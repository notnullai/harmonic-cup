"""The knowledge graph in Spanner Graph (HARM-S8-G07).

A third implementation of `kg.store.KGStore`, beside NetworkX and Neo4j, so
the diagnosis code does not change: the graph lives in Google Cloud Spanner
and every traversal is a GQL graph query.

Schema — one node table and one edge table under a property graph. Labels and
relation names are columns, so a new machine family's node kinds need no DDL:

    KgNode (id, label, props JSON)                      PRIMARY KEY (id)
    KgEdge (id = source node, rel, dst, props JSON, source)
                                                        PRIMARY KEY (id, rel, dst)
                                                        INTERLEAVED IN KgNode
    PROPERTY GRAPH HarmonicKG
        NODE TABLES (KgNode)  LABEL Entity
        EDGE TABLES (KgEdge)  LABEL Relates   id -> KgNode, dst -> KgNode

A traversal, as `out_edges` issues it:

    GRAPH HarmonicKG
    MATCH (a:Entity {id: @id})-[e:Relates]->(b:Entity)
    WHERE e.rel = @rel
    RETURN b.id, b.label, b.props, e.rel, e.props, e.source

The graph does not change while the service runs, so reads are cached and the
cache is dropped on every write.

    HARMONIC_KG_BACKEND=spanner
    HARMONIC_SPANNER_INSTANCE=harmonic  HARMONIC_SPANNER_DATABASE=kg
    GOOGLE_CLOUD_PROJECT=...            (SPANNER_EMULATOR_HOST for the emulator)
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from kg.model import Graph, Node
from kg.store import Edge, LoadCounts

GRAPH = "HarmonicKG"

DDL = [
    """CREATE TABLE KgNode (
        id STRING(MAX) NOT NULL,
        label STRING(64) NOT NULL,
        props JSON,
    ) PRIMARY KEY (id)""",
    "CREATE INDEX KgNodeByLabel ON KgNode(label)",
    """CREATE TABLE KgEdge (
        id STRING(MAX) NOT NULL,
        rel STRING(64) NOT NULL,
        dst STRING(MAX) NOT NULL,
        props JSON,
        source STRING(MAX),
    ) PRIMARY KEY (id, rel, dst),
      INTERLEAVE IN PARENT KgNode ON DELETE CASCADE""",
    "CREATE INDEX KgEdgeByDst ON KgEdge(dst, rel)",
    f"""CREATE PROPERTY GRAPH {GRAPH}
        NODE TABLES (KgNode KEY (id) LABEL Entity PROPERTIES (id, label, props))
        EDGE TABLES (KgEdge KEY (id, rel, dst)
            SOURCE KEY (id) REFERENCES KgNode (id)
            DESTINATION KEY (dst) REFERENCES KgNode (id)
            LABEL Relates PROPERTIES (rel, props, source))""",
]


def _props(v: Any) -> dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, str):
        return dict(json.loads(v))
    return dict(v)


class SpannerGraphStore:
    backend = "spanner"

    def __init__(self, instance: str | None = None, database: str | None = None,
                 project: str | None = None) -> None:
        from google.cloud import spanner

        self._spanner = spanner
        client = spanner.Client(project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"))
        inst = client.instance(instance or os.environ.get("HARMONIC_SPANNER_INSTANCE",
                                                          "harmonic"))
        self.db = inst.database(database or os.environ.get("HARMONIC_SPANNER_DATABASE", "kg"))
        self._lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], Any] = {}
        self.ensure_schema()

    # -- schema

    def ensure_schema(self) -> None:
        with self.db.snapshot() as snap:
            names = {r[0] for r in snap.execute_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = ''")}
        if {"KgNode", "KgEdge"} <= names:
            return
        self.db.update_ddl(DDL).result(timeout=600)

    # -- reads

    def _sql(self, sql: str, **params: Any) -> list[tuple[Any, ...]]:
        types = {k: self._spanner.param_types.STRING for k in params}
        with self.db.snapshot() as snap:
            return [tuple(r) for r in snap.execute_sql(sql, params=params or None,
                                                       param_types=types or None)]

    def _cached(self, key: tuple[Any, ...], fetch: Any) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        value = fetch()
        with self._lock:
            self._cache[key] = value
        return value

    def counts(self) -> tuple[int, int]:
        n = self._sql("SELECT COUNT(*) FROM KgNode")[0][0]
        r = self._sql("SELECT COUNT(*) FROM KgEdge")[0][0]
        return int(n), int(r)

    def node(self, node_id: str) -> Node | None:
        def fetch() -> Node | None:
            rows = self._sql("SELECT label, props FROM KgNode WHERE id = @id", id=node_id)
            return Node(rows[0][0], node_id, _props(rows[0][1])) if rows else None
        return self._cached(("node", node_id), fetch)

    def nodes_by_label(self, label: str) -> list[Node]:
        def fetch() -> list[Node]:
            return [Node(label, r[0], _props(r[1])) for r in self._sql(
                "SELECT id, props FROM KgNode WHERE label = @label", label=label)]
        return list(self._cached(("label", label), fetch))

    def _edges(self, node_id: str, relation: str | None, outgoing: bool) -> list[Edge]:
        pattern = ("(a:Entity {id: @id})-[e:Relates]->(b:Entity)" if outgoing
                   else "(b:Entity)-[e:Relates]->(a:Entity {id: @id})")
        where = "WHERE e.rel = @rel " if relation else ""
        gql = (f"GRAPH {GRAPH} MATCH {pattern} {where}"
               "RETURN b.id AS oid, b.label AS olabel, b.props AS oprops, "
               "e.rel AS rel, e.props AS rprops, e.source AS source")
        params = {"id": node_id, **({"rel": relation} if relation else {})}

        def fetch() -> list[Edge]:
            return [Edge(Node(r[1], r[0], _props(r[2])), r[3], _props(r[4]), r[5] or "seed")
                    for r in self._sql(gql, **params)]
        return list(self._cached(("edges", node_id, relation, outgoing), fetch))

    def out_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        return self._edges(node_id, relation, True)

    def in_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        return self._edges(node_id, relation, False)

    # -- writes

    def merge_graph(self, graph: Graph) -> LoadCounts:
        """Upsert, later properties winning — the NetworkX store's semantics."""
        from google.cloud.spanner_v1.data_types import JsonObject
        nodes = {r[0]: (r[1], _props(r[2]))
                 for r in self._sql("SELECT id, label, props FROM KgNode")}
        edges = {(r[0], r[1], r[2]): _props(r[3])
                 for r in self._sql("SELECT id, rel, dst, props FROM KgEdge")}
        node_rows, n_written = [], 0
        for n in graph.nodes:
            old = nodes.get(n.id)
            merged = {**(old[1] if old else {}), **n.props}
            if old is None or old != (n.label, merged):
                n_written += 1
            nodes[n.id] = (n.label, merged)
            node_rows.append((n.id, n.label, JsonObject(merged)))
        edge_rows, r_written = [], 0
        for t in graph.triples:
            key = (t.subject, t.relation, t.object)
            merged = {**edges.get(key, {}), **t.props}
            if key not in edges or edges[key] != merged:
                r_written += 1
            edges[key] = merged
            edge_rows.append((*key, JsonObject(merged), t.source))
        # Nodes first: an edge row is interleaved under its source node.
        for i in range(0, len(node_rows), 500):
            with self.db.batch() as b:
                b.insert_or_update("KgNode", ("id", "label", "props"), node_rows[i:i + 500])
        for i in range(0, len(edge_rows), 500):
            with self.db.batch() as b:
                b.insert_or_update("KgEdge", ("id", "rel", "dst", "props", "source"),
                                   edge_rows[i:i + 500])
        with self._lock:
            self._cache.clear()
        n_nodes, n_rels = self.counts()
        return LoadCounts(n_written, r_written, n_nodes, n_rels)

    def clear(self) -> None:
        keys = self._spanner.KeySet(all_=True)
        with self.db.batch() as b:
            b.delete("KgEdge", keys)
            b.delete("KgNode", keys)
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        pass
