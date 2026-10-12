"""Graph storage behind one interface (HARM-S4-T04).

Two implementations of `KGStore`:

- `Neo4jStore` — Neo4j Community from docker-compose, the real thing.
- `NetworkXStore` — an in-process DiGraph persisted as JSON, the fallback.

The interface exists from the start rather than being retrofitted, because a
fallback added under time pressure is a fallback that does not work. Both are
exercised by the same tests, and `kg/query.py` is written against the interface
and has no Cypher in it: a query that only ran on one backend would mean the
fallback silently answered differently on the day it was needed.

The traversal surface is deliberately small — node lookup, label scan, one hop
out, one hop in. Everything in `query.py` is built from those four, so the
NetworkX implementation is twenty lines and the Neo4j one is four statements.
Pushing the whole traversal into Cypher would be faster and would make the
fallback a second, divergent implementation of the ranking logic.

    HARMONIC_KG_BACKEND=neo4j|networkx|spanner|auto   (default: auto)
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Protocol

from kg.model import Graph, Node, Triple

DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "graph.json"


@dataclasses.dataclass(frozen=True)
class Edge:
    other: Node
    relation: str
    props: dict[str, Any]
    source: str


@dataclasses.dataclass(frozen=True)
class LoadCounts:
    nodes_written: int
    relations_written: int
    nodes_total: int
    relations_total: int

    def __str__(self) -> str:
        return (f"{self.nodes_total} nodes, {self.relations_total} relations "
                f"({self.nodes_written} nodes and {self.relations_written} "
                f"relations written this run)")


class KGStore(Protocol):
    backend: str

    def merge_graph(self, graph: Graph) -> LoadCounts: ...
    def clear(self) -> None: ...
    def counts(self) -> tuple[int, int]: ...
    def node(self, node_id: str) -> Node | None: ...
    def nodes_by_label(self, label: str) -> list[Node]: ...
    def out_edges(self, node_id: str, relation: str | None = None) -> list[Edge]: ...
    def in_edges(self, node_id: str, relation: str | None = None) -> list[Edge]: ...
    def close(self) -> None: ...


# --- NetworkX -----------------------------------------------------------------


class NetworkXStore:
    """DiGraph in memory, JSON on disk.

    A MultiDiGraph would be more faithful — the schema allows two different
    relations between the same pair — so the edge key is (relation) and the
    data is stored per relation under one edge record.
    """

    backend = "networkx"

    def __init__(self, path: Path | str = DEFAULT_JSON_PATH, *, autoload: bool = True,
                 persist: bool = True):
        import networkx as nx

        self.path = Path(path)
        self.g = nx.MultiDiGraph()
        # False for a store that reads the seeded catalogue but must never
        # write back to it — an ephemeral node merged in for one query (a
        # try-mode diagnosis) is never allowed to leak into the shared graph
        # file, whatever the caller merges into it.
        self.persist = persist
        if autoload and self.path.exists():
            self._load()

    # -- persistence
    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for n in raw["nodes"]:
            self.g.add_node(n["id"], label=n["label"], props=n["props"])
        for e in raw["edges"]:
            self.g.add_edge(e["subject"], e["object"], key=e["relation"],
                            props=e["props"], source=e.get("source", "seed"))

    def save(self) -> None:
        if not self.persist:
            return
        payload = {
            "nodes": [
                {"id": nid, "label": d["label"], "props": d["props"]}
                for nid, d in sorted(self.g.nodes(data=True))
            ],
            "edges": [
                {"subject": u, "relation": k, "object": v,
                 "props": d["props"], "source": d["source"]}
                for u, v, k, d in sorted(self.g.edges(keys=True, data=True),
                                         key=lambda e: (e[0], e[2], e[1]))
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    # -- writes
    def merge_graph(self, graph: Graph) -> LoadCounts:
        nodes_written = 0
        for n in graph.nodes:
            if n.id in self.g.nodes:
                existing = self.g.nodes[n.id]
                merged = {**existing["props"], **n.props}
                if merged != existing["props"] or existing["label"] != n.label:
                    nodes_written += 1
                existing["props"] = merged
                existing["label"] = n.label
            else:
                self.g.add_node(n.id, label=n.label, props=dict(n.props))
                nodes_written += 1

        rels_written = 0
        for t in graph.triples:
            if self.g.has_edge(t.subject, t.object, key=t.relation):
                d = self.g.edges[t.subject, t.object, t.relation]
                merged = {**d["props"], **t.props}
                if merged != d["props"]:
                    rels_written += 1
                d["props"] = merged
                d["source"] = t.source
            else:
                self.g.add_edge(t.subject, t.object, key=t.relation,
                                props=dict(t.props), source=t.source)
                rels_written += 1

        self.save()
        n_nodes, n_rels = self.counts()
        return LoadCounts(nodes_written, rels_written, n_nodes, n_rels)

    def clear(self) -> None:
        self.g.clear()
        self.save()

    # -- reads
    def counts(self) -> tuple[int, int]:
        return self.g.number_of_nodes(), self.g.number_of_edges()

    def node(self, node_id: str) -> Node | None:
        if node_id not in self.g.nodes:
            return None
        d = self.g.nodes[node_id]
        return Node(d["label"], node_id, dict(d["props"]))

    def nodes_by_label(self, label: str) -> list[Node]:
        return [Node(d["label"], nid, dict(d["props"]))
                for nid, d in self.g.nodes(data=True) if d["label"] == label]

    def out_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        if node_id not in self.g.nodes:
            return []
        out = []
        for _, v, k, d in self.g.out_edges(node_id, keys=True, data=True):
            if relation is None or k == relation:
                other = self.node(v)
                if other is not None:
                    out.append(Edge(other, k, dict(d["props"]), d["source"]))
        return out

    def in_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        if node_id not in self.g.nodes:
            return []
        out = []
        for u, _, k, d in self.g.in_edges(node_id, keys=True, data=True):
            if relation is None or k == relation:
                other = self.node(u)
                if other is not None:
                    out.append(Edge(other, k, dict(d["props"]), d["source"]))
        return out

    def close(self) -> None:
        return None


# --- Neo4j --------------------------------------------------------------------


class Neo4jStore:
    """Neo4j Community. Every write is a MERGE on `id`, so loading is idempotent."""

    backend = "neo4j"

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        from neo4j import GraphDatabase

        uri = uri or os.environ.get("HARMONIC_NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("HARMONIC_NEO4J_USER", "neo4j")
        password = password or os.environ.get("HARMONIC_NEO4J_PASSWORD", "harmonic-dev")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    def merge_graph(self, graph: Graph) -> LoadCounts:
        from kg.model import NODE_LABELS

        for label in NODE_LABELS:
            self._run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                      f"FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        self._run("CREATE INDEX sensor_kind IF NOT EXISTS FOR (n:Sensor) ON (n.kind)")
        self._run("CREATE INDEX symptom_kind IF NOT EXISTS FOR (n:Symptom) ON (n.feature_kind)")

        before = self.counts()
        by_label: dict[str, list[dict[str, Any]]] = {}
        for n in graph.nodes:
            by_label.setdefault(n.label, []).append({"id": n.id, "props": n.props})
        for label, rows in by_label.items():
            self._run(
                f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props",
                rows=rows,
            )

        labels = graph.node_labels()
        by_signature: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for t in graph.triples:
            sig = (t.relation, labels[t.subject], labels[t.object])
            by_signature.setdefault(sig, []).append(
                {"s": t.subject, "o": t.object,
                 "props": {**t.props, "source": t.source}}
            )
        for (rel, from_label, to_label), rows in by_signature.items():
            self._run(
                f"UNWIND $rows AS row "
                f"MATCH (a:{from_label} {{id: row.s}}), (b:{to_label} {{id: row.o}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += row.props",
                rows=rows,
            )

        after = self.counts()
        return LoadCounts(after[0] - before[0], after[1] - before[1], after[0], after[1])

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    def counts(self) -> tuple[int, int]:
        n = self._run("MATCH (n) RETURN count(n) AS c")[0]["c"]
        r = self._run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        return int(n), int(r)

    def node(self, node_id: str) -> Node | None:
        rows = self._run(
            "MATCH (n {id: $id}) RETURN labels(n) AS labels, properties(n) AS props",
            id=node_id,
        )
        if not rows:
            return None
        props = dict(rows[0]["props"])
        props.pop("id", None)
        return Node(rows[0]["labels"][0], node_id, props)

    def nodes_by_label(self, label: str) -> list[Node]:
        rows = self._run(f"MATCH (n:{label}) RETURN n.id AS id, properties(n) AS props")
        out = []
        for r in rows:
            props = dict(r["props"])
            props.pop("id", None)
            out.append(Node(label, r["id"], props))
        return out

    def _edges(self, node_id: str, relation: str | None, outgoing: bool) -> list[Edge]:
        pattern = "(n {id: $id})-[r]->(m)" if outgoing else "(m)-[r]->(n {id: $id})"
        where = "WHERE type(r) = $rel " if relation else ""
        rows = self._run(
            f"MATCH {pattern} {where}"
            f"RETURN type(r) AS rel, properties(r) AS rprops, "
            f"m.id AS oid, labels(m) AS olabels, properties(m) AS oprops",
            id=node_id, rel=relation,
        )
        out = []
        for r in rows:
            oprops = dict(r["oprops"])
            oprops.pop("id", None)
            rprops = dict(r["rprops"])
            source = str(rprops.pop("source", "seed"))
            out.append(Edge(Node(r["olabels"][0], r["oid"], oprops), r["rel"], rprops, source))
        return out

    def out_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        return self._edges(node_id, relation, outgoing=True)

    def in_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        return self._edges(node_id, relation, outgoing=False)

    def close(self) -> None:
        self._driver.close()


# --- selection ----------------------------------------------------------------


def open_store(backend: str | None = None, **kwargs: Any) -> KGStore:
    """Open the configured store.

    `auto` prefers Neo4j and falls back to NetworkX, printing which one it got.
    Silently degrading would mean a demo that runs against an empty fallback
    while everyone believes they are looking at the database.
    """
    backend = backend or os.environ.get("HARMONIC_KG_BACKEND", "auto")

    if backend == "spanner":
        # Google Cloud Spanner Graph (HARM-S8-G07). No silent fallback: a
        # deployment that asked for Spanner and got a file would be a lie.
        from kg.store_spanner import SpannerGraphStore

        return SpannerGraphStore(**{k: v for k, v in kwargs.items()
                                    if k in ("instance", "database", "project")})

    if backend in ("neo4j", "auto"):
        try:
            return Neo4jStore(**{k: v for k, v in kwargs.items()
                                 if k in ("uri", "user", "password")})
        except Exception as exc:
            if backend == "neo4j":
                raise
            import sys
            print(f"note: Neo4j unavailable ({type(exc).__name__}), "
                  f"using the NetworkX fallback", file=sys.stderr)

    return NetworkXStore(**{k: v for k, v in kwargs.items() if k in ("path", "autoload")})


def unpack(triples: list[Triple]) -> list[tuple[str, str, str]]:
    """Convenience for tests and logs."""
    return [t.key() for t in triples]
