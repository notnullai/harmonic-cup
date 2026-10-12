"""Load the seed and the extracted triples into the store (HARM-S4-T04).

Idempotent by construction: every write is a MERGE on `id`, so running the
loader twice leaves the same graph. The second run is part of the test, not an
afterthought — a loader that duplicates on re-run makes every count downstream
a lie, and the counts are the acceptance criterion.

Order matters. The seed is merged LAST so a curated `confidence` overrides an
extracted one on the same edge: the extractor infers confidence from hedging
words in the prose, which is a reasonable guess and not a reliability
engineer's judgement.

    python -m kg.load                       # rebuild from seed + kg/extracted.json
    python -m kg.load --backend networkx
    python -m kg.load --verify              # counts only, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kg.extract import EXTRACTED_JSON
from kg.model import Graph, Node, Triple, validate_graph
from kg.seed import build_seed
from kg.store import KGStore, open_store
from rag.ingest import load_manifest

# The acceptance thresholds from the tracker. Kept here rather than in the test
# so the loader itself reports against them.
MIN_NODES = 100
MIN_RELATIONS = 150


def manual_section_nodes() -> list[Node]:
    """A `ManualSection` node per chunk in the manifest.

    Generated from the manifest rather than written by hand, so the set of
    citable ids in the graph is by construction the set of chunks the retriever
    can return. That equality is what makes `rag/guard.py` a real boundary
    instead of a second list to keep in step.
    """
    return [
        Node("ManualSection", c.id, {
            "doc": c.doc, "section": c.section, "page": c.page,
            "title": c.title, "text_chunk": c.text,
        })
        for c in load_manifest()
    ]


def load_extracted(path: Path = EXTRACTED_JSON) -> list[Triple]:
    if not path.exists():
        print(f"note: {path} not found; loading the seed only. "
              f"Run: python -m kg.extract", file=sys.stderr)
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Triple(t["subject"], t["relation"], t["object"],
                   t.get("props", {}), t.get("source", "extracted"))
            for t in raw["triples"]]


def build_graph(extracted_path: Path = EXTRACTED_JSON) -> Graph:
    """Manual sections, then extracted triples, then the curated seed on top."""
    sections = Graph(manual_section_nodes(), [])
    extracted = Graph([], load_extracted(extracted_path))
    graph = sections.merge(extracted).merge(build_seed())
    validate_graph(graph, citable_ids=[n.id for n in sections.nodes])
    return graph


def load(store: KGStore, graph: Graph | None = None) -> tuple[int, int]:
    graph = graph if graph is not None else build_graph()
    counts = store.merge_graph(graph)
    print(f"{store.backend}: {counts}")
    return counts.nodes_total, counts.relations_total


def report(n_nodes: int, n_relations: int) -> int:
    ok_nodes = n_nodes >= MIN_NODES
    ok_rels = n_relations >= MIN_RELATIONS
    print(f"  nodes:     {n_nodes:>4}  (target >= {MIN_NODES})  "
          f"{'PASS' if ok_nodes else 'FAIL'}")
    print(f"  relations: {n_relations:>4}  (target >= {MIN_RELATIONS})  "
          f"{'PASS' if ok_rels else 'FAIL'}")
    return 0 if (ok_nodes and ok_rels) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load the knowledge graph")
    # No default here: an unset flag defers to HARMONIC_KG_BACKEND, so the
    # entrypoint loads the store the API will query (Spanner on Google Cloud).
    ap.add_argument("--backend", choices=["auto", "neo4j", "networkx", "spanner"], default=None)
    ap.add_argument("--path", default=None, help="networkx: where to persist the JSON")
    ap.add_argument("--extracted", default=str(EXTRACTED_JSON))
    ap.add_argument("--verify", action="store_true", help="count only, write nothing")
    ap.add_argument("--rebuild", action="store_true",
                    help="empty the store first. MERGE never deletes, so an edge "
                         "removed from the seed survives an ordinary load: the "
                         "graph keeps growing and the counts stop meaning "
                         "anything. Use this whenever the schema changed.")
    ap.add_argument("--twice", action="store_true",
                    help="load twice and assert the counts do not move")
    args = ap.parse_args(argv)

    kwargs = {"path": args.path} if args.path else {}
    store = open_store(args.backend, **kwargs)
    try:
        if args.verify:
            n, r = store.counts()
            print(f"{store.backend}: {n} nodes, {r} relations")
            return report(n, r)

        graph = build_graph(Path(args.extracted))
        if args.rebuild:
            store.clear()
            print(f"{store.backend}: cleared")
        n, r = load(store, graph)
        if args.twice:
            n2, r2 = load(store, graph)
            if (n, r) != (n2, r2):
                print(f"FAIL: not idempotent — {n}/{r} became {n2}/{r2}", file=sys.stderr)
                return 1
            print("idempotent: a second load left the counts unchanged")
        return report(n, r)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
