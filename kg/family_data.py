"""A machine family's knowledge, as data (HARM-S7-T10).

Each family is one file, `kg/families/<family>.json`: its equipment and parts,
sensors, symptoms (keyed on the engine's feature names), failure modes, causes,
actions and standards, the relations between them in the ISO 13379-1 shape
(symptom INDICATES failure mode, MITIGATED_BY action, DOCUMENTED_IN a manual
section), the class-to-failure-mode table its trained network reads through,
and the phrases its manual uses for each node, for the rules extractor.

Adding a machine family is adding a file here and one in `profiles/`; no code
changes. Relations may point at nodes another file defines — the pump file
holds the vocabulary every family shares (generic symptoms, causes, the ISO
standards) — and the merged graph is validated as a whole (`kg/load.py`).

    {"family": "...", "about": "...", "manual": "DOC-ID", "classes": {class: fm},
     "aliases": {node id: [phrase, ...]},
     "nodes": {Label: [{"id": ..., prop: value, ...}, ...]},
     "relations": {RELATION: [[subject, object], [subject, object, {props}], ...]}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from kg.model import Graph, Node, Triple

FAMILIES_DIR = Path(__file__).resolve().parent / "families"
# The pump comes first: it defines the shared vocabulary, and a machine with no
# family is a pump.
FIRST = "pump"


@dataclass(frozen=True)
class FamilyData:
    family: str
    about: str
    manual: str
    classes: dict[str, str]
    aliases: dict[str, tuple[str, ...]]
    graph: Graph


def parse(raw: dict[str, Any]) -> FamilyData:
    nodes = [Node(label, str(n["id"]), {k: v for k, v in n.items() if k != "id"})
             for label, items in raw.get("nodes", {}).items() for n in items]
    triples = []
    for rel, rows in raw.get("relations", {}).items():
        for row in rows:
            if len(row) not in (2, 3):
                raise ValueError(f"{raw.get('family')}: a {rel} row is [subject, object"
                                 f"(, props)], not {row!r}")
            triples.append(Triple(str(row[0]), rel, str(row[1]),
                                  dict(row[2]) if len(row) == 3 else {}))
    return FamilyData(
        family=str(raw["family"]), about=str(raw.get("about", "")),
        manual=str(raw.get("manual", "")), classes=dict(raw.get("classes", {})),
        aliases={k: tuple(v) for k, v in raw.get("aliases", {}).items()},
        graph=Graph(nodes, triples))


@lru_cache(maxsize=1)
def load_all(folder: Path = FAMILIES_DIR) -> tuple[FamilyData, ...]:
    """Every family file, the pump first, the rest by name."""
    paths = sorted(folder.glob("*.json"), key=lambda p: (p.stem != FIRST, p.stem))
    out = []
    for p in paths:
        data = parse(json.loads(p.read_text(encoding="utf-8")))
        if data.family != p.stem:
            raise ValueError(f"{p.name} declares family {data.family!r}")
        out.append(data)
    return tuple(out)


def by_family() -> dict[str, FamilyData]:
    return {f.family: f for f in load_all()}
