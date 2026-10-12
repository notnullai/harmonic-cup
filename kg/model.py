"""The knowledge-graph data model (docs/KG_SCHEMA.md), in one place.

Everything that writes to the graph — the hand-curated seed, the extractor, the
loader — goes through `validate_graph` first. The schema is frozen and the
loader is idempotent, which together mean a malformed triple does not fail
loudly at load time: it MERGEs a node with the wrong label and sits there
looking like data. So the validation is here, ahead of the store, and both
store backends get the same checked input.

No Neo4j or NetworkX import in this module. It is the vocabulary, not a client.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

# --- vocabulary ---------------------------------------------------------------

NODE_LABELS: dict[str, tuple[str, ...]] = {
    "Equipment": ("model", "serial", "nominal_rpm", "iso_class"),
    "Component": ("kind", "position"),
    "Sensor": ("kind", "unit", "sample_rate_hz", "axis"),
    "FailureMode": ("iso_code", "description", "severity"),
    "Symptom": ("description", "feature_kind", "direction"),
    "Cause": ("description", "category"),
    "Action": ("description", "tool", "duration_min", "skill_level"),
    "ManualSection": ("doc", "section", "page", "title", "text_chunk"),
    "Standard": ("name", "clause"),
}

# (relation, from_label, to_label) -> allowed relation properties.
RELATIONS: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("HAS_COMPONENT", "Equipment", "Component"): (),
    ("MONITORED_BY", "Component", "Sensor"): (),
    ("DETECTS", "Sensor", "Symptom"): (),
    ("INDICATES", "Symptom", "FailureMode"): ("confidence",),
    ("CAUSES", "Cause", "FailureMode"): ("likelihood",),
    ("MITIGATED_BY", "FailureMode", "Action"): ("priority",),
    ("DOCUMENTED_IN", "Action", "ManualSection"): (),
    # Separate from DOCUMENTED_IN on purpose. The section that says how to do
    # the work and the section that says how not to be hurt doing it are both
    # citations, but they belong in different places in the answer: one in the
    # work step, one in the safety list. One relation for both made a work
    # order whose first instruction cited the lockout procedure.
    ("SAFEGUARDED_BY", "Action", "ManualSection"): (),
    ("GOVERNED_BY", "FailureMode", "Standard"): (),
    ("AFFECTS", "FailureMode", "Component"): (),
    # A failure mode is explained by manual prose as well as by its actions.
    # Without this edge the only citations reachable from a diagnosis are the
    # procedure sections, and the guardrail then rejects a reference to the
    # section that actually describes the symptom.
    ("DESCRIBED_IN", "FailureMode", "ManualSection"): (),
    ("DOCUMENTED_IN", "Cause", "ManualSection"): (),
    ("DOCUMENTED_IN", "Symptom", "ManualSection"): (),
}

RELATION_NAMES = frozenset(r for r, _, _ in RELATIONS)

DIRECTIONS = frozenset({"up", "down", "oscillating"})
SKILL_LEVELS = frozenset({"operator", "technician", "specialist"})
CAUSE_CATEGORIES = frozenset(
    {"lubrication", "installation", "process", "wear", "contamination", "electrical"}
)
COMPONENT_KINDS = frozenset(
    {"bearing", "impeller", "seal", "motor", "coupling", "shaft", "foundation", "gear",
     "rotor", "stator"}
)


class SchemaError(ValueError):
    """A node or triple that does not fit docs/KG_SCHEMA.md."""


# --- records ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Node:
    label: str
    id: str
    props: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "id": self.id, "props": dict(self.props)}


@dataclasses.dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    props: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Where this triple came from: "seed", "rules:<chunk id>", "gemini:<chunk id>".
    # Provenance is not decoration. A diagnosis drawn from an extracted triple
    # is weaker evidence than one drawn from the curated seed, and an engineer
    # asking "why does it think that" is asking this question.
    source: str = "seed"

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)

    def to_json(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "props": dict(self.props),
            "source": self.source,
        }


@dataclasses.dataclass
class Graph:
    nodes: list[Node] = dataclasses.field(default_factory=list)
    triples: list[Triple] = dataclasses.field(default_factory=list)

    def node_labels(self) -> dict[str, str]:
        return {n.id: n.label for n in self.nodes}

    def merge(self, other: Graph) -> Graph:
        """Union, later properties winning on a collision, triples deduplicated.

        The seed is merged last so a curated confidence overrides an extracted
        one for the same edge.
        """
        by_id: dict[str, Node] = {}
        for n in [*self.nodes, *other.nodes]:
            if n.id in by_id:
                prev = by_id[n.id]
                if prev.label != n.label:
                    raise SchemaError(
                        f"{n.id} appears as both {prev.label} and {n.label}"
                    )
                merged = {**prev.props, **{k: v for k, v in n.props.items() if v is not None}}
                by_id[n.id] = Node(n.label, n.id, merged)
            else:
                by_id[n.id] = n

        by_key: dict[tuple[str, str, str], Triple] = {}
        for t in [*self.triples, *other.triples]:
            if t.key() in by_key:
                earlier = by_key[t.key()]
                by_key[t.key()] = Triple(
                    t.subject, t.relation, t.object,
                    {**earlier.props, **t.props},
                    # Keep both origins; the loader writes the joined string.
                    source=earlier.source if earlier.source == t.source
                    else f"{earlier.source}+{t.source}",
                )
            else:
                by_key[t.key()] = t
        return Graph(list(by_id.values()), list(by_key.values()))


# --- validation ---------------------------------------------------------------


def _check_node(node: Node) -> list[str]:
    errors: list[str] = []
    if node.label not in NODE_LABELS:
        return [f"{node.id}: unknown label {node.label!r}"]
    if not node.id:
        return ["a node has an empty id"]

    p = node.props
    if node.label == "Symptom":
        if p.get("direction") not in DIRECTIONS:
            errors.append(f"{node.id}: direction {p.get('direction')!r} not in {sorted(DIRECTIONS)}")
        if not p.get("feature_kind"):
            errors.append(f"{node.id}: Symptom needs feature_kind, the join back to the engine")
    if node.label == "Action" and p.get("skill_level") not in SKILL_LEVELS:
        errors.append(f"{node.id}: skill_level {p.get('skill_level')!r} not in {sorted(SKILL_LEVELS)}")
    if node.label == "Cause" and p.get("category") not in CAUSE_CATEGORIES:
        errors.append(f"{node.id}: category {p.get('category')!r} not in {sorted(CAUSE_CATEGORIES)}")
    if node.label == "Component" and p.get("kind") not in COMPONENT_KINDS:
        errors.append(f"{node.id}: kind {p.get('kind')!r} not in {sorted(COMPONENT_KINDS)}")
    if node.label == "FailureMode":
        sev = p.get("severity")
        if not isinstance(sev, int) or not 1 <= sev <= 5:
            errors.append(f"{node.id}: severity {sev!r} must be an int in 1..5")
    return errors


def _check_triple(t: Triple, labels: dict[str, str]) -> list[str]:
    if t.relation not in RELATION_NAMES:
        return [f"{t.key()}: unknown relation {t.relation!r}"]
    if t.subject not in labels:
        return [f"{t.key()}: subject {t.subject!r} is not a node in this graph"]
    if t.object not in labels:
        return [f"{t.key()}: object {t.object!r} is not a node in this graph"]

    signature = (t.relation, labels[t.subject], labels[t.object])
    if signature not in RELATIONS:
        allowed = sorted(f"{f}->{o}" for r, f, o in RELATIONS if r == t.relation)
        return [f"{t.key()}: {labels[t.subject]}-[:{t.relation}]->{labels[t.object]} "
                f"is not in the schema; {t.relation} allows {allowed}"]

    errors: list[str] = []
    for name in ("confidence", "likelihood"):
        if name in t.props:
            v = t.props[name]
            if not isinstance(v, int | float) or not 0.0 <= float(v) <= 1.0:
                errors.append(f"{t.key()}: {name} {v!r} must be a number in 0..1")
    if "priority" in t.props:
        v = t.props["priority"]
        if not isinstance(v, int) or v < 1:
            errors.append(f"{t.key()}: priority {v!r} must be a positive int")
    return errors


def validate_graph(graph: Graph, *, citable_ids: Iterable[str] | None = None) -> None:
    """Raise `SchemaError` listing everything wrong, not just the first thing."""
    errors: list[str] = []
    seen: set[str] = set()
    for n in graph.nodes:
        if n.id in seen:
            errors.append(f"{n.id}: duplicate node id")
        seen.add(n.id)
        errors.extend(_check_node(n))

    labels = graph.node_labels()
    for t in graph.triples:
        errors.extend(_check_triple(t, labels))

    if citable_ids is not None:
        allowed = set(citable_ids)
        for n in graph.nodes:
            if n.label == "ManualSection" and n.id not in allowed:
                errors.append(
                    f"{n.id}: ManualSection is not in the chunk manifest, so a "
                    f"diagnosis citing it would be rejected by rag/guard.py"
                )
    if errors:
        raise SchemaError(
            f"{len(errors)} schema problem(s):\n  " + "\n  ".join(sorted(errors))
        )
