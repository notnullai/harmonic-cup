"""Turn an engine event into a ranked diagnosis subgraph (HARM-S4-T05).

The engine says *what changed*: which class neurons fired, which features are
surprised, in which regime, how fast the health index is moving. It cannot say
what that means, because it has never seen this machine fail. This module is
the join.

Two entry points into the graph, per docs/KG_SCHEMA.md, and both are used:

- the **trained class** maps straight to a `FailureMode`. A strong prior.
- the **surprised features** map to `Sensor.kind`, then `DETECTS` to `Symptom`,
  then `INDICATES` to `FailureMode` with a confidence. This is the *evidence*,
  and it is what makes the answer explainable rather than asserted.

When the two disagree the query returns both and the ranking decides. A
disagreement is genuine information — the class neurons were trained on a
simulator and the confidences come from engineering practice, so the two
disagreeing is worth showing an engineer, not hiding behind a single winner.

Scores combine by probabilistic OR, the same rule the engine's health index
uses, because two independent symptoms pointing at one failure mode should
reinforce rather than average. Averaging would make a second piece of evidence
*lower* the score whenever it is weaker than the first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Any

from kg.seed import CLASS_TO_FAILURE_MODE, FAMILY_CLASS_TO_FAILURE_MODE
from kg.store import KGStore, open_store

# How much a surprised feature counts when the event does not say. Events from
# `harmonic_bench` and the API carry a magnitude; hand-written ones may not.
DEFAULT_FEATURE_WEIGHT = 1.0

# Candidates below this are dropped rather than shown. A long tail of 0.05
# "possibilities" reads as thoroughness and is noise: every failure mode in the
# graph is reachable from a raised overall vibration.
MIN_CANDIDATE_SCORE = 0.15

MAX_CANDIDATES = 5


@dataclasses.dataclass(frozen=True)
class SupportingSymptom:
    symptom_id: str
    description: str
    feature_kind: str
    direction: str
    confidence: float
    weight: float
    source: str


@dataclasses.dataclass(frozen=True)
class RankedAction:
    action_id: str
    description: str
    tool: str
    duration_min: float
    skill_level: str
    priority: int
    references: tuple[str, ...]
    safety_references: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Candidate:
    failure_mode_id: str
    description: str
    severity: int
    iso_code: str
    score: float
    evidence_score: float
    class_prior: float
    from_class: str | None
    symptoms: tuple[SupportingSymptom, ...]
    causes: tuple[dict[str, Any], ...]
    actions: tuple[RankedAction, ...]
    standards: tuple[str, ...]
    affects: tuple[str, ...]
    references: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for key in ("symptoms", "actions", "causes", "standards", "affects", "references"):
            d[key] = list(d[key])
        for a in d["actions"]:
            a["references"] = list(a["references"])
            a["safety_references"] = list(a["safety_references"])
        return d


@dataclasses.dataclass(frozen=True)
class KGSubgraph:
    equipment: dict[str, Any]
    candidates: tuple[Candidate, ...]
    citable_ids: tuple[str, ...]
    unmatched_features: tuple[str, ...]
    agreement: str
    # Failure modes the evidence or the class pointed at, but which affect no
    # component this machine has — cavitation on a rig with no impeller.
    ruled_out: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "equipment": self.equipment,
            "candidates": [c.to_json() for c in self.candidates],
            "citable_ids": list(self.citable_ids),
            "unmatched_features": list(self.unmatched_features),
            "agreement": self.agreement,
            "ruled_out": list(self.ruled_out),
        }

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


def _prob_or(scores: list[float]) -> float:
    """1 - Π(1 - s). Independent evidence reinforcing, not averaging."""
    survive = 1.0
    for s in scores:
        survive *= 1.0 - max(0.0, min(1.0, s))
    return 1.0 - survive


def _normalise_features(event: dict[str, Any]) -> list[tuple[str, str | None, float]]:
    """(feature kind, direction or None, weight) from an event.

    Accepts the three shapes that reach this function: bare strings from a
    hand-written event, the engine's `{kind, sigma}` and the API's
    `{kind, direction, weight}`.
    """
    out: list[tuple[str, str | None, float]] = []
    for item in event.get("top_surprised_features", []):
        if isinstance(item, str):
            out.append((item, None, DEFAULT_FEATURE_WEIGHT))
            continue
        kind = item.get("kind") or item.get("feature_kind")
        if not kind:
            continue
        direction = item.get("direction")
        if "weight" in item:
            weight = float(item["weight"])
        elif "sigma" in item:
            # Sigma is unbounded; saturate at 4 so one very loud channel cannot
            # by itself carry a diagnosis past everything else.
            weight = min(1.0, abs(float(item["sigma"])) / 4.0)
        else:
            weight = DEFAULT_FEATURE_WEIGHT
        out.append((str(kind), direction, weight))
    return out


def _symptoms_for(store: KGStore, kind: str, direction: str | None) -> list[Any]:
    matches = [
        n for n in store.nodes_by_label("Symptom")
        if n.props.get("feature_kind") == kind
    ]
    if direction:
        narrowed = [n for n in matches if n.props.get("direction") == direction]
        # A direction the graph does not model for this feature is not a reason
        # to discard the evidence; it is a reason not to narrow on it.
        if narrowed:
            return narrowed
    return matches


def _actions_for(store: KGStore, fm_id: str) -> tuple[list[RankedAction], set[str]]:
    actions: list[RankedAction] = []
    refs: set[str] = set()
    for edge in store.out_edges(fm_id, "MITIGATED_BY"):
        a = edge.other
        a_refs = tuple(sorted(
            e.other.id for e in store.out_edges(a.id, "DOCUMENTED_IN")
            if e.other.label == "ManualSection"
        ))
        # Safety sections are citable too, but they belong in the answer's
        # safety list rather than inside a work step.
        s_refs = tuple(sorted(
            e.other.id for e in store.out_edges(a.id, "SAFEGUARDED_BY")
            if e.other.label == "ManualSection"
        ))
        refs.update(a_refs)
        refs.update(s_refs)
        actions.append(RankedAction(
            action_id=a.id,
            description=str(a.props.get("description", a.id)),
            tool=str(a.props.get("tool", "")),
            duration_min=float(a.props.get("duration_min", 0) or 0),
            skill_level=str(a.props.get("skill_level", "technician")),
            priority=int(edge.props.get("priority", 99)),
            references=a_refs,
            safety_references=s_refs,
        ))
    actions.sort(key=lambda a: (a.priority, a.action_id))
    return actions, refs


def query_for_event(event: dict[str, Any], store: KGStore | None = None) -> KGSubgraph:
    """Rank the failure modes this event is evidence for.

    `event` carries at least `top_surprised_features`; `top_classes`, `regime`,
    `hhi`, `state` and `rul_hours` are used when present.
    """
    own = store is None
    store = store or open_store()
    try:
        return _query(event, store)
    finally:
        if own:
            store.close()


def _query(event: dict[str, Any], store: KGStore) -> KGSubgraph:
    # -- evidence path: features -> symptoms -> failure modes
    evidence: dict[str, list[tuple[float, SupportingSymptom]]] = {}
    unmatched: list[str] = []

    for kind, direction, weight in _normalise_features(event):
        symptoms = _symptoms_for(store, kind, direction)
        if not symptoms:
            unmatched.append(kind)
            continue
        for sym in symptoms:
            for edge in store.out_edges(sym.id, "INDICATES"):
                confidence = float(edge.props.get("confidence", 0.5))
                contribution = confidence * weight
                evidence.setdefault(edge.other.id, []).append((
                    contribution,
                    SupportingSymptom(
                        symptom_id=sym.id,
                        description=str(sym.props.get("description", sym.id)),
                        feature_kind=kind,
                        direction=str(sym.props.get("direction", "")),
                        confidence=confidence,
                        weight=round(weight, 3),
                        source=edge.source,
                    ),
                ))

    # -- which machine, and which family's vocabulary
    # A failure mode belongs to a machine family (`family`, "pump" when
    # absent), and so does the equipment. Only the machine's own family's
    # modes are candidates: the same feature means a different fault on a
    # different machine — 2X is the motor coupling on a pump and the input
    # shaft on a gearbox, and each is documented in its own manual.
    equipment_id = event.get("equipment_id", "P-101")
    eq_node = store.node(equipment_id)
    family = str((eq_node.props if eq_node else {}).get("family", "pump"))
    class_map = FAMILY_CLASS_TO_FAILURE_MODE.get(family, CLASS_TO_FAILURE_MODE)
    for fm_id in list(evidence):
        node = store.node(fm_id)
        if node is not None and str(node.props.get("family", "pump")) != family:
            evidence.pop(fm_id)

    # -- prior path: trained class -> failure mode
    priors: dict[str, tuple[float, str]] = {}
    for item in event.get("top_classes", []):
        name = item if isinstance(item, str) else item.get("name", "")
        score = 1.0 if isinstance(item, str) else float(item.get("score", 1.0))
        fm = class_map.get(name)
        if fm and store.node(fm) is not None:
            prev = priors.get(fm, (0.0, name))
            if score >= prev[0]:
                priors[fm] = (score, name)

    # -- can this machine have it at all?
    # A failure mode AFFECTS components; the equipment HAS_COMPONENT some. A
    # mode that affects only kinds of component this machine does not have is
    # not a candidate, whatever the evidence or the class says. This is data,
    # not a rule about any machine: P-101 has every kind and loses nothing.
    kinds = _component_kinds(store, equipment_id)
    ruled_out: list[str] = []
    if kinds:
        for fm_id in sorted(set(evidence) | set(priors)):
            affected = {str(e.other.props.get("kind", ""))
                        for e in store.out_edges(fm_id, "AFFECTS")}
            if affected and not affected & kinds:
                evidence.pop(fm_id, None)
                priors.pop(fm_id, None)
                ruled_out.append(fm_id)

    # -- combine
    candidates: list[Candidate] = []
    for fm_id in set(evidence) | set(priors):
        node = store.node(fm_id)
        if node is None or node.label != "FailureMode":
            continue

        supports = evidence.get(fm_id, [])
        evidence_score = _prob_or([c for c, _ in supports])
        prior, from_class = priors.get(fm_id, (0.0, None))
        score = _prob_or([evidence_score, prior])
        if score < MIN_CANDIDATE_SCORE:
            continue

        actions, refs = _actions_for(store, fm_id)
        described_in = tuple(sorted(
            e.other.id for e in store.out_edges(fm_id, "DESCRIBED_IN")
            if e.other.label == "ManualSection"
        ))
        refs.update(described_in)
        for _, sup in supports:
            for e in store.out_edges(sup.symptom_id, "DOCUMENTED_IN"):
                if e.other.label == "ManualSection":
                    refs.add(e.other.id)

        causes: list[dict[str, Any]] = []
        for e in store.in_edges(fm_id, "CAUSES"):
            c_refs = tuple(sorted(
                d.other.id for d in store.out_edges(e.other.id, "DOCUMENTED_IN")
                if d.other.label == "ManualSection"
            ))
            refs.update(c_refs)
            causes.append({
                "cause_id": e.other.id,
                "description": str(e.other.props.get("description", e.other.id)),
                "category": str(e.other.props.get("category", "")),
                "likelihood": float(e.props.get("likelihood", 0.5)),
                "references": list(c_refs),
            })
        causes.sort(key=lambda c: (-float(c["likelihood"]), str(c["cause_id"])))

        # De-duplicate the supporting symptoms, keeping the strongest reading
        # of each: one symptom can be reached through two feature channels.
        best: dict[str, tuple[float, SupportingSymptom]] = {}
        for contribution, sup in supports:
            if sup.symptom_id not in best or contribution > best[sup.symptom_id][0]:
                best[sup.symptom_id] = (contribution, sup)
        ordered = tuple(s for _, s in sorted(best.values(), key=lambda p: -p[0]))

        candidates.append(Candidate(
            failure_mode_id=fm_id,
            description=str(node.props.get("description", fm_id)),
            severity=int(node.props.get("severity", 3)),
            iso_code=str(node.props.get("iso_code", "")),
            score=round(score, 4),
            evidence_score=round(evidence_score, 4),
            class_prior=round(prior, 4),
            from_class=from_class,
            symptoms=ordered,
            causes=tuple(causes),
            actions=tuple(actions),
            standards=tuple(sorted(e.other.id for e in store.out_edges(fm_id, "GOVERNED_BY"))),
            affects=tuple(sorted(e.other.id for e in store.out_edges(fm_id, "AFFECTS"))),
            references=tuple(sorted(refs)),
        ))

    candidates.sort(key=lambda c: (-c.score, -c.severity, c.failure_mode_id))
    candidates = candidates[:MAX_CANDIDATES]

    # -- do the two paths agree?
    top_evidence = max(evidence, key=lambda k: _prob_or([c for c, _ in evidence[k]]),
                       default=None)
    top_prior = max(priors, key=lambda k: priors[k][0], default=None)
    if top_prior is None:
        agreement = "no-class"          # features only; nothing to disagree with
    elif top_evidence is None:
        agreement = "no-evidence"       # class fired with no surprised feature behind it
    elif top_prior == top_evidence:
        agreement = "agree"
    else:
        agreement = f"disagree:{top_prior}|{top_evidence}"

    eq = store.node(equipment_id)
    equipment = {"id": equipment_id, **(eq.props if eq else {})}

    citable = sorted({r for c in candidates for r in c.references})
    return KGSubgraph(
        equipment=equipment,
        candidates=tuple(candidates),
        citable_ids=tuple(citable),
        unmatched_features=tuple(dict.fromkeys(unmatched)),
        agreement=agreement,
        ruled_out=tuple(ruled_out),
    )


def classes_ruled_out(store: KGStore, equipment_id: str) -> dict[str, str]:
    """The engine's classes this machine cannot have, and what it lacks.

    The same test `query_for_event` applies to candidates — a failure mode
    that affects only kinds of component the machine does not have — asked of
    every class of the machine's family up front, so the console need not show
    "cavitation" as the leading class of a machine with no impeller.
    """
    kinds = _component_kinds(store, equipment_id)
    if not kinds:
        return {}
    eq = store.node(equipment_id)
    family = str((eq.props if eq else {}).get("family", "pump"))
    out: dict[str, str] = {}
    for cls, fm_id in FAMILY_CLASS_TO_FAILURE_MODE.get(family, CLASS_TO_FAILURE_MODE).items():
        if not fm_id or store.node(fm_id) is None:
            continue
        affected = {str(e.other.props.get("kind", ""))
                    for e in store.out_edges(fm_id, "AFFECTS")}
        if affected and not affected & kinds:
            out[cls] = ", ".join(sorted(affected))
    return out


def feature_meanings(store: KGStore, equipment_id: str,
                     kinds: list[str]) -> dict[str, dict[str, Any]]:
    """What each feature's movement means on this machine, for the console's
    evidence: kind -> direction -> the symptom it is and the failure modes it
    is typical of. The same path `query_for_event` walks (feature -> Symptom
    -INDICATES-> FailureMode), with the same scoping: the machine's own
    family, and no mode that affects only parts the machine lacks."""
    eq = store.node(equipment_id)
    family = str((eq.props if eq else {}).get("family", "pump"))
    parts = _component_kinds(store, equipment_id)
    fm_ok: dict[str, bool] = {}

    def allowed(fm_id: str) -> bool:
        if fm_id not in fm_ok:
            node = store.node(fm_id)
            ok = (node is not None and node.label == "FailureMode"
                  and str(node.props.get("family", "pump")) == family)
            if ok and parts:
                affected = {str(e.other.props.get("kind", ""))
                            for e in store.out_edges(fm_id, "AFFECTS")}
                ok = not affected or bool(affected & parts)
            fm_ok[fm_id] = ok
        return fm_ok[fm_id]

    out: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        per_dir: dict[str, Any] = {}
        for sym in _symptoms_for(store, kind, None):
            direction = str(sym.props.get("direction", "up"))
            points = [{"id": e.other.id, "confidence": float(e.props.get("confidence", 0.5))}
                      for e in store.out_edges(sym.id, "INDICATES") if allowed(e.other.id)]
            if not points:
                continue                  # another family's symptom
            best = per_dir.get(direction)
            if best is None:
                per_dir[direction] = {"description": str(sym.props.get("description", sym.id)),
                                      "points_to": points}
            else:
                seen = {p["id"] for p in best["points_to"]}
                best["points_to"] += [p for p in points if p["id"] not in seen]
        for d in per_dir.values():
            d["points_to"].sort(key=lambda p: -p["confidence"])
        if per_dir:
            out[kind] = per_dir
    return out


def _component_kinds(store: KGStore, equipment_id: str) -> set[str]:
    """The kinds of component the equipment has; empty when it is unknown."""
    if store.node(equipment_id) is None:
        return set()
    return {str(e.other.props.get("kind", ""))
            for e in store.out_edges(equipment_id, "HAS_COMPONENT")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Query the graph for an event")
    ap.add_argument("--event", help="path to an event JSON file, or - for stdin")
    ap.add_argument("--class", dest="cls", help="trained class name, e.g. bearing_wear")
    ap.add_argument("--features", nargs="*", default=[],
                    help="surprised feature kinds, e.g. HARMONIC_FEAT_VIB_BPFO")
    ap.add_argument("--backend", choices=["auto", "neo4j", "networkx"], default="auto")
    args = ap.parse_args(argv)

    if args.event:
        text = sys.stdin.read() if args.event == "-" else Path(args.event).read_text("utf-8")
        event = json.loads(text)
    else:
        event = {"top_surprised_features": args.features}
        if args.cls:
            event["top_classes"] = [{"name": args.cls, "score": 1.0}]

    store = open_store(args.backend)
    try:
        result = query_for_event(event, store)
    finally:
        store.close()
    print(json.dumps(result.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    from pathlib import Path

    raise SystemExit(main())
