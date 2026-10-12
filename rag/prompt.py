"""Prompt assembly for the diagnosis call (HARM-S4-T07, SPEC 8.3).

The prompt is built from two sources and nothing else: the knowledge-graph
subgraph for this event, and the manual chunks retrieved for it. That is the
whole anti-hallucination design — the model is not asked what it knows about
pumps, it is asked to write up what the graph and the manual already say.

The retrieval query is built from the event rather than typed by a person. It
joins the candidate failure modes, their symptom descriptions and the surprised
feature names, because those are the words the manual uses for the same things.
Searching with the raw feature enumerators (`HARMONIC_FEAT_VIB_BPFO`) retrieves
nothing: no manual contains them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg import families, standards
from kg.query import KGSubgraph
from rag.index import Hit

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "GENAI_SCHEMA.json"

# Below this the pipeline discards the answer and says so instead (SPEC 8.4).
MIN_CONFIDENCE = 0.4

INSUFFICIENT_EVIDENCE = "insufficient evidence — recommend manual inspection"

SYSTEM = """You are a senior rotating-equipment reliability engineer.
Use ONLY the provided context. Never invent manual sections, part numbers or
measurements. Every id in `references` must appear in the ALLOWED REFERENCES
list below, and you must cite the sections that support what you actually say.
If the context does not support a diagnosis, set confidence below 0.4 and say
what further measurement would settle it.
Return JSON matching the schema."""

TASK = """TASK
Diagnose the machine. Return JSON with:
  diagnosis       one or two sentences a maintenance engineer can act on
  confidence      0..1, your confidence in the diagnosis
  root_causes     why this arose, most likely first, from the KG causes
  actions         ordered work steps, each with step, tool, duration_min,
                  and skill_level (operator | technician | specialist)
  safety          precautions before the work starts; never empty
  references      ManualSection ids from ALLOWED REFERENCES only
  rul_hours       remaining useful life in hours
  failure_mode_id the KG id of the mode you are diagnosing"""


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_query(event: dict[str, Any], kg: KGSubgraph) -> str:
    """The words the manual would use for what the engine just saw."""
    parts: list[str] = []
    for c in kg.candidates[:3]:
        parts.append(c.description)
        parts.extend(s.description for s in c.symptoms[:3])
    if not parts:
        # No candidate: fall back to whatever the event itself names, with the
        # enumerator prefix stripped so the words are English.
        for item in event.get("top_surprised_features", []):
            kind = item if isinstance(item, str) else item.get("kind", "")
            parts.append(kind.removeprefix("HARMONIC_FEAT_").replace("_", " ").lower())
    return " ".join(parts)


def _fmt_features(event: dict[str, Any]) -> list[str]:
    lines = []
    for item in event.get("top_surprised_features", []):
        if isinstance(item, str):
            lines.append(f"  - {item}")
            continue
        kind = item.get("kind", "?")
        bits = []
        if "sigma" in item:
            bits.append(f"{float(item['sigma']):+.1f} sigma")
        if item.get("direction"):
            bits.append(str(item["direction"]))
        if "value" in item:
            bits.append(f"value {item['value']}")
        lines.append(f"  - {kind}" + (f": {', '.join(bits)}" if bits else ""))
    return lines or ["  (none reported)"]


def build_prompt(event: dict[str, Any], kg: KGSubgraph, hits: list[Hit]) -> str:
    eq = kg.equipment
    out: list[str] = ["SYSTEM", SYSTEM, "", "CONTEXT"]
    out.append(
        f"Machine: {eq.get('id', '?')}, {eq.get('model', 'pump')}, "
        f"nominal {eq.get('nominal_rpm', '?')} RPM, {standards.short_label(eq.get('iso_class'))}"
    )
    if event.get("unrecognised"):
        out.append("Unrecognised change: no trained fault class accounts for this level of "
                   "alarm. Reason from the surprised features and the graph, and say plainly "
                   "how uncertain the diagnosis is.")
    if event.get("spread"):
        out.append("Spread change: many features moved a little together, each under its "
                   "own threshold, and their combined change kept growing. No single "
                   "feature stands out; weigh the group, not the top one.")
    if event.get("early"):
        out.append("Early stage: the fault is recognised but no measurement is severe yet, so "
                   "the state is WARNING, not CRITICAL. Recommend planned repair, not shutdown.")
    if event.get("point"):
        out.append(f"Measurement point: {event['point']} (name it in the diagnosis)")
    out.append(f"Regime: {event.get('regime', 'UNKNOWN')}")
    hhi = event.get("hhi")
    out.append(f"Harmonic Health Index: {hhi if hhi is not None else 'n/a'} "
               f"({event.get('state', 'UNKNOWN')})")
    if event.get("rul_hours") is not None:
        out.append(f"RUL estimate: {event['rul_hours']} hours")
    out.append("Top surprised sensors:")
    out.extend(_fmt_features(event))

    if event.get("top_classes"):
        out.append("Engine fault classes (trained on simulated faults, a prior "
                   "rather than a measurement):")
        for c in event["top_classes"]:
            name = c if isinstance(c, str) else c.get("name", "?")
            score = "" if isinstance(c, str) else f" {float(c.get('score', 0)):.2f}"
            out.append(f"  - {name}{score}")

    out.append("")
    out.append("KG candidates:")
    if kg.candidates:
        for c in kg.candidates:
            out.append(f"  - {c.failure_mode_id}: {c.description} "
                       f"(score {c.score:.2f}, severity {c.severity}, {c.iso_code})")
            for s in c.symptoms[:4]:
                out.append(f"      supported by {s.description} "
                           f"[{s.feature_kind} {s.direction}, confidence {s.confidence:.2f}]")
            for cause in c.causes[:4]:
                out.append(f"      cause: {cause['description']} "
                           f"({cause['category']}, likelihood {cause['likelihood']:.2f})")
            for a in c.actions[:5]:
                out.append(f"      action {a.priority}: {a.description} "
                           f"[{a.tool or 'no special tool'}, {a.duration_min:.0f} min, "
                           f"{a.skill_level}]")
    else:
        out.append("  (none — the surprised features match no symptom in the graph)")

    if kg.agreement.startswith("disagree"):
        _, _, detail = kg.agreement.partition(":")
        prior, _, evidence = detail.partition("|")
        out.append("")
        out.append(f"NOTE: the trained class points at {prior} while the surprised "
                   f"features point at {evidence}. Say so in the diagnosis and give "
                   f"the measurement that would separate them.")
    elif kg.agreement == "no-evidence":
        out.append("")
        out.append("NOTE: a fault class fired but no surprised feature supports it. "
                   "Treat the diagnosis as provisional.")
    if kg.unmatched_features:
        out.append(f"NOTE: no symptom is modelled for {', '.join(kg.unmatched_features)}; "
                   f"that evidence is unused.")

    out.append("")
    out.append("Manual excerpts:")
    for h in hits:
        out.append(f"  [{h.chunk_id}] §{h.section} {h.title}")
        out.append(f"      {h.text.strip()}")

    out.append("")
    out.append("ALLOWED REFERENCES (use these ids and no others):")
    out.append("  " + ", ".join(allowed_references(kg, hits)) or "  (none)")
    out.append("")
    out.append(TASK)
    return "\n".join(out)


def allowed_references(kg: KGSubgraph, hits: list[Hit]) -> list[str]:
    """The citation allow-list: what the graph offered plus what was retrieved.

    Both, not just the retrieved chunks. The graph reaches a section because a
    curated edge says that section documents the action — which is a stronger
    reason to cite it than a lexical match, and excluding it would force the
    model to justify a bearing replacement without citing the bearing
    replacement procedure whenever the retriever ranked something else higher.

    The machine family's own alarm-response section is always allowed too:
    it is the section the fallback answer cites (kg.families.alarm_response),
    and for the generic family — whose one section a lexical retrieval may
    well miss — it is the only guidance there is.
    """
    return sorted({*kg.citable_ids, *(h.chunk_id for h in hits),
                   families.alarm_response(kg.equipment)})
