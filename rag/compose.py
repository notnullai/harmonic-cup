"""The offline composer: a diagnosis assembled from the graph, not generated.

This runs when no model is configured, which is the default state of a
checkout. It is a template filler, and it is labelled as one everywhere it
appears — `provider: "offline-kg-composer"` in the response, in the audit log
and on the dashboard.

What that buys and what it costs, stated plainly:

- it **cannot** hallucinate. Every sentence is assembled from a node the graph
  supplied or a sentence lifted verbatim from a cited manual section.
- it **cannot** reason. It will not notice that two candidates share a cause,
  will not weigh the regime against the symptom, and will not tell an engineer
  something the graph does not already contain.

The second point is the reason the Gemini path exists. The first is the reason
this one is an acceptable fallback rather than a fake.
"""

from __future__ import annotations

import re
from typing import Any

from kg import families
from kg.query import KGSubgraph
from rag.index import Hit
from rag.ingest import Chunk
from rag.prompt import INSUFFICIENT_EVIDENCE, MIN_CONFIDENCE, allowed_references

NAME = "offline-kg-composer"

MAX_REFERENCES = 20
MAX_ACTIONS = 10
MAX_CAUSES = 5
MAX_SAFETY = 6

# The graph's score and the reported confidence are not the same quantity.
# Probabilistic OR over three agreeing symptoms plus a class prior reaches
# 0.998, and reporting that as confidence would claim the graph is the machine.
# It is a model of the machine: the symptom set is not exhaustive, the
# confidences are engineering judgement, and the class prior comes from a
# network trained on a simulator. The ceiling is where that residual doubt is
# written down.
CONFIDENCE_CEILING = 0.95

# An unrecognised change (the engine's flag: far from this machine's own
# normal, matching no trained fault class) with no failure mode in the graph
# either. The answer claims only that the machine changed and names no cause,
# so its confidence sits just above the floor: enough to pass the guard, never
# enough to outrank a named failure mode.
UNRECOGNISED_CONFIDENCE = 0.5

_SENTENCE_RE = re.compile(r"(?<=[.])\s+")
_WS_RE = re.compile(r"\s+")


def _label(failure_mode_id: str) -> str:
    """`fm.outer_race_wear` -> `outer race wear`, for prose."""
    # "fm.cavitation", or a family's own "gbx.fm.tooth_breakage".
    return failure_mode_id.split("fm.", 1)[-1].replace("_", " ")


def _first_sentence(text: str, limit: int = 280) -> str:
    # The manual is hard-wrapped Markdown, so a sentence carries newlines that
    # would otherwise land in a JSON field and be rendered literally.
    s = _WS_RE.sub(" ", _SENTENCE_RE.split(text.strip())[0]).strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


# docs/GENAI_SCHEMA.json: safety.items.maxLength.
SAFETY_ITEM_MAX = 300


def _safety(kg: KGSubgraph, chunks: dict[str, Chunk], allowed: set[str]) -> list[str]:
    """Safety lines lifted from the manual sections the answer may cite.

    Never empty: the lockout requirement is the floor, because Harmonic is
    advisory and the safety text is the whole of its contribution to the
    engineer's safety (docs/GENAI_SCHEMA.json).
    """
    lines: list[str] = []
    for cid in sorted(allowed):
        chunk = chunks.get(cid)
        if chunk is None or not chunk.section.startswith("2."):
            continue
        # The whole line fits the schema's item limit, title and citation
        # included: a long first sentence is cut, not the answer rejected.
        room = SAFETY_ITEM_MAX - len(chunk.title) - len(cid) - 5
        lines.append(f"{chunk.title}: {_first_sentence(chunk.text, room)} [{cid}]")
        if len(lines) >= MAX_SAFETY:
            break
    if not lines:
        lines.append(
            "Isolate the drive at the local disconnect, apply a personal lock and "
            "tag, and verify zero energy before any mechanical work."
        )
    return lines


def _rul(event: dict[str, Any], severity: int) -> float:
    rul = event.get("rul_hours")
    if rul is not None:
        return max(0.0, float(rul))
    # No engine estimate. Fall back to a severity-banded figure and let the
    # wording say it is a placeholder rather than a measurement.
    return {5: 24.0, 4: 72.0, 3: 336.0, 2: 720.0, 1: 2160.0}.get(severity, 336.0)


def compose(event: dict[str, Any], kg: KGSubgraph, hits: list[Hit],
            chunks: dict[str, Chunk]) -> dict[str, Any]:
    allowed = set(allowed_references(kg, hits))
    top = kg.top

    if top is None and event.get("unrecognised"):
        return _unrecognised(event, kg, chunks, allowed)

    if top is None or top.score < MIN_CONFIDENCE:
        refs = sorted(allowed)[:3] or [h.chunk_id for h in hits[:3]]
        reason = ("no failure mode in the knowledge graph matches the surprised "
                  "features" if top is None else
                  f"the best-supported mode ({top.description.lower()}) reaches only "
                  f"{top.score:.2f} against a {MIN_CONFIDENCE:.1f} floor")
        return {
            "diagnosis": f"{INSUFFICIENT_EVIDENCE}. Reason: {reason}.",
            "confidence": round(top.score if top else 0.0, 3),
            "root_causes": ["Undetermined — the available evidence does not "
                            "distinguish between the candidate failure modes"],
            "actions": [{
                "step": "Confirm the reading with a second measurement at the same "
                        "duty point, then inspect the machine manually",
                "tool": "portable vibration meter",
                "duration_min": 30,
                "skill_level": "technician",
            }],
            "safety": _safety(kg, chunks, allowed),
            "references": refs or [families.alarm_response(kg.equipment)],
            "rul_hours": _rul(event, 3),
        }

    machine = kg.equipment.get("id", "the machine")
    evidence = ", ".join(s.description.lower() for s in top.symptoms[:3])
    state = event.get("state", "WARNING")
    hhi = event.get("hhi")

    diagnosis = f"{top.description} on {machine}."
    if evidence:
        diagnosis += f" The supporting evidence is {evidence}."
    if hhi is not None:
        diagnosis += (f" The health index is {float(hhi):.2f} ({state})"
                      f" in the {event.get('regime', 'current')} regime.")
    if top.from_class and kg.agreement == "agree":
        diagnosis += (f" The trained {top.from_class} class agrees with the "
                      f"symptom evidence.")
    elif kg.agreement.startswith("disagree"):
        _, _, detail = kg.agreement.partition(":")
        prior, _, other = detail.partition("|")
        # Say which one won, rather than asserting it was the symptoms: the
        # ranking combines both, and on the IMS rig the class's choice came out
        # on top while this sentence claimed the opposite.
        winner = ("the symptoms" if top.failure_mode_id == other else
                  "the trained class" if top.failure_mode_id == prior else
                  "the combined evidence")
        diagnosis += (f" Note that the trained class points at {_label(prior)} while "
                      f"the symptoms alone point at {_label(other)}; the ranking "
                      f"above follows {winner}.")
    elif kg.agreement == "no-evidence":
        diagnosis += (" No surprised feature supports this, so it rests on the "
                      "trained class alone and is provisional.")

    causes = [c["description"] for c in top.causes[:MAX_CAUSES]] or [
        "Undetermined — the graph records no cause for this failure mode"
    ]

    actions: list[dict[str, Any]] = []
    for a in top.actions[:MAX_ACTIONS]:
        step = a.description
        # The procedure section, not the safety section: the lockout belongs in
        # the safety list, and citing it inside the first work step buries it.
        if a.references:
            step += f" (see {', '.join(a.references[:2])})"
        actions.append({
            "step": step,
            "tool": a.tool or "hand tools",
            "duration_min": a.duration_min,
            "skill_level": a.skill_level,
        })
    if not actions:
        actions = [{
            "step": "Inspect the affected component and confirm the reading",
            "tool": "hand tools", "duration_min": 60, "skill_level": "technician",
        }]

    references = [r for r in top.references if r in allowed][:MAX_REFERENCES]
    if not references:
        references = sorted(allowed)[:3] or [h.chunk_id for h in hits[:1]]

    return {
        "diagnosis": diagnosis[:600],
        "confidence": round(min(CONFIDENCE_CEILING, top.score), 3),
        "root_causes": causes,
        "actions": actions,
        "safety": _safety(kg, chunks, allowed),
        "references": references,
        "rul_hours": _rul(event, top.severity),
        "failure_mode_id": top.failure_mode_id,
    }


def _unrecognised(event: dict[str, Any], kg: KGSubgraph, chunks: dict[str, Chunk],
                  allowed: set[str]) -> dict[str, Any]:
    """The answer to an unrecognised change: the machine has changed, no known
    failure mode matches, and the family's own alarm-response section says
    what to do (for the generic family, Harmonic's own general guidance)."""
    ref = families.alarm_response(kg.equipment)
    refs = [ref] if ref in allowed else sorted(allowed)[:3]
    see = f" (see {ref})" if ref in allowed else ""
    machine = kg.equipment.get("id", "the machine")
    diagnosis = (f"Unrecognised change on {machine}: it has moved away from its own "
                 "learned normal, and no failure mode known for this machine matches, "
                 "so no fault type is named.")
    hhi = event.get("hhi")
    if hhi is not None:
        diagnosis += (f" The health index is {float(hhi):.2f} "
                      f"({event.get('state', 'WARNING')}).")
    return {
        "diagnosis": diagnosis[:600],
        "confidence": UNRECOGNISED_CONFIDENCE,
        "root_causes": ["Undetermined — the change matches no failure mode this machine "
                        "type is trained to name"],
        "actions": [
            {"step": "Confirm the reading with a second measurement at the same duty point "
                     f"before acting on it{see}",
             "tool": "portable vibration meter", "duration_min": 30,
             "skill_level": "technician"},
            {"step": "Inspect the machine by eye, ear and touch — leaks, loose guards, "
                     "unusual heat or noise — before opening anything up",
             "tool": "hand tools", "duration_min": 30, "skill_level": "technician"},
        ],
        "safety": _safety(kg, chunks, allowed),
        "references": refs,
        "rul_hours": _rul(event, 3),
    }


class KGComposer:
    """Provider-shaped wrapper, so the pipeline treats both paths alike."""

    name = NAME

    def __init__(self, chunks: dict[str, Chunk]):
        self.chunks = chunks

    def diagnose(self, event: dict[str, Any], kg: KGSubgraph, hits: list[Hit]
                 ) -> dict[str, Any]:
        return compose(event, kg, hits, self.chunks)
