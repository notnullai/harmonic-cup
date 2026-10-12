"""The citation guardrail (HARM-S4-T08, SPEC 8.4).

Three checks, in order, and all three run even when the provider enforced a
response schema on its side:

1. **Schema.** Re-validated locally against docs/GENAI_SCHEMA.json. A schema
   the service applied is evidence about the shape of the reply, not about the
   content, and it is not evidence at all for the offline composer.
2. **Citations.** Every id in `references` must be in the allow-list built from
   the KG subgraph and the retrieved chunks. This is the anti-hallucination
   boundary: a model that invents "§7.4 Bearing torque values" produces an
   answer an engineer cannot check, and an answer an engineer cannot check is
   worse than no answer, because it will be believed.
3. **Confidence.** Below 0.4 the answer is discarded in favour of an explicit
   "insufficient evidence". A confident-sounding guess is the failure mode of
   this whole layer.

A rejected response is retried once with the failure quoted back, then falls
back. The retry is not optimism — a model told exactly which id was not in the
list usually drops it — but one retry is the limit, because a second failure
means the context is wrong rather than the sampling.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from rag.prompt import INSUFFICIENT_EVIDENCE, MIN_CONFIDENCE, load_schema

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema is a declared dependency
    jsonschema = None


@dataclasses.dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...] = ()
    bad_references: tuple[str, ...] = ()
    low_confidence: bool = False

    def as_feedback(self) -> str:
        """What to tell the model on the retry. Specific, not 'try again'."""
        lines = ["Your previous response was rejected:"]
        lines.extend(f"  - {r}" for r in self.reasons)
        if self.bad_references:
            lines.append(
                "  Remove these ids from `references`; they are not in the "
                "ALLOWED REFERENCES list: " + ", ".join(self.bad_references)
            )
        lines.append("Return corrected JSON. Do not add any id that is not allowed.")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "bad_references": list(self.bad_references),
            "low_confidence": self.low_confidence,
        }


def validate(response: dict[str, Any], allowed: set[str] | list[str],
             schema: dict[str, Any] | None = None) -> Verdict:
    allowed = set(allowed)
    reasons: list[str] = []

    schema = schema if schema is not None else load_schema()
    if jsonschema is not None:
        validator = jsonschema.Draft202012Validator(schema)
        for err in sorted(validator.iter_errors(response), key=lambda e: list(e.path)):
            path = "/".join(str(p) for p in err.path) or "(root)"
            reasons.append(f"schema: {path}: {err.message}")
    elif not isinstance(response, dict) or "diagnosis" not in response:
        reasons.append("schema: response is not an object with a diagnosis")

    refs = response.get("references") or []
    bad = tuple(sorted({str(r) for r in refs if str(r) not in allowed}))
    if bad:
        reasons.append(
            f"citations: {len(bad)} reference(s) are not in the allow-list: "
            + ", ".join(bad)
        )

    low = False
    try:
        low = float(response.get("confidence", 0.0)) < MIN_CONFIDENCE
    except (TypeError, ValueError):
        reasons.append("confidence: not a number")
    if low:
        reasons.append(
            f"confidence: {response.get('confidence')} is below the "
            f"{MIN_CONFIDENCE} floor, so the answer is withheld"
        )

    return Verdict(ok=not reasons, reasons=tuple(reasons),
                   bad_references=bad, low_confidence=low)


def insufficient_evidence(allowed: set[str] | list[str], reason: str,
                          fallback_ref: str,
                          rul_hours: float = 168.0) -> dict[str, Any]:
    """The fallback answer. Schema-valid, so nothing downstream special-cases it.
    `fallback_ref` is cited when nothing else is: the machine family's
    alarm-response section (kg.families.alarm_response)."""
    refs = sorted(allowed)[:3] or [fallback_ref]
    return {
        "diagnosis": f"{INSUFFICIENT_EVIDENCE}. {reason}"[:600],
        "confidence": 0.0,
        "root_causes": ["Undetermined — the evidence available does not support "
                        "a specific failure mode"],
        "actions": [{
            "step": "Confirm the reading with a second measurement at the same duty "
                    "point, then inspect the machine manually before acting",
            "tool": "portable vibration meter",
            "duration_min": 30,
            "skill_level": "technician",
        }],
        "safety": ["Isolate the drive at the local disconnect, apply a personal lock "
                   "and tag, and verify zero energy before any mechanical work."],
        "references": refs,
        "rul_hours": rul_hours,
    }
