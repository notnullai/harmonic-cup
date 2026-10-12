"""Grounded question answering (HARM-S5-T05).

`POST /chat` differs from a diagnosis in what starts it, not in what grounds
it. A diagnosis begins with an event and asks "what is wrong"; a chat turn
begins with a question and asks "what does the manual say about this, given
what the machine is doing right now". Both are answered only from the
knowledge graph and the retrieved manual sections, and both go through the
same citation guardrail.

The retrieval query is the question plus the words the graph uses for the
machine's current evidence. Searching on the question alone answers "how do I
replace a bearing" out of context; adding the live symptoms lets "what is
wrong with the pump?" find the sections about what is actually happening.

The answer schema is narrower than a diagnosis: an answer, its citations and a
confidence. A chat turn that returned a work order every time would bury the
question.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from api.feature_labels import label
from kg.query import KGSubgraph, query_for_event
from rag import audit
from rag.guard import Verdict
from rag.index import Hit
from rag.pipeline import Pipeline
from rag.prompt import MIN_CONFIDENCE, allowed_references

MAX_QUESTION_CHARS = 1000

CHAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answer", "confidence", "references"],
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "minLength": 1, "maxLength": 1500},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "references": {
            "type": "array", "minItems": 0, "maxItems": 20,
            "items": {"type": "string", "pattern": "^[A-Za-z0-9._-]+$"},
        },
    },
}

SYSTEM = """You are a senior rotating-equipment reliability engineer answering a
maintenance engineer's question about one machine.

Use ONLY the context below. Never invent manual sections, part numbers or
measurements. Every id in `references` must appear in ALLOWED REFERENCES. If
the context does not answer the question, say so plainly and say what would —
a measurement, an inspection, a section that is not in this manual. Do not
guess.

The MACHINE STATE is the engine's verdict. When it says there is no alarm,
the machine is operating normally: say so, and do not name a fault. Small
movements below alarm level are normal variation, not evidence. Write for a
maintenance engineer in plain words; never quote internal feature ids.

Return JSON: {"answer": string, "confidence": number 0..1, "references": [ids]}"""

NO_ANSWER = ("The manual sections available do not answer that. ")


@dataclasses.dataclass
class ChatTurn:
    question: str
    answer: str
    confidence: float
    references: list[str]
    provider: str
    retrieved: list[str]
    candidates: list[str]
    grounded: bool
    duration_ms: float

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _alarmed(snapshot: dict[str, Any] | None) -> bool:
    """Whether the engine sees anything: a state above OK, or an unrecognised
    change. Below that, what moved is the machine's normal variation."""
    if not snapshot:
        return False
    return snapshot.get("state", "OK") != "OK" or bool(snapshot.get("unrecognised"))


def _state_event(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """The machine's current state, in the shape the graph query wants. With
    no alarm there is no evidence to query on: a graph asked about noise
    returns a failure mode anyway, and the answer then names a fault the
    engine does not see."""
    if not snapshot:
        return {"top_surprised_features": []}
    alarmed = _alarmed(snapshot)
    return {
        "equipment_id": snapshot.get("equipment_id", "P-101"),
        "regime": snapshot.get("regime"),
        "hhi": snapshot.get("hhi"),
        "state": snapshot.get("state"),
        "rul_hours": snapshot.get("rul_hours"),
        "top_classes": snapshot.get("top_classes", []) if alarmed else [],
        "top_surprised_features": snapshot.get("top_surprised_features", []) if alarmed else [],
    }


def _context(question: str, kg: KGSubgraph, hits: list[Hit],
             snapshot: dict[str, Any] | None) -> str:
    out = ["SYSTEM", SYSTEM, "", "MACHINE STATE"]
    if snapshot:
        family = str(snapshot.get("family") or "pump")
        machine = snapshot.get("equipment_id", "the machine")
        out.append(f"  {machine}: {snapshot.get('state', 'UNKNOWN')} at health index "
                   f"{snapshot.get('hhi', 'n/a')} in the "
                   f"{snapshot.get('regime', 'UNKNOWN')} regime; overall vibration "
                   f"{snapshot.get('vib_rms_mm_s', 'n/a')} mm/s")
        if not _alarmed(snapshot):
            out.append("  No alarm: the engine sees no fault. The machine is operating "
                       "normally; nothing below is a symptom.")
        else:
            if snapshot.get("spread"):
                out.append("  An unrecognised change: many features moved a little together, "
                           "matching no trained fault.")
            elif snapshot.get("unrecognised"):
                out.append("  An unrecognised change: far from normal, matching no trained fault.")
            for f in snapshot.get("top_surprised_features", [])[:4]:
                out.append(f"  moved {f.get('direction', '')}: "
                           f"{label(str(f.get('kind', '')), family)}")
    else:
        out.append("  (the engine is not running; answer from the manual alone)")

    out.append("")
    out.append("KG candidates:")
    for c in kg.candidates[:3]:
        out.append(f"  - {c.failure_mode_id}: {c.description} (score {c.score:.2f})")
        for cause in c.causes[:3]:
            out.append(f"      cause: {cause['description']}")
        for a in c.actions[:3]:
            out.append(f"      action: {a.description} [{a.tool or 'no special tool'}, "
                       f"{a.duration_min:.0f} min, {a.skill_level}]")
    if not kg.candidates:
        out.append("  (none)")

    out.append("")
    eq = str((snapshot or {}).get("equipment_id") or "")
    docs = sorted({h.chunk_id.rsplit("-", 1)[0] for h in hits})
    if eq and docs and not any(d.startswith(eq) for d in docs):
        out.append(f"Manual excerpts (this machine has no manual of its own: these are from "
                   f"{', '.join(docs)}, the reference manual of its family; say so if you "
                   f"cite them):")
    else:
        out.append("Manual excerpts:")
    for h in hits:
        out.append(f"  [{h.chunk_id}] §{h.section} {h.title}")
        out.append(f"      {h.text.strip()}")

    out.append("")
    out.append("ALLOWED REFERENCES (use these ids and no others):")
    out.append("  " + ", ".join(allowed_references(kg, hits)))
    out.append("")
    out.append(f"QUESTION\n{question}")
    return "\n".join(out)


def _compose_offline(question: str, kg: KGSubgraph, hits: list[Hit],
                     snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """An answer assembled from what was retrieved, with no model.

    It quotes rather than explains — the opening of each relevant section, in
    rank order, with its id. That is genuinely useful (it is what an engineer
    would do with a search box) and it is not dressed up as understanding.
    """
    if not hits:
        return {"answer": NO_ANSWER + "Nothing in the manual matched the question.",
                "confidence": 0.0, "references": []}

    lines: list[str] = []
    if snapshot and not _alarmed(snapshot):
        lines.append(f"The engine raises no alarm: {snapshot.get('equipment_id', 'the machine')} "
                     f"is operating normally. The manual sections that bear on the question:")
    elif snapshot and kg.top is not None:
        lines.append(
            f"The machine is {snapshot.get('state', 'in an unknown state')} and the "
            f"best-supported failure mode is {kg.top.description.lower()} "
            f"(score {kg.top.score:.2f}). The manual sections that bear on the "
            f"question:"
        )
    else:
        lines.append("From the manual sections that match the question:")

    for h in hits[:3]:
        first = " ".join(h.text.split())
        cut = first.find(". ")
        summary = first[: cut + 1] if 0 < cut < 320 else first[:320].rstrip() + "…"
        lines.append(f"\n§{h.section} {h.title} [{h.chunk_id}]\n{summary}")

    lines.append(
        "\n(Assembled from the retrieved sections by the offline composer, which "
        "quotes the manual rather than reasoning about it. Configure a model for "
        "a written answer.)"
    )
    return {
        "answer": "\n".join(lines)[:1500],
        # A quotation is only as confident as the retrieval behind it, and BM25
        # scores are not probabilities. 0.5 says "relevant, not reasoned".
        "confidence": 0.5,
        "references": [h.chunk_id for h in hits[:5]],
    }


def _validate(response: dict[str, Any], allowed: set[str]) -> Verdict:
    from rag.guard import validate
    return validate(response, allowed, CHAT_SCHEMA)


def ask(question: str, pipeline: Pipeline,
        snapshot: dict[str, Any] | None = None, *, k: int = 5) -> ChatTurn:
    """One grounded turn."""
    t0 = time.perf_counter()
    question = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not question:
        raise ValueError("a question is required")

    event = _state_event(snapshot)
    kg = query_for_event(event, pipeline.store)

    # The question, plus the words the graph uses for what is happening now.
    # The question alone answers out of context; the state alone ignores what
    # was asked.
    terms = [question]
    for c in kg.candidates[:2]:
        terms.append(c.description)
        terms.extend(s.description for s in c.symptoms[:2])
    # This machine's own manual sections, as a diagnosis retrieves them.
    hits = pipeline.retrieve(" ".join(terms), kg)[:k]
    allowed = set(allowed_references(kg, hits))

    record = audit.start({"event_id": f"chat-{int(t0 * 1000)}", "question": question},
                         {"generation": pipeline.provider_name,
                          "retrieval": pipeline.index.name,
                          "graph": pipeline.store.backend})
    record.kg = kg.to_json()
    record.retrieval = [{"chunk_id": h.chunk_id, "score": h.score, "title": h.title}
                        for h in hits]
    record.allowed_references = sorted(allowed)

    if pipeline.llm is None:
        response = _compose_offline(question, kg, hits, snapshot)
        provider = pipeline.composer.name
        record.prompt = "(offline composer; no prompt was sent)"
    else:
        prompt = _context(question, kg, hits, snapshot)
        record.prompt = prompt
        provider = pipeline.llm.name
        try:
            response = pipeline.llm.generate_json(prompt, CHAT_SCHEMA)
        except Exception as exc:
            record.attempts.append({"provider": provider, "error": repr(exc)})
            response = _compose_offline(question, kg, hits, snapshot)
            provider = f"{provider}->fallback"

    verdict = _validate(response, allowed)
    record.attempts.append({"provider": provider, "verdict": verdict.to_json(),
                            "response": response})

    grounded = True
    if verdict.bad_references:
        # Strip the invented ids rather than discarding the whole answer: a
        # chat turn is advisory and its value survives losing a citation, while
        # a diagnosis is a work order and does not.
        response["references"] = [r for r in response.get("references", [])
                                  if r in allowed]
        grounded = False
    if float(response.get("confidence", 0.0)) < MIN_CONFIDENCE and not response.get("answer"):
        response["answer"] = NO_ANSWER + "Recommend a manual inspection."
        grounded = False

    duration = (time.perf_counter() - t0) * 1000.0
    record.final = response
    record.outcome = "ok" if grounded else "degraded"
    record.duration_ms = duration
    audit.write(record, pipeline.audit_path)

    return ChatTurn(
        question=question,
        answer=str(response.get("answer", "")),
        confidence=float(response.get("confidence", 0.0)),
        references=list(response.get("references", [])),
        provider=provider,
        retrieved=[h.chunk_id for h in hits],
        candidates=[c.failure_mode_id for c in kg.candidates],
        grounded=grounded,
        duration_ms=round(duration, 1),
    )
