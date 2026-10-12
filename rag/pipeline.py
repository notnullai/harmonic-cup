"""event → KG → RAG → model → validated JSON (HARM-S4-T09).

The order matters. The graph runs first and its output shapes the retrieval
query, because the manual does not contain the words the engine uses: nothing
in a pump manual says `HARMONIC_FEAT_VIB_BPFO`. The graph is what translates a
surprised feature into "elevated amplitude at the ball-pass frequency", and
that phrase is what finds the right section.

One model call per event, per SPEC 8.1, plus at most one retry when the
guardrail rejects the first answer. If the retry also fails the pipeline
returns an explicit "insufficient evidence" rather than the rejected text: the
guardrail exists to stop an uncheckable answer reaching an engineer, and
passing it on with a warning attached would defeat it.

    python -m rag.pipeline --class bearing_wear \\
        --features HARMONIC_FEAT_VIB_BPFO HARMONIC_FEAT_TEMP_BEARING
    python -m rag.pipeline --canned          # one event per fault mode
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from kg import families
from kg.query import KGSubgraph, query_for_event
from kg.store import KGStore, open_store
from rag import audit
from rag.compose import KGComposer
from rag.guard import insufficient_evidence, validate
from rag.index import DEFAULT_K, BM25Index, Hit, Retriever, open_index
from rag.ingest import load_manifest
from rag.prompt import build_prompt, load_schema, retrieval_query
from rag.provider import LLMProvider, get_llm

log = logging.getLogger("harmonic.rag")

# Canned events, one per trained fault class. These are the acceptance cases
# for HARM-S4-T09 and they are written in the shape the API emits, not a
# convenient shape: the pipeline has to cope with what the engine sends.
CANNED_EVENTS: list[dict[str, Any]] = [
    {
        "event_id": "canned-bearing-wear",
        "equipment_id": "P-101", "regime": "NOMINAL",
        "hhi": 0.71, "state": "WARNING", "rul_hours": 42,
        "top_classes": [{"name": "bearing_wear", "score": 0.88}],
        "top_surprised_features": [
            {"kind": "HARMONIC_FEAT_VIB_BPFO", "sigma": 3.1, "direction": "up"},
            {"kind": "HARMONIC_FEAT_VIB_RMS", "sigma": 2.4, "direction": "up"},
            {"kind": "HARMONIC_FEAT_TEMP_BEARING", "sigma": 1.8, "direction": "up"},
        ],
        "expect": "fm.outer_race_wear",
    },
    {
        "event_id": "canned-misalignment",
        "equipment_id": "P-101", "regime": "NOMINAL",
        "hhi": 0.58, "state": "WARNING", "rul_hours": 320,
        "top_classes": [{"name": "misalignment", "score": 0.81}],
        "top_surprised_features": [
            {"kind": "HARMONIC_FEAT_VIB_HARM_2X", "sigma": 3.4, "direction": "up"},
            {"kind": "HARMONIC_FEAT_VIB_HARM_3X", "sigma": 2.1, "direction": "up"},
            {"kind": "HARMONIC_FEAT_TEMP_MOTOR", "sigma": 1.2, "direction": "up"},
        ],
        "expect": "fm.misalignment",
    },
    {
        "event_id": "canned-cavitation",
        "equipment_id": "P-101", "regime": "LOW_LOAD",
        "hhi": 0.66, "state": "WARNING", "rul_hours": 96,
        "top_classes": [{"name": "cavitation", "score": 0.93}],
        "top_surprised_features": [
            {"kind": "HARMONIC_FEAT_VIB_KURTOSIS", "sigma": 3.8, "direction": "up"},
            {"kind": "HARMONIC_FEAT_VIB_CREST", "sigma": 2.9, "direction": "up"},
            {"kind": "HARMONIC_FEAT_PRESS_SUCTION", "sigma": 2.2,
             "direction": "oscillating"},
        ],
        "expect": "fm.cavitation",
    },
    {
        "event_id": "canned-seal-leak",
        "equipment_id": "P-101", "regime": "NOMINAL",
        "hhi": 0.52, "state": "WARNING", "rul_hours": 210,
        "top_classes": [{"name": "seal_leak", "score": 0.77}],
        "top_surprised_features": [
            {"kind": "HARMONIC_FEAT_FLOW", "sigma": 2.7, "direction": "down"},
            {"kind": "HARMONIC_FEAT_PRESS_DISCHARGE", "sigma": 1.9, "direction": "down"},
            {"kind": "HARMONIC_FEAT_CURRENT", "sigma": 1.5, "direction": "down"},
        ],
        "expect": "fm.seal_leakage",
    },
]


@dataclasses.dataclass
class Diagnosis:
    event_id: str
    response: dict[str, Any]
    kg: KGSubgraph
    hits: list[Hit]
    provider: str
    attempts: int
    fell_back: bool
    duration_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "provider": self.provider,
            "attempts": self.attempts,
            "fell_back": self.fell_back,
            "duration_ms": round(self.duration_ms, 1),
            "top_candidate": self.kg.top.failure_mode_id if self.kg.top else None,
            "agreement": self.kg.agreement,
            "retrieved": [h.chunk_id for h in self.hits],
            "response": self.response,
        }


class Pipeline:
    """Holds the store, the index and the provider open across events."""

    def __init__(self, store: KGStore | None = None, index: Retriever | None = None,
                 llm: LLMProvider | None = None, *, k: int = DEFAULT_K,
                 audit_path: Path | str | None = None, offline: bool = False):
        self.store = store if store is not None else open_store()
        n_nodes, _n_rels = self.store.counts()
        if n_nodes == 0:
            # kg/graph.json is generated, not committed (HARM-S9-T08): a
            # fresh clone or a machine where `kg.load` hasn't run yet falls
            # back to an empty graph silently otherwise, and every event
            # gets no diagnosis without saying why.
            log.warning(
                "knowledge graph is empty (0 nodes, 0 relations, backend=%s): "
                "every event will get no diagnosis until it is loaded "
                "(python -m kg.load --rebuild)", self.store.backend,
            )
        self.chunks = {c.id: c for c in load_manifest()}
        # `offline=True` never builds an embedder either: retrieval is BM25,
        # in process, so no key in the environment can send a query to an
        # embedding model (HARM-S10-T03 M3).
        self.offline = offline
        self.index = index if index is not None else self._open_index()
        # `offline=True` never calls `get_llm()`, so no environment variable
        # or key present at runtime can turn a call to a model on: the only
        # path to Gemini is through `get_llm()` itself. Used by try-mode
        # (api/try_mode.py), where the owner's decision is that a model is
        # never reachable, whatever the deployment's environment says.
        self.llm = None if offline else (llm if llm is not None else get_llm())
        self.composer = KGComposer(self.chunks)
        self.schema = load_schema()
        self.k = k
        self.audit_path = Path(audit_path) if audit_path else audit.DEFAULT_LOG

    def add_chunks(self, chunks: list[Any]) -> int:
        """Sections of a manual uploaded while the service runs (HARM-S7-T11):
        into the chunk table, the composer and a rebuilt index. Returns how many
        the index now holds."""
        if not chunks:
            return len(self.chunks)
        self.chunks.update({c.id: c for c in chunks})
        self.composer = KGComposer(self.chunks)
        self.index = self._open_index()
        return len(self.chunks)

    def _open_index(self) -> Retriever:
        chunks = list(self.chunks.values())
        return BM25Index(chunks) if self.offline else open_index(chunks)

    @property
    def provider_name(self) -> str:
        return self.llm.name if self.llm is not None else self.composer.name

    def diagnose(self, event: dict[str, Any]) -> Diagnosis:
        t0 = time.perf_counter()
        record = audit.start(event, {
            "generation": self.provider_name,
            "retrieval": self.index.name,
            "graph": self.store.backend,
        })

        kg = query_for_event(event, self.store)
        query = retrieval_query(event, kg)
        hits = self._search(query, kg) if query.strip() else []
        prompt = build_prompt(event, kg, hits)
        from rag.prompt import allowed_references
        allowed = set(allowed_references(kg, hits))

        record.kg = kg.to_json()
        record.retrieval = [{"chunk_id": h.chunk_id, "score": h.score,
                             "title": h.title} for h in hits]
        record.allowed_references = sorted(allowed)
        record.prompt = prompt

        response, attempts, fell_back = self._answer(prompt, allowed, kg, hits,
                                                     event, record)

        duration = (time.perf_counter() - t0) * 1000.0
        record.final = response
        record.outcome = "fallback" if fell_back else "ok"
        record.duration_ms = duration
        audit.write(record, self.audit_path)

        return Diagnosis(record.event_id, response, kg, hits, self.provider_name,
                         attempts, fell_back, duration)

    def retrieve(self, query: str, kg: KGSubgraph) -> list[Hit]:
        """This machine's manual sections for a query (the chat's retrieval)."""
        return self._search(query, kg)

    def _search(self, query: str, kg: KGSubgraph) -> list[Hit]:
        """Retrieve from this machine's own manuals only: the equipment's
        `manuals`, or its family's. A gearbox alarm answered from the pump
        manual would cite a procedure for a machine nobody is standing at."""
        eq = kg.equipment
        docs = [d for d in str(eq.get("manuals") or "").split(",") if d.strip()]
        if not docs:
            try:
                docs = list(families.get(str(eq.get("family") or "pump")).get("manuals", []))
            except ValueError:
                docs = []
        # No manual known for this machine or its family (the generic family
        # has none): nothing is retrieved, rather than searching every
        # family's manual unrestricted and citing one for a machine nobody is
        # standing at.
        if not docs:
            return []
        hits = self.index.search(query, self.k * 4)
        hits = [h for h in hits if (c := self.chunks.get(h.chunk_id)) and c.doc in docs]
        return hits[: self.k]

    def _answer(self, prompt: str, allowed: set[str], kg: KGSubgraph,
                hits: list[Hit], event: dict[str, Any], record: audit.AuditRecord
                ) -> tuple[dict[str, Any], int, bool]:
        # The composer is deterministic and validated the same way, but it has
        # nothing to retry with: a rejected composition means the graph is
        # wrong, and asking it again produces the same bytes.
        if self.llm is None:
            response = self.composer.diagnose(event, kg, hits)
            verdict = validate(response, allowed, self.schema)
            record.attempts.append({"provider": self.composer.name,
                                    "verdict": verdict.to_json(),
                                    "response": response})
            if verdict.ok:
                return response, 1, False
            reason = verdict.reasons[0] if verdict.reasons else "composition rejected"
            return insufficient_evidence(allowed, reason,
                                         families.alarm_response(kg.equipment),
                                         float(event.get("rul_hours") or 168.0)), 1, True

        attempts = 0
        current = prompt
        for _ in range(2):          # one call, then at most one retry (SPEC 8.1)
            attempts += 1
            try:
                response = self.llm.generate_json(current, self.schema)
            except Exception as exc:
                record.attempts.append({"provider": self.llm.name, "error": repr(exc)})
                break
            verdict = validate(response, allowed, self.schema)
            record.attempts.append({"provider": self.llm.name,
                                    "verdict": verdict.to_json(),
                                    "response": response})
            if verdict.ok:
                return response, attempts, False
            # A low-confidence answer is the model doing what it was asked; a
            # retry would only pressure it into overstating.
            if verdict.low_confidence and len(verdict.reasons) == 1:
                break
            current = f"{prompt}\n\n{verdict.as_feedback()}"

        reason = ("the model's answer did not survive the citation and schema "
                  "checks after one retry")
        return insufficient_evidence(allowed, reason,
                                     families.alarm_response(kg.equipment),
                                     float(event.get("rul_hours") or 168.0)), attempts, True

    def close(self) -> None:
        self.store.close()


def _summarise(d: Diagnosis, expect: str | None) -> tuple[str, bool]:
    got = d.response.get("failure_mode_id") or (
        d.kg.top.failure_mode_id if d.kg.top else None)
    ok = expect is None or got == expect
    mark = "PASS" if ok else "FAIL"
    return (f"  {mark}  {d.event_id:<22} top-1 {got or '-':<24} "
            f"conf {d.response.get('confidence', 0):.2f}  "
            f"refs {len(d.response.get('references', []))}  "
            f"{'FALLBACK ' if d.fell_back else ''}{d.provider}"), ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose an engine event")
    ap.add_argument("--event", help="event JSON file, or - for stdin")
    ap.add_argument("--class", dest="cls", help="trained class name")
    ap.add_argument("--features", nargs="*", default=[])
    ap.add_argument("--canned", action="store_true",
                    help="run the four acceptance events, one per fault mode")
    ap.add_argument("--backend", choices=["auto", "neo4j", "networkx"], default="auto")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    ap.add_argument("--audit-log", default=None)
    ap.add_argument("--json", action="store_true", help="print the full response")
    args = ap.parse_args(argv)

    pipeline = Pipeline(store=open_store(args.backend), k=args.k,
                        audit_path=args.audit_log)
    try:
        if args.canned:
            print(f"providers: generation={pipeline.provider_name} "
                  f"retrieval={pipeline.index.name} graph={pipeline.store.backend}")
            failures = 0
            for event in CANNED_EVENTS:
                expect = event.get("expect")
                d = pipeline.diagnose({k: v for k, v in event.items() if k != "expect"})
                line, ok = _summarise(d, expect)
                failures += not ok
                print(line)
                if args.json:
                    print(json.dumps(d.response, indent=2))
            print(f"  RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
            print(f"  audit log: {pipeline.audit_path}")
            return 1 if failures else 0

        if args.event:
            text = sys.stdin.read() if args.event == "-" else Path(args.event).read_text("utf-8")
            event = json.loads(text)
        else:
            event = {"top_surprised_features": args.features}
            if args.cls:
                event["top_classes"] = [{"name": args.cls, "score": 1.0}]
        d = pipeline.diagnose(event)
        print(json.dumps(d.to_json(), indent=2))
        return 0
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
