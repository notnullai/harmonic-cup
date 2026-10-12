"""Retrieval, prompt, guardrail and pipeline (HARM-S4-T06..T10).

The guardrail tests carry most of the weight. It is the boundary between an
answer an engineer can check and an answer they cannot, and the failure mode it
guards against — a fabricated manual section — is the one that looks most
convincing when it happens.
"""

from __future__ import annotations

import json

import pytest

from kg import load, query
from kg.store import NetworkXStore
from rag import audit, guard
from rag.compose import compose
from rag.index import BM25Index, tokenise
from rag.ingest import load_manifest
from rag.pipeline import CANNED_EVENTS, Pipeline
from rag.prompt import MIN_CONFIDENCE, allowed_references, build_prompt, load_schema


@pytest.fixture(scope="module")
def chunks():
    return load_manifest()


@pytest.fixture(scope="module")
def index(chunks):
    return BM25Index(chunks)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = NetworkXStore(tmp_path_factory.mktemp("rag-kg") / "graph.json")
    s.merge_graph(load.build_graph())
    return s


@pytest.fixture(scope="module")
def pipeline(store, index, tmp_path_factory):
    return Pipeline(store=store, index=index, llm=None,
                    audit_path=tmp_path_factory.mktemp("audit") / "log.jsonl")


# --- retrieval ----------------------------------------------------------------


def _in_doc(index, q: str, doc: str, k: int = 3) -> list[str]:
    """Retrieval as the pipeline does it: from one machine's manual. The index
    holds every manual, so an unscoped "bearing replacement" may as well find
    the gearbox's (HARM-S7-T10)."""
    return [h.chunk_id for h in index.search(q, k=k * 8)
            if h.chunk_id.startswith(doc + "-")][:k]


def test_bearing_replacement_retrieves_section_4_2(index):
    """HARM-S4-T06's acceptance case."""
    hits = _in_doc(index, "bearing replacement", "P-101-OM", k=5)
    assert hits and hits[0] == "P-101-OM-4.2", hits


@pytest.mark.parametrize(("q", "expected"), [
    ("how do I align the shafts", "P-101-OM-4.4"),
    ("cavitation kurtosis crest factor", "P-101-OM-5.8"),
    ("mechanical seal leaking, flow has dropped", "P-101-OM-5.9"),
    ("grease type and regrease interval", "P-101-OM-3.1"),
    ("vibration severity zones", "P-101-OM-5.1"),
])
def test_each_fault_topic_retrieves_its_section(index, q, expected):
    hits = _in_doc(index, q, expected.rsplit("-", 1)[0])
    assert hits[0] == expected, hits


@pytest.mark.parametrize(("q", "expected"), [
    ("gear backlash measurement", "GB-201-OM-4.6"),
    ("broken tooth impulsive sidebands", "GB-201-OM-5.5"),
    ("gear oil change interval", "GB-201-OM-3.1"),
])
def test_the_gearbox_manual_answers_gearbox_questions(index, q, expected):
    hits = _in_doc(index, q, "GB-201-OM")
    assert hits[0] == expected, hits


def test_a_gearbox_alarm_is_diagnosed_from_the_gearbox_s_own_graph_and_manual(pipeline):
    """GB-201 is a gearbox: a broken tooth, cited from GB-201-OM only — never
    the pump's cavitation, never the pump's manual (HARM-S7-T10)."""
    event = {"equipment_id": "GB-201", "regime": "NOMINAL", "state": "WARNING",
             "hhi": 0.6, "rul_hours": 300.0,
             "top_classes": [{"name": "broken_tooth", "score": 0.8}],
             "top_surprised_features": [
                 {"kind": "HARMONIC_FEAT_VIB_ORDER:mesh_sb", "direction": "up", "weight": 1.0},
                 {"kind": "HARMONIC_FEAT_VIB_KURTOSIS", "direction": "up", "weight": 0.8}]}
    d = pipeline.diagnose(event)
    assert d.kg.candidates[0].failure_mode_id == "gbx.fm.tooth_breakage"
    assert all(c.failure_mode_id.startswith("gbx.") for c in d.kg.candidates)
    assert d.hits and all(h.chunk_id.startswith("GB-201-OM-") for h in d.hits)
    refs = d.response.get("references", [])
    assert refs and all(r.startswith("GB-201-OM-") for r in refs), refs


def test_the_variant_table_bridges_leak_and_leakage():
    """Lexical retrieval matches words, so "leaking" has to reach "leakage".

    A stemmer would do it and would also turn "bearing" into "bear", which is
    most of the signal in this corpus — hence a short domain table instead.
    """
    assert tokenise("leaking") == tokenise("leakage") == ["leak"]
    assert "bearing" in tokenise("the bearings")


def test_retrieval_of_an_unrelated_query_returns_nothing_rather_than_noise(index):
    assert index.search("quarterly revenue forecast", k=5) == []


# --- prompt -------------------------------------------------------------------


def test_the_prompt_carries_the_allow_list_and_the_evidence(store, index):
    event = CANNED_EVENTS[0]
    kg = query.query_for_event(event, store)
    hits = index.search("outer race wear bearing", k=5)
    prompt = build_prompt(event, kg, hits)
    assert "ALLOWED REFERENCES" in prompt
    assert "fm.outer_race_wear" in prompt
    assert "P-101-OM-4.2" in prompt
    # The engine's own numbers have to be in front of the model.
    assert "0.71" in prompt and "WARNING" in prompt


def test_the_prompt_says_when_the_spread_floor_raised_the_alarm(store, index):
    """A spread-floor alarm (HARM-S11-T12) is unrecognised too, but its
    evidence is many features moved a little: the model is told not to rank
    on the single most surprised one."""
    event = {**CANNED_EVENTS[0], "unrecognised": True, "spread": True}
    kg = query.query_for_event(event, store)
    prompt = build_prompt(event, kg, index.search("bearing", k=3))
    assert "Unrecognised change: no trained fault class accounts for this level" in prompt
    assert "Spread change: many features moved a little together" in prompt
    plain = build_prompt({**CANNED_EVENTS[0], "unrecognised": True}, kg, [])
    assert "Spread change" not in plain


def test_the_bridge_event_carries_the_spread_flag():
    from api.bridge import Event

    fields = {f: None for f in Event.__dataclass_fields__}
    ev = Event(**{**fields, "id": "ev-1", "equipment_id": "P-101", "regime": "NOMINAL",
                  "hhi": 0.5, "state": "WARNING", "rul_hours": 10.0, "top_classes": [],
                  "top_surprised_features": [], "unrecognised": True, "spread": True,
                  "early": False, "point": "", "episode": "ep-1", "diagnosis_of": None})
    out = ev.as_engine_event()
    assert out["spread"] is True and out["unrecognised"] is True
    ev.spread = False
    assert "spread" not in ev.as_engine_event()


def test_the_prompt_says_when_the_class_and_the_evidence_disagree(store, index):
    event = {"top_classes": [{"name": "cavitation", "score": 0.9}],
             "top_surprised_features": [{"kind": "HARMONIC_FEAT_VIB_BPFO", "sigma": 3.0}]}
    kg = query.query_for_event(event, store)
    prompt = build_prompt(event, kg, index.search("bearing", k=3))
    assert "NOTE: the trained class points at" in prompt


# --- guardrail ----------------------------------------------------------------


def _valid_response(refs: list[str]) -> dict:
    return {
        "diagnosis": "Outer race spalling in the drive-end bearing on P-101.",
        "confidence": 0.85,
        "root_causes": ["Loss or degradation of the bearing grease"],
        "actions": [{"step": "Replace the drive-end bearing", "tool": "bearing puller",
                     "duration_min": 180, "skill_level": "technician"}],
        "safety": ["Isolate, lock and tag before any mechanical work."],
        "references": refs,
        "rul_hours": 42,
    }


def test_a_valid_response_passes():
    assert guard.validate(_valid_response(["P-101-OM-4.2"]), {"P-101-OM-4.2"}).ok


def test_an_invented_manual_section_is_rejected():
    """The whole point. A fabricated citation is the most convincing kind of
    hallucination, because it looks exactly like a checkable claim."""
    verdict = guard.validate(_valid_response(["P-101-OM-7.4"]), {"P-101-OM-4.2"})
    assert not verdict.ok
    assert verdict.bad_references == ("P-101-OM-7.4",)
    assert "P-101-OM-7.4" in verdict.as_feedback()


def test_a_response_missing_a_required_field_is_rejected():
    bad = _valid_response(["P-101-OM-4.2"])
    del bad["safety"]
    verdict = guard.validate(bad, {"P-101-OM-4.2"})
    assert not verdict.ok
    assert any("safety" in r for r in verdict.reasons)


def test_an_empty_safety_list_is_rejected():
    """Never empty, per docs/GENAI_SCHEMA.json: the safety text is the whole of
    an advisory system's contribution to the engineer's safety."""
    bad = _valid_response(["P-101-OM-4.2"])
    bad["safety"] = []
    assert not guard.validate(bad, {"P-101-OM-4.2"}).ok


def test_a_low_confidence_answer_is_withheld():
    bad = _valid_response(["P-101-OM-4.2"])
    bad["confidence"] = 0.2
    verdict = guard.validate(bad, {"P-101-OM-4.2"})
    assert not verdict.ok and verdict.low_confidence


def test_the_fallback_is_itself_schema_valid():
    """So nothing downstream needs a special case for it."""
    fallback = guard.insufficient_evidence({"P-101-OM-4.2"}, "no candidate matched",
                                            "HARMONIC-GEN-1")
    verdict = guard.validate(fallback, {"P-101-OM-4.2"})
    # It is deliberately below the confidence floor, which the validator flags;
    # everything else about it must be clean.
    assert verdict.low_confidence
    assert [r for r in verdict.reasons if not r.startswith("confidence")] == []


# --- composer -----------------------------------------------------------------


def test_the_composer_only_cites_what_it_is_allowed_to(store, index, chunks):
    by_id = {c.id: c for c in chunks}
    for event in CANNED_EVENTS:
        kg = query.query_for_event(event, store)
        hits = index.search("bearing seal alignment cavitation", k=5)
        response = compose(event, kg, hits, by_id)
        allowed = set(allowed_references(kg, hits))
        assert set(response["references"]) <= allowed, event["event_id"]


def test_the_composer_withholds_when_the_evidence_is_thin(store, index, chunks):
    """Below the confidence floor it must say so rather than name a mode."""
    by_id = {c.id: c for c in chunks}
    event = {"top_surprised_features": ["HARMONIC_FEAT_ACOUSTIC_RMS"]}
    kg = query.query_for_event(event, store)
    response = compose(event, kg, index.search("acoustic", k=3), by_id)
    assert "insufficient evidence" in response["diagnosis"]
    assert response["confidence"] < MIN_CONFIDENCE


def test_the_composer_never_claims_certainty(store, index, chunks):
    """Probabilistic OR over agreeing symptoms reaches 0.998. Reporting that
    would claim the graph is the machine rather than a model of it."""
    by_id = {c.id: c for c in chunks}
    for event in CANNED_EVENTS:
        kg = query.query_for_event(event, store)
        response = compose(event, kg, index.search("bearing", k=5), by_id)
        assert response["confidence"] <= 0.95


def test_a_generic_machine_s_insufficient_evidence_cites_only_its_own_guidance(chunks):
    """The generic family (HARM-S10-T01) has no manual beyond Harmonic's own
    generic-guidance section. When nothing survives the confidence floor, the
    citation must be `HARMONIC-GEN-1`, never the pump's manual that used to be
    the silent default for any family without an `alarm_response`."""
    by_id = {c.id: c for c in chunks}
    kg = query.KGSubgraph(equipment={"id": "GEN-1", "family": "generic"},
                          candidates=(), citable_ids=(),
                          unmatched_features=("HARMONIC_FEAT_VIB_RMS",),
                          agreement="none")
    event = {"equipment_id": "GEN-1",
             "top_surprised_features": ["HARMONIC_FEAT_VIB_RMS"]}
    response = compose(event, kg, [], by_id)
    assert "insufficient evidence" in response["diagnosis"]
    assert response["references"] == ["HARMONIC-GEN-1"]
    assert not any(r.startswith("P-101") for r in response["references"])


def test_safety_text_is_lifted_from_a_cited_section(store, index, chunks):
    by_id = {c.id: c for c in chunks}
    kg = query.query_for_event(CANNED_EVENTS[0], store)
    response = compose(CANNED_EVENTS[0], kg, index.search("bearing", k=5), by_id)
    assert response["safety"]
    assert any("P-101-OM-2." in line for line in response["safety"])
    assert all("\n" not in line for line in response["safety"])


# --- pipeline -----------------------------------------------------------------


def test_the_four_canned_events_each_diagnose_correctly(pipeline):
    """HARM-S4-T09's acceptance case: all four return the right mode top-1."""
    for event in CANNED_EVENTS:
        expected = event["expect"]
        d = pipeline.diagnose({k: v for k, v in event.items() if k != "expect"})
        assert d.response.get("failure_mode_id") == expected, d.response["diagnosis"]
        assert not d.fell_back, d.response["diagnosis"]


def test_every_pipeline_response_survives_its_own_guardrail(pipeline):
    schema = load_schema()
    for event in CANNED_EVENTS:
        d = pipeline.diagnose({k: v for k, v in event.items() if k != "expect"})
        allowed = set(allowed_references(d.kg, d.hits))
        assert guard.validate(d.response, allowed, schema).ok, d.response


def test_the_pipeline_reports_which_provider_answered(pipeline):
    d = pipeline.diagnose(CANNED_EVENTS[0] | {"expect": None})
    assert d.provider == "offline-kg-composer"


def test_an_event_with_no_usable_evidence_falls_back(pipeline):
    d = pipeline.diagnose({"event_id": "empty", "top_surprised_features": []})
    assert "insufficient evidence" in d.response["diagnosis"]


def test_an_empty_graph_is_logged_as_a_warning_at_start_up(index, tmp_path_factory, caplog):
    """HARM-S9-T08: `kg/graph.json` missing (a fresh clone; it's generated,
    not committed) must not silently run every event with no diagnosis."""
    empty_store = NetworkXStore(tmp_path_factory.mktemp("empty-kg") / "graph.json")
    assert empty_store.counts() == (0, 0)
    with caplog.at_level("WARNING", logger="harmonic.rag"):
        Pipeline(store=empty_store, index=index, llm=None,
                 audit_path=tmp_path_factory.mktemp("audit-empty") / "log.jsonl")
    assert any("graph is empty" in r.message for r in caplog.records)


# --- audit --------------------------------------------------------------------


def test_every_diagnosis_leaves_an_audit_record(pipeline):
    """HARM-S4-T10. The guardrail is only meaningful if the reasoning survives
    for someone to check afterwards."""
    pipeline.diagnose({"event_id": "audit-probe",
                       "top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"]})
    rows = audit.read(pipeline.audit_path)
    record = next(r for r in rows if r["event_id"] == "audit-probe")
    assert record["prompt"]
    assert record["allowed_references"]
    assert record["attempts"] and "verdict" in record["attempts"][0]
    assert record["final"]["diagnosis"]
    assert record["providers"]["generation"] == "offline-kg-composer"
    assert record["outcome"] in ("ok", "fallback")


def test_the_audit_log_is_one_json_object_per_line(pipeline):
    text = pipeline.audit_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            json.loads(line)


# --- the chat -------------------------------------------------------------------


def _snap(state: str, **kw) -> dict:
    return {"equipment_id": "M-301", "family": "motor", "state": state, "hhi": 0.0,
            "regime": "NOMINAL", "vib_rms_mm_s": 0.8, "rul_hours": 720.0,
            "top_classes": [{"name": "bearing_wear", "score": 0.1}],
            "top_surprised_features": [{"kind": "HARMONIC_FEAT_VIB_BPFO", "sigma": 0.4,
                                        "direction": "up"}], **kw}


def test_a_healthy_machine_is_not_given_a_fault_by_the_chat(pipeline):
    from api.chat import _context, _state_event, ask

    turn = ask("What is wrong with the machine?", pipeline, _snap("OK"))
    assert turn.candidates == []                     # noise is not evidence
    assert "operating normally" in turn.answer
    # The motor's own manual, not the pump's.
    assert turn.retrieved and all(r.startswith("M-301") for r in turn.retrieved)
    kg = query.query_for_event(_state_event(_snap("WARNING")), pipeline.store)
    prompt = _context("What is wrong?", kg, [], _snap("WARNING"))
    assert "Bearing outer-race tone" in prompt and "HARMONIC_FEAT" not in prompt
    assert "No alarm" in _context("What is wrong?", kg, [], _snap("OK"))
