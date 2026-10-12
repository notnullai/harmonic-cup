"""End to end: simulator → engine → event → graph → diagnosis → API (HARM-S5-T10).

This is the test that would have caught the Phase 3 and Phase 4 defects if it
had existed then, because it is the only one that runs every layer against the
real artefacts: the compiled engine through ctypes, the commissioned brain, the
loaded graph, the chunk manifest, and the HTTP surface the dashboard consumes.

It drives the bridge by calling `step()` directly rather than starting the
background thread. The thread is tested separately; stepping deterministically
means a failure names a tick rather than a race.

Skipped, loudly, when the shared library or the brain is missing. Silently
passing a suite that tested nothing is how a project convinces itself it works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")

from fastapi.testclient import TestClient  # noqa: E402

from api import main as api_main  # noqa: E402
from api.bridge import DEFAULT_BRAIN, EngineBridge  # noqa: E402
from api.genai_hook import GenAIHook  # noqa: E402
from api.rig import RigUnavailable, find_library  # noqa: E402
from kg.store import open_store  # noqa: E402
from rag.guard import validate  # noqa: E402
from rag.pipeline import Pipeline  # noqa: E402
from rag.prompt import load_schema  # noqa: E402

# How far the fault is ramped before the assertions. One frame is a machine
# minute at the default time scale, so 90 covers an hour and a half of
# degradation — comfortably past the 5-11% severity the engine detects at.
RAMP_FRAMES = 90
SETTLE_FRAMES = 40


def _require_engine() -> None:
    if find_library() is None:
        pytest.skip("libharmonic_rig not built; run: cmake --build build -j")
    if not Path(DEFAULT_BRAIN).exists():
        pytest.skip(f"{DEFAULT_BRAIN} not built; run: python scripts/build_brain.py")


def _require_graph() -> None:
    # kg/graph.json is generated and gitignored, so a fresh clone or worktree
    # has none and open_store() falls back to an empty graph. Fail here, not
    # later as "no candidates": the graph is a prerequisite, not optional.
    store = open_store()
    try:
        present = store.node("P-101") is not None
    finally:
        store.close()
    if not present:
        pytest.fail("knowledge graph has no P-101; build it first: "
                    "python -m kg.load --rebuild --backend networkx")


@pytest.fixture(scope="module")
def stack():
    """Bridge + hook + API, wired the way the service wires them."""
    _require_engine()
    _require_graph()
    try:
        hook = GenAIHook(Pipeline())
    except Exception as exc:
        pytest.skip(f"diagnosis layer unavailable: {exc}")
    try:
        bridge = EngineBridge(DEFAULT_BRAIN, on_event=hook.submit,
                              event_cooldown_sec=0)
    except RigUnavailable as exc:
        hook.stop()
        pytest.skip(str(exc))

    api_main.attach(bridge, hook)
    client = TestClient(api_main.app)
    try:
        yield bridge, hook, client
    finally:
        client.close()
        bridge.stop()
        hook.stop()
        api_main.attach(None, None)


@pytest.fixture(scope="module")
def after_fault(stack):
    """Settle healthy, inject bearing wear, ramp it in."""
    bridge, hook, _client = stack
    for _ in range(SETTLE_FRAMES):
        bridge.step()
    healthy = bridge.latest
    bridge.inject("bearing_wear", severity=1.0, ramp_minutes=30.0)
    for _ in range(RAMP_FRAMES):
        bridge.step()
    hook.wait_idle(timeout=60)
    return healthy, bridge.latest


# --- the engine actually runs -------------------------------------------------


def test_the_engine_produces_ticks(stack):
    bridge, _hook, _client = stack
    first = bridge.step()
    second = bridge.step()
    assert second.tick == first.tick + 1
    assert second.state in ("OK", "WATCH", "WARNING", "CRITICAL", "FAILED")
    assert second.sensors["rpm"] > 2000


def test_the_engine_is_the_compiled_one(stack):
    """Not a Python re-implementation: the numbers come through ctypes from
    the same object the benchmark and the determinism fixtures pin."""
    bridge, _hook, _client = stack
    info = bridge.info()
    assert info["commissioned"], "the demo brain must be commissioned"
    assert info["feature_count"] == 16
    assert info["class_count"] == 5


def test_a_healthy_machine_reads_healthy(stack):
    bridge, _hook, _client = stack
    bridge.clear_faults()
    for _ in range(30):
        snap = bridge.step()
    assert snap.state in ("OK", "WATCH"), snap.to_json()
    assert snap.vib_rms_mm_s < 1.8, "a healthy machine must sit inside ISO zone A"


# --- a fault raises an event --------------------------------------------------


def test_the_health_index_rises_with_the_fault(after_fault):
    healthy, faulted = after_fault
    assert faulted.hhi > healthy.hhi, (
        f"HHI did not rise: {healthy.hhi} -> {faulted.hhi}")


def test_the_fault_raises_an_event(stack, after_fault):
    bridge, _hook, _client = stack
    events = bridge.events(50)
    assert events, "a ramped bearing fault raised no event"
    assert any(e.state in ("WARNING", "CRITICAL") for e in events), \
        [e.state for e in events]


def test_the_event_carries_its_evidence(stack, after_fault):
    """An event without the features behind it cannot be diagnosed, and cannot
    be argued with either."""
    bridge, _hook, _client = stack
    event = bridge.events(50)[-1]
    assert event.top_surprised_features, event.to_json()
    assert all(f["kind"].startswith("HARMONIC_FEAT_")
               for f in event.top_surprised_features)
    assert event.top_classes


# --- the diagnosis ------------------------------------------------------------


def test_the_event_gets_a_diagnosis(stack, after_fault):
    bridge, hook, _client = stack
    hook.wait_idle(timeout=60)
    diagnosed = [e for e in bridge.events(50) if e.diagnosis is not None]
    assert diagnosed, [e.diagnosis_status for e in bridge.events(50)]


def test_the_diagnosis_is_schema_valid_and_correctly_cited(stack, after_fault):
    """The whole point of the layer: an answer an engineer can check."""
    bridge, hook, _client = stack
    hook.wait_idle(timeout=60)
    schema = load_schema()
    chunks = {c.id for c in hook.pipeline.chunks.values()} | set(hook.pipeline.chunks)
    for event in bridge.events(50):
        if event.diagnosis is None:
            continue
        verdict = validate(event.diagnosis, chunks, schema)
        assert verdict.ok, (event.id, verdict.reasons)
        assert event.diagnosis["references"], event.id


def test_the_diagnosis_names_a_bearing_fault(stack, after_fault):
    bridge, hook, _client = stack
    hook.wait_idle(timeout=60)
    diagnosed = [e for e in bridge.events(50) if e.diagnosis is not None]
    modes = {e.diagnosis.get("failure_mode_id") for e in diagnosed}
    assert modes & {"fm.outer_race_wear", "fm.lubrication_failure",
                    "fm.inner_race_wear", "fm.rolling_element_wear"}, modes


# --- the HTTP surface ---------------------------------------------------------


def test_health_endpoint(stack, after_fault):
    _bridge, _hook, client = stack
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["engine"]["available"] is True
    assert body["machine"]["state"] in ("OK", "WATCH", "WARNING", "CRITICAL")
    assert 0.0 <= body["machine"]["hhi"] <= 1.0


def test_sensors_endpoint(stack, after_fault):
    _bridge, _hook, client = stack
    body = client.get("/sensors", params={"history": 20}).json()
    assert body["sensors"]["rpm"] > 2000
    assert body["features"], "the engine's feature view must be exposed"
    assert len(body["history"]) > 1


def test_events_endpoint_carries_the_diagnosis(stack, after_fault):
    _bridge, hook, client = stack
    hook.wait_idle(timeout=60)
    body = client.get("/events", params={"limit": 50}).json()
    assert body["events"]
    with_diagnosis = [e for e in body["events"] if e.get("diagnosis")]
    assert with_diagnosis, [e["diagnosis_status"] for e in body["events"]]
    one = with_diagnosis[-1]
    assert one["diagnosis"]["actions"]
    assert one["diagnosis"]["safety"]
    assert one["diagnosis_provider"]


def test_kg_endpoint_returns_the_path_behind_an_event(stack, after_fault):
    _bridge, _hook, client = stack
    events = client.get("/events", params={"limit": 5}).json()["events"]
    body = client.get(f"/kg/{events[-1]['id']}").json()
    assert body["candidates"], body
    assert body["citable_ids"]


def test_chat_answers_with_citations(stack, after_fault):
    _bridge, _hook, client = stack
    body = client.post("/chat", json={"question": "what's wrong with the pump?"}).json()
    assert body["answer"], body
    assert body["references"], body
    assert body["grounded"] is True


def test_chat_rejects_an_empty_question(stack):
    _bridge, _hook, client = stack
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_unknown_event_is_404(stack):
    _bridge, _hook, client = stack
    assert client.get("/events/nope").status_code == 404


def test_inject_rejects_an_unknown_mode(stack):
    _bridge, _hook, client = stack
    r = client.post("/inject", json={"mode": "gremlins"})
    assert r.status_code == 422


# --- the websocket ------------------------------------------------------------


def test_websocket_streams_snapshots(stack, after_fault):
    """The dashboard polls REST, but the socket is the documented interface and
    an untested interface is a broken one."""
    bridge, _hook, client = stack
    with client.websocket_connect("/ws/stream") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["engine"]["available"] is True
        bridge.step()
        payload = ws.receive_json()
        assert payload["type"] == "tick"
        assert payload["snapshot"]["tick"] > 0
        assert "hhi" in payload["snapshot"]


# --- the service without an engine --------------------------------------------


def test_the_api_says_so_when_the_engine_is_absent(monkeypatch):
    """Zeros would render as a machine in perfect condition."""
    # The lifespan autostarts the engine, which is right for the service and
    # wrong for this test: it would quietly re-attach a working bridge and the
    # assertion would pass for the opposite reason.
    monkeypatch.setattr(api_main, "AUTOSTART", False)
    api_main.attach(None, None)
    try:
        with TestClient(api_main.app) as client:
            body = client.get("/health").json()
            assert body["engine"]["available"] is False
            assert body["machine"] is None
            assert client.get("/sensors").status_code == 503
            assert client.get("/events").status_code == 503
    finally:
        api_main.attach(None, None)


def test_the_audit_log_records_the_run(stack, after_fault):
    _bridge, hook, _client = stack
    hook.wait_idle(timeout=60)
    path = hook.pipeline.audit_path
    if not path.exists():
        pytest.skip("no audit log was written")
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()
            if line.strip()]
    assert rows
    assert rows[-1]["providers"]["generation"]
    assert rows[-1]["allowed_references"]
