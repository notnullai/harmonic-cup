"""The timeline history endpoint and the PDF report, against the real engine.

The web app scrubs back through `/history` and exports `/report.pdf` for the
period on screen; both are built from the bridge's own history and events,
so these drive a real bridge through a real fault.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")
pytest.importorskip("reportlab", reason="reportlab is needed for the PDF report")

from fastapi.testclient import TestClient

from api import main as api_main
from api.bridge import DEFAULT_BRAIN, EngineBridge
from api.genai_hook import GenAIHook
from api.rig import RigUnavailable, find_library


@pytest.fixture(scope="module")
def client():
    if find_library() is None:
        pytest.skip("libharmonic_rig not built; run: cmake --build build -j")
    if not Path(DEFAULT_BRAIN).exists():
        pytest.skip(f"{DEFAULT_BRAIN} not built; run: python scripts/build_brain.py")
    hook = GenAIHook()
    try:
        bridge = EngineBridge(DEFAULT_BRAIN, on_event=hook.submit)
    except RigUnavailable as exc:
        hook.stop()
        pytest.skip(str(exc))
    bridge.warm_up()
    bridge.inject("bearing_wear", 1.0, 0.0)
    for _ in range(20):
        bridge.step()
    hook.wait_idle(30)
    api_main.attach(bridge, hook)
    c = TestClient(api_main.app)
    try:
        yield c, bridge
    finally:
        c.close()
        bridge.stop()
        hook.stop()
        api_main.attach(None, None)


def test_history_is_incremental(client):
    c, _ = client
    full = c.get("/history").json()
    ticks = [p["tick"] for p in full["points"]]
    assert full["source"] == "simulated"
    assert ticks == sorted(ticks) and len(ticks) >= 80
    # Only newer readings come back: the web app polls with the last tick seen.
    since = ticks[-5]
    newer = c.get(f"/history?since_tick={since}").json()["points"]
    assert [p["tick"] for p in newer] == ticks[-4:]
    p = full["points"][-1]
    assert {"h", "state", "hhi", "vib", "sensors"} <= set(p)


def test_events_carry_their_time(client):
    c, _ = client
    events = c.get("/events").json()["events"]
    assert events, "the injected fault raised nothing"
    assert all(e["machine_hours"] > 0 for e in events)


def test_report_is_a_pdf_and_respects_the_period(client):
    c, bridge = client
    whole = c.get("/report.pdf")
    assert whole.status_code == 200
    assert whole.headers["content-type"] == "application/pdf"
    assert whole.content.startswith(b"%PDF")
    assert "attachment" in whole.headers["content-disposition"]

    # A period after the last reading holds no readings and no events, and
    # still comes back as a valid, and much shorter, report.
    last_h = bridge.latest.machine_hours
    empty = c.get(f"/report.pdf?from_h={last_h + 10}&to_h={last_h + 20}")
    assert empty.status_code == 200 and empty.content.startswith(b"%PDF")
    assert len(empty.content) < len(whole.content)
