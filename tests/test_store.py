"""The event store and technician feedback (HARM-S7-T12, T13), and the fleet
that feeds it."""

from __future__ import annotations

import subprocess
import time

import pytest

from api.store import EventStore

CLASSES = ("healthy", "bearing_wear", "misalignment", "cavitation", "seal_leak")


def _event(i: int, top: str = "bearing_wear") -> dict:
    return {"id": f"ev-{i}", "equipment_id": "P-101", "state": "WARNING",
            "machine_hours": float(i), "recorded_at": None,
            "top_classes": [{"name": top, "score": 0.9}], "diagnosis": None}


def _frame(seed: int):
    from api.bridge import DEFAULT_BRAIN
    from api.sources import SimulatedSource

    return SimulatedSource(brain_path=DEFAULT_BRAIN, seed=seed).next()


def test_events_and_verdicts_survive_a_restart(tmp_path):
    db = tmp_path / "events.db"
    s = EventStore(db)
    s.save(_event(1), "simulated", "pump")
    s.update({**_event(1), "diagnosis": {"diagnosis": "outer race wear"}})
    fb = s.set_feedback("ev-1", "wrong", "misalignment", "  coupling was cracked ", CLASSES)
    assert fb["actual"] == "misalignment" and fb["note"] == "coupling was cracked"
    s.close()

    s = EventStore(db)                     # a restart
    [e] = s.recent("simulated")
    assert e["diagnosis"]["diagnosis"] == "outer race wear"
    assert e["feedback"]["verdict"] == "wrong"
    assert s.counts() == {"simulated": {"events": 1, "feedback": 1}}


def test_a_verdict_is_checked_against_the_family_s_classes(tmp_path):
    s = EventStore(tmp_path / "e.db")
    s.save(_event(1), "simulated", "pump")
    with pytest.raises(ValueError, match="actual must be one of"):
        s.set_feedback("ev-1", "wrong", "broken_tooth", "", CLASSES)   # a gearbox fault
    with pytest.raises(ValueError, match="verdict"):
        s.set_feedback("ev-1", "maybe", None, "", CLASSES)
    with pytest.raises(KeyError):
        s.set_feedback("ev-404", "confirmed", None, "", CLASSES)
    # "confirmed" needs no fault named; "healthy" records a false alarm.
    assert s.set_feedback("ev-1", "confirmed", "cavitation", "", CLASSES)["actual"] is None
    assert s.set_feedback("ev-1", "wrong", "healthy", "", CLASSES)["actual"] == "healthy"


def test_judged_events_export_as_training_frames_the_encoder_reads(tmp_path):
    """The acceptance case: feedback becomes labelled words for the trainer."""
    from api.bridge import DEFAULT_BRAIN
    from scripts.build_brain import binary

    if not DEFAULT_BRAIN.exists():
        pytest.skip("pump brain not built")
    s = EventStore(tmp_path / "e.db")
    for i in range(4):
        s.save(_event(i), "simulated", "pump", _frame(i), 25600)
    s.set_feedback("ev-0", "confirmed", None, "", CLASSES)          # -> bearing_wear (1)
    s.set_feedback("ev-1", "wrong", "cavitation", "", CLASSES)      # -> cavitation (3)
    s.set_feedback("ev-2", "wrong", "other", "", CLASSES)           # no class: left out
    data, n = s.export_frames("simulated", CLASSES)                 # ev-3: not judged
    assert n == 2
    frames = tmp_path / "fb.hframe"
    frames.write_bytes(data)
    words = tmp_path / "fb.csv"
    r = subprocess.run([str(binary("harmonic_sim_words")), str(frames), "--brain",
                        str(DEFAULT_BRAIN), "--out", str(words)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [ln for ln in words.read_text().splitlines() if not ln.startswith("#")]
    assert [ln.split(",")[0] for ln in rows] == ["1", "3"]


def test_the_fleet_watches_every_machine_and_stores_what_they_raise(tmp_path, monkeypatch):
    from api import fleet as fleet_mod
    from api.rig import find_library
    from kg import families

    if find_library() is None or not families.path("gearbox", "brain").exists():
        pytest.skip("engine or gearbox brain not built")
    monkeypatch.setattr(fleet_mod, "FLEET_MODE", "current")
    store = EventStore(tmp_path / "e.db")
    f = fleet_mod.Fleet(store, None, period_s=0.01)
    try:
        f.start("simulated")
        f.select("simulated_gearbox")
        assert f.current_key == "simulated_gearbox"
        f.current.inject("broken_tooth", 1.0, 0.0)
        pump = f.bridges["simulated"]
        deadline = time.time() + 30
        while time.time() < deadline and not store.recent("simulated_gearbox"):
            time.sleep(0.2)
        [row] = [m for m in f.summary() if m["key"] == "simulated_gearbox"]
        assert row["running"] and row["family"] == "gearbox" and row["current"]
        assert store.recent("simulated_gearbox"), "the gearbox raised nothing"
        # The pump kept running while the gearbox was on screen.
        assert pump.latest is not None and pump.latest.tick > 0
        # Most urgent first.
        assert f.summary()[0]["key"] == "simulated_gearbox"
        _, n = store.export_frames("simulated_gearbox", families.class_names("gearbox"))
        assert n == 0                       # nothing judged yet
    finally:
        f.stop()
        store.close()
