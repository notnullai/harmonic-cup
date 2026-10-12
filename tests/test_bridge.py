"""The ctypes bridge and the engine bridge (HARM-S5-T02).

The struct-layout tests are the important ones. A ctypes binding that
disagrees with the C struct does not raise — it reads the wrong offsets and
returns plausible nonsense, which is exactly what the packed frame record cost
this project once already.
"""

from __future__ import annotations

import ctypes
import re
from pathlib import Path

import pytest

from api.rig import FrameRecord, RigOut, find_library

REPO = Path(__file__).resolve().parents[1]
TYPES_H = REPO / "engine" / "include" / "harmonic" / "harmonic_types.h"
RIG_CPP = REPO / "engine" / "bridge" / "harmonic_rig.cpp"


def _library_or_skip() -> Path:
    lib = find_library()
    if lib is None:
        pytest.skip("libharmonic_rig not built; run: cmake --build build -j")
    return lib


def _brain_or_skip() -> Path:
    from api.bridge import DEFAULT_BRAIN
    if not Path(DEFAULT_BRAIN).exists():
        pytest.skip(f"{DEFAULT_BRAIN} not built; run: python scripts/build_brain.py")
    return Path(DEFAULT_BRAIN)


# --- the name table cannot drift from the enum --------------------------------


def _enum_names() -> list[str]:
    source = TYPES_H.read_text(encoding="utf-8")
    block = re.search(r"HARMONIC_FEAT_VIB_RMS\s*=\s*0,(.*?)HARMONIC_FEAT__COUNT",
                      source, re.DOTALL)
    assert block, "the feature-kind enum moved; this test needs updating"
    names = ["HARMONIC_FEAT_VIB_RMS"]
    names += re.findall(r"(HARMONIC_FEAT_[A-Z0-9_]+)\s*,", block.group(1))
    return [n for n in names if not n.endswith("__COUNT")]


def test_the_bridge_names_every_feature_kind():
    """`harmonic_rig_feature_kind_name` is a hand-written table beside an enum.

    A C static_assert catches a MISSING entry; only this catches a WRONG one,
    and a wrong name silently breaks the join to the knowledge graph, which
    matches symptoms on exactly these strings.
    """
    table = re.search(r"FEATURE_KIND_NAMES\[\]\s*=\s*\{(.*?)\};",
                      RIG_CPP.read_text(encoding="utf-8"), re.DOTALL)
    assert table, "FEATURE_KIND_NAMES not found"
    in_table = re.findall(r'"(HARMONIC_FEAT_[A-Z0-9_]+)"', table.group(1))
    assert in_table == _enum_names(), (
        "the name table and the enum disagree:\n"
        f"  table: {in_table}\n  enum:  {_enum_names()}"
    )


# --- struct layout ------------------------------------------------------------


def test_the_frame_record_is_packed_at_52_bytes():
    """The same 52 bytes the format documents and the simulator writes."""
    assert ctypes.sizeof(FrameRecord) == 52


def test_the_library_agrees_about_both_struct_sizes():
    lib = ctypes.CDLL(str(_library_or_skip()))
    lib.harmonic_rig_out_size.restype = ctypes.c_uint32
    lib.harmonic_rig_frame_size.restype = ctypes.c_uint32
    assert int(lib.harmonic_rig_out_size()) == ctypes.sizeof(RigOut)
    assert int(lib.harmonic_rig_frame_size()) == ctypes.sizeof(FrameRecord)


def test_a_disagreeing_binding_refuses_to_load(monkeypatch):
    """The check has to actually fail when the layouts differ."""
    from api import rig as rig_mod

    _library_or_skip()
    brain = _brain_or_skip()

    class Wider(ctypes.Structure):
        _fields_ = [*RigOut._fields_, ("extra", ctypes.c_int64)]

    monkeypatch.setattr(rig_mod, "RigOut", Wider)
    with pytest.raises(rig_mod.RigUnavailable, match="would read the wrong fields"):
        rig_mod.HarmonicRig(brain)


# --- the rig ------------------------------------------------------------------


@pytest.fixture(scope="module")
def rig():
    from api.rig import HarmonicRig
    _library_or_skip()
    r = HarmonicRig(_brain_or_skip())
    yield r
    r.close()


def test_the_rig_reports_the_brain_it_loaded(rig):
    assert rig.feature_count == 16
    assert rig.class_count == 5
    assert rig.commissioned
    assert abs(rig.nominal_rpm - 2950.0) < 1.0
    assert rig.feature_kinds[0] == "HARMONIC_FEAT_VIB_RMS"
    assert rig.feature_units[0] == "mm/s"


def test_a_missing_brain_is_an_explicit_failure():
    from api.rig import HarmonicRig, RigUnavailable
    _library_or_skip()
    with pytest.raises(RigUnavailable, match="brain not found"):
        HarmonicRig("/nonexistent/brain.harmonic")


def test_feeding_a_frame_returns_real_units(rig):
    from sim.pump_sim import PumpSimulator

    sim = PumpSimulator(seed=11, fs_vib=25600.0)
    dt = rig.window_samples / 25600.0
    for i in range(6):
        frame = sim.step(dt)
        tick = rig.feed(frame.vibration.astype("float32"), rpm=frame.rpm,
                        load=0.65,
                        slow=[frame.bearing_temp_c, frame.motor_temp_c,
                              frame.discharge_bar, frame.suction_bar,
                              frame.flow_m3h, frame.current_a, 0.0],
                        timestamp_us=i * 1_000_000)
    assert tick.tick == 6
    assert tick.state in ("OK", "WATCH", "WARNING", "CRITICAL", "FAILED")
    # A healthy pump sits inside ISO zone A. If the Q16 conversion were wrong
    # this would be off by a factor of 65536 in one direction or the other.
    assert 0.05 < tick.vib_rms_mm_s < 1.8, tick.vib_rms_mm_s
    assert set(tick.features) == set(rig.feature_kinds)
    assert len(tick.class_scores) == rig.class_count


def test_a_fixed_speed_pump_at_65_percent_duty_is_nominal(rig):
    """The regime defect found while wiring this bridge, pinned from Python.

    The classifier banded on the speed fraction first, which for a machine at
    its rated speed is always ~1.0 — so a fixed-speed pump was always
    HIGH_LOAD, NOMINAL and LOW_LOAD were unreachable, and every per-regime
    predictor shared one bucket.
    """
    from sim.pump_sim import PumpSimulator

    sim = PumpSimulator(seed=3, fs_vib=25600.0)
    dt = rig.window_samples / 25600.0
    for _ in range(4):
        frame = sim.step(dt)
        tick = rig.feed(frame.vibration.astype("float32"), rpm=frame.rpm, load=0.65)
    assert tick.regime == "NOMINAL", tick.regime


def test_top_surprised_features_are_ranked_and_directional(rig):
    from api.rig import Tick

    tick = Tick(tick=1, state="OK", iso_state="OK", regime="NOMINAL", warm=True,
                hhi=0.1, hhi_fast=0.1, hhi_slow=0.1, slope_per_tick=0.0,
                rul_hours=100.0, vib_rms_mm_s=0.5, event=False,
                features={"a": 1.0, "b": 2.0, "c": 3.0},
                surprise={"a": 0.1, "b": -2.0, "c": 1.0})
    top = tick.top_surprised(n=3)
    assert [t["kind"] for t in top] == ["b", "c"]      # "a" is under the floor
    assert top[0]["direction"] == "down"
    assert top[1]["direction"] == "up"


# --- the engine bridge --------------------------------------------------------


@pytest.fixture
def bridge():
    from api.bridge import EngineBridge
    _library_or_skip()
    b = EngineBridge(_brain_or_skip(), event_cooldown_sec=0)
    yield b
    b.stop()


def test_the_bridge_steps_without_a_thread(bridge):
    first = bridge.step()
    second = bridge.step()
    assert second.tick == first.tick + 1
    assert second.sensors["flow_m3h"] > 0
    assert second.class_scores, "class scores must be named, not indexed"
    assert "healthy" in second.class_scores


def test_healthy_ranking_excludes_the_healthy_class(bridge):
    """Ranking "healthy" first every time the machine is fine is true and
    useless, and it would become the top class of every diagnosis."""
    snap = bridge.step()
    assert all(c["name"] != "healthy" for c in snap.top_classes)


def test_injection_shows_up_in_the_snapshot(bridge):
    bridge.step()
    bridge.inject("cavitation", severity=1.0, ramp_minutes=1.0)
    for _ in range(4):
        snap = bridge.step()
    assert "cavitation" in snap.injected


def test_an_unknown_fault_mode_is_rejected(bridge):
    with pytest.raises(ValueError, match="unknown fault mode"):
        bridge.inject("gremlins")


def test_events_carry_what_the_graph_needs(bridge):
    # Long enough to pass the per-regime warm-up. Before it, surprise has no
    # baseline to be measured against and no surprise-driven event is raised.
    for _ in range(20):
        bridge.step()
    bridge.inject("bearing_wear", severity=1.0, ramp_minutes=2.0)
    for _ in range(80):
        bridge.step()
    events = bridge.events(10)
    assert events, "a full-severity bearing fault raised no event"
    payload = events[-1].as_engine_event()
    assert payload["event_id"] and payload["top_surprised_features"]
    assert payload["state"] in ("WARNING", "CRITICAL")


def test_an_absolute_iso_breach_is_reported_during_warm_up(bridge):
    """A machine already in zone D when monitoring starts must not read OK.

    Warm-up is what the PREDICTORS need — before it, surprise has no baseline
    and means nothing. A published absolute limit needs no baseline, and
    gating it on warm-up made a pump sitting at 15 mm/s report OK for the
    whole warm-up window. Found on the dashboard, in exactly the situation the
    dashboard exists to prevent.
    """
    bridge.inject("bearing_wear", severity=1.0, ramp_minutes=0.5)
    states, warm_flags = [], []
    for _ in range(25):
        snap = bridge.step()
        states.append(snap.state)
        warm_flags.append(snap.warm)
    assert not any(warm_flags), "this test is about the pre-warm-up window"
    assert snap.vib_rms_mm_s > 11.2, (
        f"the fault did not reach ISO zone D: {snap.vib_rms_mm_s} mm/s")
    assert any(s in ("WARNING", "CRITICAL") for s in states), states


# --- a recording of a real machine ---------------------------------------------


def _ims_bridge_or_skip():
    from api import sources
    from api.bridge import EngineBridge

    status = sources.available()["ims_test2_b1"]
    if not status["available"]:
        pytest.skip(status["reason"])
    return EngineBridge(source="ims_test2_b1")


def test_a_recording_cannot_be_injected_into():
    b = _ims_bridge_or_skip()
    try:
        with pytest.raises(ValueError, match="recording"):
            b.inject("cavitation")
        with pytest.raises(ValueError, match="recording"):
            b.clear_faults()
    finally:
        b.stop()


def test_a_recording_reports_what_it_did_not_measure():
    b = _ims_bridge_or_skip()
    try:
        snap = b.step()
        assert snap is not None
        assert snap.vib_unit == "g"
        # Velocity is integrated from the accelerometer, not copied from it:
        # a healthy bearing on this rig is well under a millimetre per second,
        # and no ISO zones are set for it.
        assert snap.vib_rms_mm_s is not None and 0 < snap.vib_rms_mm_s < 1.0
        assert snap.vib_rms_mm_s != snap.vib_rms
        assert b.info()["source"]["standard"] is None
        assert snap.sensors["bearing_temp_c"] is None
        assert snap.recorded_at == "2004-02-12 10:32"
        assert b.info()["source"]["real"] is True
    finally:
        b.stop()


def test_a_recording_ends_rather_than_loops():
    b = _ims_bridge_or_skip()
    try:
        n = 0
        while b.step() is not None:
            n += 1
        assert n == 984
        assert b.step() is None
        assert b.info()["source"]["progress"]["finished"] is True
    finally:
        b.stop()


def test_the_exported_ims_bundle_replays_as_the_raw_data_does(tmp_path, monkeypatch):
    """On Google Cloud the raw NASA files are absent; the exported bundle
    (scripts/export_ims.py) must give the engine exactly the same run."""
    import shutil

    from api import sources
    from sim.datasets import ims_loader, recordings

    bundle = REPO / "var" / "ims" / "test2"
    if not (bundle / "stamps.json").exists() or not ims_loader.test_dir(2).is_dir():
        pytest.skip("needs both the raw IMS data and the export (scripts/export_ims.py)")
    raw = recordings.ImsRecording()
    data_dir = tmp_path / "data"
    shutil.copytree(bundle, data_dir / "ims" / "test2")
    monkeypatch.setenv("HARMONIC_DATA_DIR", str(data_dir))
    monkeypatch.setattr(ims_loader, "DEFAULT_ROOT", tmp_path / "no-raw-data")
    monkeypatch.setattr(sources, "IMS_BRAIN", tmp_path / "not-built.harmonic")
    exported = recordings.ImsRecording()
    # The export may hold more bearings than channel 0 (--channels 0 2).
    assert exported._bundle is not None and 0 in exported._channels
    assert list(exported.hours()) == list(raw.hours())
    assert exported.stamps() == raw.stamps()
    assert (exported.load(0, 16384) == raw.load(0, 16384)).all()
    assert sources.available()[sources.IMS_KEY]["available"]
    assert sources.ims_brain().parent == data_dir / "ims" / "test2"
