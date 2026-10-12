"""Commissioning a machine from a recording: the wizard's backend.

The end-to-end case uploads a synthetic recording — healthy simulator
snapshots saved as a NumPy file — and commissions it through the same
pipeline the console runs, so it needs no downloaded data and runs in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from api import machines
from sim.datasets import recordings

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(machines, "MACHINES_DIR", tmp_path / "machines")
    monkeypatch.setattr(machines, "UPLOADS_DIR", tmp_path / "uploads")
    return tmp_path


def _needs_engine() -> None:
    from api.rig import find_library

    if find_library() is None:
        pytest.skip("libharmonic_rig not built; run: cmake --build build -j")
    if not machines.TRAINED.exists():
        pytest.skip("trained network not built; run: python scripts/build_brain.py")
    try:
        machines._binary("mint_brain")
        machines._binary("commission")
    except machines.CommissionError as exc:
        pytest.skip(str(exc))


# --- recordings ------------------------------------------------------------------


def test_timestamps_are_read_from_file_names():
    assert recordings.parse_stamp("2004.02.12.10.32.39").hour == 10
    assert recordings.parse_stamp("2024-05-01T10-00-00.csv").day == 1
    assert recordings.parse_stamp("20240501_101500.txt").minute == 15
    assert recordings.parse_stamp("snapshot_7.csv") is None


def test_window_is_the_largest_power_of_two_the_snapshot_holds():
    assert recordings.window_for(20480) == 16384
    assert recordings.window_for(8192) == 8192
    assert recordings.window_for(70000) == 16384
    assert recordings.window_for(2560) == 2048          # FEMTO's 0.1 s bursts
    with pytest.raises(ValueError, match="too short"):
        recordings.window_for(1000)



def _tick(rms: float, surprise: float, warm: bool = True):
    from types import SimpleNamespace
    return SimpleNamespace(warm=warm, features={machines.RMS_KIND: rms},
                           surprise={"HARMONIC_FEAT_VIB_RMS": surprise})


def test_a_reading_with_the_machine_stopped_does_not_set_the_noise_floor():
    # 300 running readings with surprise up to 0.3, one stop 3 doublings below
    # its neighbours with a surprise of 3.7 (FEMTO bearing 2_1's case).
    ticks = [_tick(0.0, 0.3 * ((i % 7) / 6)) for i in range(300)]
    ticks[150] = _tick(-3.0, -3.7)
    info: dict = {}
    floor = machines.measure_floor(ticks, info)
    assert info["not_running"] == 1
    assert floor["VIB_RMS"] == pytest.approx(0.6, abs=0.01)


def test_a_baseline_that_rises_is_running_and_sets_the_floor_in_full():
    # 4x quieter early than late, rising gradually (XJTU-SY 3_5's horizontal
    # sensor): nothing is left out, unlike a sudden drop.
    ticks = [_tick(-2.5 + 2.5 * i / 299, 0.1) for i in range(300)]
    info: dict = {}
    machines.measure_floor(ticks, info)
    assert info["not_running"] == 0

def test_a_folder_of_csv_snapshots_is_ordered_by_its_timestamps(tmp_path):
    rng = np.random.default_rng(1)
    names = ["2024-05-01T12-00-00.csv", "2024-05-01T10-00-00.csv", "2024-05-01T11-00-00.csv"]
    for i, n in enumerate(names):
        cols = rng.normal(size=(4096, 2)) + i
        (tmp_path / n).write_text("ax,ay\n" + "\n".join(f"{a},{b}" for a, b in cols))
    (tmp_path / "recording.json").write_text(json.dumps({"sample_rate_hz": 10000, "unit": "g"}))
    r = recordings.FolderRecording(tmp_path)
    assert r.n_channels == 2 and r.snapshot_samples() == 4096
    assert list(r.hours()) == [0.0, 1.0, 2.0]
    data = r.load(1, 4096)
    assert data.shape == (3, 4096)
    # Sorted by time, not by name: 10:00 was written second, with offset 1.
    assert abs(float(data[0].mean()) - 1.0) < 0.1


def test_a_numpy_recording_without_timestamps_uses_the_interval(tmp_path):
    np.save(tmp_path / "run.npy", np.zeros((5, 8192), dtype=np.float32))
    (tmp_path / "recording.json").write_text(json.dumps({"sample_rate_hz": 25600,
                                                         "interval_s": 1800}))
    r = recordings.FolderRecording(tmp_path)
    assert list(r.hours()) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert r.stamps() == [None] * 5


# --- validation --------------------------------------------------------------------


def test_upload_file_names_cannot_escape_the_folder(data_dir):
    uid = machines.new_upload({"sample_rate_hz": 20000})
    for bad in ("../x.csv", "recording.json", "_cache_ch0_4096.npy", "x.exe"):
        with pytest.raises(machines.CommissionError):
            machines.save_upload_file(uid, bad, b"1\n")
    with pytest.raises(machines.CommissionError, match="unknown upload"):
        machines.save_upload_file("../../etc", "x.csv", b"1\n")


def _upload_healthy(snapshots: int) -> str:
    from sim.pump_sim import PumpSimulator

    sim = PumpSimulator(seed=11, fs_vib=25600.0)
    frames = np.stack([sim.step(16384 / 25600).vibration.astype(np.float32)
                       for _ in range(snapshots)])
    uid = machines.new_upload({"name": "synthetic pump", "sample_rate_hz": 25600,
                               "interval_s": 600, "unit": "mm/s"})
    import io
    buf = io.BytesIO()
    np.save(buf, frames)
    machines.save_upload_file(uid, "run.npy", buf.getvalue())
    return f"upload:{uid}"


def _spec(recording: str, **over):
    spec = {"name": "Test pump", "nominal_rpm": 2950, "vib_unit": "mm/s",
            "iso_class": "10816-1-III", "bearing": machines.BEARINGS["6308"],
            "components": ["bearing", "shaft", "impeller", "seal"],
            "recording": recording, "channel": 0, "baseline_h": (0.0, 1000.0)}
    spec.update(over)
    return spec


def test_an_acceleration_channel_takes_iso_zones_and_unknown_classes_are_refused(data_dir):
    # The engine integrates acceleration to velocity, so zones apply to both.
    key = _upload_healthy(3)
    m = machines.create(_spec(key, vib_unit="g", iso_class="10816-3-G2-rigid"))
    assert m.iso_class == "10816-3-G2-rigid"
    with pytest.raises(machines.CommissionError, match="unknown ISO class"):
        machines.create(_spec(key, iso_class="10816-7-I"))


def test_too_short_a_baseline_is_refused_with_the_reason(data_dir):
    _needs_engine()
    key = _upload_healthy(40)
    m = machines.create(_spec(key))
    with pytest.raises(machines.CommissionError, match="at least"):
        machines.commission(m.id)
    saved = machines.get(m.id)
    assert saved is not None and saved.status == "failed" and "Widen" in (saved.error or "")


# --- the whole pipeline --------------------------------------------------------------


def test_a_healthy_upload_commissions_with_no_false_alarm(data_dir):
    _needs_engine()
    key = _upload_healthy(200)
    m = machines.create(_spec(key))
    registered = []
    done = machines.commission(m.id, register=registered.append)

    assert done.status == "ready", done.error
    assert done.brain_path.exists()
    assert done.report["readings"] == 200
    assert done.report["false_alarms"] == 0
    assert set(done.report["dead_zones"]) >= {"VIB_RMS", "VIB_BPFO", "VIB_KURTOSIS"}
    assert registered and registered[0].id == m.id
    # The profile it wrote is a real profile: mint_brain read it.
    profile = (done.folder / f"{m.id}.profile").read_text()
    assert "dead_zone.VIB_RMS" in profile and "iso_zone_bounds = 1.8 4.5 11.2 28" in profile

    # It can be watched: the bridge replays it through its own brain.
    from api import sources
    from api.bridge import EngineBridge

    listed = sources.available()
    assert listed[m.id]["available"] and listed[m.id]["kind"] == "commissioned"
    b = EngineBridge(source=m.id)
    try:
        b.warm_up()
        snaps = [b.step() for _ in range(20)]
        assert all(s is not None for s in snaps)
        assert b.info()["source"]["badge"] == "Recorded data"
        assert b.info()["source"]["equipment_id"] == done.equipment_id
    finally:
        b.stop()

    # And the graph learns what it has.
    from kg.store import NetworkXStore

    store = NetworkXStore(data_dir / "graph.json", autoload=False)
    store.merge_graph(machines.kg_graph(done))
    kinds = {e.other.props["kind"] for e in store.out_edges(done.equipment_id, "HAS_COMPONENT")}
    assert kinds == {"bearing", "shaft", "impeller", "seal"}

    assert machines.delete(m.id)
    assert machines.get(m.id) is None


def test_commissioning_writes_the_spread_floor_and_the_baseline_never_raises_it(data_dir):
    """The commission tool measures the spread floor's healthy statistics and
    writes them after the class peaks (docs/BRAIN_FORMAT.md); replaying the
    healthy baseline through the result never raises it."""
    _needs_engine()
    import struct

    key = _upload_healthy(200)
    done = machines.commission(machines.create(_spec(key)).id)
    assert done.status == "ready", done.error
    raw = done.brain_path.read_bytes()
    flags = struct.unpack_from("<I", raw, 12)[0]
    assert flags & 0b001 and flags & 0b100          # COMMISSIONED and SPREAD_FLOOR
    n_features = struct.unpack_from("<I", raw, 28)[0]
    block = raw[-312:]
    n, margin, drop, persist, window, rising = struct.unpack_from("<IiIIII", block, 0)
    assert (n, margin, drop, persist, window, rising) == (n_features, 2 * 65536, 1, 5, 5, 3)
    a_max, z_max, u_max = struct.unpack_from("<qqq", block, 32)
    assert a_max > 0 and z_max > 0 and u_max > 0
    sd = struct.unpack_from("<32i", block, 56 + 128)
    assert any(v > 0 for v in sd[:n_features]) and not any(sd[n_features:])

    r = machines.recording(key)
    data = r.load(0, done.report["window_samples"])
    ticks = machines.replay(done.brain_path, r.hours(), data, r.sample_rate_hz, 2950.0)
    assert len(ticks) == 200 and not any(t.spread for t in ticks)

    # Re-commissioning replaces the peaks and the block rather than stacking
    # them: on the same frames it gives back the same bytes.
    frames = data_dir / "again.hframe"
    machines.write_frames(frames, r.hours(), data, r.sample_rate_hz,
                          machines.speeds_for(done, r))
    again = data_dir / "again.harmonic"
    machines._run([machines._binary("commission"), frames, "--brain", done.brain_path,
                   "--out", again])
    assert again.read_bytes() == raw


def test_the_upload_endpoints_accept_a_recording(data_dir):
    pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")
    import io

    from fastapi.testclient import TestClient

    from api import main as api_main
    buf = io.BytesIO()
    np.save(buf, np.random.default_rng(2).normal(size=(4, 8192)).astype(np.float32))
    with TestClient(api_main.app) as c:
        uid = c.post("/uploads", json={"name": "bench", "sample_rate_hz": 20000,
                                        "interval_s": 60, "unit": "g"}).json()["id"]
        r = c.put(f"/uploads/{uid}/run.npy", content=buf.getvalue())
        assert r.status_code == 200 and r.json()["bytes"] == len(buf.getvalue())
        # Refused before it reaches the handler (the router does not match an
        # encoded slash) or by it; either way nothing lands outside the upload.
        assert c.put(f"/uploads/{uid}/..%2Fescape.csv", content=b"1").status_code != 200
        assert not (machines.UPLOADS_DIR / "escape.csv").exists()
        listed = {x["key"]: x for x in c.get("/recordings").json()["recordings"]}
        assert listed[f"upload:{uid}"]["snapshots"] == 4
        ov = c.get(f"/recordings/upload:{uid}/overview").json()
        assert len(ov["rms"]) == 4 and ov["unit"] == "g"
        cat = c.get("/machines").json()
        assert "10816-1-III" in cat["iso_classes"] and "za2115" in cat["bearings"]
        bad = c.post("/machines", json={**_spec(f"upload:{uid}"), "vib_unit": "g"})
        assert bad.status_code == 422


# --- shaft speed (HARM-S7-T03) ------------------------------------------------------


def test_a_recording_can_carry_its_shaft_speed(tmp_path):
    np.save(tmp_path / "run.npy", np.zeros((3, 8192), dtype=np.float32))
    meta = {"sample_rate_hz": 25600, "rpm": [1500, 1600, 1700]}
    (tmp_path / "recording.json").write_text(json.dumps(meta))
    assert list(recordings.FolderRecording(tmp_path).speeds()) == [1500, 1600, 1700]
    meta["rpm"] = [1500, 1600]              # one short
    (tmp_path / "recording.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="one positive speed per snapshot"):
        recordings.FolderRecording(tmp_path)


def test_measured_speed_needs_a_speed_column(data_dir):
    key = _upload_healthy(3)
    with pytest.raises(machines.CommissionError, match="no speed column"):
        machines.create(_spec(key, speed="measured"))
    assert machines.create(_spec(key, speed="estimated")).speed == "estimated"


def test_the_engine_follows_a_changing_speed_it_is_not_told():
    """A shaft ramping 2000 -> 3500 rpm on a 2950 rpm machine. Told nothing,
    the engine estimates the speed within 1% and measures the third harmonic
    where it actually is; held at the nominal speed, it looks in the wrong
    place and loses it."""
    from api.bridge import DEFAULT_BRAIN
    from api.rig import HarmonicRig, find_library

    if find_library() is None or not DEFAULT_BRAIN.exists():
        pytest.skip("engine or P-101 brain not built")
    fs, n = 25600, 16384
    t = np.arange(n) / fs
    rig = {mode: HarmonicRig(DEFAULT_BRAIN, window_samples=n, sample_rate_hz=fs)
           for mode in ("estimated", "given", "nominal")}
    try:
        for i, true_rpm in enumerate(np.linspace(2000.0, 3500.0, 7)):
            f = true_rpm / 60.0
            x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 3 * f * t)).astype(np.float32)
            ticks = {mode: r.feed(x, rpm={"estimated": 0.0, "given": true_rpm,
                                          "nominal": 2950.0}[mode],
                                  timestamp_us=i * 1_000_000, dt_us=1_000_000, present=0)
                     for mode, r in rig.items()}
            est = ticks["estimated"]
            assert abs(est.rpm - true_rpm) / true_rpm < 0.01, (true_rpm, est.rpm)
            assert ticks["given"].rpm == pytest.approx(true_rpm, abs=1e-3)
            k = "HARMONIC_FEAT_VIB_HARM_3X"
            assert abs(est.features[k] - ticks["given"].features[k]) < 0.15, true_rpm
            if abs(true_rpm - 2950.0) > 200:
                assert ticks["nominal"].features[k] < ticks["given"].features[k] - 2.0, true_rpm
    finally:
        for r in rig.values():
            r.close()


def test_an_unsteady_speed_estimate_is_refused():
    steady = machines.speed_check(np.full(200, 2950.0) + np.linspace(-20, 20, 200), 2950.0)
    assert steady["stable"]
    # What the IMS rig gives: a louder line at 29 Hz, and the estimate wandering.
    rng = np.random.default_rng(3)
    wandering = machines.speed_check(rng.choice([1738.0, 2124.0, 2577.0], size=200), 2000.0)
    assert not wandering["stable"]


def test_a_machine_commissions_on_a_speed_estimated_from_its_vibration(data_dir):
    """The simulated pump's shaft line is its loudest near 2950 rpm, so the
    estimate is steady and commissioning accepts it — through the C
    commissioning tool as well, which now classifies the regime on it."""
    _needs_engine()
    key = _upload_healthy(200)
    done = machines.commission(machines.create(_spec(key, speed="estimated")).id)
    assert done.status == "ready", done.error
    assert done.report["speed"]["stable"]
    assert abs(done.report["speed"]["median_rpm"] - 2950) / 2950 < 0.01
    assert done.report["false_alarms"] == 0


# --- machine families (HARM-S7-T06) --------------------------------------------------


def test_a_gearbox_commissions_as_a_gearbox(data_dir):
    """A gearbox upload, commissioned as the gearbox family: its own network,
    its tooth count written into the profile, and its alarms named in the
    gearbox's vocabulary."""
    _needs_engine()
    from kg import families
    from sim.gearbox_sim import GearboxSimulator

    if not families.path("gearbox", "network").exists():
        pytest.skip("gearbox network not built; run python scripts/build_brain.py "
                    "--family gearbox")
    sim = GearboxSimulator(seed=21, fs_vib=25600.0)
    frames = np.stack([sim.step(16384 / 25600).vibration.astype(np.float32)
                       for _ in range(200)])
    uid = machines.new_upload({"name": "synthetic gearbox", "sample_rate_hz": 25600,
                               "interval_s": 600, "unit": "g"})
    import io
    buf = io.BytesIO()
    np.save(buf, frames)
    machines.save_upload_file(uid, "run.npy", buf.getvalue())
    spec = _spec(f"upload:{uid}", name="Test gearbox", nominal_rpm=1480, vib_unit="g",
                 iso_class="10816-3-G2-rigid", family="gearbox", params={"teeth": 37},
                 bearing={"pitch_mm": 46.0, "ball_mm": 11.1, "n_balls": 9,
                          "contact_angle_deg": 0.0},
                 components=["gear", "bearing", "shaft", "coupling", "foundation"])
    m = machines.create(spec)
    assert m.family == "gearbox" and m.params == {"teeth": 37.0}
    done = machines.commission(m.id)
    assert done.status == "ready", done.error
    profile = (done.folder / f"{m.id}.profile").read_text()
    assert "order.gear_mesh = 37 spectrum" in profile and "order.mesh_2x = 74 spectrum" in profile
    assert "order.cage = 0.379 envelope" in profile
    assert "dead_zone.gear_mesh" in profile             # its own floor, by label
    assert done.report["false_alarms"] == 0

    from api import sources
    from api.bridge import EngineBridge

    b = EngineBridge(source=m.id, period_s=0.01)
    try:
        assert b.info()["classes"] == list(families.class_names("gearbox"))
        assert "HARMONIC_FEAT_VIB_ORDER:gear_mesh" in b.rig.feature_kinds
        assert sources.build(m.id).family == "gearbox"
    finally:
        b.stop()


def test_a_family_parameter_is_checked(data_dir):
    key = _upload_healthy(3)
    with pytest.raises(machines.CommissionError, match="input pinion teeth"):
        machines.create(_spec(key, family="gearbox", params={"teeth": 2}))
    with pytest.raises(machines.CommissionError, match="unknown machine family"):
        machines.create(_spec(key, family="turbine"))


def test_a_machine_re_learns_normal_after_maintenance_and_keeps_its_past(data_dir):
    """HARM-S7-T09: a new baseline, the old brain kept, the engine swapped in
    place with its history; a failed re-baseline puts everything back."""
    _needs_engine()
    key = _upload_healthy(400)            # 600 s apart: 0 to 66.5 h
    m = machines.create(_spec(key, baseline_h=(0.0, 30.0)))
    machines.commission(m.id)

    from api.bridge import EngineBridge

    b = EngineBridge(source=m.id)
    try:
        b.warm_up()
        for _ in range(20):
            b.step()
        before = [s.tick for s in b.history(10_000)]

        with pytest.raises(machines.CommissionError, match="same recording"):
            machines.rebaseline(m.id, {"baseline_h": (35.0, 66.0), "recording": "upload:x"})
        machines.rebaseline(m.id, {"baseline_h": (35.0, 66.0), "note": "bearing replaced"})
        assert machines.run_rebaseline(m.id, lambda done: b.swap_brain("Re-baselined"))

        done = machines.get(m.id)
        assert done is not None and done.status == "ready"
        assert done.baseline_h == (35.0, 66.0) and done.report["false_alarms"] == 0
        assert [x["baseline_h"] for x in done.baselines] == [[0.0, 30.0]]
        assert done.baselines[0]["note"] == "bearing replaced"
        assert (done.folder / "brain-1.harmonic").exists()

        after = [s.tick for s in b.history(10_000)]
        assert after[:len(before)] == before                # the past is kept
        assert after == sorted(set(after))                   # ticks carry on, no repeats
        assert len(b.marks) == 1 and b.marks[0]["tick"] == before[-1]

        # Too short a period: refused, and the machine keeps the brain it had.
        machines.rebaseline(m.id, {"baseline_h": (60.0, 66.0)})
        assert not machines.run_rebaseline(m.id)
        kept = machines.get(m.id)
        assert kept is not None and kept.status == "ready"
        assert kept.baseline_h == (35.0, 66.0) and len(kept.baselines) == 1
        assert "at least" in (kept.rebaseline_error or "")
        assert not (kept.folder / "brain-2.harmonic").exists()
    finally:
        b.stop()


def test_a_machine_with_two_points_names_the_point_that_alarms(data_dir):
    """HARM-S7-T07: one machine, two measurement points, each its own engine;
    a fault at one point alarms there, names it, and leaves the other quiet."""
    _needs_engine()
    from sim.pump_sim import PumpSimulator

    sim = PumpSimulator(seed=11, fs_vib=25600.0)
    de = np.stack([sim.step(16384 / 25600).vibration.astype(np.float32) for _ in range(320)])
    sim2 = PumpSimulator(seed=12, fs_vib=25600.0)
    nde = np.stack([sim2.step(16384 / 25600).vibration.astype(np.float32) for _ in range(320)])
    nde[250:] *= 6.0                                   # the non-drive end goes bad
    uid = machines.new_upload({"name": "two points", "sample_rate_hz": 25600,
                               "interval_s": 600, "unit": "mm/s"})
    import io
    buf = io.BytesIO()
    np.save(buf, np.stack([de, nde], axis=-1))
    machines.save_upload_file(uid, "run.npy", buf.getvalue())

    with pytest.raises(machines.CommissionError, match="own name and its own channel"):
        machines.create(_spec(f"upload:{uid}", points=[{"name": "DE", "channel": 0},
                                                      {"name": "DE", "channel": 1}]))
    m = machines.create(_spec(f"upload:{uid}", baseline_h=(0.0, 30.0),
                              points=[{"name": "DE", "channel": 0}, {"name": "NDE", "channel": 1}]))
    done = machines.commission(m.id)
    assert done.status == "ready" and done.point_brain(1).exists()
    assert [p["name"] for p in done.report["points"]] == ["DE", "NDE"]
    assert done.report["false_alarms"] == 0

    from api import sources
    from api.bridge import EngineBridge

    listed = sources.available()
    assert listed[m.id]["point"] == "DE" and listed[f"{m.id}@1"]["point"] == "NDE"
    assert listed[f"{m.id}@1"]["machine"] == m.id

    worst: dict[str, str] = {}
    points: dict[str, set[str]] = {}
    for key in (m.id, f"{m.id}@1"):
        b = EngineBridge(source=key)
        try:
            b.warm_up()
            while b.step() is not None:
                pass
            evs = [e for e in b.events(1000) if e.state in ("WARNING", "CRITICAL")]
            worst[key] = evs[-1].state if evs else "OK"
            points[key] = {e.point for e in evs}
        finally:
            b.stop()
    assert worst[m.id] == "OK"
    assert worst[f"{m.id}@1"] in ("WARNING", "CRITICAL") and points[f"{m.id}@1"] == {"NDE"}


def test_bearing_orders_ims_rexnord_za2115():
    # profiles/ims_rexnord_za2115.profile: 16 rollers, 0.331 in (8.41 mm) roller
    # diameter, 2.815 in (71.50 mm) pitch diameter, 15.17 deg contact angle, at
    # 2000 rpm -> BPFO 236.4 Hz, BPFI 296.9 Hz, BSF 139.9 Hz (Qiu, Lee & Lin).
    orders = machines.bearing_orders({"pitch_mm": 71.50, "ball_mm": 8.41,
                                      "n_balls": 16, "contact_angle_deg": 15.17})
    fr = 2000.0 / 60.0
    assert orders["bpfo"] * fr == pytest.approx(236.4, abs=0.1)
    assert orders["bpfi"] * fr == pytest.approx(296.9, abs=0.1)
    assert orders["bsf"] * fr == pytest.approx(139.9, abs=0.1)


def test_bearing_orders_cwru_6205():
    # profiles/motor_m301.profile: SKF 6205-2RS JEM (CWRU rig), 9 balls,
    # 7.94 mm ball, 39.04 mm pitch, 0 deg contact angle. The published CWRU
    # bearing constants (multiples of shaft speed) are BPFO 3.5848,
    # BPFI 5.4152, BSF 2.357, FTF 0.3983 -- the profile's own order.cage
    # (FTF) is already 0.398, so this also checks the new orders agree with
    # the one this path already derived.
    orders = machines.bearing_orders({"pitch_mm": 39.04, "ball_mm": 7.94,
                                      "n_balls": 9, "contact_angle_deg": 0.0})
    assert orders["ftf"] == pytest.approx(0.398, abs=0.001)
    assert orders["bpfo"] == pytest.approx(3.5848, abs=0.001)
    assert orders["bpfi"] == pytest.approx(5.4152, abs=0.001)
    assert orders["bsf"] == pytest.approx(2.357, abs=0.001)
    assert orders["bsf2"] == pytest.approx(2 * 2.357, abs=0.001)


# --- sidebands the window cannot tell from their carrier (HARM-S11-T15) -----------


def _motor(rpm: float, **over) -> machines.Machine:
    spec = {"id": "m", "name": "motor", "equipment_id": "M", "nominal_rpm": rpm,
            "vib_unit": "g", "iso_class": "none",
            "bearing": {"pitch_mm": 39.04, "ball_mm": 7.94, "n_balls": 9,
                        "contact_angle_deg": 0.0},
            "components": [], "recording": "", "channel": 0, "baseline_h": (0.0, 1.0),
            "family": "motor", "params": {"poles": 4, "line_hz": 60}}
    spec.update(over)
    return machines.Machine(**spec)


def test_pole_pass_sidebands_near_synchronous_speed_are_reported_unmeasured():
    # The CWRU benchmark motor: 4 poles on 60 Hz at 1797 rpm, 4096-sample
    # windows at 12 kHz. Pole-pass sidebands 0.0067 orders (0.20 Hz) from 1X,
    # against bins of 2.93 Hz: they read exactly twice 1X before.
    u = machines.unmeasured_sidebands(_motor(1797.0), 12000, 4096)
    assert [x["feature"] for x in u] == ["rotor_sb"]
    reason = u[0]["reason"]
    assert "0.20 Hz" in reason and "9.09 Hz" in reason and "pole-pass" in reason
    # The speed allowance alone (0.5% of 1X either way) spans 0.20 Hz.
    assert "No longer snapshot separates them" in reason
    # The family's reference machine, M-301 at 1710 rpm, at the engine's
    # reference window: 6.0 Hz against a 4.97 Hz limit, measured.
    assert machines.unmeasured_sidebands(_motor(1710.0), 25600, 16384) == []
    # The same machine on short snapshots: a longer one would do.
    short = machines.unmeasured_sidebands(_motor(1710.0), 25600, 4096)
    assert [x["feature"] for x in short] == ["rotor_sb"]
    assert "16384 samples or more" in short[0]["reason"]


def test_the_resolution_rule_is_the_engine_s_search_band():
    # 2 x (1.5 bins + order x 0.5% x 1X): at 1797 rpm, 12 kHz and 4096 samples
    # the bins are 2.93 Hz and the limit for order 1 is 9.09 Hz.
    r = machines.sideband_resolution(1.0, 0.0067, 1797.0, 12000, 4096)
    assert r["bin_hz"] == pytest.approx(12000 / 4096)
    assert r["limit_hz"] == pytest.approx(2 * (1.5 * 12000 / 4096 + 0.005 * 1797 / 60))
    assert not r["resolved"] and r["window_needed"] is None
    assert machines.sideband_resolution(1.0, 0.32, 1797.0, 12000, 4096)["resolved"]
    assert not machines.sideband_resolution(1.0, 0.29, 1797.0, 12000, 4096)["resolved"]
    # A gear mesh's sidebands at 1X on the real OpenEI gearbox: resolved.
    assert machines.sideband_resolution(32.0, 1.0, 1800.0, 12939, 4096)["resolved"]


def test_a_motor_near_synchronous_speed_commissions_without_its_rotor_sidebands(data_dir):
    """End to end: commissioning writes the reason into the report and the
    profile, measures no noise floor for the feature, and the engine reads it
    as 0 instead of 1X a second time."""
    _needs_engine()
    from api.rig import HarmonicRig
    from kg import families
    from sim.motor_sim import MotorSimulator

    if not families.path("motor", "network").exists():
        pytest.skip("motor network not built; run python scripts/build_brain.py "
                    "--family motor")
    sim = MotorSimulator(rpm=1797.0, seed=5, fs_vib=12000.0)
    frames = np.stack([sim.step(4096 / 12000).vibration.astype(np.float32)
                       for _ in range(200)])
    uid = machines.new_upload({"name": "synthetic motor", "sample_rate_hz": 12000,
                               "interval_s": 600, "unit": "g"})
    import io
    buf = io.BytesIO()
    np.save(buf, frames)
    machines.save_upload_file(uid, "run.npy", buf.getvalue())
    m = machines.create(_spec(f"upload:{uid}", name="Test motor", nominal_rpm=1797,
                              vib_unit="g", iso_class="none", family="motor",
                              params={"poles": 4, "line_hz": 60},
                              bearing=machines.BEARINGS["6205"],
                              components=["bearing", "rotor", "stator", "shaft"]))
    done = machines.commission(m.id)
    assert done.status == "ready", done.error
    assert [u["feature"] for u in done.report["unmeasured"]] == ["rotor_sb"]
    assert "rotor_sb" not in done.report["dead_zones"]
    profile = (done.folder / f"{m.id}.profile").read_text()
    assert "order.rotor_sb = 1 spectrum sidebands 0.0067" in profile
    assert "# rotor_sb: not measured" in profile and "dead_zone.rotor_sb" not in profile
    rig = HarmonicRig(done.brain_path, window_samples=4096, sample_rate_hz=12000)
    try:
        tick = rig.feed(frames[0], rpm=1797.0, present=0)
        assert tick.features["HARMONIC_FEAT_VIB_ORDER:rotor_sb"] == 0.0
        assert tick.features["HARMONIC_FEAT_VIB_HARM_1X"] != 0.0
    finally:
        rig.close()
