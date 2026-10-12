"""Loader tests (HARM-S1-T07).

MFPT and C-MAPSS are not vendored, so these run against synthetic fixtures
written in the real on-disk formats. That checks the parsing, the unit
conversion and the Frame contract — everything except whether the real files
match their documented shape, which only a real download can settle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.datasets.cmapss_loader import (
    INFORMATIVE_SENSORS,
    N_SENSORS,
    N_SETTINGS,
    load_cmapss,
)
from sim.datasets.mfpt_loader import (
    G_TO_MM_S2,
    MFPTRecord,
    acceleration_to_velocity,
    load_mfpt,
    mfpt_to_frames,
)

# --- MFPT --------------------------------------------------------------------


def make_record(fault: str = "outer_race", n: int = 4096, fs: float = 4096.0) -> MFPTRecord:
    t = np.arange(n) / fs
    # 1 g of acceleration at 50 Hz, plus a DC offset an accelerometer really has.
    accel = 1.0 * np.sin(2 * np.pi * 50.0 * t) + 0.02
    return MFPTRecord(
        name=f"{fault}_test",
        acceleration_g=accel,
        sample_rate_hz=fs,
        shaft_rate_hz=25.0,
        load_lbs=270.0,
        fault=fault,
    )


def test_acceleration_to_velocity_amplitude():
    """Integrating a sine divides its amplitude by omega. Getting this wrong
    would put every MFPT-derived level in the wrong ISO zone."""
    fs, f0, n = 4096.0, 50.0, 4096
    t = np.arange(n) / fs
    accel_g = np.sin(2 * np.pi * f0 * t)
    vel = acceleration_to_velocity(accel_g, fs)
    expected_peak = G_TO_MM_S2 / (2 * np.pi * f0)  # mm/s
    # Ignore the edges, where the high-pass has transients.
    interior = vel[n // 8 : -n // 8]
    assert np.max(np.abs(interior)) == pytest.approx(expected_peak, rel=0.15)


def test_acceleration_to_velocity_removes_dc_ramp():
    """A DC offset integrates into an unbounded ramp. If the high-pass is not
    doing its job the signal drifts away and swamps everything."""
    fs, n = 4096.0, 8192
    accel_g = np.full(n, 0.05)  # pure offset
    vel = acceleration_to_velocity(accel_g, fs)
    first_half = float(np.mean(np.abs(vel[: n // 2])))
    second_half = float(np.mean(np.abs(vel[n // 2 :])))
    assert second_half < 3.0 * first_half + 1.0


def test_mfpt_to_frames_shapes_and_labels():
    rec = make_record("outer_race", n=4096 * 3)
    frames = mfpt_to_frames(rec, window_s=1.0)
    assert len(frames) == 3
    assert all(len(f.vibration) == 4096 for f in frames)
    assert frames[0].rpm == pytest.approx(25.0 * 60.0)
    assert frames[0].label == "bearing_wear"


def test_mfpt_healthy_record_is_unlabelled():
    rec = make_record("healthy", n=4096 * 2)
    frames = mfpt_to_frames(rec, window_s=1.0)
    assert frames[0].label == "healthy"


def test_mfpt_unmeasured_channels_are_zero_not_invented():
    """A consumer must be able to tell a missing channel from a fabricated one."""
    rec = make_record("outer_race")
    frame = mfpt_to_frames(rec, window_s=1.0)[0]
    assert frame.bearing_temp_c == 0.0
    assert frame.flow_m3h == 0.0
    assert frame.discharge_bar == 0.0
    assert np.all(frame.vibration_axial == 0.0)


def test_mfpt_missing_file_says_where_to_get_it():
    with pytest.raises(FileNotFoundError, match=r"mfpt\.org"):
        load_mfpt("sim/datasets/raw/mfpt/does_not_exist.mat")


def test_mfpt_record_duration():
    rec = make_record(n=8192, fs=4096.0)
    assert rec.duration_s == pytest.approx(2.0)


# --- C-MAPSS ------------------------------------------------------------------


def write_cmapss(path: Path, n_units: int = 3, n_cycles: int = 40) -> Path:
    """Write a fixture in the real C-MAPSS layout: unit, cycle, 3 settings, 21 sensors."""
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, n_units + 1):
        life = n_cycles + unit * 5
        for cycle in range(1, life + 1):
            degrade = (cycle / life) ** 2
            sensors = np.zeros(N_SENSORS)
            for s in range(1, N_SENSORS + 1):
                base = 500.0 + s
                if s in INFORMATIVE_SENSORS:
                    sensors[s - 1] = base + 40.0 * degrade + rng.normal(0, 0.4)
                else:
                    sensors[s - 1] = base + rng.normal(0, 0.01)
            rows.append([unit, cycle, *np.zeros(N_SETTINGS), *sensors])
    np.savetxt(path, np.array(rows), fmt="%.5f")
    return path


def test_load_cmapss_groups_by_unit(tmp_path: Path):
    p = write_cmapss(tmp_path / "train_FD001.txt", n_units=3)
    units = load_cmapss(p)
    assert len(units) == 3
    assert [u.unit_id for u in units] == [1, 2, 3]
    assert units[0].sensors.shape[1] == N_SENSORS
    assert units[0].settings.shape[1] == N_SETTINGS


def test_cmapss_cycles_are_ordered(tmp_path: Path):
    p = write_cmapss(tmp_path / "train.txt", n_units=1)
    unit = load_cmapss(p)[0]
    assert np.all(np.diff(unit.cycles) > 0)


def test_cmapss_rul_counts_down_to_zero(tmp_path: Path):
    p = write_cmapss(tmp_path / "train.txt", n_units=1)
    unit = load_cmapss(p)[0]
    rul = unit.rul()
    assert rul[-1] == 0
    assert np.all(np.diff(rul) < 0)


def test_cmapss_health_proxy_rises_towards_failure(tmp_path: Path):
    """The whole point of C-MAPSS here: a real, noisy, non-linear degradation
    trajectory for the drift tracker to be checked against."""
    p = write_cmapss(tmp_path / "train.txt", n_units=1, n_cycles=80)
    unit = load_cmapss(p)[0]
    h = unit.health_proxy()
    assert h.min() >= 0.0 and h.max() <= 1.0
    assert float(np.mean(h[:10])) < float(np.mean(h[-10:]))


def test_cmapss_max_units_limits_the_read(tmp_path: Path):
    p = write_cmapss(tmp_path / "train.txt", n_units=5)
    assert len(load_cmapss(p, max_units=2)) == 2


def test_cmapss_rejects_a_short_table(tmp_path: Path):
    p = tmp_path / "bad.txt"
    np.savetxt(p, np.zeros((4, 6)))
    with pytest.raises(ValueError, match="expected at least"):
        load_cmapss(p)


def test_cmapss_missing_file_says_where_to_get_it():
    with pytest.raises(FileNotFoundError, match=r"nasa\.gov"):
        load_cmapss("sim/datasets/raw/cmapss/does_not_exist.txt")
