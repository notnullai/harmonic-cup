"""The pump simulator's inner-race and rolling-element faults (HARM-S9-T15).

The first test is the guard the task asks for: adding the two location faults
must not move a single sample of the four original modes for the same seed.
The production brain trains on those frames, so a change there would quietly
change the production brain.

The pinned digests were computed with the simulator as it was before the
location faults were added (commit 94a78a0). Values are rounded to 1e-6 before
hashing so the digest does not depend on the platform's last-bit `sin`; the
one-off full-precision comparison of build_brain's whole training and baseline
frame sets (float64, every field) was identical too.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sim import pump_sim
from sim.pump_sim import (
    ALL_FAULT_MODES,
    FAULT_MODES,
    LOCATION_FAULT_MODES,
    PumpSimulator,
    bearing_frequencies,
)

WINDOW = 4096
FS = 25600.0

# Computed on the pre-T15 simulator; see the module docstring.
PINNED = {
    "healthy": "b05b82f077227ff6de0fcc42039e8fa303046784e1e9a12972563ccb00bd5aa3",
    "bearing_wear": "de4dcafec29cb1c5323b4f77560dad2671e0ed9564d28e7b94c373c15d2f2ae4",
    "misalignment": "b54b11573c88b46a713133150361558e1bc510693c7f1fae504054074bb45473",
    "cavitation": "b3cd820164a84f02a6361f23afe465acd9c6baf3bd823689fb952ad56235a320",
    "seal_leak": "62701f9ce5b816727ef3d01f564eb17d2b252f8e146377e9fddff9fce8cc708a",
}


def digest(module, mode: str) -> str:
    """Twelve frames of one mode, ramped in the way build_brain does it."""
    seed = module.mode_seed(5150, mode, "0.7") if mode != "healthy" else 5150
    sim = module.PumpSimulator(seed=seed, fs_vib=FS, load=0.8)
    sim.time_scale = 60.0
    dt = WINDOW / FS
    for _ in range(3):
        sim.step(dt)
    if mode != "healthy":
        sim.inject(mode, severity=0.7, onset_s=sim.t_s, ramp_hours=0.05)
    h = hashlib.sha256()
    for _ in range(12):
        f = sim.step(dt)
        for arr in (f.vibration, f.vibration_axial):
            h.update(np.round(arr, 6).tobytes())
        h.update(repr((round(f.rpm, 6), round(f.bearing_temp_c, 6), round(f.motor_temp_c, 6),
                       round(f.discharge_bar, 6), round(f.suction_bar, 6), round(f.flow_m3h, 6),
                       round(f.current_a, 6), sorted(f.severities), f.label)).encode())
    return h.hexdigest()


@pytest.mark.parametrize("mode", list(PINNED))
def test_original_modes_are_unchanged_for_the_same_seed(mode: str):
    assert digest(pump_sim, mode) == PINNED[mode]


def test_production_modes_are_the_original_four():
    assert FAULT_MODES == ("bearing_wear", "misalignment", "cavitation", "seal_leak")
    assert ALL_FAULT_MODES == FAULT_MODES + LOCATION_FAULT_MODES
    assert set(PumpSimulator().severities()) == set(FAULT_MODES)


def _envelope_line(sim: PumpSimulator, mode: str, hz: float) -> tuple[float, float]:
    """Envelope amplitude at `hz`, faulted against healthy, same seed."""
    from scipy.signal import hilbert

    def env_spec(inject: bool) -> tuple[np.ndarray, np.ndarray]:
        s = PumpSimulator(seed=11, fs_vib=FS)
        if inject:
            s.inject(mode, severity=1.0)
        x = s.step(16384 / FS).vibration
        e = np.abs(hilbert(x))
        spec = np.abs(np.fft.rfft(e - e.mean())) / len(e)
        return np.fft.rfftfreq(len(e), 1 / FS), spec

    f, healthy = env_spec(False)
    _, faulty = env_spec(True)
    band = np.abs(f - hz) <= 3.0
    return float(faulty[band].max()), float(healthy[band].max())


@pytest.mark.parametrize("mode,line,side", [
    ("bearing_inner_race", "bpfi", "shaft"),
    ("bearing_rolling_element", "bsf2", "ftf"),
])
def test_location_fault_shows_its_line_and_side_peaks(mode: str, line: str, side: str):
    fq = bearing_frequencies(pump_sim.NOMINAL_RPM)
    carrier = 2.0 * fq["bsf"] if line == "bsf2" else fq[line]
    faulty, healthy = _envelope_line(PumpSimulator(), mode, carrier)
    assert faulty > 10 * healthy
    upper, base = _envelope_line(PumpSimulator(), mode, carrier + fq[side])
    assert upper > 5 * base


def test_location_faults_are_deterministic_and_labelled():
    a, b = PumpSimulator(seed=3), PumpSimulator(seed=3)
    for s in (a, b):
        s.inject("bearing_rolling_element", severity=0.5)
    fa, fb = a.step(0.1), b.step(0.1)
    assert np.array_equal(fa.vibration, fb.vibration)
    assert fa.labels == ["bearing_rolling_element"]
    assert set(fa.severities) == set(ALL_FAULT_MODES)


def test_location_fault_heats_the_bearing():
    cold = PumpSimulator(seed=5).step(0.1).bearing_temp_c
    s = PumpSimulator(seed=5)
    s.inject("bearing_inner_race", severity=1.0)
    assert s.step(0.1).bearing_temp_c > cold + 15.0
