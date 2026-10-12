"""Acceptance tests for the pump simulator (HARM-S1-T01..T05).

Each fault is checked against the signature a vibration analyst would use to
identify it, not against the code that produced it — so the test would still
be meaningful if the implementation were replaced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import analysis as A
from sim.pump_sim import (
    BEARING_TEMP_C,
    DISCHARGE_BAR,
    FLOW_M3H,
    NOMINAL_RPM,
    PumpSimulator,
    bearing_frequencies,
)

SETTLE_FRAMES = 5


def settled(mode: str | None = None, severity: float = 1.0, seed: int = 11):
    """Run a few frames so transients pass, then return (sim, last_frame)."""
    sim = PumpSimulator(seed=seed)
    if mode is not None:
        sim.inject(mode, severity=severity)
    frame = None
    for _ in range(SETTLE_FRAMES):
        frame = sim.step(1.0)
    assert frame is not None
    return sim, frame


# --- HARM-S1-T01: healthy baseline ------------------------------------------


def test_bearing_frequencies_match_hand_calculation():
    f = bearing_frequencies(NOMINAL_RPM)
    # 8 balls, 15 mm ball, 65 mm pitch, 0 deg contact angle, 49.167 Hz shaft.
    assert f["shaft"] == pytest.approx(49.167, abs=0.01)
    assert f["bpfo"] == pytest.approx(151.28, abs=0.1)
    assert f["bpfi"] == pytest.approx(242.05, abs=0.1)
    assert f["bsf"] == pytest.approx(100.85, abs=0.1)
    # The two race frequencies must sum to n * shaft rate.
    assert f["bpfo"] + f["bpfi"] == pytest.approx(8 * f["shaft"], rel=1e-9)


def test_healthy_spectrum_peaks_at_1x():
    sim, frame = settled()
    freqs, amp = A.spectrum(frame.vibration, sim.fs_vib)
    band = (freqs > 10) & (freqs < 500)
    peak_hz = float(freqs[band][np.argmax(amp[band])])
    assert peak_hz == pytest.approx(NOMINAL_RPM / 60.0, abs=1.5)


def test_healthy_is_in_iso_zone_a():
    """A healthy machine must sit below the 1.8 mm/s zone-A bound, or every
    downstream threshold is calibrated against a machine already in trouble."""
    _, frame = settled()
    assert A.rms(frame.vibration) < 1.8


def test_healthy_slow_channels_near_nominal():
    _, frame = settled()
    assert frame.rpm == pytest.approx(NOMINAL_RPM, rel=0.02)
    assert frame.flow_m3h == pytest.approx(FLOW_M3H, rel=0.02)
    assert frame.bearing_temp_c == pytest.approx(BEARING_TEMP_C, abs=1.5)
    assert frame.discharge_bar == pytest.approx(DISCHARGE_BAR, rel=0.02)


def test_healthy_label_is_healthy():
    _, frame = settled()
    assert frame.label == "healthy"
    assert frame.labels == []


def test_determinism():
    """Same seed, same call sequence, bit-identical output."""
    a = PumpSimulator(seed=99)
    b = PumpSimulator(seed=99)
    a.inject("bearing_wear", severity=0.6)
    b.inject("bearing_wear", severity=0.6)
    for _ in range(3):
        fa, fb = a.step(1.0), b.step(1.0)
        assert np.array_equal(fa.vibration, fb.vibration)
        assert fa.flow_m3h == fb.flow_m3h
        assert fa.rpm == fb.rpm


def test_different_seeds_differ():
    a = PumpSimulator(seed=1).step(1.0)
    b = PumpSimulator(seed=2).step(1.0)
    assert not np.array_equal(a.vibration, b.vibration)


def test_frames_are_continuous():
    """Consecutive frames must join without a phase jump, or every window
    boundary would look like an impulse to the feature extractor."""
    sim = PumpSimulator(seed=5)
    f1 = sim.step(1.0)
    f2 = sim.step(1.0)
    joined = np.concatenate([f1.vibration[-256:], f2.vibration[:256]])
    # A discontinuity would show up as a step far larger than the local noise.
    steps = np.abs(np.diff(joined))
    assert float(np.max(steps)) < 10.0 * float(np.median(steps)) + 1.0


# --- HARM-S1-T02: bearing wear ----------------------------------------------


def test_bearing_wear_shows_bpfo_in_envelope_spectrum():
    """The acceptance criterion: the fault is visible at BPFO in the envelope
    spectrum. Raw spectrum would not do — the defect rate appears as
    modulation of a high-frequency resonance, not as a tone."""
    sim, frame = settled("bearing_wear")
    bf = bearing_frequencies(sim.rpm)
    freqs, amp = A.envelope_spectrum(frame.vibration, sim.fs_vib)
    band = (freqs > 20) & (freqs < 600)
    at_bpfo = A.amplitude_at(freqs, amp, bf["bpfo"], tolerance_hz=3.0)
    floor = float(np.median(amp[band]))
    assert at_bpfo > 10.0 * floor


def test_bearing_wear_absent_when_healthy():
    sim, frame = settled()
    bf = bearing_frequencies(sim.rpm)
    freqs, amp = A.envelope_spectrum(frame.vibration, sim.fs_vib)
    band = (freqs > 20) & (freqs < 600)
    at_bpfo = A.amplitude_at(freqs, amp, bf["bpfo"], tolerance_hz=3.0)
    floor = float(np.median(amp[band]))
    assert at_bpfo < 10.0 * floor


def test_bearing_wear_raises_temperature():
    """Friction heats the bearing — but well after the vibration moves, which
    is why temperature alone is a late indicator."""
    _, healthy = settled()
    _, worn = settled("bearing_wear")
    assert worn.bearing_temp_c > healthy.bearing_temp_c + 15.0


def test_bearing_wear_rms_is_monotone_in_severity():
    levels = []
    for sev in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, frame = settled("bearing_wear", severity=sev) if sev else settled()
        levels.append(A.rms(frame.vibration))
    assert all(a < b for a, b in zip(levels, levels[1:], strict=False))


# --- HARM-S1-T03: misalignment ----------------------------------------------


def test_misalignment_2x_over_1x_exceeds_half():
    """Acceptance: 2X/1X > 0.5 at severity 1.0. The ratio is the signature —
    a high 1X alone is imbalance, not misalignment."""
    sim, frame = settled("misalignment")
    freqs, amp = A.spectrum(frame.vibration, sim.fs_vib)
    f1 = sim.rpm / 60.0
    a1 = A.amplitude_at(freqs, amp, f1)
    a2 = A.amplitude_at(freqs, amp, 2 * f1)
    assert a1 > 0
    assert a2 / a1 > 0.5


def test_misalignment_raises_3x_too():
    sim, frame = settled("misalignment")
    _, healthy = settled()
    freqs, amp = A.spectrum(frame.vibration, sim.fs_vib)
    hfreqs, hamp = A.spectrum(healthy.vibration, sim.fs_vib)
    f1 = sim.rpm / 60.0
    assert A.amplitude_at(freqs, amp, 3 * f1) > A.amplitude_at(hfreqs, hamp, 3 * f1)


def test_misalignment_drives_axial_vibration():
    """Strong axial motion is what separates misalignment from imbalance."""
    _, healthy = settled()
    _, mis = settled("misalignment")
    healthy_ratio = A.rms(healthy.vibration_axial) / A.rms(healthy.vibration)
    mis_ratio = A.rms(mis.vibration_axial) / A.rms(mis.vibration)
    assert mis_ratio > healthy_ratio * 1.5


def test_healthy_2x_over_1x_is_low():
    sim, frame = settled()
    freqs, amp = A.spectrum(frame.vibration, sim.fs_vib)
    f1 = sim.rpm / 60.0
    ratio = A.amplitude_at(freqs, amp, 2 * f1) / A.amplitude_at(freqs, amp, f1)
    assert ratio < 0.5


# --- HARM-S1-T04: cavitation -------------------------------------------------


@pytest.mark.parametrize("severity", [0.2, 0.5, 1.0])
def test_cavitation_kurtosis_above_four(severity: float):
    """Acceptance: kurtosis > 4. Checked across severities, because the point
    of the feature is early detection — passing only at severity 1.0 would
    mean the machine is already failing by the time it fires."""
    _, frame = settled("cavitation", severity=severity)
    assert A.kurtosis(frame.vibration) > 4.0


def test_healthy_kurtosis_is_not_impulsive():
    _, frame = settled()
    assert A.kurtosis(frame.vibration) < 4.0


def test_cavitation_raises_crest_factor():
    _, healthy = settled()
    _, cav = settled("cavitation")
    assert A.crest_factor(cav.vibration) > A.crest_factor(healthy.vibration) * 2.0


def test_cavitation_oscillates_discharge_pressure():
    """The 0.3 Hz cavity growth-and-collapse cycle shows in discharge pressure."""
    sim = PumpSimulator(seed=3)
    sim.inject("cavitation", severity=1.0)
    p = np.array([sim.step(1.0).discharge_bar for _ in range(60)])
    healthy_sim = PumpSimulator(seed=3)
    q = np.array([healthy_sim.step(1.0).discharge_bar for _ in range(60)])
    assert float(np.std(p)) > 3.0 * float(np.std(q))


def test_cavitation_drops_suction_pressure():
    _, healthy = settled()
    _, cav = settled("cavitation")
    assert cav.suction_bar < healthy.suction_bar * 0.7


# --- HARM-S1-T05: seal leak --------------------------------------------------


def test_seal_leak_drops_flow_by_fifteen_percent():
    """Acceptance: flow drops 15% at severity 1.0."""
    _, frame = settled("seal_leak")
    drop = 1.0 - frame.flow_m3h / FLOW_M3H
    assert drop == pytest.approx(0.15, abs=0.02)


def test_seal_leak_drops_discharge_pressure():
    _, healthy = settled()
    _, leak = settled("seal_leak")
    assert leak.discharge_bar < healthy.discharge_bar * 0.9


def test_seal_leak_is_quiet_in_vibration():
    """A seal leak is a process fault, not a mechanical one: it must NOT show
    up as vibration, or the engine would diagnose it as a bearing problem."""
    _, healthy = settled()
    _, leak = settled("seal_leak")
    assert A.rms(leak.vibration) == pytest.approx(A.rms(healthy.vibration), rel=0.25)


# --- ramps, labels and the fault API ----------------------------------------


def test_ramp_reaches_severity_over_the_ramp_period():
    sim = PumpSimulator(seed=4)
    fault = sim.inject("bearing_wear", severity=1.0, ramp_hours=2.0)
    assert fault.level(0.0) == 0.0
    assert fault.level(3600.0) == pytest.approx(0.5)
    assert fault.level(7200.0) == pytest.approx(1.0)
    assert fault.level(99999.0) == pytest.approx(1.0)


def test_onset_delays_the_fault():
    sim = PumpSimulator(seed=4)
    fault = sim.inject("cavitation", severity=0.8, onset_s=300.0)
    assert fault.level(299.0) == 0.0
    assert fault.level(300.0) == pytest.approx(0.8)


def test_faults_can_co_occur():
    """Class neurons are independent (AGENTS.md constraint 5), so the
    simulator must be able to produce genuinely multi-label frames."""
    sim = PumpSimulator(seed=8)
    sim.inject("bearing_wear", severity=0.8)
    sim.inject("misalignment", severity=0.7)
    frame = sim.step(1.0)
    assert frame.labels == ["bearing_wear", "misalignment"]
    assert frame.label == "bearing_wear"  # the dominant one


def test_unknown_fault_mode_is_rejected():
    sim = PumpSimulator()
    with pytest.raises(ValueError, match="unknown fault mode"):
        sim.inject("gremlins")


def test_severity_out_of_range_is_rejected():
    sim = PumpSimulator()
    with pytest.raises(ValueError, match="severity"):
        sim.inject("cavitation", severity=1.5)


def test_step_requires_positive_dt():
    sim = PumpSimulator()
    with pytest.raises(ValueError, match="dt"):
        sim.step(0.0)


def test_run_yields_the_expected_number_of_frames():
    sim = PumpSimulator(seed=2)
    frames = list(sim.run(duration_s=5.0, dt=1.0))
    assert len(frames) == 5
    assert frames[0].t_s == 0.0
    assert frames[-1].t_s == 4.0


def test_window_length_follows_sample_rate():
    sim = PumpSimulator(seed=2, fs_vib=8192.0)
    frame = sim.step(0.5)
    assert len(frame.vibration) == 4096
    assert math.isclose(len(frame.vibration_axial), 4096)


def test_a_compressed_ramp_spans_the_frames_it_is_measured_over():
    """One frame is `dt * time_scale` of machine time, not `time_scale`.

    This is the arithmetic that broke the Phase 3 detection benchmark. A ramp
    was scheduled to last `onset` minutes on the belief that one frame was one
    machine minute; at a 4096-sample window and 60x compression a frame is 9.6
    machine-seconds, so the ramp took 6.25x as many frames as the window it
    was measured over. The onset never passed 16% severity, and the fault that
    crossed the alarm threshold last appeared to be undetectable.
    """
    window, fs, time_scale, onset_frames = 4096, 25600.0, 60.0, 150
    dt = window / fs
    sim = PumpSimulator(seed=11, fs_vib=fs)
    sim.time_scale = time_scale
    frame_machine_s = dt * time_scale
    assert math.isclose(frame_machine_s, 9.6)

    sim.inject("misalignment", severity=1.0, onset_s=sim.t_s,
               ramp_hours=onset_frames * frame_machine_s / 3600.0)
    for _ in range(onset_frames):
        sim.step(dt)
    assert math.isclose(sim.severities()["misalignment"], 1.0, abs_tol=0.01)


def test_no_source_file_is_git_ignored():
    """A .gitignore pattern must never swallow source.

    `.gitignore` carried an unanchored `datasets/`, meant for downloaded
    corpora, which also matched `sim/datasets/` — the MFPT and C-MAPSS loader
    source. The loaders were never committed, while the test that imports them
    was, so the suite passed only on the machine it was written on and every
    clean clone failed at collection.

    Cheap to check and the failure mode is invisible without it: an ignored
    file looks present locally and is absent everywhere else.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        pytest.skip("not a git checkout")

    sources = [
        p for d in ("sim", "kg", "rag", "api", "dashboard", "scripts", "engine", "tests")
        for p in (repo / d).rglob("*")
        if p.suffix in {".py", ".cpp", ".h", ".sh"} and "__pycache__" not in p.parts
    ]
    assert sources, "found no source files to check"

    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(str(p.relative_to(repo).as_posix()) for p in sources),
        capture_output=True, text=True, cwd=repo,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these source files are git-ignored and would be missing from a clone:\n  "
        + "\n  ".join(ignored)
    )
