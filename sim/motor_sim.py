"""An induction motor with injectable faults — the third machine family.

A family is a simulator, a profile and a graph file (HARM-S7-T05/T10); this is
the motor's simulator. The profile is `profiles/motor_m301.profile`; the
frequencies below are the ones it declares.

The machine, M-301: a 4-pole induction motor on a 60 Hz supply, 1710 rpm at
full load (5% slip), an SKF 6205 ball bearing at the drive end — the CWRU
rig's bearing, so the family's bearing faults can be checked on CWRU's real
motor data (HARM-S7-T15) — one accelerometer in g, the winding temperature on
slow-channel slot 0 and the phase current on slot 1.

The fault signatures, from the motor diagnostics literature (motor current and
vibration signature analysis), each chosen to move its own features:

  bearing_wear    an outer-race defect at the drive end: impacts at BPFO,
                  modulated by the cage; kurtosis, crest and peak climb.
  rotor_bar       broken rotor bars: the rotor's field is uneven, so 1X grows
                  sidebands at the pole-pass frequency (poles x slip frequency)
                  and the current swings.
  stator_winding  a shorted turn or an unbalanced supply: vibration at twice
                  line frequency (120 Hz) grows, the winding runs hot and the
                  current rises.
  unbalance       mass unbalance of the rotor: 1X grows, and little else.

It is a signal model, not a physical one, and it is labelled as simulated
wherever it appears. What a real motor does is for real data (HARM-S7-T15).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NOMINAL_RPM = 1710.0
LINE_HZ = 60.0
POLES = 4
# SKF 6205-2RS JEM, as documented by the CWRU Bearing Data Center.
BEARING_PITCH_DIA_MM = 39.04
BEARING_BALL_DIA_MM = 7.94
BEARING_N_BALLS = 9
FS_VIB_DEFAULT = 25600.0

WINDING_TEMP_C = 72.0
CURRENT_A = 8.2
SLOT_WINDING_TEMP = 0
SLOT_CURRENT = 1

# Healthy amplitudes, g.
A_1X, A_2X = 0.012, 0.005        # zone A healthy: about 0.5 mm/s at 1X
A_LINE2X = 0.012
A_POLE_SB = 0.002
NOISE_G = 0.03

FAULT_MODES = ("bearing_wear", "rotor_bar", "stator_winding", "unbalance")
CLASS_NAMES = ("healthy", *FAULT_MODES)
# What the console shows for the slow channels: sensor key -> frame slot.
SENSORS = {"winding_temp_c": SLOT_WINDING_TEMP, "current_a": SLOT_CURRENT}


def mode_seed(base: int, mode: str, variant: str = "") -> int:
    """A reproducible per-fault seed (see sim.pump_sim.mode_seed for why)."""
    offset = FAULT_MODES.index(mode) * 101 if mode in FAULT_MODES else 0
    return base + offset + sum(ord(c) for c in variant) * 7


def pole_pass_hz(rpm: float) -> float:
    """Poles times the slip frequency: the spacing of rotor-bar sidebands."""
    sync_rpm = 120.0 * LINE_HZ / POLES
    return POLES * (sync_rpm - rpm) / 60.0


def bearing_frequencies(rpm: float) -> dict[str, float]:
    fr = rpm / 60.0
    ratio = BEARING_BALL_DIA_MM / BEARING_PITCH_DIA_MM
    n = BEARING_N_BALLS
    return {"shaft": fr, "bpfo": (n / 2.0) * (1.0 - ratio) * fr,
            "ftf": 0.5 * (1.0 - ratio) * fr}


@dataclass
class Fault:
    mode: str
    severity: float = 1.0
    onset_s: float = 0.0
    ramp_hours: float = 0.0

    def level(self, t_s: float) -> float:
        if t_s < self.onset_s:
            return 0.0
        if self.ramp_hours <= 0.0:
            return self.severity
        return self.severity * min(1.0, (t_s - self.onset_s) / 3600.0 / self.ramp_hours)


@dataclass
class MotorFrame:
    """One window: vibration in g, the shaft speed, and the slow channels in
    frame-slot order (winding temperature and current are measured)."""

    t_s: float
    vibration: np.ndarray
    rpm: float
    slow: list[float]
    present: int
    severities: dict[str, float] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return sorted(m for m, s in self.severities.items() if s > 0.05)


class MotorSimulator:
    """Same interface as PumpSimulator: inject, clear_faults, severities, step."""

    def __init__(self, rpm: float = NOMINAL_RPM, fs_vib: float = FS_VIB_DEFAULT,
                 seed: int = 0, time_scale: float = 1.0) -> None:
        self.rpm_nominal = rpm
        self.rpm = rpm
        self.fs_vib = fs_vib
        self.time_scale = time_scale
        self.rng = np.random.default_rng(seed)
        self.t_s = 0.0
        self.wall_s = 0.0
        self._phase = 0.0          # shaft revolutions, carried across frames
        self._line_phase = 0.0     # supply cycles, carried across frames
        self.faults: list[Fault] = []

    def inject(self, mode: str, severity: float = 1.0, onset_s: float = 0.0,
               ramp_hours: float = 0.0) -> Fault:
        if mode not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {mode!r}; known: {FAULT_MODES}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in 0..1, got {severity}")
        fault = Fault(mode, severity, onset_s, ramp_hours)
        self.faults.append(fault)
        return fault

    def clear_faults(self) -> None:
        self.faults.clear()

    def severities(self) -> dict[str, float]:
        out = {m: 0.0 for m in FAULT_MODES}
        for f in self.faults:
            out[f.mode] = min(1.0, out[f.mode] + f.level(self.t_s))
        return out

    def step(self, dt: float = 1.0) -> MotorFrame:
        if dt <= 0:
            raise ValueError("dt must be positive")
        n = round(self.fs_vib * dt)
        sev = self.severities()
        bearing, rotor = sev["bearing_wear"], sev["rotor_bar"]
        stator, unbalance = sev["stator_winding"], sev["unbalance"]

        self.rpm += 0.05 * (self.rpm_nominal - self.rpm) + self.rng.normal(0.0, 0.3)
        fr = self.rpm / 60.0
        fp = pole_pass_hz(self.rpm)
        t = np.arange(n, dtype=np.float64) / self.fs_vib
        rev = t * fr + self._phase
        two_pi = 2.0 * math.pi

        a1 = A_1X * (1.0 + 5.0 * unbalance + 0.6 * rotor)
        a2 = A_2X * (1.0 + 0.3 * unbalance)
        side = A_POLE_SB * (1.0 + 25.0 * rotor)
        line2x = A_LINE2X * (1.0 + 8.0 * stator)
        vib = (a1 * np.sin(two_pi * rev) + a2 * np.sin(two_pi * 2 * rev + 0.7)
               + side * np.sin(two_pi * (fr - fp) * t + two_pi * self._phase + 0.4)
               + side * np.sin(two_pi * (fr + fp) * t + two_pi * self._phase + 1.9)
               + line2x * np.sin(two_pi * 2.0 * LINE_HZ * t + two_pi * self._line_phase + 0.3))
        vib += self.rng.normal(0.0, NOISE_G, n)

        if bearing > 0.0:
            f = bearing_frequencies(self.rpm)
            since = np.mod(t + (self.wall_s * f["bpfo"] % 1.0) / f["bpfo"], 1.0 / f["bpfo"])
            ring = np.exp(-900.0 * since) * np.sin(two_pi * 2900.0 * since)
            cage = 1.0 + 0.5 * np.sin(two_pi * f["ftf"] * (t + self.wall_s))
            vib += 1.1 * bearing**1.5 * ring * (0.65 + 0.35 * np.sin(two_pi * rev)) * cage

        winding = (WINDING_TEMP_C + 18.0 * stator + 4.0 * rotor + 1.0 * bearing
                   + self.rng.normal(0.0, 0.3))
        current = (CURRENT_A * (1.0 + 0.12 * stator + 0.04 * rotor)
                   + 0.35 * rotor * math.sin(two_pi * fp * self.wall_s)
                   + self.rng.normal(0.0, 0.05))
        slow = [0.0] * 7
        slow[SLOT_WINDING_TEMP] = winding
        slow[SLOT_CURRENT] = current

        self._phase = float((self._phase + n / self.fs_vib * fr) % 1.0)
        self._line_phase = float((self._line_phase + n / self.fs_vib * 2.0 * LINE_HZ) % 1.0)
        frame = MotorFrame(t_s=self.t_s, vibration=vib, rpm=self.rpm, slow=slow,
                           present=(1 << SLOT_WINDING_TEMP) | (1 << SLOT_CURRENT),
                           severities=sev)
        self.t_s += dt * self.time_scale
        self.wall_s += dt
        return frame


SIMULATOR = MotorSimulator
