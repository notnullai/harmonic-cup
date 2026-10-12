"""A single-stage gearbox with injectable faults — the second machine family.

The pump simulator (sim/pump_sim.py) taught the engine a pump. This teaches it
a gearbox, through the same pipeline and the same trainer, so that a family of
machines is added by a simulator, a profile and a fault table rather than by
engine code (HARM-S7-T05). The profile is `profiles/example_gearbox.profile`;
the frequencies below are the ones it declares.

The machine: a 37-tooth input pinion at 1480 rpm (gear mesh 912.7 Hz), a 6207
ball bearing on the input shaft, one accelerometer in g, and the oil
temperature on slow-channel slot 0.

The fault signatures, from the gearbox diagnostics literature, each chosen to
move its own features so the class neurons can tell them apart:

  tooth_wear      distributed wear: the mesh and its second harmonic grow
                  (the tooth profile distorts), the oil warms.
  broken_tooth    one damaged tooth: an impact once per revolution, so
                  kurtosis, crest and peak climb and the mesh grows strong
                  sidebands at +-1X.
  misalignment    shaft misalignment: 1X and 2X grow.
  bearing_wear    an outer-race defect on the input bearing: impacts at BPFO,
                  modulated by the cage, and the oil runs hot.

It is a signal model, not a physical one, and it is labelled as simulated
wherever it appears. What it proves is that the engine and the trainer are not
pump-shaped; what a real gearbox does is for real data (HARM-S7-T15).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NOMINAL_RPM = 1480.0
PINION_TEETH = 37
BEARING_PITCH_DIA_MM = 46.0
BEARING_BALL_DIA_MM = 11.1
BEARING_N_BALLS = 9
FS_VIB_DEFAULT = 25600.0

OIL_TEMP_C = 55.0
SLOT_OIL_TEMP = 0

# Healthy amplitudes, g.
A_1X, A_2X = 0.020, 0.008
A_MESH, A_MESH2 = 0.25, 0.06
A_SIDEBAND = 0.015
NOISE_G = 0.03

FAULT_MODES = ("tooth_wear", "broken_tooth", "misalignment", "bearing_wear")
CLASS_NAMES = ("healthy", *FAULT_MODES)
# What the console shows for the slow channels: sensor key -> frame slot.
SENSORS = {"oil_temp_c": SLOT_OIL_TEMP}


def mode_seed(base: int, mode: str, variant: str = "") -> int:
    """A reproducible per-fault seed (see sim.pump_sim.mode_seed for why)."""
    offset = FAULT_MODES.index(mode) * 101 if mode in FAULT_MODES else 0
    return base + offset + sum(ord(c) for c in variant) * 7


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
class GearFrame:
    """One window: vibration in g, the shaft speed, and the slow channels in
    frame-slot order (only the oil temperature is measured)."""

    t_s: float
    vibration: np.ndarray
    rpm: float
    slow: list[float]
    present: int
    severities: dict[str, float] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return sorted(m for m, s in self.severities.items() if s > 0.05)


class GearboxSimulator:
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

    def step(self, dt: float = 1.0) -> GearFrame:
        if dt <= 0:
            raise ValueError("dt must be positive")
        n = round(self.fs_vib * dt)
        sev = self.severities()
        wear, broken = sev["tooth_wear"], sev["broken_tooth"]
        misalign, bearing = sev["misalignment"], sev["bearing_wear"]

        self.rpm += 0.05 * (self.rpm_nominal - self.rpm) + self.rng.normal(0.0, 0.3)
        fr = self.rpm / 60.0
        t = np.arange(n, dtype=np.float64) / self.fs_vib
        rev = t * fr + self._phase
        two_pi = 2.0 * math.pi

        a1 = A_1X * (1.0 + 4.0 * misalign + 0.5 * broken)
        a2 = A_2X + 0.12 * misalign
        mesh = A_MESH * (1.0 + 2.5 * wear)
        mesh2 = A_MESH2 * (1.0 + 5.0 * wear)
        side = A_SIDEBAND * (1.0 + 0.8 * wear + 10.0 * broken)
        z = PINION_TEETH
        vib = (a1 * np.sin(two_pi * rev) + a2 * np.sin(two_pi * 2 * rev + 0.7)
               + mesh * np.sin(two_pi * z * rev + 0.2)
               + mesh2 * np.sin(two_pi * 2 * z * rev + 1.1)
               + side * np.sin(two_pi * (z - 1) * rev + 0.5)
               + side * np.sin(two_pi * (z + 1) * rev + 2.3))
        vib += self.rng.normal(0.0, NOISE_G * (1.0 + 0.5 * wear), n)

        if broken > 0.0:
            # One impact per revolution, as the damaged tooth enters mesh,
            # ringing a 3.2 kHz housing resonance.
            since = np.mod(rev, 1.0) / fr
            vib += 1.5 * broken**1.2 * np.exp(-600.0 * since) * np.sin(two_pi * 3200.0 * since)

        if bearing > 0.0:
            f = bearing_frequencies(self.rpm)
            since = np.mod(t + (self.wall_s * f["bpfo"] % 1.0) / f["bpfo"], 1.0 / f["bpfo"])
            ring = np.exp(-850.0 * since) * np.sin(two_pi * 2400.0 * since)
            # Load zone (1X) and cage (FTF) modulation of the impacts.
            cage = 1.0 + 0.5 * np.sin(two_pi * f["ftf"] * (t + self.wall_s))
            vib += 1.2 * bearing**1.5 * ring * (0.65 + 0.35 * np.sin(two_pi * rev)) * cage

        oil = (OIL_TEMP_C + 6.0 * wear + 3.0 * misalign + 10.0 * bearing**1.5
               + 1.0 * broken + self.rng.normal(0.0, 0.2))
        slow = [0.0] * 7
        slow[SLOT_OIL_TEMP] = oil

        self._phase = float((self._phase + n / self.fs_vib * fr) % 1.0)
        frame = GearFrame(t_s=self.t_s, vibration=vib, rpm=self.rpm, slow=slow,
                          present=1 << SLOT_OIL_TEMP, severities=sev)
        self.t_s += dt * self.time_scale
        self.wall_s += dt
        return frame


SIMULATOR = GearboxSimulator
