"""A centrifugal fan with injectable faults — the fourth machine family.

A family is a simulator, a profile and a graph file (HARM-S7-T05/T10); this is
the fan's simulator. The profile is `profiles/fan_f401.profile`; the
frequencies below are the ones it declares.

The machine, F-401: a direct-driven centrifugal fan, 8 blades, 1180 rpm (a
6-pole motor on 60 Hz), its shaft on an LDK UER204 insert bearing — the
XJTU-SY rig's bearing, whose geometry is published — one accelerometer in g,
the airflow on slow-channel slot 0 and the bearing temperature on slot 1.

The fault signatures, from the fan diagnostics literature, each chosen to move
its own features:

  unbalance      dirt build-up or erosion on the wheel: 1X grows, the
                 commonest fan fault.
  blade_damage   a cracked, bent or fouled blade: the blade-pass tone is
                 modulated at 1X, so its sidebands grow, and the airflow falls.
  looseness      a loose bearing housing or base: the response clips, so 2X,
                 3X and a half-order subharmonic appear.
  bearing_wear   an outer-race defect on the bearing: impacts at BPFO; the
                 bearing runs hot.

It is a signal model, not a physical one, and it is labelled as simulated
wherever it appears. What a real fan does is for real data (HARM-S7-T15).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NOMINAL_RPM = 1180.0
BLADES = 8
# LDK UER204, as published with the XJTU-SY bearing datasets.
BEARING_PITCH_DIA_MM = 34.55
BEARING_BALL_DIA_MM = 7.92
BEARING_N_BALLS = 8
FS_VIB_DEFAULT = 25600.0

AIRFLOW_M3S = 4.0
BEARING_TEMP_C = 45.0
SLOT_AIRFLOW = 0
SLOT_BEARING_TEMP = 1

# Healthy amplitudes, g.
A_1X, A_2X, A_3X, A_HALF = 0.035, 0.010, 0.004, 0.002
A_BPF, A_BPF_SB = 0.080, 0.006
NOISE_G = 0.03

FAULT_MODES = ("unbalance", "blade_damage", "looseness", "bearing_wear")
CLASS_NAMES = ("healthy", *FAULT_MODES)
# What the console shows for the slow channels: sensor key -> frame slot.
SENSORS = {"airflow_m3s": SLOT_AIRFLOW, "bearing_temp_c": SLOT_BEARING_TEMP}


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
class FanFrame:
    """One window: vibration in g, the shaft speed, and the slow channels in
    frame-slot order (airflow and bearing temperature are measured)."""

    t_s: float
    vibration: np.ndarray
    rpm: float
    slow: list[float]
    present: int
    severities: dict[str, float] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return sorted(m for m, s in self.severities.items() if s > 0.05)


class FanSimulator:
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

    def step(self, dt: float = 1.0) -> FanFrame:
        if dt <= 0:
            raise ValueError("dt must be positive")
        n = round(self.fs_vib * dt)
        sev = self.severities()
        unbalance, blade = sev["unbalance"], sev["blade_damage"]
        loose, bearing = sev["looseness"], sev["bearing_wear"]

        self.rpm += 0.05 * (self.rpm_nominal - self.rpm) + self.rng.normal(0.0, 0.3)
        fr = self.rpm / 60.0
        t = np.arange(n, dtype=np.float64) / self.fs_vib
        rev = t * fr + self._phase
        two_pi = 2.0 * math.pi

        a1 = A_1X * (1.0 + 5.0 * unbalance + 0.4 * blade + 0.5 * loose)
        a2 = A_2X * (1.0 + 7.0 * loose)
        a3 = A_3X * (1.0 + 12.0 * loose)
        half = A_HALF * (1.0 + 20.0 * loose)
        bpf = A_BPF * (1.0 + 0.4 * blade)
        side = A_BPF_SB * (1.0 + 12.0 * blade)
        z = BLADES
        vib = (a1 * np.sin(two_pi * rev) + a2 * np.sin(two_pi * 2 * rev + 0.7)
               + a3 * np.sin(two_pi * 3 * rev + 1.3) + half * np.sin(two_pi * 0.5 * rev + 0.2)
               + bpf * np.sin(two_pi * z * rev + 0.9)
               + side * np.sin(two_pi * (z - 1) * rev + 0.5)
               + side * np.sin(two_pi * (z + 1) * rev + 2.3))
        vib += self.rng.normal(0.0, NOISE_G, n)
        if loose > 0.0:
            # A loose mount clips the response in one direction.
            cap = float(np.percentile(vib, 99.5)) * (1.0 - 0.45 * loose)
            vib = np.minimum(vib, cap)

        if bearing > 0.0:
            f = bearing_frequencies(self.rpm)
            since = np.mod(t + (self.wall_s * f["bpfo"] % 1.0) / f["bpfo"], 1.0 / f["bpfo"])
            ring = np.exp(-800.0 * since) * np.sin(two_pi * 2600.0 * since)
            cage = 1.0 + 0.5 * np.sin(two_pi * f["ftf"] * (t + self.wall_s))
            vib += 1.1 * bearing**1.5 * ring * (0.65 + 0.35 * np.sin(two_pi * rev)) * cage

        airflow = AIRFLOW_M3S * (1.0 - 0.10 * blade - 0.02 * unbalance) + self.rng.normal(0.0, 0.02)
        btemp = (BEARING_TEMP_C + 14.0 * bearing**1.5 + 3.0 * loose + 2.0 * unbalance
                 + self.rng.normal(0.0, 0.2))
        slow = [0.0] * 7
        slow[SLOT_AIRFLOW] = airflow
        slow[SLOT_BEARING_TEMP] = btemp

        self._phase = float((self._phase + n / self.fs_vib * fr) % 1.0)
        frame = FanFrame(t_s=self.t_s, vibration=vib, rpm=self.rpm, slow=slow,
                         present=(1 << SLOT_AIRFLOW) | (1 << SLOT_BEARING_TEMP),
                         severities=sev)
        self.t_s += dt * self.time_scale
        self.wall_s += dt
        return frame


SIMULATOR = FanSimulator
