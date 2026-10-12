"""A generic "any rotating machine" signal model (HARM-S10-T01).

This is not a fifth machine family in the usual sense: it has no fault modes
at all. It exists so a machine type outside the four families (pump, gearbox,
motor, fan) can still be commissioned honestly, instead of being forced into
whichever template happens to look closest. The class list is `("healthy",)`
alone, so the trained network can never name a fault — the built-in
"unrecognised change" (novelty) path in the engine is the only thing that can
ever alarm a generic machine (see AGENTS.md, engine/src/harmonic_health.cpp).

The signal itself is deliberately plain: a 1X/2X vibration tone plus noise,
around a generic small/medium deep-groove ball bearing (SKF 6205 — CWRU's own
drive-end bearing, reused from api/try_mode.py's GENERIC_BEARING_KEY so the
same "unknown machine" default appears in both the try-it flow and the fleet
console). There is nothing here to make the class neuron discriminate,
because there is nothing for it to discriminate between.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NOMINAL_RPM = 1800.0
# SKF 6205-2RS JEM deep groove ball (CWRU drive end) — profiles/bearings.json
# key "6205", the same generic stand-in api/try_mode.py uses.
BEARING_PITCH_DIA_MM = 39.04
BEARING_BALL_DIA_MM = 7.94
BEARING_N_BALLS = 9
FS_VIB_DEFAULT = 25600.0

A_1X, A_2X = 0.06, 0.015
NOISE_G = 0.03

# No fault modes: a generic machine's brain has one class, "healthy", and
# names no fault (AGENTS.md constraint 8; docs/BUSINESS_STRATEGY.md sec. 5).
FAULT_MODES: tuple[str, ...] = ()
CLASS_NAMES = ("healthy", *FAULT_MODES)
SENSORS: dict[str, int] = {}


def mode_seed(base: int, mode: str, variant: str = "") -> int:
    """No fault mode ever selects a variant offset — kept only so this module
    has the same shape as every other family's simulator (sim.pump_sim)."""
    del mode
    return base + sum(ord(c) for c in variant) * 7


def bearing_frequencies(rpm: float) -> dict[str, float]:
    fr = rpm / 60.0
    ratio = BEARING_BALL_DIA_MM / BEARING_PITCH_DIA_MM
    n = BEARING_N_BALLS
    return {"shaft": fr, "bpfo": (n / 2.0) * (1.0 - ratio) * fr,
            "ftf": 0.5 * (1.0 - ratio) * fr}


@dataclass
class GenericFrame:
    """One window: vibration alone, no slow channels — a generic machine's
    commissioning wizard asks for nothing family-specific to write one."""

    t_s: float
    vibration: np.ndarray
    rpm: float
    slow: list[float] = field(default_factory=lambda: [0.0] * 7)
    present: int = 0
    severities: dict[str, float] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return []


class GenericSimulator:
    """Same interface as PumpSimulator (inject, clear_faults, severities,
    step), but `inject` always refuses: there is no fault mode to inject."""

    def __init__(self, rpm: float = NOMINAL_RPM, fs_vib: float = FS_VIB_DEFAULT,
                 seed: int = 0, time_scale: float = 1.0) -> None:
        self.rpm_nominal = rpm
        self.rpm = rpm
        self.fs_vib = fs_vib
        self.time_scale = time_scale
        self.rng = np.random.default_rng(seed)
        self.t_s = 0.0
        self.wall_s = 0.0
        self._phase = 0.0

    def inject(self, mode: str, severity: float = 1.0, onset_s: float = 0.0,
               ramp_hours: float = 0.0) -> None:
        del severity, onset_s, ramp_hours
        raise ValueError(f"the generic family names no fault; {mode!r} is not one of {FAULT_MODES}")

    def clear_faults(self) -> None:
        pass

    def severities(self) -> dict[str, float]:
        return {}

    def step(self, dt: float = 1.0) -> GenericFrame:
        if dt <= 0:
            raise ValueError("dt must be positive")
        n = round(self.fs_vib * dt)
        self.rpm += 0.05 * (self.rpm_nominal - self.rpm) + self.rng.normal(0.0, 0.3)
        fr = self.rpm / 60.0
        t = np.arange(n, dtype=np.float64) / self.fs_vib
        rev = t * fr + self._phase
        two_pi = 2.0 * math.pi

        vib = (A_1X * np.sin(two_pi * rev) + A_2X * np.sin(two_pi * 2 * rev + 0.7))
        vib += self.rng.normal(0.0, NOISE_G, n)

        self._phase = float((self._phase + n / self.fs_vib * fr) % 1.0)
        frame = GenericFrame(t_s=self.t_s, vibration=vib, rpm=self.rpm)
        self.t_s += dt * self.time_scale
        self.wall_s += dt
        return frame


SIMULATOR = GenericSimulator
