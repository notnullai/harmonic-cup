"""NASA C-MAPSS turbofan degradation dataset -> degradation trends.

C-MAPSS is run-to-failure data: many engines, each logged once per cycle until
it fails. It contributes something the bearing datasets cannot — real
*degradation trajectories*, which is what the health index slope and the RUL
estimate are actually modelling.

It is not vibration data and does not become Harmonic Frames. It is used to
check that the drift tracker behaves sensibly on real monotone-ish degradation:
noisy, non-linear, and with a knee, rather than the clean ramp a simulator
produces.

File format — one row per engine cycle, space separated, no header:

    unit  cycle  setting_1 setting_2 setting_3  sensor_1 ... sensor_21
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SETTINGS = 3
N_SENSORS = 21

# The sensors that actually carry degradation signal in FD001; the rest are
# constant for that operating condition and only add noise to a trend fit.
INFORMATIVE_SENSORS = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)


@dataclass
class CMAPSSUnit:
    """One engine's life, from first cycle to failure."""

    unit_id: int
    cycles: np.ndarray       # (n,)
    settings: np.ndarray     # (n, 3)
    sensors: np.ndarray      # (n, 21)

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    def rul(self) -> np.ndarray:
        """Remaining useful life at each cycle, in cycles.

        In the training files every unit runs to failure, so RUL is simply the
        countdown to the last recorded cycle — the ground truth a heuristic
        slope-based estimate gets scored against.
        """
        return self.cycles[-1] - self.cycles

    def health_proxy(self) -> np.ndarray:
        """A 0..1 degradation index from the informative sensors.

        Each informative sensor is z-scored against its own first ten cycles
        (the healthy period for that engine), and the magnitudes are averaged.
        Normalising per unit matters: engines start in visibly different
        condition, so a fleet-wide normalisation would encode that spread as
        degradation.
        """
        idx = [s - 1 for s in INFORMATIVE_SENSORS]
        sel = self.sensors[:, idx]
        healthy = sel[: min(10, len(sel))]
        mu = healthy.mean(axis=0)
        sd = healthy.std(axis=0)
        sd[sd == 0] = 1.0
        z = np.abs((sel - mu) / sd)
        raw = z.mean(axis=1)
        span = float(raw.max() - raw.min())
        if span == 0:
            return np.zeros_like(raw)
        return (raw - raw.min()) / span


def load_cmapss(path: str | Path, max_units: int | None = None) -> list[CMAPSSUnit]:
    """Read a C-MAPSS train_FDxxx.txt / test_FDxxx.txt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. C-MAPSS is not vendored — download the PCoE "
            "turbofan degradation set from https://data.nasa.gov/ into "
            "sim/datasets/raw/cmapss/"
        )

    raw = np.loadtxt(p)
    if raw.ndim != 2:
        raise ValueError(f"{p}: expected a 2-D table, got shape {raw.shape}")
    expected = 2 + N_SETTINGS + N_SENSORS
    if raw.shape[1] < expected:
        raise ValueError(
            f"{p}: expected at least {expected} columns "
            f"(unit, cycle, {N_SETTINGS} settings, {N_SENSORS} sensors), "
            f"got {raw.shape[1]}"
        )

    units: list[CMAPSSUnit] = []
    for unit_id in np.unique(raw[:, 0]).astype(int):
        rows = raw[raw[:, 0] == unit_id]
        rows = rows[np.argsort(rows[:, 1])]
        units.append(
            CMAPSSUnit(
                unit_id=int(unit_id),
                cycles=rows[:, 1].astype(int),
                settings=rows[:, 2 : 2 + N_SETTINGS],
                sensors=rows[:, 2 + N_SETTINGS : 2 + N_SETTINGS + N_SENSORS],
            )
        )
        if max_units is not None and len(units) >= max_units:
            break
    return units
