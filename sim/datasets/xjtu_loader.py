"""XJTU-SY bearing run-to-failure data -> arrays for the engine.

Xi'an Jiaotong University and Changxing Sumyoung Technology ran fifteen LDK
UER204 bearings to failure on one test bed, five under each of three
operating conditions. Cite: B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid
Prognostics Approach for Estimating Remaining Useful Life of Rolling Element
Bearings", IEEE Trans. Reliability 69(1), 401-412, 2020. Licence and use:
docs/BENCHMARK_DATASETS.md (evaluation and publication with the citation;
not for training anything that ships).

Each bearing is a directory of `N.csv` files, one per snapshot: 1.28 s at
25.6 kHz (32768 rows), one snapshot a minute, `N` counting minutes from 1.
Column 1 is the horizontal accelerometer (the load direction), column 2 the
vertical one. The data is acceleration, in g, and is passed through as such.

Getting the data (not committed; `datasets/` is ignored):

    datasets/xjtu-sy/XJTU-SY_Bearing_Datasets/<condition>/Bearing<c>_<n>/<N>.csv

Five files span two parts of the published archive and unpack damaged; they
are skipped (DAMAGED), so those minutes are gaps in the run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ["HARMONIC_XJTU_ROOT"]) if "HARMONIC_XJTU_ROOT" in os.environ \
    else REPO / "datasets" / "xjtu-sy" / "XJTU-SY_Bearing_Datasets"

SAMPLE_RATE_HZ = 25600
SNAPSHOT_SAMPLES = 32768
SNAPSHOT_INTERVAL_S = 60.0
CHANNELS = ("horizontal", "vertical")
HORIZONTAL = 0          # the load direction: the default

# Condition -> (directory, shaft speed in rpm, radial load in kN), from the
# authors' write-up (docs/BENCHMARK_DATASETS.md).
CONDITIONS: dict[int, tuple[str, float, float]] = {
    1: ("35Hz12kN", 2100.0, 12.0),
    2: ("37.5Hz11kN", 2250.0, 11.0),
    3: ("40Hz10kN", 2400.0, 10.0),
}

# What failed, per bearing, as the authors label it.
FAILED: dict[str, tuple[str, ...]] = {
    "1_1": ("outer race",), "1_2": ("outer race",), "1_3": ("outer race",),
    "1_4": ("cage",), "1_5": ("inner race", "outer race"),
    "2_1": ("inner race",), "2_2": ("outer race",), "2_3": ("cage",),
    "2_4": ("outer race",), "2_5": ("outer race",),
    "3_1": ("outer race",),
    "3_2": ("inner race", "rolling element", "cage", "outer race"),
    "3_3": ("inner race",), "3_4": ("inner race",), "3_5": ("outer race",),
}
BEARINGS = tuple(FAILED)

# Damaged in unpacking (part null-padded): skipped.
DAMAGED = frozenset({
    ("2_3", 358), ("3_1", 966), ("3_1", 1832), ("3_2", 2317), ("3_4", 1105),
})


@dataclass(frozen=True)
class Snapshot:
    path: Path
    minute: int              # N of N.csv

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def hours(self) -> float:
        """Time since the first snapshot, in hours."""
        return (self.minute - 1) * SNAPSHOT_INTERVAL_S / 3600.0


def condition(bearing: str) -> int:
    return int(bearing.split("_")[0])


def rpm(bearing: str) -> float:
    return CONDITIONS[condition(bearing)][1]


def bearing_dir(bearing: str, root: Path = DEFAULT_ROOT) -> Path:
    if bearing not in FAILED:
        raise ValueError(f"unknown XJTU-SY bearing {bearing!r}; known: {BEARINGS}")
    return root / CONDITIONS[condition(bearing)][0] / f"Bearing{bearing}"


def list_snapshots(bearing: str, root: Path = DEFAULT_ROOT) -> list[Snapshot]:
    """Every usable snapshot of one bearing, in time order."""
    d = bearing_dir(bearing, root)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found. See sim/datasets/xjtu_loader.py.")
    snaps = [Snapshot(p, int(p.stem)) for p in d.glob("*.csv") if p.stem.isdigit()]
    snaps = [s for s in snaps if (bearing, s.minute) not in DAMAGED]
    snaps.sort(key=lambda s: s.minute)
    return snaps


def read_channel(snapshot: Snapshot | Path, channel: int = HORIZONTAL,
                 n_samples: int = 16384) -> np.ndarray:
    """One channel of one snapshot, in g, as float32: its first `n_samples`
    (the engine's FFT wants a power of two)."""
    path = snapshot.path if isinstance(snapshot, Snapshot) else snapshot
    lines = path.read_text().split("\n", 1)
    values = np.array(lines[1].replace(",", " ").split(), dtype=np.float32)
    if values.size % len(CHANNELS):
        raise ValueError(f"{path}: {values.size} values do not divide into 2 channels")
    data = values.reshape(-1, len(CHANNELS))
    if data.shape[0] < n_samples:
        raise ValueError(f"{path}: {data.shape[0]} samples, need {n_samples}")
    return np.ascontiguousarray(data[:n_samples, channel])


def load_run(bearing: str, channel: int = HORIZONTAL, n_samples: int = 16384,
             root: Path = DEFAULT_ROOT, cache: bool = True
             ) -> tuple[list[Snapshot], np.ndarray]:
    """A whole bearing for one channel: the snapshots and a (n, n_samples)
    array. Cached beside the data as .npy, as the IMS loader does."""
    snaps = list_snapshots(bearing, root)
    cache_path = root / f"xjtu_{bearing}_ch{channel}_{n_samples}.npy"
    if cache and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape == (len(snaps), n_samples):
            return snaps, arr
    arr = np.stack([read_channel(s, channel, n_samples) for s in snaps])
    if cache:
        np.save(cache_path, arr)
    return snaps, arr


def windows(snaps: list[Snapshot], data: np.ndarray, window: int = 16384, hop: int = 1024):
    """Each snapshot as several engine ticks: windows of `window` samples,
    `hop` apart, inside the snapshot. Yields (hours, samples, last-of-its-
    snapshot). With the defaults a 32768-sample snapshot gives 17 ticks.

    Why: a run can be as short as 42 snapshots, and the commissioning tool
    needs at least 100 warm ticks to learn from; one tick per snapshot would
    leave most runs unable to be commissioned at all."""
    per = (data.shape[1] - window) // hop + 1
    for s, row in zip(snaps, data, strict=True):
        for j in range(per):
            yield (s.hours + j * hop / SAMPLE_RATE_HZ / 3600.0,
                   row[j * hop: j * hop + window], j == per - 1)
