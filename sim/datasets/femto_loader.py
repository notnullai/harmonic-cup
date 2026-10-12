"""FEMTO-ST PRONOSTIA bearing run-to-failure data -> arrays for the engine.

The IEEE PHM 2012 Prognostic Challenge data: seventeen NSK 6804DD ball
bearings run to failure on the PRONOSTIA platform (FEMTO-ST Institute), under
three operating conditions. Cite: P. Nectoux, R. Gouriveau, K. Medjaher,
E. Ramasso, B. Morello, N. Zerhouni, C. Varnier, "PRONOSTIA: An Experimental
Platform for Bearings Accelerated Degradation Tests", IEEE International
Conference on Prognostics and Health Management, 2012 (hal-00719503).
Licence and use: docs/BENCHMARK_DATASETS.md (evaluation and publication with
the citation; ask before training anything that ships).

From the challenge's details sheet (IEEEPHM2012-Challenge-Details.pdf):
  * each acc_NNNNN.csv is one snapshot of 2560 samples (0.1 s) at 25.6 kHz,
    one every 10 seconds; columns: hour, minute, second, microsecond,
    horizontal acceleration, vertical acceleration, in g;
  * each test was stopped when the vibration amplitude passed 20 g, so the
    last snapshot of a run is its failure;
  * the bearings degrade in all their elements at once ("frequency signatures
    are difficult to obtain"): there is no per-element label.

The six Learning_set bearings and the eleven Full_Test_Set bearings are all
complete runs to failure; the truncated Test_set is not used.

Getting the data (not committed; `datasets/` is ignored). The NASA
Prognostics Data Repository mirror holds them as nested archives:

    curl -o femto.zip "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
    # femto.zip > FEMTOBearingDataSet.zip > Training_set.zip, Validation_Set.zip
    datasets/femto/Training_set.zip      (Learning_set/Bearing<c>_<n>/acc_*.csv)
    datasets/femto/Validation_Set.zip    (Full_Test_Set/Bearing<c>_<n>/acc_*.csv)

`load_run` reads a bearing out of those archives once and caches it beside
them as femto_<bearing>.npy (both channels), so the archives can then go.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ["HARMONIC_FEMTO_ROOT"]) if "HARMONIC_FEMTO_ROOT" in os.environ \
    else REPO / "datasets" / "femto"

SAMPLE_RATE_HZ = 25600
SNAPSHOT_SAMPLES = 2560
SNAPSHOT_INTERVAL_S = 10.0
CHANNELS = ("horizontal", "vertical")
HORIZONTAL = 0

# Condition -> (shaft speed in rpm, radial load in N), from the details sheet.
CONDITIONS: dict[int, tuple[float, float]] = {
    1: (1800.0, 4000.0),
    2: (1650.0, 4200.0),
    3: (1500.0, 5000.0),
}

# Bearing -> (archive, folder inside it).
ARCHIVES: dict[str, tuple[str, str]] = {
    **{b: ("Training_set.zip", "Learning_set")
       for b in ("1_1", "1_2", "2_1", "2_2", "3_1", "3_2")},
    **{b: ("Validation_Set.zip", "Full_Test_Set")
       for b in ("1_3", "1_4", "1_5", "1_6", "1_7", "2_3", "2_4", "2_5", "2_6", "2_7", "3_3")},
}
BEARINGS = tuple(ARCHIVES)

_SPLIT = re.compile(rb"[,;\s]+")


def condition(bearing: str) -> int:
    return int(bearing.split("_")[0])


def rpm(bearing: str) -> float:
    return CONDITIONS[condition(bearing)][0]


def _parse(raw: bytes, name: str) -> np.ndarray:
    """One acc_*.csv as a (2560, 2) float32 array. A few files separate with
    ';' instead of ','; both are read."""
    cells = [c for c in _SPLIT.split(raw.strip()) if c]
    values = np.array(cells, dtype=np.float64)
    if values.size % 6:
        raise ValueError(f"{name}: {values.size} values do not divide into 6 columns")
    rows = values.reshape(-1, 6)
    if rows.shape[0] != SNAPSHOT_SAMPLES:
        raise ValueError(f"{name}: {rows.shape[0]} rows, expected {SNAPSHOT_SAMPLES}")
    return rows[:, 4:6].astype(np.float32)


def load_run(bearing: str, root: Path = DEFAULT_ROOT, cache: bool = True) -> np.ndarray:
    """A whole bearing: (n_snapshots, 2560, 2) in g, horizontal then vertical,
    in time order. Snapshot i was recorded i * 10 s after the first."""
    if bearing not in ARCHIVES:
        raise ValueError(f"unknown FEMTO bearing {bearing!r}; known: {BEARINGS}")
    cache_path = root / f"femto_{bearing}.npy"
    if cache and cache_path.exists():
        return np.load(cache_path)
    archive, folder = ARCHIVES[bearing]
    path = root / archive
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See sim/datasets/femto_loader.py.")
    with zipfile.ZipFile(path) as z:
        prefix = f"{folder}/Bearing{bearing}/acc_"
        names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".csv"))
        arr = np.stack([_parse(z.read(n), n) for n in names])
    if cache:
        np.save(cache_path, arr)
    return arr


def hours(n: int) -> np.ndarray:
    """Time of each snapshot since the first, in hours."""
    return np.arange(n) * SNAPSHOT_INTERVAL_S / 3600.0


def windows(data: np.ndarray, window: int, hop: int):
    """Each snapshot of one channel as several engine ticks: windows of
    `window` samples, `hop` apart, inside the snapshot. Yields (hours,
    samples, last-of-its-snapshot), as xjtu_loader.windows does."""
    per = (data.shape[1] - window) // hop + 1
    for t, row in zip(hours(len(data)), data, strict=True):
        for j in range(per):
            yield (t + j * hop / SAMPLE_RATE_HZ / 3600.0,
                   row[j * hop: j * hop + window], j == per - 1)
