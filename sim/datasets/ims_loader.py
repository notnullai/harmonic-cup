"""IMS bearing run-to-failure data -> Harmonic frames.

The NSF I/UCR Center for Intelligent Maintenance Systems (University of
Cincinnati) ran Rexnord ZA-2115 bearings on a shaft at 2000 rpm under a
6000 lb radial load until they failed. The NASA Prognostics Data Repository
distributes it as "4. Bearings". It is the real-machine counterpart to the
simulator: nothing in it was injected, and the failure at the end of each test
developed on its own.

Each test is a directory of ASCII files, one per snapshot. A snapshot is one
second at 20.48 kHz (20480 rows), one column per channel, and the file name is
the time it was taken — every 10 minutes, with longer gaps where the test was
paused overnight. Test 2 has four channels, one accelerometer per bearing, and
ended with an outer-race failure in bearing 1.

The data is ACCELERATION and is passed through as acceleration, in g. It is
not integrated to velocity the way the MFPT loader does: the defect impacts
are high-frequency, and velocity divides them away — bearing 1 stays near
0.5-0.75 mm/s for 160 of its 164 hours. The machine profile
(`profiles/ims_rexnord_za2115.profile`) sets the vibration unit to g; the
engine integrates to velocity itself, over 10-1000 Hz, for ISO zones.

Getting the data (not committed; `datasets/` is ignored):

    datasets/ims/4. Bearings/IMS.7z      from the NASA repository
    datasets/ims/test2/2nd_test/...      2nd_test.rar, extracted from it

`7z` extracts the .7z. The .rar inside needs a RAR decoder: p7zip-full does
not ship one, so on Windows use the built-in `tar -xf 2nd_test.rar`
(libarchive), or install `unrar` / `p7zip-rar` on Linux.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# HARMONIC_IMS_ROOT overrides where the (large, gitignored) data lives — for
# a worktree checkout, which has no datasets/ of its own, point it at the
# main checkout's: HARMONIC_IMS_ROOT=/path/to/datasets/ims.
DEFAULT_ROOT = Path(os.environ["HARMONIC_IMS_ROOT"]) if "HARMONIC_IMS_ROOT" in os.environ \
    else REPO / "datasets" / "ims"

# 20.48 kHz, not the 20 kHz the readme states: a one-second snapshot holds
# 20,480 rows, and at 20.48 kHz test 2 bearing 1's outer-race line sits at
# 236.0 Hz in the envelope spectrum of its last 10 hours, and its second
# harmonic at 473.0 Hz, against 236.4 and 472.9 Hz from the bearing's
# geometry; read at 20 kHz they land 2.4% low, at 230.5 and 461.9 Hz.
# Gousseau, Antoni, Girardin and Griffaton (CM2016, hal-01715193) reached the
# same conclusion. Every IMS replay before 2026-10-03 used 20 kHz.
SAMPLE_RATE_HZ = 20480
SNAPSHOT_SAMPLES = 20480
RPM = 2000.0

# Which directory holds each test, and how many channels its files carry.
# Test 1 carries two channels per bearing (x and y): the multi-point test
# (HARM-S7-T07). Test 3's archive (inside "4. Bearings/IMS.7z") unpacks to a
# directory named "4th_test/txt" — a naming quirk of the NASA repackaging,
# not a different test: its own file timestamps (4 Mar - 18 Apr 2004) and the
# readme's "outer race failure, bearing 3" line agree with data set 3. It
# holds 6324 snapshots, not the 4448 the readme's table states; the readme is
# quoted as printed and the file count actually found is reported separately
# (HARM-S8, "IMS: all four recorded failures").
TESTS: dict[int, tuple[str, int]] = {
    1: ("test1/1st_test", 8),
    2: ("test2/2nd_test", 4),
    3: ("test3/4th_test/txt", 4),
}

# How many accelerometer channels each test gives per bearing (readme p.1:
# "two accelerometers for each bearing [x- and y-axes] for data set 1, one
# accelerometer for each bearing for data sets 2 and 3").
CHANNELS_PER_BEARING: dict[int, int] = {1: 2, 2: 1, 3: 1}


def bearing_axis(test: int, channel: int) -> tuple[int, str | None]:
    """0-based channel -> (1-based bearing number, axis label or None)."""
    cpb = CHANNELS_PER_BEARING[test]
    if cpb == 1:
        return channel + 1, None
    return channel // cpb + 1, ("x", "y")[channel % cpb]

# What the readme records at the end of each test: (channel, what failed).
# Channels are 0-based here; the readme counts from 1.
FAILURES: dict[int, list[tuple[int, str]]] = {
    1: [(4, "inner race defect, bearing 3"), (6, "roller element defect, bearing 4")],
    2: [(0, "outer race failure, bearing 1")],
    3: [(2, "outer race failure, bearing 3")],
}

# Frame file format, mirroring engine/include/harmonic/harmonic_frames.h.
FRAME_MAGIC = b"HFRM"
FRAME_VERSION = 1
FRAME_HEADER = "<4sIIIIIII"
FRAME_RECORD = "<QiiI7fI"
HEALTHY_LABEL = 1        # class 0 set: what `commission` accepts
# Only vibration was recorded. No temperature, pressure, flow or current, and
# the frames say so rather than filling them in.
SLOW_PRESENT_NONE = 0


@dataclass(frozen=True)
class Snapshot:
    path: Path
    taken: datetime

    @property
    def name(self) -> str:
        return self.path.name


def test_dir(test: int, root: Path = DEFAULT_ROOT) -> Path:
    if test not in TESTS:
        raise ValueError(f"unknown IMS test {test}; known: {sorted(TESTS)}")
    return root / TESTS[test][0]


def list_snapshots(test: int = 2, root: Path = DEFAULT_ROOT) -> list[Snapshot]:
    """Every snapshot of one test, in time order."""
    d = test_dir(test, root)
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found. Extract the IMS data first — see sim/datasets/ims_loader.py.")
    snaps = []
    for p in d.iterdir():
        try:
            taken = datetime.strptime(p.name, "%Y.%m.%d.%H.%M.%S")
        except ValueError:
            continue
        snaps.append(Snapshot(p, taken))
    snaps.sort(key=lambda s: s.taken)
    return snaps


def read_channel(snapshot: Snapshot | Path, channel: int, n_samples: int = 16384,
                 n_channels: int = 4) -> np.ndarray:
    """One channel of one snapshot, in g, as float32.

    The engine's FFT wants a power of two, so it takes the first `n_samples`
    of the 20480 recorded.
    """
    path = snapshot.path if isinstance(snapshot, Snapshot) else snapshot
    values = np.array(path.read_text().split(), dtype=np.float32)
    if values.size % n_channels:
        raise ValueError(f"{path}: {values.size} values do not divide into "
                         f"{n_channels} channels")
    data = values.reshape(-1, n_channels)
    if data.shape[0] < n_samples:
        raise ValueError(f"{path}: {data.shape[0]} samples, need {n_samples}")
    return np.ascontiguousarray(data[:n_samples, channel])


def load_run(test: int = 2, channel: int = 0, n_samples: int = 16384,
             root: Path = DEFAULT_ROOT, cache: bool = True
             ) -> tuple[list[Snapshot], np.ndarray]:
    """A whole test for one channel: the snapshots and a (n, n_samples) array.

    Parsing 984 ASCII files takes a minute, so the array is cached beside the
    data as .npy and reused while the snapshot list is unchanged.
    """
    snaps = list_snapshots(test, root)
    n_channels = TESTS[test][1]
    cache_path = root / f"test{test}_ch{channel}_{n_samples}.npy"
    if cache and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape == (len(snaps), n_samples):
            return snaps, arr
    arr = np.stack([read_channel(s, channel, n_samples, n_channels) for s in snaps])
    if cache:
        np.save(cache_path, arr)
    return snaps, arr


def write_frames(path: Path, snaps: list[Snapshot], data: np.ndarray,
                 label_mask: int = HEALTHY_LABEL) -> int:
    """Write snapshots as a Harmonic frame file for the C tools.

    Timestamps are the real ones from the file names, so the gaps where the
    rig was stopped overnight stay gaps.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = data.shape[1]
    t0 = snaps[0].taken
    with path.open("wb") as fh:
        fh.write(struct.pack(FRAME_HEADER, FRAME_MAGIC, FRAME_VERSION, len(snaps),
                             n_samples, SAMPLE_RATE_HZ, 5, 0, 0))
        for snap, samples in zip(snaps, data, strict=True):
            ts_us = int((snap.taken - t0).total_seconds() * 1e6)
            fh.write(struct.pack(FRAME_RECORD, ts_us, int(RPM * 65536),
                                 int(0.65 * 65536), label_mask,
                                 *([0.0] * 7), SLOW_PRESENT_NONE))
            fh.write(samples.astype(np.float32).tobytes())
    return len(snaps)
