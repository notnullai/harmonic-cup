"""Build try mode's sample recording: NASA IMS test 2, bearing 1 (channel 0).

The run ended with an outer-race failure on bearing 1. The IMS data is a U.S.
Government Work (NASA Prognostics Data Repository; University of Cincinnati
Center for Intelligent Maintenance Systems), so it may be redistributed.
XJTU-SY and CWRU may NOT be used for this (docs/BENCHMARK_DATASETS.md).

To stay small in the image, every second snapshot is kept (984 -> 492, so
snapshots are 20 minutes apart instead of 10) and samples are stored as
float16, whose rounding is under 0.05% of each value (at most 0.0015 g, on the
largest peaks near the failure). Try mode converts the
array back to float32 when it copies it into a run.

    python scripts/make_try_sample.py   # needs datasets/ims/test2_ch0_16384.npy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "datasets" / "ims" / "test2_ch0_16384.npy"
OUT = REPO / "api" / "try_samples" / "ims_test2_bearing1.npy"


def main() -> int:
    arr = np.load(SRC)
    kept = np.ascontiguousarray(arr[::2]).astype(np.float16)
    err = float(np.abs(kept.astype(np.float32) - arr[::2]).max())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, kept)
    print(f"{SRC.name} {arr.shape} -> {OUT.name} {kept.shape} float16, "
          f"{OUT.stat().st_size / 1e6:.1f} MB, max rounding {err:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
