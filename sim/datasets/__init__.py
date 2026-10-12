"""Loaders that turn public datasets into the simulator's Frame format.

The point is that anything consuming simulator output — the feature
extractor, the words encoder, the trainer — consumes real measured data
unchanged. Synthetic data is for coverage; real data is what keeps the
baseline honest.

The datasets are not vendored. Download them and point the loaders at them:

  MFPT    https://www.mfpt.org/fault-data-sets/          -> sim/datasets/raw/mfpt/
  C-MAPSS https://data.nasa.gov/ (PCoE turbofan degradation) -> sim/datasets/raw/cmapss/
"""

from sim.datasets.cmapss_loader import CMAPSSUnit, load_cmapss
from sim.datasets.mfpt_loader import MFPTRecord, load_mfpt, mfpt_to_frames

__all__ = [
    "CMAPSSUnit",
    "MFPTRecord",
    "load_cmapss",
    "load_mfpt",
    "mfpt_to_frames",
]
