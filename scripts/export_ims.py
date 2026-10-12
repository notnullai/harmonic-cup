"""Export the IMS real-machine demo for Google Cloud (HARM-S8).

The raw NASA IMS files are 6 GB of text and live only on the PC that
downloaded them. The console needs far less: the failed bearing's snapshots
cut to the engine's window, their timestamps, and the brain commissioned on
the rig. This writes that bundle into the data folder, which on Google Cloud
is a Cloud Storage bucket (deploy/gcp/deploy.sh copies it there):

    var/ims/test2/stamps.json           984 timestamps
    var/ims/test2/ch0_16384.npy         bearing 1, float32, 984 x 16384 (64 MB)
    var/ims/test2/ims_test2_b1.harmonic the rig's commissioned brain

    python scripts/export_ims.py                  # bearing 1
    python scripts/export_ims.py --channels 0 2   # and bearing 3
    python scripts/export_ims.py --test 1 --channels 4 5
                                                  # test 1, bearing 3 (x and y):
                                                  # a machine commissioned on it

The data is the NASA Prognostics Data Repository's IMS bearing set, credited
wherever it is shown.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, choices=(1, 2), default=2)
    ap.add_argument("--channels", type=int, nargs="+", default=[0])
    ap.add_argument("--window", type=int, default=16384)
    args = ap.parse_args(argv)

    import numpy as np

    from api.sources import IMS_BRAIN
    from sim.datasets import ims_loader
    from sim.datasets.recordings import ims_bundle_dir

    test = args.test
    if not ims_loader.test_dir(test).is_dir():
        sys.exit(f"ERROR: the raw IMS test {test} data is not here (see sim/datasets/ims_loader.py)")
    # Test 2 is the built-in demo and carries its brain; a machine commissioned
    # on another test carries its own.
    if test == 2 and not IMS_BRAIN.exists():
        sys.exit("ERROR: the IMS brain is not built; run python scripts/evaluate_ims.py")
    out = ims_bundle_dir(test)
    out.mkdir(parents=True, exist_ok=True)
    snaps = ims_loader.list_snapshots(test)
    (out / "stamps.json").write_text(json.dumps([s.taken.isoformat() for s in snaps]))
    for ch in args.channels:
        _, data = ims_loader.load_run(test, ch, args.window)
        np.save(out / f"ch{ch}_{args.window}.npy", data.astype(np.float32))
        print(f"channel {ch}: {data.shape}", file=sys.stderr)
    if test == 2:
        shutil.copyfile(IMS_BRAIN, out / IMS_BRAIN.name)
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"wrote {out} ({size:.0f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
