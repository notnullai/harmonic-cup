"""XJTU-SY, held out: detection and the bearing-location refiner (HARM-S9-T15).

XJTU-SY was never used to set anything. The refiner margin is fixed at 0.15
(chosen by the owner on IMS); nothing is tuned here. One fixed method for all
15 bearings, stated before any result was seen:

  * Learning window: the first 30% of each run's usable snapshots,
    int(0.3 * n). Warm-up: 60 ticks, as on IMS and in the product.
  * Ticks: each 1.28 s snapshot is 17 ticks, windows of 16384 samples 1024
    apart (xjtu_loader.windows). One tick per snapshot is not possible: the
    commissioning tool needs 100 warm ticks, and the shortest run's learning
    window holds 12 snapshots. Each snapshot is reported by its last tick
    (state, scores, surprise), so every count below is in snapshots.
  * Commissioning, the product's per-sensor method (api/machines.py): mint a
    draft brain, replay the learning window with the guard off, measure each
    feature's floor (measure_floor), mint again with those floors, commission
    on the learning window. Done separately for each bearing and each brain.
  * Input: the horizontal channel (the load direction); 16384-sample
    windows, the window both networks were trained at, at the same 25.6 kHz;
    the condition's speed fed as known.
  * Production is the pump network (sim/artifacts/trained.harmonic) on
    profiles/xjtu_ldk_uer204.profile; the candidate is
    var/candidates/pump_loc_trained.harmonic on
    profiles/pump_loc_xjtu_uer204.profile.
  * Alarmed = every warm tick not OK (evaluate_ims.diagnose's convention).
    Names come from the knowledge graph as for IMS, with the IMS-RIG
    equipment node (a bearing test rig: bearing, shaft, foundation), then
    evaluate_location_refiner.refine at 0.15.
  * Right: the name is one of the elements the authors say failed. A cage
    failure is right only if the name is cage.

    python scripts/evaluate_xjtu.py [--bearings 1_1,2_3] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HARMONIC_DATA_DIR", tempfile.mkdtemp(prefix="xjtu-"))

from scripts.build_brain import binary  # noqa: E402
from scripts.build_location_candidate import CLASSES as CAND  # noqa: E402
from scripts.build_location_candidate import OUT_DIR  # noqa: E402
from scripts.evaluate_location_candidate import CURRENT_CLASSES  # noqa: E402
from scripts.evaluate_location_refiner import (  # noqa: E402
    INNER,
    OUTER,
    ROLL,
    production_names,
    refine,
)
from sim.datasets import xjtu_loader as xl  # noqa: E402

MARGIN = 0.15
LEARN_FRACTION = 0.30
WARMUP = 60
WINDOW = 16384
HOP = 1024
WORK = REPO / "var" / "xjtu"
KINDS = {
    "prod": (REPO / "profiles" / "xjtu_ldk_uer204.profile",
             REPO / "sim" / "artifacts" / "trained.harmonic", CURRENT_CLASSES),
    "cand": (REPO / "profiles" / "pump_loc_xjtu_uer204.profile",
             OUT_DIR / "pump_loc_trained.harmonic", CAND),
}
# The knowledge graph has no cage failure mode, so a cage-only failure can
# never be named right; it is scored, not dropped.
ELEMENT_FM = {"outer race": OUTER, "inner race": INNER, "rolling element": ROLL,
              "cage": None}


def _run(cmd: list) -> str:
    r = subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ERROR: {Path(str(cmd[0])).name} failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r.stdout


def profile_for(template: Path, rpm: float, warmup: int,
                floor: dict[str, float] | None) -> str:
    text = template.read_text()
    text = re.sub(r"(?m)^nominal_rpm = .*$", f"nominal_rpm = {rpm:g}", text)
    text = re.sub(r"(?m)^warmup_ticks = .*$", f"warmup_ticks = {warmup}", text)
    if floor:
        text += "\n# Measured on this bearing's learning window (measure_floor).\n"
        text += "".join(f"dead_zone.{k} = {v:.2f}\n" for k, v in floor.items())
    return text


def replay(brain: Path, items, rpm: float, guard: bool = True, last_only: bool = False):
    """Ticks through the engine; `items` from xjtu_loader.windows."""
    from api.rig import HarmonicRig

    rig = HarmonicRig(brain, window_samples=WINDOW, sample_rate_hz=xl.SAMPLE_RATE_HZ)
    if not guard:
        rig.set_guard(False)
    out, t0, prev = [], None, None
    try:
        for h, x, last in items:
            t0 = h if t0 is None else t0
            prev = h if prev is None else prev
            tick = rig.feed(np.ascontiguousarray(x, dtype=np.float32), rpm=rpm, load=0.65,
                            timestamp_us=int((h - t0) * 3.6e9),
                            dt_us=max(int((h - prev) * 3.6e9), 1), present=0)
            prev = h
            if last or not last_only:
                out.append(tick)
    finally:
        rig.close()
    return out


def write_frames(path: Path, items: list, rpm: float) -> None:
    """The learning window as a frame file, labelled healthy, vibration only
    (api/machines.py write_frames, over sub-window ticks)."""
    import struct

    from api.machines import FRAME_HEADER, FRAME_MAGIC, FRAME_RECORD, FRAME_VERSION, HEALTHY_LABEL

    h0 = items[0][0]
    with path.open("wb") as fh:
        fh.write(struct.pack(FRAME_HEADER, FRAME_MAGIC, FRAME_VERSION, len(items), WINDOW,
                             xl.SAMPLE_RATE_HZ, 5, 0, 0))
        for h, x, _ in items:
            fh.write(struct.pack(FRAME_RECORD, int((h - h0) * 3.6e9), int(rpm * 65536),
                                 int(0.65 * 65536), HEALTHY_LABEL, *([0.0] * 7), 0))
            fh.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())


def commission(bearing: str, kind: str, learn: list, channel: int = xl.HORIZONTAL) -> Path:
    from api.machines import measure_floor, net_args

    template, network, _ = KINDS[kind]
    rpm = xl.rpm(bearing)
    # The vertical channel is its own sensor, commissioned on its own floor.
    tag = f"{bearing}_{kind}" + ("" if channel == xl.HORIZONTAL else f"_ch{channel}")
    WORK.mkdir(parents=True, exist_ok=True)
    mint = [binary("mint_brain")]
    draft_p, draft = WORK / f"{tag}_draft.profile", WORK / f"{tag}_draft.harmonic"
    draft_p.write_text(profile_for(template, rpm, WARMUP, None))
    _run([*mint, draft, *net_args(network), "--profile", draft_p, "--weights-from", network])
    floor = measure_floor(replay(draft, learn, rpm, guard=False))
    prof, base = WORK / f"{tag}.profile", WORK / f"{tag}_base.harmonic"
    prof.write_text(profile_for(template, rpm, WARMUP, floor))
    _run([*mint, base, *net_args(network), "--profile", prof, "--weights-from", network])
    frames, out = WORK / f"{tag}_learn.hframe", WORK / f"{tag}.harmonic"
    write_frames(frames, learn, rpm)
    _run([binary("commission"), frames, "--brain", base, "--out", out])
    for p in (draft_p, draft, base, frames):
        p.unlink(missing_ok=True)
    return out


def run_bearing(bearing: str, channel: int = xl.HORIZONTAL) -> dict:
    snaps, data = xl.load_run(bearing, channel, xl.SNAPSHOT_SAMPLES)
    hours = np.array([s.hours for s in snaps])
    n = len(snaps)
    n_learn = int(LEARN_FRACTION * n)
    learn = list(xl.windows(snaps[:n_learn], data[:n_learn], WINDOW, HOP))
    ticks = {}
    for kind in KINDS:
        brain = commission(bearing, kind, learn, channel)
        ticks[kind] = replay(brain, xl.windows(snaps, data, WINDOW, HOP), xl.rpm(bearing),
                             last_only=True)
    prod = ticks["prod"]
    end_h = float(hours[-1])
    learn_warm = [t for t in prod[:n_learn] if t.warm]
    after = [(h, t) for h, t in zip(hours[n_learn:], prod[n_learn:], strict=True)
             if t.warm and t.state != "OK"]
    crit = next((h for h, t in zip(hours, prod, strict=True)
                 if t.warm and t.state in ("CRITICAL", "FAILED")), None)
    frames = [{"state": a.state if a.warm else "COLD", "hhi": a.hhi,
               "cur": {c: round(float(s), 4) for c, s in zip(CURRENT_CLASSES, a.class_scores,
                                                             strict=False)},
               "cand": {c: round(float(s), 4) for c, s in zip(CAND, b.class_scores,
                                                              strict=False)},
               "top_surprised": a.top_surprised()}
              for a, b in zip(prod, ticks["cand"], strict=True)]
    return {"bearing": bearing, "channel": channel, "failed": list(xl.FAILED[bearing]),
            "snapshots": n,
            "n_learn": n_learn, "end_min": round(end_h * 60),
            "learn_false_alarms": sum(t.state != "OK" for t in learn_warm),
            "learn_warm": len(learn_warm),
            "first_alarm_min": None if not after else round(after[0][0] * 60),
            "lead_min": None if not after else round((end_h - after[0][0]) * 60),
            "first_critical_min": None if crit is None else round(crit * 60),
            "frames": frames}


def score(r: dict) -> dict:
    expected = {ELEMENT_FM[e] for e in r["failed"]}
    al = [f for f in r["frames"] if f["alarmed"]]
    prod = [f["prod"] for f in al]
    ref = [refine(f, MARGIN) for f in al]
    count = lambda names: {k: names.count(k) for k in sorted(set(names))}  # noqa: E731
    return {"alarmed": len(al),
            "prod_right": sum(n in expected for n in prod),
            "ref_right": sum(n in expected for n in ref),
            "lost": sum(a in expected and b not in expected for a, b in zip(prod, ref, strict=True)),
            "gained": sum(b in expected and a not in expected for a, b in zip(prod, ref, strict=True)),
            "prod_names": count(prod), "ref_names": count(ref)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bearings", default=",".join(xl.BEARINGS))
    ap.add_argument("--json", default=str(WORK / "xjtu_result.json"))
    args = ap.parse_args(argv)
    runs = []
    for b in args.bearings.split(","):
        r = run_bearing(b)
        runs.append(r)
        print(f"{b}: replayed {r['snapshots']} snapshots", file=sys.stderr, flush=True)
    for r in runs:
        r["equipment_id"], r["alarm"] = "IMS-RIG", "not_ok"
    production_names(runs)
    print("bearing  failed                 n  learn  FA(learn)  1st alarm  lead   "
          "CRIT  alarmed  prod R/W   +ref R/W  lost gain")
    out = []
    for r in runs:
        s = score(r)
        fmt = lambda v: "-" if v is None else f"{v}"  # noqa: E731
        print(f"{r['bearing']:7}  {'+'.join(r['failed'])[:20]:20} {r['snapshots']:5} "
              f"{r['n_learn']:5}  {r['learn_false_alarms']:3}/{r['learn_warm']:<5} "
              f"{fmt(r['first_alarm_min']):>6} min {fmt(r['lead_min']):>5}  "
              f"{fmt(r['first_critical_min']):>5}  {s['alarmed']:6}  "
              f"{s['prod_right']:4}/{s['alarmed'] - s['prod_right']:<4} "
              f"{s['ref_right']:4}/{s['alarmed'] - s['ref_right']:<4} {s['lost']:4} {s['gained']:4}")
        print(f"         names: prod {s['prod_names']}  +ref {s['ref_names']}")
        out.append({k: v for k, v in r.items() if k != "frames"} | s)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
