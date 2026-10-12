"""Real data for the motor family: the CWRU bearing rig (HARM-S7-T15).

Case Western Reserve University's Bearing Data Center recorded a 2 hp induction
motor with seeded bearing faults at its drive end — an SKF 6205, the bearing of
the motor family's reference machine. This takes the motor family through the
path a user would: commission a machine on the rig's normal running, then watch
recordings it never saw.

    datasets/cwru/<n>.mat     from https://engineering.case.edu/bearingdatacenter
                              (97-100 normal at 0-3 hp; 105, 118, 130, 209, 222,
                              234 drive-end faults at 0 hp, 12 kHz)
    python scripts/evaluate_cwru.py [--json docs/artifacts/cwru.json]

What the data allows, and what it does not:
- Each record is 5-10 seconds. Commissioning needs at least 160 readings, so
  the records are cut into 4096-sample windows (0.34 s at 12 kHz) with a
  1024-sample hop: neighbouring readings overlap by three quarters. The
  noise floor measured on them is therefore optimistic.
- The normal records are sampled at 48 kHz and are decimated to 12 kHz; the
  fault records are 12 kHz as published.
- The commissioning baseline is the normal running at 0, 1 and 2 hp. The 3 hp
  record is held out: the healthy test the engine has not seen.
- Each fault record is replayed straight after the healthy baseline, as if the
  fault appeared. The motor family has one bearing class, and its features
  follow the outer race (BPFO) only: an inner-race or ball fault can be
  detected as bearing wear, not named by race.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "cwru"
FS = 12000
WINDOW = 4096
HOP = 1024
INTERVAL_S = 600.0
# The motor's speed per record, where the file does not carry it (CWRU tables).
NORMAL_RPM = {97: 1797.0, 98: 1772.0, 99: 1750.0, 100: 1730.0}
BASELINE = (97, 98, 99)
HOLDOUT = 100
FAULTS = {105: "inner race, 0.007 in", 118: "ball, 0.007 in", 130: "outer race @6, 0.007 in",
          209: "inner race, 0.021 in", 222: "ball, 0.021 in", 234: "outer race @6, 0.021 in"}


def load(n: int) -> tuple[np.ndarray, float]:
    from scipy.io import loadmat
    from scipy.signal import decimate

    d = loadmat(DATA / f"{n}.mat")
    x = np.asarray(d[f"X{n:03d}_DE_time"], dtype=np.float64).ravel()
    rpm_key = f"X{n:03d}RPM"
    rpm = float(np.asarray(d[rpm_key]).ravel()[0]) if rpm_key in d else NORMAL_RPM[n]
    if n in NORMAL_RPM:
        x = decimate(x, 4, ftype="fir", zero_phase=True)     # 48 kHz -> 12 kHz
    return x.astype(np.float32), rpm


def windows(x: np.ndarray) -> np.ndarray:
    return np.stack([x[i:i + WINDOW] for i in range(0, len(x) - WINDOW + 1, HOP)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The motor family on the CWRU bearing data")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    missing = [n for n in (*BASELINE, HOLDOUT, *FAULTS) if not (DATA / f"{n}.mat").exists()]
    if missing:
        sys.exit(f"ERROR: {DATA} lacks {missing}; download them from the CWRU Bearing Data Center")

    os.environ["HARMONIC_DATA_DIR"] = tempfile.mkdtemp(prefix="cwru-")
    sys.path.insert(0, str(REPO))
    import io

    from api import machines
    from kg import families
    from kg.load import build_graph
    from kg.query import query_for_event
    from kg.seed import FAMILY_CLASS_TO_FAILURE_MODE
    from kg.store import NetworkXStore

    base_x, base_rpm = [], []
    for n in BASELINE:
        x, rpm = load(n)
        w = windows(x)
        base_x.append(w)
        base_rpm += [rpm] * len(w)
    base = np.concatenate(base_x)

    uid = machines.new_upload({"name": "CWRU normal, 0-2 hp", "sample_rate_hz": FS,
                               "interval_s": INTERVAL_S, "unit": "g", "rpm": base_rpm})
    buf = io.BytesIO()
    np.save(buf, base)
    machines.save_upload_file(uid, "normal.npy", buf.getvalue())
    m = machines.create({
        "name": "CWRU motor", "nominal_rpm": 1797, "vib_unit": "g", "iso_class": "none",
        "bearing": machines.BEARINGS["6205"], "family": "motor",
        "params": {"poles": 4, "line_hz": 60},
        "components": ["bearing", "rotor", "stator", "shaft", "foundation"],
        "recording": f"upload:{uid}", "baseline_h": (0.0, 1e6), "speed": "measured"})
    m = machines.commission(m.id)
    print(f"commissioned on {len(base)} normal readings (0-2 hp): "
          f"{m.report['false_alarms']} of {m.report['warm_readings']} alarmed", file=sys.stderr)

    classes = families.class_names("motor")
    store = NetworkXStore(Path(os.environ["HARMONIC_DATA_DIR"]) / "g.json", autoload=False)
    store.merge_graph(build_graph())
    store.merge_graph(machines.kg_graph(m))
    expected = FAMILY_CLASS_TO_FAILURE_MODE["motor"]["bearing_wear"]

    def run(x: np.ndarray, rpm: float) -> dict:
        w = windows(x)
        data = np.concatenate([base, w])
        hours = np.arange(len(data)) * INTERVAL_S / 3600.0
        rpms = np.array(base_rpm + [rpm] * len(w))
        ticks = machines.replay(m.brain_path, hours, data, FS, rpms)[len(base):]
        alarmed = [t for t in ticks if t.state in ("WARNING", "CRITICAL", "FAILED")]
        lead: dict[str, int] = {}
        for t in alarmed:
            s = t.class_scores[1:]
            if s and max(s) > 0:
                name = classes[1 + int(np.argmax(s))]
                lead[name] = lead.get(name, 0) + 1
        out = {"readings": len(ticks), "alarmed": len(alarmed),
               "first_alarm": next((i for i, t in enumerate(ticks)
                                    if t.state in ("WARNING", "CRITICAL", "FAILED")), None),
               "engine_lead": lead, "diagnosis": None, "features": [],
               # Alarmed on the unrecognised-change path: no class, no ISO zone.
               "unrecognised": sum(1 for t in alarmed if t.unrecognised)}
        if alarmed:
            t = alarmed[-1]
            s = t.class_scores[1:]
            top = [{"name": classes[1 + i], "score": float(v)} for i, v in
                   sorted(enumerate(s), key=lambda kv: -kv[1]) if v > 0][:3]
            ev = {"equipment_id": m.equipment_id, "state": t.state, "hhi": t.hhi,
                  "rul_hours": t.rul_hours, "top_classes": top,
                  "top_surprised_features": t.top_surprised()}
            kg = query_for_event(ev, store)
            out["diagnosis"] = kg.candidates[0].failure_mode_id if kg.candidates else None
            out["features"] = [f["kind"] for f in t.top_surprised()]
        return out

    report: dict = {"baseline": {"readings": len(base), "false_alarms": m.report["false_alarms"],
                                 "checked": m.report["warm_readings"]}}
    x, rpm = load(HOLDOUT)
    report["holdout_3hp"] = run(x, rpm)
    h = report["holdout_3hp"]
    print(f"held-out normal, 3 hp: {h['alarmed']} of {h['readings']} readings alarmed",
          file=sys.stderr)
    report["faults"] = {}
    for n, what in FAULTS.items():
        x, rpm = load(n)
        r = run(x, rpm)
        r["fault"] = what
        report["faults"][str(n)] = r
        named = r["diagnosis"] == expected
        print(f"{n} {what:24} alarmed {r['alarmed']:3}/{r['readings']} "
              f"first at reading {r['first_alarm']}; engine lead {r['engine_lead']}; unrecognised {r['unrecognised']}; "
              f"graph: {r['diagnosis']}{'' if named else '  (not the bearing)'}; "
              f"features {r['features']}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
