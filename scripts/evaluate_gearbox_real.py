"""Real data for the gearbox family: a broken tooth on a test gearbox (HARM-S7-T15).

"Gearbox Fault Diagnosis Data" (OpenEI, submission 623, CC-BY 4.0): a gearbox
test rig recorded healthy and with a broken tooth, at ten loads from 0 to 90%,
four accelerometers, the shaft at 30 Hz according to the file names.

    datasets/openei_gearbox/{Healthy Data,BrokenTooth Data}/*.txt
                              from https://data.openei.org/submissions/623
    python scripts/evaluate_gearbox_real.py [--json docs/artifacts/gearbox_real.json]

What the data allows, and what it does not:
- The sampling rate and the tooth counts are not published. From the healthy
  record: a periodicity of 431.3 samples (autocorrelation) whose 32nd harmonic
  is the strongest line — a 32-tooth gear meshing on a shaft turning once per
  431.3 samples. Taking that shaft as the 30 Hz of the file names gives
  12.94 kHz. The gearbox family's features are orders of the shaft, so the
  rate only has to be consistent; absolute frequencies and the ISO velocity
  zones do not apply (the units are not published either), and none is set.
- The machine is commissioned on the healthy records at the even loads
  (0, 20, 40, 60, 80%); the odd loads are held out as healthy running the
  engine has not seen. Each broken-tooth record is replayed straight after the
  healthy baseline, as if the tooth broke.
- Records are cut into 4096-sample windows with a 1024-sample hop: readings
  overlap, so the noise floor measured on them is optimistic.
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
DATA = REPO / "datasets" / "openei_gearbox"
SHAFT_PERIOD_SAMPLES = 431.3
SHAFT_HZ = 30.0
FS = round(SHAFT_HZ * SHAFT_PERIOD_SAMPLES)      # 12939
TEETH = 32
CHANNEL = 0
WINDOW = 4096
HOP = 1024
INTERVAL_S = 600.0
BASELINE_LOADS = (0, 20, 40, 60, 80)
HOLDOUT_LOADS = (10, 30, 50, 70, 90)


def load(kind: str, pct: int) -> np.ndarray:
    folder, prefix = ("Healthy Data", "h") if kind == "healthy" else ("BrokenTooth Data", "b")
    x = np.loadtxt(DATA / folder / f"{prefix}30hz{pct}.txt")[:, CHANNEL]
    return (x - x.mean()).astype(np.float32)


def windows(x: np.ndarray) -> np.ndarray:
    return np.stack([x[i:i + WINDOW] for i in range(0, len(x) - WINDOW + 1, HOP)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The gearbox family on real broken-tooth data")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if not (DATA / "Healthy Data").is_dir():
        sys.exit(f"ERROR: extract gearboxdata.zip into {DATA} "
                 "(from https://data.openei.org/submissions/623)")

    os.environ["HARMONIC_DATA_DIR"] = tempfile.mkdtemp(prefix="gbx-real-")
    sys.path.insert(0, str(REPO))
    import io

    from api import machines
    from kg import families
    from kg.load import build_graph
    from kg.query import query_for_event
    from kg.seed import FAMILY_CLASS_TO_FAILURE_MODE
    from kg.store import NetworkXStore

    base = np.concatenate([windows(load("healthy", p)) for p in BASELINE_LOADS])
    uid = machines.new_upload({"name": "Gearbox rig, healthy, even loads", "sample_rate_hz": FS,
                               "interval_s": INTERVAL_S, "unit": "g"})
    buf = io.BytesIO()
    np.save(buf, base)
    machines.save_upload_file(uid, "healthy.npy", buf.getvalue())
    m = machines.create({
        "name": "OpenEI gearbox rig", "nominal_rpm": SHAFT_HZ * 60, "vib_unit": "g",
        "iso_class": "none", "bearing": machines.BEARINGS["6207"], "family": "gearbox",
        "params": {"teeth": TEETH}, "components": ["gear", "bearing", "shaft", "coupling", "foundation"],
        "recording": f"upload:{uid}", "baseline_h": (0.0, 1e6), "speed": "constant"})
    m = machines.commission(m.id)
    print(f"commissioned on {len(base)} healthy readings (loads {BASELINE_LOADS}): "
          f"{m.report['false_alarms']} of {m.report['warm_readings']} alarmed", file=sys.stderr)

    classes = families.class_names("gearbox")
    store = NetworkXStore(Path(os.environ["HARMONIC_DATA_DIR"]) / "g.json", autoload=False)
    store.merge_graph(build_graph())
    store.merge_graph(machines.kg_graph(m))
    expected = FAMILY_CLASS_TO_FAILURE_MODE["gearbox"]["broken_tooth"]
    alarm = ("WARNING", "CRITICAL", "FAILED")

    def run(x: np.ndarray) -> dict:
        w = windows(x)
        data = np.concatenate([base, w])
        hours = np.arange(len(data)) * INTERVAL_S / 3600.0
        ticks = machines.replay(m.brain_path, hours, data, FS, SHAFT_HZ * 60)[len(base):]
        alarmed = [t for t in ticks if t.state in alarm]
        lead: dict[str, int] = {}
        for t in alarmed:
            s = t.class_scores[1:]
            if s and max(s) > 0:
                name = classes[1 + int(np.argmax(s))]
                lead[name] = lead.get(name, 0) + 1
        out = {"readings": len(ticks), "alarmed": len(alarmed),
               "first_alarm": next((i for i, t in enumerate(ticks) if t.state in alarm), None),
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

    report: dict = {"sample_rate_hz_inferred": FS, "teeth_inferred": TEETH,
                    "baseline": {"readings": len(base), "false_alarms": m.report["false_alarms"],
                                 "checked": m.report["warm_readings"]},
                    "healthy_holdout": {}, "broken_tooth": {}}
    for p in HOLDOUT_LOADS:
        r = run(load("healthy", p))
        report["healthy_holdout"][str(p)] = r
        print(f"healthy {p:2}% (held out): {r['alarmed']:3}/{r['readings']} alarmed", file=sys.stderr)
    for p in (*BASELINE_LOADS, *HOLDOUT_LOADS):
        r = run(load("broken", p))
        report["broken_tooth"][str(p)] = r
        named = "" if r["diagnosis"] == expected else "  (not the broken tooth)"
        print(f"broken  {p:2}%: {r['alarmed']:3}/{r['readings']} alarmed, first at reading "
              f"{r['first_alarm']}; engine lead {r['engine_lead']}; graph: {r['diagnosis']}{named}; "
              f"features {r['features']}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
