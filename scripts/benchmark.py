"""One benchmark runner: one fixed method, every dataset on disk (HARM-S10-T08).

The method is not reimplemented here — it is the existing one in
`scripts/evaluate_ims.py` and `scripts/evaluate_xjtu.py` (commission on the
healthy start of the run, floor measured per sensor over that window, replay
the rest) and, for CWRU and the real gearbox, the same method as run by
`scripts/evaluate_cwru.py` and `scripts/evaluate_gearbox_real.py`. This
script calls those four scripts' own functions and reports their own
numbers, standardised into one table. Nothing is tuned or retuned here.

Datasets, each already on disk (see docs/BENCHMARK_DATASETS.md):
  * IMS, all four recorded failures (test 2 bearing 1, test 1 bearings 3 and
    4, test 3 bearing 3);
  * XJTU-SY, all 15 bearings;
  * CWRU, six seeded faults plus the held-out 3 hp normal record;
  * the real (OpenEI) gearbox, the broken tooth at 10 loads plus 5 held-out
    healthy loads.

Per run this reports: detection and lead, false alarms while learning
normal, the fault named (right or wrong, at whatever granularity the
underlying script already measures it), and a fair sustained-3-sigma-RMS
comparator (the same threshold-plus-sustain rule as
`evaluate_ims.rms_baseline`: threshold = commissioning window's mean + 3
standard deviations of overall RMS, sustained = 3 consecutive readings over
threshold — `evaluate_ims.WATCH_CONFIRM_TICKS`, Harmonic's own WATCH sustain
rule). For IMS that comparator is `evaluate_ims.rms_baseline` itself,
unchanged; for the other three datasets — which do not already compute it —
the same rule is applied here to the RMS series built from each dataset's
own `load`/`windows` functions (data loading only, not the detection
method).

    python scripts/benchmark.py [--json docs/artifacts/benchmark.json] [--md docs/artifacts/benchmark.md]
    python scripts/benchmark.py --datasets ims,xjtu     # skip CWRU / gearbox, e.g. while iterating

Runs IMS and XJTU-SY through the engine (many commission+replay passes) and
can take tens of minutes; CWRU and the real gearbox are fast (small records).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

WORK = REPO / "var" / "benchmark"
DEFAULT_JSON = REPO / "docs" / "artifacts" / "benchmark.json"
DEFAULT_MD = REPO / "docs" / "artifacts" / "benchmark.md"

# Harmonic's own WATCH sustain rule (harmonic_health_config.confirm_ticks,
# defaulted to 3 in harmonic_health_config_from_brain) — the fair comparator
# uses the same number, not a separately chosen one.
CONFIRM_TICKS = 3


def sustained_3sigma(hours: np.ndarray, rms: np.ndarray, n_commission: int,
                     confirm_ticks: int = CONFIRM_TICKS) -> dict:
    """The classical comparator: a threshold on overall RMS set from the
    first `n_commission` readings' mean + 3 standard deviations, reported
    both with no persistence requirement and with `confirm_ticks` consecutive
    readings required (Harmonic's own WATCH sustain rule) — the same
    definition as `evaluate_ims.rms_baseline`, generalised to any RMS series.
    This is a comparator baseline, not part of Harmonic's method; nothing
    here is tuned per dataset."""
    hours = np.asarray(hours, dtype=float)
    rms = np.asarray(rms, dtype=float)
    comm = rms[:n_commission]
    if len(comm) == 0 or n_commission >= len(rms):
        return {"threshold": None, "no_sustain": None, "sustained": None}
    mean, std = float(comm.mean()), float(comm.std())
    thr = mean + 3.0 * std
    end = float(hours[-1])

    def first_over(min_run: int) -> dict | None:
        run = 0
        for i in range(n_commission, len(rms)):
            if rms[i] > thr:
                run += 1
                if run >= min_run:
                    h = float(hours[i])
                    return {"hours": round(h, 4), "lead_hours": round(end - h, 4)}
            else:
                run = 0
        return None

    return {"threshold": round(thr, 6), "no_sustain": first_over(1),
            "sustained": first_over(confirm_ticks)}


# ---------------------------------------------------------------- IMS -----

def run_ims() -> list[dict]:
    from scripts import evaluate_ims as ei
    from sim.datasets.ims_loader import load_run

    WORK_IMS = WORK / "ims"
    WORK_IMS.mkdir(parents=True, exist_ok=True)
    runs = [(2, 0, ei.COMMISSION_HOURS), (1, 4, 250.0), (1, 6, 250.0), (3, 2, ei.COMMISSION_HOURS)]
    out = []
    for test, channel, comm_hours in runs:
        snaps, data = load_run(test, channel, ei.WINDOW)
        brain = ei.build_brain(test, channel, snaps, data, WORK_IMS, ei.PROFILE, comm_hours)
        timeline = ei.replay(brain, snaps, data)
        summary = ei.report(timeline, test, channel, comm_hours)
        diagnosis = ei.diagnose(timeline, test, channel)
        comparator = ei.rms_baseline(timeline, comm_hours)
        detect = summary["first"].get("WATCH")
        out.append({
            "dataset": "IMS", "run": f"test {test}, {summary['bearing']}",
            "failed": summary["failure"],
            "commission": {"false_alarms": summary["commission_window_alarms"],
                           "warm": summary["commission_window_warm"]},
            "detection": (None if detect is None else
                          {"state": "WATCH", "hours": detect["hours"],
                           "lead_hours": detect["hours_before_end"]}),
            "diagnosis": {"expected": diagnosis["expected"], "granularity": "per_alarmed_snapshot",
                         "right": diagnosis["top1_match"],
                         "wrong": diagnosis["alarmed_snapshots"] - diagnosis["top1_match"],
                         "names": diagnosis["top1_counts"]},
            "comparator_3sigma": {
                "threshold": None,
                "no_sustain": (None if comparator["3sigma"] is None else
                              {"hours": comparator["3sigma"]["hours"],
                               "lead_hours": comparator["3sigma"]["hours_before_end"]}),
                "sustained": (None if comparator[f"3sigma_sustain{ei.WATCH_CONFIRM_TICKS}"] is None
                             else {"hours": comparator[f"3sigma_sustain{ei.WATCH_CONFIRM_TICKS}"]["hours"],
                                   "lead_hours": comparator[f"3sigma_sustain{ei.WATCH_CONFIRM_TICKS}"]["hours_before_end"]})},
        })
    return out


# --------------------------------------------------------------- XJTU -----

def _earliest(*hits: dict | None) -> dict | None:
    """The earliest of several first detections (each {hours, lead_hours} or None)."""
    found = [h for h in hits if h]
    return min(found, key=lambda h: h["hours"]) if found else None


def run_xjtu() -> list[dict]:
    """Both of each bearing's accelerometers, each commissioned as its own
    sensor; one row per bearing. A failure is the bearing's, not a sensor's:
    it is caught when either sensor alarms (docs/DETECTION_MISSES.md, "What
    counts as caught"), and every alarm on either sensor's learning window is
    a false alarm. The comparator gets both channels too, scored the same
    way, so it stays a like-for-like comparison."""
    from scripts import evaluate_xjtu as ex
    from sim.datasets import xjtu_loader as xl

    channels = (xl.HORIZONTAL, xl.HORIZONTAL + 1)
    runs = []
    for bearing in xl.BEARINGS:
        for ch in channels:
            r = ex.run_bearing(bearing, ch)
            runs.append(r)
            print(f"  xjtu {bearing} ch{ch}: replayed {r['snapshots']} snapshots",
                  file=sys.stderr, flush=True)
    for r in runs:
        r["equipment_id"], r["alarm"] = "IMS-RIG", "not_ok"
    ex.production_names(runs)

    out = []
    for bearing in xl.BEARINGS:
        per = [r for r in runs if r["bearing"] == bearing]
        scores = [ex.score(r) for r in per]   # evaluate_xjtu's own scoring (production only)
        expected = sorted({ex.ELEMENT_FM[e] for e in per[0]["failed"] if ex.ELEMENT_FM[e]})

        comparators = []
        for r in per:
            snaps, data = xl.load_run(bearing, r["channel"], xl.SNAPSHOT_SAMPLES)
            hours = np.array([snap.hours for snap in snaps])
            rms = np.array([float(x.std()) for x in data])
            comparators.append(sustained_3sigma(hours, rms, r["n_learn"]))
        comparator = {"threshold": None,
                      "no_sustain": _earliest(*(c["no_sustain"] for c in comparators)),
                      "sustained": _earliest(*(c["sustained"] for c in comparators))}

        hits = [None if r["first_alarm_min"] is None else
                {"hours": round(r["first_alarm_min"] / 60.0, 4),
                 "lead_hours": round(r["lead_min"] / 60.0, 4), "channel": r["channel"]}
                for r in per]
        first = _earliest(*hits)
        names: dict[str, int] = {}
        for s in scores:
            for k, v in s["prod_names"].items():
                names[k] = names.get(k, 0) + v
        out.append({
            "dataset": "XJTU-SY", "run": bearing, "failed": "+".join(per[0]["failed"]),
            "commission": {"false_alarms": sum(r["learn_false_alarms"] for r in per),
                           "warm": sum(r["learn_warm"] for r in per)},
            "detection": (None if first is None else
                          {"state": "not_ok", "hours": first["hours"],
                           "lead_hours": first["lead_hours"],
                           "channel": "horizontal" if first["channel"] == xl.HORIZONTAL
                           else "vertical"}),
            "diagnosis": {"expected": expected, "granularity": "per_alarmed_snapshot",
                         "right": sum(s["prod_right"] for s in scores),
                         "wrong": sum(s["alarmed"] - s["prod_right"] for s in scores),
                         "names": names},
            "comparator_3sigma": comparator,
        })
    return out


# --------------------------------------------------------- CWRU / gearbox -

def run_cwru() -> list[dict]:
    from kg.seed import FAMILY_CLASS_TO_FAILURE_MODE
    from scripts import evaluate_cwru as cw

    WORK.mkdir(parents=True, exist_ok=True)
    out_json = WORK / "cwru.json"
    rc = cw.main(["--json", str(out_json)])
    if rc:
        sys.exit("ERROR: evaluate_cwru failed")
    report = json.loads(out_json.read_text())
    expected = FAMILY_CLASS_TO_FAILURE_MODE["motor"]["bearing_wear"]

    base_x = [cw.windows(cw.load(n)[0]) for n in cw.BASELINE]
    base = np.concatenate(base_x)
    n_base = len(base)

    def comparator_for(n: int) -> dict:
        x, _ = cw.load(n)
        data = np.concatenate([base, cw.windows(x)])
        rms = data.std(axis=1)
        hours = np.arange(len(data)) * cw.INTERVAL_S / 3600.0
        return sustained_3sigma(hours, rms, n_base)

    commission = {"false_alarms": report["baseline"]["false_alarms"],
                  "warm": report["baseline"]["checked"]}
    out = []
    h = report["holdout_3hp"]
    out.append({"dataset": "CWRU", "run": "100 (held out)", "failed": "normal, 3 hp (held out)",
               "commission": commission,
               "detection": (None if h["first_alarm"] is None else
                             {"state": "WARNING+", "hours": round(h["first_alarm"] * cw.INTERVAL_S / 3600.0, 4),
                              "lead_hours": round((h["readings"] - h["first_alarm"] - 1) * cw.INTERVAL_S / 3600.0, 4)}),
               "diagnosis": None, "comparator_3sigma": comparator_for(cw.HOLDOUT)})
    for n_str, r in report["faults"].items():
        n = int(n_str)
        out.append({
            "dataset": "CWRU", "run": str(n), "failed": r["fault"],
            "commission": commission,
            "detection": (None if r["first_alarm"] is None else
                          {"state": "WARNING+", "hours": round(r["first_alarm"] * cw.INTERVAL_S / 3600.0, 4),
                           "lead_hours": round((r["readings"] - r["first_alarm"] - 1) * cw.INTERVAL_S / 3600.0, 4)}),
            "diagnosis": (None if r["diagnosis"] is None else
                         {"expected": expected, "granularity": "last_alarmed_reading",
                          "right": int(r["diagnosis"] == expected),
                          "wrong": int(r["diagnosis"] != expected), "names": {r["diagnosis"]: 1}}),
            "comparator_3sigma": comparator_for(n),
        })
    return out


def run_gearbox_real() -> list[dict]:
    from kg.seed import FAMILY_CLASS_TO_FAILURE_MODE
    from scripts import evaluate_gearbox_real as gb

    WORK.mkdir(parents=True, exist_ok=True)
    out_json = WORK / "gearbox_real.json"
    rc = gb.main(["--json", str(out_json)])
    if rc:
        sys.exit("ERROR: evaluate_gearbox_real failed")
    report = json.loads(out_json.read_text())
    expected = FAMILY_CLASS_TO_FAILURE_MODE["gearbox"]["broken_tooth"]

    base = np.concatenate([gb.windows(gb.load("healthy", p)) for p in gb.BASELINE_LOADS])
    n_base = len(base)

    def comparator_for(kind: str, pct: int) -> dict:
        data = np.concatenate([base, gb.windows(gb.load(kind, pct))])
        rms = data.std(axis=1)
        hours = np.arange(len(data)) * gb.INTERVAL_S / 3600.0
        return sustained_3sigma(hours, rms, n_base)

    commission = {"false_alarms": report["baseline"]["false_alarms"],
                  "warm": report["baseline"]["checked"]}
    out = []
    for p_str, r in report["healthy_holdout"].items():
        p = int(p_str)
        out.append({"dataset": "real gearbox", "run": f"healthy {p}% (held out)",
                   "failed": "healthy, held-out load", "commission": commission,
                   "detection": (None if r["first_alarm"] is None else
                                 {"state": "WARNING+", "hours": round(r["first_alarm"] * gb.INTERVAL_S / 3600.0, 4),
                                  "lead_hours": round((r["readings"] - r["first_alarm"] - 1) * gb.INTERVAL_S / 3600.0, 4)}),
                   "diagnosis": None, "comparator_3sigma": comparator_for("healthy", p)})
    for p_str, r in report["broken_tooth"].items():
        p = int(p_str)
        out.append({
            "dataset": "real gearbox", "run": f"broken tooth {p}%", "failed": "broken tooth",
            "commission": commission,
            "detection": (None if r["first_alarm"] is None else
                          {"state": "WARNING+", "hours": round(r["first_alarm"] * gb.INTERVAL_S / 3600.0, 4),
                           "lead_hours": round((r["readings"] - r["first_alarm"] - 1) * gb.INTERVAL_S / 3600.0, 4)}),
            # r["diagnosis"] is None when nothing ever alarmed (never attempted a
            # name) — that miss is already visible via "detection": None, not
            # double-counted as a wrong name here.
            "diagnosis": (None if r["diagnosis"] is None else
                         {"expected": expected, "granularity": "last_alarmed_reading",
                          "right": int(r["diagnosis"] == expected),
                          "wrong": int(r["diagnosis"] != expected), "names": {r["diagnosis"]: 1}}),
            "comparator_3sigma": comparator_for("broken", p),
        })
    return out


# --------------------------------------------------------------- report ---

def to_markdown(runs: list[dict]) -> str:
    lines = ["# Harmonic benchmark: one fixed method, every dataset", "",
            "Commission on the healthy start of the run, floor measured per sensor over",
            "that window, replay the rest — the same method as `scripts/evaluate_ims.py` /",
            "`scripts/evaluate_xjtu.py`, run through `scripts/benchmark.py`. Nothing is",
            "tuned per dataset. Generated by `python scripts/benchmark.py`.", "",
            "XJTU-SY replays both accelerometers, each commissioned as its own sensor: a",
            "bearing is caught when either alarms, and the first is named. Commissioning",
            "false alarms are both sensors' together. The comparator gets both too.", "",
            "| Dataset | Run | Failed | Commissioning FA | Detected (lead) | Named right/wrong | 3-sigma sustained comparator (lead) |",
            "|---|---|---|---|---|---|---|"]
    for r in runs:
        fa = f"{r['commission']['false_alarms']} of {r['commission']['warm']}"
        det = "-" if not r["detection"] else (
            f"{r['detection']['hours']:.1f} h ({r['detection']['lead_hours']:.1f} h lead)"
            + (f", {r['detection']['channel']}" if r["detection"].get("channel") else ""))
        if r["diagnosis"] is None:
            named = "-"
        else:
            named = f"{r['diagnosis']['right']}/{r['diagnosis']['wrong']}"
        cmp_ = r["comparator_3sigma"].get("sustained")
        cmp_s = "never" if not cmp_ else f"{cmp_['hours']:.1f} h ({cmp_['lead_hours']:.1f} h lead)"
        lines.append(f"| {r['dataset']} | {r['run']} | {r['failed']} | {fa} | {det} | {named} | {cmp_s} |")
    lines.append("")
    lines.append("Misses are in the table as measured: a `-` detection means no alarm ever fired; "
                "`0/1` or `0/N` under \"named right/wrong\" means the graph named the wrong failure "
                "mode on every alarmed reading counted there. See `docs/METRICS.md` for the full "
                "discussion per dataset (bearing sub-fault naming, the real-gearbox broken-tooth "
                "miss before the learning guard, etc.).")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--datasets", default="ims,xjtu,cwru,gearbox",
                    help="comma-separated subset to run (default all four)")
    args = ap.parse_args(argv)
    args.json = args.json.resolve()
    args.md = args.md.resolve()
    wanted = set(args.datasets.split(","))

    runs: list[dict] = []
    for key, fn, label in (("ims", run_ims, "IMS"), ("xjtu", run_xjtu, "XJTU-SY"),
                           ("cwru", run_cwru, "CWRU"), ("gearbox", run_gearbox_real, "real gearbox")):
        if key not in wanted:
            continue
        t0 = time.time()
        print(f"=== {label} ===", file=sys.stderr, flush=True)
        runs += fn()
        print(f"=== {label} done in {time.time() - t0:.0f}s ===", file=sys.stderr, flush=True)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"runs": runs}, indent=1) + "\n")
    args.md.write_text(to_markdown(runs))
    print(f"\nwrote {args.json} and {args.md} ({len(runs)} runs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
