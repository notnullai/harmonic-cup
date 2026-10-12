"""Harmonic on a real machine: an IMS bearing test, run to failure.

Builds a brain for the IMS rig and replays a whole test through the engine,
then reports when it raised each state against when the bearing failed.

    python scripts/evaluate_ims.py                          # test 2, bearing 1 (default, unchanged)
    python scripts/evaluate_ims.py --test 1 --channel 4      # test 1, bearing 3, x axis
    python scripts/evaluate_ims.py --test 1 --channel 6      # test 1, bearing 4, x axis
    python scripts/evaluate_ims.py --test 3 --channel 2      # test 3, bearing 3
    python scripts/evaluate_ims.py --json out.json           # and write the timeline

What the brain is made of — and what it is not:

  * The NETWORK is the one trained on the pump simulator
    (`sim/artifacts/trained.harmonic`, from scripts/build_brain.py). Nothing was
    trained on IMS data, on any test. The classes are learned on surprise —
    each feature's distance from the machine's own baseline — which is why
    they can move.
  * The MACHINE is `profiles/ims_rexnord_za2115.profile` for every test: the
    rig is the same Rexnord ZA-2115 at 2000 rpm in all three data sets. Its
    dead zones were measured once, on test 2 bearing 1's commissioning window
    (--measure-floor), and are reused unchanged for every other test and
    bearing — this script does not retune them per run.
  * The BASELINE is commissioned on each test's first COMMISSION_HOURS of
    running, the way a newly installed machine would be. The engine never sees
    a label, and nothing after that window is used to build anything.

Needs the IMS data extracted — see sim/datasets/ims_loader.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_brain import binary  # noqa: E402

ARTIFACTS = REPO / "sim" / "artifacts"
PROFILE = REPO / "profiles" / "ims_rexnord_za2115.profile"
TRAINED = ARTIFACTS / "trained.harmonic"

DEFAULT_TEST = 2
DEFAULT_CHANNEL = 0              # bearing 1, the one that failed
WINDOW = 16384
# The first ~50 hours. A commissioning run is what a new install gets, and it
# has to end before anything could plausibly be wrong; the degradation reported
# in the literature for test 2 bearing 1 starts on day four.
COMMISSION_HOURS = 50.0
STATES = ("OK", "WATCH", "WARNING", "CRITICAL")
# For reference only — the brain has no zones set, so they raise nothing. The
# IMS readme gives no motor power; class I (small machines) has the lowest
# limits of ISO 10816-1, so it is the choice that lets ISO react earliest.
REFERENCE_CLASS = "10816-1-I"


def out_brain_path(test: int, channel: int, out_dir: Path = ARTIFACTS) -> Path:
    if test == DEFAULT_TEST and channel == DEFAULT_CHANNEL:
        return out_dir / "ims_test2_b1.harmonic"   # unchanged name/path
    return out_dir / f"ims_test{test}_ch{channel}.harmonic"


def failure_desc(test: int, channel: int) -> str:
    """The readme's description for this channel, without the redundant
    ', bearing N' suffix (the report prints the bearing number itself)."""
    from sim.datasets.ims_loader import FAILURES

    for c, desc in FAILURES.get(test, []):
        if c == channel:
            return re.sub(r", bearing \d+$", "", desc)
    return "no recorded failure on this channel"


def bearing_label(test: int, channel: int) -> str:
    from sim.datasets.ims_loader import bearing_axis

    bearing, axis = bearing_axis(test, channel)
    return f"bearing {bearing}" if axis is None else f"bearing {bearing} ({axis})"


def build_brain(test: int, channel: int, snaps, data, work: Path,
                profile: Path, commission_hours: float, trained: Path = TRAINED) -> Path:
    from api.machines import net_args
    from sim.datasets.ims_loader import write_frames

    if not trained.exists():
        sys.exit(f"ERROR: {trained} not found. Run: python scripts/build_brain.py")
    out_brain = out_brain_path(test, channel, work)
    t0 = snaps[0].taken
    n_comm = sum((s.taken - t0).total_seconds() / 3600.0 < commission_hours for s in snaps)
    frames = work / f"ims_test{test}_ch{channel}_commission.hframe"
    write_frames(frames, snaps[:n_comm], data[:n_comm])
    base = work / f"ims_test{test}_ch{channel}_base.harmonic"
    for cmd in (
        [binary("mint_brain"), base, *net_args(trained), "--profile", profile,
         "--weights-from", trained],
        [binary("commission"), frames, "--brain", base, "--out", out_brain],
    ):
        r = subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
            sys.exit(f"ERROR: {Path(str(cmd[0])).name} failed")
    print(f"brain: {out_brain.relative_to(REPO)} — commissioned on the first "
          f"{n_comm} snapshots ({commission_hours:.0f} h)", file=sys.stderr)
    return out_brain


def replay(brain: Path, snaps, data, class_names: tuple[str, ...] | None = None) -> list[dict]:
    from api.bridge import CLASS_NAMES as DEFAULT_CLASSES
    from api.rig import HarmonicRig

    CLASS_NAMES = class_names or DEFAULT_CLASSES
    from sim.datasets.ims_loader import RPM, SAMPLE_RATE_HZ, SLOW_PRESENT_NONE

    rig = HarmonicRig(brain, window_samples=WINDOW, sample_rate_hz=SAMPLE_RATE_HZ)
    t0 = snaps[0].taken
    out = []
    prev = t0
    try:
        for snap, samples in zip(snaps, data, strict=True):
            dt_us = max(int((snap.taken - prev).total_seconds() * 1e6), 1)
            prev = snap.taken
            tick = rig.feed(samples, rpm=RPM, load=0.65,
                            timestamp_us=int((snap.taken - t0).total_seconds() * 1e6),
                            dt_us=dt_us, present=SLOW_PRESENT_NONE)
            scores = {CLASS_NAMES[i]: round(s, 4) for i, s in enumerate(tick.class_scores)
                      if i < len(CLASS_NAMES)}
            out.append({
                "snapshot": snap.name,
                "hours": round((snap.taken - t0).total_seconds() / 3600.0, 2),
                "state": tick.state, "warm": tick.warm, "hhi": round(tick.hhi, 4),
                "acc_rms_g": round(float(samples.std()), 4),
                "vel_mm_s": (None if tick.vib_velocity_mm_s is None
                             else round(tick.vib_velocity_mm_s, 3)),
                "scores": scores,
                "top_surprised": tick.top_surprised(),
            })
    finally:
        rig.close()
    return out


def measure_floor(test: int, channel: int, brain: Path, snaps, data,
                  commission_hours: float) -> None:
    """The noise floor, measured the way scripts/noise_floor.py measures the
    pump's: dead zone = 2x the 99.9th percentile of |surprise| over healthy
    running — here the commissioning window only, and nothing after it.

    Prints profile lines to paste into profiles/ims_rexnord_za2115.profile.
    Surprise is taken before the dead zone is applied, so the brain used to
    measure it does not need the right floor already.

    This is a diagnostic: evaluate_ims.py's own replay always uses the
    profile's existing dead zones (measured once, on test 2 bearing 1), the
    same for every test — see the module docstring.
    """
    import numpy as np

    from api.rig import HarmonicRig
    from sim.datasets.ims_loader import RPM, SAMPLE_RATE_HZ, SLOW_PRESENT_NONE

    rig = HarmonicRig(brain, window_samples=WINDOW, sample_rate_hz=SAMPLE_RATE_HZ)
    rig.set_guard(False)       # measuring normal: not judged by it
    t0 = prev = snaps[0].taken
    seen: dict[str, list[float]] = {}
    try:
        for snap, samples in zip(snaps, data, strict=True):
            hours = (snap.taken - t0).total_seconds() / 3600.0
            if hours >= commission_hours:
                break
            tick = rig.feed(samples, rpm=RPM, load=0.65,
                            timestamp_us=int(hours * 3.6e9),
                            dt_us=max(int((snap.taken - prev).total_seconds() * 1e6), 1),
                            present=SLOW_PRESENT_NONE)
            prev = snap.taken
            if tick.warm:
                for kind, s in tick.surprise.items():
                    seen.setdefault(kind, []).append(abs(s))
    finally:
        rig.close()
    print(f"# Measured by scripts/evaluate_ims.py --measure-floor on the first "
          f"{commission_hours:.0f} h of test {test}, {bearing_label(test, channel)}:")
    print("# 2x the 99.9th percentile of healthy |surprise|.")
    for kind, values in seen.items():
        if not kind.startswith("HARMONIC_FEAT_VIB_"):
            continue            # not recorded on this rig
        dz = 2.0 * float(np.percentile(values, 99.9))
        name = kind.removeprefix("HARMONIC_FEAT_")
        print(f"dead_zone.{name} = {dz:.2f}")


def zone(vel: float | None) -> str:
    from kg import standards

    bounds = standards.classes()[REFERENCE_CLASS]["bounds"]
    if vel is None:
        return "-"
    return next((z for z, b in zip("ABC", bounds[:3], strict=True) if vel < b), "D")


def report(timeline: list[dict], test: int, channel: int, commission_hours: float) -> dict:
    end_h = timeline[-1]["hours"]
    label = bearing_label(test, channel)
    desc = failure_desc(test, channel)
    first: dict[str, dict] = {}
    for rec in timeline:
        level = STATES.index(rec["state"]) if rec["state"] in STATES else 0
        for s in STATES[1:level + 1]:
            first.setdefault(s, rec)
    in_comm = [r for r in timeline if r["hours"] < commission_hours and r["warm"]]
    comm_alarms = sum(r["state"] != "OK" for r in in_comm)
    after_comm = [r for r in timeline if r["hours"] >= commission_hours and r["warm"]]
    first_alarm_idx = next((i for i, r in enumerate(after_comm) if r["state"] != "OK"), None)
    alarms_before_first = 0 if first_alarm_idx is None else sum(
        r["state"] != "OK" for r in after_comm[:first_alarm_idx])

    print(f"\nIMS test {test}, {label}: {len(timeline)} snapshots over "
          f"{end_h:.1f} h; {desc} at the end of the test")
    print(f"commissioning window ({commission_hours:.0f} h): {comm_alarms} of "
          f"{len(in_comm)} warm snapshots not OK")
    if after_comm:
        print(f"after commissioning, before the first alarm: {alarms_before_first} "
              f"false alarm(s) of {len(after_comm)} warm snapshots")
    summary: dict = {"test": test, "channel": channel, "bearing": label,
                     "failure": desc, "snapshots": len(timeline), "hours": end_h,
                     "commission_hours": commission_hours,
                     "commission_window_alarms": comm_alarms,
                     "commission_window_warm": len(in_comm),
                     "false_alarms_after_commission": alarms_before_first,
                     "first": {}}
    for s in STATES[1:]:
        rec = first.get(s)
        if rec is None:
            print(f"  {s:<8} never")
            continue
        lead = end_h - rec["hours"]
        top = max((kv for kv in rec["scores"].items() if kv[0] != "healthy"),
                  key=lambda kv: kv[1], default=("-", 0))
        feats = ", ".join(f"{f['kind'].removeprefix('HARMONIC_FEAT_')}"
                          for f in rec["top_surprised"][:3])
        print(f"  {s:<8} at {rec['hours']:6.1f} h ({rec['snapshot']}), "
              f"{lead:5.1f} h before the end; acc {rec['acc_rms_g']:.3f} g, "
              f"velocity {rec['vel_mm_s']} mm/s (zone {zone(rec['vel_mm_s'])} of "
              f"{REFERENCE_CLASS}); "
              f"top class {top[0]} {top[1]:.2f}; surprised: {feats}")
        summary["first"][s] = {"hours": rec["hours"], "snapshot": rec["snapshot"],
                               "hours_before_end": round(lead, 1),
                               "vel_mm_s": rec["vel_mm_s"],
                               "reference_zone": zone(rec["vel_mm_s"]),
                               "top_class": top[0], "surprised": feats}
    peak = max(timeline, key=lambda r: r["vel_mm_s"] or 0)
    print(f"  velocity peaked at {peak['vel_mm_s']} mm/s (zone {zone(peak['vel_mm_s'])}) "
          f"at {peak['hours']:.1f} h; zones are {REFERENCE_CLASS}, for reference")
    summary["peak_velocity"] = {"hours": peak["hours"], "vel_mm_s": peak["vel_mm_s"],
                                "reference_zone": zone(peak["vel_mm_s"])}
    return summary


# The engine's classes are the simulator's five (healthy, bearing_wear,
# cavitation, misalignment, imbalance) — it has no notion of race vs. roller.
# The graph's evidence path (which symptom a surprised feature indicates) is
# what can tell the sub-faults apart; this only says which one the
# composer actually names, honestly, not which one "should" win.
EXPECTED_FAILURE_MODE = (
    ("outer race", "fm.outer_race_wear"),
    ("inner race", "fm.inner_race_wear"),
    ("roller element", "fm.rolling_element_wear"),
)


def diagnose(timeline: list[dict], test: int, channel: int) -> dict:
    """Top-1 diagnosis through the graph, offline (no language model), on
    every alarmed, warm snapshot — the same knowledge graph and the same
    IMS-RIG equipment node (kg/families/pump.json) used by the console."""
    import tempfile

    from kg.load import build_graph
    from kg.query import query_for_event
    from kg.store import NetworkXStore

    desc = failure_desc(test, channel)
    expected = next((fm for key, fm in EXPECTED_FAILURE_MODE if key in desc), None)

    path = Path(tempfile.mkdtemp(prefix="ims-kg-")) / "g.json"
    store = NetworkXStore(path, autoload=False)
    store.merge_graph(build_graph())
    try:
        alarmed = [r for r in timeline if r["warm"] and r["state"] != "OK"]
        counts: dict[str, int] = {}
        matched = 0
        for rec in alarmed:
            event = {
                "equipment_id": "IMS-RIG",
                "top_classes": [{"name": k, "score": v} for k, v in rec["scores"].items()],
                "top_surprised_features": rec["top_surprised"],
            }
            top = query_for_event(event, store).top
            name = top.failure_mode_id if top else "none"
            counts[name] = counts.get(name, 0) + 1
            if expected and name == expected:
                matched += 1
    finally:
        store.close()
    print(f"  diagnosis: expected {expected}, top-1 matched {matched} of "
          f"{len(alarmed)} alarmed snapshots; seen: {counts}")
    return {"expected": expected, "alarmed_snapshots": len(alarmed),
           "top1_match": matched, "top1_counts": counts}


# The sustain rule Harmonic's WATCH state uses before it adopts a new state:
# harmonic_health_config.confirm_ticks, defaulted to 3 consecutive ticks in
# harmonic_health_config_from_brain (engine/src/harmonic_health.cpp) and
# documented in engine/include/harmonic/harmonic_health.h. One IMS snapshot
# is one engine tick in this replay, so the fair comparator needs 3
# consecutive snapshots over threshold, and fires on the 3rd — the same tick
# the engine would adopt the state on (harmonic_health_update's candidate/
# candidate_ticks logic), not the 1st single noisy snapshot.
WATCH_CONFIRM_TICKS = 3


def class_lead(timeline: list[dict]) -> dict:
    """The engine's leading class over the warm, alarmed snapshots: what the
    class neurons themselves name, before the knowledge graph."""
    counts: dict[str, int] = {}
    for rec in timeline:
        if rec["warm"] and rec["state"] != "OK":
            top = max((kv for kv in rec["scores"].items() if kv[0] != "healthy"),
                      key=lambda kv: kv[1], default=("-", 0))
            name = top[0] if top[1] > 0 else "none"
            counts[name] = counts.get(name, 0) + 1
    print(f"  engine lead over alarmed snapshots: {counts}")
    return counts


def rms_baseline(timeline: list[dict], commission_hours: float) -> dict:
    """The same 3sigma-RMS comparison baseline as the existing IMS section:
    a threshold on overall acceleration RMS, set from the commissioning
    window's mean and standard deviation, and where it first fires — plus a
    3sigma variant that only fires once WATCH_CONFIRM_TICKS consecutive
    snapshots are over threshold, the same sustain rule Harmonic's WATCH
    state uses (no-sustain and sustained are reported side by side; neither
    replaces the other)."""
    import numpy as np

    comm = [r["acc_rms_g"] for r in timeline if r["hours"] < commission_hours]
    mean, std = float(np.mean(comm)), float(np.std(comm))
    end_h = timeline[-1]["hours"]
    out = {}
    for name, thr in (("3sigma", mean + 3 * std), ("1.5x_mean", 1.5 * mean)):
        rec = next((r for r in timeline if r["hours"] >= commission_hours
                   and r["acc_rms_g"] > thr), None)
        out[name] = (None if rec is None else
                    {"hours": rec["hours"], "hours_before_end": round(end_h - rec["hours"], 1)})

    thr_3sigma = mean + 3 * std
    run = 0
    rec = None
    for r in timeline:
        if r["hours"] < commission_hours:
            continue
        if r["acc_rms_g"] > thr_3sigma:
            run += 1
            if run >= WATCH_CONFIRM_TICKS:
                rec = r
                break
        else:
            run = 0
    out[f"3sigma_sustain{WATCH_CONFIRM_TICKS}"] = (
        None if rec is None else
        {"hours": rec["hours"], "hours_before_end": round(end_h - rec["hours"], 1)})
    return out


def main(argv: list[str] | None = None) -> int:
    from sim.datasets.ims_loader import load_run

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, default=DEFAULT_TEST,
                    help="IMS data set: 1, 2 or 3 (default 2)")
    ap.add_argument("--channel", type=int, default=DEFAULT_CHANNEL,
                    help="0-based channel, i.e. bearing/axis (default 0)")
    ap.add_argument("--commission-hours", type=float, default=COMMISSION_HOURS,
                    help=f"length of the healthy commissioning window (default {COMMISSION_HOURS:.0f} h)")
    ap.add_argument("--profile", type=Path, default=PROFILE,
                    help="machine profile (default profiles/ims_rexnord_za2115.profile, "
                         "reused unchanged for every test)")
    ap.add_argument("--json", help="write the timeline and summary here")
    ap.add_argument("--no-build", action="store_true", help="reuse the existing brain")
    ap.add_argument("--measure-floor", action="store_true",
                    help="print the dead zones measured on the commissioning window "
                         "(diagnostic only — does not change what --json/report use)")
    ap.add_argument("--rms-baseline", action="store_true",
                    help="also print the 3sigma-RMS / 1.5x-mean comparison baseline")
    ap.add_argument("--diagnose", action="store_true",
                    help="also run the offline graph composer's top-1 diagnosis "
                         "on every alarmed snapshot")
    ap.add_argument("--trained", type=Path, default=TRAINED,
                    help="the trained network to graft on (default the pump's)")
    ap.add_argument("--classes", default=None,
                    help="its class names, comma-separated (default the pump's)")
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS,
                    help="where the IMS brain goes (default sim/artifacts)")
    ap.add_argument("--class-lead", action="store_true",
                    help="also count the engine's leading class over alarmed snapshots")
    args = ap.parse_args(argv)
    classes = tuple(args.classes.split(",")) if args.classes else None

    snaps, data = load_run(args.test, args.channel, WINDOW)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    brain = (out_brain_path(args.test, args.channel, args.out_dir) if args.no_build else
             build_brain(args.test, args.channel, snaps, data, args.out_dir,
                        args.profile, args.commission_hours, args.trained))
    if args.measure_floor:
        measure_floor(args.test, args.channel, brain, snaps, data, args.commission_hours)
        return 0
    timeline = replay(brain, snaps, data, classes)
    summary = report(timeline, args.test, args.channel, args.commission_hours)
    if args.class_lead:
        summary["class_lead"] = class_lead(timeline)
    if args.diagnose:
        summary["diagnosis"] = diagnose(timeline, args.test, args.channel)
    if args.rms_baseline:
        rms = rms_baseline(timeline, args.commission_hours)
        summary["rms_baseline"] = rms
        for name, rec in rms.items():
            if rec is None:
                print(f"  RMS baseline ({name}) never fired")
            else:
                print(f"  RMS baseline ({name}) at {rec['hours']:.1f} h, "
                      f"{rec['hours_before_end']:.1f} h before the end")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "timeline": timeline},
                                              indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
