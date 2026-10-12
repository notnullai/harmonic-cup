"""A machine family's brain on runs it was not trained on (HARM-S7-T05).

Engine level: the commissioned brain through the real compiled engine (the
ctypes rig), on simulator seeds the trainer never saw. For each fault mode it
ramps the fault in against a settled baseline and reports when the engine left
OK, how severe the fault was by then, and which class neuron led; then it runs
healthy machines and counts anything not OK.

The knowledge graph is not in this loop — the class neuron is what is measured
here. The pump's end-to-end diagnosis accuracy, with the graph, is
scripts/evaluate.py.

    python scripts/evaluate_family.py --family gearbox
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_brain import family  # noqa: E402

SEEDS = (9001, 31337, 777)          # none of them used in training (5150, 7001+)
SETTLE, RAMP, HOLD = 120, 240, 60   # frames; the brain warms up in 60
HEALTHY = 600
WINDOW, FS = 16384, 25600.0


def _feed(rig, fr, i, dt_us):
    import numpy as np

    slow = getattr(fr, "slow", None)
    present = getattr(fr, "present", 0)
    if slow is None:                              # the pump's named channels
        slow = [fr.bearing_temp_c, fr.motor_temp_c, fr.discharge_bar, fr.suction_bar,
                fr.flow_m3h, fr.current_a, 0.0]
        present = 0x7F
    return rig.feed(np.asarray(fr.vibration, dtype=np.float32), rpm=fr.rpm, slow=slow,
                    present=present, timestamp_us=i * dt_us, dt_us=dt_us)


def run(fam_name: str, brain: Path) -> dict:
    from api.rig import HarmonicRig

    fam = family(fam_name)
    Sim, classes = fam["sim"], fam["classes"]
    dt = WINDOW / FS
    dt_us = int(dt * 60 * 1e6)                    # machine time runs 60x
    out: dict = {"family": fam_name, "brain": str(brain), "faults": {}, "soak": {}}

    for mode in fam["modes"]:
        runs = []
        for seed in SEEDS:
            s = Sim(seed=seed, fs_vib=FS)
            s.time_scale = 60.0
            rig = HarmonicRig(brain, window_samples=WINDOW, sample_rate_hz=int(FS))
            try:
                i = 0
                for _ in range(SETTLE):
                    _feed(rig, s.step(dt), i, dt_us)
                    i += 1
                s.inject(mode, severity=1.0, onset_s=s.t_s,
                         ramp_hours=RAMP * dt * 60 / 3600.0)
                caught, leads = None, Counter()
                for _ in range(RAMP + HOLD):
                    fr = s.step(dt)
                    tick = _feed(rig, fr, i, dt_us)
                    i += 1
                    if tick.state != "OK":
                        top = max(range(1, len(tick.class_scores)),
                                  key=lambda c: tick.class_scores[c])
                        leads[classes[top]] += 1
                        if caught is None:
                            caught = {"severity": round(fr.severities[mode], 3),
                                      "state": tick.state, "lead": classes[top]}
            finally:
                rig.close()
            runs.append({"seed": seed, "caught": caught,
                         "leading_while_alarmed": leads.most_common(1)[0][0] if leads else None,
                         "right_share": round(leads[mode] / sum(leads.values()), 3)
                         if leads else 0.0})
        out["faults"][mode] = runs

    alarms = warm = 0
    for seed in SEEDS[:2]:
        s = Sim(seed=seed, fs_vib=FS)
        s.time_scale = 60.0
        rig = HarmonicRig(brain, window_samples=WINDOW, sample_rate_hz=int(FS))
        try:
            for i in range(HEALTHY):
                tick = _feed(rig, s.step(dt), i, dt_us)
                if tick.warm:
                    warm += 1
                    alarms += tick.state != "OK"
        finally:
            rig.close()
    out["soak"] = {"warm_frames": warm, "not_ok": alarms}
    return out


def measure_floor(fam_name: str, brain: Path) -> None:
    """Profile lines for the family's noise floor: 2x the 99.9th percentile of
    healthy |surprise| per feature, the rule the pump and the IMS rig use. On
    the commissioning seeds (7001, 7002), never the evaluation ones."""
    import numpy as np

    from api.rig import HarmonicRig

    Sim = family(fam_name)["sim"]
    dt = WINDOW / FS
    dt_us = int(dt * 60 * 1e6)
    seen: dict[str, list[float]] = {}
    for seed in (7001, 7002):
        s = Sim(seed=seed, fs_vib=FS)
        s.time_scale = 60.0
        rig = HarmonicRig(brain, window_samples=WINDOW, sample_rate_hz=int(FS))
        rig.set_guard(False)       # measuring normal: not judged by it
        try:
            for i in range(HEALTHY):
                tick = _feed(rig, s.step(dt), i, dt_us)
                if tick.warm:
                    for name, v in tick.surprise.items():
                        seen.setdefault(name, []).append(abs(v))
        finally:
            rig.close()
    print(f"# Measured by scripts/evaluate_family.py --family {fam_name} --measure-floor:")
    print("# 2x the 99.9th percentile of healthy |surprise|, seeds 7001-7002.")
    for name, vals in seen.items():
        key = name.split(":", 1)[1] if ":" in name else name.removeprefix("HARMONIC_FEAT_")
        print(f"dead_zone.{key} = {max(0.02, 2.0 * float(np.percentile(vals, 99.9))):.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="gearbox")
    ap.add_argument("--brain", default=None, help="default: the family's built brain")
    ap.add_argument("--json", default=None)
    ap.add_argument("--measure-floor", action="store_true",
                    help="print the family's measured noise floor as profile lines")
    args = ap.parse_args(argv)
    brain = Path(args.brain) if args.brain else Path(family(args.family)["out"])
    if not brain.exists():
        sys.exit(f"ERROR: {brain} not built; run python scripts/build_brain.py "
                 f"--family {args.family}")
    if args.measure_floor:
        measure_floor(args.family, brain)
        return 0
    res = run(args.family, brain)
    for mode, runs in res["faults"].items():
        caught = [r["caught"] for r in runs if r["caught"]]
        sev = ", ".join(f"{c['severity']:.0%}" for c in caught) or "never"
        right = sum(r["leading_while_alarmed"] == mode for r in runs)
        share = sum(r["right_share"] for r in runs) / len(runs)
        print(f"{mode:<14} caught {len(caught)}/{len(runs)} at {sev} severity; "
              f"right class leads in {right}/{len(runs)} runs "
              f"({share:.0%} of alarmed frames)")
    soak = res["soak"]
    print(f"healthy soak: {soak['not_ok']} of {soak['warm_frames']} warm frames not OK")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
