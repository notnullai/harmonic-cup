"""Final validation: diagnosis accuracy and a healthy soak (HARM-S6-T01, T02).

Two runs, both against the real compiled engine through the ctypes bridge and
the real commissioned brain — not the trainer's own view of its weights, which
is what the Phase 3 class-separation table measures and is a separability
result rather than a capability.

**Diagnosis accuracy** walks all twelve fault datasets — four modes at three
severities — and asks, for every frame the engine flags, which failure mode the
knowledge graph ranks first. That is the number the product makes: the class
neuron says WHAT and the graph says what it MEANS, and the join between them
is where a wrong answer would appear.

**The soak** runs healthy data across every duty regime the machine has and
counts WARNING-or-worse. The multi-regime part only became meaningful in Phase
5: before the regime fix a fixed-speed pump occupied one bucket whatever its
duty, so a "multi-regime soak" exercised one regime three times.

    python scripts/evaluate.py --all
    python scripts/evaluate.py --accuracy --out docs/artifacts/accuracy.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Four modes at three severities, per the tracker's "all 12 fault datasets".
SEVERITIES = (0.35, 0.7, 1.0)

# Frames of healthy running before the fault, so the per-regime predictors have
# a baseline. Below the brain's warm-up the surprise path is meaningless.
SETTLE_FRAMES = 80

# Frames of fault. One frame is a machine minute at the default time scale.
FAULT_FRAMES = 60

# The soak: 30 minutes of engine time at the 1 Hz tick, split across the duty
# regimes the machine actually runs in.
SOAK_MINUTES = 30
SOAK_LOADS = (0.65, 0.95, 0.30)        # NOMINAL, HIGH_LOAD, LOW_LOAD

TARGET_TOP1_ACCURACY = 0.80
TARGET_FALSE_POSITIVES = 0


def _engine_or_exit():
    from api.bridge import DEFAULT_BRAIN
    from api.rig import find_library

    if find_library() is None:
        sys.exit("ERROR: libharmonic_rig not built. Run: cmake --build build -j")
    if not Path(DEFAULT_BRAIN).exists():
        sys.exit(f"ERROR: {DEFAULT_BRAIN} not built. Run: python scripts/build_brain.py")
    return DEFAULT_BRAIN


# --- diagnosis accuracy -------------------------------------------------------


def run_accuracy(store: Any, verbose: bool = True) -> dict[str, Any]:
    from api.bridge import EngineBridge
    from kg.query import query_for_event
    from kg.seed import CLASS_TO_FAILURE_MODE
    from sim.pump_sim import FAULT_MODES

    brain = _engine_or_exit()
    rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()

    for mode in FAULT_MODES:
        expected = CLASS_TO_FAILURE_MODE[mode]
        for severity in SEVERITIES:
            # One seed per MODE, not per severity: varying both means a
            # difference between two rows could be either, and the first
            # run of this produced a non-monotonic seal-leak column that
            # looked like a detection bug and was a harness bug.
            bridge = EngineBridge(brain, seed=9100 + FAULT_MODES.index(mode),
                                  event_cooldown_sec=0)
            try:
                for _ in range(SETTLE_FRAMES):
                    bridge.step()
                bridge.inject(mode, severity=severity, ramp_minutes=10.0)
                flagged = 0
                correct = 0
                predictions: Counter[str] = Counter()
                for _ in range(FAULT_FRAMES):
                    snap = bridge.step()
                    if snap.state not in ("WARNING", "CRITICAL", "FAILED"):
                        continue
                    flagged += 1
                    result = query_for_event({
                        "regime": snap.regime,
                        "hhi": snap.hhi,
                        "state": snap.state,
                        "top_classes": snap.top_classes,
                        "top_surprised_features": snap.top_surprised_features,
                    }, store)
                    top = result.top.failure_mode_id if result.top else "none"
                    predictions[top] += 1
                    confusion[(expected, top)] += 1
                    correct += top == expected
            finally:
                bridge.stop()

            row = {
                "mode": mode,
                "severity": severity,
                "expected": expected,
                "frames_flagged": flagged,
                "correct": correct,
                "accuracy": round(correct / flagged, 4) if flagged else None,
                "predictions": dict(predictions),
            }
            rows.append(row)
            if verbose:
                acc = "—" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
                print(f"  {mode:<14} sev {severity:<4} flagged {flagged:>3}  "
                      f"top-1 {acc}", file=sys.stderr)

    total_flagged = sum(r["frames_flagged"] for r in rows)
    total_correct = sum(r["correct"] for r in rows)
    overall = round(total_correct / total_flagged, 4) if total_flagged else 0.0

    # Per-mode precision and recall over the flagged frames. Recall is over the
    # frames of THIS mode that were flagged; precision is over the frames
    # predicted as this mode, whatever their true mode.
    per_mode: dict[str, dict[str, float | int]] = {}
    modes = sorted({CLASS_TO_FAILURE_MODE[m] for m in FAULT_MODES})
    for fm in modes:
        tp = confusion[(fm, fm)]
        fp = sum(n for (true, pred), n in confusion.items() if pred == fm and true != fm)
        fn = sum(n for (true, pred), n in confusion.items() if true == fm and pred != fm)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_mode[fm] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(precision, 4),
                        "recall": round(recall, 4), "f1": round(f1, 4)}

    return {
        "datasets": len(rows),
        "frames_flagged": total_flagged,
        "top1_accuracy": overall,
        "target": TARGET_TOP1_ACCURACY,
        "pass": overall >= TARGET_TOP1_ACCURACY,
        "per_dataset": rows,
        "per_mode": per_mode,
        "confusion": {f"{true}->{pred}": n for (true, pred), n in sorted(confusion.items())},
    }


# --- healthy soak -------------------------------------------------------------


def run_soak(verbose: bool = True) -> dict[str, Any]:
    from api.bridge import EngineBridge

    brain = _engine_or_exit()
    per_load = max(1, (SOAK_MINUTES * 60) // len(SOAK_LOADS))
    states: Counter[str] = Counter()
    by_regime: Counter[str] = Counter()
    warned = 0
    watched = 0
    worst_vib = 0.0
    ticks = 0
    warm_ticks = 0

    for i, load in enumerate(SOAK_LOADS):
        bridge = EngineBridge(brain, load=load, seed=7300 + i, event_cooldown_sec=0)
        try:
            for _ in range(per_load):
                snap = bridge.step()
                ticks += 1
                states[snap.state] += 1
                by_regime[snap.regime] += 1
                worst_vib = max(worst_vib, snap.vib_rms_mm_s)
                if not snap.warm:
                    continue
                warm_ticks += 1
                if snap.state in ("WARNING", "CRITICAL", "FAILED"):
                    warned += 1
                elif snap.state == "WATCH":
                    watched += 1
            events = len(bridge.events(200))
        finally:
            bridge.stop()
        if verbose:
            print(f"  load {load:.2f}: {per_load} ticks, regime "
                  f"{snap.regime}, {events} events", file=sys.stderr)

    return {
        "ticks": ticks,
        "warm_ticks": warm_ticks,
        "minutes": round(ticks / 60.0, 1),
        "regimes": dict(by_regime),
        "states": dict(states),
        "warning_or_above": warned,
        "watch": watched,
        "worst_vibration_mm_s": round(worst_vib, 4),
        "target": TARGET_FALSE_POSITIVES,
        "pass": warned == TARGET_FALSE_POSITIVES,
    }


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Final validation runs")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--soak", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backend", choices=["auto", "neo4j", "networkx"], default="networkx")
    ap.add_argument("--out", default="docs/artifacts/validation.json")
    args = ap.parse_args(argv)
    if args.all or not (args.accuracy or args.soak):
        args.accuracy = args.soak = True

    payload: dict[str, Any] = {"measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    failures = 0

    if args.accuracy:
        from kg.store import open_store
        store = open_store(args.backend)
        print("diagnosis accuracy over 12 fault datasets", file=sys.stderr)
        try:
            payload["accuracy"] = run_accuracy(store)
        finally:
            store.close()
        a = payload["accuracy"]
        print(f"  top-1 {a['top1_accuracy']:.3f} over {a['frames_flagged']} flagged "
              f"frames (target {a['target']:.2f})  "
              f"{'PASS' if a['pass'] else 'FAIL'}\n", file=sys.stderr)
        failures += not a["pass"]

    if args.soak:
        print(f"healthy soak, {SOAK_MINUTES} min across "
              f"{len(SOAK_LOADS)} duty regimes", file=sys.stderr)
        payload["soak"] = run_soak()
        s = payload["soak"]
        print(f"  {s['warning_or_above']} WARNING+ and {s['watch']} WATCH over "
              f"{s['warm_ticks']} warm ticks ({s['minutes']} min), regimes "
              f"{sorted(s['regimes'])}  {'PASS' if s['pass'] else 'FAIL'}",
              file=sys.stderr)
        failures += not s["pass"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
