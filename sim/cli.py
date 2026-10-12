"""Command line for the pump simulator.

    python -m sim.cli scenario run_to_failure --out out.csv
    python -m sim.cli scenario run_to_failure --realtime
    python -m sim.cli dataset baseline --hours 24 --out sim/artifacts/baseline_normal.npz
    python -m sim.cli dataset faults --out-dir sim/artifacts/faults
    python -m sim.cli inspect run_to_failure

CSV carries the slow channels and a per-frame feature summary — enough to plot
a health trend — while NPZ carries the raw vibration the engine actually needs.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from sim import analysis as A
from sim.pump_sim import (
    FAULT_MODES,
    Frame,
    PumpSimulator,
    bearing_frequencies,
    mode_seed,
)
from sim.scenario import SCENARIO_DIR, Scenario

# Class vocabulary for training. Index 0 is "healthy" and must stay index 0:
# the health layer gives class 0 zero weight, so a fault in that slot would be
# invisible to the index.
CLASS_NAMES = ("healthy", *FAULT_MODES)

# Frame file format, mirroring engine/include/harmonic/harmonic_frames.h.
FRAME_MAGIC = b"HFRM"
FRAME_VERSION = 1
FRAME_HEADER = "<4sIIIIIII"          # magic, version, n_frames, n_samples, rate, classes, 2 reserved
FRAME_RECORD = "<QiiI7fI"            # ts, rpm_q16, load_q16, label_mask, 7 slow, present
SLOW_PRESENT_ALL = 0x7F

CSV_FIELDS = [
    "wall_s",
    "machine_s",
    "machine_h",
    "rpm",
    "vib_rms",
    "vib_peak",
    "vib_kurtosis",
    "vib_crest",
    "vib_1x",
    "vib_2x",
    "vib_3x",
    "env_bpfo",
    "env_bpfi",
    "env_bsf",
    "bearing_temp_c",
    "motor_temp_c",
    "discharge_bar",
    "suction_bar",
    "flow_m3h",
    "current_a",
    "label",
    "labels",
    *[f"sev_{m}" for m in FAULT_MODES],
]


def summarise(frame: Frame, fs_vib: float) -> dict[str, Any]:
    """Per-frame feature summary — the reference the C extractor is checked against."""
    vib = frame.vibration
    freqs, amp = A.spectrum(vib, fs_vib)
    efreqs, eamp = A.envelope_spectrum(vib, fs_vib)
    bf = bearing_frequencies(frame.rpm)
    f1 = frame.rpm / 60.0
    row: dict[str, Any] = {
        "wall_s": round(frame.wall_s, 3),
        "machine_s": round(frame.t_s, 3),
        "machine_h": round(frame.t_s / 3600.0, 4),
        "rpm": round(frame.rpm, 2),
        "vib_rms": round(A.rms(vib), 5),
        "vib_peak": round(A.peak(vib), 5),
        "vib_kurtosis": round(A.kurtosis(vib), 4),
        "vib_crest": round(A.crest_factor(vib), 4),
        "vib_1x": round(A.amplitude_at(freqs, amp, f1), 5),
        "vib_2x": round(A.amplitude_at(freqs, amp, 2 * f1), 5),
        "vib_3x": round(A.amplitude_at(freqs, amp, 3 * f1), 5),
        "env_bpfo": round(A.amplitude_at(efreqs, eamp, bf["bpfo"], 3.0), 5),
        "env_bpfi": round(A.amplitude_at(efreqs, eamp, bf["bpfi"], 3.0), 5),
        "env_bsf": round(A.amplitude_at(efreqs, eamp, bf["bsf"], 3.0), 5),
        "bearing_temp_c": round(frame.bearing_temp_c, 3),
        "motor_temp_c": round(frame.motor_temp_c, 3),
        "discharge_bar": round(frame.discharge_bar, 4),
        "suction_bar": round(frame.suction_bar, 4),
        "flow_m3h": round(frame.flow_m3h, 3),
        "current_a": round(frame.current_a, 3),
        "label": frame.label,
        "labels": "|".join(frame.labels),
    }
    for m in FAULT_MODES:
        row[f"sev_{m}"] = round(frame.severities.get(m, 0.0), 4)
    return row


# --- commands ---------------------------------------------------------------


def cmd_scenario(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    sim = scenario.build()
    n_steps = round(scenario.duration_s / scenario.dt)

    print(
        f"{scenario.name}: {n_steps} frames, {scenario.duration_s:.0f} s of signal, "
        f"{scenario.machine_hours:.2f} h of machine time "
        f"(time_scale {scenario.time_scale:g}x, seed {scenario.seed})",
        file=sys.stderr,
    )

    rows: list[dict[str, Any]] = []
    raw_vib: list[np.ndarray] = []
    t_start = time.monotonic()

    for i in range(n_steps):
        frame = sim.step(scenario.dt)
        rows.append(summarise(frame, scenario.fs_vib))
        if args.raw:
            raw_vib.append(frame.vibration.astype(np.float32))
        if args.realtime:
            target = t_start + (i + 1) * scenario.dt
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if args.progress and (i + 1) % max(1, n_steps // 10) == 0:
            r = rows[-1]
            print(
                f"  {r['wall_s']:>6}s  rms={r['vib_rms']:>7}  kurt={r['vib_kurtosis']:>7}"
                f"  bpfo={r['env_bpfo']:>7}  {r['label']}",
                file=sys.stderr,
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows)", file=sys.stderr)

    if args.raw:
        raw_path = Path(args.raw)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            vibration=np.stack(raw_vib),
            fs_vib=scenario.fs_vib,
            labels=np.array([r["label"] for r in rows]),
        )
        print(f"wrote {raw_path} ({len(raw_vib)} windows)", file=sys.stderr)

    return 0


def _generate(
    sim: PumpSimulator, n_steps: int, dt: float, fs_vib: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vib, rows = [], []
    for _ in range(n_steps):
        frame = sim.step(dt)
        vib.append(frame.vibration.astype(np.float32))
        rows.append(summarise(frame, fs_vib))
    return np.stack(vib), rows


def cmd_dataset(args: argparse.Namespace) -> int:
    if args.kind == "baseline":
        return _dataset_baseline(args)
    return _dataset_faults(args)


def _dataset_baseline(args: argparse.Namespace) -> int:
    """Healthy data across regimes, for commissioning the encoder baseline."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Split the budget between NOMINAL and HIGH_LOAD, so the per-regime
    # predictors each have a baseline to learn. Frames are subsampled: a
    # commissioning run needs coverage across conditions, not every second.
    per_regime_steps = max(1, int(args.hours * 3600 / args.stride / 2))
    blocks = {}
    all_rows: list[dict[str, Any]] = []
    for regime, load in (("nominal", 0.65), ("high_load", 0.95)):
        sim = PumpSimulator(seed=mode_seed(args.seed, "", regime), load=load)
        sim.time_scale = args.stride  # each window stands for `stride` seconds
        vib, rows = _generate(sim, per_regime_steps, 1.0, sim.fs_vib)
        blocks[f"vib_{regime}"] = vib
        for r in rows:
            r["regime"] = regime
        all_rows.extend(rows)
        print(f"  {regime}: {len(vib)} windows", file=sys.stderr)

    payload: dict[str, Any] = {
        "fs_vib": 25600.0,
        "hours": args.hours,
        "stride_s": args.stride,
        "seed": args.seed,
        "summary": json.dumps(all_rows),
        **blocks,
    }
    np.savez_compressed(out, **payload)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB, {len(all_rows)} windows)", file=sys.stderr)
    return 0


def _dataset_faults(args: argparse.Namespace) -> int:
    """One run per fault mode per severity, with ground-truth labels."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    severities = {"low": 0.3, "med": 0.6, "high": 1.0}

    for mode in FAULT_MODES:
        for level, sev in severities.items():
            sim = PumpSimulator(seed=mode_seed(args.seed, mode, level))
            sim.inject(mode, severity=sev)
            n = args.windows
            vib, rows = _generate(sim, n, 1.0, sim.fs_vib)
            path = out_dir / f"{mode}_{level}.npz"
            np.savez_compressed(
                path,
                vibration=vib,
                fs_vib=sim.fs_vib,
                mode=mode,
                severity=sev,
                label=mode,
                summary=json.dumps(rows),
            )
            manifest.append(
                {
                    "file": path.name,
                    "mode": mode,
                    "level": level,
                    "severity": sev,
                    "windows": n,
                    "bytes": path.stat().st_size,
                    "mean_rms": round(
                        float(np.mean([float(r["vib_rms"]) for r in rows])), 4
                    ),
                    "mean_kurtosis": round(
                        float(np.mean([float(r["vib_kurtosis"]) for r in rows])), 4
                    ),
                }
            )
            print(f"  {path.name}: {n} windows", file=sys.stderr)

    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {mpath} ({len(manifest)} runs)", file=sys.stderr)
    return 0


def label_mask(frame: Frame) -> int:
    """Bitmask of the classes active in this frame.

    A SET, not an index: fault modes co-occur and the class neurons are
    independent, so a frame with both misalignment and bearing wear must be
    able to say so.
    """
    mask = 0
    for m in frame.labels:
        mask |= 1 << CLASS_NAMES.index(m)
    return mask if mask else 1 << 0      # nothing active means healthy


def write_frames(path: Path, frames: list[Frame], sample_rate: float) -> int:
    """Write raw frames for the C tooling to encode.

    The words the trainer learns from have to come from the real C extractor
    and encoder, or the trainer learns one encoding and the runtime sees
    another. So Python hands over raw samples and stays out of it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(frames[0].vibration)
    if n_samples & (n_samples - 1):
        raise ValueError(f"window must be a power of two, got {n_samples}")

    with path.open("wb") as fh:
        fh.write(
            struct.pack(
                FRAME_HEADER, FRAME_MAGIC, FRAME_VERSION, len(frames),
                n_samples, int(sample_rate), len(CLASS_NAMES), 0, 0,
            )
        )
        for fr in frames:
            if len(fr.vibration) != n_samples:
                raise ValueError("frames must all be the same length")
            fh.write(
                struct.pack(
                    FRAME_RECORD,
                    int(fr.t_s * 1e6),
                    int(fr.rpm * 65536),
                    int(0.6 * 65536),
                    label_mask(fr),
                    fr.bearing_temp_c, fr.motor_temp_c,
                    fr.discharge_bar, fr.suction_bar,
                    fr.flow_m3h, fr.current_a, 0.0,
                    SLOW_PRESENT_ALL,
                )
            )
            fh.write(fr.vibration.astype(np.float32).tobytes())
    return len(frames)


def write_family_frames(path: Path, frames: list[Any], sample_rate: float,
                        class_names: tuple[str, ...]) -> int:
    """`write_frames` for any machine family. A frame carries `t_s`, `rpm`,
    `vibration`, `slow` (seven values in frame-slot order), `present` and
    `labels`; the label mask indexes `class_names`, healthy first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(frames[0].vibration)
    if n_samples & (n_samples - 1):
        raise ValueError(f"window must be a power of two, got {n_samples}")
    with path.open("wb") as fh:
        fh.write(struct.pack(FRAME_HEADER, FRAME_MAGIC, FRAME_VERSION, len(frames),
                             n_samples, int(sample_rate), len(class_names), 0, 0))
        for fr in frames:
            if len(fr.vibration) != n_samples:
                raise ValueError("frames must all be the same length")
            mask = 0
            for m in fr.labels:
                mask |= 1 << class_names.index(m)
            fh.write(struct.pack(FRAME_RECORD, int(fr.t_s * 1e6), int(fr.rpm * 65536),
                                 int(0.6 * 65536), mask or 1, *fr.slow[:7], fr.present))
            fh.write(fr.vibration.astype(np.float32).tobytes())
    return len(frames)


def cmd_frames(args: argparse.Namespace) -> int:
    """Export training frames: healthy plus every fault at every severity."""
    out = Path(args.out)
    window = args.window
    dt = window / 25600.0

    frames: list[Frame] = []
    counts: dict[str, int] = {}

    # Healthy frames first, and plenty of them: the trainer needs a solid
    # picture of normal or every fault class learns "not quite normal".
    sim = PumpSimulator(seed=args.seed, fs_vib=25600.0)
    for _ in range(args.healthy):
        frames.append(sim.step(dt))
    counts["healthy"] = args.healthy

    for mode in FAULT_MODES:
        for level, sev in (("low", 0.35), ("med", 0.7), ("high", 1.0)):
            sim = PumpSimulator(
                seed=mode_seed(args.seed, mode, level), fs_vib=25600.0
            )
            sim.inject(mode, severity=sev)
            # Discard the first few: the fault has not taken effect yet and
            # the frames would be labelled as a fault while looking healthy.
            for _ in range(3):
                sim.step(dt)
            for _ in range(args.per_fault):
                frames.append(sim.step(dt))
            counts[mode] = counts.get(mode, 0) + args.per_fault

    n = write_frames(out, frames, 25600.0)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({n} frames, {window} samples each, {size_mb:.1f} MB)",
          file=sys.stderr)
    for name, count in counts.items():
        print(f"  {name}: {count}", file=sys.stderr)
    return 0


def cmd_latency_frames(args: argparse.Namespace) -> int:
    """Frames for an honest detection-latency measurement.

    Each fault gets a healthy settling period and then a RAMPED onset, so the
    measurement is "how long until the engine notices a fault appearing"
    rather than "how long until it notices a fault that was already there".
    A step change measures almost nothing: the fault is fully present in the
    first frame, so any detector that looks at one frame scores zero.
    """
    out = Path(args.out)
    dt = args.window / 25600.0
    frames: list[Frame] = []

    # How much MACHINE time one frame covers. This is the number that has to be
    # right for the severity column of the benchmark to mean anything, and it
    # is not `time_scale`: a frame carries `window / fs` seconds of signal, and
    # the simulator's clock advances that multiplied by time_scale. At a 4096
    # sample window and 60x compression it is 9.6 s, not 60 s. Reading it as 60
    # made every ramp 6.25x longer in frames than intended, so a 150-frame
    # onset window reached 16% severity rather than 100% and every "severity at
    # detection" figure was scaled against a ramp that had not happened.
    frame_machine_s = dt * args.time_scale
    # The ramp spans the onset window exactly, so severity at frame k is k/onset
    # by construction rather than by assumption. An explicit --ramp-minutes
    # overrides it for anyone who wants a ramp deliberately longer or shorter
    # than the window they are watching.
    ramp_minutes = (args.ramp_minutes if args.ramp_minutes is not None
                    else args.onset * frame_machine_s / 60.0)

    for mode in FAULT_MODES:
        sim = PumpSimulator(seed=mode_seed(args.seed, mode), fs_vib=25600.0)
        # Settle healthy first: the predictors and regimes must be warm or the
        # measurement includes start-up time.
        for _ in range(args.settle):
            frames.append(sim.step(dt))
        # Then ramp from nothing to full severity over the onset window. The
        # ramp is in MACHINE time, so time_scale controls how abrupt it is.
        sim.time_scale = args.time_scale
        onset_s = sim.t_s
        sim.inject(mode, severity=1.0, onset_s=onset_s,
                   ramp_hours=ramp_minutes / 60.0)
        for _ in range(args.onset):
            frames.append(sim.step(dt))

    n = write_frames(out, frames, 25600.0)
    print(f"wrote {out} ({n} frames: {len(FAULT_MODES)} faults x "
          f"({args.settle} healthy + {args.onset} ramped))", file=sys.stderr)
    print(f"  one frame is {frame_machine_s:g} s of machine time "
          f"({dt:g} s of signal, compressed {args.time_scale:g}x)", file=sys.stderr)
    print(f"  ramp: 0 to full severity over {ramp_minutes:g} min of machine time, "
          f"i.e. {ramp_minutes * 60.0 / frame_machine_s:.0f} frames", file=sys.stderr)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    scenario = Scenario.load(args.scenario)
    print(f"name:         {scenario.name}")
    print(f"seed:         {scenario.seed}")
    print(f"signal:       {scenario.duration_s:g} s at dt={scenario.dt:g}")
    print(f"machine time: {scenario.machine_hours:.2f} h (x{scenario.time_scale:g})")
    if scenario.description:
        print(f"\n{scenario.description}")
    print("\nfaults:")
    for f in scenario.faults or [{"mode": "(none)"}]:
        print(f"  {f}")
    print("\nexpected:")
    for e in scenario.expect:
        fault = f" ({e.fault})" if e.fault else ""
        print(f"  {e.state:<9} by {e.by_wall_s:g}s{fault}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        s = Scenario.load(path)
        print(f"{s.name:<20} {s.duration_s:>6.0f}s  {s.machine_hours:>6.2f}h machine")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sim.cli", description="Harmonic pump simulator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scenario", help="run a scenario")
    p.add_argument("scenario", help="scenario name or path to a YAML file")
    p.add_argument("--out", help="write the per-frame feature summary as CSV")
    p.add_argument("--raw", help="write raw vibration windows as NPZ")
    p.add_argument(
        "--realtime",
        action="store_true",
        help="pace the run to wall clock instead of running flat out",
    )
    p.add_argument("--progress", action="store_true", help="print progress to stderr")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("dataset", help="generate a dataset")
    p.add_argument("kind", choices=["baseline", "faults"])
    p.add_argument("--hours", type=float, default=24.0, help="baseline: machine hours")
    p.add_argument(
        "--stride",
        type=float,
        default=60.0,
        help="baseline: machine seconds each window stands for",
    )
    p.add_argument("--windows", type=int, default=300, help="faults: windows per run")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="sim/artifacts/baseline_normal.npz")
    p.add_argument("--out-dir", default="sim/artifacts/faults")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("frames", help="export raw frames for the C words encoder")
    p.add_argument("--out", default="sim/artifacts/train_frames.hframe")
    p.add_argument("--window", type=int, default=16384,
                   help="samples per frame; must be a power of two. The "
                        "reference is 16384: at 4096 the 6.25 Hz bins cannot "
                        "separate 3X from BPFO.")
    p.add_argument("--healthy", type=int, default=400, help="healthy frames")
    p.add_argument("--per-fault", type=int, default=120,
                   help="frames per fault mode per severity")
    p.add_argument("--seed", type=int, default=5150)
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("latency-frames",
                       help="frames with ramped fault onsets, for latency measurement")
    p.add_argument("--out", default="sim/artifacts/latency_frames.hframe")
    p.add_argument("--window", type=int, default=16384)
    p.add_argument("--settle", type=int, default=150,
                   help="healthy frames before each onset")
    p.add_argument("--onset", type=int, default=150, help="frames covering the ramp")
    p.add_argument("--ramp-minutes", type=float, default=None,
                   help="machine minutes from nothing to full severity; the "
                        "default spans exactly --onset frames, which is what "
                        "makes the benchmark's severity column mean what it "
                        "says. Set it only to study a ramp deliberately longer "
                        "or shorter than the observation window.")
    p.add_argument("--time-scale", type=float, default=60.0,
                   help="machine seconds per frame during the ramp")
    p.add_argument("--seed", type=int, default=8080)
    p.set_defaults(func=cmd_latency_frames)

    p = sub.add_parser("inspect", help="describe a scenario without running it")
    p.add_argument("scenario")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("list", help="list the available scenarios")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
