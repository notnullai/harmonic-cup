"""Machines: what is watched, and commissioning a new one.

A machine is a description (its profile), a recording to replay, and a brain
commissioned on that machine's own healthy running. This module keeps the
registry and runs commissioning — the same steps `scripts/evaluate_ims.py`
runs by hand for the IMS rig, so the console can do them for any machine:

  1. write the machine's profile: speed, bearing geometry, sensor unit
  2. mint a brain from it, carrying over the trained network
  3. measure the noise floor on the baseline — 2x the 99.9th percentile of
     healthy surprise, the rule scripts/noise_floor.py set for the pump
  4. mint again with that floor, and commission on the baseline
  5. replay the baseline through the result and count false alarms

Nothing in the engine changes for a new machine; that is the point of the
profile. Everything here is data: `<data dir>/machines/<id>/`.

The baseline has to be HEALTHY. Commissioning on a failing machine teaches the
engine that failing is normal, and nothing downstream can undo it — which is
why the wizard draws the recording first and asks the user to choose.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kg import families, standards
from sim.datasets import recordings as rec_mod
from sim.datasets.ims_loader import (
    FRAME_HEADER,
    FRAME_MAGIC,
    FRAME_RECORD,
    FRAME_VERSION,
    HEALTHY_LABEL,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "sim" / "artifacts"
TRAINED = ARTIFACTS / "trained.harmonic"
DATA_DIR = Path(os.environ.get("HARMONIC_DATA_DIR", REPO / "var"))
MACHINES_DIR = DATA_DIR / "machines"
UPLOADS_DIR = DATA_DIR / "uploads"

# The network's geometry, fixed by the trained brain it is copied from: the
# pump's, and what `net_args` reads from any other family's network.
NET_ARGS = ("128", "5", "16")
WARMUP_TICKS = 60
# `commission` refuses fewer than 100 usable readings after the warm-up.
MIN_BASELINE = WARMUP_TICKS + 100
MIN_DEAD_ZONE = 0.02
# measure_floor: a baseline reading this many doublings (here 4x, 12 dB) below
# the overall RMS of the readings around it is not the machine running.
RUNNING_GATE_DOUBLINGS = 2.0
RMS_KIND = "HARMONIC_FEAT_VIB_RMS"
MAX_POINTS = 8
SPEED_MODES = ("constant", "measured", "estimated")

# Vibration severity classes (profiles/vibration_standards.json): velocity RMS
# in mm/s, the A/B, B/C and C/D bounds and a top-of-D. They apply to an
# accelerometer too: the engine integrates acceleration to velocity.
ISO_CLASSES: dict[str, dict[str, Any]] = standards.classes()

# Rolling-element bearings a user is likely to have in front of them. Geometry
# per row: pitch diameter, ball diameter, ball count, contact angle.
# The bearing catalogue (HARM-S7-T10): designation to geometry, each entry with
# the public source its numbers come from. Anything else is entered by hand.
BEARINGS: dict[str, dict[str, Any]] = json.loads(
    (REPO / "profiles" / "bearings.json").read_text(encoding="utf-8"))["bearings"]

STEPS = ("Reading the recording", "Writing the machine profile",
         "Measuring the noise floor", "Minting the brain", "Commissioning",
         "Checking the baseline")


class CommissionError(Exception):
    """A reason the machine cannot be commissioned, in words for the user."""


@dataclass
class Machine:
    id: str
    name: str
    equipment_id: str
    nominal_rpm: float
    vib_unit: str                       # "g" or "mm/s"
    iso_class: str                      # a key of ISO_CLASSES
    bearing: dict[str, float]           # pitch_mm, ball_mm, n_balls, contact_angle_deg
    components: list[str]
    recording: str                      # a recording key
    channel: int
    baseline_h: tuple[float, float]
    # Where the shaft speed comes from: the nominal speed throughout
    # ("constant"), the recording's own speed column ("measured"), or the
    # engine's estimate from the 1X peak ("estimated").
    speed: str = "constant"
    # The machine family (profiles/families.json): which trained network it
    # runs, what its classes mean, which graph and manual diagnose it — and
    # the family's own parameters, such as a gearbox's tooth count.
    family: str = "pump"
    params: dict[str, float] = field(default_factory=dict)
    # Keep watching a live sensor after commissioning (HARM-S7-T08):
    # {"broker": "host:port", "topic": "..."}; None replays the recording.
    live: dict[str, str] | None = None
    created: float = field(default_factory=time.time)
    status: str = "draft"               # draft | commissioning | ready | failed
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    report: dict[str, Any] = field(default_factory=dict)
    builtin: bool = False
    # Earlier baselines, oldest first (HARM-S7-T09): after a repair the machine
    # re-learns normal, and what it learned before is kept with its brain.
    baselines: list[dict[str, Any]] = field(default_factory=list)
    # Why the last re-baseline was refused or failed; the machine kept its brain.
    rebaseline_error: str | None = None
    # Measurement points (HARM-S7-T07): drive end, non-drive end; horizontal,
    # vertical. [{"name": "DE horizontal", "channel": 0}, ...]. Each runs its
    # own engine on its own brain; the machine is as bad as its worst point.
    # Empty: one point, on `channel`.
    points: list[dict[str, Any]] = field(default_factory=list)
    # Empty: the shared registry (MACHINES_DIR), listed by `list_machines()`
    # and joined to the knowledge graph. Set for a machine commissioned
    # outside the registry — api/try_mode.py commissions a public, anonymous
    # try-mode run this way: it never appears in `/machines`, is never
    # touched by another run, and is simply a folder removed on delete.
    base_dir: str = ""

    @property
    def folder(self) -> Path:
        return (Path(self.base_dir) if self.base_dir else MACHINES_DIR) / self.id

    @property
    def brain_path(self) -> Path:
        return self.folder / "brain.harmonic"

    def point_list(self) -> list[dict[str, Any]]:
        return self.points or [{"name": "", "channel": self.channel}]

    def point_brain(self, i: int) -> Path:
        """Point 0's brain is the machine's brain, as before points existed."""
        return self.brain_path if i == 0 else self.folder / f"point{i}.harmonic"

    def point_key(self, i: int) -> str:
        """The fleet key of point i: the machine's id for the first point."""
        return self.id if i == 0 else f"{self.id}@{i}"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["baseline_h"] = list(self.baseline_h)
        return d


# --- the registry ----------------------------------------------------------------

_lock = threading.RLock()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "machine"


def _save(m: Machine) -> None:
    m.folder.mkdir(parents=True, exist_ok=True)
    tmp = m.folder / "machine.json.tmp"
    tmp.write_text(json.dumps(m.to_json(), indent=1))
    tmp.replace(m.folder / "machine.json")


def _load(path: Path) -> Machine:
    d = json.loads(path.read_text())
    d["baseline_h"] = tuple(d["baseline_h"])
    return Machine(**d)


def list_machines() -> list[Machine]:
    if not MACHINES_DIR.is_dir():
        return []
    out = []
    for p in sorted(MACHINES_DIR.glob("*/machine.json")):
        try:
            out.append(_load(p))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(out, key=lambda m: m.created)


def get(machine_id: str) -> Machine | None:
    p = MACHINES_DIR / machine_id / "machine.json"
    return _load(p) if p.exists() else None


def delete(machine_id: str) -> bool:
    m = get(machine_id)
    if m is None or m.builtin:
        return False
    shutil.rmtree(m.folder, ignore_errors=True)
    return True


def recordings() -> dict[str, rec_mod.Recording]:
    """Server recordings plus every upload."""
    out = rec_mod.available()
    if UPLOADS_DIR.is_dir():
        for folder in sorted(UPLOADS_DIR.iterdir()):
            if not folder.is_dir():
                continue
            try:
                r = rec_mod.FolderRecording(folder, key=f"upload:{folder.name}")
            except (ValueError, OSError, json.JSONDecodeError):
                continue
            out[r.key] = r
    return out


def recording(key: str) -> rec_mod.Recording:
    r = recordings().get(key)
    if r is None:
        raise CommissionError(f"no recording {key!r} on the server")
    return r


def family_params(family: str, given: dict[str, Any] | None
                  ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """A machine's family parameters (a gearbox's teeth, a motor's poles and
    supply frequency ...), read from the family's wizard definition in
    profiles/families.json: each given value is checked against the wizard's
    range, and each one not given (absent, None or "") takes the wizard's
    default. Returns the parameters and the wizard entries whose default
    stood in, so a caller can say plainly what was assumed."""
    params: dict[str, float] = {}
    assumed: list[dict[str, Any]] = []
    given = given or {}
    for p in families.get(family).get("wizard", {}).get("parameters", []):
        raw = given.get(p["key"])
        if raw is None or raw == "":
            v = float(p["default"])
            assumed.append(p)
        else:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = float("nan")
        if not p["min"] <= v <= p["max"] or v != int(v):
            raise CommissionError(f"{p['label'].lower()} must be a whole number from "
                                  f"{p['min']} to {p['max']}")
        params[p["key"]] = v
    return params, assumed


def create(spec: dict[str, Any]) -> Machine:
    """Validate a wizard submission and save it as a draft."""
    name = str(spec.get("name", "")).strip()
    if not name:
        raise CommissionError("the machine needs a name")
    rpm = float(spec.get("nominal_rpm", 0))
    if not 60 <= rpm <= 60000:
        raise CommissionError("nominal speed must be between 60 and 60000 rpm")
    unit = spec.get("vib_unit", "g")
    if unit not in ("g", "mm/s"):
        raise CommissionError("the vibration unit must be g or mm/s")
    fam_name = str(spec.get("family") or families.DEFAULT)
    try:
        families.get(fam_name)
    except ValueError as exc:
        raise CommissionError(str(exc)) from None
    params, _ = family_params(fam_name, spec.get("params"))
    iso = spec.get("iso_class", "none")
    if iso not in ISO_CLASSES:
        raise CommissionError(f"unknown ISO class {iso!r}")
    b = spec.get("bearing") or {}
    bearing = {k: float(b.get(k, 0)) for k in ("pitch_mm", "ball_mm", "n_balls",
                                               "contact_angle_deg")}
    if not (bearing["pitch_mm"] > bearing["ball_mm"] > 0 and bearing["n_balls"] >= 3):
        raise CommissionError("bearing geometry: the pitch diameter must exceed the ball "
                              "diameter, with at least 3 balls")
    from kg.model import COMPONENT_KINDS

    components = [c for c in spec.get("components", ["bearing", "shaft", "foundation"])
                  if c in COMPONENT_KINDS]
    if "bearing" not in components:
        components.insert(0, "bearing")
    r = recording(str(spec.get("recording", "")))
    points: list[dict[str, Any]] = [
        {"name": str(p.get("name", "")).strip()[:40], "channel": int(p.get("channel", 0))}
        for p in (spec.get("points") or [])]
    if len(points) > MAX_POINTS:
        raise CommissionError(f"at most {MAX_POINTS} measurement points per machine")
    if points and (len({p["channel"] for p in points}) != len(points)
                   or len({p["name"] for p in points}) != len(points)
                   or not all(p["name"] for p in points)):
        raise CommissionError("each measurement point needs its own name and its own channel")
    if len(points) > 1 and spec.get("live"):
        raise CommissionError("a live sensor carries one channel: one measurement point")
    channel = int(points[0]["channel"]) if points else int(spec.get("channel", 0))
    for c in [p["channel"] for p in points] or [channel]:
        if not 0 <= c < r.n_channels:
            raise CommissionError(f"channel {c} does not exist; the recording has "
                                  f"{r.n_channels}")
    speed = str(spec.get("speed", "constant"))
    if speed not in SPEED_MODES:
        raise CommissionError(f"unknown speed source {speed!r}")
    if speed == "measured" and r.speeds() is None:
        raise CommissionError("the recording has no speed column; choose the nominal "
                              "speed or the estimate from vibration")
    live = spec.get("live") or None
    if live is not None:
        from api import adapters

        try:
            b, t = adapters.normalise(str(live.get("broker", "")), str(live.get("topic", "")))
        except ValueError as exc:
            raise CommissionError(str(exc)) from None
        live = {"broker": b, "topic": t}
    lo, hi = (float(x) for x in spec.get("baseline_h", (0.0, 0.0)))
    if hi <= lo:
        raise CommissionError("choose a baseline period")

    with _lock:
        base = _slug(name)
        mid, n = base, 2
        while (MACHINES_DIR / mid).exists():
            mid, n = f"{base}-{n}", n + 1
        m = Machine(id=mid, name=name, equipment_id=f"M-{mid.upper()}", nominal_rpm=rpm,
                    vib_unit=unit, iso_class=iso, bearing=bearing, components=components,
                    recording=r.key, channel=channel, baseline_h=(lo, hi), speed=speed,
                    family=fam_name, params=params, live=live,
                    points=points if len(points) > 1 else [])
        _save(m)
    return m


# --- uploads -------------------------------------------------------------------------


def new_upload(meta: dict[str, Any]) -> str:
    fs = int(meta.get("sample_rate_hz", 0))
    if fs <= 0:
        raise CommissionError("the upload needs its sample rate")
    uid = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    folder = UPLOADS_DIR / uid
    folder.mkdir(parents=True)
    (folder / "recording.json").write_text(json.dumps({
        "name": str(meta.get("name") or f"Upload {uid}"),
        "sample_rate_hz": fs,
        "interval_s": float(meta.get("interval_s", 600.0)),
        "unit": meta.get("unit", "g"),
        "provenance": "uploaded through the console",
        **({"rpm": [float(v) for v in meta["rpm"]]} if meta.get("rpm") else {}),
    }))
    return uid


_SAFE_NAME = re.compile(r"^[A-Za-z0-9._ -]{1,120}$")


def save_upload_file(uid: str, filename: str, body: bytes) -> None:
    folder = UPLOADS_DIR / uid
    if not folder.is_dir() or not re.fullmatch(r"[0-9a-f-]+", uid):
        raise CommissionError("unknown upload")
    if not _SAFE_NAME.match(filename) or filename.startswith(("_", ".")) \
            or filename == "recording.json":
        raise CommissionError(f"refused file name {filename!r}")
    if Path(filename).suffix.lower() not in (".csv", ".txt", ".npy"):
        raise CommissionError("snapshots must be .csv, .txt or .npy")
    (folder / filename).write_bytes(body)


# --- commissioning -------------------------------------------------------------------


def _binary(name: str) -> Path:
    for c in (REPO / "build" / name, REPO / "build" / f"{name}.exe"):
        if c.exists():
            return c
    found = shutil.which(name)
    if found:
        return Path(found)
    raise CommissionError(f"{name} is not built; run: cmake --build build -j")


def _run(cmd: list[Any]) -> str:
    r = subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        raise CommissionError(f"{Path(str(cmd[0])).name} failed: "
                              f"{(r.stderr or r.stdout).strip()[-600:]}")
    return r.stdout


def profile_text(m: Machine, sample_rate_hz: int,
                 dead_zones: dict[str, float] | None = None,
                 unmeasured: list[dict[str, Any]] | None = None) -> str:
    iso = ISO_CLASSES[m.iso_class]["bounds"]
    lines = [
        f"# {m.name} — written by the commissioning wizard",
        f"# {time.strftime('%Y-%m-%d %H:%M')}; recording {m.recording}, channel {m.channel},",
        f"# baseline {m.baseline_h[0]:.2f}-{m.baseline_h[1]:.2f} h.",
        f"name = {m.id}",
        f"nominal_rpm = {m.nominal_rpm:g}",
        "iso_zone_bounds = " + ("none" if iso is None else " ".join(f"{v:g}" for v in iso)),
        f"warmup_ticks = {WARMUP_TICKS}",
        "rul_horizon_h = 720",
        f"bearing_pitch_mm = {m.bearing['pitch_mm']:g}",
        f"bearing_ball_mm = {m.bearing['ball_mm']:g}",
        f"bearing_n_balls = {int(m.bearing['n_balls'])}",
        f"bearing_contact_angle_deg = {m.bearing['contact_angle_deg']:g}",
        f"vib_sample_rate_hz = {sample_rate_hz}",
        f"vib_unit = {m.vib_unit}",
    ]
    lines += family_lines(m)
    for u in unmeasured or []:
        lines.append(f"# {u['feature']}: not measured at this window, the engine reads it as "
                     f"0 (sidebands {u['gap_hz']:g} Hz from the carrier, limit "
                     f"{u['limit_hz']:g} Hz).")
    if dead_zones:
        lines.append("# The noise floor, measured on the baseline: 2x the 99.9th "
                     "percentile of healthy |surprise|.")
        lines += [f"dead_zone.{k} = {v:.2f}" for k, v in dead_zones.items()]
    return "\n".join(lines) + "\n"


def net_args(network: Path) -> tuple[str, str, str]:
    """mint_brain's hidden, class and feature counts, read from the header of
    the trained network that will be grafted on — so a family's machine is
    minted in the shape of its family's network."""
    with network.open("rb") as fh:
        head = fh.read(36)
    _, _, _, _, hidden, classes, features = struct.unpack("<8sIIIIII", head[:32])
    return str(hidden), str(classes), str(features)


def _arith(expr: str, names: dict[str, float]) -> float:
    """+ - * / over numbers and the given names; nothing else is evaluated."""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv}

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return float(ops[type(node.op)](ev(node.left), ev(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise CommissionError(f"the family's derived value {expr!r} is not plain arithmetic "
                              "over the machine's numbers")
    try:
        return ev(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        raise CommissionError(f"the family's derived value {expr!r} divides by zero") from None


def bearing_orders(b: dict[str, float]) -> dict[str, float]:
    """FTF, BPFO, BPFI, BSF and 2xBSF as shaft orders (multiples of 1x), from
    the bearing geometry alone: ball count, ball and pitch diameter, contact
    angle. Standard rolling-element-bearing kinematics.

    This mirrors, order-for-order, what the engine itself derives at runtime
    from a brain's bearing header — harmonic_bearing_frequencies in
    engine/src/harmonic_features.cpp, which is what actually computes the
    VIB_BPFO/VIB_BPFI/VIB_BSF feature kinds (the engine already reads the
    geometry back out of the brain; nothing here duplicates that at the
    feature-extraction level). This copy exists only so a family's own order
    features — a sideband search around BPFI, say, which the built-in
    VIB_BPFO/BPFI/BSF kinds don't support — can be written as an order
    template the same way {ftf} already is, instead of a hand-typed number.
    """
    ratio = b["ball_mm"] / b["pitch_mm"] * math.cos(math.radians(b["contact_angle_deg"]))
    n = b["n_balls"]
    bsf = (b["pitch_mm"] / (2.0 * b["ball_mm"])) * (1.0 - ratio * ratio)
    return {
        "ftf": 0.5 * (1.0 - ratio),
        "bpfo": (n / 2.0) * (1.0 - ratio),
        "bpfi": (n / 2.0) * (1.0 + ratio),
        "bsf": bsf,
        "bsf2": 2.0 * bsf,
    }


def derived_values(family: str, params: dict[str, float], rpm: float,
                   bearing: dict[str, float]) -> dict[str, float]:
    """The family's derived numbers (families.json `derived`) for these
    parameters, speed and bearing, beside the values they come from."""
    values: dict[str, float] = {k: int(v) if float(v).is_integer() else float(v)
                                for k, v in params.items()}
    values.update({"rpm": rpm, **bearing_orders(bearing)})
    for k, expr in families.get(family).get("wizard", {}).get("derived", {}).items():
        v = _arith(expr, values)
        values[k] = int(v) if v.is_integer() else v
    return values


def fit_assumed(family: str, params: dict[str, float], assumed: set[str], rpm: float,
                bearing: dict[str, float]) -> tuple[dict[str, float], str | None]:
    """Make assumed parameters fit a known speed. A family's derived values
    feed its order templates, and an order must lie in (0, 1000]; a default
    may not fit the speed a visitor gave (a 4-pole motor on 60 Hz cannot turn
    at 2250 rpm). Then the one assumed parameter whose smallest relative move
    from its default, in its wizard range and `step`, makes every derived value
    fit is moved. Nothing given is ever changed, and nothing is special to
    one family. Returns the parameters and the key moved (None if none)."""
    wiz = families.get(family).get("wizard", {})
    derived = list(wiz.get("derived", {}))

    def fits(ps: dict[str, float]) -> bool:
        try:
            vals = derived_values(family, ps, rpm, bearing)
        except CommissionError:
            return False
        return all(0 < vals[k] <= 1000 for k in derived)

    if fits(params):
        return params, None
    best: tuple[float, str, float] | None = None
    for p in wiz.get("parameters", []):
        if p["key"] not in assumed:
            continue
        d, step = float(p["default"]), float(p.get("step", 1))
        k = 1
        while d - k * step >= p["min"] or d + k * step <= p["max"]:
            for v in (d - k * step, d + k * step):
                if p["min"] <= v <= p["max"] and fits({**params, p["key"]: v}):
                    rel = abs(v - d) / max(abs(d), 1.0)
                    if best is None or rel < best[0]:
                        best = (rel, p["key"], v)
                    break
            else:
                k += 1
                continue
            break
    if best is None:
        return params, None
    return {**params, best[1]: best[2]}, best[1]


def family_orders(m: Machine) -> dict[str, str]:
    """The family's order features with this machine's numbers, label ->
    "<order> spectrum|envelope [sidebands <spacing>]": a 29-tooth gearbox
    meshes at order 29."""
    wiz = families.get(m.family).get("wizard", {})
    if not wiz.get("features"):
        return {}
    # The machine's own numbers: its parameters (whole numbers, so a tooth
    # count prints as 37), its speed and its bearing's defect orders, and
    # what the family derives from them.
    values = derived_values(m.family, m.params, m.nominal_rpm, m.bearing)
    return {k: v.format(**values) for k, v in wiz.get("orders", {}).items()}


def family_lines(m: Machine) -> list[str]:
    """The family's own features, written into the machine's profile with
    this machine's numbers. The labels stay the family's, which is what lets
    the family's trained network be grafted on (mint_brain matches features
    by label)."""
    wiz = families.get(m.family).get("wizard", {})
    if not wiz.get("features"):
        return []
    lines = [f"# {families.get(m.family)['label']}: the family's features."]
    lines += [f"order.{k} = {v}" for k, v in family_orders(m).items()]
    lines += [f"process.{k} = {v}" for k, v in wiz.get("process", {}).items()]
    lines.append(f"features = {wiz['features']}")
    return lines


# How far from where it should be the engine looks for a line at an order
# (engine/src/harmonic_features.cpp): a bin and a half of spectral leakage,
# plus half a percent of the shaft speed per order for drift.
TOL_LEAK_BINS = 1.5
TOL_SPEED_PER_ORDER = 0.005


def sideband_resolution(order: float, spacing: float, rpm: float, sample_rate_hz: float,
                        window: int) -> dict[str, Any]:
    """Whether sidebands at order +- spacing can be told from their carrier at
    this window, sample rate and speed. The engine looks for each sideband,
    and for the carrier, within its tolerance of where the line should be;
    when the two searches overlap, the strongest line in a sideband's band can
    be the carrier itself, and the "sidebands" read the carrier again. The
    engine does not measure such a feature (it reads 0); this is the same
    rule, for the commissioning report.

    `window_needed` is the shortest power-of-two window that would separate
    them at this rate and speed, or None when none can: the drift allowance
    alone already spans the spacing."""
    f1 = rpm / 60.0
    bin_hz = sample_rate_hz / window
    tol = TOL_LEAK_BINS * bin_hz + order * f1 * TOL_SPEED_PER_ORDER
    gap, limit = spacing * f1, 2.0 * tol
    # gap > 2 * (1.5 * fs / n + order * drift * f1)  <=>  n > 3 fs / (gap - 2 * order * drift * f1)
    room = gap - 2.0 * order * f1 * TOL_SPEED_PER_ORDER
    needed = None
    if room > 0:
        needed = 1
        while not needed * room > 2.0 * TOL_LEAK_BINS * sample_rate_hz:
            needed *= 2
    return {"gap_hz": gap, "limit_hz": limit, "bin_hz": bin_hz, "resolved": gap > limit,
            "window_needed": needed}


def unmeasured_sidebands(m: Machine, sample_rate_hz: float, window: int) -> list[dict[str, Any]]:
    """The family's sideband features this machine cannot measure at this
    window and sample rate (see sideband_resolution), each with the reason in
    words for the commissioning report. The feature keeps its place, so the
    family's network still grafts on; the engine reads it as 0."""
    from api.feature_labels import label as plain_label

    out = []
    for name, spec in family_orders(m).items():
        words = spec.split()
        if len(words) != 4 or words[2] != "sidebands":
            continue
        order, spacing = float(words[0]), float(words[3])
        res = sideband_resolution(order, spacing, m.nominal_rpm, sample_rate_hz, window)
        if res["resolved"]:
            continue
        kind = "VIB_ENV_ORDER" if words[1] == "envelope" else "VIB_ORDER"
        plain = plain_label(f"{kind}:{name}", m.family)
        need = res["window_needed"]
        cure = ("No longer snapshot separates them at this speed: the speed allowance "
                "alone spans the spacing." if need is None else
                f"Snapshots of {need} samples or more at this rate would separate them."
                + ("" if need <= rec_mod.MAX_WINDOW else
                   f" The engine's longest window is {rec_mod.MAX_WINDOW}."))
        out.append({
            "feature": name, "name": plain, "order": order, "spacing": spacing,
            "gap_hz": round(res["gap_hz"], 3), "limit_hz": round(res["limit_hz"], 3),
            "reason": (
                f"Not measured on this machine: {plain} ({name}). Its sidebands lie "
                f"{res['gap_hz']:.2f} Hz either side of order {order:g} at "
                f"{m.nominal_rpm:g} rpm, and with {window}-sample windows at "
                f"{sample_rate_hz:g} Hz the engine can only tell a sideband from its carrier "
                f"when they are more than {res['limit_hz']:.2f} Hz apart: here they would "
                "read the carrier a second time. The engine reads this feature as 0: it "
                "adds nothing to any fault class or to the alarm, and a fault that shows "
                "only there is not seen. " + cure),
        })
    return out


def speeds_for(m: Machine, r: rec_mod.Recording) -> np.ndarray:
    """The speed fed to the engine for each snapshot of the recording, in rpm;
    0 asks the engine to estimate it from the spectrum."""
    n = len(r.hours())
    if m.speed == "measured":
        measured = r.speeds()
        if measured is None:
            raise CommissionError("the recording has no speed column")
        return np.asarray(measured, dtype=np.float64)
    if m.speed == "estimated":
        return np.zeros(n)
    return np.full(n, float(m.nominal_rpm))


def write_frames(path: Path, hours: np.ndarray, data: np.ndarray, sample_rate_hz: int,
                 rpm: float | np.ndarray) -> None:
    """Snapshots as a Harmonic frame file, labelled healthy, vibration only.
    `rpm` is one speed, or one per snapshot; 0 means estimate it."""
    rpms = np.broadcast_to(np.asarray(rpm, dtype=np.float64), (len(data),))
    with path.open("wb") as fh:
        fh.write(struct.pack(FRAME_HEADER, FRAME_MAGIC, FRAME_VERSION, len(data),
                             data.shape[1], sample_rate_hz, 5, 0, 0))
        for h, samples, s in zip(hours, data, rpms, strict=True):
            fh.write(struct.pack(FRAME_RECORD, int((h - hours[0]) * 3.6e9), int(s * 65536),
                                 int(0.65 * 65536), HEALTHY_LABEL, *([0.0] * 7), 0))
            fh.write(np.ascontiguousarray(samples, dtype=np.float32).tobytes())


def replay(brain: Path, hours: np.ndarray, data: np.ndarray, sample_rate_hz: int,
           rpm: float | np.ndarray, guard: bool = True,
           check: Callable[[], None] | None = None) -> list[Any]:
    """Every tick of a replay through the engine, in order. `rpm` is one speed,
    or one per snapshot; 0 means estimate it. `guard=False` turns the encoder's
    learning guard off, for measuring the noise floor. `check`, if given, is
    called before every tick and stops the replay by raising (try mode's
    cancelled or timed-out run)."""
    from api.rig import HarmonicRig

    rig = HarmonicRig(brain, window_samples=data.shape[1], sample_rate_hz=sample_rate_hz)
    if not guard:
        rig.set_guard(False)
    ticks = []
    try:
        prev = hours[0]
        rpms = np.broadcast_to(np.asarray(rpm, dtype=np.float64), (len(data),))
        for h, samples, s in zip(hours, data, rpms, strict=True):
            if check is not None:
                check()
            ticks.append(rig.feed(np.ascontiguousarray(samples, dtype=np.float32),
                                  rpm=float(s),
                                  load=0.65, timestamp_us=int((h - hours[0]) * 3.6e9),
                                  dt_us=max(int((h - prev) * 3.6e9), 1), present=0))
            prev = h
    finally:
        rig.close()
    return ticks


def speed_check(rpms: np.ndarray, nominal: float) -> dict[str, Any]:
    """Whether a speed estimated from vibration can be trusted.

    The estimate is the loudest line near the nominal speed. On a machine
    whose shaft line is not the loudest — the IMS bearing rig, where belt and
    structural lines at 28-50 Hz outshout 1X — it wanders, and every order and
    defect frequency wanders with it. Over a healthy baseline the speed should
    be steady and near nominal; if it is not, the estimate is refused."""
    rpms = np.asarray(rpms, dtype=np.float64)
    med = float(np.median(rpms))
    p5, p95 = (float(v) for v in np.percentile(rpms, [5, 95]))
    stable = abs(med - nominal) / nominal <= 0.05 and (p95 - p5) / med <= 0.10
    return {"median_rpm": round(med, 1), "p5_rpm": round(p5, 1), "p95_rpm": round(p95, 1),
            "nominal_rpm": nominal, "stable": stable}


def measure_floor(ticks: list[Any], info: dict[str, Any] | None = None) -> dict[str, float]:
    """2x the 99.9th percentile of |surprise| per vibration feature, over the
    warm ticks the machine was running in. The slow channels are not recorded
    and are left alone.

    A reading whose overall RMS drops more than RUNNING_GATE_DOUBLINGS below
    the median of the readings around it (WARMUP_TICKS of them, centred) is
    the machine stopped or the sensor dropping out, not running noise, and
    does not set the floor: one such reading among a few hundred is above the
    99.9th percentile, and on FEMTO bearing 2_1 a single one widened the RMS
    floor from 1.9 to 7.4 doublings, which hid the failure. The reference is
    local because a baseline can rise: XJTU-SY 3_5's horizontal sensor ran 4x
    quieter early in its learning window than late, and that is running.
    Nothing louder than its neighbours is left out; a loud baseline is a
    different problem. `info`, if given, receives the count left out."""
    warm = [t for t in ticks if t.warm]
    rms = [t.features.get(RMS_KIND) for t in warm]
    if warm and all(v is not None for v in rms):
        v = np.asarray(rms, dtype=np.float64)
        half = WARMUP_TICKS // 2
        local = np.array([np.median(v[max(0, i - half): i + half + 1]) for i in range(len(v))])
        running = [t for t, keep in zip(warm, v >= local - RUNNING_GATE_DOUBLINGS, strict=True)
                   if keep]
    else:
        running = warm
    if info is not None:
        info["not_running"] = len(warm) - len(running)
    seen: dict[str, list[float]] = {}
    for t in running:
        if t.warm:
            for kind, s in t.surprise.items():
                if kind.startswith("HARMONIC_FEAT_VIB_"):
                    # A declared feature is keyed by its label ("gear_mesh"),
                    # a table kind by its name ("VIB_BPFO").
                    key = (kind.split(":", 1)[1] if ":" in kind
                           else kind.removeprefix("HARMONIC_FEAT_"))
                    seen.setdefault(key, []).append(abs(s))
    return {k: round(max(MIN_DEAD_ZONE, 2.0 * float(np.percentile(v, 99.9))), 2)
            for k, v in seen.items()}


def _commission_point(m: Machine, r: Any, network: Path, channel: int, out: Path,
                      tag: str, step: Callable[[int], None]) -> dict[str, Any]:
    """Commission one measurement point: its channel of the recording, its own
    noise floor and brain. Returns its report."""
    window = rec_mod.window_for(r.snapshot_samples())
    hours = r.hours()
    data = r.load(channel, window)
    sel = np.flatnonzero((hours >= m.baseline_h[0]) & (hours <= m.baseline_h[1]))
    if len(sel) < MIN_BASELINE:
        raise CommissionError(
            f"the baseline holds {len(sel)} readings; commissioning needs at least "
            f"{MIN_BASELINE} ({WARMUP_TICKS} to warm up and 100 to learn from). "
            "Widen the period.")
    b_hours, b_data = hours[sel], data[sel]
    b_rpm = speeds_for(m, r)[sel]
    if not np.isfinite(b_data).all():
        raise CommissionError("the baseline contains missing or non-numeric samples")
    if float(rec_mod.rms(b_data).min()) <= 0.0:
        raise CommissionError("the baseline contains a flat, all-zero snapshot — a "
                              "sensor that was off")

    m.folder.mkdir(parents=True, exist_ok=True)
    step(1)
    # Sidebands this window cannot tell from their carrier: the engine reads
    # them as 0, and the report says which and why.
    unmeasured = unmeasured_sidebands(m, r.sample_rate_hz, window)
    prof0 = m.folder / f"draft-{tag}.profile"
    prof0.write_text(profile_text(m, r.sample_rate_hz, unmeasured=unmeasured))
    draft = m.folder / f"draft-{tag}.harmonic"
    _run([_binary("mint_brain"), draft, *net_args(network), "--profile", prof0,
          "--weights-from", network])

    step(2)
    floor = measure_floor(replay(draft, b_hours, b_data, r.sample_rate_hz, b_rpm,
                                 guard=False))
    # A feature that reads a constant has no noise floor to measure.
    for u in unmeasured:
        floor.pop(u["feature"], None)
    if not floor:
        raise CommissionError("no warm readings to measure the noise floor on")

    step(3)
    prof = m.folder / (f"{m.id}.profile" if tag == "p0" else f"{m.id}.{tag}.profile")
    prof.write_text(profile_text(m, r.sample_rate_hz, floor, unmeasured))
    base = m.folder / f"base-{tag}.harmonic"
    _run([_binary("mint_brain"), base, *net_args(network), "--profile", prof,
          "--weights-from", network])

    step(4)
    frames = m.folder / f"baseline-{tag}.hframe"
    write_frames(frames, b_hours, b_data, r.sample_rate_hz, b_rpm)
    log_text = _run([_binary("commission"), frames, "--brain", base, "--out", out])

    step(5)
    ticks = replay(out, b_hours, b_data, r.sample_rate_hz, b_rpm)
    warm = [t for t in ticks if t.warm]
    alarms = sum(t.state != "OK" for t in warm)
    speed = None
    if m.speed == "estimated":
        speed = speed_check(np.array([t.rpm for t in ticks]), m.nominal_rpm)
        if not speed["stable"]:
            raise CommissionError(
                f"the speed estimated from vibration is not steady: median "
                f"{speed['median_rpm']:.0f} rpm, 5-95% {speed['p5_rpm']:.0f}-"
                f"{speed['p95_rpm']:.0f} rpm, against a nominal {m.nominal_rpm:g}. "
                "The shaft's own line is not the loudest near that speed, so every "
                "frequency would be looked for in the wrong place. Use the nominal "
                "speed, or a recording with a speed column.")

    for p in (draft, prof0, base, frames):
        p.unlink(missing_ok=True)
    return {
        "readings": len(sel),
        "span_h": round(float(b_hours[-1] - b_hours[0]), 2),
        "window_samples": window,
        "sample_rate_hz": r.sample_rate_hz,
        "dead_zones": floor,
        "warm_readings": len(warm),
        "false_alarms": alarms,
        "verdict": "ok" if alarms == 0 else "check",
        "note": ("No alarm on the baseline: the engine accepts this period as "
                 "normal." if alarms == 0 else
                 f"{alarms} of {len(warm)} baseline readings raised an alarm after "
                 "commissioning. The period may include abnormal running, or be "
                 "too short to learn from; choose a steadier stretch."),
        "commission_log": log_text.strip().splitlines()[:12],
        "speed_source": m.speed,
        "speed": speed,
        "unmeasured": [{k: u[k] for k in ("feature", "name", "reason")} for u in unmeasured],
    }


def commission(machine: str | Machine,
               register: Callable[[Machine], None] | None = None) -> Machine:
    """Run the whole pipeline. Progress and the outcome are saved as it goes,
    so the console can poll; a failure is saved with its reason.

    `machine` is a registry id (looked up in `MACHINES_DIR`), or a `Machine`
    already built with its own `base_dir` — how api/try_mode.py commissions
    a run without ever putting it in the shared registry."""
    m = get(machine) if isinstance(machine, str) else machine
    if m is None:
        raise CommissionError(f"no machine {machine!r}")

    def step(i: int, tag: str = "") -> None:
        m.status = "commissioning"
        m.progress = {"step": i + 1, "of": len(STEPS), "label": STEPS[i] + tag}
        _save(m)

    try:
        step(0)
        network = families.path(m.family, "network")
        if not network.exists():
            raise CommissionError(f"the {m.family} network is missing; run: "
                                  f"python scripts/build_brain.py --family {m.family}")
        r = recording(m.recording)
        pts = m.point_list()
        reports = []
        for i, pt in enumerate(pts):
            tag = f" — {pt['name']}" if len(pts) > 1 else ""

            def point_step(k: int, tag: str = tag) -> None:
                step(k, tag)

            reports.append(_commission_point(m, r, network, int(pt["channel"]),
                                             m.point_brain(i), f"p{i}", point_step))
        m.report = dict(reports[0])
        if len(pts) > 1:
            m.report["points"] = [{"name": pt["name"], "channel": pt["channel"],
                                   **{k: rep[k] for k in ("readings", "warm_readings",
                                                          "false_alarms", "verdict")}}
                                  for pt, rep in zip(pts, reports, strict=True)]
            alarms = sum(int(rep["false_alarms"]) for rep in reports)
            checked = sum(int(rep["warm_readings"]) for rep in reports)
            m.report.update({
                "false_alarms": alarms, "warm_readings": checked,
                "verdict": "ok" if alarms == 0 else "check",
                "note": (f"No alarm on the baseline at any of the {len(pts)} points." if
                         alarms == 0 else
                         f"{alarms} of {checked} baseline readings raised an alarm across "
                         f"the {len(pts)} points; choose a steadier stretch."),
            })
        m.status = "ready"
        m.progress = {"step": len(STEPS), "of": len(STEPS), "label": "Done"}
        m.error = None
        _save(m)
        if register is not None:
            register(m)
        return m
    except CommissionError as exc:
        m.status, m.error = "failed", str(exc)
        _save(m)
        raise
    except Exception as exc:
        m.status, m.error = "failed", f"{type(exc).__name__}: {exc}"
        _save(m)
        raise


def start(machine_id: str, register: Callable[[Machine], None] | None = None) -> None:
    """Commission in the background; the console polls `get`."""
    def run() -> None:
        # A failure is already saved on the machine record, with its reason.
        with contextlib.suppress(Exception):
            commission(machine_id, register)
    threading.Thread(target=run, name=f"commission-{machine_id}", daemon=True).start()


def rebaseline(machine_id: str, spec: dict[str, Any]) -> Machine:
    """Re-learn normal after maintenance (HARM-S7-T09): prepare the machine to
    be commissioned again on a new healthy period, keeping the old baseline.

    The brain in use is archived beside the machine and stays in service until
    the new one is proven: `start_rebaseline` swaps it in only when
    commissioning succeeds, and restores the old one when it does not. Events
    and history are the fleet's and are not touched.

    A machine that replays a recording re-learns from a period of that same
    recording. A live machine re-learns from any recording — typically one its
    sensor made after the repair, through the wizard's live listener.
    """
    m = get(machine_id)
    if m is None:
        raise CommissionError(f"no machine {machine_id!r}")
    if m.builtin:
        raise CommissionError("a built-in machine is not re-baselined")
    if m.status != "ready":
        raise CommissionError(f"the machine is {m.status}; only a commissioned machine "
                              "can re-learn normal")
    rec_key = str(spec.get("recording") or m.recording)
    if m.live is None and rec_key != m.recording:
        raise CommissionError(
            "a machine that replays a recording re-learns from a period of that same "
            "recording; to learn from another recording, commission it as a new machine")
    recording(rec_key)                      # exists, or CommissionError
    iso = str(spec.get("iso_class") or m.iso_class)
    if iso not in ISO_CLASSES:
        raise CommissionError(f"unknown ISO class {iso!r}")
    lo, hi = (float(x) for x in spec["baseline_h"])
    if not hi > lo:
        raise CommissionError("the baseline period must end after it starts")
    n = len(m.baselines) + 1
    brains = []
    for i in range(len(m.point_list())):
        archived = m.folder / (f"brain-{n}.harmonic" if i == 0 else f"point{i}-{n}.harmonic")
        shutil.copyfile(m.point_brain(i), archived)
        brains.append(archived.name)
    m.baselines.append({
        "n": n, "until": time.time(), "brain": brains[0], "brains": brains,
        "recording": m.recording, "channel": m.channel, "baseline_h": list(m.baseline_h),
        "iso_class": m.iso_class,
        "report": {k: v for k, v in m.report.items() if k != "commission_log"},
        "note": str(spec.get("note", ""))[:200],
    })
    m.recording, m.channel = rec_key, int(spec.get("channel", m.channel))
    m.baseline_h = (lo, hi)
    # The alarm zones can change with it: a machine commissioned without ISO
    # zones gets them by commissioning again, on the same healthy period.
    m.iso_class = iso
    m.rebaseline_error = None
    # Commissioning from here on; the console polls until it is ready again.
    m.status, m.progress = "commissioning", {"step": 0, "of": len(STEPS), "label": "Queued"}
    _save(m)
    return m


def run_rebaseline(machine_id: str, on_ready: Callable[[Machine], None] | None = None
                   ) -> bool:
    """Commission again; on failure put the previous brain and baseline back.
    Returns whether the new baseline was taken."""
    try:
        commission(machine_id)
    except Exception as exc:
        m = get(machine_id)
        if m is None or not m.baselines:
            return False
        prev = m.baselines.pop()
        for i, name in enumerate(prev.get("brains") or [prev["brain"]]):
            shutil.copyfile(m.folder / name, m.point_brain(i))
            (m.folder / name).unlink(missing_ok=True)
        m.recording, m.channel = prev["recording"], prev["channel"]
        m.baseline_h = (prev["baseline_h"][0], prev["baseline_h"][1])
        m.iso_class = prev.get("iso_class", m.iso_class)
        m.report = prev["report"]
        m.status, m.error = "ready", None
        m.rebaseline_error = (str(exc) if isinstance(exc, CommissionError)
                              else f"{type(exc).__name__}: {exc}")
        m.progress = {}
        _save(m)
        return False
    m = get(machine_id)
    if m is not None and on_ready is not None:
        on_ready(m)
    return True


def start_rebaseline(machine_id: str, on_ready: Callable[[Machine], None]) -> None:
    """`run_rebaseline` in the background; the console polls `get`."""
    threading.Thread(target=run_rebaseline, args=(machine_id, on_ready),
                     name=f"rebaseline-{machine_id}", daemon=True).start()


def kg_graph(m: Machine) -> Any:
    """The machine as graph nodes: equipment plus its component kinds, so the
    diagnosis can rule out what the machine cannot have."""
    from kg.model import Graph, Node, Triple

    iso = "none" if m.iso_class == "none" else m.iso_class
    props: dict[str, Any] = {"model": f"commissioned: {m.name}", "serial": m.id,
                             "nominal_rpm": m.nominal_rpm, "iso_class": iso,
                             "family": m.family}
    # Its own manual when one was uploaded (api/manuals.py), with the family's
    # reference manual beside it; without one, retrieval uses the family's.
    from api import manuals

    own = manuals.info(m.id)
    if own is not None:
        from kg import families

        fam_docs = families.get(m.family).get("manuals", [])
        props["manuals"] = ",".join([own["doc_id"], *fam_docs])
    nodes = [Node("Equipment", m.equipment_id, props)]
    triples = []
    for kind in m.components:
        cid = f"{m.equipment_id}.{kind}"
        nodes.append(Node("Component", cid, {"kind": kind, "position": kind}))
        triples.append(Triple(m.equipment_id, "HAS_COMPONENT", cid))
    return Graph(nodes, triples)
