"""Where the engine's frames come from.

The bridge ticks the engine; a source hands it one frame at a time and says
what it is. Two exist:

  * `SimulatedSource` — the P-101 pump model, with fault injection.
  * `RecordedSource`  — a real recording replayed in order. The IMS bearing
    run-to-failure is the one registered; nothing can be injected into it.

Each source names the brain it needs, because a brain describes one machine:
the pump brain knows 2950 rpm and a 6308 bearing; the IMS brain knows 2000 rpm
and a ZA-2115. Swapping the data without swapping the brain would ask the
engine to look for the wrong bearing's defect frequencies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kg import families, standards
from sim.pump_sim import PumpSimulator

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "sim" / "artifacts"

SLOW_PRESENT_ALL = 0x7F


@dataclass
class SourceFrame:
    """One frame, in the shape the engine takes, plus what to display."""

    t_s: float                    # machine time since the start of the source
    dt_s: float                   # machine time this frame stands for
    vibration: np.ndarray         # float32, the source's own unit
    rpm: float
    load: float
    slow: list[float]             # the 7 slow channels, in the frame order
    present: int                  # which of them were measured
    sensors: dict[str, float | None]   # display values; None = not measured
    recorded_at: str | None = None     # the original timestamp, for recordings


class Source(Protocol):
    key: str
    brain_path: Path
    window_samples: int
    sample_rate_hz: int
    vib_unit: str
    equipment_id: str
    event_cooldown_s: int
    supports_injection: bool

    def next(self) -> SourceFrame | None: ...
    def info(self) -> dict[str, Any]: ...
    def severities(self) -> dict[str, float]: ...


# --- the simulator ----------------------------------------------------------------


class FaultGrowth:
    """Pausing and resuming an injected fault's growth, for the simulators
    (every simulator's Fault has `severity`, `onset_s`, `ramp_hours` and
    `level(t_s)`). A demo runs machine time far faster than real time, so a
    fault ramped over an hour crosses WATCH, WARNING and CRITICAL in seconds;
    holding it lets the machine stay in a state long enough to act on it."""

    sim: Any

    def _held_faults(self) -> dict[int, tuple[float, float]]:
        # id(fault) -> (target severity, ramp hours), for the held ones.
        held = getattr(self, "_held", None)
        if held is None:
            held = {}
            self._held = held
        return held

    def hold_faults(self) -> None:
        t = self.sim.t_s
        held = self._held_faults()
        for f in self.sim.faults:
            if id(f) in held or f.ramp_hours <= 0.0:
                continue
            level = f.level(t)
            if level >= f.severity:
                continue                  # already at its target: nothing to hold
            held[id(f)] = (f.severity, f.ramp_hours)
            f.severity, f.ramp_hours, f.onset_s = level, 0.0, min(f.onset_s, t)

    def resume_faults(self) -> None:
        t = self.sim.t_s
        held = self._held_faults()
        for f in self.sim.faults:
            if id(f) not in held:
                continue
            target, ramp_h = held.pop(id(f))
            # Carry on from the level it was held at, at the same rate.
            done = f.severity / target if target > 0 else 1.0
            f.severity, f.ramp_hours = target, ramp_h
            f.onset_s = t - done * ramp_h * 3600.0
        held.clear()

    def growth(self) -> dict[str, Any]:
        """Each injected mode's target severity, and whether growth is held."""
        held = self._held_faults()
        live = {id(f) for f in self.sim.faults}
        for k in [k for k in held if k not in live]:
            del held[k]
        targets: dict[str, float] = {}
        for f in self.sim.faults:
            target = held[id(f)][0] if id(f) in held else f.severity
            targets[f.mode] = round(min(1.0, targets.get(f.mode, 0.0) + target), 3)
        return {"targets": targets, "held": bool(held)}


@dataclass
class SimulatedSource(FaultGrowth):
    """The P-101 pump model."""

    brain_path: Path
    sample_rate_hz: int = 25600
    window_samples: int = 16384
    time_scale: float = 60.0
    load: float = 0.65
    seed: int = 4242
    event_cooldown_s: int = 1800

    key: str = "simulated"
    vib_unit: str = "mm/s"
    equipment_id: str = "P-101"
    iso_class: str = "10816-1-III"
    family: str = "pump"
    supports_injection: bool = True
    sim: PumpSimulator = field(init=False)

    def __post_init__(self) -> None:
        self.sim = PumpSimulator(seed=self.seed, fs_vib=float(self.sample_rate_hz),
                                 load=self.load)
        self.sim.time_scale = self.time_scale
        self.dt = self.window_samples / float(self.sample_rate_hz)

    def next(self) -> SourceFrame:
        fr = self.sim.step(self.dt)
        return SourceFrame(
            t_s=fr.t_s, dt_s=self.dt * self.time_scale,
            vibration=fr.vibration.astype(np.float32), rpm=fr.rpm, load=self.load,
            slow=[fr.bearing_temp_c, fr.motor_temp_c, fr.discharge_bar,
                  fr.suction_bar, fr.flow_m3h, fr.current_a, 0.0],
            present=SLOW_PRESENT_ALL,
            sensors={
                "bearing_temp_c": round(fr.bearing_temp_c, 2),
                "motor_temp_c": round(fr.motor_temp_c, 2),
                "discharge_bar": round(fr.discharge_bar, 3),
                "suction_bar": round(fr.suction_bar, 3),
                "flow_m3h": round(fr.flow_m3h, 2),
                "current_a": round(fr.current_a, 2),
                "rpm": round(fr.rpm, 1),
            },
        )

    def info(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": "simulated", "real": False,
            "name": "PumpSimulator — synthetic centrifugal pump P-101",
            "detail": "Physics-parameterised signal model (sim/pump_sim.py): shaft "
                      "harmonics, bearing defect frequencies from the bearing "
                      "geometry, and the four fault signatures injected on "
                      "demand. Not recorded from a real machine.",
            "supports_injection": True, "vib_unit": self.vib_unit,
            "equipment_id": self.equipment_id,
            "standard": standards.describe(self.iso_class),
            "family": self.family,
            "fault_modes": list(families.class_names(self.family)[1:]),
        }

    def severities(self) -> dict[str, float]:
        return self.sim.severities()

    def inject(self, mode: str, severity: float, ramp_minutes: float) -> None:
        self.sim.inject(mode, severity=severity, onset_s=self.sim.t_s,
                        ramp_hours=ramp_minutes / 60.0)

    def clear_faults(self) -> None:
        self.sim.clear_faults()
        self._held_faults().clear()


# --- the simulated gearbox ----------------------------------------------------------


GEARBOX_KEY = "simulated_gearbox"


@dataclass
class SimulatedFamilySource(FaultGrowth):
    """A machine family's simulated reference machine (HARM-S7-T05/T10): the
    gearbox GB-201, the motor M-301, the fan F-401 — watched by the family's
    own brain and diagnosed from its own graph and manual. Everything that
    differs between them is data: the family's `demo` block in
    profiles/families.json and its simulator module (SIMULATOR, SENSORS)."""

    family: str = "gearbox"
    sample_rate_hz: int = 25600
    window_samples: int = 16384
    time_scale: float = 60.0
    seed: int = 4242
    event_cooldown_s: int = 1800
    supports_injection: bool = True

    def __post_init__(self) -> None:
        import importlib

        fam = families.get(self.family)
        self.demo = fam["demo"]
        self.key = str(self.demo["key"])
        self.vib_unit = str(self.demo["vib_unit"])
        self.equipment_id = str(self.demo["equipment_id"])
        self.iso_class = str(self.demo["iso_class"])
        mod = importlib.import_module(fam["simulator"])
        self.sensors: dict[str, int] = dict(mod.SENSORS)
        self.brain_path = families.path(self.family, "brain")
        self.sim = mod.SIMULATOR(seed=self.seed, fs_vib=float(self.sample_rate_hz))
        self.sim.time_scale = self.time_scale
        self.dt = self.window_samples / float(self.sample_rate_hz)

    def next(self) -> SourceFrame:
        fr = self.sim.step(self.dt)
        return SourceFrame(
            t_s=fr.t_s, dt_s=self.dt * self.time_scale,
            vibration=fr.vibration.astype(np.float32), rpm=fr.rpm, load=0.65,
            slow=list(fr.slow), present=fr.present,
            sensors={**{k: round(fr.slow[slot], 2) for k, slot in self.sensors.items()},
                     "rpm": round(fr.rpm, 1)},
        )

    def info(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": "simulated", "real": False,
            "name": self.demo["name"], "detail": self.demo["detail"],
            "supports_injection": True, "vib_unit": self.vib_unit,
            "equipment_id": self.equipment_id,
            "standard": standards.describe(self.iso_class),
            "family": self.family,
            "fault_modes": list(families.class_names(self.family)[1:]),
        }

    def severities(self) -> dict[str, float]:
        return dict(self.sim.severities())

    def inject(self, mode: str, severity: float, ramp_minutes: float) -> None:
        self.sim.inject(mode, severity=severity, onset_s=self.sim.t_s,
                        ramp_hours=ramp_minutes / 60.0)

    def clear_faults(self) -> None:
        self.sim.clear_faults()
        self._held_faults().clear()


def demo_families() -> dict[str, str]:
    """Source key -> family, for every family with a simulated reference machine."""
    return {str(f["demo"]["key"]): name for name, f in families.families().items()
            if f.get("demo")}


# --- a recording ------------------------------------------------------------------


@dataclass
class RecordedSource:
    """A recording replayed through a machine's own brain.

    Replays the snapshots in order with their real timestamps; each tick is
    one snapshot. It ends — a recording does not loop, because a failed
    bearing followed by a healthy one would be a lie.

    The built-in one is IMS test 2, bearing 1 (`ims_builtin`); the rest are
    machines commissioned through the console (`from_machine`).
    """

    key: str
    name: str
    detail: str
    brain_path: Path
    recording: Any                       # sim.datasets.recordings.Recording
    channel: int
    vib_unit: str
    equipment_id: str
    nominal_rpm: float
    badge: str = "Recorded data"
    real: bool = False
    iso_class: str = "none"             # a key of profiles/vibration_standards.json
    # Why a built-in machine has no zones, for the console (it cannot set them).
    iso_note: str = ""
    family: str = "pump"                # a key of profiles/families.json
    # One speed per snapshot, rpm; 0 = the engine estimates it. None = nominal.
    speeds: Any = None
    outcome: str | None = None
    # A steady CRITICAL is re-reported every 6 recorded hours; at the pump's
    # 30-minute cooldown a ten-minute snapshot rate would repeat every third tick.
    event_cooldown_s: int = 6 * 3600
    supports_injection: bool = False
    # The measurement point this source watches, and the machine it belongs
    # to (HARM-S7-T07); "" and the key itself for a one-point machine.
    point: str = ""
    machine: str = ""

    def __post_init__(self) -> None:
        from sim.datasets.recordings import window_for

        self.window_samples = window_for(self.recording.snapshot_samples())
        self.sample_rate_hz = self.recording.sample_rate_hz
        self.hours = self.recording.hours()
        self.at = self.recording.stamps()
        self.data = self.recording.load(self.channel, self.window_samples)
        if self.speeds is None:
            self.speeds = np.full(len(self.hours), float(self.nominal_rpm))
        self.i = 0

    @property
    def finished(self) -> bool:
        return self.i >= len(self.hours)

    def next(self) -> SourceFrame | None:
        if self.finished:
            return None
        i = self.i
        self.i += 1
        prev = self.hours[i - 1] if i else self.hours[i]
        return SourceFrame(
            t_s=float(self.hours[i]) * 3600.0,
            dt_s=max(float(self.hours[i] - prev) * 3600.0, 1e-6),
            vibration=self.data[i], rpm=float(self.speeds[i]), load=0.65,
            slow=[0.0] * 7, present=0,
            # Only vibration was recorded. Nothing else is shown, rather than
            # a plausible-looking number.
            sensors={"bearing_temp_c": None, "motor_temp_c": None,
                     "discharge_bar": None, "suction_bar": None, "flow_m3h": None,
                     "current_a": None, "rpm": None},   # filled from the engine
            recorded_at=self.at[i],
        )

    def info(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": "recorded", "real": self.real, "badge": self.badge,
            "name": self.name, "detail": self.detail,
            "supports_injection": False, "vib_unit": self.vib_unit,
            "equipment_id": self.equipment_id,
            "standard": standards.describe(self.iso_class),
            "iso_note": self.iso_note,
            "progress": {"snapshot": self.i, "of": len(self.hours),
                         "finished": self.finished},
            "outcome": self.outcome,
            "family": self.family,
            "fault_modes": list(families.class_names(self.family)[1:]),
            "point": self.point, "machine": self.machine or self.key,
        }

    def severities(self) -> dict[str, float]:
        return {}


# --- a live sensor ----------------------------------------------------------------


@dataclass
class LiveSource:
    """A commissioned machine watched live: snapshots from an MQTT topic
    (HARM-S7-T08), through the brain commissioned on its own healthy running.

    Each tick takes the oldest snapshot waiting, or nothing — the engine sees
    the machine at the rate the gateway sends it, and its clock is the
    snapshots' own timestamps. A snapshot of another rate or too short for the
    commissioned window is refused and counted.
    """

    key: str
    name: str
    brain_path: Path
    broker: str
    topic: str
    window_samples: int
    sample_rate_hz: int
    vib_unit: str
    equipment_id: str
    nominal_rpm: float
    family: str = "pump"
    iso_class: str = "none"
    speed: str = "constant"
    event_cooldown_s: int = 1800
    supports_injection: bool = False
    backlog: int = 64

    def __post_init__(self) -> None:
        from collections import deque

        from api import adapters

        self._queue: Any = deque(maxlen=self.backlog)
        self._lock = threading.Lock()
        self.refused = 0
        self.dropped = 0
        self._t0: float | None = None
        self._prev: float | None = None
        self.subscriber = adapters.subscriber(self.broker, self.topic, self._on_snapshot,
                                              consumer=f"machine-{self.key}")
        self.subscriber.start()

    def _on_snapshot(self, snap: Any) -> None:
        if snap.sample_rate_hz != self.sample_rate_hz or len(snap.samples) < self.window_samples:
            self.refused += 1
            return
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1          # the engine fell behind; oldest goes
            self._queue.append(snap)

    def next(self) -> SourceFrame | None:
        with self._lock:
            if not self._queue:
                return None
            snap = self._queue.popleft()
        if self._t0 is None:
            self._t0 = self._prev = snap.t
        dt = max(snap.t - (self._prev or snap.t), 1e-6)
        self._prev = snap.t
        rpm = (snap.rpm if self.speed == "measured" and snap.rpm
               else 0.0 if self.speed == "estimated" else self.nominal_rpm)
        return SourceFrame(
            t_s=snap.t - self._t0, dt_s=dt,
            vibration=np.ascontiguousarray(snap.samples[: self.window_samples]),
            rpm=float(rpm), load=0.65, slow=list(snap.slow), present=snap.present,
            sensors={"rpm": None},        # filled from the engine
            recorded_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(snap.t)),
        )

    def info(self) -> dict[str, Any]:
        live = self.subscriber.status()
        return {
            "key": self.key, "kind": "live", "real": True, "badge": "Live sensor",
            "name": self.name,
            "detail": f"Live from {live['broker']}, topic {live['topic']}, "
                      "through the brain commissioned on this machine's own healthy "
                      "running.",
            "supports_injection": False, "vib_unit": self.vib_unit,
            "equipment_id": self.equipment_id,
            "standard": standards.describe(self.iso_class),
            "family": self.family,
            "fault_modes": list(families.class_names(self.family)[1:]),
            "live": {**live, "refused": self.refused, "dropped": self.dropped,
                     "waiting": len(self._queue)},
            "point": "", "machine": self.key,
        }

    def severities(self) -> dict[str, float]:
        return {}

    def close(self) -> None:
        self.subscriber.stop()


IMS_KEY = "ims_test2_b1"
IMS_BRAIN = ARTIFACTS / "ims_test2_b1.harmonic"


def ims_brain() -> Path:
    """The rig's brain: built here, or from the exported bundle on Google Cloud."""
    from sim.datasets.recordings import ims_bundle_dir

    if IMS_BRAIN.exists():
        return IMS_BRAIN
    return ims_bundle_dir(2) / "ims_test2_b1.harmonic"


def ims_builtin() -> RecordedSource:
    from sim.datasets.recordings import ImsRecording

    return RecordedSource(
        key=IMS_KEY, name="IMS bearing run-to-failure, test 2, bearing 1",
        detail="Real accelerometer recordings from the IMS / University of "
               "Cincinnati bearing rig (NASA Prognostics Data Repository): "
               "Rexnord ZA-2115 bearings at 2000 rpm, one 1-second snapshot "
               "every 10 minutes from 12 to 19 Feb 2004, ending in an outer "
               "race failure. Vibration only. The engine's fault classes "
               "were trained on the pump simulator and have not seen this "
               "rig; its baseline was learned on the rig's first 50 hours.",
        brain_path=ims_brain(), recording=ImsRecording(), channel=0, vib_unit="g",
        equipment_id="IMS-RIG", nominal_rpm=2000.0, badge="Real recording", real=True,
        outcome="outer race failure at the end of the test",
        iso_note="The rig's ISO class is unknown: ISO 10816 classes go by machine size and "
                 "power, and the IMS documentation gives no motor power. Its alarms come "
                 "from what the engine learned on the rig's first 50 hours.")


def from_machine(m: Any, point: int = 0) -> RecordedSource | LiveSource:
    """A machine commissioned through the console (`api.machines.Machine`):
    watched live when it was commissioned to keep listening, replayed from its
    recording otherwise. A machine with several measurement points has one
    source per point, each on its own channel and brain."""
    from api import machines

    pts = m.point_list()
    pt = pts[point]
    multi = len(pts) > 1

    if m.live:
        return LiveSource(
            key=m.id, name=m.name, brain_path=m.brain_path, broker=m.live["broker"],
            topic=m.live["topic"], window_samples=int(m.report["window_samples"]),
            sample_rate_hz=int(m.report["sample_rate_hz"]), vib_unit=m.vib_unit,
            equipment_id=m.equipment_id, nominal_rpm=m.nominal_rpm, family=m.family,
            iso_class=m.iso_class, speed=m.speed)

    r = machines.recording(m.recording)
    real = m.recording.startswith("ims:")
    channel = int(pt["channel"])
    return RecordedSource(
        key=m.point_key(point), name=f"{m.name} · {pt['name']}" if multi else m.name,
        detail=f"Commissioned in the console on its own healthy running "
               f"({m.baseline_h[0]:.1f}-{m.baseline_h[1]:.1f} h of {r.name}, "
               f"channel {channel + 1}). {r.provenance}.",
        point=pt["name"] if multi else "", machine=m.id,
        brain_path=m.point_brain(point), recording=r, channel=channel, vib_unit=m.vib_unit,
        equipment_id=m.equipment_id, nominal_rpm=m.nominal_rpm, iso_class=m.iso_class,
        family=m.family,
        speeds=machines.speeds_for(m, r),
        badge="Real recording" if real else "Recorded data", real=real)


def build(key: str) -> RecordedSource | SimulatedFamilySource | LiveSource:
    demos = demo_families()
    if key in demos:
        return SimulatedFamilySource(family=demos[key])
    if key == IMS_KEY:
        return ims_builtin()
    from api import machines

    mid, _, idx = key.partition("@")
    m = machines.get(mid)
    if m is None or not (m.status == "ready" or (m.status == "commissioning" and m.baselines)):
        raise ValueError(f"unknown or uncommissioned machine {key!r}")
    point = int(idx) if idx.isdigit() else 0
    if not 0 <= point < len(m.point_list()):
        raise ValueError(f"machine {mid!r} has no point {idx}")
    return from_machine(m, point)


def available() -> dict[str, dict[str, Any]]:
    """What can be selected, and whether its data and brain are present."""
    from api import machines
    from sim.datasets.recordings import ims_bundle_dir

    ims_data = ((REPO / "datasets" / "ims" / "test2" / "2nd_test").is_dir()
                or (ims_bundle_dir(2) / "stamps.json").exists())
    reason = None
    if not ims_data:
        reason = "IMS data not extracted (see sim/datasets/ims_loader.py)"
    elif not ims_brain().exists():
        reason = "IMS brain not built (python scripts/evaluate_ims.py)"
    out: dict[str, dict[str, Any]] = {
        "simulated": {"label": "Simulated pump P-101", "available": True, "kind": "simulated"},
        IMS_KEY: {"label": "IMS bearing rig (real)", "available": reason is None,
                  "reason": reason, "kind": "recorded"},
    }
    for key, fam_name in demo_families().items():
        built = families.path(fam_name, "brain").exists()
        out[key] = {"label": families.get(fam_name)["demo"]["label"], "available": built,
                    "reason": None if built else
                    f"{fam_name} brain not built (python scripts/build_brain.py --family {fam_name})",
                    "kind": "simulated"}
    for m in machines.list_machines():
        # A machine re-learning normal (HARM-S7-T09) keeps running on the
        # brain it had until the new one is proven.
        if m.status == "ready" or (m.status == "commissioning" and m.baselines):
            pts = m.point_list()
            for i, pt in enumerate(pts):
                ok = m.point_brain(i).exists()
                out[m.point_key(i)] = {
                    "label": f"{m.name} · {pt['name']}" if len(pts) > 1 else m.name,
                    "available": ok, "reason": None if ok else "brain missing",
                    "kind": "commissioned", "machine": m.id,
                    "point": pt["name"] if len(pts) > 1 else ""}
    return out
