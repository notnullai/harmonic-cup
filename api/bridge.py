"""The engine bridge: frames in, engine ticks out, events on the queue.

Owns the two stateful things the service needs and nothing else: a data
source producing frames (`api/sources.py` — the pump simulator, or a real
recording) and a `HarmonicRig` consuming them. One background thread, one
lock, a bounded history and a bounded event log.

The thread exists because the engine is stateful and its ticks are ordered.
Running it inside a request handler would mean two requests interleaving ticks,
which is not a race that shows up as an error — it shows up as a health index
that wanders.

**Nothing here recomputes what the engine already decided.** The health state,
the index, the RUL and the decision to raise an event all come out of the C
side. The bridge names things (class index to class name, feature kind to
string) and keeps the history. Where it was tempting to threshold something in
Python, it asks the engine instead.

When the shared library is missing the bridge does not start and says so.
Serving zeros would render as a perfectly healthy machine.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api import sources as sources_mod
from api.rig import HarmonicRig, RigUnavailable, Tick
from api.sources import SimulatedSource, Source, SourceFrame
from kg import families
from sim.pump_sim import FAULT_MODES

REPO = Path(__file__).resolve().parents[1]

# Index into the pump brain's class-score vector. Index 0 is healthy and
# carries no weight in the index; the rest are the trained fault classes in the
# order `sim/cli.py::CLASS_NAMES` writes them into the training labels. Other
# machine families name their classes in profiles/families.json, and the
# bridge reads the source's family (`_class_names`).
CLASS_NAMES: tuple[str, ...] = ("healthy", *FAULT_MODES)

DEFAULT_BRAIN = REPO / "sim" / "artifacts" / "pump_p101.harmonic"
DEFAULT_WINDOW = 16384
DEFAULT_SAMPLE_RATE = 25600

# How much machine time one frame stands for: a frame is `window / fs` of
# signal (0.64 s at 16384 samples and 25.6 kHz) and advances the machine clock
# by that times this. At 60 a frame is ~38 machine seconds, so a degradation
# that takes hours on a real pump plays out in minutes. It is the compression
# the scenario files already use.
DEFAULT_TIME_SCALE = 60.0

# Wall-clock seconds between frames. The engine needs a fixed number of ticks
# to confirm a fault — its class score fills over a 32-tick window, so even an
# instant full-severity fault is 8 ticks from WARNING — and that count is a
# property of the frozen brain, not of this service. At one frame a second the
# demo took 8 s to say anything after a click; at 0.25 s it takes 2. A frame
# costs ~3 ms to simulate and tick, so the engine is nowhere near its limit.
DEFAULT_PERIOD_S = float(os.environ.get("HARMONIC_PERIOD_S", "0.25"))

# How long a steady state stays quiet before it is reported again, in machine
# seconds (the guard runs on the frame timestamps). It was 60, which is under
# two frames at this compression: a pump sitting in CRITICAL raised an event,
# and a diagnosis, every other tick — 140 of them a minute. Half a machine hour
# is ~47 frames. An escalation is reported at once whatever this says; the
# guard lets a worse state through the cooldown.
DEFAULT_EVENT_COOLDOWN_S = 1800

# Upper bound on the fast-forward through warm-up. The brain says how long its
# warm-up is; this only stops a brain that never warms from hanging a start.
MAX_WARM_UP_TICKS = 2000

# The timeline can be scrubbed back through all of it: the whole IMS
# recording (984 snapshots), and about 17 minutes of the simulated pump at
# four frames a second.
HISTORY = 4000
EVENT_LOG = 200



# Alarm states in order, for the demo's "pause the fault at" (EngineBridge.pause_at).
_RANK = {"OK": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3, "FAILED": 4}


@dataclass
class Snapshot:
    """One tick, in the shape the API and the dashboard consume."""

    tick: int
    ts: float
    machine_hours: float
    state: str
    iso_state: str
    regime: str
    warm: bool
    hhi: float
    hhi_fast: float
    hhi_slow: float
    slope_per_tick: float
    rul_hours: float
    # RMS velocity in mm/s, the unit ISO zones are written in: the channel
    # itself for a velocity sensor, integrated over 10-1000 Hz by the engine
    # for an accelerometer. `vib_rms` is the channel's RMS in its own unit.
    vib_rms_mm_s: float | None
    vib_rms: float
    vib_unit: str
    rpm: float
    load: float
    sensors: dict[str, float | None]
    features: dict[str, float]
    surprise: dict[str, float]
    class_scores: dict[str, float]
    top_classes: list[dict[str, Any]]
    top_surprised_features: list[dict[str, Any]]
    injected: dict[str, float]
    recorded_at: str | None = None
    # The state stands on an unrecognised change: the machine has moved from
    # its baseline, and no trained class and no ISO zone accounts for the
    # level. Far on a few features (novelty), or a little on many together
    # (`spread`).
    unrecognised: bool = False
    # A fault recognised early: the classes are sure, no measurement is
    # severe yet, so the state is held at WARNING.
    early: bool = False
    # The spread floor raised the state: many features moved a little
    # together, each under its own threshold. Always unrecognised too.
    spread: bool = False

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Event:
    """A raised alarm, plus the diagnosis once it arrives."""

    id: str
    tick: int
    ts: float
    state: str
    hhi: float
    rul_hours: float
    regime: str
    vib_rms_mm_s: float | None
    top_classes: list[dict[str, Any]]
    top_surprised_features: list[dict[str, Any]]
    equipment_id: str = "P-101"
    # When it happened on the machine's clock, and the recording's own
    # timestamp for a recording: what the timeline places it by.
    machine_hours: float = 0.0
    recorded_at: str | None = None
    vib_rms: float | None = None
    vib_unit: str = "mm/s"
    diagnosis: dict[str, Any] | None = None
    diagnosis_status: str = "pending"
    diagnosis_provider: str | None = None
    diagnosis_error: str | None = None
    # What the technician found (HARM-S7-T12): verdict, the actual fault when
    # the diagnosis was wrong, a note. Kept in the event store as well.
    feedback: dict[str, Any] | None = None
    # The work order this event opened or joined (HARM-S7-T14): id, delivery.
    work_order: dict[str, Any] | None = None
    # One excursion from OK back to OK. An alarm is re-raised every cooldown
    # while a fault lasts; the diagnosis is made once per episode and fault,
    # and again on escalation (api/genai_hook.py).
    episode: str = ""
    # Set when this event carries the diagnosis made for an earlier one.
    diagnosis_of: str | None = None
    # Where on the machine it was measured (HARM-S7-T07); "" for one point.
    point: str = ""
    # Raised on an unrecognised change: no trained class accounts for the level.
    unrecognised: bool = False
    # Raised on a fault recognised early: sure, not yet severe.
    early: bool = False
    # Raised by the spread floor: many features moved a little together.
    spread: bool = False

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def as_engine_event(self) -> dict[str, Any]:
        """The shape `kg.query.query_for_event` and `rag.pipeline` expect."""
        return {
            "event_id": self.id,
            "equipment_id": self.equipment_id,
            "regime": self.regime,
            "hhi": round(self.hhi, 4),
            "state": self.state,
            "rul_hours": round(self.rul_hours, 1),
            "top_classes": self.top_classes,
            "top_surprised_features": self.top_surprised_features,
            **({"point": self.point} if self.point else {}),
            **({"unrecognised": True} if self.unrecognised else {}),
            **({"early": True} if self.early else {}),
            **({"spread": True} if self.spread else {}),
        }


class EngineBridge:
    """A data source and the engine, running together on one thread."""

    def __init__(self, brain_path: str | Path = DEFAULT_BRAIN, *,
                 window_samples: int = DEFAULT_WINDOW,
                 sample_rate_hz: int = DEFAULT_SAMPLE_RATE,
                 time_scale: float = DEFAULT_TIME_SCALE,
                 period_s: float = DEFAULT_PERIOD_S,
                 load: float = 0.65, seed: int = 4242,
                 event_cooldown_sec: int = DEFAULT_EVENT_COOLDOWN_S,
                 source: str = "simulated",
                 on_event: Callable[[Event], object] | None = None):
        # The arguments describe the simulated pump; a recording brings its
        # own brain, rate and cooldown.
        self._sim_args: dict[str, Any] = {
            "brain_path": Path(brain_path), "window_samples": window_samples,
            "sample_rate_hz": sample_rate_hz, "time_scale": time_scale,
            "load": load, "seed": seed, "event_cooldown_s": event_cooldown_sec,
        }
        self.window_samples = window_samples
        self.period_s = period_s
        self.load = load
        self.on_event = on_event
        self.source_key = source
        self.rig, self.source = self._build(source)
        self.last_frame: SourceFrame | None = None
        # The demo's "pause at": an injected fault stops growing once the
        # machine reaches this state, so there is time to act on it.
        self.pause_at: str | None = None
        self.paused_at: str | None = None

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: deque[Snapshot] = deque(maxlen=HISTORY)
        self._events: deque[Event] = deque(maxlen=EVENT_LOG)
        self._latest: Snapshot | None = None
        self._counter = itertools.count(1)
        self._episode = ""
        self._episode_open = False
        # A new brain restarts the engine's tick at 0; the timeline's ticks
        # continue from here. And the moments the brain changed, for the
        # timeline to mark (HARM-S7-T09).
        self._tick_base = 0
        self.marks: list[dict[str, Any]] = []

    def _build(self, key: str) -> tuple[HarmonicRig, Source]:
        source: Source = (SimulatedSource(**self._sim_args) if key == "simulated"
                          else sources_mod.build(key))
        rig = HarmonicRig(source.brain_path, window_samples=source.window_samples,
                          sample_rate_hz=source.sample_rate_hz,
                          event_cooldown_sec=source.event_cooldown_s)
        return rig, source

    # -- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        # Fast-forward the warm-up, so the service comes up already watching
        # rather than spending its first minute unable to say anything.
        self.warm_up()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="harmonic-engine",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self.rig.close()
        _close(self.source)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the loop

    def _run(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            self.step()
            next_at += self.period_s
            # Sleep to the next deadline rather than for a fixed period, so a
            # slow tick does not make the stream drift late for ever.
            delay = next_at - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_at = time.monotonic()

    def warm_up(self) -> int:
        """Tick as fast as the machine allows until the engine is warm.

        The ticks are real: the same frames and the same engine, only not
        paced to the wall clock. Returns how many it took.
        """
        for n in range(1, MAX_WARM_UP_TICKS + 1):
            snap = self.step()
            if snap is None or snap.warm:
                return n
        return MAX_WARM_UP_TICKS

    def step(self) -> Snapshot | None:
        """One frame. Public so tests can drive the bridge without a thread.

        Returns None once a recording has run out: the last reading stays up
        and nothing more is ticked.
        """
        # The source and the engine are advanced under the lock: `inject` and
        # `reset` change them, and a reset that swapped the engine out halfway
        # through a frame would feed one engine and snapshot another.
        with self._lock:
            frame = self.source.next()
            if frame is None:
                return None
            tick = self.rig.feed(
                frame.vibration, rpm=frame.rpm, load=frame.load, slow=frame.slow,
                timestamp_us=int(frame.t_s * 1e6), dt_us=int(frame.dt_s * 1e6),
                present=frame.present,
            )
            snapshot = self._snapshot(frame, tick)
            # The frame that produced this reading: an event's raw window is
            # what a technician's verdict turns into training data.
            self.last_frame = frame
            self._latest = snapshot
            self._history.append(snapshot)
            event = self._maybe_event(tick, snapshot) if tick.event else None
            if snapshot.state == "OK":
                self._episode_open = False
            if (self.pause_at and _RANK.get(snapshot.state, 0) >= _RANK[self.pause_at]
                    and hasattr(self.source, "hold_faults")):
                self.source.hold_faults()
                self.paused_at, self.pause_at = snapshot.state, None
        # The hook runs outside the lock: it calls the LLM, and holding the
        # lock across a network round trip would stall every reader.
        if event is not None and self.on_event is not None:
            self.on_event(event)
        return snapshot

    @property
    def family(self) -> str:
        return str(getattr(self.source, "family", None) or families.DEFAULT)

    def _class_names(self) -> tuple[str, ...]:
        """What each class neuron of the source's brain means. A family whose
        list does not match the brain's class count is not trusted: the
        neurons are then reported by index rather than under wrong names."""
        names = families.class_names(getattr(self.source, "family", None))
        if len(names) != self.rig.class_count:
            return tuple(f"class_{i}" for i in range(self.rig.class_count))
        return names

    def _snapshot(self, frame: SourceFrame, tick: Tick) -> Snapshot:
        names = self._class_names()
        scores = {names[i] if i < len(names) else f"class_{i}": s
                  for i, s in enumerate(tick.class_scores)}
        # Class 0 is healthy and is not a fault; ranking it would put "healthy"
        # at the top of a diagnosis every time the machine is fine, which is
        # true and useless.
        faults = {k: v for k, v in scores.items() if k != "healthy"}
        top = sorted(faults.items(), key=lambda kv: -kv[1])[:3]
        unit = self.source.vib_unit
        # The speed the engine analysed at: the frame's, or its own estimate
        # when the frame carried none.
        rpm = round(tick.rpm if tick.rpm > 0 else frame.rpm, 1)
        # The rig reports the RMS of whatever the channel carries, and the
        # velocity the ISO zones are judged on.
        vib = round(tick.vib_rms_mm_s, 4)
        velocity = (None if tick.vib_velocity_mm_s is None
                    else round(tick.vib_velocity_mm_s, 4))
        return Snapshot(
            tick=tick.tick + self._tick_base,
            ts=time.time(),
            machine_hours=round(frame.t_s / 3600.0, 4),
            state=tick.state,
            iso_state=tick.iso_state,
            regime=tick.regime,
            warm=tick.warm,
            hhi=round(tick.hhi, 4),
            hhi_fast=round(tick.hhi_fast, 4),
            hhi_slow=round(tick.hhi_slow, 4),
            slope_per_tick=round(tick.slope_per_tick, 6),
            rul_hours=round(tick.rul_hours, 1),
            vib_rms_mm_s=velocity,
            vib_rms=vib,
            vib_unit=unit,
            rpm=rpm,
            load=frame.load,
            sensors={"vib_rms_mm_s": velocity, "vib_rms": vib, **frame.sensors,
                     "rpm": rpm},
            features={k: round(v, 4) for k, v in tick.features.items()},
            surprise={k: round(v, 4) for k, v in tick.surprise.items()},
            class_scores={k: round(v, 4) for k, v in scores.items()},
            top_classes=[{"name": k, "score": round(v, 4)} for k, v in top if v > 0.0],
            top_surprised_features=tick.top_surprised(),
            injected={m: round(s, 3) for m, s in self.source.severities().items() if s > 0},
            recorded_at=frame.recorded_at,
            unrecognised=tick.unrecognised,
            early=tick.early,
            spread=tick.spread,
        )

    def _maybe_event(self, tick: Tick, snap: Snapshot) -> Event | None:
        """Build an event. The guard already decided there should be one.

        The rate limiting lives in the C guard, so this does not second-guess
        it: a cooldown in two places is a cooldown nobody can reason about.
        """
        if not self._episode_open:
            self._episode = f"ep-{uuid.uuid4().hex[:10]}"
            self._episode_open = True
        event = Event(
            id=f"ev-{uuid.uuid4().hex[:10]}-{next(self._counter)}", episode=self._episode,
            point=str(getattr(self.source, "point", "") or ""),
            unrecognised=snap.unrecognised,
            early=snap.early,
            spread=snap.spread,
            tick=snap.tick, ts=snap.ts, state=snap.state, hhi=snap.hhi,
            rul_hours=snap.rul_hours, regime=snap.regime,
            vib_rms_mm_s=snap.vib_rms_mm_s, equipment_id=self.source.equipment_id,
            machine_hours=snap.machine_hours, recorded_at=snap.recorded_at,
            vib_rms=snap.vib_rms, vib_unit=snap.vib_unit,
            top_classes=list(snap.top_classes),
            top_surprised_features=list(snap.top_surprised_features),
        )
        self._events.append(event)
        return event

    # -- fault injection, for the demo

    def _injectable(self) -> Any:
        if not getattr(self.source, "supports_injection", False):
            raise ValueError("this is a recording of a real machine; nothing can "
                             "be injected into it")
        return self.source

    def inject(self, mode: str, severity: float = 1.0,
               ramp_minutes: float = 5.0, pause_at: str | None = None) -> dict[str, Any]:
        modes = families.class_names(getattr(self.source, "family", None))[1:]
        if mode not in modes:
            raise ValueError(f"unknown fault mode {mode!r}; known: {modes}")
        with self._lock:
            src = self._injectable()
            src.inject(mode, severity, ramp_minutes)
            if pause_at is not None and pause_at not in _RANK:
                raise ValueError(f"unknown state {pause_at!r}")
            self.pause_at, self.paused_at = pause_at, None
            t_s = src.sim.t_s
        return {"injected": mode, "severity": severity, "ramp_minutes": ramp_minutes,
                "at_machine_hours": round(t_s / 3600.0, 4)}

    def clear_faults(self) -> dict[str, Any]:
        with self._lock:
            self._injectable().clear_faults()
            self.pause_at = self.paused_at = None
        return {"cleared": True}

    def hold_faults(self, hold: bool) -> dict[str, Any]:
        """Pause or resume the injected faults' growth (sources.FaultGrowth)."""
        with self._lock:
            src = self._injectable()
            if hold:
                src.hold_faults()
            else:
                src.resume_faults()
                self.pause_at = self.paused_at = None
            return src.growth()

    def fault_growth(self) -> dict[str, Any] | None:
        """For a simulator: targets, held, and how much faster than real time
        machine time runs here."""
        src = self.source
        if not getattr(src, "supports_injection", False) or not hasattr(src, "growth"):
            return None
        with self._lock:
            g = src.growth()
        per_tick = src.window_samples / float(src.sample_rate_hz) * float(getattr(src, "time_scale", 1.0))
        g["speedup"] = round(per_tick / self.period_s, 1) if self.period_s > 0 else None
        g["pause_at"], g["paused_at"] = self.pause_at, self.paused_at
        return g

    def swap_brain(self, label: str) -> int:
        """Put the source's brain — just re-commissioned — into service
        without restarting the machine: the source carries on where it is, the
        history and the events stay, and the moment is marked. The new engine
        warms up on the next readings. Returns the warm-up ticks."""
        rig = HarmonicRig(self.source.brain_path, window_samples=self.source.window_samples,
                          sample_rate_hz=self.source.sample_rate_hz,
                          event_cooldown_sec=self.source.event_cooldown_s)
        with self._lock:
            old = self.rig
            last = self._latest
            self._tick_base = last.tick if last is not None else self._tick_base
            self.rig = rig
            self._episode_open = False
            if last is not None:
                self.marks.append({"tick": last.tick, "h": last.machine_hours,
                                   "at": last.recorded_at, "label": label})
            old.close()
        return self.warm_up()

    def reset(self, source: str | None = None) -> dict[str, Any]:
        """Start the machine and the engine again from nothing — or switch to
        another data source, which is the same operation with a different
        source.

        Clearing a fault is not enough to undo one: the engine's predictors
        have been learning the faulty machine while it ran, and its health
        averages carry the excursion. A restart builds a fresh engine from the
        source's brain and a fresh source (the same seed, or the recording from
        its first snapshot), drops the history and the events, and
        fast-forwards the warm-up — so the second run of a demo starts where
        the first one did.
        """
        key = source or self.source_key
        rig, src = self._build(key)
        with self._lock:
            old, old_src = self.rig, self.source
            self.rig, self.source, self.source_key = rig, src, key
            self._history.clear()
            self._events.clear()
            self._latest = None
            self._episode_open = False
            self._tick_base = 0
            self.marks = []
            old.close()
        _close(old_src)
        warm_after = self.warm_up()
        return {"reset": True, "source": key, "warm_after_ticks": warm_after}

    # -- reads

    @property
    def latest(self) -> Snapshot | None:
        with self._lock:
            return self._latest

    def history(self, n: int = 120) -> list[Snapshot]:
        with self._lock:
            return list(self._history)[-n:]

    def events(self, n: int = 20) -> list[Event]:
        with self._lock:
            return list(self._events)[-n:]

    def event(self, event_id: str) -> Event | None:
        with self._lock:
            for e in reversed(self._events):
                if e.id == event_id:
                    return e
        return None

    def info(self) -> dict[str, Any]:
        return {
            "brain": str(self.rig.brain_path),
            "window_samples": self.rig.window_samples,
            "sample_rate_hz": self.rig.sample_rate_hz,
            "feature_count": self.rig.feature_count,
            "class_count": self.rig.class_count,
            "commissioned": self.rig.commissioned,
            "nominal_rpm": self.rig.nominal_rpm,
            "feature_kinds": list(self.rig.feature_kinds),
            "classes": list(self._class_names()),
            "source": self.source.info(),
            "period_s": self.period_s,
            "running": self.running,
        }


@dataclass
class BridgeStatus:
    """What the API reports when the engine could not be started.

    An explicit unavailable state, not zeros: a health index of 0.0 renders as
    a machine in perfect condition, which is the most dangerous thing this
    system could display.
    """

    available: bool
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _close(source: Any) -> None:
    """A live source holds a network subscription; the others hold nothing."""
    close = getattr(source, "close", None)
    if callable(close):
        close()


def try_start(brain_path: str | Path = DEFAULT_BRAIN,
              on_event: Callable[[Event], object] | None = None,
              **kwargs: Any) -> tuple[EngineBridge | None, BridgeStatus]:
    """Start the bridge, or explain why not."""
    try:
        bridge = EngineBridge(brain_path, on_event=on_event, **kwargs)
    except RigUnavailable as exc:
        return None, BridgeStatus(False, str(exc))
    except (OSError, ValueError) as exc:
        return None, BridgeStatus(False, f"{type(exc).__name__}: {exc}")
    bridge.start()
    return bridge, BridgeStatus(True, detail=bridge.info())
