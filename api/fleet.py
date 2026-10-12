"""Every machine at once (HARM-S7-T13).

One engine bridge per machine, each on its own thread, the way each machine
would have its own controller. The console looks at one of them — the current
one — and the fleet page at all of them. Selecting a machine changes what is
looked at, not what is watched: the others keep running.

Every event is written to the event store with the window that raised it, and
updated when its diagnosis arrives.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from api import sources as sources_mod
from api.bridge import BridgeStatus, EngineBridge, Event
from api.rig import RigUnavailable
from api.store import EventStore

log = logging.getLogger("harmonic.fleet")

# "all" watches every available machine from start-up; "current" only the one
# on screen, starting others when they are selected.
FLEET_MODE = os.environ.get("HARMONIC_FLEET", "all")
STATE_RANK = {"FAILED": 4, "CRITICAL": 3, "WARNING": 2, "WATCH": 1, "OK": 0}


class Fleet:
    def __init__(self, store: EventStore, submit: Callable[[Event], object] | None,
                 **bridge_kwargs: Any) -> None:
        self.store = store
        self.submit = submit
        self.bridge_kwargs = bridge_kwargs
        self.bridges: dict[str, EngineBridge] = {}
        self.current_key: str | None = None
        self.status = BridgeStatus(False, "not started")
        self._lock = threading.RLock()
        # Work orders to a maintenance system, when one is configured
        # (api/workorders.py); set by the API.
        self.workorders: Any = None

    # -- lifecycle

    def start(self, first: str = "simulated") -> EngineBridge | None:
        try:
            self.select(first)
        except (RigUnavailable, OSError, ValueError) as exc:
            self.status = BridgeStatus(False, f"{type(exc).__name__}: {exc}"
                                       if not isinstance(exc, RigUnavailable) else str(exc))
            return None
        current = self.bridges[first]
        self.status = BridgeStatus(True, detail=current.info())
        if FLEET_MODE == "all":
            threading.Thread(target=self._start_rest, name="harmonic-fleet",
                             daemon=True).start()
        return self.current

    def _start_rest(self) -> None:
        for key, info in sources_mod.available().items():
            if info.get("available") and key not in self.bridges:
                try:
                    self._bridge(key)
                except Exception as exc:          # one bad machine must not stop the rest
                    log.warning("could not start %s: %s", key, exc)

    def stop(self) -> None:
        with self._lock:
            for b in self.bridges.values():
                b.stop()
            self.bridges.clear()

    # -- which machine

    def _bridge(self, key: str) -> EngineBridge:
        with self._lock:
            b = self.bridges.get(key)
            if b is None:
                b = EngineBridge(source=key, **self.bridge_kwargs)
                b.on_event = self._on_event_for(b)
                self.bridges[key] = b
                b.start()
            return b

    def select(self, key: str) -> EngineBridge:
        """Look at `key`, starting its engine if it is not running."""
        b = self._bridge(key)
        self.current_key = key
        return b

    def reset(self, key: str) -> dict[str, Any]:
        """Restart one machine from nothing, and look at it."""
        b = self.select(key)
        return b.reset(key)

    def swap_brain(self, machine: str, label: str, iso_class: str | None = None) -> int:
        """A machine re-commissioned in place: each of its points' engines
        takes its new brain, and its source the alarm zones it was given.
        Returns how many were swapped."""
        with self._lock:
            found = [b for k, b in self.bridges.items()
                     if k == machine or k.startswith(machine + "@")]
        for b in found:
            if iso_class is not None and hasattr(b.source, "iso_class"):
                b.source.iso_class = iso_class
            b.swap_brain(label)
        return len(found)

    def drop(self, key: str) -> None:
        """A machine that no longer exists (deleted), with all its points."""
        with self._lock:
            gone = [self.bridges.pop(k) for k in list(self.bridges)
                    if k == key or k.startswith(key + "@")]
        for b in gone:
            b.stop()
        if self.current_key is not None and (self.current_key == key
                                             or self.current_key.startswith(key + "@")):
            self.current_key = "simulated" if "simulated" in self.bridges else next(
                iter(self.bridges), None)

    @property
    def current(self) -> EngineBridge | None:
        return self.bridges.get(self.current_key or "")

    def find_event(self, event_id: str) -> Event | None:
        for b in list(self.bridges.values()):
            e = b.event(event_id)
            if e is not None:
                return e
        return None

    # -- events

    def _on_event_for(self, b: EngineBridge) -> Callable[[Event], object]:
        def on_event(event: Event) -> object:
            try:
                self.store.save(event.to_json(), b.source_key, b.family,
                                b.last_frame, b.source.sample_rate_hz)
            except Exception as exc:
                log.warning("could not store event %s: %s", event.id, exc)
            return self.submit(event) if self.submit is not None else None
        return on_event

    def diagnosed(self, event: Event) -> None:
        """The hook's callback: the stored event gets its diagnosis, and a
        work order when it warrants one."""
        if self.workorders is not None:
            key = next((k for k, b in list(self.bridges.items()) if b.event(event.id)), None)
            if key is not None:
                try:
                    name = self.bridges[key].source.info().get("name", key)
                    wo = self.workorders.consider(event.to_json(), key, str(name))
                    if wo is not None:
                        event.work_order = {"id": wo["id"], "delivery": wo["delivery"],
                                            "update": wo.get("update", False),
                                            "repeat": wo.get("repeat", False)}
                except Exception as exc:
                    log.warning("work order for %s failed: %s", event.id, exc)
        try:
            self.store.update(event.to_json())
        except Exception as exc:
            log.warning("could not update event %s: %s", event.id, exc)

    # -- the fleet page

    def summary(self) -> list[dict[str, Any]]:
        counts = self.store.counts()
        out = []
        for key, avail in sources_mod.available().items():
            b = self.bridges.get(key)
            row: dict[str, Any] = {"key": key, "label": avail["label"],
                                   "kind": avail.get("kind"),
                                   "available": avail.get("available", False),
                                   "reason": avail.get("reason"), "running": b is not None,
                                   "current": key == self.current_key,
                                   "machine": avail.get("machine", key),
                                   "point": avail.get("point", ""),
                                   "stored": counts.get(key, {"events": 0, "feedback": 0})}
            if b is not None:
                info = b.source.info()
                latest = b.latest
                events = b.events(50)
                last = events[-1] if events else None
                row.update({
                    "name": info.get("name"), "real": info.get("real", False),
                    "equipment_id": info.get("equipment_id"), "family": b.family,
                    "vib_unit": info.get("vib_unit"), "progress": info.get("progress"),
                    "standard": info.get("standard"),
                    "state": latest.state if latest else None,
                    "hhi": latest.hhi if latest else None,
                    "rul_hours": latest.rul_hours if latest else None,
                    "vib": latest.vib_rms if latest else None,
                    "vel": latest.vib_rms_mm_s if latest else None,
                    "machine_hours": latest.machine_hours if latest else None,
                    "events": len(events),
                    "last_event": None if last is None else {
                        "id": last.id, "state": last.state,
                        "machine_hours": last.machine_hours,
                        "top": last.top_classes[:1],
                        "failure_mode_id": ((last.diagnosis or {}).get("failure_mode_id")
                                         or None),
                        "diagnosis": (last.diagnosis or {}).get("diagnosis"),
                        "diagnosis_status": last.diagnosis_status,
                        "feedback": last.feedback,
                    },
                })
            out.append(row)
        # Most urgent first: the worst state, then the lowest remaining life.
        out.sort(key=lambda r: (-STATE_RANK.get(r.get("state") or "", -1),
                                r.get("rul_hours") if r.get("rul_hours") is not None else 1e9,
                                r["label"]))
        return out
