"""Listening to a live sensor, to commission from it (HARM-S7-T08).

A listener subscribes to one MQTT topic and writes every valid snapshot into
an upload folder, one file per snapshot, as it arrives. The folder is an
ordinary recording: the wizard draws it, picks a healthy stretch and
commissions from it like any other. What makes it live is only that it grows —
and that the machine, once commissioned, can keep watching the same topic
(`api.sources.LiveSource`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from api import adapters, machines
from api.adapters.mqtt import LiveSnapshot

MAX_SNAPSHOTS = 20_000        # ~140 days at one a minute; a listener is for commissioning


class LiveRecorder:
    def __init__(self, folder: Path, broker: str, topic: str, name: str) -> None:
        broker, topic = adapters.normalise(broker, topic)
        self.folder = folder
        self.name = name
        self.count = 0
        self.mismatched = 0
        self._meta: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self.subscriber = adapters.subscriber(broker, topic, self._on_snapshot,
                                              consumer=f"listen-{folder.name}")

    def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.subscriber.start()

    def stop(self) -> None:
        self.subscriber.stop()

    def _on_snapshot(self, snap: LiveSnapshot) -> None:
        with self._lock:
            if self.count >= MAX_SNAPSHOTS:
                return
            if self._meta is None:
                self._meta = {
                    "name": self.name, "sample_rate_hz": snap.sample_rate_hz,
                    "unit": snap.unit or "g", "interval_s": 60.0,
                    "provenance": f"live: {self.subscriber.broker} "
                                  f"{self.subscriber.topic}",
                    "samples": len(snap.samples), "t_s": [], "rpm": [],
                }
            meta = self._meta
            # Every snapshot of one recording has the same rate and length, or
            # the engine would be fed windows it was not commissioned on.
            if snap.sample_rate_hz != meta["sample_rate_hz"] or len(snap.samples) != meta["samples"]:
                self.mismatched += 1
                return
            n = self.count + 1
            tmp = self.folder / f"snap_{n:06d}.npy.part"
            with tmp.open("wb") as fh:
                np.save(fh, snap.samples.astype(np.float32))
            # The metadata first, then the file: a reader may see one more
            # timestamp than files, never a file without its timestamp.
            meta["t_s"].append(snap.t)
            if meta["rpm"] is not None:
                meta["rpm"] = meta["rpm"] + [snap.rpm] if snap.rpm else None
            out = {k: v for k, v in meta.items() if k != "samples" and v is not None}
            jtmp = self.folder / "recording.json.part"
            jtmp.write_text(json.dumps(out))
            os.replace(jtmp, self.folder / "recording.json")
            os.replace(tmp, self.folder / f"snap_{n:06d}.npy")
            self.count = n

    def status(self) -> dict[str, Any]:
        return {**self.subscriber.status(), "snapshots": self.count,
                "mismatched": self.mismatched, "name": self.name,
                "recording": f"upload:{self.folder.name}",
                "sample_rate_hz": (self._meta or {}).get("sample_rate_hz"),
                "samples": (self._meta or {}).get("samples")}


_listeners: dict[str, LiveRecorder] = {}
_lock = threading.Lock()


def listen(broker: str, topic: str, name: str) -> tuple[str, LiveRecorder]:
    """Start recording a topic into a new upload folder."""
    uid = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    rec = LiveRecorder(machines.UPLOADS_DIR / uid, broker, topic,
                       name.strip() or f"Live: {topic}")
    rec.start()
    with _lock:
        _listeners[uid] = rec
    return uid, rec


def get(uid: str) -> LiveRecorder | None:
    with _lock:
        return _listeners.get(uid)


def stop(uid: str) -> bool:
    with _lock:
        rec = _listeners.pop(uid, None)
    if rec is None:
        return False
    rec.stop()
    return True


def stop_all() -> None:
    with _lock:
        recs = list(_listeners.values())
        _listeners.clear()
    for r in recs:
        r.stop()
