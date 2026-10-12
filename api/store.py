"""The event store: every raised event, its diagnosis and its feedback, kept
across restarts (HARM-S7-T13), plus the vibration window that raised it, so a
technician's verdict can become training data (HARM-S7-T12).

SQLite, one file under the data directory. It is the edge-to-cloud record of
what the engines said, what the diagnosis said, and what the person at the
machine found — the last of which is the only ground truth a deployed
system ever gets.
"""

from __future__ import annotations

import io
import json
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

VERDICTS = ("confirmed", "wrong")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    equipment_id  TEXT,
    family        TEXT,
    created       REAL NOT NULL,
    machine_hours REAL,
    recorded_at   TEXT,
    state         TEXT,
    body          TEXT NOT NULL,     -- the event as the API serves it
    feedback      TEXT,              -- the technician's verdict, JSON, or NULL
    sample_rate   INTEGER,
    rpm           REAL,
    slow          TEXT,              -- the seven slow channels, JSON
    present       INTEGER,
    window        BLOB               -- float32 vibration samples
);
CREATE INDEX IF NOT EXISTS events_source ON events (source, created);
"""


class EventStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writes

    def save(self, event: dict[str, Any], source: str, family: str,
             frame: Any | None = None, sample_rate: int = 0) -> None:
        """A new event, with the frame that raised it when there is one."""
        window = rpm = slow = present = None
        if frame is not None:
            window = np.asarray(frame.vibration, dtype=np.float32).tobytes()
            rpm, slow, present = float(frame.rpm), json.dumps(list(frame.slow)), int(frame.present)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO events (id, source, equipment_id, family, created, "
                "machine_hours, recorded_at, state, body, feedback, sample_rate, rpm, slow, "
                "present, window) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event["id"], source, event.get("equipment_id"), family, time.time(),
                 event.get("machine_hours"), event.get("recorded_at"), event.get("state"),
                 json.dumps(event, default=str), None, sample_rate or None, rpm, slow,
                 present, window))
            self._db.commit()

    def update(self, event: dict[str, Any]) -> None:
        """The event again, once its diagnosis has arrived."""
        with self._lock:
            self._db.execute("UPDATE events SET body = ? WHERE id = ?",
                             (json.dumps(event, default=str), event["id"]))
            self._db.commit()

    def set_feedback(self, event_id: str, verdict: str, actual: str | None,
                     note: str, classes: tuple[str, ...]) -> dict[str, Any]:
        """Record what the technician found. `actual` names the fault when the
        diagnosis was wrong: one of the machine family's classes ("healthy"
        for a false alarm), or "other" for a fault the family has no class
        for."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if verdict == "wrong":
            if actual not in (*classes, "other"):
                raise ValueError(f"actual must be one of {[*classes, 'other']}")
        else:
            actual = None
        fb = {"verdict": verdict, "actual": actual, "note": note.strip()[:1000],
              "at": time.time()}
        with self._lock:
            cur = self._db.execute("UPDATE events SET feedback = ? WHERE id = ?",
                                   (json.dumps(fb), event_id))
            self._db.commit()
        if cur.rowcount == 0:
            raise KeyError(event_id)
        return fb

    # -- reads

    def recent(self, source: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q = "SELECT body, feedback, source FROM events"
        args: tuple[Any, ...] = ()
        if source:
            q += " WHERE source = ?"
            args = (source,)
        q += " ORDER BY created DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(q, (*args, limit)).fetchall()
        out = []
        for body, fb, src in rows:
            e = json.loads(body)
            e["feedback"] = json.loads(fb) if fb else None
            e["source"] = src
            out.append(e)
        return out

    def feedback(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT feedback FROM events WHERE id = ?",
                                   (event_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def counts(self) -> dict[str, dict[str, int]]:
        """Per source: events stored, and how many carry a verdict."""
        with self._lock:
            rows = self._db.execute(
                "SELECT source, COUNT(*), COUNT(feedback) FROM events GROUP BY source"
            ).fetchall()
        return {s: {"events": n, "feedback": f} for s, n, f in rows}

    # -- training data

    def export_frames(self, source: str, classes: tuple[str, ...]) -> tuple[bytes, int]:
        """The events of one machine that a technician has judged, as a
        Harmonic frame file labelled with what was actually there — the format
        `harmonic_sim_words` encodes for the trainer. A confirmed diagnosis is
        labelled with the class the engine named; a wrong one with the class
        the technician found. "other" and events without a window are left
        out: there is no class to learn them as."""
        with self._lock:
            rows = self._db.execute(
                "SELECT body, feedback, sample_rate, rpm, slow, present, window "
                "FROM events WHERE source = ? AND feedback IS NOT NULL AND window IS NOT NULL "
                "ORDER BY created", (source,)).fetchall()
        return frames_file([(json.loads(b), json.loads(f), r, rpm, json.loads(s), p, w)
                            for b, f, r, rpm, s, p, w in rows], classes)


def frames_file(rows: list[tuple[Any, ...]], classes: tuple[str, ...]) -> tuple[bytes, int]:
    """Judged events as a labelled frame file. Each row is (event, feedback,
    sample rate, rpm, slow readings, present mask, window bytes), oldest
    first; shared by the SQLite and Firestore stores."""
    from sim.cli import FRAME_HEADER, FRAME_MAGIC, FRAME_RECORD, FRAME_VERSION

    records = []
    for ev, fb, rate, rpm, slow, present, window in rows:
        top = ((ev.get("top_classes") or [{}])[0].get("name")
               if fb["verdict"] == "confirmed" else fb.get("actual"))
        if top not in classes or not window:
            continue
        records.append((ev, classes.index(top), rate, rpm, slow, present,
                        np.frombuffer(window, dtype=np.float32)))
    if not records:
        return b"", 0
    n_samples, rate = len(records[0][6]), int(records[0][2])
    records = [r for r in records if len(r[6]) == n_samples and int(r[2]) == rate]
    buf = io.BytesIO()
    buf.write(struct.pack(FRAME_HEADER, FRAME_MAGIC, FRAME_VERSION, len(records),
                          n_samples, rate, len(classes), 0, 0))
    for i, (ev, label, _, rpm, slow, present, window) in enumerate(records):
        t_us = int(float(ev.get("machine_hours") or i) * 3.6e9)
        buf.write(struct.pack(FRAME_RECORD, t_us, int(rpm * 65536), int(0.6 * 65536),
                              1 << label, *[float(v) for v in slow[:7]], int(present)))
        buf.write(window.tobytes())
    return buf.getvalue(), len(records)


def open_stores(data_dir: Path) -> tuple[Any, Any]:
    """The event store and the work-order store: SQLite files under the data
    folder, or Firestore when HARMONIC_STORE=firestore (on Google Cloud)."""
    import os

    if os.environ.get("HARMONIC_STORE", "sqlite") == "firestore":
        from api.store_firestore import FirestoreEventStore, FirestoreWorkOrders

        return FirestoreEventStore(), FirestoreWorkOrders()
    return EventStore(data_dir / "events.db"), data_dir / "events.db"
