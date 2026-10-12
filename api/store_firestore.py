"""Events, feedback and work orders in Firestore (HARM-S8-G01).

The same interface as the SQLite store (`api.store.EventStore`) and the
work-order repository (`api.workorders.SqliteRepo`), so the API does not know
which it has: SQLite on a PC, Firestore on Google Cloud, where Cloud Run's
disk does not outlive a redeploy.

Collections (prefix `HARMONIC_FIRESTORE_PREFIX`, default "harmonic_"):

    <p>events/{event id}       the event as JSON text, its source, state, times,
                               the verdict, and the raw window as bytes
    <p>counts/{source}         events and verdicts per machine, kept by increment
    <p>work_orders/{order id}  one row of api/workorders.py

The event body is stored as JSON text rather than a Firestore map: a diagnosis
can hold nested lists, which Firestore maps cannot. Only equality filters are
queried; ordering is done here, so no composite index has to exist first.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from api.store import VERDICTS, frames_file

MAX_WINDOW_BYTES = 900_000        # under Firestore's 1 MiB document limit


def _client(project: str | None = None) -> Any:
    from google.cloud import firestore

    return firestore.Client(project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                            database=os.environ.get("HARMONIC_FIRESTORE_DATABASE", "(default)"))


def _prefix() -> str:
    return os.environ.get("HARMONIC_FIRESTORE_PREFIX", "harmonic_")


class FirestoreEventStore:
    def __init__(self, client: Any | None = None, prefix: str | None = None) -> None:
        self.db = client or _client()
        p = prefix if prefix is not None else _prefix()
        self.events = self.db.collection(f"{p}events")
        self.counts_col = self.db.collection(f"{p}counts")

    def close(self) -> None:
        close = getattr(self.db, "close", None)
        if callable(close):
            close()

    # -- writes

    def save(self, event: dict[str, Any], source: str, family: str,
             frame: Any | None = None, sample_rate: int = 0) -> None:
        from google.cloud import firestore

        doc: dict[str, Any] = {
            "source": source, "equipment_id": event.get("equipment_id"), "family": family,
            "created": time.time(), "machine_hours": event.get("machine_hours"),
            "recorded_at": event.get("recorded_at"), "state": event.get("state"),
            "body": json.dumps(event, default=str), "feedback": None, "judged": False,
            "sample_rate": sample_rate or None,
        }
        if frame is not None:
            window = np.asarray(frame.vibration, dtype=np.float32).tobytes()
            if len(window) <= MAX_WINDOW_BYTES:
                doc.update(window=window, rpm=float(frame.rpm),
                           slow=[float(v) for v in frame.slow], present=int(frame.present))
        self.events.document(event["id"]).set(doc)
        self.counts_col.document(source).set({"events": firestore.Increment(1)}, merge=True)

    def update(self, event: dict[str, Any]) -> None:
        self.events.document(event["id"]).update({"body": json.dumps(event, default=str)})

    def set_feedback(self, event_id: str, verdict: str, actual: str | None,
                     note: str, classes: tuple[str, ...]) -> dict[str, Any]:
        from google.cloud import firestore

        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if verdict == "wrong":
            if actual not in (*classes, "other"):
                raise ValueError(f"actual must be one of {[*classes, 'other']}")
        else:
            actual = None
        fb = {"verdict": verdict, "actual": actual, "note": note.strip()[:1000],
              "at": time.time()}
        ref = self.events.document(event_id)
        snap = ref.get()
        if not snap.exists:
            raise KeyError(event_id)
        first = not snap.get("judged")
        ref.update({"feedback": json.dumps(fb), "judged": True})
        if first:
            self.counts_col.document(snap.get("source")).set(
                {"feedback": firestore.Increment(1)}, merge=True)
        return fb

    # -- reads

    def _event(self, d: dict[str, Any]) -> dict[str, Any]:
        e = json.loads(d["body"])
        e["feedback"] = json.loads(d["feedback"]) if d.get("feedback") else None
        e["source"] = d["source"]
        return e

    def recent(self, source: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        from google.cloud import firestore

        if source:
            docs = [s.to_dict() for s in self.events.where(
                filter=firestore.FieldFilter("source", "==", source)).stream()]
            docs.sort(key=lambda d: d["created"], reverse=True)
            docs = docs[:limit]
        else:
            docs = [s.to_dict() for s in self.events.order_by(
                "created", direction=firestore.Query.DESCENDING).limit(limit).stream()]
        return [self._event(d) for d in docs]

    def feedback(self, event_id: str) -> dict[str, Any] | None:
        snap = self.events.document(event_id).get()
        fb = snap.get("feedback") if snap.exists else None
        return json.loads(fb) if fb else None

    def counts(self) -> dict[str, dict[str, int]]:
        return {s.id: {"events": int((s.to_dict() or {}).get("events", 0)),
                       "feedback": int((s.to_dict() or {}).get("feedback", 0))}
                for s in self.counts_col.stream()}

    # -- training data

    def export_frames(self, source: str, classes: tuple[str, ...]) -> tuple[bytes, int]:
        from google.cloud import firestore

        docs = [s.to_dict() for s in self.events
                .where(filter=firestore.FieldFilter("source", "==", source))
                .where(filter=firestore.FieldFilter("judged", "==", True)).stream()]
        docs.sort(key=lambda d: d["created"])
        rows = [(json.loads(d["body"]), json.loads(d["feedback"]), d.get("sample_rate"),
                 d.get("rpm"), d.get("slow"), d.get("present"), d.get("window"))
                for d in docs if d.get("window")]
        return frames_file(rows, classes)


class FirestoreWorkOrders:
    """The work-order repository of api/workorders.py, in Firestore."""

    def __init__(self, client: Any | None = None, prefix: str | None = None) -> None:
        self.db = client or _client()
        p = prefix if prefix is not None else _prefix()
        self.col = self.db.collection(f"{p}work_orders")

    def _row(self, snap: Any) -> dict[str, Any] | None:
        if not snap.exists:
            return None
        d = snap.to_dict()
        d["payload"] = json.loads(d["payload"])
        return d

    def find_open(self, source: str, fault: str) -> dict[str, Any] | None:
        from google.cloud import firestore

        q = (self.col.where(filter=firestore.FieldFilter("source", "==", source))
             .where(filter=firestore.FieldFilter("fault", "==", fault))
             .where(filter=firestore.FieldFilter("status", "==", "open")).limit(1))
        for snap in q.stream():
            return self._row(snap)
        return None

    def insert(self, row: dict[str, Any]) -> None:
        self.col.document(row["id"]).set({**row, "payload": json.dumps(row["payload"])})

    def update(self, wo_id: str, fields: dict[str, Any], if_status: str | None = None) -> bool:
        ref = self.col.document(wo_id)
        snap = ref.get()
        if not snap.exists or (if_status and snap.get("status") != if_status):
            return False
        out = dict(fields)
        if "payload" in out:
            out["payload"] = json.dumps(out["payload"])
        ref.update(out)
        return True

    def get(self, wo_id: str) -> dict[str, Any] | None:
        return self._row(self.col.document(wo_id).get())

    def list(self, source: str | None, limit: int) -> list[dict[str, Any]]:
        from google.cloud import firestore

        q = (self.col.where(filter=firestore.FieldFilter("source", "==", source))
             if source else self.col)
        rows = [r for r in (self._row(s) for s in q.stream()) if r is not None]
        rows.sort(key=lambda r: r["created"], reverse=True)
        return rows[:limit]

    def close(self) -> None:
        pass
