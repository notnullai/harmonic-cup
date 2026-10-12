"""Work orders to a maintenance system (HARM-S7-T14).

When a WARNING-or-worse event's diagnosis settles, a work order is posted to a
webhook as JSON. The fields are the ones a CMMS needs to open a job, named so
an adapter maps them one to one:

    field            SAP PM notification        IBM Maximo work order
    ---------------  -------------------------  -----------------------
    equipment_id     EQUNR (equipment)          ASSETNUM
    title            QMTXT (short text)         DESCRIPTION
    priority         PRIOK (1 very high ...)    WOPRIORITY
    summary          long text                  DESCRIPTION_LONGDESCRIPTION
    actions          task list operations       job plan tasks
    references       document links             attached documents

One open work order per machine and failure mode: the engine re-reports a
standing CRITICAL every cooldown, and a planner who receives the same job ten
times stops reading them. A repeat is recorded against the open order; an
escalation (WARNING to CRITICAL) is posted again, marked as an update. Closing
the order — done on the job — lets the next occurrence open a new one.

Without a webhook (HARMONIC_WORKORDER_URL unset) the orders go to Harmonic's
own demo inbox instead: kept, listed by `GET /workorders`, marked "simulated",
and shown in the console as the planner would receive them.

Advisory only, as everything Harmonic emits: the order asks a person to look.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("harmonic.workorders")

TRIGGER = {"WARNING": 2, "CRITICAL": 1, "FAILED": 1}     # state -> priority (1 highest)
RETRIES = (0.0, 2.0, 10.0)                                # seconds before each attempt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    fault       TEXT NOT NULL,
    state       TEXT NOT NULL,
    status      TEXT NOT NULL,     -- open | closed
    delivery    TEXT NOT NULL,     -- pending | sent | failed | simulated (no webhook)
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    events      TEXT NOT NULL,     -- JSON list of event ids
    payload     TEXT NOT NULL,
    created     REAL NOT NULL,
    updated     REAL NOT NULL
);
"""


_COLUMNS = ("id", "source", "fault", "state", "status", "delivery", "attempts", "last_error",
            "events", "payload", "created", "updated")
_JSON = ("events", "payload")


class SqliteRepo:
    """Work orders in the local SQLite file, beside the events. The Firestore
    repository (`api.store_firestore.FirestoreWorkOrders`) has the same five
    methods; `WorkOrders` holds the logic and does not know which it has."""

    def __init__(self, db_path: Path | str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()

    @staticmethod
    def _row(r: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if r is None:
            return None
        d = dict(zip(_COLUMNS, r, strict=True))
        for k in _JSON:
            d[k] = json.loads(d[k])
        return d

    def find_open(self, source: str, fault: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._db.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM work_orders "
                "WHERE source = ? AND fault = ? AND status = 'open'", (source, fault)).fetchone())

    def insert(self, row: dict[str, Any]) -> None:
        vals = [json.dumps(row[c]) if c in _JSON else row[c] for c in _COLUMNS]
        marks = ",".join("?" for _ in _COLUMNS)
        with self._lock:
            self._db.execute(f"INSERT INTO work_orders VALUES ({marks})", vals)
            self._db.commit()

    def update(self, wo_id: str, fields: dict[str, Any], if_status: str | None = None) -> bool:
        sets = ", ".join(f"{k} = ?" for k in fields)
        args = [json.dumps(v) if k in _JSON else v for k, v in fields.items()] + [wo_id]
        q = f"UPDATE work_orders SET {sets} WHERE id = ?"
        if if_status:
            q += " AND status = ?"
            args.append(if_status)
        with self._lock:
            cur = self._db.execute(q, args)
            self._db.commit()
        return cur.rowcount > 0

    def get(self, wo_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._db.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM work_orders WHERE id = ?",
                (wo_id,)).fetchone())

    def list(self, source: str | None, limit: int) -> list[dict[str, Any]]:
        q = f"SELECT {', '.join(_COLUMNS)} FROM work_orders"
        args: tuple[Any, ...] = ()
        if source:
            q, args = q + " WHERE source = ?", (source,)
        with self._lock:
            rows = self._db.execute(q + " ORDER BY created DESC LIMIT ?",
                                    (*args, limit)).fetchall()
        return [r for r in (self._row(x) for x in rows) if r is not None]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class WorkOrders:
    def __init__(self, store: Path | str | Any, url: str | None = None,
                 retries: tuple[float, ...] = RETRIES) -> None:
        """`store` is a SQLite path, or a repository with SqliteRepo's methods."""
        self.url = url if url is not None else os.environ.get("HARMONIC_WORKORDER_URL") or None
        self.retries = retries
        self._lock = threading.Lock()
        self.repo = SqliteRepo(store) if isinstance(store, str | Path) else store

    def close(self) -> None:
        self.repo.close()

    # -- the decision

    def consider(self, event: dict[str, Any], source: str, machine_name: str = "",
                 wait: bool = False) -> dict[str, Any] | None:
        """An event whose diagnosis has settled. Returns the work order it
        opened or updated, or None when it warrants none."""
        state = str(event.get("state"))
        dx = event.get("diagnosis") or {}
        fault = dx.get("failure_mode_id")
        if state not in TRIGGER or event.get("diagnosis_status") != "ok" or not fault:
            return None                   # no alarm, or no diagnosis to act on
        with self._lock:
            row = self.repo.find_open(source, fault)
            now = time.time()
            if row is not None:
                wo_id = row["id"]
                events = [*row["events"], event["id"]]
                escalated = TRIGGER[state] < TRIGGER.get(row["state"], 3)
                payload = (self._payload(wo_id, event, source, machine_name, update=True)
                           if escalated else row["payload"])
                payload["event_ids"] = events
                self.repo.update(wo_id, {"state": state if escalated else row["state"],
                                         "events": events, "payload": payload, "updated": now})
                if not escalated:
                    # Recorded against the open order, not sent again.
                    got = self.get(wo_id)
                    return None if got is None else {**got, "repeat": True}
            else:
                wo_id = f"WO-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                payload = self._payload(wo_id, event, source, machine_name, update=False)
                payload["event_ids"] = [event["id"]]
                self.repo.insert({
                    "id": wo_id, "source": source, "fault": fault, "state": state,
                    "status": "open", "delivery": "pending" if self.url else "simulated",
                    "attempts": 0, "last_error": None, "events": [event["id"]],
                    "payload": payload, "created": now, "updated": now})
        if self.url:
            t = threading.Thread(target=self._deliver, args=(wo_id, payload), daemon=True,
                                 name=f"workorder-{wo_id}")
            t.start()
            if wait:
                t.join()
        return self.get(wo_id)

    def _payload(self, wo_id: str, event: dict[str, Any], source: str, machine_name: str,
                 update: bool) -> dict[str, Any]:
        dx = event.get("diagnosis") or {}
        fault = str(dx.get("failure_mode_id", ""))
        label = fault.split("fm.", 1)[-1].replace("_", " ")
        return {
            "type": "harmonic.work_order", "version": 1, "id": wo_id, "update": update,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source, "machine": machine_name or source,
            "equipment_id": event.get("equipment_id"),
            "state": event.get("state"), "priority": TRIGGER[str(event.get("state"))],
            "title": f"{label.capitalize()} suspected on {event.get('equipment_id')}"[:40],
            "failure_mode_id": fault, "confidence": dx.get("confidence"),
            "summary": dx.get("diagnosis"),
            "actions": dx.get("actions", []), "safety": dx.get("safety", []),
            "references": dx.get("references", []),
            "remaining_life_hours": dx.get("rul_hours", event.get("rul_hours")),
            "machine_time": event.get("recorded_at") or event.get("machine_hours"),
            "advisory": "Advisory only: Harmonic never drives an actuator; the plant's own "
                        "protection stays authoritative.",
        }

    # -- delivery

    def _deliver(self, wo_id: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        error = None
        for attempt, delay in enumerate(self.retries, start=1):
            if delay:
                time.sleep(delay)
            try:
                req = urllib.request.Request(
                    str(self.url), data=body, method="POST",
                    headers={"Content-Type": "application/json",
                             "Idempotency-Key": f"{wo_id}-{payload.get('state')}"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        self._mark(wo_id, "sent", attempt, None)
                        return
                    error = f"HTTP {resp.status}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                error = str(getattr(exc, "reason", exc))
            self._mark(wo_id, "pending", attempt, error)
        self._mark(wo_id, "failed", len(self.retries), error)
        log.warning("work order %s not delivered: %s", wo_id, error)

    def _mark(self, wo_id: str, delivery: str, attempts: int, error: str | None) -> None:
        self.repo.update(wo_id, {"delivery": delivery, "attempts": attempts,
                                 "last_error": error, "updated": time.time()})

    # -- reads and the job being done

    @staticmethod
    def _view(row: dict[str, Any]) -> dict[str, Any]:
        return {**row["payload"], "status": row["status"], "delivery": row["delivery"],
                "attempts": row["attempts"], "last_error": row["last_error"],
                "event_ids": row["events"], "alarms": len(row["events"]),
                "opened_at": row["created"], "updated_at": row["updated"]}

    def get(self, wo_id: str) -> dict[str, Any] | None:
        row = self.repo.get(wo_id)
        return None if row is None else self._view(row)

    def list(self, source: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [self._view(r) for r in self.repo.list(source, limit)]

    def close_order(self, wo_id: str) -> bool:
        return bool(self.repo.update(wo_id, {"status": "closed", "updated": time.time()},
                                     if_status="open"))
