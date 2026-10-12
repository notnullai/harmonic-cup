"""Work orders to a maintenance system (HARM-S7-T14)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from api.workorders import WorkOrders


@pytest.fixture
def webhook():
    """A CMMS stand-in: records every POST; answers 503 while `fail` is set."""
    got: list[dict] = []
    state = {"fail": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if state["fail"]:
                state["fail"] -= 1
                self.send_response(503)
            else:
                got.append({"body": body, "key": self.headers.get("Idempotency-Key")})
                self.send_response(201)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/hooks/workorder", got, state
    srv.shutdown()


def _event(i: int, state: str = "WARNING", fault: str = "gbx.fm.tooth_breakage") -> dict:
    return {"id": f"ev-{i}", "state": state, "equipment_id": "GB-201", "rul_hours": 40.0,
            "machine_hours": float(i), "recorded_at": None, "diagnosis_status": "ok",
            "diagnosis": {"failure_mode_id": fault, "confidence": 0.9,
                          "diagnosis": "A cracked or broken pinion tooth on GB-201.",
                          "actions": [{"step": "Inspect the gear teeth"}],
                          "references": ["GB-201-OM-4.1"], "rul_hours": 40.0}}


def test_one_open_order_per_fault_with_escalation_as_an_update(tmp_path, webhook):
    url, got, _ = webhook
    wo = WorkOrders(tmp_path / "e.db", url=url, retries=(0.0,))
    first = wo.consider(_event(1), "gb", "Gearbox GB-201", wait=True)
    assert first["delivery"] == "sent" and first["priority"] == 2 and not first["update"]
    repeat = wo.consider(_event(2), "gb", wait=True)
    assert repeat["repeat"] and repeat["id"] == first["id"] and repeat["alarms"] == 2
    escalated = wo.consider(_event(3, "CRITICAL"), "gb", wait=True)
    assert escalated["id"] == first["id"] and escalated["update"] and escalated["priority"] == 1
    other = wo.consider(_event(4, fault="gbx.fm.tooth_wear"), "gb", wait=True)
    assert other["id"] != first["id"]
    assert [g["body"]["id"] for g in got] == [first["id"], first["id"], other["id"]]
    assert got[1]["body"]["event_ids"] == ["ev-1", "ev-2", "ev-3"]
    # What a CMMS needs to open a job.
    body = got[0]["body"]
    assert body["equipment_id"] == "GB-201" and len(body["title"]) <= 40
    assert body["actions"] and body["references"] == ["GB-201-OM-4.1"]
    assert got[0]["key"] == f"{first['id']}-WARNING"
    # The job done, the next occurrence opens a new order.
    assert wo.close_order(first["id"])
    assert wo.consider(_event(5), "gb", wait=True)["id"] not in (first["id"], other["id"])


def test_no_order_without_an_alarm_or_a_diagnosis(tmp_path, webhook):
    url, got, _ = webhook
    wo = WorkOrders(tmp_path / "e.db", url=url, retries=(0.0,))
    assert wo.consider(_event(1, "WATCH"), "gb", wait=True) is None
    assert wo.consider({**_event(2), "diagnosis_status": "fallback"}, "gb", wait=True) is None
    assert got == []


def test_delivery_is_retried_then_reported(tmp_path, webhook):
    url, got, state = webhook
    state["fail"] = 1
    wo = WorkOrders(tmp_path / "e.db", url=url, retries=(0.0, 0.05))
    order = wo.consider(_event(1), "gb", wait=True)
    assert order["delivery"] == "sent" and order["attempts"] == 2 and len(got) == 1
    dead = WorkOrders(tmp_path / "f.db", url="http://127.0.0.1:9/none", retries=(0.0, 0.01))
    lost = dead.consider(_event(1), "gb", wait=True)
    assert lost["delivery"] == "failed" and lost["last_error"]


def test_without_a_webhook_orders_go_to_the_demo_inbox(tmp_path, monkeypatch):
    monkeypatch.delenv("HARMONIC_WORKORDER_URL", raising=False)
    wo = WorkOrders(tmp_path / "e.db")
    assert wo.consider(_event(1), "gb")["delivery"] == "simulated"
    assert len(wo.list()) == 1
