"""Google Cloud Pub/Sub: vibration snapshots from a sensor gateway (HARM-S8-G02).

The same messages as MQTT (`api/adapters/mqtt.py`), carried by Pub/Sub: a
gateway at the plant publishes one message per snapshot to a topic; each
consumer here — a listener recording for commissioning, a machine watching
live — reads it through its own subscription, so both see every snapshot.

Subscriptions are created on first use, named after the topic and the
consumer, and expire after a day without one, so a consumer that goes away
leaves nothing behind. The project is `GOOGLE_CLOUD_PROJECT`, or the one in
the broker string: "pubsub" or "pubsub:<project>".
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from api.adapters.mqtt import LiveSnapshot, parse_payload

log = logging.getLogger("harmonic.pubsub")

_ID = re.compile(r"^[A-Za-z][A-Za-z0-9._~+%-]{2,254}$")


def is_pubsub(broker: str) -> bool:
    return broker.strip() == "pubsub" or broker.strip().startswith("pubsub:")


def project_of(broker: str) -> str:
    project = broker.strip().partition(":")[2] or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise ValueError("a Pub/Sub broker needs a project: 'pubsub:<project>', or "
                         "GOOGLE_CLOUD_PROJECT set")
    return project


def valid_topic(topic: str) -> bool:
    """A Pub/Sub topic id: 3-255 characters, starting with a letter."""
    return bool(_ID.match(topic)) and not topic.lower().startswith("goog")


def subscription_id(topic: str, consumer: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9._~+%-]", "-", f"harmonic-{topic}-{consumer}")
    return sid[:255]


class Subscriber:
    """The MQTT subscriber's interface (start, stop, status, counters) over a
    Pub/Sub streaming pull."""

    def __init__(self, broker: str, topic: str,
                 on_snapshot: Callable[[LiveSnapshot], None], consumer: str) -> None:
        if not valid_topic(topic):
            raise ValueError(f"topic {topic!r}: a Pub/Sub topic id (letters, digits, "
                             "- _ . ~ + %; starting with a letter)")
        self.project = project_of(broker)
        self.broker = f"pubsub:{self.project}"
        self.topic = topic
        self.consumer = consumer
        self.on_snapshot = on_snapshot
        self.received = 0
        self.rejected = 0
        self.connected = False
        self.last_error: str | None = None
        self.last_at: float | None = None
        self._lock = threading.Lock()
        self._future: Any = None
        self._client: Any = None

    def start(self) -> None:
        from google.api_core import exceptions
        from google.cloud import pubsub_v1  # type: ignore[attr-defined, unused-ignore]
        from google.protobuf import duration_pb2

        self._client = pubsub_v1.SubscriberClient()
        topic_path = f"projects/{self.project}/topics/{self.topic}"
        sub_path = self._client.subscription_path(self.project,
                                                  subscription_id(self.topic, self.consumer))
        try:
            self._client.create_subscription(request={
                "name": sub_path, "topic": topic_path, "ack_deadline_seconds": 30,
                "message_retention_duration": duration_pb2.Duration(seconds=3600),
                "expiration_policy": {"ttl": duration_pb2.Duration(seconds=86400)},
            })
        except exceptions.AlreadyExists:
            pass
        except Exception as exc:
            self.last_error = f"cannot subscribe: {exc}"
            raise
        self._future = self._client.subscribe(sub_path, callback=self._on_message)
        self.connected = True

    def stop(self) -> None:
        if self._future is not None:
            self._future.cancel()
            with contextlib.suppress(Exception):
                self._future.result(timeout=5)
            self._future = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self.connected = False

    def _on_message(self, message: Any) -> None:
        try:
            snap = parse_payload(message.data)
        except ValueError as exc:
            with self._lock:
                self.rejected += 1
                self.last_error = f"message rejected: {exc}"
            message.ack()                 # a bad message will not get better
            return
        with self._lock:
            self.received += 1
            self.last_at = time.time()
        try:
            self.on_snapshot(snap)
        except Exception as exc:
            log.warning("snapshot handler failed: %s", exc)
            self.last_error = f"{type(exc).__name__}: {exc}"
        message.ack()

    def status(self) -> dict[str, Any]:
        return {"broker": self.broker, "topic": self.topic, "connected": self.connected,
                "received": self.received, "rejected": self.rejected,
                "last_at": self.last_at, "last_error": self.last_error}
