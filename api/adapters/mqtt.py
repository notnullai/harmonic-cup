"""MQTT: vibration snapshots from a sensor gateway (HARM-S7-T08).

One message per snapshot, JSON:

    {
      "t": 1727258400.25,          # when it was taken, epoch seconds (optional:
                                   #   the time it arrived)
      "sample_rate_hz": 25600,
      "samples_b64": "...",        # float32 little-endian, base64 — or
      "samples": [0.01, ...],      #   a plain list, for hand-written tests
      "unit": "g",                 # optional; checked against the machine's
      "rpm": 1480.2,               # optional: a tachometer reading
      "slow": [61.2, null, ...]    # optional: up to 7 slow readings in frame
                                   #   slot order; null = not measured
    }

A message is input from the network, so everything is checked before it goes
near the engine: size, sample count, rate, finiteness. A bad message is
counted and dropped, never passed on.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("harmonic.mqtt")

MAX_PAYLOAD = 4 * 1024 * 1024          # bytes: 2^18 float32 samples, base64
MIN_SAMPLES, MAX_SAMPLES = 2048, 1 << 18   # MIN_SAMPLES: recordings.MIN_WINDOW
SLOW_SLOTS = 7


@dataclass
class LiveSnapshot:
    t: float                         # epoch seconds
    samples: np.ndarray              # float32
    sample_rate_hz: int
    unit: str | None = None
    rpm: float | None = None
    slow: list[float] = field(default_factory=lambda: [0.0] * SLOW_SLOTS)
    present: int = 0                 # which slow slots were measured


def parse_payload(raw: bytes, now: float | None = None) -> LiveSnapshot:
    """One message, validated. Raises ValueError with the reason."""
    if len(raw) > MAX_PAYLOAD:
        raise ValueError(f"message of {len(raw)} bytes exceeds {MAX_PAYLOAD}")
    try:
        msg = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not JSON: {exc}") from None
    if not isinstance(msg, dict):
        raise ValueError("a snapshot is a JSON object")

    rate = msg.get("sample_rate_hz")
    if not isinstance(rate, int) or isinstance(rate, bool) or not 100 <= rate <= 1_000_000:
        raise ValueError("sample_rate_hz must be an integer from 100 to 1000000")

    if "samples_b64" in msg:
        try:
            buf = base64.b64decode(str(msg["samples_b64"]), validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("samples_b64 is not valid base64") from None
        if len(buf) % 4:
            raise ValueError("samples_b64 is not a whole number of float32 values")
        samples = np.frombuffer(buf, dtype="<f4").astype(np.float32)
    elif isinstance(msg.get("samples"), list):
        try:
            samples = np.asarray(msg["samples"], dtype=np.float32)
        except (TypeError, ValueError):
            raise ValueError("samples must be numbers") from None
        if samples.ndim != 1:
            raise ValueError("samples must be a flat list")
    else:
        raise ValueError("no samples_b64 or samples")
    if not MIN_SAMPLES <= len(samples) <= MAX_SAMPLES:
        raise ValueError(f"{len(samples)} samples; a snapshot needs {MIN_SAMPLES} to "
                         f"{MAX_SAMPLES}")
    if not np.isfinite(samples).all():
        raise ValueError("samples contain NaN or infinity")

    t = msg.get("t", now if now is not None else time.time())
    if not isinstance(t, int | float) or isinstance(t, bool) or not math.isfinite(t) \
            or not 0 < t < 1e11:
        raise ValueError("t must be epoch seconds")

    rpm = msg.get("rpm")
    if rpm is not None and (not isinstance(rpm, int | float) or isinstance(rpm, bool)
                            or not math.isfinite(rpm) or not 0 <= rpm <= 100_000):
        raise ValueError("rpm must be a number from 0 to 100000")

    unit = msg.get("unit")
    if unit is not None and (not isinstance(unit, str) or len(unit) > 8):
        raise ValueError("unit must be a short string")

    slow = [0.0] * SLOW_SLOTS
    present = 0
    raw_slow = msg.get("slow")
    if raw_slow is not None:
        if not isinstance(raw_slow, list) or len(raw_slow) > SLOW_SLOTS:
            raise ValueError(f"slow is a list of at most {SLOW_SLOTS} readings")
        for i, v in enumerate(raw_slow):
            if v is None:
                continue
            if not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"slow[{i}] must be a number or null")
            slow[i] = float(v)
            present |= 1 << i

    return LiveSnapshot(t=float(t), samples=samples, sample_rate_hz=rate, unit=unit,
                        rpm=None if rpm is None else float(rpm), slow=slow, present=present)


def encode_payload(samples: np.ndarray, sample_rate_hz: int, *, t: float | None = None,
                   rpm: float | None = None, unit: str | None = None,
                   slow: list[float | None] | None = None) -> bytes:
    """The message a gateway sends — used by the publisher and the tests."""
    msg: dict[str, Any] = {
        "t": time.time() if t is None else t, "sample_rate_hz": int(sample_rate_hz),
        "samples_b64": base64.b64encode(
            np.asarray(samples, dtype="<f4").tobytes()).decode("ascii"),
    }
    if rpm is not None:
        msg["rpm"] = float(rpm)
    if unit is not None:
        msg["unit"] = unit
    if slow is not None:
        msg["slow"] = slow
    return json.dumps(msg).encode()


def parse_broker(broker: str) -> tuple[str, int]:
    """"host:port" or "host" (1883)."""
    host, _, port = broker.strip().rpartition(":") if ":" in broker else (broker, "", "1883")
    host = host or "localhost"
    try:
        p = int(port)
    except ValueError:
        raise ValueError(f"bad broker port in {broker!r}") from None
    if not 0 < p < 65536:
        raise ValueError(f"bad broker port in {broker!r}")
    return host, p


def valid_topic(topic: str) -> bool:
    """A subscription to one topic: no wildcards, no empty levels."""
    return (0 < len(topic) <= 200 and "+" not in topic and "#" not in topic
            and "//" not in topic and not topic.startswith("$"))


class Subscriber:
    """Subscribes to one topic and hands each valid snapshot to `on_snapshot`.

    Reconnects on its own (paho's network loop). `received`, `rejected`,
    `connected` and `last_error` are for the console's status line.
    """

    def __init__(self, broker: str, topic: str,
                 on_snapshot: Callable[[LiveSnapshot], None]) -> None:
        if not valid_topic(topic):
            raise ValueError(f"topic {topic!r}: one topic, without wildcards")
        self.host, self.port = parse_broker(broker)
        self.broker, self.topic = f"{self.host}:{self.port}", topic
        self.on_snapshot = on_snapshot
        self.received = 0
        self.rejected = 0
        self.connected = False
        self.last_error: str | None = None
        self.last_at: float | None = None
        self._lock = threading.Lock()
        self._client: Any = None

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("paho-mqtt is not installed (pip install paho-mqtt)") from exc
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message
        c.reconnect_delay_set(min_delay=1, max_delay=10)
        c.connect_async(self.host, self.port, keepalive=30)
        c.loop_start()
        self._client = c

    def stop(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client.loop_stop()
            self._client = None
        self.connected = False

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason: Any,
                    properties: Any = None) -> None:
        if getattr(reason, "is_failure", False):
            self.connected = False
            self.last_error = f"broker refused the connection: {reason}"
            return
        self.connected = True
        self.last_error = None
        client.subscribe(self.topic, qos=1)

    def _on_disconnect(self, client: Any, userdata: Any, flags: Any, reason: Any,
                       properties: Any = None) -> None:
        self.connected = False

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            snap = parse_payload(msg.payload)
        except ValueError as exc:
            with self._lock:
                self.rejected += 1
                self.last_error = f"message rejected: {exc}"
            return
        with self._lock:
            self.received += 1
            self.last_at = time.time()
        try:
            self.on_snapshot(snap)
        except Exception as exc:                 # never kill the network loop
            log.warning("snapshot handler failed: %s", exc)
            self.last_error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        return {"broker": self.broker, "topic": self.topic, "connected": self.connected,
                "received": self.received, "rejected": self.rejected,
                "last_at": self.last_at, "last_error": self.last_error}
