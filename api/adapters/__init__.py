"""Data-acquisition adapters: live snapshots from outside, into the engine.

Each adapter turns one transport into `LiveSnapshot`s and knows nothing about
machines, brains or the engine: MQTT (`mqtt.py`) for a plant's own broker,
Google Cloud Pub/Sub (`pubsub.py`) for gateways that publish to the cloud.
OPC UA and Modbus gateways would sit beside them behind the same callback.

A broker string picks the transport: "host:port" is MQTT, "pubsub" or
"pubsub:<project>" is Pub/Sub.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def normalise(broker: str, topic: str) -> tuple[str, str]:
    """The broker and topic checked for their transport; ValueError otherwise."""
    from api.adapters import mqtt, pubsub

    if pubsub.is_pubsub(broker):
        if not pubsub.valid_topic(topic):
            raise ValueError("the topic must be a Pub/Sub topic id (letters, digits, "
                             "- _ . ~ + %; starting with a letter)")
        return f"pubsub:{pubsub.project_of(broker)}", topic
    host, port = mqtt.parse_broker(broker)
    if not mqtt.valid_topic(topic):
        raise ValueError("the MQTT topic must be one topic, without wildcards")
    return f"{host}:{port}", topic


def subscriber(broker: str, topic: str, on_snapshot: Callable[[Any], None],
               consumer: str) -> Any:
    """A subscriber for the broker's transport. `consumer` names who reads —
    Pub/Sub gives each consumer its own subscription."""
    from api.adapters import mqtt, pubsub

    if pubsub.is_pubsub(broker):
        return pubsub.Subscriber(broker, topic, on_snapshot, consumer)
    return mqtt.Subscriber(broker, topic, on_snapshot)
