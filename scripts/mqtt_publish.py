"""Publish vibration snapshots over MQTT or Pub/Sub, as a sensor gateway would
(HARM-S7-T08, HARM-S8-G02).

For demos and tests: a simulated machine, or a recording replayed with its
own timestamps.

    python scripts/mqtt_publish.py --topic plant/gb1/vib --family gearbox
    python scripts/mqtt_publish.py --topic plant/gb1/vib --family gearbox \\
        --inject broken_tooth@300 --rate 4
    python scripts/mqtt_publish.py --topic rig/b1 --recording ims:test2 --channel 0
    python scripts/mqtt_publish.py --broker pubsub:my-project --topic gb7-vibration \
        --family gearbox                  # Google Cloud Pub/Sub; the topic must exist

`--interval` is the machine time between snapshots (what the engine's clock
advances by); `--rate` is how many are sent per wall-clock second.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def simulated(family: str, seed: int, window: int, inject: tuple[str, int] | None):
    from scripts.build_brain import family as fam

    sim = fam(family)["sim"](seed=seed, fs_vib=25600.0)
    sim.time_scale = 60.0
    dt = window / 25600.0
    i = 0
    while True:
        if inject and i == inject[1]:
            sim.inject(inject[0], severity=1.0, onset_s=sim.t_s, ramp_hours=1.0)
        fr = sim.step(dt)
        slow = list(fr.slow) if hasattr(fr, "slow") else [
            fr.bearing_temp_c, fr.motor_temp_c, fr.discharge_bar, fr.suction_bar,
            fr.flow_m3h, fr.current_a, None]
        present = getattr(fr, "present", 0x3F)
        slow = [v if present & (1 << k) else None for k, v in enumerate(slow)]
        yield fr.vibration, 25600, fr.rpm, slow, None
        i += 1


def recorded(key: str, channel: int):
    from api import machines

    r = machines.recording(key)
    data = r.load(channel, r.snapshot_samples())
    hours = r.hours()
    speeds = r.speeds()
    for k, samples in enumerate(data):
        rpm = None if speeds is None else float(speeds[k])
        yield samples, r.sample_rate_hz, rpm, None, float(hours[k]) * 3600.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="localhost:1883")
    ap.add_argument("--topic", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--family", help="a simulated machine: pump or gearbox")
    src.add_argument("--recording", help="a recording key, e.g. ims:test2 or upload:<id>")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--rate", type=float, default=2.0, help="snapshots per second")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="machine seconds between simulated snapshots")
    ap.add_argument("--count", type=int, default=0, help="stop after N (0: run on)")
    ap.add_argument("--inject", default=None, help="fault@snapshot, e.g. broken_tooth@300")
    ap.add_argument("--unit", default=None)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args(argv)

    from api import adapters
    from api.adapters import pubsub
    from api.adapters.mqtt import encode_payload, parse_broker

    try:
        broker, topic = adapters.normalise(args.broker, args.topic)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    inject = None
    if args.inject:
        mode, _, at = args.inject.partition("@")
        inject = (mode, int(at or 0))
    frames = (simulated(args.family, args.seed, 16384, inject) if args.family
              else recorded(args.recording, args.channel))
    unit = args.unit or ("g" if args.family == "gearbox" or args.recording else "mm/s")

    if pubsub.is_pubsub(broker):
        from google.cloud import pubsub_v1

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(pubsub.project_of(broker), topic)

        def send(payload: bytes) -> None:
            publisher.publish(topic_path, payload).result(timeout=30)

        def done() -> None:
            pass
    else:
        import paho.mqtt.client as mqtt

        host, port = parse_broker(broker)
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(host, port, keepalive=30)
        client.loop_start()

        def send(payload: bytes) -> None:
            client.publish(topic, payload, qos=1).wait_for_publish(timeout=10)

        def done() -> None:
            client.loop_stop()
            client.disconnect()
    t0 = time.time()
    sent = 0
    try:
        for samples, rate, rpm, slow, t_rel in frames:
            t = t0 + (t_rel if t_rel is not None else sent * args.interval)
            send(encode_payload(samples, rate, t=t, rpm=rpm, unit=unit, slow=slow))
            sent += 1
            if sent % 20 == 0:
                print(f"published {sent}", file=sys.stderr)
            if args.count and sent >= args.count:
                break
            time.sleep(1.0 / args.rate if args.rate > 0 else 0)
    except KeyboardInterrupt:
        pass
    finally:
        done()
    print(f"published {sent} snapshots to {broker} {topic}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
