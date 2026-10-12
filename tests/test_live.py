"""Live sensors over MQTT (HARM-S7-T08).

The payload checks run anywhere. The end-to-end cases start a real mosquitto
broker on a free port and are skipped where none is installed
(`sudo apt-get install mosquitto`).
"""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from api.adapters.mqtt import encode_payload, parse_broker, parse_payload, valid_topic
from sim.datasets import recordings

# --- the payload, no broker needed ----------------------------------------------------


def test_a_snapshot_round_trips_bit_for_bit():
    x = np.random.default_rng(1).normal(size=8192).astype(np.float32)
    snap = parse_payload(encode_payload(x, 25600, t=1_727_000_000.5, rpm=1480.0, unit="g",
                                        slow=[61.5, None, 2.0]))
    assert np.array_equal(snap.samples, x) and snap.sample_rate_hz == 25600
    assert snap.t == 1_727_000_000.5 and snap.rpm == 1480.0 and snap.unit == "g"
    assert snap.slow[:3] == [61.5, 0.0, 2.0] and snap.present == 0b101


@pytest.mark.parametrize("msg, reason", [
    (b"not json", "not JSON"),
    (b"[1, 2]", "JSON object"),
    (json.dumps({"sample_rate_hz": 25600}).encode(), "no samples"),
    (json.dumps({"sample_rate_hz": 7, "samples": [0.0] * 4096}).encode(), "sample_rate_hz"),
    (json.dumps({"sample_rate_hz": 25600, "samples": [0.0] * 100}).encode(), "100 samples"),
    (json.dumps({"sample_rate_hz": 25600, "samples_b64": "***"}).encode(), "base64"),
    (json.dumps({"sample_rate_hz": 25600,
                 "samples_b64": base64.b64encode(np.full(4096, np.nan, "<f4").tobytes())
                 .decode()}).encode(), "NaN"),
    (json.dumps({"sample_rate_hz": 25600, "samples": [0.0] * 4096, "rpm": -5}).encode(), "rpm"),
    (json.dumps({"sample_rate_hz": 25600, "samples": [0.0] * 4096,
                 "slow": [0] * 9}).encode(), "at most 7"),
    (b"x" * (5 * 1024 * 1024), "exceeds"),
])
def test_a_bad_message_is_refused_with_its_reason(msg, reason):
    with pytest.raises(ValueError, match=reason):
        parse_payload(msg)


def test_brokers_and_topics_are_checked():
    assert parse_broker("mqtt.local:1884") == ("mqtt.local", 1884)
    assert parse_broker("mqtt.local") == ("mqtt.local", 1883)
    with pytest.raises(ValueError):
        parse_broker("host:99999")
    assert valid_topic("plant/gb1/vib")
    for bad in ("plant/+/vib", "plant/#", "", "$SYS/x", "a//b"):
        assert not valid_topic(bad)


def test_a_growing_folder_of_snapshots_reads_as_a_recording(tmp_path):
    """What a listener writes: one file per snapshot, the timestamps one ahead
    of the files while a snapshot is being written."""
    for i in range(3):
        np.save(tmp_path / f"snap_{i + 1:06d}.npy", np.full(4096, i, np.float32))
    (tmp_path / "recording.json").write_text(json.dumps({
        "sample_rate_hz": 25600, "t_s": [1000.0, 1060.0, 1120.0, 1180.0],
        "rpm": [1480, 1481, 1482, 1483]}))
    r = recordings.FolderRecording(tmp_path)
    assert r.snapshot_samples() == 4096 and list(r.hours() * 60) == [0.0, 1.0, 2.0]
    assert list(r.speeds()) == [1480, 1481, 1482]
    assert r.load(0, 4096)[2, 0] == 2.0


# --- with a broker ----------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def broker(tmp_path):
    exe = shutil.which("mosquitto") or next(
        (p for p in ("/usr/sbin/mosquitto", "/usr/bin/mosquitto") if Path(p).exists()), None)
    if exe is None:
        pytest.skip("no mosquitto broker installed (sudo apt-get install mosquitto)")
    port = _free_port()
    conf = tmp_path / "mosquitto.conf"
    conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\n")
    proc = subprocess.Popen([exe, "-c", str(conf)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.skip("mosquitto did not start")
    yield f"127.0.0.1:{port}"
    proc.terminate()
    proc.wait(5)


def _publish(broker: str, topic: str, frames, t0: float = 1_727_000_000.0) -> None:
    import paho.mqtt.client as mqtt

    host, port = parse_broker(broker)
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.connect(host, port)
    c.loop_start()
    for i, x in enumerate(frames):
        c.publish(topic, encode_payload(x, 25600, t=t0 + 600.0 * i), qos=1).wait_for_publish(5)
    c.loop_stop()
    c.disconnect()


def _wait(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.05)
    return cond()


def test_a_listener_records_a_topic_into_a_recording(broker, tmp_path, monkeypatch):
    from api import live, machines

    monkeypatch.setattr(machines, "UPLOADS_DIR", tmp_path / "uploads")
    frames = [np.random.default_rng(i).normal(size=8192).astype(np.float32) for i in range(5)]
    uid, rec = live.listen(broker, "test/vib", "a live pump")
    try:
        assert _wait(lambda: rec.subscriber.connected)
        _publish(broker, "test/vib", frames)
        assert _wait(lambda: rec.count == 5), rec.status()
        r = machines.recording(f"upload:{uid}")
        assert np.array_equal(r.load(0, 8192), np.stack(frames))
        assert list(r.hours() * 6) == [0.0, 1.0, 2.0, 3.0, 4.0]      # 10-minute spacing
    finally:
        live.stop(uid)


def test_live_gives_the_same_states_as_the_recording(broker, tmp_path):
    """The acceptance case: the same snapshots, replayed from a folder or
    arriving over MQTT, go through the engine identically."""
    from api.bridge import DEFAULT_BRAIN, EngineBridge
    from api.rig import find_library
    from api.sources import LiveSource, RecordedSource
    from sim.pump_sim import PumpSimulator

    if find_library() is None or not DEFAULT_BRAIN.exists():
        pytest.skip("engine or pump brain not built")
    sim = PumpSimulator(seed=5, fs_vib=25600.0)
    sim.time_scale = 60.0
    frames = []
    for i in range(90):
        if i == 60:
            sim.inject("misalignment", severity=1.0, onset_s=sim.t_s)
        frames.append(sim.step(16384 / 25600).vibration.astype(np.float32))
    folder = tmp_path / "rec"
    folder.mkdir()
    np.save(folder / "run.npy", np.stack(frames))
    (folder / "recording.json").write_text(json.dumps(
        {"sample_rate_hz": 25600, "interval_s": 600, "unit": "mm/s"}))
    common = dict(brain_path=DEFAULT_BRAIN, vib_unit="mm/s", equipment_id="P-101",
                  nominal_rpm=2950.0)
    recorded = RecordedSource(key="r", name="r", detail="", channel=0,
                              recording=recordings.FolderRecording(folder), **common)
    live = LiveSource(key="l", name="l", broker=broker, topic="test/p101",
                      window_samples=16384, sample_rate_hz=25600, backlog=256, **common)
    assert _wait(lambda: live.subscriber.connected)
    _publish(broker, "test/p101", frames)
    assert _wait(lambda: live.subscriber.received == 90)

    def states(src, n):
        b = EngineBridge(DEFAULT_BRAIN, source="simulated")
        b.rig.close()
        from api.rig import HarmonicRig
        b.source, b.rig = src, HarmonicRig(DEFAULT_BRAIN, window_samples=16384,
                                           sample_rate_hz=25600)
        out = []
        while len(out) < n:
            snap = b.step()
            if snap is None:
                time.sleep(0.02)
                continue
            out.append((snap.state, round(snap.hhi, 6)))
        b.stop()
        return out

    live_states = states(live, 90)
    assert live.dropped == 0 and live.refused == 0
    assert live_states == states(recorded, 90)
    assert any(s != "OK" for s, _ in live_states)            # the fault was seen


def test_a_broker_string_picks_its_transport(monkeypatch):
    from api import adapters
    from api.adapters import pubsub

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    assert adapters.normalise("localhost", "plant/gb1/vib") == ("localhost:1883", "plant/gb1/vib")
    assert adapters.normalise("pubsub", "gb7-vibration") == ("pubsub:demo-project", "gb7-vibration")
    assert adapters.normalise("pubsub:other", "gb7-vibration")[0] == "pubsub:other"
    with pytest.raises(ValueError, match="Pub/Sub topic id"):
        adapters.normalise("pubsub", "plant/gb1/vib")          # MQTT-style levels
    with pytest.raises(ValueError, match="wildcards"):
        adapters.normalise("localhost:1883", "plant/#")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    with pytest.raises(ValueError, match="needs a project"):
        adapters.normalise("pubsub", "gb7-vibration")
    assert pubsub.subscription_id("gb7-vibration", "machine-gb 7") == \
        "harmonic-gb7-vibration-machine-gb-7"
