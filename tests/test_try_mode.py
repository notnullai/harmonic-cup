""""Try Harmonic on your own data" — try-mode routing, the untrusted-input
checks and one end-to-end run (api/try_mode.py, HARM-S8-try).

Most of this needs no compiled engine: the upload caps, the format adapters
and the mode gate are plain Python. The one end-to-end case commissions a
synthetic recording through the real pipeline, like tests/test_machines.py,
and is skipped the same way when the engine or the trained network is not
built.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from api import machines as machines_mod
from api import try_mode
from api.adapters import formats
from kg.store import NetworkXStore
from sim.datasets import recordings


def _needs_engine() -> None:
    from api.rig import find_library

    if find_library() is None:
        pytest.skip("libharmonic_rig not built; run: cmake --build build -j")
    if not machines_mod.TRAINED.exists():
        pytest.skip("trained network not built; run: python scripts/build_brain.py")
    try:
        machines_mod._binary("mint_brain")
        machines_mod._binary("commission")
    except machines_mod.CommissionError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def try_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(try_mode, "TRY_DIR", tmp_path / "try")
    monkeypatch.setattr(try_mode, "KEPT_DIR", tmp_path / "kept")
    for name in ("_RATE", "_RUN_BYTES", "_DISK", "_UPLOADS", "_ACTIVE"):
        monkeypatch.setattr(try_mode, name, {})
    monkeypatch.setattr(try_mode, "_CANCELLED", set())
    monkeypatch.setattr(try_mode, "_INFLIGHT", set())
    return tmp_path / "try"


# --- the mode gate -----------------------------------------------------------------------


def test_try_mode_serves_only_try_routes(try_dir, monkeypatch):
    pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")
    from fastapi.testclient import TestClient

    from api import main as api_main

    monkeypatch.setattr(api_main, "MODE", "try")
    # No real worker thread: it would outlive this test and take other tests' queued runs.
    monkeypatch.setattr(try_mode, "start_background", lambda: None)
    with TestClient(api_main.app) as c:
        assert c.get("/try/limits").status_code == 200
        for path, method in (("/health", "get"), ("/machines", "get"), ("/fleet", "get"),
                             ("/sources", "get"), ("/events", "get")):
            assert getattr(c, method)(path).status_code == 404, path
        assert c.post("/chat", json={"question": "why"}).status_code == 404
        assert c.post("/inject", json={"mode": "bearing_wear"}).status_code == 404
        # "/" redirects to the wizard, not the console.
        r = c.get("/", follow_redirects=False)
        assert r.status_code in (307, 308) and r.headers["location"] == "/try/"


# --- Gemini is unreachable, whatever the environment says --------------------------------


def test_offline_pipeline_never_calls_get_llm(monkeypatch):
    from rag import pipeline as pipeline_mod

    def _boom(*a, **k):
        raise AssertionError("get_llm() must never be called for an offline pipeline")

    monkeypatch.setattr(pipeline_mod, "get_llm", _boom)
    monkeypatch.setenv("GEMINI_API_KEY", "definitely-not-a-real-key")
    p = pipeline_mod.Pipeline(store=NetworkXStore(autoload=False), offline=True)
    assert p.llm is None
    assert p.provider_name == "offline-kg-composer"


def test_networkx_store_with_persist_false_never_writes_back(tmp_path):
    from kg.model import Graph, Node

    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": [], "edges": []}))
    store = NetworkXStore(path, persist=False)
    store.merge_graph(Graph([Node("Equipment", "TRY-X", {"family": "pump"})], []))
    assert store.node("TRY-X") is not None                 # visible in memory
    assert json.loads(path.read_text()) == {"nodes": [], "edges": []}  # never on disk


# --- untrusted input ------------------------------------------------------------------------


def _run(try_dir) -> try_mode.Run:
    return try_mode.new_run("203.0.113.7")


def test_a_pickled_npy_is_rejected(try_dir):
    run = _run(try_dir)
    buf = io.BytesIO()
    # allow_pickle=True on write is the attacker's choice, not ours; the
    # server must refuse to *read* it back as pickle regardless.
    np.save(buf, np.array([{"evil": "payload"}], dtype=object), allow_pickle=True)
    try_mode.save_file(run.id, "run.npy", buf.getvalue())
    with pytest.raises(try_mode.TryError):
        try_mode.configure(run.id, {"family": "pump", "sample_rate_hz": 20000,
                                    "unit": "g", "format": "auto"})


def test_a_traversal_filename_is_rejected(try_dir):
    run = _run(try_dir)
    for bad in ("../x.csv", "..\\x.csv", "/etc/passwd.csv", "a/b.csv", "recording.json",
               ".hidden.csv", "run.json"):
        with pytest.raises(try_mode.TryError):
            try_mode.save_file(run.id, bad, b"1,2,3\n")
    assert not (try_dir.parent / "x.csv").exists()
    assert list(run.upload_folder.iterdir()) == []


def test_an_oversized_upload_is_rejected(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "MAX_TOTAL_BYTES", 100)
    run = _run(try_dir)
    with pytest.raises(try_mode.TryError, match="limit"):
        try_mode.save_file(run.id, "run.csv", b"1,2,3\n" * 100)


def test_a_non_numeric_wide_csv_is_rejected(try_dir):
    run = _run(try_dir)
    text = "1,2,not-a-number\n1,2,3\n"
    try_mode.save_file(run.id, "run.csv", text.encode())
    with pytest.raises(try_mode.TryError, match="non-numeric"):
        try_mode.configure(run.id, {"family": "pump", "sample_rate_hz": 20000, "unit": "g",
                                    "format": "wide_csv"})


def test_too_many_snapshots_is_rejected(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "MAX_SNAPSHOTS", 3)
    run = _run(try_dir)
    rows = "\n".join(",".join("0.1" for _ in range(8)) for _ in range(10))
    try_mode.save_file(run.id, "run.csv", rows.encode())
    with pytest.raises(try_mode.TryError, match="snapshots"):
        try_mode.configure(run.id, {"family": "pump", "sample_rate_hz": 20000, "unit": "g",
                                    "format": "wide_csv"})


def test_an_unknown_family_is_rejected(try_dir):
    run = _run(try_dir)
    with pytest.raises(try_mode.TryError, match="family"):
        try_mode.configure(run.id, {"family": "spaceship", "sample_rate_hz": 20000,
                                    "unit": "g", "format": "auto"})


# --- the wide-CSV adapter --------------------------------------------------------------------


def test_wide_csv_adapter_reads_one_row_per_snapshot(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    rows = [[float(i + j) for j in range(4096)] for i in range(5)]
    text = "\n".join(",".join(str(v) for v in row) for row in rows)
    (folder / "run.csv").write_text(text)
    formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000, "unit": "g"},
                  max_snapshots=100, max_samples=100_000)
    r = recordings.FolderRecording(folder, key="t")
    assert len(r.hours()) == 5 and r.snapshot_samples() == 4096


def test_wide_csv_adapter_reads_a_leading_timestamp_column(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    rows = ["1700000000,1,2,3,4", "1700000600,1,2,3,4"]
    (folder / "run.csv").write_text("\n".join(rows))
    formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000, "unit": "g",
                                       "has_timestamp": True}, max_snapshots=100,
                  max_samples=1000)
    meta = json.loads((folder / "recording.json").read_text())
    assert meta["t_s"] == [1700000000.0, 1700000600.0]


@pytest.mark.parametrize("content, has_ts, match", [
    (b"1,2,3\n1,2\n", False, "same length"),
    (b"1,2,nan\n1,2,3\n", False, "missing, infinite"),
    (b"1,2,inf\n1,2,3\n", False, "missing, infinite"),
    (b"\xff\xfe\x00bad", False, "UTF-8"),
    (b"\n\n  \n", False, "empty"),
    (b"header only\n", False, "no data rows"),
    (b"2026-13-45T00:00,1,2\n", True, "timestamp"),
    (b"1700000000\n", True, "no samples"),
    (b",".join(b"1" for _ in range(50)) + b"\n", False, "exceeds"),
])
def test_wide_csv_adapter_rejects_malformed_input(tmp_path, content, has_ts, match):
    folder = tmp_path / "u"
    folder.mkdir()
    (folder / "run.csv").write_bytes(content)
    with pytest.raises(formats.AdapterError, match=match):
        formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000, "has_timestamp": has_ts},
                      max_snapshots=100, max_samples=40)
    assert not (folder / "data.npy").exists()


def test_wide_csv_adapter_takes_exactly_one_file(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    (folder / "a.csv").write_text("1,2\n")
    (folder / "b.csv").write_text("1,2\n")
    with pytest.raises(formats.AdapterError, match="exactly one"):
        formats.adapt("wide_csv", folder, {}, max_snapshots=10, max_samples=10)


def test_an_unknown_adapter_is_rejected(tmp_path):
    with pytest.raises(formats.AdapterError, match="unknown upload format"):
        formats.adapt("xlsx", tmp_path, {}, max_snapshots=1, max_samples=1)


# --- the WAV adapter -------------------------------------------------------------------------


def _wav(samples: np.ndarray, rate: int = 25600, *, bits: int = 16, code: int = 1,
         extensible: bool = False, channels: int | None = None) -> bytes:
    """A WAV file's bytes, written by hand so every field can be broken."""
    import struct

    a = samples if samples.ndim == 2 else samples[:, None]
    ch = channels if channels is not None else a.shape[1]
    if code == 3:
        data = a.astype("<f4" if bits == 32 else "<f8").tobytes()
    elif bits == 24:
        v = np.round(a * (1 << 23)).astype(np.int64).clip(-(1 << 23), (1 << 23) - 1)
        v = np.where(v < 0, v + (1 << 24), v).astype(np.uint32).ravel()
        data = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1) \
            .astype(np.uint8).tobytes()
    elif bits == 8:
        data = np.round((a + 1.0) * 127.0).astype("<u1").tobytes()
    else:
        dt = {16: "<i2", 32: "<i4"}[bits]
        data = np.round(a * ((1 << (bits - 1)) - 1)).astype(dt).tobytes()
    block = ch * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, ch, rate, rate * block, block, bits, 22, bits,
                          0) + struct.pack("<H", code) + formats._KS_GUID_TAIL
    else:
        fmt = struct.pack("<HHIIHH", code, ch, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt \
        + b"data" + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _tone(n: int, channels: int = 1) -> np.ndarray:
    t = np.arange(n) / 25600.0
    return np.stack([0.5 * np.sin(2 * np.pi * 157.0 * (c + 1) * t) for c in range(channels)],
                    axis=1)


def _adapt_wav(tmp_path, files: dict[str, bytes], **caps):
    folder = tmp_path / "u"
    folder.mkdir()
    for name, b in files.items():
        (folder / name).write_bytes(b)
    formats.adapt("wav", folder, {"sample_rate_hz": 0, "unit": "g", "interval_s": 600.0},
                  max_snapshots=caps.get("max_snapshots", 5000),
                  max_samples=caps.get("max_samples", 262_144))
    return folder


def test_one_long_wav_is_cut_into_snapshots_at_the_header_rate(tmp_path):
    folder = _adapt_wav(tmp_path, {"rec.wav": _wav(_tone(16384 * 6 + 100), rate=25600)})
    r = recordings.FolderRecording(folder, key="t")
    assert len(r.hours()) == 6 and r.snapshot_samples() == 16384
    assert r.sample_rate_hz == 25600
    meta = json.loads((folder / "recording.json").read_text())
    assert meta["interval_s"] == pytest.approx(16384 / 25600)
    assert not (folder / "rec.wav").exists()
    data = np.load(folder / "data.npy")
    assert np.allclose(data[0], _tone(16384)[:, 0], atol=1e-4)


def test_a_long_wav_is_sampled_evenly_when_it_holds_too_many_snapshots(tmp_path):
    folder = _adapt_wav(tmp_path, {"rec.wav": _wav(_tone(16384 * 20))}, max_snapshots=5)
    assert np.load(folder / "data.npy").shape == (5, 16384)
    meta = json.loads((folder / "recording.json").read_text())
    assert meta["interval_s"] == pytest.approx(16384 * 4 / 25600)


def test_several_wavs_are_one_snapshot_each_cut_to_the_shortest(tmp_path):
    files = {f"clip_{i:02d}.wav": _wav(_tone(5000 + 100 * i)) for i in range(4)}
    folder = _adapt_wav(tmp_path, files)
    assert np.load(folder / "data.npy").shape == (4, 5000)


@pytest.mark.parametrize("kw", [dict(bits=24), dict(bits=32), dict(code=3, bits=32),
                                dict(code=3, bits=64), dict(extensible=True, bits=24),
                                dict(extensible=True, code=3, bits=32)])
def test_wav_sample_layouts_decode_to_full_scale(tmp_path, kw):
    folder = _adapt_wav(tmp_path, {"rec.wav": _wav(_tone(16384 * 2), **kw)})
    data = np.load(folder / "data.npy")
    assert np.allclose(data[0], _tone(16384)[:, 0], atol=1e-4)


def test_a_stereo_wav_keeps_its_channels(tmp_path):
    folder = _adapt_wav(tmp_path, {"rec.wav": _wav(_tone(16384 * 2, channels=2))})
    r = recordings.FolderRecording(folder, key="t")
    assert r.n_channels == 2


def _mutate(b: bytes, at: int, new: bytes) -> bytes:
    return b[:at] + new + b[at + len(new):]


_GOOD = _wav(_tone(8192))


@pytest.mark.parametrize("name, content, match", [
    ("not-riff", b"RIFX" + _GOOD[4:], "not a RIFF/WAVE"),
    ("not-wave", _GOOD[:8] + b"AVI " + _GOOD[12:], "not a RIFF/WAVE"),
    ("tiny", b"RIFF", "not a RIFF/WAVE"),
    ("truncated", _GOOD[:-100], "truncated"),
    ("chunk-overrun", _mutate(_GOOD, 40, (10 ** 9).to_bytes(4, "little")), "past the end"),
    ("8-bit", _wav(_tone(8192), bits=8), "8-bit PCM"),
    ("adpcm", _mutate(_GOOD, 20, (2).to_bytes(2, "little")), "encoding 2"),
    ("no-channels", _mutate(_GOOD, 22, (0).to_bytes(2, "little")), "channels"),
    ("nine-channels", _mutate(_GOOD, 22, (9).to_bytes(2, "little")), "channels"),
    ("block-mismatch", _mutate(_GOOD, 32, (3).to_bytes(2, "little")), "frame size"),
    ("zero-rate", _mutate(_GOOD, 24, (0).to_bytes(4, "little")), "not valid"),
    ("too-short", _wav(_tone(1000)), "at least"),
    ("nan", _wav(np.full((16384 * 2, 1), np.nan), code=3, bits=32), "infinite"),
    ("no-data", b"RIFF" + (28).to_bytes(4, "little") + _GOOD[8:36], "no data chunk"),
    # The extensible sub-format GUID sits at bytes 44..60; its shared tail at 46..60.
    ("bad-guid", _mutate(_wav(_tone(8192), extensible=True), 46, b"\x01" * 14), "sub-format"),
    ("odd-frames", _mutate(_GOOD, 40, (8192 * 2 - 1).to_bytes(4, "little")), "whole number"),
])
def test_the_wav_adapter_rejects_malformed_files(tmp_path, name, content, match):
    with pytest.raises(formats.AdapterError, match=match):
        _adapt_wav(tmp_path, {"rec.wav": content})
    assert not (tmp_path / "u" / "data.npy").exists(), name


@pytest.mark.parametrize("order, match", [
    (("data", "fmt"), "before the format"),
    (("fmt", "fmt", "data"), "more than one format"),
    (("fmt", "data", "data"), "more than one data"),
])
def test_the_wav_adapter_rejects_misordered_or_repeated_chunks(tmp_path, order, match):
    import struct

    chunk = {"fmt": _GOOD[12:36], "data": _GOOD[36:]}
    body = b"WAVE" + b"".join(chunk[c] for c in order)
    with pytest.raises(formats.AdapterError, match=match):
        _adapt_wav(tmp_path, {"rec.wav": b"RIFF" + struct.pack("<I", len(body)) + body})


def test_the_wav_adapter_rejects_too_many_chunks(tmp_path):
    import struct

    junk = b"JUNK" + struct.pack("<I", 0)
    body = b"WAVE" + junk * 300 + _GOOD[12:]
    with pytest.raises(formats.AdapterError, match="too many"):
        _adapt_wav(tmp_path, {"rec.wav": b"RIFF" + struct.pack("<I", len(body)) + body})


def test_the_wav_adapter_rejects_mixed_rates(tmp_path):
    with pytest.raises(formats.AdapterError, match="must match"):
        _adapt_wav(tmp_path, {"a.wav": _wav(_tone(8192), rate=25600),
                              "b.wav": _wav(_tone(8192), rate=48000)})


def test_the_wav_adapter_rejects_other_files_beside_the_wavs(tmp_path):
    with pytest.raises(formats.AdapterError, match=r"only \.wav"):
        _adapt_wav(tmp_path, {"a.wav": _GOOD, "b.csv": b"1,2\n"})


def test_a_wav_upload_runs_through_configure_at_its_own_rate(try_dir):
    n = try_mode.machines_mod.MIN_BASELINE + try_mode.MIN_WATCHED_SNAPSHOTS
    run = _run(try_dir)
    try_mode.save_file(run.id, "rec.wav", _wav(_tone(16384 * n), rate=25600))
    # The form's rate is ignored for WAV: the header's is used.
    ov = try_mode.configure(run.id, {"family": "other", "sample_rate_hz": 0, "unit": "g",
                                     "format": "auto"})
    assert ov["sample_rate_hz"] == 25600 and ov["snapshots"] == n
    assert ov["default_baseline_h"][1] == pytest.approx(
        ov["hours"][ov["min_baseline_snapshots"] - 1], abs=1e-4)
    run = try_mode.get(run.id)
    assert run.format == "wav" and run.sample_rate_hz == 25600
    # Configuring again (say, a different family) keeps what the adapter read.
    ov = try_mode.configure(run.id, {"family": "pump", "sample_rate_hz": 1000, "unit": "mm/s"})
    assert ov["sample_rate_hz"] == 25600 and ov["snapshots"] == n
    assert any("WAV samples" in c for c in try_mode._caveats(try_mode.get(run.id)))


def test_a_recording_too_short_to_learn_normal_is_refused_at_configure(try_dir):
    run = _run(try_dir)
    try_mode.save_file(run.id, "rec.wav", _wav(_tone(16384 * 12)))
    with pytest.raises(try_mode.TryError, match="needs at least"):
        try_mode.configure(run.id, {"family": "other", "unit": "g", "format": "wav"})


# --- the rejected-format log --------------------------------------------------------------------


def _rejections(caplog) -> list[dict]:
    return [json.loads(r.getMessage().split("try-rejected-format ", 1)[1])
            for r in caplog.records if "try-rejected-format" in r.getMessage()]


def test_a_refused_extension_is_logged_without_its_name(try_dir, caplog):
    caplog.set_level("WARNING", logger="harmonic.try.formats")
    run = _run(try_dir)
    with pytest.raises(try_mode.TryError, match="tell us"):
        try_mode.save_file(run.id, "plant-secret-name.mat", b"MATLAB 5.0 MAT-file")
    (entry,) = _rejections(caplog)
    assert entry["stage"] == "upload" and entry["ext"] == ".mat"
    assert entry["magic"] == b"MATLAB 5".hex()
    assert "plant-secret-name" not in caplog.text


def test_an_adapter_refusal_is_logged_without_the_file_name(try_dir, caplog):
    caplog.set_level("WARNING", logger="harmonic.try.formats")
    run = _run(try_dir)
    try_mode.save_file(run.id, "acme-line-4.wav", b"RIFF\x00\x00\x00\x00junk")
    with pytest.raises(try_mode.TryError, match="RIFF"):
        try_mode.configure(run.id, {"family": "pump", "unit": "g", "format": "wav"})
    (entry,) = _rejections(caplog)
    assert entry["stage"] == "adapter" and entry["format"] == "wav" and entry["ext"] == ".wav"
    assert "acme-line-4" not in caplog.text and "<file>" in entry["reason"]


def test_run_creation_is_rate_limited_per_address(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "RUNS_CREATED_PER_HOUR", 2)
    try_mode.new_run("192.0.2.9")
    try_mode.new_run("192.0.2.9")
    with pytest.raises(try_mode.TryError, match="per hour"):
        try_mode.new_run("192.0.2.9")
    try_mode.new_run("192.0.2.10")          # another address is not affected


def test_families_come_from_the_registry():
    from kg import families

    assert set(try_mode.FAMILIES) == set(families.families())
    assert all(k in try_mode.FAMILY_DEFAULT_RPM for k in try_mode.FAMILIES)


def test_manual_citations_are_stripped_from_a_try_diagnosis():
    diag = {"provider": "offline-kg-composer", "retrieved": ["P-101-OM-5.2", "FM-7"],
            "response": {
                "diagnosis": "Outer race defect", "confidence": 0.7,
                "root_causes": ["Lubrication (P-101-OM-5.2)"],
                "actions": [{"step": "Grease the bearing (see P-101-OM-5.3, P-101-OM-5.4)",
                             "tool": "grease gun"}],
                "safety": ["Lock out per P-101-OM-5.4"],
                "references": ["P-101-OM-5.3", "P-101-OM-5.4"]}}
    out = try_mode._without_manual_citations(diag)
    text = json.dumps(out)
    assert "P-101" not in text and "(see" not in text and "retrieved" not in out
    assert out["response"]["actions"][0]["step"] == "Grease the bearing"
    assert out["response"]["guidance_note"] == try_mode.GUIDANCE_NOTE
    assert "references" in diag["response"]           # the console's copy is untouched


def test_limits_state_the_minimum_recording_length():
    lim = try_mode.limits()
    assert lim["min_snapshots"] == try_mode.machines_mod.MIN_BASELINE         + try_mode.MIN_WATCHED_SNAPSHOTS == 180
    assert lim["wav_snapshot_samples"] == 16384


# --- off by default ------------------------------------------------------------------------------


def _client(monkeypatch, *, mode: str, beside: bool):
    pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")
    from fastapi.testclient import TestClient

    from api import main as api_main

    monkeypatch.setattr(api_main, "MODE", mode)
    monkeypatch.setattr(api_main, "TRY_BESIDE_CONSOLE", beside)
    monkeypatch.setattr(api_main, "AUTOSTART", False)
    started: list[bool] = []
    monkeypatch.setattr(try_mode, "start_background", lambda: started.append(True))
    return TestClient(api_main.app), started


def test_try_mode_is_off_by_default_in_the_console(try_dir, monkeypatch):
    c, started = _client(monkeypatch, mode="console", beside=False)
    with c:
        for path in ("/try", "/try/", "/try/limits", "/try/index.html"):
            assert c.get(path).status_code == 404, path
        assert c.post("/try/runs").status_code == 404
        assert c.get("/health").status_code == 200
    assert started == []
    assert not try_dir.exists()


def test_harmonic_try_switches_the_routes_on_beside_the_console(try_dir, monkeypatch):
    c, started = _client(monkeypatch, mode="console", beside=True)
    with c:
        assert c.get("/try/limits").status_code == 200
        assert c.get("/health").status_code == 200
    assert started == [True]


def test_an_upload_over_the_cap_is_refused_before_it_is_buffered(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "MAX_TOTAL_BYTES", 1000)
    c, _ = _client(monkeypatch, mode="try", beside=False)
    with c:
        rid = c.post("/try/runs").json()["id"]
        r = c.put(f"/try/runs/{rid}/uploads/a.csv", content=b"1," * 1000)
        assert r.status_code == 413
        r = c.put(f"/try/runs/{rid}/uploads/a.csv", content=b"1,2\n")
        assert r.status_code == 200


# --- deletion --------------------------------------------------------------------------------


def test_delete_now_removes_the_run(try_dir):
    run = _run(try_dir)
    assert run.folder.exists()
    try_mode.delete_run(run.id)
    assert not run.folder.exists()
    assert try_mode.get(run.id) is None


def test_the_sweeper_deletes_expired_runs_but_keeps_consented_ones(try_dir):
    old = _run(try_dir)
    old.created -= try_mode.RETENTION_DAYS * 86400 + 3600
    try_mode._save(old)

    # Consented AND finished: the sweeper moves it to kept/ instead of
    # deleting it (HARM-S10-T05).
    kept = _run(try_dir)
    kept.created -= try_mode.RETENTION_DAYS * 86400 + 3600
    kept.consent = True
    kept.status = "done"
    try_mode._save(kept)

    fresh = _run(try_dir)

    n = try_mode.sweep()
    assert n == 1
    assert try_mode.get(old.id) is None
    reloaded = try_mode.get(kept.id)
    assert reloaded is not None
    assert reloaded.kept is True
    assert reloaded.folder == try_mode.KEPT_DIR / kept.id
    assert try_mode.get(fresh.id) is not None


# --- kept/ prefix (HARM-S10-T05) --------------------------------------------------------------


def test_an_unkept_run_is_deleted_after_the_normal_retention(try_dir):
    run = _run(try_dir)
    run.created -= try_mode.RETENTION_DAYS * 86400 + 1
    try_mode._save(run)
    assert try_mode.sweep() == 1
    assert try_mode.get(run.id) is None


def test_a_kept_run_survives_the_7_day_sweep_and_expires_after_90_days(try_dir):
    run = _run(try_dir)
    run.consent, run.status = True, "done"
    run.created -= try_mode.RETENTION_DAYS * 86400 + 1     # past the 7-day cutoff
    try_mode._save(run)

    assert try_mode.sweep() == 0                            # moved, not deleted
    moved = try_mode.get(run.id)
    assert moved is not None and moved.kept is True
    assert moved.folder == try_mode.KEPT_DIR / run.id
    assert not (try_mode.TRY_DIR / run.id).exists()

    moved.created -= try_mode.CONSENT_RETENTION_DAYS * 86400
    try_mode._save(moved)
    assert try_mode.sweep() == 1
    assert try_mode.get(run.id) is None


def test_unticking_before_the_run_finishes_keeps_it_under_the_normal_prefix(try_dir):
    # consent stays False, so a finished run is never moved even after the
    # sweeper visits it (HARM-S10-T05: "unticking before the move").
    run = _run(try_dir)
    run.status = "done"
    try_mode._save(run)
    try_mode.sweep()
    still = try_mode.get(run.id)
    assert still is not None and still.kept is False
    assert still.folder == try_mode.TRY_DIR / run.id


def test_delete_now_on_a_kept_run(try_dir):
    run = _run(try_dir)
    run.consent, run.status = True, "done"
    try_mode._save(run)
    try_mode._move_to_kept(run)
    assert run.kept is True
    assert (try_mode.KEPT_DIR / run.id).exists()

    try_mode.delete_run(run.id)
    assert try_mode.get(run.id) is None
    assert not (try_mode.KEPT_DIR / run.id).exists()
    assert not (try_mode.TRY_DIR / run.id).exists()


def test_the_disk_quota_counts_both_the_normal_and_kept_prefixes(try_dir):
    run = _run(try_dir)
    try_mode.save_file(run.id, "a.csv", b"0" * 5000)
    run.consent, run.status = True, "done"
    try_mode._save(run)
    try_mode._move_to_kept(run)
    assert run.kept is True

    # Ground truth: what is really on disk under each prefix, compared
    # against the running total the quota checks read (HARM-S10-T05).
    real_total = try_mode._du(try_mode.TRY_DIR) + try_mode._du(try_mode.KEPT_DIR)
    assert try_mode._disk_used() == real_total
    assert try_mode._run_used(run.id) == try_mode._du(try_mode.KEPT_DIR / run.id)

    other = try_mode.new_run("203.0.113.9")
    assert try_mode._disk_used() == real_total + try_mode._du(try_mode.TRY_DIR / other.id)


# --- one end-to-end run ------------------------------------------------------------------------


def test_a_try_run_commissions_and_produces_a_report(try_dir, monkeypatch):
    _needs_engine()
    # A private queue: no worker thread can take this run from under the test.
    import queue

    monkeypatch.setattr(try_mode, "_QUEUE", queue.Queue(maxsize=try_mode.QUEUE_DEPTH))
    from sim.pump_sim import PumpSimulator

    sim = PumpSimulator(seed=7, fs_vib=25600.0)
    window = 16384
    n_baseline, n_fault = 200, 60
    frames = [sim.step(window / 25600.0).vibration.astype(np.float32)
             for _ in range(n_baseline)]
    sim.inject("bearing_wear", severity=1.0, onset_s=0.0, ramp_hours=0.0)
    frames += [sim.step(window / 25600.0).vibration.astype(np.float32)
              for _ in range(n_fault)]
    arr = np.stack(frames)

    run = try_mode.new_run("198.51.100.4")
    buf = io.BytesIO()
    np.save(buf, arr)
    try_mode.save_file(run.id, "run.npy", buf.getvalue())
    overview = try_mode.configure(run.id, {"family": "pump", "sample_rate_hz": 25600,
                                           "rpm": 2950, "unit": "mm/s", "format": "auto"})
    assert overview["snapshots"] == n_baseline + n_fault

    baseline_hi = overview["hours"][n_baseline - 1]
    started = try_mode.start(run.id, {"baseline_h": [overview["hours"][0], baseline_hi],
                                      "consent": False})
    assert started["status"] == "queued"

    # Run synchronously, deterministically — the same pipeline the worker
    # thread calls (`_run_with_timeout`), without the thread and the timeout.
    run = try_mode.get(run.id)
    try_mode._do_run(run)
    run = try_mode.get(run.id)

    assert run.status == "done", run.error
    report = run.report
    assert report["first_warning"] is not None or report["first_critical"] is not None
    assert report["evidence"]
    assert report["diagnosis"] is not None
    assert report["diagnosis"]["provider"] == "offline-kg-composer"
    # No manual that is not the visitor's own is cited (owner, 2026-09-29).
    assert "references" not in report["diagnosis"]["response"]
    assert "P-101" not in json.dumps(report) and "(see " not in json.dumps(report)
    assert report["diagnosis"]["response"]["guidance_note"] == try_mode.GUIDANCE_NOTE
    assert report["false_alarms_in_baseline"] == 0, report
    assert report["contact_url"] is None       # HARMONIC_CONTACT_URL unset in tests
    # S10-T06: one plain answer, a chart for every machine type, no health index.
    assert report["headline"].startswith("Harmonic ")
    assert "health index" not in json.dumps(report).lower() and "hhi" not in report
    pts = [p["d"] for p in report["chart"]["points"] if p["d"] is not None]
    assert pts and max(pts) > report["chart"]["alarm"]
    assert report["first_alarm"] is not None
    assert report["chart"]["first_alarm_h"] == report["first_alarm"]["hours"]
    assert all(f"{try_mode.TIMES} its normal range" in e for e in report["evidence"])

    try_mode.delete_run(run.id)
    assert try_mode.get(run.id) is None


# --- S10-T06: plain words, the probe, the sample ------------------------------------------------


def _rows_csv(n: int = 200, width: int = 4096) -> bytes:
    rng = np.random.default_rng(3)
    return "\n".join(",".join(f"{v:.3f}" for v in rng.normal(size=width))
                     for _ in range(n)).encode()


def test_a_probe_says_what_the_file_holds_before_anything_is_asked(try_dir, monkeypatch):
    c, _ = _client(monkeypatch, mode="try", beside=False)
    with c:
        rid = c.post("/try/runs").json()["id"]
        assert c.put(f"/try/runs/{rid}/uploads/rows.csv", content=_rows_csv()).status_code == 200
        # No machine type, no sampling rate: the probe field must reach try_mode
        # through the route's request model.
        r = c.post(f"/try/runs/{rid}/configure", json={"probe": True, "format": "wide_csv"})
        assert r.status_code == 200, r.text
        p = r.json()
        assert p["probe"] is True and p["snapshots"] == 200
        assert p["samples_per_snapshot"] == 4096 and p["n_channels"] == 1
        assert p["timed_by"] == "interval" and p["interval_s"] is None
        assert p["sample_rate_hz"] is None                  # the stand-in is never shown
        assert c.get(f"/try/runs/{rid}").json()["status"] == "collecting"
        r = c.post(f"/try/runs/{rid}/configure", json={
            "family": "generic", "sample_rate_hz": 25600, "format": "wide_csv",
            "interval_s": 60})
        assert r.status_code == 200, r.text
        assert r.json()["interval_s"] == 60
        # Once configured, a probe is refused rather than rewriting the settings.
        r = c.post(f"/try/runs/{rid}/configure", json={"probe": True})
        assert r.status_code == 422


def test_the_sample_goes_through_the_same_upload_path(try_dir, monkeypatch):
    if not try_mode.SAMPLE_PATH.is_file():
        pytest.skip("the sample recording is not in this checkout")
    c, _ = _client(monkeypatch, mode="try", beside=False)
    with c:
        rid = c.post("/try/runs").json()["id"]
        r = c.post(f"/try/runs/{rid}/sample")
        assert r.status_code == 200, r.text
        info = r.json()
        assert "Cincinnati" in info["credit"] and "NASA" in info["credit"]
        assert c.post(f"/try/runs/{rid}/sample").status_code == 422   # once per run
        p = c.post(f"/try/runs/{rid}/configure", json={"probe": True}).json()
        assert p["snapshots"] == 492 and p["samples_per_snapshot"] == 16384
        body = {k: info[k] for k in ("family", "sample_rate_hz", "rpm", "interval_s", "unit")}
        r = c.post(f"/try/runs/{rid}/configure", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["interval_s"] == info["interval_s"]
        assert c.get(f"/try/runs/{rid}").json()["sample"] is True


def test_human_times():
    assert try_mode.human_duration(1.35) == "1 h 21 min"
    assert try_mode.human_duration(2.5333) == "2 h 32 min"
    assert try_mode.human_duration(0.25) == "15 min"
    assert try_mode.human_duration(20 / 3600) == "20 seconds"
    assert try_mode.human_duration(26.4) == "26 hours"
    assert try_mode.human_duration(60.3) == "2.5 days"
    assert try_mode.human_span(5.63) == "5.6 hours"
    assert try_mode.human_span(163.7) == "6.8 days"


def test_the_public_diagnosis_has_no_health_index_and_no_internal_id():
    diag = {"top_candidate": None, "response": {
        "diagnosis": "Unrecognised change on TRY-ABC: it has moved away from its own learned "
                     "normal. The health index is 0.00 (WATCH).",
        "actions": [{"step": "Check TRY-ABC"}], "rul_hours": 5.0}}
    out = try_mode._plain_diagnosis(diag, "TRY-ABC")
    text = json.dumps(out)
    assert "health index" not in text and "TRY-ABC" not in text and "rul_hours" not in text
    assert out["response"]["diagnosis"].startswith("Unrecognised change on your machine")
    named = {"top_candidate": "FM-X", "response": {
        "diagnosis": "Outer race spalling on TRY-ABC. The health index is 0.71 (WARNING) in "
                     "the steady regime."}}
    named = try_mode._plain_diagnosis(named, "TRY-ABC")
    assert named["response"]["diagnosis"] == "Outer race spalling on your machine."


def test_the_headline_answers_in_plain_words():
    watch = {"hours": 4.28, "before_end_h": 1.35, "before_end": "1 h 21 min"}
    assert try_mode._headline({}, True, None) == \
        "Harmonic saw nothing unusual in this recording."
    other = try_mode._headline({"WATCH": watch}, False, None)
    assert other == ("Harmonic noticed a change 1 h 21 min before the end of your recording, "
                     "and would have warned you. It can't name the fault type, because you "
                     "chose Other.")
    diag = {"top_candidate": "FM-X",
            "response": {"diagnosis": "Outer race spalling on your machine."}}
    named = try_mode._headline({"WARNING": watch}, True, diag)
    assert named.startswith("Harmonic would have warned you 1 h 21 min before the end")
    assert named.endswith("What it points to: outer race spalling.")
