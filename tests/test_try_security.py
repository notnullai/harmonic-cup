"""HARM-S10-T03: one regression test per finding of the security review of the
public "Try it" service (api/try_mode.py, api/main.py, the format adapters).

Test names carry the finding's id (h1, m2, l4, g3 ...), so the review's
list and this file can be read side by side. The review's own proof-of-
concept scripts live in var/t03/ and are run by hand (see docs/TRY_IT_SPEC.md).
None of these needs the compiled engine: the worker's commissioning step is
replaced by a stand-in where a run has to be in flight.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tracemalloc

import numpy as np
import pytest

from api import machines as machines_mod
from api import try_mode
from api.adapters import formats
from sim.datasets import recordings

PUMP = {"family": "pump", "sample_rate_hz": 20000, "unit": "g", "format": "auto"}


@pytest.fixture
def try_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(try_mode, "TRY_DIR", tmp_path / "try")
    monkeypatch.setattr(try_mode, "KEPT_DIR", tmp_path / "kept")
    for name in ("_RATE", "_RUN_BYTES", "_DISK", "_UPLOADS", "_ACTIVE"):
        monkeypatch.setattr(try_mode, name, {})
    monkeypatch.setattr(try_mode, "_CANCELLED", set())
    monkeypatch.setattr(try_mode, "_INFLIGHT", set())
    monkeypatch.setattr(try_mode, "_QUEUE", queue.Queue(maxsize=try_mode.QUEUE_DEPTH))
    return tmp_path / "try"


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _snapshots(n: int = 200, samples: int = 4096, channels: int = 1) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (n, samples) if channels == 1 else (n, samples, channels)
    return rng.normal(0, 0.1, shape).astype(np.float32)


def _configured(ip: str = "203.0.113.7", **cfg) -> try_mode.Run:
    run = try_mode.new_run(ip)
    try_mode.save_file(run.id, "run.npy", _npy(_snapshots()))
    try_mode.configure(run.id, {**PUMP, **cfg})
    return try_mode.get(run.id)


def _client(monkeypatch, mode: str = "try"):
    pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")
    from fastapi.testclient import TestClient

    from api import main as api_main

    monkeypatch.setattr(api_main, "MODE", mode)
    monkeypatch.setattr(api_main, "TRY_BESIDE_CONSOLE", mode != "try")
    monkeypatch.setattr(api_main, "AUTOSTART", False)
    monkeypatch.setattr(try_mode, "start_background", lambda: None)
    return TestClient(api_main.app, raise_server_exceptions=False)


def _peak_mb(fn) -> float:
    tracemalloc.start()
    with contextlib.suppress(ValueError, formats.AdapterError):
        fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


# --- H1: a small CSV must not exhaust memory ----------------------------------------------------


def test_h1_a_text_snapshot_of_short_lines_is_refused_while_streaming(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    (folder / "a.csv").write_bytes(b"0\n" * 3_000_000)        # 6 MB, 3 M rows
    (folder / "recording.json").write_text(json.dumps({"sample_rate_hz": 25600}))
    with pytest.raises(ValueError, match="rows"):
        recordings.FolderRecording(folder)
    # The old reader built a list of 3 M lists of strings (~0.5 GB).
    assert _peak_mb(lambda: recordings.FolderRecording(folder)) < 64


def test_h1_the_wide_csv_adapter_streams_rows_to_disk(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    row = ",".join(["0.125"] * 2000) + "\n"
    (folder / "a.csv").write_text(row * 2000)                 # 4 M samples, 16 MB as float32
    peak = _peak_mb(lambda: formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000},
                                          max_snapshots=5000, max_samples=262144))
    assert np.load(folder / "data.npy", mmap_mode="r").shape == (2000, 2000)
    assert peak < 8, f"peak {peak:.1f} MB: rows were held in memory"


def test_h1_one_endless_wide_row_is_refused_by_its_length(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    (folder / "a.csv").write_bytes(b"0," * 3_000_000)
    with pytest.raises(formats.AdapterError, match=r"exceeds the limit|longer than"):
        formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000},
                      max_snapshots=5000, max_samples=262144)
    assert _peak_mb(lambda: formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000},
                                          max_snapshots=5000, max_samples=262144)) < 40


def test_h1_the_wide_csv_total_samples_are_capped(tmp_path):
    folder = tmp_path / "u"
    folder.mkdir()
    (folder / "a.csv").write_text(("1," * 99 + "1\n") * 50)
    with pytest.raises(formats.AdapterError, match="samples in all"):
        formats.adapt("wide_csv", folder, {"sample_rate_hz": 20000}, max_snapshots=5000,
                      max_samples=1000, max_total_samples=1000)
    assert not (folder / "data.npy").exists()


def test_h1_configure_is_rate_limited_per_run(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "CONFIGURES_PER_RUN_PER_HOUR", 3)
    run = try_mode.new_run("203.0.113.7")
    try_mode.save_file(run.id, "a.csv", b"0.1\n" * 10)
    for _ in range(3):
        with pytest.raises(try_mode.TryError):
            try_mode.configure(run.id, PUMP)
    with pytest.raises(try_mode.RateLimited):
        try_mode.configure(run.id, PUMP)


# --- H2: the rate limit counts the visitor, not Google's proxy -----------------------------------


def test_h2_the_rightmost_forwarded_for_entry_is_the_visitor(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "RUNS_CREATED_PER_HOUR", 2)
    with _client(monkeypatch) as c:
        # A spoofed left entry changes nothing: the right one is who asks.
        codes = [c.post("/try/runs", headers={"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.7"})
                 .status_code for i in range(4)]
        assert codes == [200, 200, 429, 429]
        # Different visitors behind the same proxy each get their own limit.
        assert c.post("/try/runs", headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200


def test_h2_the_console_never_reads_forwarded_for(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "RUNS_CREATED_PER_HOUR", 1)
    with _client(monkeypatch, mode="console") as c:
        assert c.post("/try/runs", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert c.post("/try/runs", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_h2_the_instance_has_a_global_cap_on_runs(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "GLOBAL_RUNS_CREATED_PER_HOUR", 3)
    for i in range(3):
        try_mode.new_run(f"192.0.2.{i}")
    with pytest.raises(try_mode.RateLimited, match="this hour"):
        try_mode.new_run("192.0.2.99")
    monkeypatch.setattr(try_mode, "GLOBAL_RUNS_CREATED_PER_HOUR", 100)
    monkeypatch.setattr(try_mode, "GLOBAL_RUNS_STARTED_PER_HOUR", 1)
    a, b = _configured("192.0.2.50"), _configured("192.0.2.51")
    try_mode.start(a.id, {"baseline_h": [0.0, 30.0]})
    with pytest.raises(try_mode.RateLimited, match="this hour"):
        try_mode.start(b.id, {"baseline_h": [0.0, 30.0]})


# --- H3: storage is bounded -----------------------------------------------------------------------


def test_h3_parallel_uploads_cannot_pass_the_run_cap(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "MAX_TOTAL_BYTES", 100_000)
    run = try_mode.new_run("203.0.113.7")
    results: list[str] = []

    def put(i: int) -> None:
        try:
            try_mode.save_file(run.id, f"f{i}.npy", b"0" * 30_000)
            results.append("ok")
        except try_mode.TooLarge:
            results.append("413")

    ts = [threading.Thread(target=put, args=(i,)) for i in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    on_disk = sum(p.stat().st_size for p in run.upload_folder.iterdir())
    assert results.count("ok") == 3 and on_disk == 90_000


def test_h3_a_global_disk_quota_refuses_new_runs_and_uploads(try_dir, monkeypatch):
    run = try_mode.new_run("203.0.113.7")
    monkeypatch.setattr(try_mode, "DISK_QUOTA_BYTES", try_mode._disk_used() + 1000)
    with pytest.raises(try_mode.Busy, match="full"):
        try_mode.save_file(run.id, "a.csv", b"0" * 2000)
    monkeypatch.setattr(try_mode, "MAX_TOTAL_BYTES", 500)
    try_mode.save_file(run.id, "a.csv", b"0" * 400)
    monkeypatch.setattr(try_mode, "DISK_QUOTA_BYTES", try_mode._disk_used() + 100)
    with pytest.raises(try_mode.Busy, match="full"):
        try_mode.new_run("203.0.113.8")


def test_h3_derived_files_count_against_the_run_cap(try_dir, monkeypatch):
    run = try_mode.new_run("203.0.113.7")
    rows = "\n".join(",".join("0.1" for _ in range(4096)) for _ in range(200))
    try_mode.save_file(run.id, "day.csv", rows.encode())
    uploaded = try_mode._run_used(run.id)
    # Room for the upload, not for data.npy and its cache beside it.
    monkeypatch.setattr(try_mode, "MAX_RUN_DISK_BYTES", uploaded + 1000)
    with pytest.raises(try_mode.TooLarge, match="derives"):
        try_mode.configure(run.id, {**PUMP, "format": "wide_csv"})
    monkeypatch.setattr(try_mode, "MAX_RUN_DISK_BYTES", 640 << 20)
    monkeypatch.setattr(try_mode, "_RATE", {})
    run2 = _configured()
    assert try_mode._run_used(run2.id) == try_mode._du(run2.folder)
    assert any(p.name.startswith("_cache_") for p in run2.work_folder.iterdir())


def test_h3_put_is_streamed_capped_per_request_and_rate_limited(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "MAX_REQUEST_BYTES", 1000)
    monkeypatch.setattr(try_mode, "UPLOAD_REQUESTS_PER_ADDRESS_PER_HOUR", 2)
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]

        def chunks():                     # no Content-Length: the stream itself is cut off
            for _ in range(10):
                yield b"0" * 500
        assert c.put(f"/try/runs/{rid}/uploads/a.npy", content=chunks()).status_code == 413
        assert c.put(f"/try/runs/{rid}/uploads/a.npy", content=b"0" * 10).status_code == 200
        assert c.put(f"/try/runs/{rid}/uploads/b.npy", content=b"0" * 10).status_code == 429
    run = try_mode.get(rid)
    assert [p.stat().st_size for p in run.upload_folder.iterdir()] == [10]


def test_h3_kept_runs_expire_after_the_consent_maximum(try_dir):
    assert try_mode.CONSENT_RETENTION_DAYS == 90
    # Consent alone is not enough (HARM-S10-T05): a run is only ever moved to
    # kept/ once it has finished.
    old = try_mode.new_run("203.0.113.7")
    old.consent, old.status, old.created = True, "done", time.time() - 91 * 86400
    try_mode._save(old)
    young = try_mode.new_run("203.0.113.7")
    young.consent, young.status, young.created = True, "done", time.time() - 30 * 86400
    try_mode._save(young)
    assert try_mode.sweep() == 1
    assert try_mode.get(old.id) is None and try_mode.get(young.id) is not None
    assert try_mode.get(young.id).kept is True


# --- M1, M2: a deleted or timed-out run writes nothing more ----------------------------------------


def _fake_commission(steps: int, pause: float = 0.0, *, on_step=None, record=None):
    def fake(m, r, net, ch, out, tag, step):
        for i in range(steps):
            if on_step:
                on_step(i)
            m.folder.mkdir(parents=True, exist_ok=True)       # as the real one does
            (m.folder / f"draft-{i}.profile").write_text("x")
            step(i % len(machines_mod.STEPS))
            if record is not None:
                record.append(i)
            time.sleep(pause)
        raise AssertionError("the stand-in never finishes: the run must be stopped")
    return fake


def test_m1_delete_now_mid_run_leaves_no_files(try_dir, monkeypatch):
    run = _configured()
    run.baseline_h = (0.0, 30.0)
    monkeypatch.setattr(machines_mod, "_commission_point", _fake_commission(
        5, on_step=lambda i: i == 2 and try_mode.delete_run(run.id)))
    monkeypatch.setattr(try_mode.families, "path", lambda *a: try_dir)
    try_mode._do_run(run)
    assert not run.folder.exists()
    assert try_mode.get(run.id) is None


def test_m1_a_delete_is_finished_after_a_restart(try_dir, monkeypatch):
    run = try_mode.new_run("203.0.113.7")
    try_mode._delete(run.id)
    run.folder.mkdir(parents=True)                # a worker wrote after the delete ...
    (run.folder / "run.json").write_text(json.dumps(run.to_json()))
    monkeypatch.setattr(try_mode, "_CANCELLED", set())       # ... and the process restarted
    assert try_mode.get(run.id) is None
    assert not try_mode._save(run)
    try_mode.sweep()
    assert not run.folder.exists()


def test_m2_a_timed_out_run_stays_failed_and_stops(try_dir, monkeypatch):
    run = _configured()
    try_mode.start(run.id, {"baseline_h": [0.0, 30.0]})
    run = try_mode.get(run.id)
    done: list[int] = []
    monkeypatch.setattr(machines_mod, "_commission_point", _fake_commission(40, 0.05,
                                                                            record=done))
    monkeypatch.setattr(try_mode.families, "path", lambda *a: try_dir)
    monkeypatch.setattr(try_mode, "RUN_TIMEOUT_S", 0.3)
    try_mode._run_with_timeout(run)
    assert try_mode.get(run.id).status == "failed"
    for t in [t for t in threading.enumerate() if t.name == f"try-run-{run.id}"]:
        t.join(5)
    after = try_mode.get(run.id)
    assert after.status == "failed" and "longer than" in after.error
    assert len(done) < 15, "the worker kept going after the timeout"


# --- M3: never an embedding model in try mode --------------------------------------------------------


def test_m3_the_offline_pipeline_builds_no_embedder_with_keys_set(monkeypatch):
    import rag.index as index_mod
    from rag import pipeline as pipeline_mod

    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.setenv(k, "definitely-not-real")

    def boom(*a, **k):
        raise AssertionError("no embedder may be built in try mode")
    monkeypatch.setattr(index_mod, "get_embedder", boom)
    monkeypatch.setattr(pipeline_mod, "open_index", boom)
    monkeypatch.setattr(try_mode, "_pipeline", None)
    p = try_mode._offline_pipeline()
    assert isinstance(p.index, index_mod.BM25Index) and p.llm is None
    p.add_chunks(list(p.chunks.values())[:1])
    assert isinstance(p.index, index_mod.BM25Index)
    monkeypatch.setattr(try_mode, "_pipeline", None)


# --- M4: an unknown mode refuses to start -------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tyr", "Try-it", "", "consle"])
def test_m4_an_unknown_harmonic_mode_refuses_to_start(mode):
    env = {**os.environ, "HARMONIC_MODE": mode, "HARMONIC_AUTOSTART": "0"}
    r = subprocess.run([sys.executable, "-c", "import api.main"], env=env,
                       capture_output=True, text=True, timeout=120,
                       cwd=str(machines_mod.REPO))
    assert r.returncode != 0 and "Refusing to start" in r.stderr


# --- M5: no CORS in try mode ---------------------------------------------------------------------------


def test_m5_try_mode_answers_no_cross_origin_request(try_dir, monkeypatch):
    hdr = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    with _client(monkeypatch) as c:
        assert "access-control-allow-origin" not in c.options("/try/runs", headers=hdr).headers
        assert "access-control-allow-origin" not in c.get("/try/limits", headers=hdr).headers
    with _client(monkeypatch, mode="console") as c:          # the console is unchanged
        assert c.get("/try/limits", headers=hdr).headers["access-control-allow-origin"] == "*"


# --- L1, L2: clean 4xx refusals that leak nothing ---------------------------------------------------------


def test_l1_bad_inputs_get_a_4xx_never_a_500(try_dir, monkeypatch):
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        for i in range(200):             # the review's mismatched-snapshot CSVs
            body = "\n".join(["0.1"] * (4096 if i != 5 else 100))
            assert c.put(f"/try/runs/{rid}/uploads/s{i:04d}.csv", content=body).status_code == 200
        r = c.post(f"/try/runs/{rid}/configure", json=PUMP)
        assert r.status_code == 422 and "every snapshot needs" in r.json()["detail"]
        rid = c.post("/try/runs").json()["id"]
        c.put(f"/try/runs/{rid}/uploads/x.npy", content=b"\x93NUMPY garbage")
        assert c.post(f"/try/runs/{rid}/configure", json=PUMP).status_code == 422
        assert c.put(f"/try/runs/{rid}/uploads/y.npy?offset=-1", content=b"0").status_code == 422
        assert c.put(f"/try/runs/{rid}/uploads/y.npy?offset=5", content=b"0").status_code == 422
        assert c.post(f"/try/runs/{rid}/start", json={"baseline_h": [0, 1]}).status_code == 422
        assert c.get("/try/runs/" + "A" * 300).status_code == 404


def test_l2_errors_carry_no_path_or_run_id_and_the_log_no_cells(try_dir, caplog):
    caplog.set_level("WARNING", logger="harmonic.try.formats")
    run = try_mode.new_run("203.0.113.7")
    try_mode.save_file(run.id, "Line 4 (pump).wav", b"RIFF\x00\x00\x00\x00junk")
    with pytest.raises(try_mode.TryError) as e:
        try_mode.configure(run.id, {"family": "pump", "unit": "g", "format": "wav"})
    msg = str(e.value)
    assert "Line 4 (pump).wav" in msg                        # the visitor's own name
    assert run.id not in msg and str(try_dir) not in msg and "/" not in msg.split(":")[0]

    run = try_mode.new_run("203.0.113.8")
    try_mode.save_file(run.id, "day.csv", b"4711.25,secret,3\n1,2,3\n")
    with pytest.raises(try_mode.TryError, match="non-numeric"):
        try_mode.configure(run.id, {**PUMP, "format": "wide_csv"})
    assert "4711" not in caplog.text and "secret" not in caplog.text
    assert '"magic": "text"' in caplog.text


# --- L3: the console is unreachable from try mode ----------------------------------------------------------


def test_l3_no_websocket_and_no_dot_segments_in_try_mode(try_dir, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    from api import main as api_main

    for bad in ("/try/../index.html", "/try/./x", "/tryx", "/try\\..\\index.html",
                "/assets/../index.html", "/ws/stream"):
        assert not api_main._try_mode_path_ok(bad), bad
    assert api_main._try_mode_path_ok("/try/") and api_main._try_mode_path_ok("/try")
    with _client(monkeypatch) as c, pytest.raises(WebSocketDisconnect),             c.websocket_connect("/ws/stream") as ws:
        ws.receive_json()


# --- T10: the public benchmark page is served in try mode, the console stays 404 -------------------------


def test_t10_the_benchmark_page_is_served_and_console_routes_stay_404(try_dir, monkeypatch, tmp_path):
    from api import main as api_main

    (tmp_path / "try").mkdir()
    (tmp_path / "try" / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
    monkeypatch.setattr(api_main, "WEB_DIST", tmp_path)

    with _client(monkeypatch) as c:
        r = c.get("/try/benchmark")
        assert r.status_code == 200
        assert "root" in r.text
        # The console never leaks through the try-mode gate, benchmark page or not.
        assert c.get("/fleet").status_code == 404
        assert c.get("/machines").status_code == 404
    with _client(monkeypatch, mode="console") as c:          # unchanged for the console
        assert c.get("/try/benchmark").status_code == 200


# --- L4, G2: server-made storage names; real export names accepted -----------------------------------------


def test_l4_g2_real_export_names_are_stored_under_server_names(try_dir):
    run = try_mode.new_run("203.0.113.7")
    names = ["Pumpe Nord [2].npy", "vibración ch1.npy", "run (copy).npy"]
    for bad in ("../x.npy", "a/b.npy", "a\\b.npy", ".hidden.npy", "nul\x00.npy", "x" * 300 + ".npy"):
        with pytest.raises(try_mode.TryError):
            try_mode.save_file(run.id, bad, b"0")
    for n in names:
        try_mode.save_file(run.id, n, _npy(np.zeros(4096, np.float32)))
    stored = sorted(p.name for p in run.upload_folder.iterdir())
    assert len(stored) == 3 and not set(stored) & set(names)
    assert all(s.startswith("u") and s.endswith(".npy") for s in stored)


def test_l4_snapshot_times_still_come_from_the_original_names(try_dir):
    run = try_mode.new_run("203.0.113.7")
    snaps = _snapshots(200)
    for i in reversed(range(200)):       # uploaded out of order, times in the names
        t = time.gmtime(1_700_000_000 + 600 * i)
        try_mode.save_file(run.id, time.strftime("%Y-%m-%d %H-%M-%S", t) + ".npy",
                           _npy(snaps[i]))
    ov = try_mode.configure(run.id, PUMP)
    assert ov["at"][0] == time.strftime("%Y-%m-%d %H:%M", time.gmtime(1_700_000_000))
    assert ov["hours"][:3] == [0.0, 0.1667, 0.3333]
    r = try_mode._recording(try_mode.get(run.id))
    assert np.allclose(r.load(0, 4096)[7], snaps[7])


# --- L5: a running total, no stat of every file per PUT -----------------------------------------------------


def test_l5_puts_keep_a_running_total(try_dir, monkeypatch):
    run = try_mode.new_run("203.0.113.7")
    calls: list[str] = []
    real = try_mode._du
    monkeypatch.setattr(try_mode, "_du", lambda p: calls.append(str(p)) or real(p))
    for i in range(100):
        try_mode.save_file(run.id, f"s{i}.csv", b"0.1\n" * 10)
    assert calls == []                                       # never re-measured per PUT
    assert try_mode._run_used(run.id) == real(run.folder)    # and still exact


# --- G3: a channel choice for multi-channel files ----------------------------------------------------------


def test_g3_the_channel_is_chosen_and_stated_on_the_report(try_dir):
    run = try_mode.new_run("203.0.113.7")
    try_mode.save_file(run.id, "two.npy", _npy(_snapshots(200, channels=2)))
    ov = try_mode.configure(run.id, PUMP)
    assert ov["n_channels"] == 2 and ov["channel"] == 0            # channel 1 by default
    ov = try_mode.configure(run.id, {**PUMP, "channel": 1})
    run = try_mode.get(run.id)
    assert run.channel == 1 and ov["channel"] == 1
    assert any("channel 2 was analysed" in c for c in try_mode._caveats(run))
    with pytest.raises(try_mode.TryError, match="channel"):
        try_mode.configure(run.id, {**PUMP, "channel": 2})


# --- G4: files bigger than one request are sent in parts ----------------------------------------------------


def test_g4_a_file_is_sent_in_ordered_parts(try_dir, monkeypatch):
    data = _npy(_snapshots(200))
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        part = 1 << 20
        for off in range(0, len(data), part):
            r = c.put(f"/try/runs/{rid}/uploads/big.npy?offset={off}", content=data[off:off + part])
            assert r.status_code == 200, r.text
        r = c.put(f"/try/runs/{rid}/uploads/big.npy?offset=7", content=b"x")
        assert r.status_code == 422 and "send the file again" in r.json()["detail"]
        assert c.post(f"/try/runs/{rid}/configure", json=PUMP).json()["snapshots"] == 200
    run = try_mode.get(rid)
    assert try_mode._recording(run).snapshot_samples() == 4096


# --- G5: a full queue is "busy, try again", not a failed run ---------------------------------------------------


def test_g5_a_full_queue_says_busy_and_costs_nothing(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "_QUEUE", queue.Queue(maxsize=1))
    try_mode._QUEUE.put_nowait("someone-else")
    run = _configured()
    with pytest.raises(try_mode.Busy, match=r"try again in about \d+ minute"):
        try_mode.start(run.id, {"baseline_h": [0.0, 30.0]})
    assert try_mode.get(run.id).status == "configured"
    assert not any(k[0] == "start" for k in try_mode._RATE)
    with _client(monkeypatch) as c:
        r = c.post(f"/try/runs/{run.id}/start", json={"baseline_h": [0.0, 30.0]})
        assert r.status_code == 503 and int(r.headers["retry-after"]) >= 60
    try_mode._QUEUE.get_nowait()
    assert try_mode.start(run.id, {"baseline_h": [0.0, 30.0]})["status"] == "queued"


# --- re-check findings (N1-N4) ------------------------------------------------------------------------


def test_n1_a_replace_while_an_append_is_writing_is_refused(try_dir):
    run = try_mode.new_run("203.0.113.7")
    try_mode.save_file(run.id, "a.csv", b"0\n" * 1000)
    part = try_mode.begin_upload(run.id, "a.csv", 2000)          # a slow append in flight
    with pytest.raises(try_mode.Conflict):
        try_mode.save_file(run.id, "a.csv", b"")                 # replace: refused, 409
    with pytest.raises(try_mode.Conflict):
        try_mode.begin_upload(run.id, "a.csv", 2000)
    part.write(b"1\n")
    part.commit()
    real = sum(p.stat().st_size for p in run.upload_folder.iterdir())
    assert real == 2002 and try_mode._upload_bytes(run) == real
    try_mode.save_file(run.id, "a.csv", b"")                      # free again once it finished
    assert try_mode._upload_bytes(run) == 0


def test_n1_commit_charges_the_real_size_on_disk(try_dir):
    run = try_mode.new_run("203.0.113.7")
    up = try_mode.begin_upload(run.id, "a.npy", 0)
    up.write(b"0" * 100)
    up._f.write(b"1" * 50)               # bytes the running total never saw
    assert up.commit()["bytes"] == 150
    assert try_mode._upload_bytes(run) == 150
    assert try_mode._run_used(run.id) == try_mode._du(run.folder)


def test_n1_the_route_answers_409(try_dir, monkeypatch):
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        c.put(f"/try/runs/{rid}/uploads/a.npy", content=b"0" * 10)
        held = try_mode.begin_upload(rid, "a.npy", 10)
        assert c.put(f"/try/runs/{rid}/uploads/a.npy", content=b"").status_code == 409
        held.abort()
        assert c.put(f"/try/runs/{rid}/uploads/a.npy", content=b"").status_code == 200


def _csv(v: str, rows: int = 4096) -> bytes:
    return "\n".join([v] * rows).encode()


def test_n2_files_added_after_a_refused_configure_stay_intact(try_dir):
    run = try_mode.new_run("203.0.113.7")
    for i in range(1, 100):                                   # 99 snapshots: too short
        try_mode.save_file(run.id, f"s{i:03d}.csv", _csv(str(i)))
    with pytest.raises(try_mode.TryError, match="needs at least"):
        try_mode.configure(run.id, PUMP)
    for i in [*range(100, 200), 0]:                     # more, one sorting first
        try_mode.save_file(run.id, f"s{i:03d}.csv", _csv(str(i)))
    ov = try_mode.configure(run.id, PUMP)
    assert ov["snapshots"] == 200
    try_mode._UPLOADS.clear()                                 # as after a restart
    for name, e in try_mode._uploads(run).items():
        first = (run.upload_folder / e.stored).read_text().split("\n", 1)[0]
        assert first == str(int(name[1:4])), name
    data = try_mode._recording(try_mode.get(run.id)).load(0, 4096)
    assert [float(data[k, 0]) for k in (0, 1, 199)] == [0.0, 1.0, 199.0]


def test_la_a_same_size_replacement_rebuilds_the_parsed_cache(try_dir):
    # One bad cell deep in one snapshot fails configure after the cache is built.
    def body(v: str) -> bytes:
        return "\n".join(["0.5"] * 100 + [v] + ["0.5"] * 3995).encode()
    run = try_mode.new_run("203.0.113.7")
    for i in range(200):
        try_mode.save_file(run.id, f"s{i:03d}.csv", body("nan" if i == 5 else "0.5"))
    with pytest.raises(try_mode.TryError):
        try_mode.configure(run.id, PUMP)
    try_mode.save_file(run.id, "s005.csv", body("1.0"))          # fixed, same size
    assert try_mode.configure(run.id, PUMP)["snapshots"] == 200


def test_n2_a_wav_upload_survives_a_refused_configure(try_dir):
    from test_try_mode import _tone, _wav

    run = try_mode.new_run("203.0.113.7")
    try_mode.save_file(run.id, "part1.wav", _wav(_tone(16384 * 100)))
    with pytest.raises(try_mode.TryError, match="needs at least"):
        try_mode.configure(run.id, {**PUMP, "format": "wav"})
    try_mode.save_file(run.id, "part2.wav", _wav(_tone(16384 * 100)))
    with pytest.raises(try_mode.TryError, match="needs at least"):
        try_mode.configure(run.id, {**PUMP, "format": "wav"})   # 2 files = 2 snapshots
    with try_mode._ACCT_LOCK:
        stored = [run.upload_folder / e.stored for e in try_mode._uploads(run).values()]
    assert len(stored) == 2 and all(p.read_bytes()[:4] == b"RIFF" for p in stored)


def test_n3_one_ipv6_slash64_is_one_visitor(try_dir, monkeypatch):
    monkeypatch.setattr(try_mode, "RUNS_CREATED_PER_HOUR", 3)
    for i in range(3):
        try_mode.new_run(f"2001:db8:0:1::{i:x}")
    with pytest.raises(try_mode.RateLimited):
        try_mode.new_run("2001:db8:0:1:ffff:ffff:ffff:ffff")
    try_mode.new_run("2001:db8:0:2::1")                        # the next /64 is someone else
    assert try_mode._visitor("::ffff:192.0.2.1") == "192.0.2.1"
    assert try_mode._visitor("192.0.2.1") == "192.0.2.1"


def test_n4_text_snapshots_are_read_only_as_far_as_the_window(try_dir):
    run = try_mode.new_run("203.0.113.7")
    body = _csv("0.5", 262144)
    for i in range(200):
        try_mode.save_file(run.id, f"s{i:03d}.csv", body)
    t0 = time.process_time()
    try_mode.configure(run.id, PUMP)
    first = time.process_time() - t0
    r = try_mode._recording(try_mode.get(run.id))
    assert r.snapshot_samples() == recordings.MAX_WINDOW
    t0 = time.process_time()
    try_mode.configure(run.id, {**PUMP, "unit": "mm/s"})     # same files: cache kept
    again = time.process_time() - t0
    assert again < first / 3, (first, again)
    # A refused configure keeps the build too, when nothing changed.
    run2 = try_mode.new_run("203.0.113.8")
    for i in range(50):
        try_mode.save_file(run2.id, f"s{i:03d}.csv", _csv("1"))
    for _ in range(2):
        with pytest.raises(try_mode.TryError, match="needs at least"):
            try_mode.configure(run2.id, PUMP)
    sig = (run2.folder / "work.sig").stat().st_mtime_ns
    with pytest.raises(try_mode.TryError, match="needs at least"):
        try_mode.configure(run2.id, PUMP)
    assert (run2.folder / "work.sig").stat().st_mtime_ns == sig


def test_n4_the_console_still_reads_whole_text_snapshots(tmp_path):
    (tmp_path / "a.csv").write_bytes(_csv("0.25", 20000))
    (tmp_path / "recording.json").write_text(json.dumps({"sample_rate_hz": 25600}))
    assert recordings.FolderRecording(tmp_path).snapshot_samples() == 20000


def test_configure_over_http_keeps_the_interval_and_family_values(try_dir, monkeypatch):
    """The route's request model once dropped `interval_s` and `params`
    before try_mode saw them: the live page's "time between snapshots" and
    the family's optional values were ignored. Through HTTP, not in-process."""
    rng = np.random.default_rng(1)
    csv = "\n".join(",".join(f"{v:.3f}" for v in rng.normal(size=4096))
                    for _ in range(200)).encode()
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        assert c.put(f"/try/runs/{rid}/uploads/rows.csv?offset=0", content=csv).status_code == 200
        r = c.post(f"/try/runs/{rid}/configure",
                   json={"family": "gearbox", "sample_rate_hz": 25600, "unit": "g",
                         "format": "wide_csv", "interval_s": 60, "params": {"teeth": 23}})
        assert r.status_code == 200, r.text
        ov = r.json()
        assert ov["hours"][-1] == pytest.approx(199 / 60, abs=1e-3)   # 1 minute apart, not 10
        assert try_mode.get(rid).params.get("teeth") == 23


# --- S10-T06: restart, back, and "change settings and run again" (owner, 2026-09-29) -------


@pytest.mark.parametrize("finished_status", ["done", "failed"])
def test_s10t06_reconfiguring_a_finished_run_resets_it_and_clears_the_report(
        try_dir, monkeypatch, finished_status):
    """"Change settings and run again" on the result: a done or failed run
    may be configured again, through HTTP, with no second upload. It goes
    back to "configured" and its old report and chart are gone."""
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        assert c.put(f"/try/runs/{rid}/uploads/run.npy?offset=0",
                     content=_npy(_snapshots())).status_code == 200
        assert c.post(f"/try/runs/{rid}/configure", json=PUMP).status_code == 200

        run = try_mode.get(rid)
        run.status = finished_status
        run.report = {"headline": "an old report"}
        run.error = "boom" if finished_status == "failed" else None
        try_mode._save(run)

        r = c.post(f"/try/runs/{rid}/configure", json={**PUMP, "family": "gearbox"})
        assert r.status_code == 200, r.text

        reloaded = try_mode.get(rid)
        assert reloaded.status == "configured"
        assert reloaded.report == {}
        assert reloaded.error is None
        assert reloaded.family == "gearbox"
        assert reloaded.n_snapshots == 200         # the same upload, kept and re-read

        body = c.get(f"/try/runs/{rid}").json()
        assert body["status"] == "configured"
        assert "report" not in body                # public_json only returns it once "done"


def test_s10t06_re_runs_count_against_the_per_address_start_limit(try_dir, monkeypatch):
    """"Change settings and run again" starts the run over: it must still
    count against the per-address runs-per-hour limit, the same as a first
    run (owner, 2026-09-29)."""
    monkeypatch.setattr(try_mode, "RATE_LIMIT_PER_HOUR", 2)
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        assert c.put(f"/try/runs/{rid}/uploads/run.npy?offset=0",
                     content=_npy(_snapshots())).status_code == 200
        assert c.post(f"/try/runs/{rid}/configure", json=PUMP).status_code == 200
        baseline = {"baseline_h": [0.0, 30.0], "consent": False}

        for _ in range(2):
            r = c.post(f"/try/runs/{rid}/start", json=baseline)
            assert r.status_code == 200, r.text
            # Stand in for the run finishing, so it can be configured again
            # without the engine: the same "done" -> configure -> start cycle
            # "change settings and run again" drives.
            run = try_mode.get(rid)
            run.status = "done"
            run.report = {"headline": "x"}
            try_mode._save(run)
            assert c.post(f"/try/runs/{rid}/configure", json=PUMP).status_code == 200

        r = c.post(f"/try/runs/{rid}/start", json=baseline)
        assert r.status_code == 429, r.text
        assert "per hour" in r.json()["detail"]


def test_s10t06_start_over_deletes_a_run_that_is_still_queued(try_dir, monkeypatch):
    """"Start over" deletes the run server-side from any step, including
    while it is queued or running: the existing delete (HARM-S10-T03 M1)
    already cancels and removes it, so the page can reuse it as-is."""
    with _client(monkeypatch) as c:
        rid = c.post("/try/runs").json()["id"]
        assert c.put(f"/try/runs/{rid}/uploads/run.npy?offset=0",
                     content=_npy(_snapshots())).status_code == 200
        assert c.post(f"/try/runs/{rid}/configure", json=PUMP).status_code == 200
        r = c.post(f"/try/runs/{rid}/start",
                   json={"baseline_h": [0.0, 30.0], "consent": False})
        assert r.status_code == 200, r.text
        assert try_mode.get(rid).status == "queued"

        r = c.post(f"/try/runs/{rid}/delete")
        assert r.status_code == 200, r.text
        assert try_mode.get(rid) is None
        assert c.get(f"/try/runs/{rid}").status_code == 404
