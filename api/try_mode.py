"""Try Harmonic on your own data — public, anonymous, no sign-up (HARM-S8-try).

A skeptical maintenance engineer should be able to upload a recording from
their own machine — ideally one that already failed — and see, without a
sales call, when Harmonic would have alarmed and what it would have said.
That is the whole point: this is the product's sales funnel, so every design
choice here follows from "a stranger's untrusted file, alone, unattended."

Three steps, each a request this module answers:

    1. upload            POST /try/runs, PUT .../uploads/{filename}
    2. configure+baseline POST .../configure  (family, rate, speed, format)
    3. run+report         POST .../start, GET /try/runs/{id}

Reused rather than reimplemented, per HARM-S8's own rule: the recording
format (`sim/datasets/recordings.py`), the commissioning pipeline
(`api/machines.py::_commission_point`, `replay`, `speeds_for`), and the
offline diagnosis composer (`rag/pipeline.py`, forced `offline=True` — see
its docstring for why that is a stronger guarantee than an environment
variable).

What makes this different from the console's own commissioning wizard:

  * **No registry.** A run's `Machine` carries `base_dir=TRY_DIR`, so it is
    never in `MACHINES_DIR`, never in `GET /machines`, and never joined to
    the shared knowledge graph — `api/machines.py`'s `base_dir` field exists
    for exactly this. Diagnosis instead runs against a private, in-memory,
    non-persisting `NetworkXStore` (`persist=False`), seeded from the same
    catalogue but never written back to — a run's equipment node is never
    saved anywhere another request could see it.
  * **One run at a time, queued, timed out.** A single background worker
    processes `_QUEUE`; a run that will not finish is abandoned, not left to
    wedge the service for everyone behind it.
  * **Untrusted by default.** Every cap in this file (bytes, files,
    snapshots, samples, sample rate) is checked before the bytes are read
    for meaning, not after; filenames are never used as paths; a run's
    folder is a UUID-like id nothing else can guess or enumerate.
  * **Deleted.** A sweeper removes runs older than `RETENTION_DAYS` that did
    not opt in; a run that did, and finished, is moved to its own `kept/`
    prefix (`KEPT_DIR`) and removed after `CONSENT_RETENTION_DAYS` instead
    (HARM-S10-T05) — the bucket's own lifecycle rule is a backstop at each
    prefix's age plus a day, see `deploy/gcp/deploy_try.sh`. `delete_run`
    removes a run immediately from wherever it is, technician's own button,
    even mid-run.
  * **Bounded.** Rate limits per address and per instance, a disk quota, a
    per-run disk cap that counts derived files, streamed parsing and
    streamed uploads (HARM-S10-T03; docs/TRY_IT_SPEC.md "Limits").
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import ipaddress
import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np

from api import machines as machines_mod
from api.adapters import formats as format_adapters
from kg import families
from sim.datasets import recordings as rec_mod

log = logging.getLogger("harmonic.try")

REPO = machines_mod.REPO
DATA_DIR = machines_mod.DATA_DIR
TRY_DIR = DATA_DIR / "try"
# A finished, consented run moves here (HARM-S10-T05): a prefix of its own,
# not nested under `try/`, because a GCS lifecycle condition can only
# *include* a prefix, never exclude one — if kept runs lived under
# `try/kept/`, the bucket's 8-day rule on `try/` would also catch them (see
# deploy/gcp/deploy_try.sh). Same reasoning on local disk: a plain `find
# var/try -mtime +7` style backstop would otherwise need to know about
# `kept/` too.
KEPT_DIR = DATA_DIR / "kept"

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
# An upload's own name is metadata only (HARM-S10-T03 L4/G2): it is kept to
# sort the snapshots and read their times, and never used as a path. So real
# exports with spaces, brackets or non-ASCII letters are accepted; only path
# separators, control characters and hidden-file names are refused.
_MAX_NAME_CHARS = 200
_BAD_NAME_CHARS = re.compile(r"[/\\\x00-\x1f\x7f]")

RETENTION_DAYS = 7
# A run whose visitor ticked "let Harmonic keep this recording" is still not
# kept for ever: it is deleted after this many days (HARM-S10-T03 H3).
CONSENT_RETENTION_DAYS = 90
SWEEP_PERIOD_S = 3600.0

RATE_LIMIT_PER_HOUR = 5                     # runs started, per address
RUNS_CREATED_PER_HOUR = 20                  # runs created, per address
# Caps for the whole instance, whatever the addresses (HARM-S10-T03 H2): an
# attacker with many addresses still cannot fill the queue or the disk.
GLOBAL_RUNS_CREATED_PER_HOUR = 200
GLOBAL_RUNS_STARTED_PER_HOUR = 30
CONFIGURES_PER_RUN_PER_HOUR = 10
CONFIGURES_PER_ADDRESS_PER_HOUR = 30
UPLOAD_REQUESTS_PER_ADDRESS_PER_HOUR = 10_000
QUEUE_DEPTH = 5
RUN_TIMEOUT_S = float(os.environ.get("HARMONIC_TRY_TIMEOUT_S", "900"))

MAX_TOTAL_BYTES = 200 * 1024 * 1024         # uploaded bytes per run
# One request's body: under Cloud Run's 32 MiB HTTP/1 limit. A bigger file is
# sent in parts (`?offset=`), each appended in order (HARM-S10-T03 G4).
MAX_REQUEST_BYTES = 30 * 1024 * 1024
UPLOAD_PART_BYTES = 16 * 1024 * 1024        # what the page sends per part
MAX_FILES = 5000
MAX_SNAPSHOTS = 5000
MAX_SAMPLES_PER_SNAPSHOT = 262_144
# Samples in the whole recording, as float32 (128 MiB): what the converted
# data and the per-channel cache may each hold (HARM-S10-T03 H1/H3).
MAX_TOTAL_SAMPLES = 1 << 25
MAX_TEXT_COLUMNS = 16
MIN_SAMPLE_RATE_HZ = 100
MAX_SAMPLE_RATE_HZ = 500_000
# Everything one run may hold on disk: its upload plus every file the service
# derives from it (data.npy, _cache_*, brains, frames).
MAX_RUN_DISK_BYTES = 768 * 1024 * 1024
# Everything every run may hold on disk, together (HARM-S10-T03 H3).
DISK_QUOTA_BYTES = int(float(os.environ.get("HARMONIC_TRY_DISK_QUOTA_MB", "8192"))
                       * 1024 * 1024)
_RUN_OVERHEAD_BYTES = 16 * 1024 * 1024      # brains, profiles, run.json

# Every family in profiles/families.json, read from the registry (kg.families)
# so a machine class added there appears here with no code change. The page
# shows a family's `public_label` where it has one (the generic family's is
# "Other or unsure — watch for any change only"), else its `label`.
_REGISTRY = families.families()
FAMILIES = tuple(_REGISTRY)
FAMILY_LABELS = {k: str(v.get("public_label") or v.get("label", k.title()))
                 for k, v in _REGISTRY.items()}
# Names an older page or run.json may still send. "other" was a choice that
# borrowed the default family's network; the generic family (a healthy-only
# network of its own) replaced it (HARM-S10-T02).
FAMILY_ALIASES = {"other": "generic"}
# Typical nameplate speed, used only when the visitor does not give one; each
# is a family-wide approximation, not this machine's own speed, and the
# report says so (`_caveats`). A family not listed gets the generic 1800 rpm.
_DEFAULT_RPM = {"pump": 2950.0, "gearbox": 1800.0, "motor": 1500.0, "fan": 1450.0}
FAMILY_DEFAULT_RPM = {k: _DEFAULT_RPM.get(k, 1800.0) for k in FAMILIES}
MIN_WATCHED_SNAPSHOTS = 20
# Time between snapshots when neither the visitor nor the recording says
# (an old page that did not send one); the page's own default is 10 minutes.
DEFAULT_INTERVAL_S = 600.0
PROBE_SAMPLE_RATE_HZ = 25600            # stands in for the rate while a probe reads a file
MIN_INTERVAL_S = 1.0
MAX_INTERVAL_S = 30 * 86400.0
# A generic deep-groove ball bearing, for when the visitor's own geometry is
# not asked for in a 3-step public wizard. profiles/bearings.json carries its
# source.
GENERIC_BEARING_KEY = "6205"

STATES = ("collecting", "configured", "queued", "running", "done", "failed")


class TryError(Exception):
    """A reason a try-mode request is refused, in words for the visitor."""


class RunNotFound(TryError):
    """No such run — deleted, expired, or never existed."""


class RateLimited(TryError):
    """Too many requests from one address, for one run, or for the service."""


class TooLarge(TryError):
    """A body, an upload or a run over its size cap."""


class Busy(TryError):
    """The service cannot take this now (queue full, disk full); try later."""

    def __init__(self, msg: str, retry_after_s: int = 300):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class Conflict(TryError):
    """The same file is already being written by another request."""


class RefusedExtension(TryError):
    """An extension no adapter takes; the route logs the file's magic bytes."""

    def __init__(self, ext: str):
        super().__init__("files must be .csv, .txt, .npy or .wav — tell us what your "
                         "logger writes and we will look at adding it")
        self.ext = ext


class _Cancelled(Exception):
    """Raised inside a run's worker once it is deleted or timed out."""


# --- the run record -------------------------------------------------------------------


@dataclass
class Run:
    id: str
    created: float = field(default_factory=time.time)
    status: str = "collecting"
    consent: bool = False
    # Whether this run's files live under KEPT_DIR right now (HARM-S10-T05).
    # Set only by `_move_to_kept`, once its copy is committed — never by hand,
    # and never turned back off (a kept run is deleted, not un-kept).
    kept: bool = False
    family: str = "pump"
    format: str = "auto"
    unit: str = "g"
    sample_rate_hz: int = 0
    rpm: float | None = None            # None: the family default stood in
    rpm_limited: bool = True
    # The family parameters the visitor gave (a gearbox's teeth ...); every
    # one not given takes the family's wizard default (`_family_params`).
    params: dict[str, float] = field(default_factory=dict)
    # Seconds between snapshots when the recording carries no time stamps.
    interval_s: float = DEFAULT_INTERVAL_S
    channel: int = 0
    baseline_h: tuple[float, float] | None = None
    n_snapshots: int = 0
    n_channels: int = 1
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    # A salted hash, never the address itself, kept only for the per-IP rate
    # limit and never returned to any client.
    client_ip_hash: str = ""
    # The run holds the bundled sample recording (`add_sample`), not an upload.
    sample: bool = False

    @property
    def folder(self) -> Path:
        return (KEPT_DIR if self.kept else TRY_DIR) / self.id

    @property
    def upload_folder(self) -> Path:
        """The visitor's files, exactly as uploaded, under the server's names;
        never renamed or converted in place."""
        return self.folder / "upload"

    @property
    def work_folder(self) -> Path:
        """What the recording is read from: the upload linked in time order,
        converted by its format adapter, plus the parsed cache. Rebuilt from
        `upload/` whenever the upload changes (HARM-S10-T03 N2)."""
        return self.folder / "work"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["baseline_h"] = list(self.baseline_h) if self.baseline_h else None
        return d

    def public_json(self) -> dict[str, Any]:
        """What a visitor holding this run's own (unguessable) id may see —
        never the IP hash, and the report only once it exists."""
        d = {
            "id": self.id, "status": self.status, "created": self.created,
            "consent": self.consent, "family": self.family, "unit": self.unit,
            "sample_rate_hz": self.sample_rate_hz, "rpm": self.rpm,
            "rpm_limited": self.rpm_limited, "channel": self.channel,
            "params": self.params, "interval_s": self.interval_s,
            "baseline_h": list(self.baseline_h) if self.baseline_h else None,
            "n_snapshots": self.n_snapshots, "n_channels": self.n_channels,
            "progress": self.progress, "error": self.error, "sample": self.sample,
        }
        if self.status == "done":
            d["report"] = self.report
        return d


# --- deleted and cancelled runs (HARM-S10-T03 M1, M2) ------------------------------------
#
# A run is cancelled when its visitor deletes it or when it times out. From
# then on nothing about it is written again: `_save` skips it, a progress step
# raises `_Cancelled` in the worker, and a deleted run's folder is removed
# again once the worker stops. Deletion also leaves a marker outside the run's
# folder, so a restart finishes a delete the process did not.

_CANCELLED: set[str] = set()
_CANCELLED_LOCK = threading.RLock()


def _marker(run_id: str) -> Path:
    return TRY_DIR / ".deleted" / run_id


def _is_deleted(run_id: str) -> bool:
    return _marker(run_id).exists()


def _dead(run_id: str) -> bool:
    with _CANCELLED_LOCK:
        return run_id in _CANCELLED or _is_deleted(run_id)


def _save(run: Run, *, force: bool = False) -> bool:
    """Write run.json — never for a deleted run, never for a cancelled one
    unless `force` (the timeout's own "failed"), and never creating the run's
    folder: only `new_run` does that. Returns whether it wrote."""
    with _CANCELLED_LOCK:
        if _is_deleted(run.id) or (run.id in _CANCELLED and not force):
            return False
        if not run.folder.is_dir():
            return False
        tmp = run.folder / "run.json.tmp"
        tmp.write_text(json.dumps(run.to_json(), indent=1))
        tmp.replace(run.folder / "run.json")
        return True


def _load(path: Path) -> Run:
    d = json.loads(path.read_text())
    if d.get("baseline_h") is not None:
        d["baseline_h"] = tuple(d["baseline_h"])
    # A run stored before HARM-S10-T02 may name "other"; it runs as generic.
    d["family"] = FAMILY_ALIASES.get(d.get("family", ""), d.get("family", "pump"))
    return Run(**d)


def _locate(run_id: str) -> Path:
    """Where run_id's folder actually is: `KEPT_DIR` once its move has
    committed (its run.json written there — HARM-S10-T05's "write, then
    remove"), else the normal prefix. A caller already holding a `Run` should
    use `run.folder` instead; this is for the handful of places (`get`,
    `_delete`, disk accounting, the sweeper) that only have an id."""
    kept = KEPT_DIR / run_id
    return kept if (kept / "run.json").exists() else TRY_DIR / run_id


def get(run_id: str) -> Run | None:
    if not RUN_ID_RE.match(run_id) or _is_deleted(run_id):
        return None
    p = _locate(run_id) / "run.json"
    try:
        return _load(p) if p.exists() else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _get_or_404(run_id: str) -> Run:
    run = get(run_id)
    if run is None:
        raise RunNotFound("no such run, or it has already been deleted")
    return run


def _ensure_status(run: Run, allowed: set[str]) -> None:
    if run.status not in allowed:
        raise TryError(f"this run is {run.status}; expected one of {sorted(allowed)}")


def _set_status(run: Run, status: str, *, progress: dict[str, Any] | None = None,
                error: str | None = None, force: bool = False) -> None:
    run.status = status
    if progress is not None:
        run.progress = progress
    if error is not None:
        run.error = error
    _save(run, force=force)


def _delete(run_id: str) -> None:
    """Delete a run now: marker first (so nothing writes to it again, even
    after a restart), then the folder — both prefixes, since a kept run
    lives under `KEPT_DIR` and a crash mid-move (HARM-S10-T05) can briefly
    leave a stray copy under the other one — then its share of the disk."""
    with _CANCELLED_LOCK:
        _CANCELLED.add(run_id)
        with contextlib.suppress(OSError):
            _marker(run_id).parent.mkdir(parents=True, exist_ok=True)
            _marker(run_id).touch()
    shutil.rmtree(TRY_DIR / run_id, ignore_errors=True)
    shutil.rmtree(KEPT_DIR / run_id, ignore_errors=True)
    _forget(run_id)


# --- disk accounting (HARM-S10-T03 H3, L5) ------------------------------------------------
#
# A running total per run and for the whole service, measured from disk once
# (at start-up, or the first time a run is touched after it) and then kept by
# adding what each write adds. Uploads are charged before they are written, so
# parallel requests cannot all pass one check; derived files are checked for
# room before they are written and re-measured after.

_ACCT_LOCK = threading.RLock()
_RUN_BYTES: dict[str, int] = {}
_DISK: dict[str, int] = {}                   # {"used": bytes}, measured lazily


def _du(path: Path) -> int:
    """Bytes under `path`, a hard-linked file counted once."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                st = os.lstat(os.path.join(root, name))
                if st.st_nlink > 1:
                    if (st.st_dev, st.st_ino) in seen:
                        continue
                    seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    return total


def _disk_used() -> int:
    """The whole service's quota is one number across both prefixes
    (HARM-S10-T05): a kept run still counts against it, it is just deleted
    later."""
    with _ACCT_LOCK:
        if "used" not in _DISK:
            _DISK["used"] = ((_du(TRY_DIR) if TRY_DIR.is_dir() else 0)
                             + (_du(KEPT_DIR) if KEPT_DIR.is_dir() else 0))
        return _DISK["used"]


def _run_used(run_id: str) -> int:
    with _ACCT_LOCK:
        if run_id not in _RUN_BYTES:
            _disk_used()
            _RUN_BYTES[run_id] = _du(_locate(run_id))
        return _RUN_BYTES[run_id]


def _room(run_id: str, extra: int) -> None:
    """Refuse `extra` more bytes for this run if it would pass the run's own
    cap or the service's quota."""
    with _ACCT_LOCK:
        if _run_used(run_id) + extra > MAX_RUN_DISK_BYTES:
            raise TooLarge(f"this run would pass its {MAX_RUN_DISK_BYTES // (1 << 20)} MB "
                           "limit on disk, counting the files Harmonic derives from the "
                           "upload; send a shorter recording")
        if _disk_used() + extra > DISK_QUOTA_BYTES:
            raise Busy("the try service is full right now; please try again later",
                       retry_after_s=3600)


def _charge(run_id: str, n: int, *, check: bool = True) -> None:
    with _ACCT_LOCK:
        if n > 0 and check:
            _room(run_id, n)
        _RUN_BYTES[run_id] = max(_run_used(run_id) + n, 0)
        _DISK["used"] = max(_disk_used() + n, 0)


def _remeasure(run_id: str) -> None:
    with _ACCT_LOCK:
        before = _run_used(run_id)
        after = 0 if _is_deleted(run_id) else _du(_locate(run_id))
        _RUN_BYTES[run_id] = after
        _DISK["used"] = max(_disk_used() + after - before, 0)


def _forget(run_id: str) -> None:
    with _ACCT_LOCK:
        n = _RUN_BYTES.pop(run_id, None)
        _UPLOADS.pop(run_id, None)
        _ACTIVE.pop(run_id, None)
        if "used" in _DISK:
            if n is None:
                _DISK.pop("used")            # not tracked: measure again next time
            else:
                _DISK["used"] = max(_DISK["used"] - n, 0)


# --- rate limiting ----------------------------------------------------------------------


_RATE_LOCK = threading.Lock()
_RATE: dict[tuple[str, str], list[float]] = {}


def _recent(key: tuple[str, str], now: float) -> list[float]:
    hist = [t for t in _RATE.get(key, ()) if now - t < 3600.0]
    if hist:
        _RATE[key] = hist
    else:
        _RATE.pop(key, None)
    return hist


def _take_rates(checks: list[tuple[str, str, int, str]]) -> None:
    """Each check is (bucket, key, per_hour, refusal). All are checked before
    any is counted, so a refusal by one does not use up another."""
    now = time.time()
    with _RATE_LOCK:
        for bucket, key, cap, msg in checks:
            if len(_recent((bucket, key), now)) >= cap:
                raise RateLimited(msg)
        for bucket, key, _cap, _msg in checks:
            _RATE.setdefault((bucket, key), []).append(now)


def _check_rate(ip_hash: str, *, bucket: str = "start",
                per_hour: int | None = None) -> None:
    cap = RATE_LIMIT_PER_HOUR if per_hour is None else per_hour
    _take_rates([(bucket, ip_hash, cap,
                  f"at most {cap} {'runs' if bucket == 'start' else 'uploads'} per hour "
                  "from one address; please try again later")])


# --- creating a run and taking the upload ----------------------------------------------


def _visitor(ip: str) -> str:
    """Who counts as one visitor for the rate limits: an IPv4 address, or an
    IPv6 /64 — one household or one machine usually holds a whole /64, so
    rotating addresses inside it must not buy fresh limits (HARM-S10-T03 N3)."""
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]").split("%", 1)[0])
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def _ip_hash(ip: str) -> str:
    return hashlib.sha256(f"harmonic-try:{_visitor(ip)}".encode()).hexdigest()[:24]


def new_run(client_ip: str) -> Run:
    ip_hash = _ip_hash(client_ip)
    # Starting a run is capped too, not only replaying one: each run may hold
    # up to MAX_TOTAL_BYTES on disk until it is swept.
    if _disk_used() + MAX_TOTAL_BYTES > DISK_QUOTA_BYTES:
        raise Busy("the try service is full right now; please try again later",
                   retry_after_s=3600)
    _take_rates([("create", ip_hash, RUNS_CREATED_PER_HOUR,
                  f"at most {RUNS_CREATED_PER_HOUR} uploads per hour from one address; "
                  "please try again later"),
                 ("create", "*", GLOBAL_RUNS_CREATED_PER_HOUR,
                  "the try service has taken as many uploads as it can this hour; "
                  "please try again later")])
    run = Run(id=secrets.token_urlsafe(18), client_ip_hash=ip_hash)
    # The one place a run's folder is created (HARM-S10-T03 M1).
    run.upload_folder.mkdir(parents=True)
    _save(run)
    with _ACCT_LOCK:
        _disk_used()
        _RUN_BYTES[run.id] = _du(run.folder)
        _DISK["used"] += _RUN_BYTES[run.id]
    return run


# Uploads, by the visitor's own name: each stored under a name the server
# makes up, and listed in `names.jsonl` beside the upload folder — appended
# once per request, so a run of 5000 files costs 5000 lines, not 5000
# rewrites (HARM-S10-T03 L4, L5).


@dataclass
class _Entry:
    stored: str
    size: int


_UPLOADS: dict[str, dict[str, _Entry]] = {}
# (run id, visitor's file name) of every PUT still writing: a second PUT on
# the same name waits its turn (409), so a replace can never free bytes an
# append is still writing (HARM-S10-T03 N1).
_INFLIGHT: set[tuple[str, str]] = set()
# Per run: how many uploads are being written (>0), or -1 while `configure`
# reads the upload; the two never overlap.
_ACTIVE: dict[str, int] = {}


def _names_file(run: Run) -> Path:
    return run.folder / "names.jsonl"


def _uploads(run: Run) -> dict[str, _Entry]:
    """This run's uploads, original name -> stored file (call under _ACCT_LOCK)."""
    ups = _UPLOADS.get(run.id)
    if ups is None:
        ups = {}
        with contextlib.suppress(OSError):
            for line in _names_file(run).read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError, KeyError, TypeError):
                    d = json.loads(line)
                    ups[str(d["n"])] = _Entry(str(d["s"]), int(d["b"]))
        for name, e in list(ups.items()):           # the disk, not the list, is the truth
            p = run.upload_folder / e.stored
            if p.is_file():
                e.size = p.stat().st_size
            else:
                del ups[name]
        _UPLOADS[run.id] = ups
    return ups


def _upload_bytes(run: Run) -> int:
    return sum(e.size for e in _uploads(run).values())


def _check_name(filename: str) -> str:
    """The visitor's file name, checked as metadata; returns its extension."""
    if (not filename or len(filename) > _MAX_NAME_CHARS or _BAD_NAME_CHARS.search(filename)
            or filename.startswith(".") or filename.strip() != filename):
        raise TryError("refused file name: use the name your logger gave the file")
    return Path(filename).suffix.lower()


class Upload:
    """One PUT: `write` each piece as it arrives, then `commit`, or `abort`.
    Bytes are charged against the run's and the service's caps before they
    are written, so parallel requests cannot overshoot them together."""

    def __init__(self, run: Run, name: str, entry: _Entry, offset: int):
        self.run, self.name, self.entry, self.offset = run, name, entry, offset
        self.written = 0
        self._f: Any = None
        path = run.upload_folder / entry.stored
        try:
            # Held open across the request's `write` calls, closed in commit/abort.
            self._f = open(path, "r+b" if offset else "wb")  # noqa: SIM115
            self._f.seek(offset)
            self._f.truncate()
        except OSError:
            self._release()
            raise RunNotFound("no such run, or it has already been deleted") from None

    def write(self, data: bytes) -> None:
        if not data:
            return
        n = len(data)
        with _ACCT_LOCK:
            if _dead(self.run.id):
                raise RunNotFound("no such run, or it has already been deleted")
            if _upload_bytes(self.run) + n > MAX_TOTAL_BYTES:
                raise TooLarge(f"this upload would pass the "
                               f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB limit for one "
                               "try-mode run")
            _charge(self.run.id, n)
            self.entry.size += n
            self.written += n
        self._f.write(data)

    def commit(self) -> dict[str, Any]:
        self._f.flush()
        real = os.fstat(self._f.fileno()).st_size
        self._f.close()
        with _ACCT_LOCK:
            if not _dead(self.run.id):
                # What the file really holds is what is counted, whatever
                # the running total believed (HARM-S10-T03 N1).
                if real != self.entry.size:
                    _charge(self.run.id, real - self.entry.size, check=False)
                    self.entry.size = real
                line = (json.dumps({"n": self.name, "s": self.entry.stored,
                                    "b": self.entry.size}) + "\n").encode()
                with contextlib.suppress(OSError), open(_names_file(self.run), "ab") as f:
                    f.write(line)
                    _charge(self.run.id, len(line), check=False)
        self._release()
        return {"saved": self.name, "bytes": self.entry.size}

    def abort(self) -> None:
        """Undo this request's bytes: the file goes back to where it started."""
        with contextlib.suppress(OSError, ValueError):
            self._f.truncate(self.offset)
        with contextlib.suppress(OSError):
            self._f.close()
        with _ACCT_LOCK:
            if not _dead(self.run.id):
                _charge(self.run.id, -self.written)
                self.entry.size -= self.written
                if self.entry.size == 0 and self.offset == 0:
                    _uploads(self.run).pop(self.name, None)
                    with contextlib.suppress(OSError):
                        (self.run.upload_folder / self.entry.stored).unlink()
        self.written = 0
        self._release()

    def _release(self) -> None:
        with _ACCT_LOCK:
            if (self.run.id, self.name) in _INFLIGHT:
                _INFLIGHT.discard((self.run.id, self.name))
                if _ACTIVE.get(self.run.id, 0) > 0:
                    _ACTIVE[self.run.id] -= 1


def begin_upload(run_id: str, filename: str, offset: int = 0,
                 client_ip: str | None = None) -> Upload:
    """Start one PUT of `filename` (the visitor's own name) at `offset`: 0
    starts or replaces the file, anything else appends a part and must equal
    what is already stored."""
    run = _get_or_404(run_id)
    _ensure_status(run, {"collecting"})
    ext = _check_name(filename)
    if ext not in format_adapters.ACCEPTED_EXTENSIONS:
        raise RefusedExtension(ext)
    if offset < 0:
        raise TryError("the part offset must not be negative")
    if client_ip is not None:
        _take_rates([("put", _ip_hash(client_ip), UPLOAD_REQUESTS_PER_ADDRESS_PER_HOUR,
                      "too many uploads from one address this hour; please try again later")])
    with _ACCT_LOCK:
        if _dead(run.id):
            raise RunNotFound("no such run, or it has already been deleted")
        if _ACTIVE.get(run.id, 0) < 0:
            raise TryError("this run's upload is being read; it takes no more files")
        if (run.id, filename) in _INFLIGHT:
            raise Conflict("this file is already being uploaded; wait for that request "
                           "to finish, then send it again")
        ups = _uploads(run)
        entry = ups.get(filename)
        if offset:
            have = entry.size if entry else 0
            if entry is None or offset != have:
                raise TryError(f"this part starts at byte {offset}, but {have} bytes of the "
                               "file are stored; send the file again from there")
        elif entry is None:
            if len(ups) >= MAX_FILES:
                raise TryError(f"at most {MAX_FILES} files per upload")
            entry = _Entry(f"u{secrets.token_hex(8)}{ext}", 0)
            ups[filename] = entry
        else:
            _charge(run.id, -entry.size)             # replaced: its old bytes are freed
            entry.size = 0
        _ACTIVE[run.id] = _ACTIVE.get(run.id, 0) + 1
        _INFLIGHT.add((run.id, filename))
    return Upload(run, filename, entry, offset)


def save_file(run_id: str, filename: str, body: bytes, offset: int = 0) -> dict[str, Any]:
    """One whole file (or part) in one call: `begin_upload`, `write`, `commit`."""
    try:
        up = begin_upload(run_id, filename, offset)
    except RefusedExtension as exc:
        format_adapters.log_rejected("upload", ext=exc.ext, head=body[:8], size=len(body),
                                     reason="extension not accepted")
        raise
    try:
        up.write(body)
    except BaseException:
        up.abort()
        raise
    return up.commit()


# --- the sample recording ---------------------------------------------------------------

# NASA IMS bearing test 2, bearing 1 (channel 0): the run ended with an outer-race
# failure. A U.S. Government Work (NASA Prognostics Data Repository; University of
# Cincinnati Center for Intelligent Maintenance Systems), so it may be shipped and
# shown. Built by scripts/make_try_sample.py: every second snapshot (20 minutes
# apart), stored as float16 and turned back into float32 here. It goes through the
# same upload, configure and run as a visitor's own file: a real run, not a canned
# result. XJTU-SY and CWRU may never be used for this (docs/BENCHMARK_DATASETS.md).
SAMPLE_PATH = Path(__file__).resolve().parent / "try_samples" / "ims_test2_bearing1.npy"
SAMPLE_FILE_NAME = "nasa_ims_test2_bearing1.npy"
SAMPLE = {
    "file_name": SAMPLE_FILE_NAME,
    "title": "NASA IMS bearing test 2, bearing 1",
    "story": ("A test rig ran four bearings at 2000 rpm, day and night, until one "
              "failed: bearing 1 ended with an outer-race failure after about a week."),
    "credit": ("Data: NSF I/UCR Center for Intelligent Maintenance Systems, University "
               "of Cincinnati, via the NASA Prognostics Data Repository (a U.S. "
               "Government work)."),
    "family": "generic", "sample_rate_hz": 20480, "rpm": 2000.0, "interval_s": 1200.0,
    "unit": "g",
}


def add_sample(run_id: str) -> dict[str, Any]:
    """Put the sample recording into a new run's upload, through the same upload
    path (caps, accounting) as a visitor's own file, and say how to configure it."""
    run = _get_or_404(run_id)
    _ensure_status(run, {"collecting"})
    with _ACCT_LOCK:
        if _uploads(run):
            raise TryError("this run already holds files; start a new one for the sample")
    if not SAMPLE_PATH.is_file():
        raise TryError("the sample recording is not available on this server")
    buf = io.BytesIO()
    np.save(buf, np.load(SAMPLE_PATH, allow_pickle=False).astype(np.float32))
    save_file(run.id, SAMPLE_FILE_NAME, buf.getvalue())
    run = _get_or_404(run.id)
    run.sample = True
    if not _save(run):
        raise RunNotFound("no such run, or it has already been deleted")
    return dict(SAMPLE)


# --- step 2: configure + baseline -------------------------------------------------------


def _generic_bearing() -> dict[str, float]:
    b = machines_mod.BEARINGS[GENERIC_BEARING_KEY]
    return {k: float(b[k]) for k in ("pitch_mm", "ball_mm", "n_balls", "contact_angle_deg")}


def _family_params(family: str, given: dict[str, Any] | None
                   ) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """The family's parameters for a run — the visitor's where given, the
    family wizard's defaults elsewhere — and the wizard entries that were
    assumed. Refusals are the visitor's error (a TryError)."""
    try:
        return machines_mod.family_params(family, given)
    except machines_mod.CommissionError as exc:
        raise TryError(str(exc)) from None


def _run_params(run: Run, rpm: float
                ) -> tuple[dict[str, float], list[dict[str, Any]], str | None]:
    """The run's family parameters: the visitor's, the wizard's defaults for
    the rest, and an assumed one moved where a default cannot fit the speed
    (`machines.fit_assumed`). Also returns what was assumed and what moved."""
    params, assumed = _family_params(run.family, run.params)
    params, moved = machines_mod.fit_assumed(run.family, params,
                                             {p["key"] for p in assumed}, rpm,
                                             _generic_bearing())
    return params, assumed, moved


def _names_faults(family: str) -> bool:
    """Whether the family's network names any fault (a class besides healthy).
    A family that names none only watches for change."""
    return len(families.class_names(family)) > 1


def _recording(run: Run) -> rec_mod.FolderRecording:
    """The run's upload, read with try mode's own caps on text files."""
    return rec_mod.FolderRecording(run.work_folder, key=f"try:{run.id}",
                                   max_text_rows=MAX_SAMPLES_PER_SNAPSHOT,
                                   max_text_cols=MAX_TEXT_COLUMNS,
                                   # No more of a text snapshot is read than the
                                   # engine's window can use (HARM-S10-T03 N4).
                                   text_rows_used=rec_mod.MAX_WINDOW)


_STORED_NAME = re.compile(r"\b[us][0-9a-f]{5,16}\.(?:csv|txt|npy|wav)\b", re.IGNORECASE)


def _visitor_message(run: Run, msg: str) -> str:
    """An error for the visitor: their own file names back in place of the
    stored ones, and no server path or run id (HARM-S10-T03 L2)."""
    with _ACCT_LOCK:
        names = {e.stored: n for n, e in _uploads(run).items()}
    with contextlib.suppress(OSError, ValueError):
        names.update(json.loads((run.folder / _WORK_NAMES).read_text(encoding="utf-8")))
    msg = _STORED_NAME.sub(lambda m: names.get(m.group(0), "a file"), msg)
    return _scrub(msg, run.id)


def _scrub(msg: str, run_id: str = "") -> str:
    for root in sorted({str(run_id and TRY_DIR / run_id), str(run_id and KEPT_DIR / run_id),
                        str(TRY_DIR), str(KEPT_DIR), str(DATA_DIR),
                        str(REPO)}, key=len, reverse=True):
        if root and root != ".":
            msg = msg.replace(root + os.sep, "").replace(root, "")
    if run_id:
        msg = msg.replace(run_id, "")
    return msg[:400]


# Beside `work/`, not in it: an adapter reads every file in its folder.
_WORK_NAMES = "work-names.json"              # work name -> the visitor's own name
_WORK_SIG = "work.sig"


def _upload_order(run: Run) -> tuple[list[tuple[str, _Entry]], list[float] | None]:
    """The uploads in the order their own names give — by the time a name
    holds, if every name holds one, else by name — and the epoch seconds per
    snapshot when the names gave times (several files only). Anything in the
    upload folder that is not a listed upload (a part never finished before a
    restart) is removed first. Call under _ACCT_LOCK."""
    ups = _uploads(run)
    keep = {e.stored for e in ups.values()}
    for p in run.upload_folder.iterdir():
        if p.name not in keep:
            p.unlink(missing_ok=True)
    items = sorted(ups.items())
    stamps = [rec_mod.parse_stamp(name) for name, _ in items]
    if len(items) > 1 and all(s is not None for s in stamps):
        order = sorted(range(len(items)), key=lambda i: stamps[i])  # type: ignore[arg-type,return-value]
        return ([items[i] for i in order],
                [stamps[i].replace(tzinfo=UTC).timestamp() for i in order])  # type: ignore[union-attr]
    return items, None


def _set_meta(folder: Path, unit: str, fs: int, fmt: str, interval_s: float) -> None:
    meta_path = folder / "recording.json"
    old = json.loads(meta_path.read_text())
    old["unit"] = unit
    if fmt != "wav":                  # a WAV's rate comes from its own header
        old["sample_rate_hz"] = fs
    # One long WAV is cut into back-to-back snapshots and its adapter sets
    # their spacing; any other recording without time stamps is spaced by
    # the visitor's interval (time stamps, where present, win in the reader).
    if not (fmt == "wav" and old.get("wav_contiguous")):
        old["interval_s"] = interval_s
    meta_path.write_text(json.dumps(old))


def _prepare_work(run: Run, fmt: str, meta: dict[str, Any]) -> None:
    """(Re)build `work/` from `upload/`: each upload linked (or, where the
    disk cannot link, copied) under s00000.ext, s00001.ext ... in time order,
    then converted by the format adapter, which may delete what it converted
    — only there, never in `upload/`. So a refused configure, more files, and
    configure again always start from the visitor's intact files. If neither
    the upload nor the format changed since the last build, the build — and
    the parsed cache in it — is kept, and only the metadata is updated."""
    with _ACCT_LOCK:
        items, t_s = _upload_order(run)
        # The file's identity and write time, not only its size: a replacement
        # of the same size must still rebuild the parsed cache.
        def ident(e: Any) -> tuple[int, int]:
            try:
                st = (run.upload_folder / e.stored).stat()
            except OSError:
                return (0, 0)
            return (st.st_ino, st.st_mtime_ns)
        sig = hashlib.sha256(json.dumps(
            [fmt, bool(meta.get("has_timestamp")),
             [(n, e.stored, e.size, *ident(e)) for n, e in items]]).encode()).hexdigest()
    work = run.work_folder
    with contextlib.suppress(OSError):
        if (run.folder / _WORK_SIG).read_text() == sig and (work / "recording.json").exists():
            _set_meta(work, meta["unit"], meta["sample_rate_hz"], fmt, meta["interval_s"])
            return
    (run.folder / _WORK_SIG).unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    _remeasure(run.id)
    work.mkdir()
    names = {}
    for rank, (name, e) in enumerate(items):
        dst = f"s{rank:05d}{Path(e.stored).suffix.lower()}"
        names[dst] = name
        try:
            os.link(run.upload_folder / e.stored, work / dst)
        except OSError:
            _room(run.id, e.size)
            shutil.copyfile(run.upload_folder / e.stored, work / dst)
            _charge(run.id, e.size, check=False)
    (run.folder / _WORK_NAMES).write_text(json.dumps(names), encoding="utf-8")
    if t_s is not None:
        meta["t_s"] = t_s
    # The converted data may hold MAX_TOTAL_SAMPLES float32, and a wide CSV
    # is staged once before it is written: room for both first.
    _room(run.id, 2 * 4 * MAX_TOTAL_SAMPLES)
    format_adapters.adapt(fmt, work, meta, max_snapshots=MAX_SNAPSHOTS,
                          max_samples=MAX_SAMPLES_PER_SNAPSHOT,
                          max_total_samples=MAX_TOTAL_SAMPLES)
    (run.folder / _WORK_SIG).write_text(sig)
    _remeasure(run.id)


@contextlib.contextmanager
def _reading(run: Run):
    """`configure` has the run's upload to itself: no upload is being
    written, and no other `configure` runs on it at the same time."""
    with _ACCT_LOCK:
        if _ACTIVE.get(run.id, 0) != 0:
            raise TryError("this run is busy (still uploading, or being read); "
                           "try again when that has finished")
        _ACTIVE[run.id] = -1
    try:
        yield
    finally:
        with _ACCT_LOCK:
            if _ACTIVE.get(run.id) == -1:
                _ACTIVE[run.id] = 0


def configure(run_id: str, body: dict[str, Any], client_ip: str | None = None) -> dict[str, Any]:
    run = _get_or_404(run_id)
    # A finished run (done or failed) may be configured again: "change
    # settings and run again" on the result page. The upload is untouched,
    # so this only resets the run to "configured" and clears its old report.
    _ensure_status(run, {"collecting", "configured", "done", "failed"})
    checks = [("configure-run", run.id, CONFIGURES_PER_RUN_PER_HOUR,
               f"this run has been configured {CONFIGURES_PER_RUN_PER_HOUR} times this "
               "hour; please try again later")]
    if client_ip is not None:
        checks.append(("configure", _ip_hash(client_ip), CONFIGURES_PER_ADDRESS_PER_HOUR,
                       "too many requests from one address this hour; please try again "
                       "later"))
    _take_rates(checks)
    with _reading(run):
        try:
            return _configure(run, body)
        except TryError as exc:
            if type(exc) in (TryError, TooLarge):
                raise type(exc)(_visitor_message(run, str(exc))) from None
            raise
        except (format_adapters.AdapterError, ValueError, OSError, IndexError,
                TypeError, KeyError, EOFError, MemoryError) as exc:
            # Whatever else an untrusted file makes a reader raise is the
            # visitor's bad input, not a server fault: a 4xx with a plain
            # reason, never a 500 (HARM-S10-T03 L1).
            with _ACCT_LOCK:
                uploaded = [run.upload_folder / e.stored for e in _uploads(run).values()]
            reason = str(exc) if isinstance(exc, (format_adapters.AdapterError, ValueError)) \
                else f"the recording could not be read ({type(exc).__name__})"
            _log_refused(uploaded, str(body.get("format", "auto")), reason)
            raise TryError(_visitor_message(run, reason)) from None
        finally:
            _remeasure(run.id)


def _configure(run: Run, body: dict[str, Any]) -> dict[str, Any]:
    # A probe reads the upload and says what it found (snapshots, samples,
    # channels, time stamps) before the visitor is asked anything else: no
    # sampling rate or machine type is needed yet, and the run is not marked
    # configured, so the real configure that follows reads it again.
    probe = bool(body.get("probe"))
    if probe and run.status != "collecting":
        raise TryError("this recording has already been read; start over to change the file")
    family = str(body.get("family", "") or ("generic" if probe else ""))
    family = FAMILY_ALIASES.get(family, family)
    if family not in FAMILIES:
        raise TryError(f"unknown machine family {family[:40]!r}; choose one of {FAMILIES}")
    # Whether the family's network is actually built is checked when the run
    # starts (`_do_run`) — a server configuration question, not something
    # wrong with the visitor's file, and irrelevant to validating it here.
    with _ACCT_LOCK:
        uploaded = [run.upload_folder / e.stored for e in _uploads(run).values()]
    # Configured once already: the upload was converted then, so the format
    # asked for the first time stands (a second adapter pass would rewrite
    # recording.json and lose what the first one read, e.g. a WAV's rate).
    fmt = run.format if run.status == "configured" else str(body.get("format", "auto"))
    if fmt == "auto" and uploaded and all(p.suffix.lower() == ".wav" for p in uploaded):
        fmt = "wav"
    try:
        fs = int(body.get("sample_rate_hz") or 0)
    except (TypeError, ValueError, OverflowError):
        raise TryError("the sampling rate must be a whole number of Hz") from None
    if probe and fmt != "wav" and fs == 0:
        fs = PROBE_SAMPLE_RATE_HZ            # a stand-in, never reported or run
    # A WAV file carries its own rate in its header, and that one is used.
    if fmt != "wav" and not MIN_SAMPLE_RATE_HZ <= fs <= MAX_SAMPLE_RATE_HZ:
        raise TryError(f"the sampling rate must be between {MIN_SAMPLE_RATE_HZ} and "
                       f"{MAX_SAMPLE_RATE_HZ} Hz")
    unit = str(body.get("unit", "g"))
    if unit not in ("g", "mm/s"):
        raise TryError("the vibration unit must be g or mm/s")
    rpm_in = body.get("rpm")
    rpm_limited = rpm_in in (None, "", 0)
    rpm = None
    if not rpm_limited:
        try:
            rpm = float(rpm_in)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise TryError("shaft speed must be a number of rpm") from None
        if not 60 <= rpm <= 60000:
            raise TryError("shaft speed must be between 60 and 60000 rpm")
    try:
        channel = int(body.get("channel") or 0)
    except (TypeError, ValueError, OverflowError):
        raise TryError("the channel must be a whole number") from None
    given = body.get("params") or {}
    if not isinstance(given, dict):
        raise TryError("the machine's details must be a set of named numbers")
    given = {str(k)[:40]: v for k, v in given.items()}
    known = {p["key"] for p in families.get(family).get("wizard", {}).get("parameters", [])}
    if set(given) - known:
        raise TryError(f"unknown machine detail(s) {sorted(set(given) - known)} for "
                       f"{FAMILY_LABELS[family].lower()}")
    _family_params(family, given)
    params = {k: float(v) for k, v in given.items() if v not in (None, "")}
    interval_in = body.get("interval_s")
    if interval_in in (None, ""):
        interval_s = DEFAULT_INTERVAL_S
    else:
        try:
            interval_s = float(interval_in)
        except (TypeError, ValueError):
            raise TryError("the time between snapshots must be a number of seconds") from None
        if not MIN_INTERVAL_S <= interval_s <= MAX_INTERVAL_S:
            raise TryError("the time between snapshots must be between 1 second and "
                           "30 days")

    meta: dict[str, Any] = {"name": "try upload", "sample_rate_hz": fs,
                            "interval_s": interval_s,
                            "unit": unit, "provenance": "uploaded through /try",
                            "has_timestamp": bool(body.get("has_timestamp"))}
    if run.status in ("configured", "done", "failed"):
        _set_meta(run.work_folder, unit, fs, fmt, interval_s)
    else:
        if not uploaded:
            raise TryError("no snapshot files were uploaded")
        _prepare_work(run, fmt, meta)
    r = _recording(run)
    fs = int(r.sample_rate_hz)
    if not MIN_SAMPLE_RATE_HZ <= fs <= MAX_SAMPLE_RATE_HZ:
        raise TryError(f"the sampling rate ({fs} Hz) must be between "
                       f"{MIN_SAMPLE_RATE_HZ} and {MAX_SAMPLE_RATE_HZ} Hz")
    n = len(r.hours())
    # Commissioning learns normal from at least MIN_BASELINE readings
    # (api/machines.py); a recording needs those plus some to watch after.
    need = machines_mod.MIN_BASELINE + MIN_WATCHED_SNAPSHOTS
    if n < need:
        raise TryError(f"the recording holds {n} snapshots; Harmonic needs at least {need}: "
                       f"{machines_mod.MIN_BASELINE} to learn normal from and "
                       f"{MIN_WATCHED_SNAPSHOTS} to watch after that")
    if n > MAX_SNAPSHOTS:
        raise TryError(f"the recording holds {n} snapshots; at most {MAX_SNAPSHOTS} "
                       "are accepted")
    if r.snapshot_samples() > MAX_SAMPLES_PER_SNAPSHOT:
        raise TryError(f"{r.snapshot_samples()} samples per snapshot exceeds the limit "
                       f"of {MAX_SAMPLES_PER_SNAPSHOT}")
    if not 0 <= channel < r.n_channels:
        raise TryError(f"the recording has {r.n_channels} channel(s); choose channel 1 to "
                       f"{r.n_channels}")

    window = rec_mod.window_for(r.snapshot_samples())
    if n * window > MAX_TOTAL_SAMPLES:
        raise TooLarge(f"{n} snapshots of {window} samples is more than the "
                       f"{MAX_TOTAL_SAMPLES} samples one run may analyse; send fewer or "
                       "shorter snapshots")
    _room(run.id, 4 * n * window)             # the channel's float32 cache
    data = r.load(channel, window)
    if not np.isfinite(data).all():
        raise TryError("the recording contains a missing or non-numeric sample")
    rms = rec_mod.rms(data)
    del data
    hours = r.hours()
    default_hi = float(hours[max(int(n * 0.3), machines_mod.MIN_BASELINE) - 1])
    if probe:
        timed_by = _timed_by(r)
        return {
            "probe": True, "snapshots": n, "samples_per_snapshot": r.snapshot_samples(),
            "n_channels": r.n_channels, "format": fmt, "files": len(uploaded),
            "timed_by": timed_by,
            # The recording's own spacing, when it has one; else the visitor chooses.
            "interval_s": float(r.interval_s) if timed_by != "interval" else None,
            "span_h": round(float(hours[-1] - hours[0]), 4)
            if timed_by != "interval" else None,
            "sample_rate_hz": int(r.sample_rate_hz) if fmt == "wav" else None,
        }

    if run.status in ("done", "failed"):
        # Re-configuring a finished run for another go: the old report and
        # chart no longer describe what this run will show; the upload and
        # (for a kept run) its consent are untouched.
        run.report = {}
        run.error = None
        run.progress = {}
    run.family, run.unit, run.sample_rate_hz, run.format = family, unit, fs, fmt
    run.rpm, run.rpm_limited = rpm, rpm_limited
    run.params, run.interval_s = params, float(r.interval_s)
    run.n_snapshots, run.n_channels, run.channel = n, r.n_channels, channel
    run.status = "configured"
    if not _save(run):
        raise RunNotFound("no such run, or it has already been deleted")
    return {
        "hours": [round(float(h), 4) for h in hours], "at": r.stamps(),
        "rms": [round(float(v), 6) for v in rms], "unit": r.unit,
        "snapshots": n, "sample_rate_hz": r.sample_rate_hz,
        "default_baseline_h": [float(hours[0]), default_hi],
        "rpm_used": rpm if rpm is not None else FAMILY_DEFAULT_RPM[family],
        "rpm_limited": rpm_limited,
        "min_baseline_snapshots": machines_mod.MIN_BASELINE,
        "n_channels": r.n_channels, "channel": channel,
        "interval_s": float(r.interval_s), "timed_by": _timed_by(r),
    }


def _timed_by(r: rec_mod.Recording) -> str:
    """What spaces the snapshots in time: "timestamps" (the recording's own),
    "wav" (one long WAV, cut back to back) or "interval" (the visitor's)."""
    if r.stamps() and r.stamps()[0] is not None:
        return "timestamps"
    folder = getattr(r, "folder", None)
    with contextlib.suppress(OSError, ValueError):
        if folder is not None and json.loads(
                (Path(folder) / "recording.json").read_text()).get("wav_contiguous"):
            return "wav"
    return "interval"


def _log_refused(uploaded: list[Path], fmt: str, reason: str) -> None:
    """The rejected-format log (formats.log_rejected): no file name (stored
    names are scrubbed too), no path, no cell of the file."""
    reason = _scrub(_STORED_NAME.sub("<file>", reason))
    exts = sorted({p.suffix.lower() for p in uploaded})
    head = b""
    if uploaded:
        with contextlib.suppress(OSError), uploaded[0].open("rb") as f:
            head = f.read(8)
    size = 0
    for p in uploaded:
        with contextlib.suppress(OSError):
            size += p.stat().st_size
    format_adapters.log_rejected("adapter", ext=",".join(exts), head=head, fmt=fmt[:32],
                                 reason=reason, size=size)


# --- the single-flight queue ------------------------------------------------------------


_QUEUE: queue.Queue[str] = queue.Queue(maxsize=QUEUE_DEPTH)
_START_LOCK = threading.Lock()
_WORKER_STARTED = threading.Event()
# How long a run takes, for the "try again in N minutes" answer (G5).
_RUN_SECONDS = {"avg": 180.0}


def _busy_minutes() -> int:
    return max(1, int(np.ceil((_QUEUE.qsize() + 1) * _RUN_SECONDS["avg"] / 60.0)))


def start(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
    run = _get_or_404(run_id)
    _ensure_status(run, {"configured"})
    bh = body.get("baseline_h")
    if not bh or len(bh) != 2:
        raise TryError("choose a baseline period")
    try:
        lo, hi = float(bh[0]), float(bh[1])
    except (TypeError, ValueError):
        raise TryError("the baseline period must be two numbers of hours") from None
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise TryError("the baseline period must end after it starts")
    # Room for what the run writes: the frames of its baseline (at most the
    # cache's size again), its brains and profiles.
    try:
        window = rec_mod.window_for(_recording(run).snapshot_samples())
    except (ValueError, OSError):
        window = rec_mod.MAX_WINDOW
    _room(run.id, 4 * run.n_snapshots * window + _RUN_OVERHEAD_BYTES)
    with _START_LOCK:
        # A full queue is "busy, try again", not a failed run, and it does
        # not count against anyone's hourly limit (HARM-S10-T03 G5).
        if _QUEUE.full():
            mins = _busy_minutes()
            raise Busy(f"the try service is busy; please try again in about {mins} "
                       f"minute{'s' if mins != 1 else ''}", retry_after_s=mins * 60)
        _take_rates([("start", run.client_ip_hash, RATE_LIMIT_PER_HOUR,
                      f"at most {RATE_LIMIT_PER_HOUR} runs per hour from one address; "
                      "please try again later"),
                     ("start", "*", GLOBAL_RUNS_STARTED_PER_HOUR,
                      "the try service has run as many recordings as it can this hour; "
                      "please try again later")])
        run.baseline_h = (lo, hi)
        run.consent = bool(body.get("consent", False))
        run.status = "queued"
        run.progress = {"step": 0, "of": len(machines_mod.STEPS), "label": "Queued",
                        "stage": "queue", "ahead": _QUEUE.qsize()}
        run.error = None
        if not _save(run):
            raise RunNotFound("no such run, or it has already been deleted")
        _QUEUE.put_nowait(run.id)          # never full: checked above, under the lock
    return {"status": "queued", "queue_depth": _QUEUE.qsize()}


def delete_run(run_id: str) -> None:
    run = _get_or_404(run_id)
    _delete(run.id)


# --- the worker --------------------------------------------------------------------------


def _baseline(run: Run) -> tuple[float, float]:
    """The run's baseline period. `start` always sets it before a run is
    queued; a run.json without one (hand-edited, or written by an older
    version) is refused as the visitor's error, never a crash."""
    if run.baseline_h is None:
        raise TryError("choose a baseline period")
    return run.baseline_h


def _build_machine(run: Run, r: rec_mod.Recording) -> machines_mod.Machine:
    rpm = run.rpm if run.rpm is not None else FAMILY_DEFAULT_RPM[run.family]
    params, _, _ = _run_params(run, rpm)
    wiz = families.get(run.family).get("wizard", {})
    return machines_mod.Machine(
        id=run.id, name=f"Try run {run.id[:8]}", equipment_id=f"TRY-{run.id[:8].upper()}",
        nominal_rpm=rpm, vib_unit=run.unit, iso_class="none",
        bearing=_generic_bearing(),
        components=list(wiz.get("components") or ["bearing", "shaft", "foundation"]),
        recording=r.key, channel=run.channel, baseline_h=_baseline(run),
        speed="constant", family=run.family, params=params, base_dir=str(TRY_DIR),
    )


def _check_live(run: Run) -> None:
    if _dead(run.id):
        raise _Cancelled(run.id)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _move_to_kept(run: Run) -> None:
    """Move a finished, consented run to `KEPT_DIR` (HARM-S10-T05). Only a
    run that reached "done" is ever moved — a failed or timed-out one stays
    under the normal 7-day prefix even if consent was ticked, since nothing
    usable was produced.

    "Write, then remove": everything but run.json is copied (hard-linked
    where possible) into the new folder first; the new run.json — the commit
    point, since `_locate` treats `KEPT_DIR` as authoritative once it exists
    — is written last; only then is the old folder removed. A crash before
    the commit leaves only the old, unmoved copy (the sweeper retries it
    next pass, HARM-S10-T05); a crash after leaves a stray old copy that
    every reader already ignores and `_delete` cleans up regardless. A
    concurrent "delete now" wins: this backs out and leaves nothing under
    `KEPT_DIR` if the run is marked dead before the commit.
    """
    if run.kept or not run.consent or _dead(run.id):
        return
    old, new = TRY_DIR / run.id, KEPT_DIR / run.id
    if not old.is_dir():
        return
    new.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(old):
        rel = Path(root).relative_to(old)
        for d in dirs:
            with contextlib.suppress(OSError):
                (new / rel / d).mkdir(exist_ok=True)
        for name in files:
            if name in ("run.json", "run.json.tmp") or _dead(run.id):
                continue
            dst = new / rel / name
            if not dst.exists():
                with contextlib.suppress(OSError):
                    _link_or_copy(Path(root) / name, dst)
    with _CANCELLED_LOCK:
        if _dead(run.id):
            shutil.rmtree(new, ignore_errors=True)
            return
        run.kept = True
        tmp = new / "run.json.tmp"
        tmp.write_text(json.dumps(run.to_json(), indent=1))
        tmp.replace(new / "run.json")
    shutil.rmtree(old, ignore_errors=True)
    _remeasure(run.id)


def _do_run(run: Run) -> None:
    """Commission and replay one run. Once the run is deleted or timed out
    it stops at its next step or tick and writes nothing more; a deleted
    run's folder is removed again after it stops (HARM-S10-T03 M1, M2)."""
    try:
        _do_run_steps(run)
    except _Cancelled:
        pass
    finally:
        if _is_deleted(run.id):
            shutil.rmtree(run.folder, ignore_errors=True)
            _forget(run.id)
        else:
            _remeasure(run.id)


def _do_run_steps(run: Run) -> None:
    _check_live(run)
    _set_status(run, "running", progress={"step": 0, "of": len(machines_mod.STEPS),
                                          "label": "Starting", "stage": "learn"})
    try:
        r = _recording(run)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise TryError(_visitor_message(run, str(exc))) from None
    network = families.path(run.family, "network")
    if not network.exists():
        raise TryError(f"the {run.family} network is not built on this server yet")
    m = _build_machine(run, r)

    def step(i: int, tag: str = "") -> None:
        _check_live(run)
        _set_status(run, "running", progress={"step": i + 1, "of": len(machines_mod.STEPS),
                                              "label": machines_mod.STEPS[i] + tag,
                                              "stage": "learn"})

    commission_report = machines_mod._commission_point(m, r, network, run.channel,
                                                        m.brain_path, "p0", step)
    _check_live(run)
    window = rec_mod.window_for(r.snapshot_samples())
    hours = r.hours()
    data = r.load(run.channel, window)
    rpm = machines_mod.speeds_for(m, r)
    # The page's plain stages: learning normal (above), watching the rest, and
    # writing the report, with how far the watching has got.
    seen = [0]
    every = max(len(hours) // 20, 1)

    def watching() -> None:
        _check_live(run)
        if seen[0] % every == 0:
            _set_status(run, "running", progress={"stage": "watch", "done": seen[0],
                                                  "total": len(hours)})
        seen[0] += 1

    ticks = machines_mod.replay(m.brain_path, hours, data, r.sample_rate_hz, rpm,
                                check=watching)
    _check_live(run)
    _set_status(run, "running", progress={"stage": "write"})
    report = _build_report(run, m, r, hours, ticks, commission_report)
    run.report = report
    run.status = "done"
    run.progress = {}
    run.error = None
    _save(run)
    if run.consent:
        _move_to_kept(run)


def _duration(s: float) -> str:
    return f"{int(s)} seconds" if s < 120 else f"{int(s // 60)} minutes"


def _run_with_timeout(run: Run) -> None:
    errors: list[str] = []
    t0 = time.time()

    def target() -> None:
        try:
            _do_run(run)
        except (TryError, machines_mod.CommissionError) as exc:
            errors.append(_scrub(str(exc), run.id))
        except Exception as exc:
            log.exception("try run failed")
            errors.append(f"the run stopped on an internal error ({type(exc).__name__})")

    t = threading.Thread(target=target, name=f"try-run-{run.id}", daemon=True)
    t.start()
    t.join(RUN_TIMEOUT_S)
    if t.is_alive():
        # Cancelled first, then the one forced write: from here the thread's
        # own progress updates are skipped and it stops at its next step.
        with _CANCELLED_LOCK:
            _CANCELLED.add(run.id)
            _set_status(run, "failed", force=True,
                        error=f"this run took longer than {_duration(RUN_TIMEOUT_S)} "
                              "and was stopped")
        return
    _RUN_SECONDS["avg"] = 0.7 * _RUN_SECONDS["avg"] + 0.3 * (time.time() - t0)
    if _dead(run.id):
        return
    if errors:
        _set_status(run, "failed", error=errors[0])


def _worker_loop() -> None:
    while True:
        run_id = _QUEUE.get()
        try:
            run = get(run_id)
            if run is None or run.status != "queued":
                continue
            _run_with_timeout(run)
        except Exception:
            log.exception("try-mode worker failed on a run")


# --- the offline diagnosis pipeline -------------------------------------------------------


_pipeline: Any = None
_pipeline_lock = threading.Lock()


def _offline_pipeline() -> Any:
    """A `rag.pipeline.Pipeline` that can never reach a model (`offline=True`
    — see its docstring) and whose knowledge-graph store never writes back to
    the shared graph file (`persist=False`): a try run's equipment node lives
    only in this process's memory, for as long as this pipeline is open."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from kg.store import DEFAULT_JSON_PATH, NetworkXStore
            from rag.pipeline import Pipeline

            store = NetworkXStore(DEFAULT_JSON_PATH, persist=False)
            # offline=True also means BM25 retrieval, built here explicitly:
            # no embedder is ever made, so no key in the environment can send
            # a query to an embedding model (HARM-S10-T03 M3).
            from rag.index import BM25Index
            from rag.pipeline import load_manifest

            _pipeline = Pipeline(store=store, offline=True,
                                 index=BM25Index(load_manifest()))
        return _pipeline


def _register_ephemeral(store: Any, m: machines_mod.Machine) -> None:
    from kg.model import Graph, Node, Triple

    nodes = [Node("Equipment", m.equipment_id, {
        "model": "try-mode run", "serial": m.id, "nominal_rpm": m.nominal_rpm,
        "iso_class": "none", "family": m.family})]
    triples = []
    for c in m.components:
        cid = f"{m.equipment_id}.{c}"
        nodes.append(Node("Component", cid, {"kind": c, "position": c}))
        triples.append(Triple(m.equipment_id, "HAS_COMPONENT", cid))
    store.merge_graph(Graph(nodes, triples))


def _diagnose(run: Run, m: machines_mod.Machine, tick: Any) -> dict[str, Any] | None:
    names = families.class_names(m.family)
    scores = {names[i] if i < len(names) else f"class_{i}": s
             for i, s in enumerate(tick.class_scores)}
    faults = {k: v for k, v in scores.items() if k != "healthy"}
    top = sorted(faults.items(), key=lambda kv: -kv[1])[:3]
    event = {
        "event_id": f"try-{run.id}", "equipment_id": m.equipment_id, "regime": tick.regime,
        "hhi": round(tick.hhi, 4), "state": tick.state, "rul_hours": round(tick.rul_hours, 1),
        "top_classes": [{"name": k, "score": round(v, 4)} for k, v in top if v > 0.0],
        "top_surprised_features": tick.top_surprised(),
        # A family that names no fault can only ever see an unrecognised
        # change; the composer then answers from the family's own guidance.
        **({"unrecognised": True} if getattr(tick, "unrecognised", False)
           or not _names_faults(m.family) else {}),
    }
    try:
        pipeline = _offline_pipeline()
        _register_ephemeral(pipeline.store, m)
        return _without_manual_citations(pipeline.diagnose(event).to_json())
    except Exception:
        log.exception("try-mode diagnosis failed for run %s", run.id)
        return None


_SEE_REF = re.compile(r"\s*\(see [^)]*\)")

GUIDANCE_NOTE = ("These steps are general guidance for this machine type, not taken from "
                 "your machine's own manual. Check them against your manufacturer's manual "
                 "before acting.")
# Documents Harmonic wrote itself (rag/manuals/HARMONIC_GEN_GUIDANCE.md), not
# any machine's manual: the public report may cite them, labelled as such.
HARMONIC_OWN_DOCS = frozenset({"HARMONIC-GEN"})
OWN_GUIDANCE_LABEL = "Harmonic's own general guidance, not a machine manual"


def _own_guidance(ref: str) -> bool:
    return any(ref == d or ref.startswith(f"{d}-") for d in HARMONIC_OWN_DOCS)


_manual_ids: re.Pattern[str] | None = None


def _manual_id_re() -> re.Pattern[str]:
    """Any section id of any machine manual in the manifest (every document
    but Harmonic's own), wherever it appears in the text: a safety line
    lifted from a manual carries its id inline even when it was never in the
    reference list."""
    global _manual_ids
    if _manual_ids is None:
        docs = json.loads((families.REPO / "rag" / "manuals" / "manifest.json")
                          .read_text(encoding="utf-8"))["docs"]
        alt = "|".join(re.escape(d) for d in docs if d not in HARMONIC_OWN_DOCS)
        _manual_ids = re.compile(rf"\b(?:{alt})(?:-\d+(?:\.\d+)*)?\b" if alt else r"(?!)")
    return _manual_ids


def _without_manual_citations(diag: dict[str, Any]) -> dict[str, Any]:
    """The public report never cites a manual that is not the visitor's own
    (owner, 2026-09-29): the knowledge graph's manuals belong to the demo
    fleet (P-101 and the like). Drops the retrieved chunk ids and every
    manual reference, and strips every citation from the text. Harmonic's own
    guidance (HARMONIC_OWN_DOCS) is not a machine manual: it stays in the
    reference list, labelled. Try mode only; the composer, and so the
    console's diagnosis, is unchanged."""
    ids = {str(i) for i in (diag.get("retrieved") or [])}
    resp = dict(diag.get("response") or {})
    ids |= {str(i) for i in (resp.get("references") or [])}
    own = [str(i) for i in (resp.get("references") or []) if _own_guidance(str(i))]

    def clean(text: Any) -> Any:
        if not isinstance(text, str):
            return text
        text = _SEE_REF.sub("", text)
        for i in sorted(ids, key=len, reverse=True):
            if not _own_guidance(i):
                text = text.replace(i, "")
        text = _manual_id_re().sub("", text)
        text = re.sub(r"\s*\(\s*[,;\s]*\)", "", text)       # "()" left by a removed id
        text = re.sub(r"\s*\[\s*[,;\s]*\]", "", text)       # "[]" likewise
        return re.sub(r"\s{2,}", " ", text).strip()

    resp.pop("references", None)
    if own:
        resp["references"] = own
        resp["references_label"] = OWN_GUIDANCE_LABEL
    for key in ("diagnosis", "evidence"):
        if key in resp:
            resp[key] = clean(resp[key])
    for key in ("root_causes", "safety"):
        if isinstance(resp.get(key), list):
            resp[key] = [clean(x) for x in resp[key]]
    if isinstance(resp.get("actions"), list):
        resp["actions"] = [{**a, "step": clean(a.get("step"))} if isinstance(a, dict) else a
                           for a in resp["actions"]]
    resp["guidance_note"] = GUIDANCE_NOTE
    out = {k: v for k, v in diag.items() if k != "retrieved"}
    out["response"] = resp
    return out


# --- the report ----------------------------------------------------------------------------


_RANK = {"OK": 0, "WATCH": 1, "WARNING": 2, "CRITICAL": 3, "FAILED": 4}


def _downsample(points: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if len(points) <= cap:
        return points
    step = len(points) / cap
    return [points[int(i * step)] for i in range(cap)]


def _caveats(run: Run) -> list[str]:
    out = [
        "A fault already present during the learning period is learned as normal and "
        "will not be flagged: choose a period before the problem started, if you can.",
        "This is one offline replay of the file you gave us; it does not show how fast "
        "an alarm would have reached a phone or a work order in real time.",
        "A generic bearing geometry was assumed (this wizard does not ask for yours), "
        "so ball-pass frequencies are approximate.",
    ]
    if run.rpm_limited:
        out.append(f"No shaft speed was given: {FAMILY_DEFAULT_RPM[run.family]:.0f} rpm "
                    f"({FAMILY_LABELS[run.family].lower()}'s typical speed) was assumed, so "
                    "speed-order diagnosis is limited.")
    if not _names_faults(run.family):
        out.append("Harmonic has no trained fault vocabulary for this machine type, so "
                   "this run only watches for change and does not name a fault type.")
    if run.n_channels > 1:
        out.append(f"The recording has {run.n_channels} channels; channel {run.channel + 1} "
                   "was analysed. To analyse another, choose it on step 3.")
    if run.format == "wav":
        out.append("WAV samples are fractions of full scale, not calibrated units: the "
                    "run compares the machine with its own normal, so its timing holds, "
                    "but absolute vibration levels in this report are not calibrated.")
    out.append("This public page uses Harmonic's offline diagnosis only. The full "
               "product's diagnosis is stronger; ask us to run it on this recording.")
    return out


def _spacing(s: float) -> str:
    """Seconds in the unit the upload form offers: minutes when whole."""
    if s >= 60 and float(s / 60).is_integer():
        m = int(s // 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    return f"{_num(s)} second{'s' if s != 1 else ''}"


def _num(v: float) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def _assumed(run: Run, m: machines_mod.Machine) -> list[str]:
    """Every value the run assumed rather than was given, in plain words."""
    out = []
    if run.rpm_limited:
        out.append(f"Shaft speed: {_num(m.nominal_rpm)} rpm, a typical speed for "
                   f"{FAMILY_LABELS[run.family].lower()} (not given)")
    params, assumed, moved = _run_params(run, m.nominal_rpm)
    for p in assumed:
        if p["key"] == moved:
            out.append(f"{p['label']}: {_num(params[p['key']])}, the nearest value to the "
                       f"typical {_num(float(p['default']))} that fits a shaft speed of "
                       f"{_num(m.nominal_rpm)} rpm (not given)")
        else:
            out.append(f"{p['label']}: {_num(float(p['default']))}, a typical value for "
                       "this machine type (not given)")
    out.append(f"Bearing: a generic {GENERIC_BEARING_KEY} deep-groove ball bearing's "
               "geometry (not asked for)")
    return out


# How far the machine is from its own normal, in units of "the edge of normal":
# half a feature's dead zone, which commissioning sets to twice the largest
# surprise seen on the learning period (api/machines.py measure_floor). So 1
# is at least the most a feature moved while the machine ran normally. The
# engine's novelty path raises an early sign when a feature passes two dead
# zones (4 on this scale) for five snapshots running (engine/src/
# harmonic_health.cpp, novelty_factor_q16); the chart draws that line.
EDGE_OF_DEAD_ZONE = 0.5
ALARM_LEVEL = 4.0
CHART_POINTS = 600

LEVEL_WORDS = {"WATCH": "Early sign (Watch)", "WARNING": "Warning", "CRITICAL": "Critical"}


def human_duration(h: float) -> str:
    """A time span as a person says it: "40 seconds", "12 min", "1 h 21 min",
    "26 hours", "3.4 days"."""
    sec = max(0.0, float(h) * 3600.0)
    if sec < 90:
        n = round(sec)
        return f"{n} second{'s' if n != 1 else ''}"
    mins = round(sec / 60)
    if mins < 60:
        return f"{mins} min"
    if h < 10:
        hh, mm = divmod(mins, 60)
        return f"{hh} h {mm} min" if mm else f"{hh} h"
    if h < 48:
        return f"{round(h)} hours"
    return f"{h / 24:.1f} days"


def human_span(h: float) -> str:
    """A recording's length: "45 min", "5.6 hours", "6.8 days"."""
    if h < 1:
        return human_duration(h)
    if h < 48:
        return f"{h:.1f} hours"
    return f"{h / 24:.1f} days"


def _dead_zones(m: machines_mod.Machine, window: int, fs: int) -> dict[str, float]:
    """Each watched feature's dead zone in the run's commissioned brain."""
    from api.rig import HarmonicRig

    rig = HarmonicRig(m.brain_path, window_samples=window, sample_rate_hz=fs)
    try:
        return {k: float(dz) for k, dz in zip(rig.feature_kinds, rig.feature_dead_zones,
                                              strict=False) if dz > 0}
    finally:
        rig.close()


def _distances(tick: Any, dz: dict[str, float]) -> dict[str, float]:
    """Each watched feature's distance from normal at one tick (1 = the edge)."""
    return {k: abs(float(v)) / (dz[k] * EDGE_OF_DEAD_ZONE)
            for k, v in tick.surprise.items() if k in dz}


def _series(hours: np.ndarray, ticks: list[Any], dz: dict[str, float],
            cap: int = CHART_POINTS) -> list[dict[str, Any]]:
    """The chart: hours since the start and the largest distance from normal,
    at most `cap` points, each the largest in its stretch so no peak is lost.
    None while the engine is still warming up (no judgement yet)."""
    t0 = float(hours[0])
    vals: list[float | None] = []
    for t in ticks:
        d = _distances(t, dz) if t.warm else {}
        vals.append(max(d.values()) if d else None)
    n = len(vals)
    step = max(1, -(-n // cap))
    out = []
    for i in range(0, n, step):
        chunk = [v for v in vals[i:i + step] if v is not None]
        out.append({"h": round(float(hours[i]) - t0, 4),
                    "d": round(max(chunk), 3) if chunk else None})
    return out


_HHI_SENTENCE = re.compile(r"\s*The health index is -?[0-9.]+\s*\([A-Z]+\)"
                           r"(?: in the [^.]*? regime)?\.")


def _plain_diagnosis(diag: dict[str, Any] | None, equipment_id: str) -> dict[str, Any] | None:
    """The public report's wording: no health index (an internal number the
    visitor never sees, which read "0.00 (WATCH)" on a change the classes do not
    name), and "your machine" for the run's internal equipment id."""
    if not diag:
        return diag
    resp = dict(diag.get("response") or {})

    def plain(text: str) -> str:
        return _HHI_SENTENCE.sub("", text).replace(equipment_id, "your machine").strip()

    if isinstance(resp.get("diagnosis"), str):
        resp["diagnosis"] = plain(resp["diagnosis"])
    for key in ("root_causes", "safety"):
        if isinstance(resp.get(key), list):
            resp[key] = [plain(x) if isinstance(x, str) else x for x in resp[key]]
    if isinstance(resp.get("actions"), list):
        resp["actions"] = [{**a, "step": plain(str(a.get("step", "")))}
                           if isinstance(a, dict) else a for a in resp["actions"]]
    resp.pop("rul_hours", None)
    return {**diag, "response": resp}


def _named_fault(diag: dict[str, Any] | None) -> str | None:
    """The fault the diagnosis names, in its own words, or None."""
    if not diag or not diag.get("top_candidate"):
        return None
    text = str((diag.get("response") or {}).get("diagnosis", ""))
    head, sep, _ = text.partition(" on your machine")
    return head.strip() if sep and head.strip() and "nrecognised" not in head else None


def _headline(first: dict[str, Any], names_faults: bool,
              diag: dict[str, Any] | None) -> str:
    """The report's one plain answer: would this have warned me, when, and about what."""
    alarms = [(s, first[s]) for s in ("WATCH", "WARNING", "CRITICAL") if s in first]
    if not alarms:
        return "Harmonic saw nothing unusual in this recording."
    state, a = min(alarms, key=lambda sa: sa[1]["hours"])
    when = (f"{a['before_end']} before the end of your recording"
            if a["before_end_h"] >= 1 / 60 else "at the very end of your recording")
    if state == "WATCH":
        out = f"Harmonic noticed a change {when}, and would have warned you."
        w = first.get("WARNING")
        if w is not None and w["before_end_h"] >= 1 / 60:
            out += f" Its first Warning came {w['before_end']} before the end."
    else:
        out = f"Harmonic would have warned you {when}."
    fault = _named_fault(diag)
    if not names_faults:
        out += " It can't name the fault type, because you chose Other."
    elif fault:
        out += f" What it points to: {fault[0].lower() + fault[1:]}."
    else:
        out += (" The change doesn't match a fault Harmonic knows for this machine type, "
                "so it names none.")
    return out


TIMES = "\u00d7"                       # the multiplication sign


def _evidence_note(d: float) -> str:
    """The one explanation of "N\u00d7 its normal range", with the report's own N."""
    return (f"\"{d:.1f}{TIMES} its normal range\" means that measure moved {d:.1f} times as "
            "far from its usual value as the most it moved while the machine was running "
            "normally (the learning period).")


def _build_report(run: Run, m: machines_mod.Machine, r: rec_mod.Recording, hours: np.ndarray,
                  ticks: list[Any], commission_report: dict[str, Any]) -> dict[str, Any]:
    lo, hi = _baseline(run)
    t0 = float(hours[0])
    interval_s = float(getattr(r, "interval_s", run.interval_s))
    timed_by = _timed_by(r)
    warm = [i for i, t in enumerate(ticks) if t.warm]
    post = [i for i in warm if hours[i] > hi]
    dz = _dead_zones(m, rec_mod.window_for(r.snapshot_samples()), int(r.sample_rate_hz))

    first: dict[str, Any] = {}
    for state in ("WATCH", "WARNING", "CRITICAL"):
        for i in post:
            if _RANK[ticks[i].state] >= _RANK[state]:
                into, left = float(hours[i] - t0), float(hours[-1] - hours[i])
                first[state] = {"hours": round(into, 3), "before_end_h": round(left, 3),
                                "into": human_duration(into),
                                "before_end": human_duration(left),
                                "at": r.stamps()[i], "level": LEVEL_WORDS[state]}
                break

    worst_i = max(post, key=lambda i: _RANK[ticks[i].state], default=None)
    evidence: list[str] = []
    top_d: float | None = None
    diagnosis = None
    if worst_i is not None and ticks[worst_i].state != "OK":
        from api.feature_labels import labels

        names = labels(m.family)
        dist = _distances(ticks[worst_i], dz)
        for kind, d in sorted(dist.items(), key=lambda kv: -kv[1])[:3]:
            if d < 1.0:
                break
            short = kind.split(":", 1)[1] if ":" in kind else kind.removeprefix(
                "HARMONIC_FEAT_")
            label = names.get(short, short.replace("_", " ").lower())
            direction = "up" if ticks[worst_i].surprise[kind] > 0 else "down"
            evidence.append(f"{label} {direction}, {d:.1f}{TIMES} its normal range")
            top_d = d if top_d is None else top_d
        diagnosis = _plain_diagnosis(_diagnose(run, m, ticks[worst_i]), m.equipment_id)

    span = float(hours[-1] - t0)
    alarms = [first[s] | {"state": s} for s in ("WATCH", "WARNING", "CRITICAL") if s in first]
    first_any = min(alarms, key=lambda a: a["hours"]) if alarms else None
    return {
        "headline": _headline(first, _names_faults(run.family), diagnosis),
        "chart": {"points": _series(hours, ticks, dz), "normal": 1.0,
                  "alarm": ALARM_LEVEL, "end_h": round(span, 4),
                  "first_alarm_h": first_any["hours"] if first_any else None},
        "first_alarm": first_any,
        "baseline_h": [round(lo - t0, 3), round(hi - t0, 3)],
        "baseline_text": human_span(hi - lo),
        "first_watch": first.get("WATCH"), "first_warning": first.get("WARNING"),
        "first_critical": first.get("CRITICAL"),
        "false_alarms_in_baseline": commission_report.get("false_alarms", 0),
        "warm_readings_in_baseline": commission_report.get("warm_readings", 0),
        "evidence": evidence, "evidence_note": _evidence_note(top_d) if top_d is not None else None,
        "diagnosis": diagnosis,
        "family": run.family, "family_label": FAMILY_LABELS[run.family],
        "names_faults": _names_faults(run.family),
        "assumed": _assumed(run, m),
        "interval_s": interval_s, "interval_text": _spacing(interval_s),
        "timed_by": timed_by,
        "rpm_used": m.nominal_rpm, "rpm_limited": run.rpm_limited,
        "unit": r.unit, "sample_rate_hz": r.sample_rate_hz, "n_snapshots": len(hours),
        "channel": run.channel + 1, "n_channels": run.n_channels,
        "span_h": round(span, 2), "span_text": human_span(span),
        "sample": dict(SAMPLE) if run.sample else None,
        "caveats": _caveats(run) + [
            u["reason"] for u in commission_report.get("unmeasured", [])] + ([
            f"The recording carries no time stamps, so its snapshots were taken to be "
            f"{_spacing(interval_s)} apart, as you chose; every time in this report "
            "rests on that."] if timed_by == "interval" else []),
        "contact_url": os.environ.get("HARMONIC_CONTACT_URL") or None,
    }


# --- limits, for the frontend to mirror -----------------------------------------------------


def limits() -> dict[str, Any]:
    return {
        "max_total_mb": MAX_TOTAL_BYTES // (1024 * 1024), "max_files": MAX_FILES,
        "max_snapshots": MAX_SNAPSHOTS, "max_samples_per_snapshot": MAX_SAMPLES_PER_SNAPSHOT,
        "min_sample_rate_hz": MIN_SAMPLE_RATE_HZ, "max_sample_rate_hz": MAX_SAMPLE_RATE_HZ,
        "families": FAMILY_LABELS, "rate_limit_per_hour": RATE_LIMIT_PER_HOUR,
        # What the page may ask per family: its wizard's own parameters
        # (optional there; each left empty takes its default and the report
        # says so), and whether the family names faults at all.
        "family_info": {k: {"names_faults": _names_faults(k),
                            "parameters": [{f: p[f] for f in ("key", "label", "default",
                                                               "min", "max")}
                                           for p in v.get("wizard", {}).get("parameters", [])]}
                        for k, v in _REGISTRY.items()},
        "default_interval_s": DEFAULT_INTERVAL_S,
        "min_interval_s": MIN_INTERVAL_S, "max_interval_s": MAX_INTERVAL_S,
        "retention_days": RETENTION_DAYS,
        "consent_retention_days": CONSENT_RETENTION_DAYS,
        "max_request_mb": MAX_REQUEST_BYTES // (1024 * 1024),
        "upload_part_bytes": UPLOAD_PART_BYTES,
        "max_total_samples": MAX_TOTAL_SAMPLES,
        "contact_url": os.environ.get("HARMONIC_CONTACT_URL") or None,
        # The shortest recording a run accepts (checked in `configure`), and
        # the snapshot length one long WAV is cut into — the page turns the
        # two into "how long a recording must be" before anything is uploaded.
        "min_snapshots": machines_mod.MIN_BASELINE + MIN_WATCHED_SNAPSHOTS,
        "wav_snapshot_samples": format_adapters._WAV_SNAPSHOT_SAMPLES,
    }


# --- deletion sweeper -----------------------------------------------------------------------


def sweep() -> int:
    """Delete runs older than `RETENTION_DAYS` in the normal prefix, and
    those older than `CONSENT_RETENTION_DAYS` in `KEPT_DIR` (HARM-S10-T05);
    finish any delete a restart interrupted (a marker without its folder
    removed), and any move a restart interrupted (a finished, consented run
    still sitting in the normal prefix — `_move_to_kept` is otherwise called
    only once, right when a run finishes). Returns how many runs were
    removed."""
    now = time.time()
    cutoff = now - RETENTION_DAYS * 86400
    consent_cutoff = now - CONSENT_RETENTION_DAYS * 86400
    n = 0
    marks = TRY_DIR / ".deleted"
    if marks.is_dir():
        for m in marks.iterdir():
            if (TRY_DIR / m.name).exists() or (KEPT_DIR / m.name).exists():
                shutil.rmtree(TRY_DIR / m.name, ignore_errors=True)
                shutil.rmtree(KEPT_DIR / m.name, ignore_errors=True)
                _forget(m.name)
            else:
                # Kept a while after the folder is gone: a timed-out worker
                # may still be running and must keep finding it deleted.
                with contextlib.suppress(OSError):
                    if now - m.stat().st_mtime > 2 * 86400:
                        m.unlink()
    if TRY_DIR.is_dir():
        for p in TRY_DIR.iterdir():
            if not p.is_dir() or not RUN_ID_RE.match(p.name):
                continue
            run = get(p.name)
            if run is None:
                continue                 # unreadable leftover; left for a human, not guessed at
            if run.consent and run.status == "done" and not run.kept:
                _move_to_kept(run)       # finish an interrupted move; revisit next sweep
                continue
            if run.created >= cutoff:
                continue
            _delete(p.name)
            n += 1
    if KEPT_DIR.is_dir():
        for p in KEPT_DIR.iterdir():
            if not p.is_dir() or not RUN_ID_RE.match(p.name):
                continue
            run = get(p.name)
            if run is None:
                continue
            if run.created >= consent_cutoff:
                continue
            _delete(p.name)
            n += 1
    return n


def _sweeper_loop() -> None:
    while True:
        try:
            n = sweep()
            if n:
                log.info("try-mode sweeper: deleted %d expired run(s)", n)
        except Exception:
            log.exception("try-mode sweeper failed")
        time.sleep(SWEEP_PERIOD_S)


def start_background() -> None:
    """Idempotent: called from the API's lifespan on every start."""
    if _WORKER_STARTED.is_set():
        return
    _WORKER_STARTED.set()
    TRY_DIR.mkdir(parents=True, exist_ok=True)
    KEPT_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        sweep()
    with _ACCT_LOCK:                         # measured once; then kept as a running total
        _DISK["used"] = _du(TRY_DIR) + _du(KEPT_DIR)
    threading.Thread(target=_worker_loop, name="try-mode-worker", daemon=True).start()
    threading.Thread(target=_sweeper_loop, name="try-mode-sweeper", daemon=True).start()
