"""Recordings a machine can be commissioned from and replayed.

One interface over two layouts, so the commissioning pipeline and the replay
source never know which one they are reading:

  * `ImsRecording` — an IMS bearing test as NASA distributes it: one ASCII file
    per snapshot, named by the time it was taken.
  * `FolderRecording` — the plain format anything else arrives in, whether
    uploaded through the console or copied onto the server:

        <folder>/recording.json   {"name", "sample_rate_hz", "interval_s", "unit",
                                   "rpm": optional, the shaft speed per snapshot}
        <folder>/*.csv|*.txt      one snapshot per file, a column per channel
        <folder>/*.npy            or one array: (snapshots, samples[, channels])
                                  or one 1-D array per snapshot, in name order —
                                  what a live listener writes (api/live.py),
                                  with "t_s" in recording.json: epoch seconds
                                  per snapshot

    A snapshot's time comes from its file name when the name is a timestamp
    (2004.02.12.10.32.39, 2024-05-01T10-00-00, 20240501_100000 ...), and from
    its position times `interval_s` otherwise.

Everything is read as float32 and cached per channel beside the data, so a
week of recording is parsed once.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
RECORDINGS_DIR = DATASETS / "recordings"

# The engine's FFT wants a power of two. A snapshot is cut to the largest one
# it holds, up to the engine's reference window. MIN_WINDOW is the shortest
# window with evidence behind it: FEMTO's 2560-sample snapshots, windowed at
# 2048 (12.5 Hz bins at 25.6 kHz), catch 16 of its 17 runs to failure held
# out, and the 17th once a stop in its baseline no longer sets the noise
# floor (docs/DETECTION_MISSES.md, "Held-out data"). Short windows cost diagnosis,
# not detection: below 16384 close lines such as 3X and BPFO on a 2950 rpm
# pump share a bin. Below MIN_WINDOW nothing has been evaluated, so the data
# is refused.
MAX_WINDOW = 16384
MIN_WINDOW = 2048

_STAMP_FORMATS = ("%Y.%m.%d.%H.%M.%S", "%Y-%m-%dT%H-%M-%S", "%Y-%m-%d_%H-%M-%S",
                  "%Y%m%d_%H%M%S", "%Y%m%dT%H%M%S", "%Y-%m-%d %H-%M-%S")


def window_for(n_samples: int) -> int:
    """The engine window a snapshot of `n_samples` supports."""
    if n_samples < MIN_WINDOW:
        raise ValueError(f"a snapshot of {n_samples} samples is too short: the engine "
                         f"needs at least {MIN_WINDOW} to resolve bearing frequencies")
    return min(MAX_WINDOW, 1 << (int(n_samples).bit_length() - 1))


# Caps on one text snapshot file, checked while it is read (HARM-S10-T03 H1):
# a small file of very short lines must not become a huge list of strings.
# Generous for the console; try mode passes its own, tighter ones.
TEXT_MAX_ROWS = 1 << 21
TEXT_MAX_COLS = 64
TEXT_MAX_LINE_BYTES = 64 * 1024

_HEADER_RE = re.compile(rb"^\s*[-+]?(\d|\.\d)")
_CELL_RE = re.compile(rb"[^,\s;]+")


def text_rows(path: Path, *, max_rows: int, max_cols: int, max_line_bytes: int,
              info: dict | None = None):
    """Stream the data rows of a CSV/TXT file as lists of cell bytes.

    One line is held at a time and no line may pass `max_line_bytes`; a row
    of more than `max_cols` cells or a file of more than `max_rows` data rows
    is refused (ValueError) as soon as it is seen, before any array is built.
    Blank lines are skipped; the first non-blank line is a header when it does
    not start with a number (`info["header"]` says whether one was seen).
    Error messages never quote the file's content."""
    header_checked = False
    n = 0
    with open(path, "rb") as f:
        while True:
            raw = f.readline(max_line_bytes + 1)
            if not raw:
                return
            if len(raw) > max_line_bytes:
                raise ValueError(f"a line is longer than {max_line_bytes} bytes")
            if not header_checked:
                raw = raw.removeprefix(b"\xef\xbb\xbf")
            if not raw.strip():
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("the file is not UTF-8 text") from None
            if not header_checked:
                header_checked = True
                if info is not None:
                    info["header"] = True
                if not _HEADER_RE.match(raw):
                    continue
            n += 1
            if n > max_rows:
                raise ValueError(f"more than {max_rows} rows in one file")
            cells: list[bytes] = []
            for m in _CELL_RE.finditer(raw):
                if len(cells) >= max_cols:
                    raise ValueError(f"row {n}: more than {max_cols} values exceeds the limit")
                cells.append(m.group())
            yield n, cells


def floats(cells: list[bytes], row: int) -> np.ndarray:
    """Cells as float32, or a ValueError that does not quote the cell."""
    try:
        return np.array(cells, dtype="S").astype(np.float32)
    except ValueError:
        raise ValueError(f"row {row}: non-numeric value") from None


def parse_stamp(name: str) -> datetime | None:
    stem = name.rsplit(".", 1)[0] if re.search(r"\.[A-Za-z]+$", name) else name
    for candidate in (name, stem):
        for fmt in _STAMP_FORMATS:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


class Recording(Protocol):
    key: str
    name: str
    sample_rate_hz: int
    n_channels: int
    unit: str
    provenance: str

    def hours(self) -> np.ndarray: ...
    def stamps(self) -> list[str | None]: ...
    def snapshot_samples(self) -> int: ...
    def load(self, channel: int, n_samples: int) -> np.ndarray: ...
    def speeds(self) -> np.ndarray | None: ...


_NUMERIC_KINDS = "fiu"   # float, signed int, unsigned int — never object ("O")


_MmapMode = Literal["r+", "r", "w+", "c"]


def _load_npy(path: Path, mmap_mode: _MmapMode | None = "r") -> np.ndarray:
    """A snapshot `.npy`, read defensively: no pickled objects — the default
    since NumPy 1.16.3, made explicit here rather than relied on — and a
    numeric dtype only. An upload has no review before the engine sees it,
    so this is the one place that gets to assume the bytes are honest."""
    arr = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if arr.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{path.name}: dtype {arr.dtype} is not numeric")
    return arr


def _cached(cache: Path, build) -> np.ndarray:
    if cache.exists():
        try:
            return _load_npy(cache, mmap_mode=None)
        except (OSError, ValueError):
            cache.unlink(missing_ok=True)
    arr = build()
    with contextlib.suppress(OSError):   # a read-only folder works, only slower
        np.save(cache, arr)
    return arr


# --- IMS ------------------------------------------------------------------------


def ims_bundle_dir(test: int = 2) -> Path:
    """The exported IMS bundle: the data folder's `ims/test<n>/`, holding
    stamps.json, ch<k>_<samples>.npy per exported channel, and the rig's brain."""
    import os

    data = Path(os.environ.get("HARMONIC_DATA_DIR", REPO / "var"))
    return data / "ims" / f"test{test}"


@dataclass
class ImsRecording:
    test: int = 2
    key: str = "ims:test2"
    name: str = "IMS bearing test 2 (NASA Prognostics Data Repository)"
    sample_rate_hz: int = 20480       # ims_loader.SAMPLE_RATE_HZ: 20.48 kHz, not the readme's 20
    unit: str = "g"
    provenance: str = ("IMS / University of Cincinnati bearing rig, NASA Prognostics "
                       "Data Repository — a real run to failure")
    n_channels: int = field(init=False)

    def __post_init__(self) -> None:
        from sim.datasets import ims_loader

        self._loader = ims_loader
        self._bundle: Path | None = None
        if ims_loader.test_dir(self.test, ims_loader.DEFAULT_ROOT).is_dir():
            self.n_channels = ims_loader.TESTS[self.test][1]
            self._taken = [s.taken for s in ims_loader.list_snapshots(self.test)]
            return
        # Where the raw NASA files are not — on Google Cloud — the exported
        # bundle (scripts/export_ims.py) in the data folder stands in.
        bundle = ims_bundle_dir(self.test)
        meta_path = bundle / "stamps.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"IMS test {self.test}: no raw data and no bundle")
        self._bundle = bundle
        self._taken = [datetime.fromisoformat(t) for t in json.loads(meta_path.read_text())]
        self._channels = sorted(int(p.stem.split("_")[0][2:]) for p in bundle.glob("ch*_*.npy"))
        self.n_channels = max(self._channels) + 1 if self._channels else 0

    def hours(self) -> np.ndarray:
        t0 = self._taken[0]
        return np.array([(t - t0).total_seconds() / 3600.0 for t in self._taken])

    def stamps(self) -> list[str | None]:
        return [t.strftime("%Y-%m-%d %H:%M") for t in self._taken]

    def snapshot_samples(self) -> int:
        if self._bundle is not None:
            return int(np.load(next(self._bundle.glob("ch*_*.npy")), mmap_mode="r").shape[1])
        return self._loader.SNAPSHOT_SAMPLES

    def load(self, channel: int, n_samples: int) -> np.ndarray:
        if self._bundle is not None:
            files = sorted(self._bundle.glob(f"ch{channel}_*.npy"))
            if not files:
                raise ValueError(f"channel {channel} is not in the exported bundle "
                                 f"(it has {self._channels})")
            arr = np.load(files[0], mmap_mode="r")
            if arr.shape[1] < n_samples:
                raise ValueError(f"the bundle holds {arr.shape[1]} samples per snapshot")
            return np.ascontiguousarray(arr[:, :n_samples], dtype=np.float32)
        return self._loader.load_run(self.test, channel, n_samples)[1]

    def speeds(self) -> np.ndarray | None:
        return None          # no tachometer; the readme gives a constant 2000 rpm


# --- the plain format ---------------------------------------------------------------


@dataclass
class FolderRecording:
    folder: Path
    key: str = ""
    name: str = ""
    sample_rate_hz: int = 0
    unit: str = "g"
    interval_s: float = 600.0
    provenance: str = "uploaded"
    # Caps on each text snapshot file (`text_rows`); try mode passes tighter ones.
    max_text_rows: int = TEXT_MAX_ROWS
    max_text_cols: int = TEXT_MAX_COLS
    # Rows of each text snapshot actually read; None (the console) reads all.
    text_rows_used: int | None = None
    n_channels: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        meta_path = self.folder / "recording.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.name = self.name or meta.get("name") or self.folder.name
        self.sample_rate_hz = int(self.sample_rate_hz or meta.get("sample_rate_hz", 0))
        if self.sample_rate_hz <= 0:
            raise ValueError(f"{self.folder}: recording.json must give sample_rate_hz")
        self.unit = meta.get("unit", self.unit)
        self.interval_s = float(meta.get("interval_s", self.interval_s))
        self.provenance = meta.get("provenance", self.provenance)
        rpm = meta.get("rpm")
        self._rpm = None if rpm is None else np.asarray(rpm, dtype=np.float64)
        self.key = self.key or f"folder:{self.folder.name}"

        self._npy = sorted(self.folder.glob("*.npy"))
        self._npy = [p for p in self._npy if not p.name.startswith("_cache")]
        self._per_snapshot = False
        self._files = sorted(p for p in self.folder.iterdir()
                             if p.suffix.lower() in (".csv", ".txt"))
        if not self._npy and not self._files:
            raise ValueError(f"{self.folder}: no .csv, .txt or .npy snapshots")
        if self._npy and _load_npy(self._npy[0]).ndim == 1:
            # One snapshot per file, as a live listener writes them. The folder
            # may be growing: the metadata lists can run one ahead of the
            # files (they are written first), never behind.
            self._per_snapshot = True
            self._count = len(self._npy)
            self._samples = int(_load_npy(self._npy[0]).shape[0])
            self.n_channels = 1
            t_s = meta.get("t_s")
            self._times = None
            if t_s is not None and len(t_s) >= self._count:
                self._times = [datetime.fromtimestamp(float(t), tz=UTC)
                               for t in t_s[: self._count]]
            if self._rpm is not None and len(self._rpm) >= self._count:
                self._rpm = self._rpm[: self._count]
        elif self._npy:
            arr = _load_npy(self._npy[0])
            if arr.ndim not in (2, 3):
                raise ValueError(f"{self._npy[0].name}: expected (snapshots, samples"
                                 f"[, channels]), got shape {arr.shape}")
            self._count = arr.shape[0]
            self._samples = arr.shape[1]
            self.n_channels = 1 if arr.ndim == 2 else arr.shape[2]
            # "t_s" in recording.json, epoch seconds per snapshot: the same
            # convention a live listener uses, honoured here too so a wide
            # single-array upload (api/adapters/formats.py) can carry a
            # timestamp column.
            t_s = meta.get("t_s")
            self._times = ([datetime.fromtimestamp(float(t), tz=UTC)
                            for t in t_s[: self._count]]
                           if t_s is not None and len(t_s) >= self._count else None)
        else:
            first = self._read_text(self._files[0])
            self._count = len(self._files)
            self._samples = first.shape[0]
            self.n_channels = first.shape[1]
            stamps = [parse_stamp(p.name) for p in self._files]
            t_s = meta.get("t_s")
            if t_s is not None and len(t_s) >= self._count:
                # Times given in recording.json, files already in time order:
                # how try mode keeps an upload's own names out of its paths.
                self._times = [datetime.fromtimestamp(float(t), tz=UTC)
                               for t in t_s[: self._count]]
            elif all(stamps):
                order = np.argsort([s.timestamp() for s in stamps])  # type: ignore[union-attr]
                self._files = [self._files[i] for i in order]
                self._times = [stamps[i] for i in order]
            else:
                self._times = None
        window_for(self._samples)        # refuse data the engine cannot use
        if self._rpm is not None and (self._rpm.shape != (self._count,)
                                      or not np.all(self._rpm > 0)):
            raise ValueError(f"{self.folder}: recording.json 'rpm' must give one positive "
                             f"speed per snapshot ({self._count}), in time order")

    def _read_text(self, path: Path) -> np.ndarray:
        """One text snapshot, streamed (`text_rows`): every row must have the
        same number of columns, and memory stays at the float32 result. Cells
        are converted in batches, not row by row. With `text_rows_used` set,
        reading stops after that many rows: the rest of the file is never
        parsed (try mode passes the engine's largest window)."""
        stop = self.text_rows_used
        parts: list[np.ndarray] = []
        batch: list[bytes] = []
        first_row = 1
        n_cols = 0
        n = 0

        def flush() -> None:
            try:
                parts.append(np.array(batch, dtype="S").astype(np.float32).reshape(-1, n_cols))
            except ValueError:
                for k in range(len(batch) // n_cols):   # name the row, never the cell
                    floats(batch[k * n_cols:(k + 1) * n_cols], first_row + k)
                raise
            batch.clear()

        for row, cells in text_rows(path, max_rows=self.max_text_rows,
                                    max_cols=self.max_text_cols,
                                    max_line_bytes=TEXT_MAX_LINE_BYTES):
            if n == 0:
                n_cols = len(cells)
                if n_cols == 0:
                    raise ValueError(f"{path.name}: row {row} is empty")
                first_row = row
            elif len(cells) != n_cols:
                raise ValueError(f"{path.name}: row {row} has {len(cells)} values; "
                                 f"the first row has {n_cols}")
            batch.extend(cells)
            n += 1
            if len(batch) >= 65536:
                flush()
                first_row = row + 1
            if stop is not None and n >= stop:
                break
        if n == 0:
            raise ValueError(f"{path.name}: no data rows")
        if batch:
            flush()
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def hours(self) -> np.ndarray:
        if self._times:
            t0 = self._times[0]
            return np.array([(t - t0).total_seconds() / 3600.0 for t in self._times])
        return np.arange(self._count) * self.interval_s / 3600.0

    def stamps(self) -> list[str | None]:
        if self._times:
            return [t.strftime("%Y-%m-%d %H:%M") for t in self._times]
        return [None] * self._count

    def snapshot_samples(self) -> int:
        return self._samples

    def speeds(self) -> np.ndarray | None:
        """The measured shaft speed per snapshot, in time order, or None."""
        return self._rpm

    def load(self, channel: int, n_samples: int) -> np.ndarray:
        if not 0 <= channel < self.n_channels:
            raise ValueError(f"channel {channel} out of range; the recording has "
                             f"{self.n_channels}")

        if self._per_snapshot:
            # Not cached: a live folder grows, and each file is one read.
            return np.stack([_load_npy(p, mmap_mode=None)[:n_samples].astype(np.float32)
                             for p in self._npy])

        def build() -> np.ndarray:
            if self._npy:
                arr = _load_npy(self._npy[0])
                sl = arr[:, :n_samples] if arr.ndim == 2 else arr[:, :n_samples, channel]
                return np.ascontiguousarray(sl, dtype=np.float32)
            out = np.empty((len(self._files), n_samples), dtype=np.float32)
            for i, p in enumerate(self._files):
                a = self._read_text(p)
                if a.shape[0] < n_samples or a.shape[1] <= channel:
                    raise ValueError(f"{p.name} holds {a.shape[0]} samples x {a.shape[1]} "
                                     f"column(s); every snapshot needs {n_samples} samples "
                                     f"and channel {channel + 1}, like the first file")
                out[i] = a[:n_samples, channel]
            return out

        return _cached(self.folder / f"_cache_ch{channel}_{n_samples}.npy", build)


# --- the catalogue --------------------------------------------------------------------


def available(extra_folders: list[Path] | None = None) -> dict[str, Recording]:
    """Every recording on the server, by key. Unreadable ones are skipped."""
    out: dict[str, Recording] = {}
    from sim.datasets import ims_loader

    for test in ims_loader.TESTS:
        try:
            ims = ImsRecording(test=test, key=f"ims:test{test}",
                               name=f"IMS bearing test {test} (NASA Prognostics Data Repository)")
            out[ims.key] = ims
        except (FileNotFoundError, KeyError, ValueError):
            pass
    folders = [p for p in RECORDINGS_DIR.iterdir() if p.is_dir()] if RECORDINGS_DIR.is_dir() else []
    for folder in [*folders, *(extra_folders or [])]:
        try:
            rec = FolderRecording(folder)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        out[rec.key] = rec
    return out


def rms(data: np.ndarray) -> np.ndarray:
    """Overall RMS of each snapshot, mean removed — what the baseline picker
    draws so a user can see where the machine was steady."""
    d = data - data.mean(axis=1, keepdims=True)
    return np.sqrt((d.astype(np.float64) ** 2).mean(axis=1))
