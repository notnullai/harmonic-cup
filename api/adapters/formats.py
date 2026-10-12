"""Recording-format adapters: raw uploaded bytes -> the plain folder layout
`sim.datasets.recordings.FolderRecording` already reads (HARM-S8-T-try).

Today's console accepts that layout directly — one `.csv`/`.txt`/`.npy` per
snapshot, or one `.npy` array — and needs no adapter at all (`ADAPT_AUTO`
below is a no-op). Each new format a visitor's own logger already produces
gets one function here and one entry in `ADAPTERS`, so the commissioning and
replay pipeline never has to know a format changed. Today: `auto` (the
console's layout), `wide_csv` (one CSV, a snapshot per row) and `wav` (PCM or
float WAV, one snapshot per file or one long file cut into snapshots).

Every refusal — an extension the upload does not take, or an adapter that
could not read the file — is logged by `log_rejected` as one JSON line
(`try-rejected-format`, no file content, no file name), so the log, not
guesswork, picks the next adapter to build (HARM-S10-T04).

Everything here runs on **untrusted** bytes: a visitor's own upload, with no
sign-up and no review before the engine sees it. Every adapter is expected to
raise `AdapterError` (never let a parser exception escape) and to enforce the
same caps `api/try_mode.py` already checked before the bytes were saved to
disk — belt and braces, since the wizard may add adapters this module's
author never sees.
"""

from __future__ import annotations

import contextlib
import json
import logging
import mmap
import re
import shutil
import struct
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("harmonic.try.formats")

_NUMERIC_START = re.compile(r"^\s*[-+]?(\d|\.\d)")
_SPLIT = re.compile(r"[,\s;]+")


class AdapterError(Exception):
    """A reason the upload could not be read, in words for the visitor."""


def _adapt_auto(folder: Path, meta: dict[str, Any], *, max_snapshots: int,
                max_samples: int, max_total_samples: int | None = None) -> None:
    """The format the console already accepts: nothing to rewrite. Just
    refuses an upload with nothing in it before `FolderRecording` has to say
    so less clearly."""
    names = [p.name for p in folder.iterdir()
             if p.is_file() and p.name != "recording.json"]
    if not names:
        raise AdapterError("no snapshot files were uploaded")
    csvs = [p for p in folder.iterdir() if p.suffix.lower() in (".csv", ".txt")]
    if len(csvs) > max_snapshots:
        raise AdapterError(f"{len(csvs)} files were uploaded; at most "
                           f"{max_snapshots} snapshots are accepted")
    (folder / "recording.json").write_text(json.dumps(meta))


# The longest line a wide-CSV row may be: every sample a full-precision float
# with its separator, plus a timestamp column.
_WIDE_BYTES_PER_CELL = 32


def _adapt_wide_csv(folder: Path, meta: dict[str, Any], *, max_snapshots: int,
                    max_samples: int, max_total_samples: int | None = None) -> None:
    """One CSV or TXT file, one row per snapshot, an optional leading
    timestamp column (`meta["has_timestamp"]`, set by the wizard — asking
    rather than guessing keeps an untrusted file from being misread). This is
    the shape most condition-monitoring loggers already export a day's
    recording in.

    Rewritten into the plain layout so nothing downstream needs to know:
    one `data.npy` of shape (snapshots, samples) plus `recording.json`, with
    `t_s` (epoch seconds per snapshot) added when a timestamp column was
    given (`sim/datasets/recordings.py` reads both).

    Streamed (HARM-S10-T03 H1): one row is held at a time, each row's bytes
    and cells and the rows and samples in total are capped as they are read,
    and the rows go straight to disk — memory never grows with the file.
    """
    from sim.datasets.recordings import floats, text_rows

    csvs = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".csv", ".txt"))
    if len(csvs) != 1:
        raise AdapterError("the single-file format takes exactly one .csv or .txt file")
    path = csvs[0]
    has_ts = bool(meta.get("has_timestamp"))
    total_cap = max_total_samples or max_snapshots * max_samples
    stamps: list[float] = []
    n_samples: int | None = None
    n_rows = 0
    raw_path = folder / "_rows.tmp"
    seen: dict[str, bool] = {}
    ok = False
    try:
        with open(raw_path, "wb") as raw:
            for i, cells in text_rows(path, max_rows=max_snapshots + 1,
                                      max_cols=max_samples + int(has_ts),
                                      max_line_bytes=(max_samples + 1) * _WIDE_BYTES_PER_CELL,
                                      info=seen):
                if i > max_snapshots:
                    raise AdapterError(f"the file holds more than {max_snapshots} rows; at "
                                       f"most {max_snapshots} snapshots are accepted")
                if has_ts:
                    if len(cells) < 2:
                        raise AdapterError(f"row {i}: no samples after the timestamp column")
                    stamps.append(_parse_ts(cells[0], i))
                    cells = cells[1:]
                if n_samples is None:
                    n_samples = len(cells)
                    if n_samples <= 0:
                        raise AdapterError(f"row {i} is empty")
                    if n_samples > max_samples:
                        raise AdapterError(f"{n_samples} samples in a row exceeds the "
                                           f"limit of {max_samples}")
                elif len(cells) != n_samples:
                    raise AdapterError(f"row {i} has {len(cells)} samples; row 1 has "
                                       f"{n_samples} — every snapshot must be the same length")
                if (n_rows + 1) * n_samples > total_cap:
                    raise AdapterError(f"the file holds more than {total_cap} samples in "
                                       "all; send fewer or shorter snapshots")
                row = floats(cells, i)
                if not np.isfinite(row).all():
                    raise AdapterError("the file contains a missing, infinite or "
                                       "non-numeric value")
                raw.write(row.tobytes())
                n_rows += 1
        if n_rows == 0 or n_samples is None:
            raise AdapterError("no data rows found" if seen else "the file is empty")
        with open(folder / "data.npy", "wb") as out, open(raw_path, "rb") as src:
            np.lib.format.write_array_header_1_0(out, {
                "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                "fortran_order": False, "shape": (n_rows, n_samples)})
            shutil.copyfileobj(src, out, 1 << 20)
        ok = True
    except ValueError as exc:                  # text_rows / floats: content-free messages
        raise AdapterError(str(exc)) from None
    except OSError:
        raise AdapterError("the file could not be read") from None
    finally:
        raw_path.unlink(missing_ok=True)
        if not ok:
            (folder / "data.npy").unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    out_meta = dict(meta)
    out_meta.pop("has_timestamp", None)
    out_meta.pop("t_s", None)
    if has_ts:
        out_meta["t_s"] = stamps
    (folder / "recording.json").write_text(json.dumps(out_meta))


def _parse_ts(raw: bytes, row: int) -> float:
    """Epoch seconds, from a plain number or an ISO-ish timestamp. The
    refusal never quotes the cell: it is the visitor's content."""
    cell = raw.decode("utf-8", errors="replace")
    try:
        return float(cell)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(cell.replace("Z", "+00:00")).replace(
            tzinfo=UTC).timestamp()
    except ValueError:
        raise AdapterError(f"row {row}: the timestamp is neither a number "
                           "(epoch seconds) nor ISO 8601") from None


# --- WAV ---------------------------------------------------------------------------------

# The engine's own window bounds (sim/datasets/recordings.py MIN_WINDOW,
# MAX_WINDOW), repeated rather than imported so this module stays a leaf.
_WAV_MIN_SAMPLES = 2048
_WAV_SNAPSHOT_SAMPLES = 16384
_WAV_MAX_CHANNELS = 8
_WAV_MAX_CHUNKS = 256
_WAVE_FORMAT_PCM, _WAVE_FORMAT_FLOAT, _WAVE_FORMAT_EXTENSIBLE = 1, 3, 0xFFFE
# KSDATAFORMAT_SUBTYPE_* GUIDs share everything after their first two bytes.
_KS_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


class _Wav:
    """A parsed, validated WAV header and a view of its sample bytes."""

    def __init__(self, fmt: int, channels: int, rate: int, bits: int, data: memoryview):
        self.fmt, self.channels, self.rate, self.bits = fmt, channels, rate, bits
        self.block = channels * bits // 8
        self.data = data
        self.frames = len(data) // self.block

    def frames_at(self, start: int, count: int) -> np.ndarray:
        """`count` frames from `start` as float32 (count, channels); integer
        PCM scaled to full scale = 1.0. Only these frames are decoded."""
        raw = self.data[start * self.block:(start + count) * self.block]
        if self.fmt == _WAVE_FORMAT_FLOAT:
            a = np.frombuffer(raw, dtype="<f4" if self.bits == 32 else "<f8").astype(np.float64)
        elif self.bits == 24:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            a = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64) / float(1 << 23)
        else:
            dt = {16: "<i2", 32: "<i4"}[self.bits]
            a = np.frombuffer(raw, dtype=dt).astype(np.float64) / float(1 << (self.bits - 1))
        if not np.isfinite(a).all():
            raise AdapterError("the WAV file contains a missing or infinite sample")
        return a.astype(np.float32).reshape(count, self.channels)


def _parse_fmt(b: bytes, name: str) -> tuple[int, int, int, int]:
    if len(b) < 16:
        raise AdapterError(f"{name}: the format chunk is too short")
    code, channels, rate, _byte_rate, block, bits = struct.unpack_from("<HHIIHH", b, 0)
    if code == _WAVE_FORMAT_EXTENSIBLE:
        if len(b) < 40 or struct.unpack_from("<H", b, 16)[0] < 22:
            raise AdapterError(f"{name}: the extensible format chunk is too short")
        valid_bits = struct.unpack_from("<H", b, 18)[0]
        sub = b[24:40]
        if sub[2:] != _KS_GUID_TAIL:
            raise AdapterError(f"{name}: unsupported WAV sub-format")
        code = struct.unpack_from("<H", sub, 0)[0]
        if valid_bits not in (0, bits):
            raise AdapterError(f"{name}: {valid_bits}-bit samples in {bits}-bit containers "
                               "are not supported")
    if code not in (_WAVE_FORMAT_PCM, _WAVE_FORMAT_FLOAT):
        raise AdapterError(f"{name}: WAV encoding {code} is not supported "
                           "(PCM or IEEE float only; no compressed audio)")
    if code == _WAVE_FORMAT_PCM and bits not in (16, 24, 32):
        raise AdapterError(f"{name}: {bits}-bit PCM is not supported (16, 24 or 32 bit)")
    if code == _WAVE_FORMAT_FLOAT and bits not in (32, 64):
        raise AdapterError(f"{name}: {bits}-bit float is not supported (32 or 64 bit)")
    if not 1 <= channels <= _WAV_MAX_CHANNELS:
        raise AdapterError(f"{name}: {channels} channels; 1 to {_WAV_MAX_CHANNELS} accepted")
    if block != channels * bits // 8:
        raise AdapterError(f"{name}: the frame size does not match channels and bit depth")
    if rate <= 0:
        raise AdapterError(f"{name}: sampling rate {rate} Hz is not valid")
    return code, channels, rate, bits


def _parse_wav(buf: bytes | mmap.mmap, name: str) -> _Wav:
    """Strict RIFF/WAVE parse: every length is bounds-checked before it is
    used, exactly one `fmt ` before exactly one `data`, and only the sample
    layouts `_Wav.frames_at` decodes — PCM 16/24/32-bit and IEEE float
    32/64-bit, plain or WAVE_FORMAT_EXTENSIBLE. Anything else is refused."""
    if len(buf) < 12 or buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise AdapterError(f"{name}: not a RIFF/WAVE file")
    end = struct.unpack_from("<I", buf, 4)[0] + 8
    if end > len(buf):
        raise AdapterError(f"{name}: the file is truncated (header says {end} bytes, "
                           f"file has {len(buf)})")
    pos, n_chunks = 12, 0
    fmt: tuple[int, int, int, int] | None = None
    data: memoryview | None = None
    while pos + 8 <= end:
        n_chunks += 1
        if n_chunks > _WAV_MAX_CHUNKS:
            raise AdapterError(f"{name}: too many RIFF chunks")
        cid, size = struct.unpack_from("<4sI", buf, pos)
        body, body_end = pos + 8, pos + 8 + size
        if body_end > end:
            raise AdapterError(f"{name}: a chunk runs past the end of the file")
        if cid == b"fmt ":
            if size > 1024:
                raise AdapterError(f"{name}: the format chunk is too long")
            if fmt is not None:
                raise AdapterError(f"{name}: more than one format chunk")
            fmt = _parse_fmt(buf[body:body_end], name)
        elif cid == b"data":
            if fmt is None:
                raise AdapterError(f"{name}: the data chunk comes before the format chunk")
            if data is not None:
                raise AdapterError(f"{name}: more than one data chunk")
            data = memoryview(buf)[body:body_end]
        pos = body_end + (size & 1)
    if fmt is None or data is None:
        raise AdapterError(f"{name}: no format chunk or no data chunk")
    wav = _Wav(*fmt, data)
    if len(data) % wav.block:
        raise AdapterError(f"{name}: the data chunk is not a whole number of frames")
    if wav.frames < _WAV_MIN_SAMPLES:
        raise AdapterError(f"{name}: {wav.frames} samples per channel; the engine needs "
                           f"at least {_WAV_MIN_SAMPLES}")
    return wav


def _adapt_wav(folder: Path, meta: dict[str, Any], *, max_snapshots: int,
               max_samples: int, max_total_samples: int | None = None) -> None:
    """WAV, as a data-acquisition front end or an acoustic logger writes it.

    Several files: one snapshot per file, in file-name order (a time stamp in
    the name is read the way the console reads CSV snapshot names), each cut
    to the shortest file and to `max_samples`. One file: a continuous
    recording, cut into snapshots of the engine's largest window, evenly
    spaced across the file when it holds more than `max_snapshots` of them.

    The sampling rate comes from the WAV header, never from the visitor's
    form. Integer PCM is scaled to full scale = 1.0 — not a calibrated unit,
    which the report says (api/try_mode.py `_caveats`). The output is the
    same plain layout as `wide_csv`: `data.npy` plus `recording.json`.

    Each file is memory-mapped, not read into memory, and the output is held
    to `max_total_samples` (snapshots x samples x channels): a long file is
    sampled more sparsely rather than decoded whole (HARM-S10-T03 H1/H3).
    """
    maps: list[mmap.mmap] = []
    try:
        _adapt_wav_mapped(folder, meta, maps, max_snapshots=max_snapshots,
                          max_samples=max_samples, max_total_samples=max_total_samples)
    finally:
        for m in maps:
            with contextlib.suppress(BufferError, ValueError):
                m.close()


def _adapt_wav_mapped(folder: Path, meta: dict[str, Any], maps: list[mmap.mmap], *,
                      max_snapshots: int, max_samples: int,
                      max_total_samples: int | None) -> None:
    files = [p for p in folder.iterdir() if p.is_file() and p.name != "recording.json"]
    wavs = sorted(p for p in files if p.suffix.lower() == ".wav")
    if not wavs:
        raise AdapterError("no .wav files were uploaded")
    if len(wavs) != len(files):
        raise AdapterError("a WAV upload takes only .wav files")
    if len(wavs) > max_snapshots:
        raise AdapterError(f"{len(wavs)} files were uploaded; at most {max_snapshots} "
                           "snapshots are accepted")
    parsed: list[_Wav] = []
    for p in wavs:
        try:
            with open(p, "rb") as f:
                if f.seek(0, 2) < 12:
                    raise AdapterError(f"{p.name}: not a RIFF/WAVE file")
                maps.append(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        except OSError:
            raise AdapterError(f"could not read {p.name}") from None
        parsed.append(_parse_wav(maps[-1], p.name))
    first = parsed[0]
    for p, w in zip(wavs, parsed, strict=True):
        if (w.rate, w.channels) != (first.rate, first.channels):
            raise AdapterError(f"{p.name}: {w.rate} Hz, {w.channels} channel(s); "
                               f"{wavs[0].name} is {first.rate} Hz, {first.channels} — "
                               "every file must match")

    out = dict(meta)
    out.pop("has_timestamp", None)
    out["sample_rate_hz"] = first.rate
    total_cap = max_total_samples or max_snapshots * max_samples * first.channels
    if len(parsed) > 1:
        n = min(min(w.frames for w in parsed), max_samples)
        if len(parsed) * n * first.channels > total_cap:
            raise AdapterError(f"{len(parsed)} files of {n} samples x {first.channels} "
                               f"channel(s) pass the limit of {total_cap} samples in all")
        arr = np.stack([w.frames_at(0, n) for w in parsed])
        from sim.datasets.recordings import parse_stamp

        stamps = [parse_stamp(p.name) for p in wavs]
        if all(s is not None for s in stamps):
            out["t_s"] = [s.timestamp() for s in stamps if s is not None]
    else:
        n = min(_WAV_SNAPSHOT_SAMPLES, first.frames, max_samples)
        count, stride = first.frames // n, n
        most = min(max_snapshots, max(total_cap // (n * first.channels), 1))
        if count > most:
            count, stride = most, first.frames // most
        if count < 2:
            raise AdapterError(f"{wavs[0].name} is too short: it makes {count} snapshot "
                               f"of {n} samples, and at least 2 are needed")
        arr = np.stack([first.frames_at(i * stride, n) for i in range(count)])
        out["interval_s"] = stride / first.rate
        out["wav_contiguous"] = True        # spaced by the file, not by the visitor
    if first.channels == 1:
        arr = arr[:, :, 0]
    del parsed, first
    np.save(folder / "data.npy", np.ascontiguousarray(arr))
    for p in wavs:
        p.unlink(missing_ok=True)
    (folder / "recording.json").write_text(json.dumps(out))


# --- the registry, and the log of what it refused -----------------------------------------

Adapter = Callable[..., None]

ADAPTERS: dict[str, Adapter] = {
    "auto": _adapt_auto,
    "wide_csv": _adapt_wide_csv,
    "wav": _adapt_wav,
}

# Extensions the upload takes at all; every other one is refused, and logged.
ACCEPTED_EXTENSIONS = (".csv", ".txt", ".npy", ".wav")


_BINARY_MAGICS = (b"RIFF", b"\x93NUMPY", b"PK", b"MATL", b"TDSm", b"\x89HDF", b"CDF")


def _magic(head: bytes) -> str:
    head = head[:8]
    if not head:
        return ""
    if head.startswith(_BINARY_MAGICS):
        return head.hex()
    for cut in range(4):                        # a multi-byte character cut at byte 8
        try:
            text = head[:len(head) - cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if all(c.isprintable() or c in "\t\r\n" for c in text):
            return "text"
        break
    return head.hex()


def log_rejected(stage: str, *, ext: str = "", head: bytes = b"", fmt: str = "",
                 reason: str = "", size: int | None = None) -> None:
    """One structured line per refused upload: the extension, the first 8
    bytes as hex (a file's magic number — `MATL`, `TDSm`, `PK` — says what it
    really is), the adapter asked for and why it failed. Never the file name
    or its content: both belong to the visitor. A head that is plain text is
    logged as "text", not as hex — the first bytes of a CSV are its first
    cells (HARM-S10-T03 L2)."""
    log.warning("try-rejected-format %s", json.dumps({
        "stage": stage, "ext": ext.lower()[:16], "magic": _magic(head),
        "format": fmt[:32], "size": size, "reason": reason[:200]}))


def adapt(name: str, folder: Path, meta: dict[str, Any], *, max_snapshots: int,
          max_samples: int, max_total_samples: int | None = None) -> None:
    fn = ADAPTERS.get(name)
    if fn is None:
        raise AdapterError(f"unknown upload format {str(name)[:32]!r}; known: "
                           f"{sorted(ADAPTERS)}")
    fn(folder, meta, max_snapshots=max_snapshots, max_samples=max_samples,
       max_total_samples=max_total_samples)
