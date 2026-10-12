"""Audit log for every diagnosis (HARM-S4-T10).

One JSON object per line: the event, the retrieval, the prompt, the raw
response, the guardrail verdict, and which provider actually answered. The
whole point of the guardrail is that an engineer can check the machine's
reasoning, and that is only true if the reasoning is still there afterwards.

Written append-only with the file opened per record, so a crash mid-pipeline
loses at most the record being written rather than the log. Rotation is by
size, oldest suffix discarded — a demo machine should not fill its disk with
prompts, and nothing here is a system of record.

The prompt is stored in full. It is the largest field by far, and it is the
one that answers "why did it say that": a response that looks wrong is usually
a context that was wrong.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path(os.environ.get("HARMONIC_AUDIT_LOG", "rag/audit/diagnoses.jsonl"))

# Rotate at 8 MB, keep 3 older files.
MAX_BYTES = 8 * 1024 * 1024
KEEP = 3


@dataclasses.dataclass
class AuditRecord:
    event_id: str
    ts: float
    providers: dict[str, str]
    event: dict[str, Any]
    kg: dict[str, Any]
    retrieval: list[dict[str, Any]]
    allowed_references: list[str]
    prompt: str
    attempts: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    final: dict[str, Any] | None = None
    outcome: str = "pending"
    duration_ms: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def new_event_id(event: dict[str, Any]) -> str:
    """Prefer the engine's own id so a log line joins back to an alarm."""
    for key in ("event_id", "id"):
        if event.get(key):
            return str(event[key])
    return f"ev-{uuid.uuid4().hex[:12]}"


def _rotate(path: Path) -> None:
    if not path.exists() or path.stat().st_size < MAX_BYTES:
        return
    for i in range(KEEP, 0, -1):
        older = path.with_suffix(path.suffix + f".{i}")
        if i == KEEP and older.exists():
            older.unlink()
        prev = path if i == 1 else path.with_suffix(path.suffix + f".{i - 1}")
        if prev.exists():
            prev.rename(older)


def write(record: AuditRecord, path: Path | str = DEFAULT_LOG) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate(path)
    line = json.dumps(record.to_json(), ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def read(path: Path | str = DEFAULT_LOG, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[-limit:] if limit else rows


def start(event: dict[str, Any], providers: dict[str, str]) -> AuditRecord:
    return AuditRecord(
        event_id=new_event_id(event),
        ts=time.time(),
        providers=providers,
        event=event,
        kg={},
        retrieval=[],
        allowed_references=[],
        prompt="",
    )
