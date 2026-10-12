"""A machine's own manual, uploaded from the console (HARM-S7-T11).

Until now a commissioned machine was diagnosed with its family's reference
manual: a user's gearbox cited GB-201's. An uploaded manual — Markdown or plain
text — is cut into citable sections (`rag.ingest.parse_generic`), added to the
retrieval index, and named in the machine's graph node, so retrieval for that
machine searches its own manual first and no other machine ever sees it.

Kept under the data folder (a Cloud Storage bucket on Google Cloud):

    <data>/manuals/<doc id>.md     the text as uploaded
    <data>/manuals/<doc id>.json   who it belongs to, its sections
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from api import machines
from rag.ingest import Chunk, parse_generic

MAX_BYTES = 2 * 1024 * 1024
SUFFIXES = (".md", ".markdown", ".txt")


def _dir() -> Any:
    return machines.DATA_DIR / "manuals"


def doc_id_for(m: machines.Machine) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "-", f"{m.equipment_id}-OM").upper()


def save(machine_id: str, filename: str, body: bytes) -> dict[str, Any]:
    m = machines.get(machine_id)
    if m is None:
        raise machines.CommissionError(f"no machine {machine_id!r}")
    if not filename.lower().endswith(SUFFIXES):
        raise machines.CommissionError("a manual is uploaded as Markdown or plain text "
                                       f"({', '.join(SUFFIXES)})")
    if len(body) > MAX_BYTES:
        raise machines.CommissionError(f"a manual is at most {MAX_BYTES // 1024} KB of text")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise machines.CommissionError("the manual is not UTF-8 text") from None
    doc_id = doc_id_for(m)
    chunks = parse_generic(text, doc_id)
    if not chunks:
        raise machines.CommissionError("the manual has no text to cite")
    folder = _dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{doc_id}.md").write_text(text, encoding="utf-8")
    info = {"doc_id": doc_id, "machine": m.id, "filename": filename,
            "sections": len(chunks), "uploaded": time.time()}
    (folder / f"{doc_id}.json").write_text(json.dumps(info))
    return {**info, "chunks": chunks}


def info(machine_id: str) -> dict[str, Any] | None:
    m = machines.get(machine_id)
    if m is None:
        return None
    path = _dir() / f"{doc_id_for(m)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_all() -> list[Chunk]:
    """Every uploaded manual's sections, for the index at start-up."""
    out: list[Chunk] = []
    folder = _dir()
    if not folder.is_dir():
        return out
    for meta in folder.glob("*.json"):
        doc = meta.stem
        md = folder / f"{doc}.md"
        if md.exists():
            out += parse_generic(md.read_text(encoding="utf-8"), doc)
    return out
