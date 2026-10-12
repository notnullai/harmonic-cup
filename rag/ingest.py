"""Manual ingestion: Markdown sources to a chunk manifest (HARM-S4-T01).

One chunk per numbered section, because the section is the unit an engineer
cites and the unit the knowledge graph's `ManualSection` node stands for. Fixed
window chunking would split a procedure across two citations and make the
guardrail in `rag/guard.py` meaningless: a reference has to point at something
a person can look up.

The manifest is the join between three things that must agree — the RAG index,
the `ManualSection` nodes in the graph, and the citation allow-list. It carries
a content hash per chunk so a stale index can be detected rather than silently
retrieved from.

    python -m rag.ingest                      # rebuild rag/manuals/manifest.json
    python -m rag.ingest --check              # fail if the manifest is stale
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path

MANUAL_DIR = Path(__file__).resolve().parent / "manuals"
MANIFEST = MANUAL_DIR / "manifest.json"

# `### 4.2 Bearing replacement procedure`
SECTION_RE = re.compile(r"^###\s+(\d+(?:\.\d+)*)\s+(.+?)\s*$")
# `Doc id: `P-101-OM` · Revision C`
DOC_ID_RE = re.compile(r"^Doc id:\s*`([^`]+)`")
# Citations inside the prose: "section 4.2", "sections 2.1", "§4.2".
CROSSREF_RE = re.compile(r"(?:sections?|§)\s*(\d+\.\d+)", re.IGNORECASE)

# Words per notional page. The source is Markdown and has no pagination, so the
# page number is derived from position rather than measured. It is carried
# because docs/KG_SCHEMA.md gives ManualSection a `page` property and an
# engineer asks for a page; it is honest only as long as this is written down.
WORDS_PER_PAGE = 450


@dataclasses.dataclass(frozen=True)
class Chunk:
    """One citable section of one document."""

    id: str
    doc: str
    section: str
    title: str
    text: str
    page: int
    words: int
    sha256: str
    cross_refs: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        out = dataclasses.asdict(self)
        out["cross_refs"] = list(self.cross_refs)
        return out


def _chunk_id(doc_id: str, section: str) -> str:
    """Citation token. Must match GENAI_SCHEMA.json's `^[A-Za-z0-9._-]+$`."""
    return f"{doc_id}-{section}"


MAX_SECTION_WORDS = 250
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_generic(text: str, doc_id: str) -> list[Chunk]:
    """Split a manual that was not written for Harmonic (HARM-S7-T11).

    Sections are cut at Markdown headings; a text with none is cut into
    paragraph groups. Sections are numbered in order ("1", "2", ...) — those
    numbers are the citation tokens — and a long one is split into parts
    ("4.1", "4.2") so a citation points at something a person can read.
    """
    blocks: list[tuple[str, list[str]]] = []
    title, lines = "", list[str]()
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if "".join(lines).strip():
                blocks.append((title, lines))
            title, lines = m.group(2).strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        blocks.append((title, lines))
    if len(blocks) == 1 and not blocks[0][0]:
        # No headings: paragraphs, grouped up to the section size.
        paras = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(blocks[0][1])) if p.strip()]
        blocks = []
        group: list[str] = []
        count = 0
        for p in paras:
            if group and count + len(p.split()) > MAX_SECTION_WORDS:
                blocks.append(("", group))
                group, count = [], 0
            group.append(p)
            count += len(p.split())
        if group:
            blocks.append(("", group))
        blocks = [(t, ["\n\n".join(g)]) for t, g in blocks]

    chunks: list[Chunk] = []
    words_before = 0
    for n, (title, body) in enumerate(blocks, start=1):
        words = " ".join(body).split()
        parts = [words[i:i + MAX_SECTION_WORDS] for i in range(0, len(words), MAX_SECTION_WORDS)]
        body_text = "\n".join(body).strip()
        for k, part in enumerate(parts, start=1):
            section = str(n) if len(parts) == 1 else f"{n}.{k}"
            part_text = body_text if len(parts) == 1 else " ".join(part)
            chunks.append(Chunk(
                id=_chunk_id(doc_id, section), doc=doc_id, section=section,
                title=title or f"Section {section}", text=part_text,
                page=1 + words_before // WORDS_PER_PAGE, words=len(part),
                sha256=hashlib.sha256(part_text.encode("utf-8")).hexdigest()[:16],
                cross_refs=()))
            words_before += len(part)
    return chunks


def parse_manual(path: Path) -> list[Chunk]:
    """Split one Markdown manual into its numbered sections."""
    lines = path.read_text(encoding="utf-8").splitlines()

    doc_id = path.stem
    for line in lines:
        m = DOC_ID_RE.match(line.strip())
        if m:
            doc_id = m.group(1)
            break

    chunks: list[Chunk] = []
    section: str | None = None
    title = ""
    body: list[str] = []
    words_before = 0

    def flush() -> None:
        nonlocal words_before
        if section is None:
            return
        text = "\n".join(body).strip()
        if not text:
            return
        words = len(text.split())
        # Cross-references are resolved to chunk ids here rather than at query
        # time so the graph loader and the retriever see the same edges.
        refs = tuple(sorted({_chunk_id(doc_id, s) for s in CROSSREF_RE.findall(text)
                             if s != section}))
        chunks.append(Chunk(
            id=_chunk_id(doc_id, section),
            doc=doc_id,
            section=section,
            title=title,
            text=text,
            page=1 + words_before // WORDS_PER_PAGE,
            words=words,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            cross_refs=refs,
        ))
        words_before += words

    for line in lines:
        m = SECTION_RE.match(line)
        if m:
            flush()
            section, title = m.group(1), m.group(2)
            body = []
            continue
        if section is not None:
            # `## 5. Condition monitoring` ends the previous subsection without
            # opening a new citable one.
            if line.startswith("## ") or line.startswith("---"):
                flush()
                section = None
                body = []
                continue
            body.append(line)
    flush()
    return chunks


def build(manual_dir: Path = MANUAL_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(manual_dir.glob("*.md")):
        chunks.extend(parse_manual(path))
    if not chunks:
        raise SystemExit(f"ERROR: no sections found under {manual_dir}")

    seen: dict[str, str] = {}
    for c in chunks:
        if c.id in seen:
            raise SystemExit(f"ERROR: duplicate chunk id {c.id} "
                             f"({seen[c.id]} and {c.doc})")
        seen[c.id] = c.doc

    # A cross-reference to a section that does not exist would put an
    # unciteable id into the graph, and the guardrail would then reject a
    # citation the graph itself supplied.
    ids = set(seen)
    for c in chunks:
        for ref in c.cross_refs:
            if ref not in ids:
                raise SystemExit(f"ERROR: {c.id} refers to {ref}, which does not exist")
    return chunks


def manifest_payload(chunks: list[Chunk]) -> dict[str, object]:
    docs = sorted({c.doc for c in chunks})
    return {
        "version": 1,
        "docs": docs,
        "n_chunks": len(chunks),
        "chunks": [c.to_json() for c in chunks],
    }


def load_manifest(path: Path = MANIFEST) -> list[Chunk]:
    """Read the manifest back. The single entry point for every consumer."""
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found. Run: python -m rag.ingest")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**{**c, "cross_refs": tuple(c["cross_refs"])}) for c in raw["chunks"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk the manuals into a manifest")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the manifest is stale, and write nothing")
    ap.add_argument("--out", default=str(MANIFEST))
    args = ap.parse_args(argv)

    chunks = build()
    payload = manifest_payload(chunks)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out = Path(args.out)

    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"STALE: {out} does not match the manuals. Run: python -m rag.ingest",
                  file=sys.stderr)
            return 1
        print(f"manifest current: {len(chunks)} chunks")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    print(f"wrote {out}: {len(chunks)} chunks from {len(by_doc)} document(s)")
    for doc, n in sorted(by_doc.items()):
        print(f"  {doc}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
