"""Retrieval over the manual chunks (HARM-S4-T06).

Two backends behind one `Retriever` interface:

- `ChromaIndex` — `text-embedding-004` vectors in Chroma, the spec's design.
- `BM25Index` — Okapi BM25 over the same chunks, in process, no service and no
  model. This is what runs in a fresh checkout and in CI.

BM25 is not a stand-in for embeddings and is not described as one. It matches
words, so it finds "bearing replacement" and misses "the part that holds the
shaft"; embeddings do the opposite trade. Both are listed in the audit record
so a retrieval can be judged by what produced it.

The title is weighted more heavily than the body. A manual section's title is
its subject stated once, without the hedging and cross-referencing of the
prose, and an unweighted bag of words puts the section that *mentions* bearing
replacement above the section that *is* bearing replacement.

    python -m rag.index --query "bearing replacement"
    python -m rag.index --selftest
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, cast

from rag.ingest import Chunk, load_manifest
from rag.provider import Embedder, get_embedder

DEFAULT_K = 5

# How many times the title is counted relative to the body.
TITLE_WEIGHT = 3

# Okapi BM25. k1 controls how fast term frequency saturates, b how strongly
# length is normalised. The standard values; the corpus is 38 documents and is
# not large enough to justify tuning them.
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")

# Words that carry no retrieval signal in a maintenance manual. Deliberately
# short: a long stop list removes "not" and "above", which change meanings.
_STOP = frozenset([
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "is", "are", "be", "been", "it", "its", "this", "that",
])

# A stemmer is the wrong tool here. Blanket -ing stripping turns "bearing" into
# "bear" and "housing" into "hous", and those two words carry most of the
# signal in this corpus. What is needed instead is a short table of the
# morphological variants a maintenance vocabulary actually uses, so a query
# saying "seal leaking" reaches the section titled "Seal leakage".
#
# Deliberately small, and machine-domain rather than general English: it lives
# in the RAG layer because it is knowledge about pumps, not about retrieval.
DOMAIN_VARIANTS: dict[str, str] = {
    "leaking": "leak", "leakage": "leak", "leaked": "leak", "leak": "leak",
    "cavitating": "cavitation", "cavitate": "cavitation",
    "misaligned": "misalignment",
    "regreasing": "grease", "regrease": "grease", "greasing": "grease",
    "vibrating": "vibration",
    "lubricating": "lubrication", "lubricant": "lubrication",
    "spalling": "spall",
    "dropped": "drop", "dropping": "drop", "fallen": "fall", "falling": "fall",
    "raised": "rise", "rising": "rise", "risen": "rise",
}


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumerics, a crude singulariser, then the variant table."""
    out = []
    for raw in _TOKEN_RE.findall(text.lower()):
        t = raw
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        t = DOMAIN_VARIANTS.get(t, DOMAIN_VARIANTS.get(raw, t))
        if t in _STOP:
            continue
        out.append(t)
    return out


@dataclasses.dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    title: str
    section: str
    text: str

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]: ...


class BM25Index:
    name = "bm25"

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_manifest()
        self._docs: list[list[str]] = []
        for c in self.chunks:
            tokens = tokenise(c.text) + tokenise(f"{c.section} {c.title}") * TITLE_WEIGHT
            self._docs.append(tokens)

        self._tf = [Counter(d) for d in self._docs]
        self._len = [len(d) for d in self._docs]
        self._avg_len = sum(self._len) / len(self._len) if self._len else 0.0

        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._docs)
        # Robertson-Sparck-Jones idf with the +1 that keeps it positive for a
        # term appearing in every document.
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        q = tokenise(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                norm = 1.0 - BM25_B + BM25_B * (self._len[i] / (self._avg_len or 1.0))
                score += idf * (f * (BM25_K1 + 1.0)) / (f + BM25_K1 * norm)
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], self.chunks[s[1]].id))
        return [
            Hit(self.chunks[i].id, round(s, 4), self.chunks[i].title,
                self.chunks[i].section, self.chunks[i].text)
            for s, i in scored[:k]
        ]


class ChromaIndex:
    """Embeddings in Chroma. Persistent locally, HTTP when the service is up."""

    name = "chroma"

    def __init__(self, embedder: Embedder, chunks: list[Chunk] | None = None,
                 collection: str = "harmonic-manuals"):
        import chromadb

        self.chunks = chunks if chunks is not None else load_manifest()
        self.embedder = embedder
        self.name = f"chroma:{embedder.name}"

        url = os.environ.get("HARMONIC_CHROMA_URL")
        if url:
            host, _, port = url.removeprefix("http://").removeprefix("https://").partition(":")
            self._client = chromadb.HttpClient(host=host, port=int(port or 8000))
        else:
            self._client = chromadb.PersistentClient(
                path=os.environ.get("HARMONIC_CHROMA_PATH", ".chroma")
            )
        self._c = self._client.get_or_create_collection(collection)

    def build(self) -> int:
        """Upsert every chunk. Idempotent: the chunk id is the Chroma id."""
        vectors = self.embedder.embed([c.text for c in self.chunks])
        self._c.upsert(
            ids=[c.id for c in self.chunks],
            embeddings=cast(Any, vectors),
            documents=[c.text for c in self.chunks],
            metadatas=cast(Any, [{"title": c.title, "section": c.section,
                                  "doc": c.doc, "sha256": c.sha256}
                                 for c in self.chunks]),
        )
        return len(self.chunks)

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        vector = self.embedder.embed([query], query=True)[0]
        res = self._c.query(query_embeddings=cast(Any, [vector]), n_results=k)
        # Chroma types every result field as optional because `include` can
        # drop it. We ask for the defaults, which carry all four.
        ids = cast(list[list[str]], res["ids"])[0]
        docs = cast(list[list[str]], res["documents"])[0]
        metas = cast(list[list[dict[str, Any]]], res["metadatas"])[0]
        dists = cast(list[list[float]], res["distances"])[0]
        hits = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            hits.append(Hit(cid, round(1.0 - float(dist), 4),
                            str(meta.get("title", "")), str(meta.get("section", "")),
                            doc))
        return hits


class VectorIndex:
    """Embeddings from Vertex AI, searched in process (HARM-S8-G03).

    The manuals are a few dozen chunks: a vector database would be a service
    to run for a matrix that fits in a kilobyte of RAM per chunk. The chunks
    are embedded once and cached beside the data, keyed by their text and the
    model, so a restart does not pay for it again.
    """

    def __init__(self, embedder: Embedder, chunks: list[Chunk] | None = None,
                 cache_dir: Path | None = None):
        import hashlib

        import numpy as np

        self.chunks = chunks if chunks is not None else load_manifest()
        self.embedder = embedder
        self.name = f"vertex:{embedder.name}"
        key = hashlib.sha256(("".join(c.text for c in self.chunks) + embedder.name)
                             .encode()).hexdigest()[:16]
        cache = (cache_dir or Path(os.environ.get("HARMONIC_DATA_DIR", "var")))             / f"embeddings-{key}.npy"
        vectors = None
        if cache.exists():
            try:
                vectors = np.load(cache)
            except (OSError, ValueError):
                vectors = None
        if vectors is None or len(vectors) != len(self.chunks):
            vectors = np.asarray(embedder.embed([c.text for c in self.chunks]),
                                 dtype=np.float32)
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, vectors)
            except OSError:
                pass
        self._m = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        import numpy as np

        q = np.asarray(self.embedder.embed([query], query=True)[0], dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-9)
        scores = self._m @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i].id, round(float(scores[i]), 4), self.chunks[i].title,
                    self.chunks[i].section, self.chunks[i].text) for i in order]


def open_index(chunks: list[Chunk] | None = None) -> Retriever:
    """Vertex AI embeddings when a model is configured (Chroma instead when
    HARMONIC_CHROMA_URL names one), BM25 otherwise or when either fails."""
    embedder = get_embedder()
    if embedder is not None:
        try:
            if os.environ.get("HARMONIC_CHROMA_URL"):
                index = ChromaIndex(embedder, chunks)
                index.build()
                return index
            return VectorIndex(embedder, chunks)
        except Exception as exc:
            print(f"note: embedding retrieval unavailable ({type(exc).__name__}: {exc}); "
                  "using BM25", file=sys.stderr)
    return BM25Index(chunks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieve manual chunks")
    ap.add_argument("--query", default=None)
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    ap.add_argument("--bm25", action="store_true", help="force BM25")
    ap.add_argument("--selftest", action="store_true",
                    help="run the acceptance queries from the tracker")
    args = ap.parse_args(argv)

    index: Retriever = BM25Index() if args.bm25 else open_index()

    if args.selftest:
        # HARM-S4-T06's acceptance case, plus one per fault mode so a
        # regression in the tokeniser shows up as more than one failure.
        cases = [
            ("bearing replacement", "P-101-OM-4.2"),
            ("how do I align the shafts", "P-101-OM-4.4"),
            ("cavitation kurtosis crest factor", "P-101-OM-5.8"),
            ("mechanical seal leaking, flow has dropped", "P-101-OM-5.9"),
            ("grease type and regrease interval", "P-101-OM-3.1"),
            ("vibration severity zones", "P-101-OM-5.1"),
        ]
        failures = 0
        print(f"retriever: {index.name}")
        for query, expected in cases:
            hits = index.search(query, k=3)
            got = hits[0].chunk_id if hits else "-"
            ok = got == expected
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {query!r}\n"
                  f"        top-1 {got}"
                  + ("" if ok else f", expected {expected}")
                  + f"   [{', '.join(h.chunk_id for h in hits)}]")
        print(f"  RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
        return 1 if failures else 0

    if not args.query:
        ap.error("--query or --selftest is required")
    for h in index.search(args.query, args.k):
        print(f"{h.score:8.4f}  {h.chunk_id:<16} {h.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
