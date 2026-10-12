"""Triple extraction from the manual chunks (HARM-S4-T02).

Two extractors behind one output format:

- `gemini` — an LLM reads each chunk and returns typed triples against a JSON
  schema. This is the one the spec asks for.
- `rules` — sentence patterns plus a lexicon that links surface phrases to the
  canonical node ids in `kg/seed.py`. This is what runs when no model is
  configured, which is the default state of a checkout.

The rules extractor is **not** a pretend LLM and is not labelled as one. Every
triple carries `source="rules:<chunk id>"` or `source="gemini:<chunk id>"`, the
loader writes that onto the relation, and `kg/query.py` can tell an engineer
where a claim came from. A pipeline that quietly substitutes a weaker component
and reports the stronger one is the failure this project has already had once.

Both paths end in the same place: canonical ids, `validate_graph`, dedup. An
extracted triple that does not fit docs/KG_SCHEMA.md is dropped with a reason,
not coerced.

    python -m kg.extract                       # rules, writes kg/extracted.json
    python -m kg.extract --extractor gemini    # needs GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kg import family_data
from kg.model import Graph, Node, SchemaError, Triple, validate_graph
from kg.seed import build_seed
from rag.ingest import Chunk, load_manifest

if TYPE_CHECKING:
    from rag.provider import LLMProvider

EXTRACTED_JSON = Path(__file__).resolve().parent / "extracted.json"

# What the LLM must return. Kept deliberately narrow: entities are chosen from
# an id list supplied in the prompt rather than invented, because an extractor
# free to mint ids produces a graph where nothing joins to anything.
MAX_TRIPLES_PER_CHUNK = 40

EXTRACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["triples"],
    "additionalProperties": False,
    "properties": {
        "triples": {
            # Bounds (item count, confidence range, evidence length) are
            # enforced in extract_llm, not here: Vertex refuses a schema whose
            # nested bounds produce "too many states for serving".
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["INDICATES", "CAUSES", "MITIGATED_BY",
                                 "DOCUMENTED_IN", "DESCRIBED_IN", "AFFECTS",
                                 "GOVERNED_BY"],
                    },
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
            },
        }
    },
}

# Surface phrases that name a node — the vocabulary a manual uses for it —
# from each family's data file (`kg/families/*.json`, "aliases"). Derived ids
# stay in the files' nodes. Longest match wins, so "outer race wear" is tried
# before "outer race".
ALIASES: dict[str, tuple[str, ...]] = {
    k: v for f in family_data.load_all() for k, v in f.aliases.items()}

# Sentence-level patterns. Each returns (subject kinds, relation, object kinds)
# and the extractor links the phrases on either side of the cue.
_INDICATE_CUE = re.compile(
    r"\b(indicates?|indicating|is the definitive indication of|is the signature of|"
    r"are the signature of|mimics|confused with)\b", re.IGNORECASE)
_CAUSE_CUE = re.compile(
    r"\b(causes are|is a cause of|causes?\b|caused by|produces?|destroys|"
    r"results? in|leads? to)\b", re.IGNORECASE)
_MITIGATE_CUE = re.compile(
    r"\b(mitigation is|the mitigation is|is cured by|correct(?:ed)? by|"
    r"remedy is)\b", re.IGNORECASE)

_CONFIDENCE_WORDS = [
    (re.compile(r"\bdefinitive\b|\bhigh confidence\b|\bstrongly\b", re.IGNORECASE), 0.85),
    (re.compile(r"\busually\b|\bcommonly\b|\btypically\b", re.IGNORECASE), 0.60),
    (re.compile(r"\bmore often\b|\bmay\b|\bcan be\b|\bweakly\b", re.IGNORECASE), 0.40),
]
_DEFAULT_CONFIDENCE = 0.55

_SENTENCE_RE = re.compile(r"(?<=[.;])\s+")


@dataclass
class Dropped:
    chunk: str
    triple: tuple[str, str, str]
    reason: str


class Lexicon:
    """Surface phrase to canonical node id, longest match first."""

    def __init__(self, labels: dict[str, str]):
        self.labels = labels
        pairs: list[tuple[str, str]] = []
        for node_id, aliases in ALIASES.items():
            if node_id not in labels:
                raise SchemaError(f"ALIASES names {node_id}, which is not a seed node")
            for alias in aliases:
                pairs.append((alias.lower(), node_id))
        # Longest alias first so "outer race wear" beats "outer race".
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))

    def find_positions(self, text: str, want: set[str] | None = None,
                       skip: frozenset[str] = frozenset()) -> list[tuple[str, int]]:
        """As `find`, but with the character offset of each mention."""
        low = text.lower()
        claimed: list[tuple[int, int]] = []
        hits: list[tuple[int, str]] = []
        for alias, node_id in self.pairs:
            if (want is not None and self.labels[node_id] not in want) or node_id in skip:
                continue
            start = low.find(alias)
            while start >= 0:
                end = start + len(alias)
                if not any(s < end and start < e for s, e in claimed):
                    claimed.append((start, end))
                    hits.append((start, node_id))
                    break
                start = low.find(alias, start + 1)
        seen: set[str] = set()
        out: list[tuple[str, int]] = []
        for pos, node_id in sorted(hits):
            if node_id not in seen:
                seen.add(node_id)
                out.append((node_id, pos))
        return out

    def find(self, text: str, want: set[str] | None = None,
             skip: frozenset[str] = frozenset()) -> list[str]:
        """Node ids mentioned in `text`, in order of first appearance.

        Overlapping matches are suppressed: once a span is claimed by the
        longest alias, a shorter alias inside it is not a second entity.
        """
        low = text.lower()
        claimed: list[tuple[int, int]] = []
        hits: list[tuple[int, str]] = []
        for alias, node_id in self.pairs:
            if (want is not None and self.labels[node_id] not in want) or node_id in skip:
                continue
            start = low.find(alias)
            while start >= 0:
                end = start + len(alias)
                if not any(s < end and start < e for s, e in claimed):
                    claimed.append((start, end))
                    hits.append((start, node_id))
                    break
                start = low.find(alias, start + 1)
        out: list[str] = []
        for _, node_id in sorted(hits):
            if node_id not in out:
                out.append(node_id)
        return out


def _pair_by_distance(mentions: list[tuple[str, int]], refs: list[tuple[int, str]]
                      ) -> list[tuple[str, str]]:
    """Match each action mention to at most one section reference, greedily.

    A sentence like "the alignment in section 4.4, after the soft-foot check
    and after the coupling inspection in section 4.5" names two actions and two
    sections. Taking the cross product claims the soft-foot check is documented
    in the coupling section; taking the nearest FOLLOWING reference gives the
    soft-foot check section 4.5 for the same reason. Closest-pair-first
    assignment, each side used once, gets both right, and when the counts are
    unequal it leaves the surplus unmatched rather than guessing.
    """
    if not mentions or not refs:
        return []
    pairs = sorted(
        ((abs(pos - rpos), act, target)
         for act, pos in mentions for rpos, target in refs),
        key=lambda p: (p[0], p[1], p[2]),
    )
    used_actions: set[str] = set()
    used_refs: set[str] = set()
    out: list[tuple[str, str]] = []
    for _, act, target in pairs:
        if act in used_actions or target in used_refs:
            continue
        used_actions.add(act)
        used_refs.add(target)
        out.append((act, target))
    return out


def _confidence(sentence: str) -> float:
    for pattern, value in _CONFIDENCE_WORDS:
        if pattern.search(sentence):
            return value
    return _DEFAULT_CONFIDENCE


def extract_rules(chunks: list[Chunk], lex: Lexicon,
                  foreign: dict[str, frozenset[str]] | None = None
                  ) -> tuple[list[Triple], list[Dropped]]:
    """Pattern-and-lexicon extraction over the manual prose.

    `foreign` maps a document to the nodes of other machine families,
    which its prose must not be matched to.
    """
    triples: list[Triple] = []
    dropped: list[Dropped] = []
    section_ids = {c.id for c in chunks}

    for chunk in chunks:
        src = f"rules:{chunk.id}"
        skip = (foreign or {}).get(chunk.doc, frozenset())
        # The failure modes this section is about, used to attach causes and
        # mitigations that the prose states without naming the mode again.
        section_modes = lex.find(f"{chunk.title}. {chunk.text}", want={"FailureMode"}, skip=skip)

        for sentence in _SENTENCE_RE.split(chunk.text):
            s = sentence.strip()
            if not s:
                continue

            if _INDICATE_CUE.search(s):
                cue = _INDICATE_CUE.search(s)
                assert cue is not None
                left, right = s[:cue.start()], s[cue.end():]
                for sym in lex.find(left, want={"Symptom"}, skip=skip):
                    for fm in lex.find(right, want={"FailureMode"}, skip=skip):
                        triples.append(Triple(sym, "INDICATES", fm,
                                              {"confidence": _confidence(s)}, source=src))

            if _CAUSE_CUE.search(s):
                causes = lex.find(s, want={"Cause"}, skip=skip)
                modes = lex.find(s, want={"FailureMode"}, skip=skip) or section_modes[:1]
                for cause in causes:
                    for fm in modes:
                        triples.append(Triple(cause, "CAUSES", fm, {}, source=src))

            if _MITIGATE_CUE.search(s):
                actions = lex.find(s, want={"Action"}, skip=skip)
                modes = lex.find(s, want={"FailureMode"}, skip=skip) or section_modes[:1]
                for act in actions:
                    for fm in modes:
                        triples.append(Triple(fm, "MITIGATED_BY", act, {}, source=src))

            # A procedure that names a section documents itself there — but
            # pair them by POSITION, not by cross product. "the alignment in
            # section 4.4, after the soft-foot check and after the coupling
            # inspection in section 4.5" names two actions and two sections,
            # and the cross product claims the soft-foot check is documented in
            # the coupling section. Each action takes the nearest section
            # reference that follows it.
            refs = [(m.start(), f"{chunk.doc}-{m.group(1)}")
                    for m in re.finditer(r"sections?\s+(\d+\.\d+)", s, re.IGNORECASE)]
            mentions = lex.find_positions(s, want={"Action"}, skip=skip)
            for act, target in _pair_by_distance(mentions, refs):
                if target in section_ids:
                    triples.append(Triple(act, "DOCUMENTED_IN", target, {}, source=src))

        # Every failure mode a section discusses is described by that section.
        for fm in section_modes:
            triples.append(Triple(fm, "DESCRIBED_IN", chunk.id, {}, source=src))

    return triples, dropped


# --- LLM path -----------------------------------------------------------------

_PROMPT = """You are building a maintenance knowledge graph for rotating machinery.

Read the manual section below and return the relationships it states. Use ONLY
the entity ids in the lists provided — do not invent ids, and do not return a
relationship the section does not actually state. An id must appear in the list
for the role it is used in.

Relations and their allowed endpoints:
  Symptom     INDICATES     FailureMode   (include a confidence 0..1)
  Cause       CAUSES        FailureMode
  FailureMode MITIGATED_BY  Action
  Action      DOCUMENTED_IN ManualSection
  FailureMode DESCRIBED_IN  ManualSection

Symptom ids:      {symptoms}
FailureMode ids:  {failure_modes}
Cause ids:        {causes}
Action ids:       {actions}
This section's ManualSection id: {chunk_id}

For each triple include `evidence`: the clause from the section that states it.
If the section states nothing of this kind, return an empty list.

SECTION {chunk_id} — {title}
{text}
"""


def build_prompt(chunk: Chunk, labels: dict[str, str],
                 skip: frozenset[str] = frozenset()) -> str:
    def ids(label: str) -> str:
        return ", ".join(sorted(n for n, lb in labels.items() if lb == label and n not in skip))

    return _PROMPT.format(
        symptoms=ids("Symptom"), failure_modes=ids("FailureMode"),
        causes=ids("Cause"), actions=ids("Action"),
        chunk_id=chunk.id, title=chunk.title, text=chunk.text,
    )


def extract_llm(chunks: list[Chunk], labels: dict[str, str], provider: LLMProvider,
                foreign: dict[str, frozenset[str]] | None = None
                ) -> tuple[list[Triple], list[Dropped]]:
    triples: list[Triple] = []
    dropped: list[Dropped] = []
    for chunk in chunks:
        skip = (foreign or {}).get(chunk.doc, frozenset())
        raw = provider.generate_json(build_prompt(chunk, labels, skip), EXTRACTION_SCHEMA)
        for item in raw.get("triples", [])[:MAX_TRIPLES_PER_CHUNK]:
            props: dict[str, object] = {}
            if "confidence" in item:
                props["confidence"] = min(1.0, max(0.0, float(item["confidence"])))
            if "evidence" in item:
                props["evidence"] = str(item["evidence"])[:400]
            triples.append(Triple(str(item["subject"]), str(item["relation"]),
                                  str(item["object"]), props,
                                  source=f"gemini:{chunk.id}"))
    return triples, dropped


# --- shared post-processing ---------------------------------------------------


def canonicalise(triples: list[Triple], labels: dict[str, str], section_ids: set[str]
                 ) -> tuple[Graph, list[Dropped]]:
    """Drop what does not fit the schema, then deduplicate.

    Dropping is on purpose. An extractor that coerces a bad triple into a
    plausible one produces a graph whose errors are invisible; one that drops
    with a reason produces a graph plus a list of what it could not use.
    """
    known = set(labels) | section_ids
    kept: dict[tuple[str, str, str], Triple] = {}
    dropped: list[Dropped] = []

    for t in triples:
        chunk = t.source.split(":", 1)[-1]
        if t.subject not in known:
            dropped.append(Dropped(chunk, t.key(), f"unknown subject {t.subject}"))
            continue
        if t.object not in known:
            dropped.append(Dropped(chunk, t.key(), f"unknown object {t.object}"))
            continue
        if t.subject == t.object:
            dropped.append(Dropped(chunk, t.key(), "self-referential"))
            continue

        probe = Graph(
            [Node(labels.get(t.subject, "ManualSection"), t.subject),
             Node(labels.get(t.object, "ManualSection"), t.object)],
            [t],
        )
        try:
            # Node property rules do not apply here: the extractor contributes
            # edges between nodes the seed already defines, not new nodes.
            from kg.model import _check_triple
            errors = _check_triple(t, probe.node_labels())
        except Exception as exc:
            dropped.append(Dropped(chunk, t.key(), str(exc)))
            continue
        if errors:
            dropped.append(Dropped(chunk, t.key(), errors[0].split(": ", 1)[-1]))
            continue

        if t.key() in kept:
            prev = kept[t.key()]
            merged = {**prev.props, **t.props}
            # Two sections agreeing is evidence; keep the higher confidence.
            if "confidence" in prev.props and "confidence" in t.props:
                merged["confidence"] = max(prev.props["confidence"], t.props["confidence"])
            kept[t.key()] = Triple(t.subject, t.relation, t.object, merged,
                                   source=f"{prev.source};{t.source}"
                                   if prev.source != t.source else prev.source)
        else:
            kept[t.key()] = t

    return Graph([], list(kept.values())), dropped


def _foreign_nodes(docs: set[str]) -> dict[str, frozenset[str]]:
    """For each manual, the nodes of every other family's file.

    A manual speaks for its own machine: the gearbox manual's "balancing" is not
    the pump's impeller balancing, and matching it there would cite the pump's
    manual for a gearbox fault. So a manual is matched only against its own
    family's nodes (`kg/families/<family>.json`). A manual listed under no family
    is matched against everything, as before.
    """
    from kg import families

    doc_family = {doc: name for name, fam in families.families().items()
                  for doc in fam.get("manuals", [])}
    owned = {f.family: {n.id for n in f.graph.nodes} for f in family_data.load_all()}
    every = set().union(*owned.values()) if owned else set()
    return {doc: frozenset(every - owned.get(doc_family[doc], set()))
            for doc in docs if doc in doc_family}


def run(extractor: str, chunks: list[Chunk], provider: LLMProvider | None = None
        ) -> tuple[Graph, list[Dropped]]:
    seed = build_seed()
    labels = seed.node_labels()
    section_ids = {c.id for c in chunks}
    foreign = _foreign_nodes({c.doc for c in chunks})

    if extractor == "rules":
        raw, dropped = extract_rules(chunks, Lexicon(labels), foreign)
    elif extractor == "gemini":
        if provider is None:
            raise SystemExit("ERROR: --extractor gemini needs a configured provider; "
                             "set GEMINI_API_KEY or use --extractor rules")
        raw, dropped = extract_llm(chunks, labels, provider, foreign)
    else:
        raise SystemExit(f"ERROR: unknown extractor {extractor!r}")

    graph, more = canonicalise(raw, labels, section_ids)
    return graph, [*dropped, *more]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract triples from the manual chunks")
    ap.add_argument("--extractor", choices=["rules", "gemini"], default="rules")
    ap.add_argument("--out", default=str(EXTRACTED_JSON))
    ap.add_argument("--min-triples", type=int, default=50,
                    help="fail if fewer than this many survive validation")
    args = ap.parse_args(argv)

    chunks = load_manifest()
    provider = None
    if args.extractor == "gemini":
        from rag.provider import get_llm
        provider = get_llm(require_model=True)

    graph, dropped = run(args.extractor, chunks, provider)

    # The extracted graph contributes edges only; the nodes belong to the seed
    # and the manifest. Validate it against that combined vocabulary.
    from kg.load import manual_section_nodes
    combined = build_seed().merge(Graph(manual_section_nodes(), []))
    validate_graph(Graph(combined.nodes, graph.triples))

    by_relation = dict(Counter(t.relation for t in graph.triples))
    payload: dict[str, object] = {
        "extractor": args.extractor,
        "n_chunks": len(chunks),
        "n_triples": len(graph.triples),
        "n_dropped": len(dropped),
        "by_relation": by_relation,
        "triples": [t.to_json() for t in graph.triples],
        "dropped": [{"chunk": d.chunk, "triple": list(d.triple), "reason": d.reason}
                    for d in dropped],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"wrote {args.out}: {len(graph.triples)} triples from {len(chunks)} chunks "
          f"via {args.extractor}, {len(dropped)} dropped")
    for rel, n in sorted(by_relation.items()):
        print(f"  {rel}: {n}")
    if len(graph.triples) < args.min_triples:
        print(f"FAIL: {len(graph.triples)} triples, target {args.min_triples}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
