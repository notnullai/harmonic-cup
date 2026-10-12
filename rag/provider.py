"""Model access behind thin interfaces, so the pipeline does not import a vendor.

Three capabilities, each with a real implementation and an honest fallback:

| capability | configured            | fallback            |
|------------|-----------------------|---------------------|
| generation | Gemini via Vertex/AI  | `KGComposer`        |
| embedding  | `text-embedding-004`  | BM25, in `index.py` |

The fallbacks are **not** imitations of the real thing and are never reported
as one. `KGComposer` is a template filler over the knowledge-graph subgraph: it
writes no sentence that is not assembled from a node the graph supplied, which
means it cannot hallucinate and also cannot reason. Every response carries the
`provider` that produced it, the audit log records it, and the dashboard shows
it. A pipeline that silently substitutes a weaker component and reports the
stronger one is the failure this project has already had once, in Phase 3.

Configuration, checked in this order:

    HARMONIC_LLM=offline            force the composer, even with a key present
    GEMINI_API_KEY / GOOGLE_API_KEY Gemini Developer API
    GOOGLE_GENAI_USE_VERTEXAI=1     Vertex AI with application default creds
      + GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION

    HARMONIC_LLM_MODEL              default gemini-2.5-pro
    HARMONIC_EMBED_MODEL            default text-embedding-004
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_EMBED_MODEL = "text-embedding-004"


class LLMProvider(Protocol):
    name: str

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


def _configured() -> bool:
    if os.environ.get("HARMONIC_LLM", "").lower() in ("offline", "none", "0"):
        return False
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    return bool(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
                and os.environ.get("GOOGLE_CLOUD_PROJECT"))


# --- Gemini -------------------------------------------------------------------


class GeminiProvider:
    """google-genai, with the response schema enforced by the service.

    The schema is enforced on the way out AND revalidated on the way in
    (`rag/guard.py`). A schema the provider applied is evidence about the
    shape, not about the content: the citations still have to be checked
    against what was actually retrieved.
    """

    def __init__(self, model: str | None = None):
        from google import genai

        self._genai = genai
        self._client = genai.Client()
        self.model = model or os.environ.get("HARMONIC_LLM_MODEL", DEFAULT_MODEL)
        self.name = f"gemini:{self.model}"

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_to_genai_schema(schema),
                temperature=0.2,
            ),
        )
        text = (response.text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some models fence the JSON even when asked for a bare object.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))


def _to_genai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip what the Vertex schema dialect does not accept.

    It is a subset of JSON Schema: `$schema`, `$id`, `additionalProperties`
    and `pattern` are rejected. The full schema is still applied locally by the
    guardrail, so nothing is lost by simplifying the copy sent to the service.
    """
    drop = {"$schema", "$id", "additionalProperties", "pattern", "title"}
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _to_genai_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _to_genai_schema(v)
        else:
            out[k] = v
    return out


class GeminiEmbedder:
    name = f"gemini:{DEFAULT_EMBED_MODEL}"
    dim = 768

    def __init__(self, model: str | None = None):
        from google import genai

        self._client = genai.Client()
        self.model = model or os.environ.get("HARMONIC_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.name = f"gemini:{self.model}"

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        # In batches: the embedding endpoints cap the texts per request.
        for i in range(0, len(texts), 16):
            batch: list[Any] = list(texts[i:i + 16])
            result = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
                ),
            )
            for e in result.embeddings or []:
                out.append([float(v) for v in (e.values or [])])
        if len(out) != len(texts):
            raise RuntimeError(f"{len(out)} embeddings for {len(texts)} texts")
        return out


# --- selection ----------------------------------------------------------------


def get_llm(*, require_model: bool = False) -> LLMProvider | None:
    """The generation provider, or None when nothing is configured.

    Returning None rather than a stub is deliberate: the caller has to decide
    what to do without a model, and `rag/pipeline.py` decides to use the
    composer and say so.
    """
    if not _configured():
        if require_model:
            raise SystemExit(
                "ERROR: no model configured. Set GEMINI_API_KEY, or "
                "GOOGLE_GENAI_USE_VERTEXAI=1 with GOOGLE_CLOUD_PROJECT."
            )
        return None
    try:
        return GeminiProvider()
    except Exception as exc:
        if require_model:
            raise SystemExit(f"ERROR: Gemini is configured but unusable: {exc}") from exc
        import sys
        print(f"note: Gemini unavailable ({type(exc).__name__}); "
              f"falling back to the knowledge-graph composer", file=sys.stderr)
        return None


def get_embedder() -> Embedder | None:
    if not _configured():
        return None
    try:
        return GeminiEmbedder()
    except Exception:
        return None


def describe() -> str:
    """One line for the logs and the dashboard: what is actually answering."""
    llm = "gemini" if _configured() else "offline-kg-composer"
    return f"generation={llm} embedding={'text-embedding-004' if _configured() else 'bm25'}"
