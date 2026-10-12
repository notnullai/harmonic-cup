"""`/health` reports the knowledge graph (HARM-S9-T08).

With `kg/graph.json` missing — a fresh clone, since it's generated and
gitignored — the API used to run on an empty graph silently: every event got
no diagnosis and nothing said why. `/health` now carries the graph's node and
relation counts and its backend, generally enough to hold for NetworkX or
Neo4j, and the empty case is flagged rather than left for the caller to infer
from a count of zero.

No compiled engine is needed: `/health` works with the bridge absent (see the
module docstring in `api/main.py`), so this runs in CI without the rig.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx is needed by the FastAPI test client")

from fastapi.testclient import TestClient  # noqa: E402

from api import main as api_main  # noqa: E402
from api.genai_hook import GenAIHook  # noqa: E402
from kg import load  # noqa: E402
from kg.store import NetworkXStore  # noqa: E402
from rag.index import BM25Index  # noqa: E402
from rag.ingest import load_manifest  # noqa: E402
from rag.pipeline import Pipeline  # noqa: E402


@pytest.fixture
def index():
    return BM25Index(load_manifest())


def _client(store, index):
    pipeline = Pipeline(store=store, index=index, llm=None)
    hook = GenAIHook(pipeline, start=False)
    api_main.attach(None, hook)
    client = TestClient(api_main.app)
    try:
        yield client
    finally:
        client.close()
        api_main.attach(None, None)


@pytest.fixture
def empty_client(tmp_path_factory, index):
    store = NetworkXStore(tmp_path_factory.mktemp("empty-kg") / "graph.json")
    yield from _client(store, index)


@pytest.fixture
def loaded_client(tmp_path_factory, index):
    store = NetworkXStore(tmp_path_factory.mktemp("loaded-kg") / "graph.json")
    store.merge_graph(load.build_graph())
    yield from _client(store, index)


def test_health_reports_zero_counts_for_a_missing_graph(empty_client):
    body = empty_client.get("/health").json()
    assert body["kg"] == {"nodes": 0, "relations": 0, "backend": "networkx", "empty": True}


def test_health_reports_the_graph_s_real_counts_and_backend(loaded_client):
    body = loaded_client.get("/health").json()
    kg = body["kg"]
    assert kg["backend"] == "networkx"
    assert kg["empty"] is False
    assert kg["nodes"] >= load.MIN_NODES
    assert kg["relations"] >= load.MIN_RELATIONS


def test_health_works_with_no_hook_at_all(tmp_path_factory):
    api_main.attach(None, None)
    client = TestClient(api_main.app)
    try:
        body = client.get("/health").json()
        assert body["kg"] == {"nodes": 0, "relations": 0, "backend": None, "empty": True}
    finally:
        client.close()
