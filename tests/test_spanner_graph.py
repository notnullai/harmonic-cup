"""The knowledge graph in Spanner Graph answers as the in-process graph does
(HARM-S8-G07).

Runs against the Spanner emulator when SPANNER_EMULATOR_HOST is set:

    docker run -d -p 9010:9010 -p 9020:9020 gcr.io/cloud-spanner-emulator/emulator
    SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/test_spanner_graph.py
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SPANNER_EMULATOR_HOST"),
                                reason="SPANNER_EMULATOR_HOST not set")


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    from google.cloud import spanner

    from kg.load import build_graph
    from kg.store import NetworkXStore
    from kg.store_spanner import SpannerGraphStore

    project = "harmonic-test"
    client = spanner.Client(project=project)
    instance = client.instance("harmonic", configuration_name=
                               f"projects/{project}/instanceConfigs/emulator-config",
                               display_name="harmonic", node_count=1)
    if not instance.exists():
        instance.create().result(timeout=60)
    db_id = f"kg-{uuid.uuid4().hex[:8]}"
    instance.database(db_id).create().result(timeout=60)

    graph = build_graph()
    sp = SpannerGraphStore(instance="harmonic", database=db_id, project=project)
    sp.merge_graph(graph)
    nx = NetworkXStore(tmp_path_factory.mktemp("kg") / "graph.json", autoload=False)
    nx.merge_graph(graph)
    yield sp, nx
    instance.database(db_id).drop()


def _key(edges):
    return sorted((e.other.id, e.relation, e.source, repr(sorted(e.props.items())))
                  for e in edges)


def test_the_same_graph_goes_in(stores):
    sp, nx = stores
    assert sp.counts() == nx.counts()
    assert sp.counts()[0] >= 200


def test_traversals_match_edge_for_edge(stores):
    sp, nx = stores
    for node_id, rel in [("gbx.sym.sidebands.high", "INDICATES"), ("sym.bpfo.high", None),
                         ("fm.cavitation", "MITIGATED_BY"), ("GB-201", "HAS_COMPONENT")]:
        assert _key(sp.out_edges(node_id, rel)) == _key(nx.out_edges(node_id, rel)), node_id
    assert _key(sp.in_edges("gbx.fm.tooth_breakage")) == _key(nx.in_edges("gbx.fm.tooth_breakage"))
    assert sp.node("GB-201").props["family"] == "gearbox"
    assert {n.id for n in sp.nodes_by_label("FailureMode")} == \
        {n.id for n in nx.nodes_by_label("FailureMode")}


@pytest.mark.parametrize("event", [
    {"equipment_id": "GB-201", "top_classes": [{"name": "broken_tooth", "score": 0.8}],
     "top_surprised_features": [{"kind": "HARMONIC_FEAT_VIB_ORDER:mesh_sb", "weight": 1.0},
                                {"kind": "HARMONIC_FEAT_VIB_KURTOSIS", "weight": 0.8}]},
    {"equipment_id": "P-101", "top_classes": [{"name": "bearing_wear", "score": 0.9}],
     "top_surprised_features": [{"kind": "HARMONIC_FEAT_VIB_BPFO", "direction": "up",
                                 "weight": 1.0}]},
])
def test_a_diagnosis_walks_spanner_to_the_same_answer(stores, event):
    from kg.query import query_for_event

    sp, nx = stores
    a, b = query_for_event(event, sp), query_for_event(event, nx)
    assert [c.failure_mode_id for c in a.candidates] == [c.failure_mode_id for c in b.candidates]
    assert [round(c.score, 6) for c in a.candidates] == [round(c.score, 6) for c in b.candidates]
    assert sorted(a.citable_ids) == sorted(b.citable_ids)
