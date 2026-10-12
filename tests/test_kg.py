"""Knowledge graph: schema, seed, extraction, loading, query (HARM-S4-T01..T05).

The acceptance criteria from docs/task_tracker.md are asserted here rather than
described in a review, so a later change that quietly drops the graph below
them fails the build.
"""

from __future__ import annotations

import json

import pytest

from kg import extract, load, query, seed
from kg.model import Graph, Node, SchemaError, Triple, validate_graph
from kg.store import NetworkXStore
from rag.ingest import build, load_manifest, manifest_payload


@pytest.fixture(scope="module")
def chunks():
    return load_manifest()


@pytest.fixture(scope="module")
def store(tmp_path_factory, chunks):
    """A graph built from scratch, exactly as `python -m kg.load --rebuild` does."""
    path = tmp_path_factory.mktemp("kg") / "graph.json"
    s = NetworkXStore(path)
    s.merge_graph(load.build_graph())
    return s


# --- ingestion ----------------------------------------------------------------


def test_manifest_has_enough_chunks(chunks):
    """HARM-S4-T01: at least 30 chunks, each with a section id."""
    assert len(chunks) >= 30, f"only {len(chunks)} chunks"
    assert all(c.section and c.id and c.text for c in chunks)


def test_manifest_is_current():
    """The committed manifest must match the manuals it was built from.

    A stale manifest means the retriever serves text the graph does not have
    and the citation allow-list is built from neither.
    """
    fresh = manifest_payload(build())
    committed = json.loads((load_manifest.__globals__["MANIFEST"]).read_text("utf-8"))
    assert fresh == committed, "run: python -m rag.ingest"


def test_chunk_ids_are_valid_citation_tokens(chunks):
    """docs/GENAI_SCHEMA.json constrains references to ^[A-Za-z0-9._-]+$."""
    import re
    bad = [c.id for c in chunks if not re.fullmatch(r"[A-Za-z0-9._-]+", c.id)]
    assert not bad, bad


# --- schema -------------------------------------------------------------------


def test_schema_rejects_an_unknown_relation_endpoint():
    g = Graph(
        [Node("Symptom", "s", {"feature_kind": "X", "direction": "up"}),
         Node("Action", "a", {"skill_level": "technician"})],
        [Triple("s", "INDICATES", "a")],
    )
    with pytest.raises(SchemaError, match="INDICATES"):
        validate_graph(g)


def test_schema_rejects_a_symptom_without_a_feature_kind():
    g = Graph([Node("Symptom", "s", {"direction": "up"})], [])
    with pytest.raises(SchemaError, match="feature_kind"):
        validate_graph(g)


def test_schema_reports_every_problem_not_just_the_first():
    g = Graph(
        [Node("Cause", "c", {"category": "nonsense"}),
         Node("Action", "a", {"skill_level": "wizard"})],
        [],
    )
    with pytest.raises(SchemaError) as exc:
        validate_graph(g)
    assert "2 schema problem" in str(exc.value)


def test_a_manual_section_outside_the_manifest_is_rejected():
    """The citation allow-list and the graph's ManualSection nodes are one set.

    If the graph can offer an id the retriever does not know, the guardrail
    rejects a citation the graph itself supplied.
    """
    g = Graph([Node("ManualSection", "P-101-OM-99.9", {})], [])
    with pytest.raises(SchemaError, match="not in the chunk manifest"):
        validate_graph(g, citable_ids=["P-101-OM-4.2"])


# --- seed ---------------------------------------------------------------------


def test_seed_validates_against_the_schema():
    g = seed.build_seed().merge(Graph(load.manual_section_nodes(), []))
    validate_graph(g)


def test_every_trained_class_maps_to_a_failure_mode_that_exists():
    """The class neurons are the engine's half of the join. A class with no
    failure mode behind it produces an alarm the graph cannot explain."""
    from sim.pump_sim import FAULT_MODES

    ids = {n.id for n in seed.build_seed().nodes}
    for fault in FAULT_MODES:
        fm = seed.CLASS_TO_FAILURE_MODE.get(fault)
        assert fm, f"no failure mode mapped for the trained class {fault}"
        assert fm in ids, f"{fault} maps to {fm}, which is not in the graph"


def test_every_symptom_names_a_real_engine_feature():
    """`feature_kind` must be a harmonic_feature_kind enumerator, or the
    query joins on nothing and every event returns no candidates."""
    header = (load.__file__.rsplit("kg", 1)[0]
              + "engine/include/harmonic/harmonic_types.h")
    with open(header, encoding="utf-8") as fh:
        source = fh.read()
    for node in seed.build_seed().nodes:
        if node.label != "Symptom":
            continue
        # A declared feature is "KIND:label" (HARMONIC_FEAT_VIB_ORDER:gear_mesh):
        # the kind must be the engine's and the label a family profile's.
        kind, _, label = node.props["feature_kind"].partition(":")
        assert kind in source, f"{node.id} names {kind}, which the engine does not define"
        if label:
            from kg import families
            profiles = "".join(families.path(f, "profile").read_text()
                               for f in families.families())
            assert (f"order.{label}" in profiles or f"process.{label}" in profiles),                 f"{node.id} names {label}, which no family profile declares"


def test_cypher_generation_round_trips():
    g = seed.build_seed().merge(Graph(load.manual_section_nodes(), []))
    cypher = seed.to_cypher(g)
    assert "CREATE CONSTRAINT" in cypher
    assert cypher.count("MERGE (n:") == len(g.nodes)
    assert "'" in cypher  # quoting happened at all
    # Every statement must terminate. An unescaped apostrophe inside a
    # description would close its string early and swallow the rest.
    assert cypher.count(";") >= len(g.nodes) + len(g.triples)


# --- extraction ---------------------------------------------------------------


def test_extraction_yields_enough_valid_triples(chunks):
    """HARM-S4-T02: at least 50 valid triples."""
    graph, _dropped = extract.run("rules", chunks)
    assert len(graph.triples) >= 50, f"only {len(graph.triples)}"
    combined = seed.build_seed().merge(Graph(load.manual_section_nodes(), []))
    validate_graph(Graph(combined.nodes, graph.triples))


def test_every_extracted_triple_records_where_it_came_from(chunks):
    graph, _ = extract.run("rules", chunks)
    assert all(t.source.startswith("rules:") for t in graph.triples)


def test_extraction_drops_rather_than_coerces():
    """A triple naming something outside the vocabulary is discarded with a
    reason. Coercing it would hide the extractor's errors in the graph."""
    labels = seed.build_seed().node_labels()
    bad = [Triple("sym.bpfo.high", "INDICATES", "fm.not_a_real_mode",
                  source="rules:P-101-OM-5.4")]
    graph, dropped = extract.canonicalise(bad, labels, set())
    assert not graph.triples
    assert dropped and "unknown object" in dropped[0].reason


def test_a_section_reference_is_paired_with_the_nearest_action():
    """The sentence that broke the cross-product version, kept as a test."""
    lex = extract.Lexicon(seed.build_seed().node_labels())
    sentence = ("Mitigation is the alignment in section 4.4, after the soft-foot "
                "check and after the coupling inspection in section 4.5.")
    import re
    refs = [(m.start(), f"P-101-OM-{m.group(1)}")
            for m in re.finditer(r"sections?\s+(\d+\.\d+)", sentence, re.IGNORECASE)]
    pairs = dict(extract._pair_by_distance(
        lex.find_positions(sentence, want={"Action"}), refs))
    assert pairs.get("act.soft_foot_check") == "P-101-OM-4.4"
    assert pairs.get("act.replace_coupling_element") == "P-101-OM-4.5"


# --- loading ------------------------------------------------------------------


def test_graph_meets_the_size_targets(store):
    """HARM-S4-T04: at least 100 nodes and 150 relations."""
    n, r = store.counts()
    assert n >= load.MIN_NODES, f"{n} nodes"
    assert r >= load.MIN_RELATIONS, f"{r} relations"


def test_an_empty_graph_reports_zero_counts_on_every_backend(tmp_path):
    """HARM-S9-T08: `/health` trusts `counts()` to say when the graph is
    empty — e.g. `kg/graph.json` missing on a fresh clone. Both backends
    must agree, not just the NetworkX fallback."""
    missing = NetworkXStore(tmp_path / "does-not-exist.json")
    assert missing.counts() == (0, 0)
    assert missing.backend == "networkx"

    neo4j = _neo4j_or_skip()
    try:
        neo4j.clear()
        assert neo4j.counts() == (0, 0)
        assert neo4j.backend == "neo4j"
    finally:
        neo4j.close()


def test_loading_is_idempotent(tmp_path):
    """A MERGE loader that duplicates on re-run makes every count a lie."""
    s = NetworkXStore(tmp_path / "g.json")
    graph = load.build_graph()
    first = s.merge_graph(graph)
    second = s.merge_graph(graph)
    assert (first.nodes_total, first.relations_total) == \
           (second.nodes_total, second.relations_total)
    assert second.nodes_written == 0
    assert second.relations_written == 0


def test_rebuild_removes_what_the_seed_no_longer_says(tmp_path):
    """MERGE never deletes, so an edge dropped from the seed survives an
    ordinary load. `--rebuild` is how the graph stops growing forever."""
    s = NetworkXStore(tmp_path / "g.json")
    s.merge_graph(Graph([Node("Standard", "stale", {"name": "gone"})], []))
    assert s.node("stale") is not None
    s.clear()
    s.merge_graph(load.build_graph())
    assert s.node("stale") is None


def test_the_curated_seed_wins_over_an_extracted_confidence(tmp_path):
    """The extractor guesses confidence from hedging words; a reliability
    engineer's number has to survive the merge."""
    extracted = Graph([], [Triple("sym.bpfo.high", "INDICATES", "fm.outer_race_wear",
                                  {"confidence": 0.4}, source="rules:x")])
    merged = Graph(load.manual_section_nodes(), []).merge(extracted).merge(seed.build_seed())
    edge = next(t for t in merged.triples
                if t.key() == ("sym.bpfo.high", "INDICATES", "fm.outer_race_wear"))
    assert edge.props["confidence"] == 0.90


# --- query --------------------------------------------------------------------


def test_bearing_symptoms_rank_outer_race_wear_first(store):
    """HARM-S4-T05's acceptance case."""
    result = query.query_for_event({"top_surprised_features": [
        "HARMONIC_FEAT_VIB_BPFO",
        "HARMONIC_FEAT_VIB_RMS",
        "HARMONIC_FEAT_TEMP_BEARING",
    ]}, store)
    assert result.top is not None
    assert result.top.failure_mode_id == "fm.outer_race_wear"


@pytest.mark.parametrize(("features", "expected"), [
    (["HARMONIC_FEAT_VIB_HARM_2X", "HARMONIC_FEAT_VIB_HARM_3X"], "fm.misalignment"),
    (["HARMONIC_FEAT_VIB_KURTOSIS", "HARMONIC_FEAT_VIB_CREST"], "fm.cavitation"),
    (["HARMONIC_FEAT_FLOW", "HARMONIC_FEAT_PRESS_DISCHARGE"], "fm.seal_leakage"),
    (["HARMONIC_FEAT_VIB_BPFI"], "fm.inner_race_wear"),
])
def test_each_signature_reaches_its_failure_mode(store, features, expected):
    result = query.query_for_event({"top_surprised_features": features}, store)
    assert result.top is not None and result.top.failure_mode_id == expected


def test_evidence_reinforces_rather_than_averages(store):
    """Two symptoms agreeing must not score lower than the stronger alone.

    Averaging would do exactly that, and it is the bug that was found in the
    engine's health index in Phase 3 — worth pinning in both places.
    """
    one = query.query_for_event(
        {"top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"]}, store)
    two = query.query_for_event(
        {"top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO",
                                    "HARMONIC_FEAT_TEMP_BEARING"]}, store)
    assert two.top.score >= one.top.score


def test_a_disagreement_between_class_and_evidence_is_reported(store):
    """Not hidden behind a single winner: the class was trained on a simulator
    and the confidences come from practice, so the two differing is news."""
    result = query.query_for_event({
        "top_classes": [{"name": "cavitation", "score": 0.9}],
        "top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"],
    }, store)
    assert result.agreement.startswith("disagree:")
    ids = {c.failure_mode_id for c in result.candidates}
    assert {"fm.cavitation", "fm.outer_race_wear"} <= ids


def test_a_class_with_no_supporting_feature_is_flagged(store):
    result = query.query_for_event({
        "top_classes": [{"name": "cavitation", "score": 0.9}],
        "top_surprised_features": [],
    }, store)
    assert result.agreement == "no-evidence"


def test_features_the_graph_does_not_model_are_reported_not_dropped(store):
    result = query.query_for_event({
        "top_surprised_features": ["HARMONIC_FEAT_ACOUSTIC_RMS"]}, store)
    assert "HARMONIC_FEAT_ACOUSTIC_RMS" in result.unmatched_features


def test_every_citable_id_is_a_real_chunk(store, chunks):
    known = {c.id for c in chunks}
    result = query.query_for_event({"top_surprised_features": [
        "HARMONIC_FEAT_VIB_BPFO", "HARMONIC_FEAT_VIB_HARM_2X",
        "HARMONIC_FEAT_FLOW", "HARMONIC_FEAT_VIB_KURTOSIS",
    ]}, store)
    assert result.citable_ids
    assert set(result.citable_ids) <= known


def test_actions_come_back_in_priority_order(store):
    result = query.query_for_event({
        "top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"]}, store)
    priorities = [a.priority for a in result.top.actions]
    assert priorities == sorted(priorities)


def test_safety_sections_are_not_cited_inside_a_work_step(store):
    """A work order whose first instruction cites the lockout procedure buries
    the lockout. They are different relations for that reason."""
    result = query.query_for_event({
        "top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"]}, store)
    for action in result.top.actions:
        assert not any(r.startswith("P-101-OM-2.") for r in action.references), action


# --- backend parity -----------------------------------------------------------


def _neo4j_or_skip():
    """Neo4j when docker-compose is up, skipped otherwise.

    Skipped rather than faked. The fallback exists so the demo survives Docker
    being unavailable, and the only way to know the two agree is to run the
    same assertions against both when both are there.
    """
    try:
        from kg.store import Neo4jStore
        return Neo4jStore()
    except Exception as exc:
        pytest.skip(f"Neo4j unavailable: {type(exc).__name__}")


@pytest.fixture(scope="module")
def neo4j_store():
    s = _neo4j_or_skip()
    s.clear()
    s.merge_graph(load.build_graph())
    yield s
    s.close()


def test_the_two_backends_hold_the_same_graph(store, neo4j_store):
    assert neo4j_store.counts() == store.counts()


def test_the_two_backends_answer_the_same_way(store, neo4j_store):
    """kg/query.py has no Cypher in it for this reason: one traversal, two
    stores. A query that only ran on one backend would mean the fallback
    answered differently on the day it was needed."""
    event = {
        "top_classes": [{"name": "bearing_wear", "score": 0.88}],
        "top_surprised_features": [
            {"kind": "HARMONIC_FEAT_VIB_BPFO", "sigma": 3.1},
            {"kind": "HARMONIC_FEAT_TEMP_BEARING", "sigma": 1.8},
        ],
    }
    a = query.query_for_event(event, store).to_json()
    b = query.query_for_event(event, neo4j_store).to_json()
    assert a == b


def test_a_failure_mode_the_machine_cannot_have_is_ruled_out(store):
    """The same event on two machines. The IMS bearing rig has no impeller, so
    cavitation — which the simulator-trained class said, on real data — is
    not a candidate there, and the bearing evidence is left to speak."""
    event = {
        "top_classes": [{"name": "cavitation", "score": 0.9}],
        "top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"],
    }
    pump = query.query_for_event({**event, "equipment_id": "P-101"}, store)
    rig = query.query_for_event({**event, "equipment_id": "IMS-RIG"}, store)

    assert "fm.cavitation" in {c.failure_mode_id for c in pump.candidates}
    assert pump.ruled_out == ()
    assert "fm.cavitation" in rig.ruled_out
    assert "fm.cavitation" not in {c.failure_mode_id for c in rig.candidates}
    assert rig.top is not None and rig.top.failure_mode_id == "fm.outer_race_wear"


def test_an_unknown_machine_rules_nothing_out(store):
    """No equipment node, no components, no basis for excluding anything."""
    result = query.query_for_event({
        "equipment_id": "NOT-IN-THE-GRAPH",
        "top_classes": [{"name": "cavitation", "score": 0.9}],
        "top_surprised_features": ["HARMONIC_FEAT_VIB_BPFO"],
    }, store)
    assert result.ruled_out == ()


def test_classes_a_machine_cannot_have_are_named_with_the_part_it_lacks(store):
    from kg.query import classes_ruled_out

    assert classes_ruled_out(store, "P-101") == {}
    assert classes_ruled_out(store, "GB-201") == {}
    rig = classes_ruled_out(store, "IMS-RIG")
    assert rig["cavitation"] == "impeller" and "bearing_wear" not in rig


def test_the_families_are_data_files_and_build_the_curated_graph():
    """HARM-S7-T10: one file per family; the graph they build is the one the
    code used to build (148 nodes, 343 relations before any new family)."""
    from kg import family_data
    from kg.seed import FAMILY_CLASS_TO_FAILURE_MODE, build_seed

    fams = family_data.by_family()
    assert next(iter(fams)) == "pump" and {"pump", "gearbox"} <= set(fams)
    g = build_seed()
    ids = {n.id for n in g.nodes}
    assert len(ids) == len(g.nodes)                           # no id defined twice
    assert all(t.subject in ids for t in g.triples)
    for name, f in fams.items():
        assert FAMILY_CLASS_TO_FAILURE_MODE[name] == f.classes
        assert all(fm in ids for fm in f.classes.values()), name
        assert all(node in ids for node in f.aliases), name


def test_every_catalogue_bearing_names_its_source():
    from api import machines

    assert {"6205", "6203", "za2115", "6308"} <= set(machines.BEARINGS)
    for key, b in machines.BEARINGS.items():
        assert b["source"] and b["pitch_mm"] > b["ball_mm"] > 0 and b["n_balls"] >= 3, key
