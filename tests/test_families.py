"""Machine families (HARM-S7-T05): the class names the console shows are the
ones the network was trained on."""

from __future__ import annotations

import importlib

from kg import families


def test_every_family_names_its_simulator_s_classes_in_training_order():
    for name, fam in families.families().items():
        sim = importlib.import_module(fam["simulator"])
        assert families.class_names(name) == ("healthy", *sim.FAULT_MODES), name
        assert families.path(name, "profile").exists(), name


def test_the_pump_family_is_what_the_bridge_always_used():
    from api.bridge import CLASS_NAMES

    assert families.class_names(None) == CLASS_NAMES


def test_the_build_script_trains_each_family_with_the_same_names():
    from scripts.build_brain import family

    for name in families.families():
        assert tuple(family(name)["classes"]) == families.class_names(name)


def test_the_generic_family_cites_its_own_guidance_not_the_pump():
    """HARM-S10-T01 added "generic" with no manual of its own family; its
    alarm-response citation must be Harmonic's own generic guidance section,
    never the pump's `P-101-OM-5.12` (which used to be the silent default for
    any family without an `alarm_response`)."""
    assert families.get("generic")["alarm_response"] == "HARMONIC-GEN-1"
    assert families.alarm_response({"family": "generic"}) == "HARMONIC-GEN-1"
    # A family name the registry does not know is exactly the case the old
    # fallback existed for, and it must land on the same safe id, not the
    # pump's.
    assert families.alarm_response({"family": "no-such-family"}) == "HARMONIC-GEN-1"


def test_a_new_family_is_data_and_diagnoses_with_its_own_manual(tmp_path, monkeypatch):
    """HARM-S7-T10: the motor and the fan were added as a simulator module, a
    profile, a graph file and a manual. Each fault's canned event is diagnosed
    as that family's failure mode, cited only from that family's manual."""
    monkeypatch.setenv("HARMONIC_LLM", "offline")
    from kg.load import build_graph
    from kg.seed import FAMILY_CLASS_TO_FAILURE_MODE
    from kg.store import NetworkXStore
    from rag.pipeline import Pipeline

    store = NetworkXStore(tmp_path / "g.json", autoload=False)
    store.merge_graph(build_graph())
    p = Pipeline(store=store, llm=None, audit_path=tmp_path / "audit.jsonl")
    lead_feature = {
        "motor": {"bearing_wear": "VIB_BPFO", "rotor_bar": "VIB_ORDER:rotor_sb",
                  "stator_winding": "VIB_ORDER:line2x", "unbalance": "VIB_HARM_1X"},
        "fan": {"unbalance": "VIB_HARM_1X", "blade_damage": "VIB_ORDER:bpf_sb",
                "looseness": "VIB_ORDER:x3", "bearing_wear": "VIB_BPFO"},
    }
    for fam, faults in lead_feature.items():
        f = families.get(fam)
        assert families.class_names(fam)[1:] == tuple(faults), fam
        for cls, feature in faults.items():
            ev = {"event_id": f"{fam}-{cls}", "equipment_id": f["demo"]["equipment_id"],
                  "state": "WARNING", "hhi": 0.7, "rul_hours": 90, "regime": "NOMINAL",
                  "top_classes": [{"name": cls, "score": 0.9}],
                  "top_surprised_features": [{"kind": f"HARMONIC_FEAT_{feature}",
                                              "direction": "up", "weight": 1.0}]}
            r = p.diagnose(ev)
            assert not r.fell_back, (fam, cls)
            assert r.response["failure_mode_id"] == FAMILY_CLASS_TO_FAILURE_MODE[fam][cls]
            refs = r.response["references"]
            assert refs and all(ref.startswith(f["manuals"][0]) for ref in refs), (fam, cls, refs)
