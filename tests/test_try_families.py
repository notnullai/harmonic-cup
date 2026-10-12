"""Try mode under every machine family (HARM-S10-T02).

The family parameters come from each family's own wizard definition, the old
"other" choice is the generic family, the generic family's diagnosis cites
Harmonic's own guidance, and snapshots without time stamps are spaced by the
visitor's interval. The end-to-end case runs every family on one generated
recording and is skipped, like tests/test_try_mode.py's, when the engine or
a family's network is not built.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from api import machines as machines_mod
from api import try_mode
from kg import families
from kg.query import KGSubgraph
from rag.compose import compose
from rag.guard import validate
from rag.pipeline import load_manifest
from rag.prompt import allowed_references


def _needs_engine() -> None:
    from api.rig import find_library

    if find_library() is None:
        pytest.skip("libharmonic_rig not built; run: cmake --build build -j")
    try:
        machines_mod._binary("mint_brain")
        machines_mod._binary("commission")
    except machines_mod.CommissionError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def try_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(try_mode, "TRY_DIR", tmp_path / "try")
    monkeypatch.setattr(try_mode, "KEPT_DIR", tmp_path / "kept")
    for name in ("_RATE", "_RUN_BYTES", "_DISK", "_UPLOADS", "_ACTIVE"):
        monkeypatch.setattr(try_mode, name, {})
    monkeypatch.setattr(try_mode, "_CANCELLED", set())
    monkeypatch.setattr(try_mode, "_INFLIGHT", set())
    return tmp_path / "try"


MACHINE_MANUALS = [d for d in json.loads(
    (families.REPO / "rag" / "manuals" / "manifest.json").read_text(encoding="utf-8"))["docs"]
    if d not in try_mode.HARMONIC_OWN_DOCS]


def _wide_csv(n: int, samples: int = 4096, ts: bool = False) -> bytes:
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        vals = ",".join(f"{v:.2f}" for v in rng.normal(0, 1, samples))
        rows.append(f"{1_700_000_000 + 300 * i},{vals}" if ts else vals)
    return "\n".join(rows).encode()


def _configured(body: dict, n: int = 200, ts: bool = False) -> tuple[try_mode.Run, dict]:
    run = try_mode.new_run("198.51.100.9")
    try_mode.save_file(run.id, "rec.csv", _wide_csv(n, ts=ts))
    ov = try_mode.configure(run.id, {"sample_rate_hz": 20000, "unit": "g",
                                     "format": "wide_csv", **body})
    got = try_mode.get(run.id)
    assert got is not None
    return got, ov


# --- 1. family parameters, from the wizard's own defaults --------------------------------------


def test_family_params_take_the_wizard_defaults_and_say_which():
    for name, fam in families.families().items():
        wiz = fam.get("wizard", {}).get("parameters", [])
        params, assumed = machines_mod.family_params(name, None)
        assert params == {p["key"]: float(p["default"]) for p in wiz}
        assert [p["key"] for p in assumed] == [p["key"] for p in wiz]


def test_a_given_family_param_is_used_and_not_listed_as_assumed():
    params, assumed = machines_mod.family_params("gearbox", {"teeth": "29"})
    assert params["teeth"] == 29 and assumed == []
    with pytest.raises(machines_mod.CommissionError, match="whole number"):
        machines_mod.family_params("gearbox", {"teeth": 2.5})


def test_an_assumed_default_that_cannot_fit_the_speed_is_moved_never_a_given_one():
    bearing = try_mode._generic_bearing()
    params, _ = machines_mod.family_params("motor", None)
    fitted, moved = machines_mod.fit_assumed("motor", params, {"poles", "line_hz"},
                                             2250.0, bearing)
    assert moved is not None and fitted != params
    vals = machines_mod.derived_values("motor", fitted, 2250.0, bearing)
    assert all(0 < vals[k] <= 1000 for k in ("line2x", "pole_pass"))
    # Nothing assumed: nothing moves, and commissioning reports the misfit.
    same, moved = machines_mod.fit_assumed("motor", params, set(), 2250.0, bearing)
    assert same == params and moved is None
    # A default that fits is left alone.
    same, moved = machines_mod.fit_assumed("motor", params, {"poles", "line_hz"},
                                           1500.0, bearing)
    assert same == params and moved is None


def test_configure_keeps_given_params_and_refuses_unknown_ones(try_dir):
    run, _ = _configured({"family": "gearbox", "params": {"teeth": 29}})
    assert run.params == {"teeth": 29.0}
    with pytest.raises(try_mode.TryError, match="unknown machine detail"):
        _configured({"family": "gearbox", "params": {"blades": 8}})
    with pytest.raises(try_mode.TryError, match="whole number"):
        _configured({"family": "fan", "params": {"blades": 999}})


def test_the_report_lists_every_assumed_value(try_dir):
    run, _ = _configured({"family": "motor", "rpm": 2250})
    run.baseline_h = (0.0, 1.0)
    m = try_mode._build_machine(run, try_mode._recording(run))
    text = " ".join(try_mode._assumed(run, m))
    assert "Poles" in text and "Supply frequency" in text and "fits a shaft speed" in text
    assert "Shaft speed" not in text                   # given, so not assumed
    assert "Bearing" in text


# --- 2. one "Other" choice ---------------------------------------------------------------------


def test_the_page_offers_the_registry_s_families_and_no_other():
    assert tuple(families.families()) == try_mode.FAMILIES
    assert "other" not in try_mode.FAMILIES
    assert try_mode.FAMILY_LABELS["generic"] == "Other or unsure — watch for any change only"
    info = try_mode.limits()["family_info"]
    assert info["generic"]["names_faults"] is False and info["pump"]["names_faults"] is True
    assert [p["key"] for p in info["motor"]["parameters"]] == ["poles", "line_hz"]


def test_an_old_other_is_taken_as_generic(try_dir):
    run, _ = _configured({"family": "other"})
    assert run.family == "generic"
    d = json.loads((run.folder / "run.json").read_text())
    d["family"] = "other"                            # as a pre-T02 run.json holds it
    (run.folder / "run.json").write_text(json.dumps(d))
    got = try_mode.get(run.id)
    assert got is not None and got.family == "generic"


# --- 3. the generic family's diagnosis cites Harmonic's own guidance -------------------------------


def _generic_kg() -> KGSubgraph:
    return KGSubgraph(equipment={"id": "TRY-X", "family": "generic"}, candidates=(),
                      citable_ids=(), unmatched_features=("HARMONIC_FEAT_VIB_RMS",),
                      agreement="none")


def test_the_family_alarm_response_is_always_citable():
    assert "HARMONIC-GEN-1" in allowed_references(_generic_kg(), [])


def test_an_unrecognised_change_on_a_generic_machine_passes_the_guard():
    chunks = {c.id: c for c in load_manifest()}
    kg = _generic_kg()
    event = {"equipment_id": "TRY-X", "state": "WATCH", "hhi": 0.6, "unrecognised": True,
             "top_surprised_features": ["HARMONIC_FEAT_VIB_RMS"]}
    response = compose(event, kg, [], chunks)
    assert validate(response, allowed_references(kg, [])).ok
    assert response["references"] == ["HARMONIC-GEN-1"]
    assert "insufficient evidence" not in response["diagnosis"]
    assert "Unrecognised change" in response["diagnosis"]


def test_own_guidance_stays_on_the_public_report_and_manuals_do_not():
    diag = {"retrieved": ["HARMONIC-GEN-1", "P-101-OM-5.2"],
            "response": {"diagnosis": "Unrecognised change", "confidence": 0.5,
                         "actions": [{"step": "Confirm (see HARMONIC-GEN-1)"}],
                         "safety": ["Lock out per P-101-OM-2.1", "Guarding [F-401-OM-2.2]"],
                         "references": ["HARMONIC-GEN-1", "P-101-OM-5.2"]}}
    out = try_mode._without_manual_citations(diag)["response"]
    assert out["references"] == ["HARMONIC-GEN-1"]
    assert out["references_label"] == try_mode.OWN_GUIDANCE_LABEL
    assert "P-101" not in json.dumps(out) and "F-401" not in json.dumps(out)
    assert out["safety"][1] == "Guarding"


# --- 4. time between snapshots -----------------------------------------------------------------


def test_the_visitor_s_interval_spaces_snapshots_without_time_stamps(try_dir):
    _, ov = _configured({"family": "pump", "interval_s": 60})
    assert ov["timed_by"] == "interval" and ov["interval_s"] == 60
    assert ov["hours"][1] == pytest.approx(1 / 60, abs=1e-4)
    assert ov["hours"][-1] == pytest.approx(199 / 60, abs=1e-3)


def test_a_reconfigure_takes_the_new_interval(try_dir):
    run, _ = _configured({"family": "pump", "interval_s": 60})
    ov = try_mode.configure(run.id, {"family": "pump", "sample_rate_hz": 20000,
                                     "unit": "g", "format": "wide_csv", "interval_s": 1})
    assert ov["hours"][-1] == pytest.approx(199 / 3600, abs=1e-3)


def test_a_time_stamp_column_wins_over_the_interval(try_dir):
    _, ov = _configured({"family": "pump", "interval_s": 60, "has_timestamp": True}, ts=True)
    assert ov["timed_by"] == "timestamps"
    assert ov["hours"][-1] == pytest.approx(199 * 300 / 3600, abs=1e-3)


@pytest.mark.parametrize("bad", [0, -5, "ten", 31 * 86400])
def test_an_impossible_interval_is_refused(try_dir, bad):
    with pytest.raises(try_mode.TryError, match="time between snapshots"):
        _configured({"family": "pump", "interval_s": bad})


# --- every family, end to end --------------------------------------------------------------------


@pytest.mark.parametrize("family", try_mode.FAMILIES)
def test_every_family_runs_end_to_end(try_dir, monkeypatch, family):
    _needs_engine()
    if not families.path(family, "network").exists():
        pytest.skip(f"the {family} network is not built")
    import queue

    monkeypatch.setattr(try_mode, "_QUEUE", queue.Queue(maxsize=try_mode.QUEUE_DEPTH))
    from sim.pump_sim import PumpSimulator

    sim = PumpSimulator(seed=7, fs_vib=25600.0)
    window = 16384
    n_baseline, n_fault = 200, 60
    frames = [sim.step(window / 25600.0).vibration.astype(np.float32)
              for _ in range(n_baseline)]
    sim.inject("bearing_wear", severity=1.0, onset_s=0.0, ramp_hours=0.0)
    frames += [sim.step(window / 25600.0).vibration.astype(np.float32)
               for _ in range(n_fault)]
    buf = io.BytesIO()
    np.save(buf, np.stack(frames))

    run = try_mode.new_run("198.51.100.4")
    try_mode.save_file(run.id, "run.npy", buf.getvalue())
    ov = try_mode.configure(run.id, {"family": family, "sample_rate_hz": 25600,
                                     "unit": "g", "format": "auto", "interval_s": 60})
    try_mode.start(run.id, {"baseline_h": [ov["hours"][0], ov["hours"][n_baseline - 1]],
                            "consent": False})
    got = try_mode.get(run.id)
    assert got is not None
    try_mode._do_run(got)
    got = try_mode.get(run.id)
    assert got is not None and got.status == "done", got and got.error

    report = got.report
    text = json.dumps(report)
    assert report["false_alarms_in_baseline"] == 0, report
    assert not any(doc in text for doc in MACHINE_MANUALS), text
    assert "P-101" not in text
    assumed = " ".join(report["assumed"])
    assert "Shaft speed" in assumed                      # no rpm was given
    for p in families.get(family).get("wizard", {}).get("parameters", []):
        assert p["label"] in assumed
    assert report["timed_by"] == "interval" and report["interval_s"] == 60
    if report["diagnosis"] and not report["names_faults"]:
        resp = report["diagnosis"]["response"]
        assert resp["references"] == ["HARMONIC-GEN-1"]
        assert "insufficient evidence" not in resp["diagnosis"]
