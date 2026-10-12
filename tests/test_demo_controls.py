"""The console's plain-language evidence and the demo's fault pause."""

from __future__ import annotations

import pytest

from api.sources import FaultGrowth
from kg.load import build_graph
from kg.query import feature_meanings
from kg.store import NetworkXStore
from sim.fan_sim import FanSimulator


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = NetworkXStore(tmp_path_factory.mktemp("g") / "g.json", autoload=False)
    s.merge_graph(build_graph())
    return s


def test_a_feature_means_its_own_familys_faults(store):
    m = feature_meanings(store, "F-401", ["HARMONIC_FEAT_VIB_ORDER:half", "HARMONIC_FEAT_VIB_BPFO"])
    half = [p["id"] for p in m["HARMONIC_FEAT_VIB_ORDER:half"]["up"]["points_to"]]
    assert half == ["fan.fm.mechanical_looseness"]
    # BPFO is in every family's graph; on the fan it means the fan's bearing only.
    bpfo = [p["id"] for p in m["HARMONIC_FEAT_VIB_BPFO"]["up"]["points_to"]]
    assert bpfo and all(i.startswith("fan.") for i in bpfo)
    pump = feature_meanings(store, "P-101", ["HARMONIC_FEAT_VIB_BPFO"])
    assert all(not p["id"].startswith("fan.")
               for p in pump["HARMONIC_FEAT_VIB_BPFO"]["up"]["points_to"])


class _Demo(FaultGrowth):
    def __init__(self) -> None:
        self.sim = FanSimulator(seed=1)


def test_a_paused_fault_holds_then_resumes_where_it_was():
    d = _Demo()
    d.sim.inject("looseness", severity=1.0, onset_s=0.0, ramp_hours=1.0)
    d.sim.t_s = 900.0                                  # a quarter of the ramp
    d.hold_faults()
    assert d.growth() == {"targets": {"looseness": 1.0}, "held": True}
    d.sim.t_s = 5000.0                                 # time passes; the level does not
    assert d.sim.severities()["looseness"] == pytest.approx(0.25)
    d.resume_faults()
    assert d.sim.severities()["looseness"] == pytest.approx(0.25)
    d.sim.t_s += 900.0                                 # and it carries on at the same rate
    assert d.sim.severities()["looseness"] == pytest.approx(0.5)
    assert d.growth()["held"] is False
