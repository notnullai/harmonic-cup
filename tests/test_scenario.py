"""Scenario tests (HARM-S1-T06).

A scenario carries its own expectations, so these check both that the YAML
loads and that the run actually reaches the states it claims — which is what
makes the demo scenario a regression test rather than a recording.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import analysis as A
from sim.scenario import SCENARIO_DIR, Scenario

# ISO 10816-1 class III bounds, mm/s RMS velocity. The brain header carries these
# too; if the two ever disagree the scenario expectations become meaningless.
ISO_ZONES = [("OK", 1.8), ("WATCH", 4.5), ("WARNING", 11.2), ("CRITICAL", float("inf"))]


def iso_state(rms: float) -> str:
    for name, upper in ISO_ZONES:
        if rms < upper:
            return name
    return "CRITICAL"


def run_states(scenario: Scenario) -> dict[str, float]:
    """First wall-clock second at which each ISO state is reached."""
    first: dict[str, float] = {}
    for frame in scenario.run():
        state = iso_state(A.rms(frame.vibration))
        first.setdefault(state, frame.wall_s)
    return first


def test_all_bundled_scenarios_load():
    paths = sorted(SCENARIO_DIR.glob("*.yaml"))
    assert paths, "no scenarios found"
    for p in paths:
        s = Scenario.load(p)
        assert s.name
        assert s.duration_s > 0
        assert s.dt > 0


def test_scenario_loads_by_bare_name():
    assert Scenario.load("run_to_failure").name == "run_to_failure"


def test_unknown_scenario_raises():
    with pytest.raises(FileNotFoundError):
        Scenario.load("no_such_scenario")


def test_unknown_fault_mode_in_yaml_is_rejected():
    with pytest.raises(ValueError, match="unknown fault mode"):
        Scenario.from_dict({"name": "bad", "faults": [{"mode": "gremlins"}]})


def test_machine_hours_accounts_for_time_compression():
    s = Scenario.load("run_to_failure")
    assert s.time_scale == 300.0
    assert s.machine_hours == pytest.approx(10.0)


def test_run_to_failure_reaches_every_expected_state_in_time():
    """The headline scenario: the machine must walk OK -> WATCH -> WARNING ->
    CRITICAL, each by its declared deadline."""
    scenario = Scenario.load("run_to_failure")
    first = run_states(scenario)
    for expected in scenario.expect:
        if expected.state == "OK":
            continue
        reached = first.get(expected.state)
        assert reached is not None, f"{expected.state} never reached"
        assert reached <= expected.by_wall_s, (
            f"{expected.state} reached at {reached}s, "
            f"expected by {expected.by_wall_s}s"
        )


def test_run_to_failure_states_arrive_in_order():
    scenario = Scenario.load("run_to_failure")
    first = run_states(scenario)
    order = ["OK", "WATCH", "WARNING", "CRITICAL"]
    seen = [first[s] for s in order if s in first]
    assert seen == sorted(seen)


def test_run_to_failure_starts_healthy():
    """If the machine is already out of zone A at the start, every downstream
    threshold is calibrated against a machine that is already failing."""
    scenario = Scenario.load("run_to_failure")
    frames = list(scenario.run())
    assert iso_state(A.rms(frames[0].vibration)) == "OK"
    assert frames[0].label == "healthy"


def test_healthy_soak_never_leaves_zone_a():
    """The false-positive budget is zero (HARM-S3-T05). If the *data* strays
    out of zone A, no amount of engine tuning can hit that target."""
    scenario = Scenario.load("healthy_soak")
    # 1800 windows is slow; a 200-window sample is enough to catch drift.
    sim = scenario.build()
    worst = 0.0
    for _ in range(200):
        worst = max(worst, A.rms(sim.step(scenario.dt).vibration))
    assert worst < 1.8, f"healthy soak reached {worst:.2f} mm/s"


def test_cavitation_onset_is_quiet_then_impulsive():
    scenario = Scenario.load("cavitation_onset")
    sim = scenario.build()
    before = [sim.step(1.0) for _ in range(290)][-1]
    for _ in range(11):
        after = sim.step(1.0)
    assert A.kurtosis(before.vibration) < 4.0
    assert A.kurtosis(after.vibration) > 4.0


def test_multi_fault_scenario_produces_multi_label_frames():
    scenario = Scenario.load("multi_fault")
    labels = {tuple(f.labels) for f in scenario.run()}
    assert any(len(t) >= 2 for t in labels), f"never saw two faults at once: {labels}"


def test_scenarios_are_reproducible():
    a = [f.vibration for f in Scenario.load("run_to_failure").run()]
    b = [f.vibration for f in Scenario.load("run_to_failure").run()]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
