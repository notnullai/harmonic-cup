"""Scenario runner: a reproducible sequence of machine states from a YAML file.

A scenario pins the seed, the time compression, the faults and their timing,
and — crucially — the events the run is *expected* to produce. That last part
is what turns the demo into a test: a scenario can assert the engine reacted
where it should have, not merely that it produced output.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sim.pump_sim import FAULT_MODES, NOMINAL_RPM, Frame, PumpSimulator

SCENARIO_DIR = Path(__file__).parent / "scenarios"


@dataclass
class ExpectedEvent:
    """A state the run should reach, and by when.

    `by_wall_s` is a deadline in signal time, so a scenario's expectations stay
    meaningful whatever time compression it runs at.
    """

    state: str
    by_wall_s: float
    fault: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExpectedEvent:
        return cls(
            state=str(d["state"]).upper(),
            by_wall_s=float(d["by_wall_s"]),
            fault=d.get("fault"),
        )


@dataclass
class Scenario:
    name: str
    description: str = ""
    seed: int = 0
    duration_s: float = 120.0
    dt: float = 1.0
    time_scale: float = 1.0
    rpm: float = NOMINAL_RPM
    fs_vib: float = 25600.0
    load_fraction: float = 1.0   # not `load`: that name is the classmethod
    faults: list[dict[str, Any]] = field(default_factory=list)
    expect: list[ExpectedEvent] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        p = Path(path)
        if not p.exists():
            candidate = SCENARIO_DIR / f"{p.stem}.yaml"
            if not candidate.exists():
                raise FileNotFoundError(f"no scenario at {p} or {candidate}")
            p = candidate
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Scenario:
        faults = list(raw.get("faults", []))
        for f in faults:
            mode = f.get("mode")
            if mode not in FAULT_MODES:
                raise ValueError(f"scenario names unknown fault mode {mode!r}")
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")).strip(),
            seed=int(raw.get("seed", 0)),
            duration_s=float(raw.get("duration_s", 120.0)),
            dt=float(raw.get("dt", 1.0)),
            time_scale=float(raw.get("time_scale", 1.0)),
            rpm=float(raw.get("rpm", NOMINAL_RPM)),
            fs_vib=float(raw.get("fs_vib", 25600.0)),
            load_fraction=float(raw.get("load", 1.0)),
            faults=faults,
            expect=[ExpectedEvent.from_dict(e) for e in raw.get("expect", [])],
        )

    def build(self) -> PumpSimulator:
        sim = PumpSimulator(
            rpm=self.rpm,
            fs_vib=self.fs_vib,
            seed=self.seed,
            load=self.load_fraction,
            time_scale=self.time_scale,
        )
        for f in self.faults:
            sim.inject(
                f["mode"],
                severity=float(f.get("severity", 1.0)),
                onset_s=float(f.get("onset_s", 0.0)),
                ramp_hours=float(f.get("ramp_hours", 0.0)),
            )
        return sim

    def run(self) -> Iterator[Frame]:
        """Yield every frame of the scenario."""
        sim = self.build()
        yield from sim.run(self.duration_s, self.dt)

    @property
    def machine_hours(self) -> float:
        """Machine time the scenario covers, after time compression."""
        return self.duration_s * self.time_scale / 3600.0
