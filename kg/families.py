"""Machine families, as data (profiles/families.json).

A family is a trained network and what its class neurons mean. The engine
emits class scores by index; this is where an index becomes "broken_tooth" for
a gearbox and "cavitation" for a pump.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FAMILIES_PATH = REPO / "profiles" / "families.json"
DEFAULT = "pump"


@lru_cache(maxsize=1)
def families() -> dict[str, dict[str, Any]]:
    return json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))["families"]


def get(name: str | None) -> dict[str, Any]:
    fam = families().get(name or DEFAULT)
    if fam is None:
        raise ValueError(f"unknown machine family {name!r}; known: {sorted(families())}")
    return fam


def class_names(name: str | None) -> tuple[str, ...]:
    return tuple(get(name)["classes"])


def alarm_response(equipment: dict[str, Any]) -> str:
    """The manual section to cite when nothing more specific survives: the
    machine family's alarm-response procedure, or, for a family that has none
    (an unrecognised machine on the generic family), Harmonic's own generic
    guidance — never another family's manual."""
    try:
        fam = get(str(equipment.get("family") or DEFAULT))
    except ValueError:
        fam = get("generic")
    return str(fam.get("alarm_response") or get("generic")["alarm_response"])


def path(name: str | None, key: str) -> Path:
    """A family's file (network, brain, profile), resolved in the repository."""
    return REPO / get(name)[key]
