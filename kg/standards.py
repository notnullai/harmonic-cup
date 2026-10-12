"""Vibration severity standards: ISO zone tables, as data.

The tables live in `profiles/vibration_standards.json`, next to the machine
profiles, so a standard is added by editing data — the console, the
commissioning wizard and the diagnosis prompt all read it from here.

Keys look like "10816-1-III" or "10816-3-G2-rigid". Equipment seeded before
the table existed stores a bare ISO 10816-1 roman numeral ("III"); that still
resolves.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

STANDARDS_PATH = Path(__file__).resolve().parents[1] / "profiles" / "vibration_standards.json"
ROMAN = ("I", "II", "III", "IV")


@lru_cache(maxsize=1)
def classes() -> dict[str, dict[str, Any]]:
    """Every class: {key: {"label", "short", "bounds"}}; bounds are A/B, B/C,
    C/D and the top of D in mm/s RMS, or None for "no zones"."""
    raw = json.loads(STANDARDS_PATH.read_text(encoding="utf-8"))["classes"]
    return {k: {**v, "bounds": tuple(v["bounds"]) if v["bounds"] else None}
            for k, v in raw.items()}


def key_of(iso_class: object) -> str:
    iso = str(iso_class or "none")
    return f"10816-1-{iso}" if iso in ROMAN else iso


def describe(iso_class: object) -> dict[str, Any] | None:
    """The class as the console shows it, or None when it names no zones."""
    key = key_of(iso_class)
    c = classes().get(key)
    if c is None or c["bounds"] is None:
        return None
    return {"key": key, "label": c["label"], "short": c["short"], "bounds": list(c["bounds"])}


def short_label(iso_class: object) -> str:
    """How a reader would write the class: "ISO 10816-1 class III"."""
    c = classes().get(key_of(iso_class))
    return c["short"] if c else "no ISO velocity zones"
