"""Plain names for the engine's features (profiles/feature_labels.json).

The console's evidence and the chat's prompt both speak of "vibration every
other turn", not of HARMONIC_FEAT_VIB_ORDER:half: an engineer reads the first,
and a language model given the second repeats it to them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "profiles" / "feature_labels.json"


@lru_cache(maxsize=1)
def _all() -> dict[str, dict[str, str]]:
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def labels(family: str) -> dict[str, str]:
    """Feature kind without HARMONIC_FEAT_ -> plain name, for one family."""
    d = _all()
    return {**d.get("common", {}), **d.get(family, {})}


def label(kind: str, family: str) -> str:
    """A feature's plain name; its bare kind when the file has none."""
    short = kind.removeprefix("HARMONIC_FEAT_")
    return labels(family).get(short) or short.replace("_", " ").lower()
