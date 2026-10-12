"""One diagnosis per fault episode, and again on escalation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from api.bridge import Event
from api.genai_hook import GenAIHook


class CountingPipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def diagnose(self, engine_event: dict[str, Any]) -> Any:
        self.calls.append(engine_event["event_id"])
        lead = engine_event["top_classes"][0]["name"]
        return SimpleNamespace(response={"failure_mode_id": f"fm.{lead}", "confidence": 0.9},
                               provider="fake", fell_back=False)

    def close(self) -> None:
        pass


def _event(n: int, state: str, episode: str, lead: str = "broken_tooth") -> Event:
    return Event(id=f"ev-{n}", tick=n, ts=float(n), state=state, hhi=0.8, rul_hours=10.0,
                 regime="NOMINAL", vib_rms_mm_s=1.0,
                 top_classes=[{"name": lead, "score": 0.9}], top_surprised_features=[],
                 equipment_id="GB-201", episode=episode)


def test_repeats_within_an_episode_reuse_the_diagnosis():
    p = CountingPipeline()
    hook = GenAIHook(p)  # type: ignore[arg-type]
    events = [_event(1, "WARNING", "ep-a"), _event(2, "WARNING", "ep-a"),
              _event(3, "CRITICAL", "ep-a"), _event(4, "CRITICAL", "ep-a"),
              _event(5, "WARNING", "ep-a"),                   # a lower state: no new call
              _event(6, "CRITICAL", "ep-a", lead="tooth_wear"),  # another fault leads
              _event(7, "WARNING", "ep-b")]                   # a new episode
    for e in events:
        assert hook.submit(e)
    assert hook.wait_idle()
    hook.stop()

    assert p.calls == ["ev-1", "ev-3", "ev-6", "ev-7"]
    assert all(e.diagnosis is not None and e.diagnosis_status == "ok" for e in events)
    by_id = {e.id: e for e in events}
    assert by_id["ev-2"].diagnosis_of == "ev-1"
    assert by_id["ev-4"].diagnosis_of == "ev-3" and by_id["ev-5"].diagnosis_of == "ev-3"
    assert by_id["ev-3"].diagnosis_of is None and by_id["ev-7"].diagnosis_of is None
    assert hook.stats()["reused"] == 3


def test_a_failed_diagnosis_is_tried_again():
    class Flaky(CountingPipeline):
        def diagnose(self, engine_event: dict[str, Any]) -> Any:
            if not self.calls:
                self.calls.append(engine_event["event_id"])
                raise RuntimeError("model unavailable")
            return super().diagnose(engine_event)

    p = Flaky()
    hook = GenAIHook(p)  # type: ignore[arg-type]
    first, second = _event(1, "WARNING", "ep-a"), _event(2, "WARNING", "ep-a")
    hook.submit(first)
    hook.submit(second)
    assert hook.wait_idle()
    hook.stop()
    assert first.diagnosis_status == "failed"
    assert second.diagnosis_status == "ok" and second.diagnosis_of is None
