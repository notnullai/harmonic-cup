"""Diagnosis on WARNING+, off the engine thread (HARM-S5-T04).

The engine ticks once a second and a diagnosis takes a graph traversal, a
retrieval and — when a model is configured — a network round trip. Doing that
inline would stall the tick that produced it, so the hook hands the event to a
worker and the engine carries on.

One worker, not a pool. The pipeline is not the bottleneck and a pool would let
two diagnoses for the same machine overlap, which is confusing to read in the
audit log for no gain.

Results are cached by event id, so a dashboard polling `/events` does not
re-run the pipeline on every refresh. The cache is the event record itself:
`Event.diagnosis` is written once and read many times.

While a fault lasts the engine re-raises its alarm every cooldown. Each of
those is not a new question, so the diagnosis is made once per episode and
leading fault, and made again only when the state escalates (WARNING to
CRITICAL) or another fault takes the lead. The repeats carry the earlier
answer and name it in `diagnosis_of`. Without this a fault left in costs a
model call every few seconds, and the event list fills with the same answer.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

from api.bridge import Event
from rag.pipeline import Pipeline

log = logging.getLogger("harmonic.genai")

# States that are worth a diagnosis. WATCH is a dashboard trend, not a call-out
# (docs/METRICS.md), and diagnosing every WATCH would fill the audit log with
# answers nobody asked for.
TRIGGER_STATES = frozenset({"WARNING", "CRITICAL", "FAILED"})

QUEUE_DEPTH = 32
RANK = {"WARNING": 1, "CRITICAL": 2, "FAILED": 3}
# Episodes remembered; the oldest are forgotten first.
EPISODES = 256


def _lead(event: Event) -> str:
    return str(event.top_classes[0].get("name", "")) if event.top_classes else ""


class GenAIHook:
    """Turns raised events into validated diagnoses, in the background."""

    def __init__(self, pipeline: Pipeline | None = None, *, start: bool = True,
                 on_done: Callable[[Event], object] | None = None):
        self._pipeline = pipeline
        # Called with each event once its diagnosis is settled, whatever the
        # outcome: the event store keeps it.
        self.on_done = on_done
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()          # guards lazy pipeline construction
        # A separate lock for the counter. Building the pipeline opens the
        # graph store and the index and takes seconds; sharing one lock would
        # block the engine thread's submit() for that whole time.
        self._count_lock = threading.Lock()
        self._done = threading.Event()
        self.processed = 0
        self.dropped = 0
        self.failed = 0
        # Items accepted but not yet finished. `queue.unfinished_tasks` would
        # need a matching task_done() on every path including the failure
        # one, and a counter that is wrong only on failures is worse than no
        # counter at all.
        self._inflight = 0
        self.reused = 0
        # (episode, leading fault) -> the event whose diagnosis stands for it.
        # Touched by the worker thread only.
        self._answered: dict[tuple[str, str], Event] = {}
        if start:
            self.start()

    # -- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="harmonic-genai",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout)
        self._thread = None
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        """Built on first use.

        Opening the graph store and the index takes a second or two, and doing
        it at import time would make the API slow to start for a capability
        that may never be exercised in a given run.
        """
        with self._lock:
            if self._pipeline is None:
                self._pipeline = Pipeline()
            return self._pipeline

    # -- the hook

    def submit(self, event: Event) -> bool:
        """Queue an event. Returns False when it was not accepted."""
        if event.state not in TRIGGER_STATES:
            event.diagnosis_status = "not_triggered"
            return False
        if event.diagnosis is not None or event.diagnosis_status == "queued":
            return False
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Dropping is better than blocking the engine thread. The event is
            # still in the log with its evidence; only the narrative is missing,
            # and the status says so rather than leaving it "pending" for ever.
            self.dropped += 1
            event.diagnosis_status = "dropped"
            event.diagnosis_error = "the diagnosis queue was full"
            log.warning("genai queue full, dropped %s", event.id)
            return False
        with self._count_lock:
            self._inflight += 1
        event.diagnosis_status = "queued"
        return True

    __call__ = submit

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._done.set()
                return
            try:
                self._diagnose(item)
                if self.on_done is not None:
                    self.on_done(item)
            except Exception:
                log.exception("after diagnosing %s", item.id)
            finally:
                with self._count_lock:
                    self._inflight -= 1

    def _diagnose(self, event: Event) -> None:
        key = (event.episode, _lead(event))
        prior = self._answered.get(key) if event.episode else None
        if (prior is not None and prior.diagnosis is not None
                and RANK.get(event.state, 0) <= RANK.get(prior.state, 0)):
            event.diagnosis = prior.diagnosis
            event.diagnosis_provider = prior.diagnosis_provider
            event.diagnosis_status = prior.diagnosis_status
            event.diagnosis_of = prior.diagnosis_of or prior.id
            self.reused += 1
            return
        self._diagnose_now(event)
        if event.episode and event.diagnosis is not None:
            self._answered.pop(key, None)
            self._answered[key] = event
            while len(self._answered) > EPISODES:
                self._answered.pop(next(iter(self._answered)))

    def _diagnose_now(self, event: Event) -> None:
        event.diagnosis_status = "running"
        try:
            result = self.pipeline.diagnose(event.as_engine_event())
        except Exception as exc:
            # take the service with it. The event keeps its evidence.
            self.failed += 1
            event.diagnosis_status = "failed"
            event.diagnosis_error = f"{type(exc).__name__}: {exc}"
            log.exception("diagnosis failed for %s", event.id)
            return
        event.diagnosis = result.response
        event.diagnosis_provider = result.provider
        event.diagnosis_status = "fallback" if result.fell_back else "ok"
        self.processed += 1

    # -- for tests and /health

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until everything accepted has been diagnosed."""
        tick = threading.Event()
        step = 0.05
        waited = 0.0
        while waited < timeout:
            with self._count_lock:
                if self._inflight == 0:
                    return True
            tick.wait(step)
            waited += step
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "queued": self._queue.qsize(),
            "in_flight": self._inflight,
            "processed": self.processed,
            "reused": self.reused,
            "dropped": self.dropped,
            "failed": self.failed,
            "running": self._thread is not None and self._thread.is_alive(),
            "provider": self._pipeline.provider_name if self._pipeline else "not started",
        }
