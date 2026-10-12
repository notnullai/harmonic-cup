"""Harmonic API (HARM-S5-T01, T03, T05).

    GET  /health        current state, HHI, RUL, and what is actually running
    GET  /sensors       latest sensor values and the engine's feature view
    GET  /events        raised events, with their diagnosis once it arrives
    GET  /events/{id}   one event
    POST /chat          grounded question answering
    POST /inject        inject a fault, for the demo
    POST /reset         restart the machine and the engine from nothing,
                        optionally switching the data source
    GET  /sources       the data sources that can be selected
    GET  /history       the timeline: every kept reading, compact
    GET  /report.pdf    the health report for a period, as a PDF
    GET  /machines      commissioned machines, and what the wizard offers
    POST /machines      commission a new machine from a recording
    GET  /recordings    recordings on the server, uploads included
    POST /uploads       start an upload; PUT /uploads/{id}/{file} adds a snapshot
    GET  /              the web app (web/dist), when it has been built
    WS   /ws/stream     the same snapshots at the tick rate
    POST /try/runs      "Try Harmonic on your own data" — see api/try_mode.py

Try mode is **off by default**. `HARMONIC_MODE=try` runs this same image as
the public, anonymous try-mode service (`harmonic-try`, not deployed until
the S10-T03 security review passes): the fleet, the console's own routes and
Gemini are never started, and everything but `/try/*` and the static `/try`
page answers 404. With `HARMONIC_MODE` unset the console is exactly as
before and every `/try` path answers 404, unless `HARMONIC_TRY=1` switches
the try routes on beside the console (local development only).

Every endpoint works with the engine absent. It reports `engine.available:
false` and a reason, rather than serving zeros — a health index of 0.0 renders
as a machine in perfect condition, which is the most dangerous thing this
service could display.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from api import bridge as bridge_mod
from api import live as live_mod
from api import machines as machines_mod
from api import manuals as manuals_mod
from api import sources as sources_mod
from api import try_mode as try_mode_mod
from api.bridge import BridgeStatus, EngineBridge, Event
from api.fleet import Fleet
from api.genai_hook import GenAIHook
from api.store import EventStore, open_stores
from api.workorders import WorkOrders
from kg import families

log = logging.getLogger("harmonic.api")

STREAM_PERIOD_S = float(os.environ.get("HARMONIC_STREAM_PERIOD", "1.0"))
AUTOSTART = os.environ.get("HARMONIC_AUTOSTART", "1") not in ("0", "false", "no")
# "console" (default): the full operator console, unchanged. "try": the
# public, anonymous "try Harmonic on your own data" service — see the module
# docstring above and api/try_mode.py.
#
# Only these two, or unset (= console). Anything else — a typo like "tyr" —
# refuses to start rather than quietly serving the console to the public
# (HARM-S10-T03 M4).
_MODE_RAW = os.environ.get("HARMONIC_MODE")
MODE = "console" if _MODE_RAW is None else _MODE_RAW.strip().lower()
if MODE not in ("console", "try"):
    raise SystemExit(f"HARMONIC_MODE={_MODE_RAW!r} is not a mode Harmonic knows: leave it "
                     "unset (the console), or set it to 'console' or 'try'. Refusing to start.")
# The try routes beside the console, for local development. Off unless set.
TRY_BESIDE_CONSOLE = os.environ.get("HARMONIC_TRY", "0").strip().lower() in ("1", "true", "yes")


def _try_enabled() -> bool:
    """Read at call time, so the gate, the lifespan and tests agree."""
    return MODE == "try" or TRY_BESIDE_CONSOLE


class State:
    """Process-wide handles: the fleet of engines, one hook, one pipeline, the store."""

    status: BridgeStatus = BridgeStatus(False, "not started")
    hook: GenAIHook | None = None
    # Every machine, each with its own engine (api/fleet.py); the console
    # looks at the fleet's current one. A test may attach one bridge instead.
    fleet: Fleet | None = None
    store: EventStore | None = None
    _attached: EngineBridge | None = None

    @property
    def bridge(self) -> EngineBridge | None:
        return self.fleet.current if self.fleet is not None else self._attached

    @bridge.setter
    def bridge(self, b: EngineBridge | None) -> None:
        self.fleet = None
        self._attached = b


state = State()


# --- request models -----------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)


class InjectRequest(BaseModel):
    mode: str
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    ramp_minutes: float = Field(default=5.0, ge=0.0, le=1440.0)
    # Stop the fault growing once the machine reaches this state (the demo).
    pause_at: str | None = Field(default=None, pattern="^(WATCH|WARNING|CRITICAL)$")


# --- lifespan -----------------------------------------------------------------


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # try-mode's own worker and sweeper (api/try_mode.py): a single-flight
    # queue for commissioning a run, and hourly deletion of expired ones.
    # Started only when try mode is on; nothing there touches the fleet, the
    # store or Gemini.
    if _try_enabled():
        try_mode_mod.start_background()
        if not os.environ.get("HARMONIC_CONTACT_URL"):
            log.warning("try mode: HARMONIC_CONTACT_URL is unset, so the report hides "
                        "its 'talk to us' link")
    if AUTOSTART and MODE != "try":
        state.store, wo_store = open_stores(machines_mod.DATA_DIR)
        state.hook = GenAIHook(on_done=lambda e: state.fleet.diagnosed(e)
                               if state.fleet is not None else None)
        state.fleet = Fleet(state.store, state.hook.submit)
        state.fleet.workorders = WorkOrders(wo_store)
        state.fleet.start("simulated")
        state.status = state.fleet.status
        if state.bridge is None:
            log.warning("engine unavailable: %s", state.status.reason)
        # Open the graph and the index now rather than on the first event:
        # built lazily they cost the first diagnosis ~1.5 s, which is the one
        # the audience is watching.
        await asyncio.to_thread(lambda: state.hook.pipeline if state.hook else None)
        # Manuals uploaded for commissioned machines join the index.
        if state.hook is not None:
            await asyncio.to_thread(state.hook.pipeline.add_chunks, manuals_mod.load_all())
        # Machines commissioned earlier are put back into the graph: a rebuilt
        # graph would otherwise forget what components they have.
        for m in machines_mod.list_machines():
            if m.status == "ready":
                _register_machine(m)
    elif MODE == "try":
        state.status = BridgeStatus(False, "try mode: the fleet is not started")
    yield
    if state.fleet is not None:
        state.fleet.stop()
        if state.fleet.workorders is not None:
            state.fleet.workorders.close()
    elif state.bridge is not None:
        state.bridge.stop()
    if state.hook is not None:
        state.hook.stop()
    if state.store is not None:
        state.store.close()
    live_mod.stop_all()


app = FastAPI(
    title="Harmonic",
    version="0.6.0",
    description="Machine health for rotating equipment: a fixed-point SNN on the "
                "edge, a knowledge graph and a grounded diagnosis layer above it.",
    lifespan=lifespan,
)

class _ConsoleOnlyCORS:
    """CORS for the console only. In try mode there is none at all: the page
    is served by the API itself, so no other website's script may read a
    visitor's run or drive the upload (HARM-S10-T03 M5). Checked per request,
    like the mode gate below, so both always agree."""

    def __init__(self, app: ASGIApp, /, **kw: Any):
        self.app = app
        self.cors = CORSMiddleware(app, **kw)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if MODE == "try":
            await self.app(scope, receive, send)
        else:
            await self.cors(scope, receive, send)


app.add_middleware(
    _ConsoleOnlyCORS,
    # The dashboard is served from a different port in compose and from a
    # different host when someone opens it from their laptop. This is a demo
    # service with no authentication and no mutating endpoints beyond fault
    # injection into a simulator, so the permissive setting costs nothing.
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# In try mode, every route except /try/* and the static pages the browser
# needs to render it answers 404 — no fleet, no machine list, no chat, no
# other user's data, whatever a future route forgets to check for itself.
# Checked on the path alone, before any handler (including auth-free ones
# like /health) runs, so a route added later is safe by default rather than
# by remembering to opt out.
_TRY_MODE_PATH_PREFIXES = ("/try/", "/assets/")
_TRY_MODE_EXACT_PATHS = frozenset({"/", "/favicon.svg", "/try"})


def _try_mode_path_ok(path: str) -> bool:
    """A path try mode serves: under /try/ or /assets/, or one of a few
    exact ones — and never with a "." or ".." segment or a backslash, so
    `/try/../index.html` cannot reach the console's own files through the
    static mount (HARM-S10-T03 L3)."""
    if "\\" in path or any(seg in (".", "..") for seg in path.split("/")):
        return False
    return path in _TRY_MODE_EXACT_PATHS or path.startswith(_TRY_MODE_PATH_PREFIXES)


class _NoWebsocketsInTryMode:
    """The console's live stream (`/ws/stream`) and any websocket added later
    are closed before the handshake in try mode; HTTP middleware never sees a
    websocket, so the mode gate below cannot (HARM-S10-T03 L3)."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket" and MODE == "try":
            await send({"type": "websocket.close", "code": 1008})
            return
        await self.app(scope, receive, send)


app.add_middleware(_NoWebsocketsInTryMode)


@app.middleware("http")
async def _mode_gate(request: Request, call_next):
    path = request.url.path
    if MODE == "try":
        if not _try_mode_path_ok(path):
            return JSONResponse(status_code=404, content={
                "detail": "this server runs in try mode; only /try is served"})
    elif not _try_enabled() and (path == "/try" or path.startswith("/try/")):
        # Off by default: the console serves no try route and no try page
        # (the static mount would otherwise serve dist/try/index.html).
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return await call_next(request)


_RULED_OUT: dict[str, dict[str, str]] = {}


def _ruled_out(b: EngineBridge) -> dict[str, str]:
    """Classes the graph rules out for this machine: class -> the part it lacks.
    The graph does not change while the API runs, so asked once per machine."""
    eq = b.source.equipment_id
    if eq not in _RULED_OUT:
        if state.hook is None:
            return {}
        from kg.query import classes_ruled_out
        try:
            _RULED_OUT[eq] = classes_ruled_out(state.hook.pipeline.store, eq)
        except Exception as exc:
            log.warning("could not ask the graph about %s: %s", eq, exc)
            return {}
    return _RULED_OUT[eq]


def _engine_block() -> dict[str, Any]:
    b = state.bridge
    return {
        "available": b is not None,
        "reason": state.status.reason,
        **(b.info() if b is not None else {}),
        "ruled_out_classes": _ruled_out(b) if b is not None else {},
    }


def _require_bridge() -> EngineBridge:
    if state.bridge is None:
        raise HTTPException(
            status_code=503,
            detail=f"the engine is not running: {state.status.reason}",
        )
    return state.bridge


# --- endpoints ----------------------------------------------------------------


_MEANINGS: dict[str, dict[str, Any]] = {}


def _ratio_scaled(kind: str) -> bool:
    """Whether the engine log2-scales this feature, so that its surprise is a
    ratio (engine/src/harmonic_features.cpp): vibration amplitudes are; the
    dimensionless kurtosis and crest factor, and the slow process channels in
    their physical units, are linear."""
    k = kind.removeprefix("HARMONIC_FEAT_")
    return k.startswith("VIB_") and k.split(":")[0] not in ("VIB_KURTOSIS", "VIB_CREST")


@app.get("/evidence", summary="What each feature of the selected machine means, in plain words")
def evidence() -> dict[str, Any]:
    """For the diagnosis panel: each feature's plain label, how its change
    reads (a ratio or a difference in its unit), and — from the graph — the
    symptom it is and the faults it is typical of on this machine."""
    b = _require_bridge()
    eq = b.source.equipment_id
    family = str(getattr(b.source, "family", None) or families.DEFAULT)
    kinds, units = list(b.rig.feature_kinds), list(b.rig.feature_units)
    if eq not in _MEANINGS and state.hook is not None:
        from kg.query import feature_meanings
        try:
            _MEANINGS[eq] = feature_meanings(state.hook.pipeline.store, eq, kinds)
        except Exception as exc:
            log.warning("could not ask the graph about %s: %s", eq, exc)
    meanings = _MEANINGS.get(eq, {})
    from api.feature_labels import labels
    names = labels(family)
    out: dict[str, Any] = {}
    for kind, unit in zip(kinds, units, strict=False):
        short = kind.removeprefix("HARMONIC_FEAT_")
        out[kind] = {"label": names.get(short), "unit": unit,
                     "scale": "ratio" if _ratio_scaled(kind) else "linear",
                     **meanings.get(kind, {})}
    return {"equipment_id": eq, "family": family, "features": out}


def _kg_block() -> dict[str, Any]:
    """Node/relation counts and backend, whatever backend is in play
    (`KGStore.counts()` and `.backend` are the same on NetworkX and Neo4j —
    HARM-S9-T08). Built lazily, same as the pipeline itself; if that build
    hasn't happened yet or fails, report it rather than raising."""
    if state.hook is None:
        return {"nodes": 0, "relations": 0, "backend": None, "empty": True}
    try:
        store = state.hook.pipeline.store
        n_nodes, n_rels = store.counts()
        return {"nodes": n_nodes, "relations": n_rels, "backend": store.backend,
                "empty": n_nodes == 0}
    except Exception as exc:
        log.warning("could not read the knowledge graph's counts: %s", exc)
        return {"nodes": 0, "relations": 0, "backend": None, "empty": True,
                "error": str(exc)}


@app.get("/health", summary="Current health state")
def health() -> dict[str, Any]:
    latest = state.bridge.latest if state.bridge is not None else None
    return {
        "ok": True,
        "engine": _engine_block(),
        "genai": state.hook.stats() if state.hook is not None else {"running": False},
        "kg": _kg_block(),
        "machine": None if latest is None else {
            "state": latest.state,
            "iso_state": latest.iso_state,
            "hhi": latest.hhi,
            "hhi_fast": latest.hhi_fast,
            "hhi_slow": latest.hhi_slow,
            "slope_per_tick": latest.slope_per_tick,
            "rul_hours": latest.rul_hours,
            "regime": latest.regime,
            "warm": latest.warm,
            "vib_rms_mm_s": latest.vib_rms_mm_s,
            "vib_rms": latest.vib_rms,
            "vib_unit": latest.vib_unit,
            "unrecognised": latest.unrecognised,
            "early": latest.early,
            "spread": latest.spread,
            "recorded_at": latest.recorded_at,
            "tick": latest.tick,
            "machine_hours": latest.machine_hours,
            "injected": latest.injected,
            "fault_growth": state.bridge.fault_growth() if state.bridge is not None else None,
        },
    }


@app.get("/sensors", summary="Latest sensor and feature values")
def sensors(history: int = 0) -> dict[str, Any]:
    b = _require_bridge()
    latest = b.latest
    if latest is None:
        return {"tick": 0, "sensors": {}, "waiting": "no frame has been processed yet"}
    payload: dict[str, Any] = {
        "tick": latest.tick,
        "ts": latest.ts,
        "sensors": latest.sensors,
        "features": latest.features,
        "surprise": latest.surprise,
        "class_scores": latest.class_scores,
        "units": dict(zip(b.rig.feature_kinds, b.rig.feature_units, strict=True)),
    }
    if history > 0:
        payload["history"] = [
            {"tick": s.tick, "ts": s.ts, "hhi": s.hhi, "state": s.state,
             "machine_hours": s.machine_hours, "sensors": s.sensors}
            for s in b.history(min(history, 1000))
        ]
    return payload


@app.get("/history", summary="The timeline: every kept reading, compact")
def history(since_tick: int = 0, limit: int = 4000) -> dict[str, Any]:
    """What the timeline draws and scrubs through. Compact on purpose — the
    whole IMS week is 984 of these — and incremental: pass the last tick seen
    and only newer readings come back."""
    b = _require_bridge()
    points = [s for s in b.history(bridge_mod.HISTORY) if s.tick > since_tick][-limit:]
    return {
        "source": b.source_key,
        # Where the brain was changed: a re-baseline after maintenance.
        "marks": list(b.marks),
        "points": [{
            "tick": s.tick, "h": s.machine_hours, "at": s.recorded_at,
            "state": s.state, "hhi": s.hhi, "vib": s.vib_rms, "vel": s.vib_rms_mm_s,
            "rul": s.rul_hours,
            "regime": s.regime, "warm": s.warm, "sensors": s.sensors,
            "top": s.top_classes[:1], "inj": s.injected,
            **({"unr": True} if s.unrecognised else {}),
            **({"early": True} if s.early else {}),
        } for s in points],
    }


@app.get("/report.pdf", summary="The health report for a period, as a PDF")
def report_pdf(from_h: float | None = Query(None, description="machine hours, inclusive"),
               to_h: float | None = Query(None, description="machine hours, inclusive")
               ) -> Response:
    from api.report import build_report

    b = _require_bridge()
    info = b.info()
    pdf = build_report(history=b.history(bridge_mod.HISTORY), events=b.events(bridge_mod.EVENT_LOG),
                       source=info["source"], engine=info, from_h=from_h, to_h=to_h)
    name = f"harmonic-{b.source.equipment_id.lower()}-report.pdf"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/events", summary="Raised events and their diagnoses")
def events(limit: int = 20) -> dict[str, Any]:
    b = _require_bridge()
    return {"events": [e.to_json() for e in b.events(min(limit, 200))]}


@app.get("/events/stored", summary="Events kept across restarts, newest first")
def stored_events(source: str | None = None, limit: int = 50) -> dict[str, Any]:
    if state.store is None:
        raise HTTPException(status_code=503, detail="the event store is not running")
    return {"events": state.store.recent(source, min(limit, 500))}


@app.get("/events/{event_id}", summary="One event")
def event(event_id: str) -> dict[str, Any]:
    b = _require_bridge()
    found = b.event(event_id) or (state.fleet.find_event(event_id) if state.fleet else None)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no event {event_id}")
    return found.to_json()


class FeedbackRequest(BaseModel):
    verdict: str = Field(pattern="^(confirmed|wrong)$")
    actual: str | None = None
    note: str = Field(default="", max_length=1000)


@app.post("/events/{event_id}/feedback",
          summary="What the technician found: the diagnosis confirmed, or what it was")
def event_feedback(event_id: str, req: FeedbackRequest) -> dict[str, Any]:
    if state.store is None:
        raise HTTPException(status_code=503, detail="the event store is not running")
    live = state.fleet.find_event(event_id) if state.fleet else None
    stored = next((e for e in state.store.recent(limit=5000) if e["id"] == event_id), None)
    if live is None and stored is None:
        raise HTTPException(status_code=404, detail=f"no event {event_id}")
    src = stored["source"] if stored else (state.bridge.source_key if state.bridge else "")
    family = next((b.family for k, b in (state.fleet.bridges.items() if state.fleet else [])
                   if k == src), families.DEFAULT)
    try:
        fb = state.store.set_feedback(event_id, req.verdict, req.actual, req.note,
                                      families.class_names(family))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if live is not None:
        live.feedback = fb
    return {"event_id": event_id, "feedback": fb}


@app.get("/feedback/export", summary="Judged events of one machine, as labelled frames")
def feedback_export(source: str) -> Response:
    """The windows of every event a technician has judged, labelled with
    what was actually there, in the frame format `harmonic_sim_words`
    encodes for the trainer: the path from a simulator-trained brain to a
    field-trained one."""
    if state.store is None:
        raise HTTPException(status_code=503, detail="the event store is not running")
    b = state.fleet.bridges.get(source) if state.fleet else None
    family = b.family if b is not None else families.DEFAULT
    data, n = state.store.export_frames(source, families.class_names(family))
    if not n:
        raise HTTPException(status_code=404, detail="no judged events with a window for "
                                                    f"{source!r} yet")
    return Response(content=data, media_type="application/octet-stream", headers={
        "Content-Disposition": f'attachment; filename="{source}-feedback.hframe"',
        "X-Frames": str(n)})


class ListenRequest(BaseModel):
    broker: str = Field(default="localhost:1883", max_length=200)
    topic: str = Field(min_length=1, max_length=200)
    name: str = Field(default="", max_length=80)


@app.post("/live/listen", summary="Record a live MQTT sensor, to commission from it")
def live_listen(req: ListenRequest) -> dict[str, Any]:
    try:
        uid, rec = live_mod.listen(req.broker, req.topic, req.name)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": uid, **rec.status()}


@app.get("/live/listen/{uid}", summary="A listener's progress")
def live_status(uid: str) -> dict[str, Any]:
    rec = live_mod.get(uid)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"no listener {uid}")
    return {"id": uid, **rec.status()}


@app.delete("/live/listen/{uid}", summary="Stop a listener; what it recorded stays")
def live_stop(uid: str) -> dict[str, Any]:
    if not live_mod.stop(uid):
        raise HTTPException(status_code=404, detail=f"no listener {uid}")
    return {"stopped": uid}


def _workorders() -> WorkOrders:
    if state.fleet is None or state.fleet.workorders is None:
        raise HTTPException(status_code=503, detail="work orders are not running")
    return state.fleet.workorders


@app.get("/workorders", summary="Work orders raised for the maintenance system")
def workorders(source: str | None = None, limit: int = 50) -> dict[str, Any]:
    wo = _workorders()
    from urllib.parse import urlparse
    return {"webhook": urlparse(wo.url).netloc if wo.url else None,
            "work_orders": wo.list(source, min(limit, 200))}


@app.post("/workorders/{wo_id}/close", summary="The job is done: the next alarm opens a new one")
def close_workorder(wo_id: str) -> dict[str, Any]:
    if not _workorders().close_order(wo_id):
        raise HTTPException(status_code=404, detail=f"no open work order {wo_id}")
    return {"closed": wo_id}


@app.get("/fleet", summary="Every machine: its state, its last event, most urgent first")
def fleet() -> dict[str, Any]:
    if state.fleet is None:
        raise HTTPException(status_code=503, detail="the fleet is not running")
    return {"current": state.fleet.current_key, "machines": state.fleet.summary()}


class SelectRequest(BaseModel):
    source: str


@app.post("/select", summary="Look at another machine; it keeps its history")
def select(req: SelectRequest) -> dict[str, Any]:
    known = sources_mod.available()
    if req.source not in known:
        raise HTTPException(status_code=422, detail=f"unknown source {req.source!r}")
    if not known[req.source]["available"]:
        raise HTTPException(status_code=409, detail=known[req.source]["reason"])
    if state.fleet is None:
        return _require_bridge().reset(req.source)
    b = state.fleet.select(req.source)
    return {"selected": req.source, "source": b.source.info()}


@app.post("/chat", summary="Ask a question, grounded in the graph and the manual")
def chat(req: ChatRequest) -> dict[str, Any]:
    from api.chat import ask

    if state.hook is None:
        raise HTTPException(status_code=503, detail="the diagnosis layer is not running")
    b = state.bridge
    latest = b.latest if b is not None else None
    snapshot = latest.to_json() if latest is not None else None
    if snapshot is not None and b is not None:
        # Which machine, so the graph and the manuals are its own.
        snapshot["equipment_id"] = b.source.equipment_id
        snapshot["family"] = b.family
    try:
        turn = ask(req.question, state.hook.pipeline, snapshot, k=req.k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return turn.to_json()


@app.post("/inject", summary="Inject a fault into the simulated machine")
def inject(req: InjectRequest) -> dict[str, Any]:
    b = _require_bridge()
    try:
        return b.inject(req.mode, req.severity, req.ramp_minutes, req.pause_at)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class HoldRequest(BaseModel):
    hold: bool = True


@app.post("/inject/hold", summary="Pause (or resume) the injected faults' growth")
def hold_faults(req: HoldRequest) -> dict[str, Any]:
    try:
        return _require_bridge().hold_faults(req.hold)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/inject/clear", summary="Clear injected faults")
def clear_faults() -> dict[str, Any]:
    try:
        return _require_bridge().clear_faults()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ResetRequest(BaseModel):
    source: str | None = None


@app.post("/reset", summary="Restart the machine and the engine from nothing")
def reset(req: ResetRequest | None = None) -> dict[str, Any]:
    b = _require_bridge()
    key = req.source if req else None
    if key is not None:
        known = sources_mod.available()
        if key not in known:
            raise HTTPException(status_code=422, detail=f"unknown source {key!r}")
        if not known[key]["available"]:
            raise HTTPException(status_code=409, detail=known[key]["reason"])
    if state.fleet is not None:
        return state.fleet.reset(key or b.source_key)
    return b.reset(key)


@app.get("/sources", summary="The data sources that can be selected")
def sources() -> dict[str, Any]:
    current = state.bridge.source_key if state.bridge is not None else None
    return {"current": current, "sources": sources_mod.available()}


# --- machines and commissioning --------------------------------------------------


def _machine_id(key: str) -> str:
    """A fleet key names a machine or one of its points ("pump-3@1")."""
    return key.partition("@")[0]


def _register_machine(m: machines_mod.Machine) -> None:
    """The machine's equipment and components into the graph, so a diagnosis
    can rule out what it cannot have."""
    if state.hook is None:
        return
    try:
        state.hook.pipeline.store.merge_graph(machines_mod.kg_graph(m))
    except Exception:
        log.exception("could not add %s to the knowledge graph", m.id)


@app.put("/machines/{machine_id}/manual/{filename}",
         summary="Upload the machine's own manual (Markdown or text), to cite in diagnoses")
async def upload_manual(machine_id: str, filename: str, request: Request) -> dict[str, Any]:
    body = await request.body()
    machine_id = _machine_id(machine_id)
    try:
        saved = manuals_mod.save(machine_id, filename, body)
    except machines_mod.CommissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    chunks = saved.pop("chunks")
    if state.hook is not None:
        await asyncio.to_thread(state.hook.pipeline.add_chunks, chunks)
    m = machines_mod.get(machine_id)
    if m is not None:
        _register_machine(m)
    return saved


@app.get("/machines/{machine_id}/manual", summary="The machine's uploaded manual, if any")
def machine_manual(machine_id: str) -> dict[str, Any]:
    found = manuals_mod.info(_machine_id(machine_id))
    if found is None:
        raise HTTPException(status_code=404, detail=f"no manual for {machine_id}")
    return found


class MachineRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    nominal_rpm: float
    vib_unit: str
    iso_class: str = "none"
    bearing: dict[str, float]
    components: list[str] = ["bearing", "shaft", "foundation"]
    recording: str
    channel: int = 0
    baseline_h: tuple[float, float]
    speed: str = "constant"
    family: str = "pump"
    params: dict[str, float] = {}
    live: dict[str, str] | None = None
    # Measurement points: [{"name": "DE horizontal", "channel": 0}, ...].
    points: list[dict[str, Any]] = []


class UploadRequest(BaseModel):
    name: str = ""
    sample_rate_hz: int = Field(gt=0, le=1_000_000)
    interval_s: float = Field(default=600.0, gt=0)
    unit: str = "g"
    rpm: list[float] | None = None      # shaft speed per snapshot, if measured


@app.get("/machines", summary="Commissioned machines, and what the wizard offers")
def machines() -> dict[str, Any]:
    from kg.model import COMPONENT_KINDS

    return {
        "machines": [m.to_json() for m in machines_mod.list_machines()],
        "iso_classes": {k: v["label"] for k, v in machines_mod.ISO_CLASSES.items()},
        "bearings": machines_mod.BEARINGS,
        "components": sorted(COMPONENT_KINDS),
        "min_baseline_readings": machines_mod.MIN_BASELINE,
        # Families a machine can be commissioned as: one whose network is not
        # built is listed, and says so.
        "families": {k: {"label": f["label"], "wizard": f.get("wizard", {}),
                          "available": families.path(k, "network").exists()}
                     for k, f in families.families().items()},
    }


@app.post("/machines", summary="Commission a new machine from a recording")
def create_machine(req: MachineRequest) -> dict[str, Any]:
    try:
        m = machines_mod.create(req.model_dump())
    except machines_mod.CommissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    machines_mod.start(m.id, register=_register_machine)
    return m.to_json()


class RebaselineRequest(BaseModel):
    baseline_h: tuple[float, float]
    iso_class: str | None = None
    recording: str | None = None
    channel: int | None = None
    note: str = Field(default="", max_length=200)


@app.post("/machines/{machine_id}/rebaseline",
          summary="Re-learn normal after maintenance, from a healthy period")
def rebaseline_machine(machine_id: str, req: RebaselineRequest) -> dict[str, Any]:
    spec = {k: v for k, v in req.model_dump().items() if v is not None}
    machine_id = _machine_id(machine_id)
    try:
        m = machines_mod.rebaseline(machine_id, spec)
    except machines_mod.CommissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def on_ready(done: machines_mod.Machine) -> None:
        _register_machine(done)
        if state.fleet is not None:
            state.fleet.swap_brain(done.id, f"Re-baselined: {req.note}" if req.note
                                   else "Re-baselined", done.iso_class)

    machines_mod.start_rebaseline(m.id, on_ready)
    return m.to_json()


@app.get("/machines/{machine_id}", summary="One machine, with commissioning progress")
def machine(machine_id: str) -> dict[str, Any]:
    m = machines_mod.get(_machine_id(machine_id))
    if m is None:
        raise HTTPException(status_code=404, detail=f"no machine {machine_id}")
    return m.to_json()


@app.delete("/machines/{machine_id}", summary="Remove a commissioned machine")
def delete_machine(machine_id: str) -> dict[str, Any]:
    machine_id = _machine_id(machine_id)
    if state.bridge is not None and _machine_id(state.bridge.source_key) == machine_id:
        raise HTTPException(status_code=409, detail="that machine is on screen; switch "
                                                    "to another before removing it")
    if not machines_mod.delete(machine_id):
        raise HTTPException(status_code=404, detail=f"no machine {machine_id}")
    if state.fleet is not None:
        state.fleet.drop(machine_id)
    return {"deleted": machine_id}


@app.get("/recordings", summary="Recordings on the server, uploads included")
def recordings() -> dict[str, Any]:
    out = []
    for key, r in machines_mod.recordings().items():
        hours = r.hours()
        out.append({
            "key": key, "name": r.name, "provenance": r.provenance, "unit": r.unit,
            "sample_rate_hz": r.sample_rate_hz, "channels": r.n_channels,
            "snapshots": len(hours), "span_h": round(float(hours[-1] - hours[0]), 2)
            if len(hours) else 0.0, "samples": r.snapshot_samples(),
            "has_speed": r.speeds() is not None,
        })
    return {"recordings": out}


@app.get("/recordings/{key}/overview", summary="RMS per snapshot, to pick a baseline")
def recording_overview(key: str, channel: int = 0) -> dict[str, Any]:
    from sim.datasets.recordings import rms, window_for

    try:
        r = machines_mod.recording(key)
        data = r.load(channel, window_for(r.snapshot_samples()))
    except (machines_mod.CommissionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"hours": [round(float(h), 4) for h in r.hours()],
            "at": r.stamps(), "rms": [round(float(v), 6) for v in rms(data)],
            "unit": r.unit}


@app.post("/uploads", summary="Start an upload of recorded snapshots")
def new_upload(req: UploadRequest) -> dict[str, Any]:
    try:
        return {"id": machines_mod.new_upload(req.model_dump())}
    except machines_mod.CommissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/uploads/{upload_id}/{filename}", summary="Add one file to an upload")
async def upload_file(upload_id: str, filename: str, request: Request) -> dict[str, Any]:
    body = await request.body()
    if len(body) > 256 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="one file may be at most 256 MB")
    try:
        machines_mod.save_upload_file(upload_id, filename, body)
    except machines_mod.CommissionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"saved": filename, "bytes": len(body)}


@app.get("/kg/{event_id}", summary="The graph subgraph behind one event")
def kg_for_event(event_id: str) -> dict[str, Any]:
    from kg.query import query_for_event

    b = _require_bridge()
    found = b.event(event_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no event {event_id}")
    if state.hook is None:
        raise HTTPException(status_code=503, detail="the diagnosis layer is not running")
    subgraph = query_for_event(found.as_engine_event(), state.hook.pipeline.store)
    return subgraph.to_json()


# --- try mode: "Try Harmonic on your own data" (api/try_mode.py) --------------


def _try_error(exc: try_mode_mod.TryError) -> HTTPException:
    """A refusal as a plain 4xx (or a 503 with Retry-After when the service
    is busy) — never a 500 for something a visitor sent (HARM-S10-T03 L1)."""
    if isinstance(exc, try_mode_mod.RunNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, try_mode_mod.RateLimited):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, try_mode_mod.Conflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, try_mode_mod.TooLarge):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, try_mode_mod.Busy):
        return HTTPException(status_code=503, detail=str(exc),
                             headers={"Retry-After": str(exc.retry_after_s)})
    return HTTPException(status_code=422, detail=str(exc))


def _client_ip(request: Request) -> str:
    """The visitor's address, for the rate limits. In try mode, behind Cloud
    Run, the socket's peer is Google's front end — every visitor would share
    one limit — so the address is the RIGHTMOST X-Forwarded-For entry: the
    one Cloud Run itself appends. Entries to its left are whatever the client
    sent and are never trusted (HARM-S10-T03 H2). The console, and try routes
    beside it, keep the socket's own peer."""
    if MODE == "try":
        values = request.headers.getlist("x-forwarded-for")
        if values:
            right = values[-1].split(",")[-1].strip()
            if right:
                return right[:64]
    return request.client.host if request.client else "unknown"


@app.get("/try/limits", summary="Try mode: the caps this server enforces")
def try_limits() -> dict[str, Any]:
    return try_mode_mod.limits()


@app.post("/try/runs", summary="Try mode: start a new, anonymous run")
def try_new_run(request: Request) -> dict[str, Any]:
    try:
        run = try_mode_mod.new_run(_client_ip(request))
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc
    return {"id": run.id}


_PUT_FLUSH_BYTES = 1 << 20


@app.put("/try/runs/{run_id}/uploads/{filename}", summary="Try mode: add one snapshot file")
async def try_upload_file(run_id: str, filename: str, request: Request,
                          offset: int = Query(0, ge=0, description="byte offset of this part; "
                                              "0 starts the file, otherwise appends")) \
        -> dict[str, Any]:
    """One file, or one part of a file bigger than a request may carry.
    Streamed to disk as it arrives — never a whole body in memory — with the
    disk work off the event loop (HARM-S10-T03 H3, L5, G4)."""
    too_big = HTTPException(status_code=413, detail=(
        f"one request may carry at most {try_mode_mod.MAX_REQUEST_BYTES // (1 << 20)} MB; "
        "the page sends bigger files in parts"))
    cap = try_mode_mod.MAX_REQUEST_BYTES
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > cap):
        raise too_big
    ip = _client_ip(request)
    try:
        up = await asyncio.to_thread(try_mode_mod.begin_upload, run_id, filename, offset, ip)
    except try_mode_mod.RefusedExtension as exc:
        head = b""
        async for chunk in request.stream():
            head = chunk[:8]
            break
        try_mode_mod.format_adapters.log_rejected(
            "upload", ext=exc.ext, head=head,
            size=int(declared) if declared else None, reason="extension not accepted")
        raise _try_error(exc) from exc
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc
    size = 0
    buf = bytearray()
    try:
        async for chunk in request.stream():
            size += len(chunk)
            if size > cap:
                raise too_big
            buf += chunk
            if len(buf) >= _PUT_FLUSH_BYTES:
                await asyncio.to_thread(up.write, bytes(buf))
                buf.clear()
        if buf:
            await asyncio.to_thread(up.write, bytes(buf))
        return await asyncio.to_thread(up.commit)
    except try_mode_mod.TryError as exc:
        await asyncio.to_thread(up.abort)
        raise _try_error(exc) from exc
    except BaseException:
        await asyncio.to_thread(up.abort)
        raise


@app.post("/try/runs/{run_id}/sample",
          summary="Try mode: put the sample recording (NASA IMS) into a new run")
def try_add_sample(run_id: str) -> dict[str, Any]:
    try:
        return try_mode_mod.add_sample(run_id)
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc


class TryConfigureRequest(BaseModel):
    # With probe=True the upload is only read, and family and sampling rate
    # may be left empty: the page says what the file holds before asking.
    family: str = ""
    sample_rate_hz: int = 0
    probe: bool = False
    rpm: float | None = None
    unit: str = "g"
    format: str = "auto"
    has_timestamp: bool = False
    channel: int = 0
    # Without these two the page's "time between snapshots" and the family's
    # optional values were dropped here, silently, before try_mode saw them.
    interval_s: float | None = None
    params: dict[str, float] | None = Field(default=None, max_length=16)


@app.post("/try/runs/{run_id}/configure",
         summary="Try mode: say what the recording is, get the baseline overview")
def try_configure(run_id: str, req: TryConfigureRequest, request: Request) -> dict[str, Any]:
    try:
        return try_mode_mod.configure(run_id, req.model_dump(), _client_ip(request))
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc


class TryStartRequest(BaseModel):
    baseline_h: tuple[float, float]
    consent: bool = False


@app.post("/try/runs/{run_id}/start", summary="Try mode: commission on the baseline and replay")
def try_start(run_id: str, req: TryStartRequest) -> dict[str, Any]:
    try:
        return try_mode_mod.start(run_id, req.model_dump())
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc


@app.get("/try/runs/{run_id}", summary="Try mode: a run's status, and its report once done")
def try_run(run_id: str) -> dict[str, Any]:
    try:
        return try_mode_mod._get_or_404(run_id).public_json()
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc


@app.post("/try/runs/{run_id}/delete", summary="Try mode: delete a run now")
def try_delete(run_id: str) -> dict[str, Any]:
    try:
        try_mode_mod.delete_run(run_id)
    except try_mode_mod.TryError as exc:
        raise _try_error(exc) from exc
    return {"deleted": run_id}


# --- websocket ----------------------------------------------------------------


def _stream_payload(bridge: EngineBridge, last_tick: int) -> dict[str, Any] | None:
    latest = bridge.latest
    if latest is None or latest.tick == last_tick:
        return None
    new_events = [e.to_json() for e in bridge.events(5) if e.tick > last_tick]
    return {"type": "tick", "snapshot": latest.to_json(), "events": new_events}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    """Snapshots at the tick rate.

    Backpressure is handled by sending only the LATEST snapshot rather than a
    queue of every tick since the client last read. A dashboard that falls
    behind wants the current state of the machine, not a backlog of stale ones,
    and a per-client queue would grow without bound behind a paused browser tab.
    """
    await ws.accept()
    if state.bridge is None:
        await ws.send_json({"type": "unavailable", "reason": state.status.reason})
        await ws.close()
        return

    last_tick = -1
    try:
        await ws.send_json({"type": "hello", "engine": _engine_block()})
        while True:
            payload = _stream_payload(state.bridge, last_tick)
            if payload is not None:
                last_tick = payload["snapshot"]["tick"]
                await ws.send_json(payload)
            await asyncio.sleep(STREAM_PERIOD_S)
    except WebSocketDisconnect:
        return
    except Exception:
        log.exception("websocket stream failed")
        with contextlib.suppress(Exception):
            await ws.close()


# --- for tests ----------------------------------------------------------------


def attach(bridge: EngineBridge | None, hook: GenAIHook | None,
           status: BridgeStatus | None = None) -> None:
    """Inject handles instead of autostarting. Used by the end-to-end test."""
    state.bridge = bridge
    state.hook = hook
    state.status = status or BridgeStatus(bridge is not None,
                                          None if bridge else "not started")


def raised_events() -> list[Event]:
    return list(state.bridge.events(200)) if state.bridge else []


# --- the web app ----------------------------------------------------------------

# Mounted last, so every API route above wins. `web/dist` is built by
# `npm run build` in web/ (and by the API image); without it the API runs on
# its own and `/` is simply not found.
WEB_DIST = Path(os.environ.get("HARMONIC_WEB_DIST",
                               Path(__file__).resolve().parents[1] / "web" / "dist"))


@app.get("/try", include_in_schema=False)
@app.get("/try/", include_in_schema=False)
def try_page() -> Response:
    """The standalone try-mode wizard (web/try.html, its own Vite entry —
    web/vite.config.ts). Registered explicitly, not left to the mount below,
    so `/try` (no trailing slash) works without a redirect and the page loads
    in both modes: an unset HARMONIC_MODE still gets the feature."""
    path = WEB_DIST / "try" / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="the web app is not built: "
                                                    "npm run build in web/")
    return FileResponse(path)


@app.get("/try/benchmark", include_in_schema=False)
def try_benchmark_page() -> Response:
    """The public benchmark page (HARM-S10-T10, docs/BENCHMARK.md rendered to
    HTML at web build time — web/scripts/gen-benchmark.mjs). Same SPA shell
    as /try/, routed client-side by pathname (web/src/try/main.tsx);
    registered explicitly, like try_page() above, so it resolves without
    falling through to the static mount's own 404."""
    path = WEB_DIST / "try" / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="the web app is not built: "
                                                    "npm run build in web/")
    return FileResponse(path)


@app.get("/", include_in_schema=False)
def root() -> Response:
    # Checked at request time (not import time) so the mode gate above and
    # this agree even if MODE were ever changed after start-up, as tests do.
    if MODE == "try":
        # The console's own index.html is still built into the same image —
        # HARMONIC_WEB_DIST is shared — but a try-mode deployment has nothing
        # else to show at "/": send a visitor straight to the wizard.
        return RedirectResponse("/try/")
    path = WEB_DIST / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="the web app is not built: "
                                                    "npm run build in web/")
    return FileResponse(path)


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
