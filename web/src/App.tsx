import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ClipboardCheck, History, Radio, ServerCrash, X } from "lucide-react";
import { api, type FeatureMeaning, type HEvent, type Sources, type WorkOrder } from "./api";
import { atOrBefore, duration, failureMode, when } from "./format";
import { useLive } from "./useLive";
import TopBar from "./components/TopBar";
import StatusHero from "./components/StatusHero";
import Timeline, { type View } from "./components/Timeline";
import Rebaseline from "./components/Rebaseline";
import SensorGrid from "./components/SensorGrid";
import DiagnosisPanel from "./components/DiagnosisPanel";
import KnowledgeGraph from "./components/KnowledgeGraph";
import EventList from "./components/EventList";
import SimControls from "./components/SimControls";
import ExportDialog from "./components/ExportDialog";
import ChatDrawer from "./components/ChatDrawer";
import Panel from "./components/Panel";
import Wizard from "./components/Wizard";
import Fleet from "./components/Fleet";
import WorkOrders from "./components/WorkOrders";

interface Toast { id: number; text: string; err?: boolean }
/** The pop-up when a work order goes to maintenance, or is escalated. */
interface OrderNotice { key: string; id: string; update: boolean; state: string; fault: string; equipment: string }

export default function App() {
  const { health, points, events, marks, connected, epoch, refresh } = useLive();
  const [rebaselineKey, setRebaselineKey] = useState<string | null>(null);
  // The re-learn dialog also sets a machine's ISO zones (same pipeline).
  const [rebaselineMode, setRebaselineMode] = useState<"baseline" | "iso">("baseline");
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? "dark");
  const [sources, setSources] = useState<Sources | null>(null);
  const [live, setLive] = useState(true);
  const [cursorH, setCursorH] = useState<number | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);
  const [view, setView] = useState<View | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [page, setPage] = useState<"console" | "fleet">("console");
  const [busy, setBusy] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [orders, setOrders] = useState<WorkOrder[]>([]);
  const [webhook, setWebhook] = useState<string | null>(null);
  const [ordersView, setOrdersView] = useState<string | null | false>(false);
  const [notices, setNotices] = useState<OrderNotice[]>([]);
  const [meanings, setMeanings] = useState<Record<string, FeatureMeaning>>({});

  const toast = useCallback((text: string, err = false) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, err }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4200);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("harmonic-theme", theme); } catch { /* private mode */ }
  }, [theme]);

  const loadSources = useCallback(() => { api.sources().then(setSources).catch(() => {}); }, []);
  useEffect(() => { loadSources(); const t = setInterval(loadSources, 15000); return () => clearInterval(t); }, [loadSources]);

  const loadOrders = useCallback(() => {
    api.workOrders().then((r) => { setOrders(r.work_orders); setWebhook(r.webhook); }).catch(() => {});
  }, []);
  useEffect(() => { loadOrders(); const t = setInterval(loadOrders, 10000); return () => clearInterval(t); }, [loadOrders]);

  // A work order sent, or re-sent as an escalation, pops up once. The alarms
  // already there when the console opened do not.
  const seenOrders = useRef<Set<string> | null>(null);
  useEffect(() => {
    const keys = events.filter((e) => e.work_order && !e.work_order.repeat
      && !(e.work_order.delivery ?? "").startsWith("not sent"))
      .map((e) => ({ e, key: `${e.work_order!.id}:${e.work_order!.update ? e.state : "new"}` }));
    if (seenOrders.current === null) {
      if (events.length === 0) return;
      seenOrders.current = new Set(keys.map((k) => k.key));
      return;
    }
    const seen = seenOrders.current;
    const fresh = keys.filter((k) => !seen.has(k.key));
    if (fresh.length === 0) return;
    fresh.forEach((k) => seen.add(k.key));
    setNotices((n) => [...n, ...fresh.map(({ e, key }) => ({
      key, id: e.work_order!.id, update: e.work_order!.update, state: e.state,
      fault: failureMode(e.diagnosis?.failure_mode_id), equipment: e.equipment_id }))].slice(-3));
    fresh.forEach(({ key }) => setTimeout(() => setNotices((n) => n.filter((x) => x.key !== key)), 15000));
    loadOrders();
  }, [events, loadOrders]);

  // A restart or a source switch starts the view afresh, following live.
  useEffect(() => { setLive(true); setCursorH(null); setPinned(null); setView(null); }, [epoch]);

  const source = health?.engine.source;
  // What each feature means on this machine, for the diagnosis's evidence.
  useEffect(() => {
    setMeanings({});
    api.evidence().then((r) => setMeanings(r.features)).catch(() => {});
  }, [source?.key, epoch]);
  const unit = source?.vib_unit ?? "mm/s";
  const recorded = source?.kind === "recorded";
  // How much the live view shows: a day and a half of the week-long recording
  // (216 snapshots), an hour and a half of the pump (~140 frames).
  const windowH = recorded ? 36 : 1.5;

  const latest = points[points.length - 1];
  const inspectH = cursorH ?? (live ? latest?.h : view?.end) ?? latest?.h ?? 0;
  const point = atOrBefore(points, inspectH, (p) => p.h);
  const event: HEvent | undefined = useMemo(() => {
    if (pinned) {
      const e = events.find((x) => x.id === pinned);
      if (e) return e;
    }
    return atOrBefore(events, inspectH + 1e-9, (e) => e.machine_hours);
  }, [events, pinned, inspectH]);
  const windowPoints = useMemo(() => {
    if (!view) return points.slice(-150);
    return points.filter((p) => p.h >= view.start && p.h <= view.end);
  }, [points, view]);

  const goLive = useCallback(() => { setLive(true); setCursorH(null); setPinned(null); }, []);
  const inspect = useCallback((h: number) => { setLive(false); setCursorH(h); setPinned(null); }, []);
  const openEvent = useCallback((e: HEvent) => { setLive(false); setCursorH(e.machine_hours); setPinned(e.id); }, []);

  // Keyboard: step the cursor through readings, jump between events, back to live.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (exportOpen || chatOpen || wizardOpen || ordersView !== false || t.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "Escape" || e.key.toLowerCase() === "l") { goLive(); return; }
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const dir = e.key === "ArrowLeft" ? -1 : 1;
      if (e.shiftKey) {
        const list = dir < 0 ? [...events].reverse() : events;
        const next = list.find((x) => (dir < 0 ? x.machine_hours < inspectH - 1e-9 : x.machine_hours > inspectH + 1e-9));
        if (next) openEvent(next);
        return;
      }
      const i = point ? points.indexOf(point) : points.length - 1;
      const j = Math.max(0, Math.min(points.length - 1, i + dir));
      if (points[j]) inspect(points[j].h);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [events, points, point, inspectH, exportOpen, chatOpen, wizardOpen, ordersView, goLive, inspect, openEvent]);

  const run = async (label: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(label);
    try { await fn(); toast(ok); } catch (e) { toast((e as Error).message, true); } finally { setBusy(null); }
  };

  // Every machine runs its own engine: switching looks at another one and
  // keeps its history. Restart is what starts one again from nothing.
  const onSource = (key: string) => run("Switching machine…", async () => {
    await api.select(key);
    refresh();
    loadSources();
  }, `Now showing ${sources?.sources[key]?.label ?? key}`);

  const onFeedback = async (e: HEvent, verdict: "confirmed" | "wrong", actual: string | null, note: string) => {
    try {
      await api.feedback(e.id, verdict, actual, note);
      toast(verdict === "confirmed" ? "Recorded: confirmed on inspection" : "Recorded: what was actually there");
      refresh();
    } catch (err) { toast((err as Error).message, true); }
  };

  const onManual = (key: string, file: File) => run("Reading the manual…", async () => {
    const r = await api.uploadManual(key, file);
    toast(`${file.name}: ${r.sections} sections, cited as ${r.doc_id} in this machine's diagnoses`);
  }, "Manual added");

  const onDelete = (key: string) => run("Removing…", async () => { await api.deleteMachine(key); loadSources(); },
    "Machine removed");

  const onRestart = () => run("Restarting…", async () => { await api.reset(); refresh(); },
    recorded ? "Replaying the recording from the first snapshot" : "Fresh engine and a fresh machine, warm and watching");

  if (health && !health.engine.available) {
    return (
      <div className="app">
        <div className="empty" style={{ margin: "12vh auto", maxWidth: 560 }}>
          <ServerCrash size={36} />
          <h2 style={{ margin: 0 }}>The engine is not running</h2>
          <div>{health.engine.reason}</div>
          <div className="hint">Build it with <kbd>cmake --build build -j</kbd> and <kbd>python scripts/build_brain.py</kbd>, then restart the API.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <TopBar health={health} sources={sources} connected={connected} theme={theme} busy={!!busy}
        onSource={onSource} onCommission={() => setWizardOpen(true)} onDelete={onDelete} onManual={onManual} onRebaseline={(k) => { setRebaselineMode("baseline"); setRebaselineKey(k); }} onRestart={onRestart} onExport={() => setExportOpen(true)}
        onChat={() => setChatOpen(true)} openOrders={orders.filter((o) => o.status === "open").length}
        onOrders={() => setOrdersView(null)} page={page} onFleet={() => setPage(page === "fleet" ? "console" : "fleet")}
        onTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} />

      {page === "fleet" ? <Fleet onOpen={(key) => { setPage("console"); if (key !== source?.key) onSource(key); }} /> : (
      <main className="page">
        <div className="col">
          {!live && point && (
            <div className="inspect-bar" role="status">
              <History size={16} />
              <span>Viewing <b>{when(point.h, point.at)}</b>
                {latest && latest.h - point.h > 1 / 60 && <> · {duration(latest.h - point.h)} before live</>}
              </span>
              <span className="grow" />
              <span className="hint"><kbd>←</kbd> <kbd>→</kbd> step · <kbd>Shift</kbd>+<kbd>←</kbd> <kbd>→</kbd> events · <kbd>Esc</kbd> live</span>
              <button className="btn sm primary" onClick={goLive}><Radio size={13} />Back to live</button>
            </div>
          )}

          <Panel name="Machine state"><StatusHero point={point} latest={latest} source={source} live={live} ruledOut={health?.engine.ruled_out_classes ?? {}}
            onSetIso={source && sources?.sources[source.key]?.kind === "commissioned"
              ? () => { setRebaselineMode("iso"); setRebaselineKey(source.key); } : undefined} /></Panel>

          <section className="card">
            <div className="card-h">
              <h2>Timeline</h2><span className="grow" />
              <div className="legend" aria-hidden="true">
                <span><i style={{ background: "var(--accent)" }} />health index</span>
                <span><i style={{ background: "var(--muted)" }} />vibration ({unit})</span>
                <span><i style={{ background: "var(--critical)", height: 8, width: 8, borderRadius: 4 }} />event</span>
              </div>
              {live ? <span className="chip" style={{ color: "var(--ok)" }}><span className="dot pulse" />following live</span>
                : <button className="btn sm" onClick={goLive}><Radio size={13} />Live</button>}
            </div>
            <div className="card-b" style={{ paddingTop: 6 }}>
              <Panel name="Timeline"><Timeline points={points} events={events} marks={marks} unit={unit} live={live} cursorH={cursorH}
                windowH={windowH} theme={theme} onCursor={inspect}
                onLive={(v) => (v ? goLive() : setLive(false))} onView={setView} onEvent={openEvent} /></Panel>
              <div className="hint">Drag to scroll back in time, scroll to zoom, click to inspect a moment — the
                state, sensors and diagnosis follow what you are looking at.</div>
            </div>
          </section>

          <Panel name="Sensors"><SensorGrid point={point} window={windowPoints} vibUnit={unit} /></Panel>
        </div>

        <div className="col">
          <Panel name="Diagnosis"><DiagnosisPanel event={event} inspectH={inspectH} live={live} equipmentId={source?.equipment_id}
            faults={source?.fault_modes ?? []} onFeedback={onFeedback} onPin={openEvent}
            meanings={meanings} onOrder={(id) => { loadOrders(); setOrdersView(id); }} /></Panel>
          <Panel name="Knowledge graph"><KnowledgeGraph event={event} theme={theme} /></Panel>
          <Panel name="Events"><EventList events={events} selectedId={event?.id} onSelect={openEvent} /></Panel>
          <Panel name="Controls"><SimControls source={source} injected={health?.machine?.injected ?? {}}
            growth={health?.machine?.fault_growth} busy={!!busy}
            onInject={(m, s, r, p) => run("Injecting…", () => api.inject(m, s, r, p),
              `Injected ${m.replace(/_/g, " ")}, growing to ${Math.round(s * 100)}%${p ? ` — pauses at ${p}` : ""}`)}
            onHold={(h) => run(h ? "Pausing…" : "Resuming…", async () => { await api.hold(h); refresh(); },
              h ? "Fault paused at its current level" : "Fault growing again")}
            onClear={() => run("Clearing…", () => api.clear(), "Faults cleared — Restart for a clean engine")} /></Panel>
        </div>
      </main>
      )}

      <footer className="footer">
        <span>Advisory only — Harmonic never drives an actuator; the plant's own protection stays authoritative.</span>
        <span>
          {health?.engine.brain ? `brain ${health.engine.brain.split("/").pop()} · ` : ""}
          diagnosis by {health?.genai.provider ?? "—"}
        </span>
      </footer>

      <ExportDialog open={exportOpen} points={points} view={view} cursorH={cursorH}
        onClose={() => setExportOpen(false)} onDone={toast} />
      <Wizard open={wizardOpen} onClose={() => { setWizardOpen(false); loadSources(); }}
        onWatch={(id) => { loadSources(); onSource(id); }} toast={toast} />
      <Rebaseline machineId={rebaselineKey} mode={rebaselineMode} onClose={() => { setRebaselineKey(null); loadSources(); }} toast={toast}
        orders={orders} onOrdersClosed={loadOrders} />
      <ChatDrawer open={chatOpen} onClose={() => setChatOpen(false)} epoch={epoch} />
      <WorkOrders view={ordersView} orders={orders} webhook={webhook} events={events}
        onView={setOrdersView} onChanged={loadOrders} toast={toast} />

      {busy && <div className="overlay-busy" role="status"><div style={{ display: "grid", placeItems: "center", gap: 12 }}>
        <div className="spinner" /><b>{busy}</b></div></div>}
      <div className="notices" aria-live="polite">
        {notices.map((n) => (
          <div key={n.key} className={`notice s-${n.state}`} role="status">
            <ClipboardCheck size={18} />
            <div>
              <b>{n.update ? `Work order updated — escalated to ${n.state}` : "Work order sent to maintenance"}</b>
              <div className="hint">{n.id} · {n.fault} suspected on {n.equipment}</div>
            </div>
            <button className="btn sm" onClick={() => { setOrdersView(n.id); setNotices((x) => x.filter((y) => y.key !== n.key)); }}>Open</button>
            <button className="btn ghost icon sm" aria-label="Dismiss"
              onClick={() => setNotices((x) => x.filter((y) => y.key !== n.key))}><X size={14} /></button>
          </div>
        ))}
      </div>
      <div className="toasts" aria-live="polite">
        {toasts.map((t) => <div key={t.id} className={`toast${t.err ? " err" : ""}`}>{t.text}</div>)}
      </div>
    </div>
  );
}
