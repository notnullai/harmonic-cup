import { useEffect, useRef, useState } from "react";
import { AlertTriangle, BookUp, ChevronDown, ClipboardList, Factory, FileDown, FlaskConical, History, LayoutGrid, MessageSquareText, Moon, Plus, RotateCcw, Sun, Trash2 } from "lucide-react";
import type { Health, Sources } from "../api";

interface Props {
  health: Health | null;
  sources: Sources | null;
  connected: boolean;
  theme: string;
  busy: boolean;
  onSource: (key: string) => void;
  onCommission: () => void;
  onDelete: (key: string) => void;
  onManual: (key: string, file: File) => void;
  onRebaseline: (key: string) => void;
  onRestart: () => void;
  onExport: () => void;
  onChat: () => void;
  /** Open work orders, for the badge; opens the work-order inbox. */
  openOrders: number;
  onOrders: () => void;
  page: "console" | "fleet";
  onFleet: () => void;
  onTheme: () => void;
}

export default function TopBar(p: Props) {
  const src = p.health?.engine.source;
  const current = p.sources?.current ?? src?.key;
  const opts = p.sources?.sources ?? {};
  return (
    <header className="topbar">
      <div className="topbar-in">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="8" fill="var(--surface-3)" />
            <path d="M5 17h4l3-8 4 15 3-10 2 5h6" fill="none" stroke="var(--accent)" strokeWidth="2.6"
              strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>HARMONIC<small>machine health</small></div>
        </div>

        <MachineMenu opts={opts} current={current} busy={p.busy} onSource={p.onSource}
          onCommission={p.onCommission} onDelete={p.onDelete} onManual={p.onManual} onRebaseline={p.onRebaseline} />

        {src && (
          <span className={`badge ${src.kind === "simulated" ? "sim" : "real"}`} title={src.detail}>
            <span className="dot" />{src.badge ?? (src.real ? "Real recording" : "Simulated data")}
          </span>
        )}

        <span className="spacer" />

        {p.health?.kg?.empty && (
          <span className="chip bad" title="No knowledge graph is loaded — raised events will get no diagnosis. Build it with: python -m kg.load --rebuild">
            <AlertTriangle size={13} />Knowledge graph empty
          </span>
        )}
        <span className="chip" title={p.connected ? "Connected to the engine" : "The API is not reachable"}
          style={{ color: p.connected ? "var(--ok)" : "var(--critical)" }}>
          <span className={`dot${p.connected ? " pulse" : ""}`} />{p.connected ? "Live" : "Offline"}
        </span>
        <button className="btn" onClick={p.onFleet} aria-pressed={p.page === "fleet"} title="Every machine, most urgent first">
          <LayoutGrid size={15} />{p.page === "fleet" ? "Console" : "Fleet"}</button>
        <button className="btn" onClick={p.onRestart} disabled={p.busy} title="Start this machine and the engine again from nothing">
          <RotateCcw size={15} />Restart</button>
        <button className="btn" onClick={p.onOrders} title="Jobs sent to maintenance">
          <ClipboardList size={15} />Work orders
          {p.openOrders > 0 && <span className="count" aria-label={`${p.openOrders} open`}>{p.openOrders}</span>}</button>
        <button className="btn" onClick={p.onChat}><MessageSquareText size={15} />Ask</button>
        <button className="btn primary" onClick={p.onExport}><FileDown size={15} />Export PDF</button>
        <button className="btn ghost icon" onClick={p.onTheme} aria-label="Toggle light and dark theme">
          {p.theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</button>
      </div>
    </header>
  );
}

type Opts = NonNullable<Props["sources"]>["sources"];

/** Which machine is on screen, and the way to add another. A menu rather than
 *  a row of buttons: commissioning makes the list as long as the fleet. */
function MachineMenu({ opts, current, busy, onSource, onCommission, onDelete, onManual, onRebaseline }: {
  opts: Opts; current: string | null | undefined; busy: boolean;
  onSource: (key: string) => void; onCommission: () => void; onDelete: (key: string) => void;
  onManual: (key: string, file: File) => void;
  onRebaseline: (key: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const off = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", off);
    window.addEventListener("keydown", esc);
    return () => { window.removeEventListener("mousedown", off); window.removeEventListener("keydown", esc); };
  }, [open]);
  const cur = current ? opts[current] : undefined;
  const icon = (k?: string) => (k === "simulated" ? <FlaskConical size={14} /> : <Factory size={14} />);
  return (
    <div className="menu-wrap" ref={ref}>
      <button className="btn machine-btn" aria-haspopup="menu" aria-expanded={open} disabled={busy}
        onClick={() => setOpen(!open)}>
        {icon(cur?.kind)}<span className="machine-name">{cur?.label ?? "Choose a machine"}</span><ChevronDown size={14} />
      </button>
      {open && (
        <div className="menu" role="menu">
          <div className="menu-h">Machines</div>
          {Object.entries(opts).map(([key, o]) => (
            <div key={key} className="menu-row">
              <button role="menuitemradio" aria-checked={current === key} disabled={!o.available}
                title={o.available ? undefined : o.reason ?? "unavailable"}
                onClick={() => { setOpen(false); if (current !== key) onSource(key); }}>
                {icon(o.kind)}<span>{o.label}{o.kind === "simulated" && <small>simulator</small>}
                  {!o.available && <small>{o.reason}</small>}</span>
              </button>
              {o.kind === "commissioned" && (
                <label className="btn ghost icon sm" title="Upload this machine's own manual (Markdown or text), to cite in its diagnoses">
                  <BookUp size={13} />
                  <input type="file" accept=".md,.markdown,.txt" hidden aria-label={`Upload a manual for ${o.label}`}
                    onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) { setOpen(false); onManual(key, f); } }} />
                </label>
              )}
              {o.kind === "commissioned" && o.available && (
                <button className="btn ghost icon sm" aria-label={`Re-learn normal for ${o.label}`}
                  title="Re-learn normal after maintenance, from a healthy period"
                  onClick={() => { setOpen(false); onRebaseline(key); }}><History size={13} /></button>
              )}
              {o.kind === "commissioned" && current !== key && (
                <button className="btn ghost icon sm" aria-label={`Remove ${o.label}`} title="Remove this machine"
                  onClick={() => { if (window.confirm(`Remove ${o.label}? Its brain and profile are deleted.`)) onDelete(key); }}>
                  <Trash2 size={13} /></button>
              )}
            </div>
          ))}
          <div className="menu-sep" />
          <button className="menu-add" role="menuitem" onClick={() => { setOpen(false); onCommission(); }}>
            <Plus size={14} />Commission a machine…</button>
        </div>
      )}
    </div>
  );
}
