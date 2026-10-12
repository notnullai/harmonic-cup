import { useState } from "react";
import { Check, Cog, Droplets, Eraser, MoveHorizontal, Pause, Play, Syringe, Timer, TrendingUp, Waves, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { FaultGrowth, SourceInfo } from "../api";

// Icons for the fault modes the families know; any other mode gets a cog.
// The list of modes itself comes from the source, so a new machine family
// needs no change here.
const ICONS: Record<string, LucideIcon> = {
  bearing_wear: Waves, misalignment: MoveHorizontal, cavitation: Zap, seal_leak: Droplets,
};
const PUMP_MODES = ["bearing_wear", "misalignment", "cavitation", "seal_leak"];
const label = (k: string) => k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " ");

interface Props {
  source: SourceInfo | undefined;
  injected: Record<string, number>;
  growth: FaultGrowth | null | undefined;
  busy: boolean;
  onInject: (mode: string, severity: number, ramp: number, pauseAt: string | null) => void;
  onHold: (hold: boolean) => void;
  onClear: () => void;
}

export default function SimControls({ source, injected, growth, busy, onInject, onHold, onClear }: Props) {
  const modes = source?.fault_modes ?? PUMP_MODES;
  const [picked, setMode] = useState(modes[0]);
  const mode = modes.includes(picked) ? picked : modes[0];
  const [severity, setSeverity] = useState(1);
  // Two machine hours: WARNING comes within seconds (every mode crosses it at
  // 8-21% severity, overall vibration still in ISO zone A), and the pause
  // holds it there.
  const [ramp, setRamp] = useState(120);
  const [pauseAt, setPauseAt] = useState("WARNING");

  if (!source) return null;

  if (source.kind === "live" && source.live) {
    const l = source.live;
    return (
      <section className="card">
        <div className="card-h"><h2>Live sensor</h2><span className="grow" />
          <span className="chip" style={{ color: l.connected ? "var(--ok)" : "var(--warning)" }}>
            <span className={`dot${l.connected ? " pulse" : ""}`} />{l.connected ? "connected" : "reconnecting"}</span></div>
        <div className="card-b">
          <p className="hint" style={{ margin: 0 }}>MQTT <b>{l.topic}</b> on {l.broker} · <b className="num">{l.received}</b> snapshots received
            {l.rejected + l.refused > 0 && <> · {l.rejected + l.refused} refused</>}
            {l.dropped > 0 && <> · {l.dropped} dropped behind</>}</p>
          {l.last_error && <p className="hint" style={{ margin: "6px 0 0" }}>{l.last_error}</p>}
          <p className="hint" style={{ margin: "6px 0 0" }}>A real machine: nothing can be injected. <b>Restart</b> starts its engine again from nothing.</p>
        </div>
      </section>
    );
  }

  if (!source.supports_injection) {
    const p = source.progress;
    const pct = p ? (100 * p.snapshot) / Math.max(p.of, 1) : 0;
    return (
      <section className="card">
        <div className="card-h"><h2>Replay</h2><span className="grow" />
          <span className="chip num">{p ? `${p.snapshot} / ${p.of}` : ""}</span></div>
        <div className="card-b">
          <div className="progress" aria-label="Replay progress"><i style={{ width: `${pct}%` }} /></div>
          <p className="hint" style={{ margin: "10px 0 0" }}>
            {p?.finished ? "The recording has ended — " : "A recording of a real machine, played snapshot by snapshot through the engine. "}
            Nothing can be injected. <b>Restart</b> replays it from the first snapshot.
          </p>
        </div>
      </section>
    );
  }

  const active = Object.entries(injected);
  const g = growth ?? null;
  const speed = g?.speedup ?? null;
  const growing = active.some(([k, v]) => v < (g?.targets[k] ?? v) - 1e-3);
  return (
    <section className="card">
      <div className="card-h"><h2>Inject a fault</h2><span className="grow" /><span className="badge sim">simulator</span></div>
      <div className="card-b">
        {speed && (
          <p className="hint sim-speed"><Timer size={12} style={{ verticalAlign: -2 }} /> Demo time runs about <b>{Math.round(speed)}×</b> real
            time: one second here is {fmtMin(speed / 60)} of machine time.</p>
        )}
        <div className="field">
          <label>Fault mode</label>
          <div className="faults" role="group" aria-label="Fault mode">
            {modes.map((key) => {
              const Icon = ICONS[key] ?? Cog;
              return (
                <button key={key} aria-pressed={mode === key} onClick={() => setMode(key)}>
                  <Icon size={15} />{label(key)}
                </button>
              );
            })}
          </div>
        </div>
        <div className="field">
          <label htmlFor="sev"><span>Grows to</span><b className="num">{Math.round(severity * 100)}% severity</b></label>
          <input id="sev" type="range" min={0.1} max={1} step={0.05} value={severity}
            onChange={(e) => setSeverity(Number(e.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="ramp"><span>Over</span><b className="num">{ramp ? `${fmtMin(ramp)} machine time${speed ? ` · ≈ ${fmtSec((ramp * 60) / speed)} here` : ""}` : "instantly"}</b></label>
          <input id="ramp" type="range" min={0} max={720} step={10} value={ramp}
            onChange={(e) => setRamp(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>Pause the fault when the machine reaches</label>
          <div className="faults" role="group" aria-label="Pause at">
            {PAUSE.map(([k, l]) => (
              <button key={k} aria-pressed={pauseAt === k} onClick={() => setPauseAt(k)}>{l}</button>
            ))}
          </div>
          <span className="hint">On a real machine WARNING usually comes days to weeks before CRITICAL. Pausing holds
            the machine there so you can read the diagnosis, open the work order and act, then resume.</span>
        </div>
        <div className="row">
          <button className="btn primary" disabled={busy} onClick={() => onInject(mode, severity, ramp, pauseAt || null)}>
            <Syringe size={15} />Inject</button>
          <button className="btn" disabled={busy || active.length === 0} onClick={onClear}>
            <Eraser size={15} />Clear faults</button>
        </div>
        {active.length > 0 && (
          <div className="fault-now">
            {active.map(([k, v]) => {
              const target = g?.targets[k] ?? v;
              return (
                <div key={k} className="fault-row">
                  <span><b>{label(k)}</b> <span className="num">{Math.round(v * 100)}%</span>
                    {target > v + 1e-3 && <span className="hint"> → {Math.round(target * 100)}%</span>}</span>
                  <div className="progress" aria-hidden="true"><i style={{ width: `${Math.min(100, 100 * v)}%` }} /></div>
                </div>
              );
            })}
            <div className="fault-status">
              {g?.held ? <><Pause size={13} /><span>Paused{g.paused_at ? ` at ${g.paused_at}` : ""} — the fault holds at this level.</span></>
                : growing ? <><TrendingUp size={13} /><span>Growing{g?.pause_at ? ` — pauses at ${g.pause_at}` : ""}</span></>
                : <><Check size={13} /><span>At full severity</span></>}
              <span className="grow" />
              {g?.held ? <button className="btn sm" disabled={busy} onClick={() => onHold(false)}><Play size={13} />Resume</button>
                : growing ? <button className="btn sm" disabled={busy} onClick={() => onHold(true)}><Pause size={13} />Pause now</button> : null}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

const PAUSE: [string, string][] = [["WATCH", "Watch"], ["WARNING", "Warning"], ["CRITICAL", "Critical"], ["", "Don't pause"]];
const fmtMin = (m: number) => m < 1 ? `${Math.round(m * 60)} s` : m < 120 ? `${Math.round(m)} min` : `${(m / 60).toFixed(m % 60 ? 1 : 0)} h`;
const fmtSec = (s: number) => s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`;
