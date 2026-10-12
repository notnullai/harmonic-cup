import { useCallback, useEffect, useState } from "react";
import { ArrowRight, ClipboardCheck, Download, LayoutGrid, PlayCircle } from "lucide-react";
import { api } from "../api";
import type { FleetMachine } from "../api";
import { failureMode, fmt, when } from "../format";
import { verdictText } from "./Feedback";

// Every machine, most urgent first (HARM-S7-T13). Each runs its own engine;
// opening one shows it in the console without restarting it.

interface Props { onOpen: (key: string) => void }

export default function Fleet({ onOpen }: Props) {
  const [machines, setMachines] = useState<FleetMachine[] | null>(null);
  const load = useCallback(() => api.fleet().then((f) => setMachines(f.machines)).catch(() => {}), []);
  useEffect(() => { load(); const t = setInterval(load, 2000); return () => clearInterval(t); }, [load]);

  if (!machines) return <div className="page"><div className="skeleton" style={{ height: 240 }} /></div>;
  const alarmed = machines.filter((m) => m.state && m.state !== "OK").length;
  // A machine with several measurement points (HARM-S7-T07) is one card: the
  // rows come most urgent first, so its first row is its worst point.
  const groups: FleetMachine[][] = [];
  const byMachine = new Map<string, FleetMachine[]>();
  for (const m of machines) {
    const k = m.machine ?? m.key;
    const g = byMachine.get(k);
    if (g) g.push(m); else { const n = [m]; byMachine.set(k, n); groups.push(n); }
  }

  return (
    <main className="page fleet-page">
      <div className="fleet-head">
        <LayoutGrid size={20} />
        <h2>Fleet</h2>
        <span className="hint">{machines.filter((m) => m.running).length} watched · {alarmed} need attention · most urgent first</span>
      </div>
      <div className="fleet-grid">
        {groups.map((g) => {
          const m = g[0];
          const multi = g.length > 1;
          const label = multi && m.point ? m.label.replace(` · ${m.point}`, "") : m.label;
          const ev = m.last_event;
          const title = ev?.failure_mode_id ? failureMode(ev.failure_mode_id)
            : ev ? (ev.diagnosis_status === "fallback" ? "Insufficient evidence" : "Diagnosing…") : null;
          return (
            <article key={m.machine ?? m.key} className={`card fleet-card s-${m.state ?? "NONE"}${g.some((x) => x.current) ? " current" : ""}`}>
              <div className="fc-top">
                <div>
                  <div className="fc-name">{label}</div>
                  <div className="hint">{m.family ?? m.kind}{m.equipment_id ? ` · ${m.equipment_id}` : ""}
                    {m.real ? " · real recording" : m.kind === "simulated" ? " · simulated" : ""}
                    {m.running && !m.standard && <span className="warn-text"> · no ISO zones</span>}</div>
                </div>
                <span className="fc-state">{m.running ? (m.state ?? "…") : "not watched"}
                  {multi && m.running && m.state && m.state !== "OK" && <small> at {m.point}</small>}</span>
              </div>
              {multi && (
                <div className="fc-points">
                  {g.map((p) => (
                    <button key={p.key} className={`chip s-${p.state ?? "NONE"}`} disabled={!p.available}
                      title={`Open ${p.point}`} onClick={() => onOpen(p.key)}>
                      {p.point} · {p.running ? (p.state ?? "…") : "not watched"}</button>
                  ))}
                </div>
              )}
              {m.running ? (
                <div className="fc-kpis">
                  <div><span className="hint">Health index</span><b className="num">{m.hhi?.toFixed(2) ?? "—"}</b></div>
                  <div><span className="hint">Vibration</span><b className="num">{fmt(m.vib, m.vib_unit ?? "")}</b><small>{m.vib_unit}</small></div>
                  <div><span className="hint">Life left</span><b className="num">{m.rul_hours == null ? "—" : m.rul_hours >= 720 ? "> 30 d" : `${(m.rul_hours / 24).toFixed(1)} d`}</b></div>
                </div>
              ) : (
                <div className="hint fc-off">{m.available ? "Its engine starts when you open it." : m.reason}</div>
              )}
              <div className="fc-event">
                {ev ? (
                  <>
                    <div><b>{title}</b> <span className="hint">· {ev.state} · {when(ev.machine_hours, null)}</span></div>
                    {ev.feedback && <div className="hint"><ClipboardCheck size={11} style={{ verticalAlign: -1 }} /> {verdictText(ev.feedback)}</div>}
                  </>
                ) : <span className="hint">{m.running ? "No event this run." : ""}</span>}
              </div>
              <div className="fc-foot">
                <span className="hint">{m.stored.events} stored events · {m.stored.feedback} judged</span>
                <span className="grow" />
                {m.stored.feedback > 0 && (
                  <a className="btn sm ghost" href={`/feedback/export?source=${encodeURIComponent(m.key)}`}
                    title="Judged events as labelled training frames"><Download size={13} />Training data</a>
                )}
                <button className="btn sm" disabled={!m.available} onClick={() => onOpen(m.key)}>
                  {m.running ? <>Open<ArrowRight size={13} /></> : <><PlayCircle size={13} />Watch</>}
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </main>
  );
}
