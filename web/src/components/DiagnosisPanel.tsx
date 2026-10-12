import { useState } from "react";
import Feedback from "./Feedback";
import Evidence from "./Evidence";
import { delivery } from "./WorkOrders";
import { AlertTriangle, BookOpen, Bot, CheckCircle2, ChevronRight, ClipboardCheck, HardHat, Loader2, SearchX, Wrench } from "lucide-react";
import type { FeatureMeaning, HEvent } from "../api";
import { duration, failureMode, when } from "../format";

interface Props {
  event: HEvent | undefined;
  inspectH: number;
  live: boolean;
  equipmentId: string | undefined;
  faults: string[];
  onFeedback: (e: HEvent, verdict: "confirmed" | "wrong", actual: string | null, note: string) => Promise<void>;
  onPin: (e: HEvent) => void;
  meanings: Record<string, FeatureMeaning>;
  onOrder: (id: string) => void;
}

type Tab = "actions" | "safety" | "causes" | "sources";

export default function DiagnosisPanel({ event, inspectH, live, equipmentId, faults, onFeedback, onPin, meanings, onOrder }: Props) {
  const [tab, setTab] = useState<Tab>("actions");

  if (!event) {
    return (
      <section className="card">
        <div className="card-h"><h2>Diagnosis</h2></div>
        <div className="card-b">
          <div className="empty">
            <CheckCircle2 size={30} />
            <div><b>No event {live ? "yet" : "before this moment"}.</b></div>
            <div>A diagnosis is produced when the engine raises WARNING or worse.</div>
          </div>
        </div>
      </section>
    );
  }

  const dx = event.diagnosis;
  const status = event.diagnosis_status;
  const ok = status === "ok" && dx && dx.failure_mode_id;
  const pending = status === "pending" || status === "queued" || status === "running";
  const offline = (event.diagnosis_provider ?? "").startsWith("offline");
  const age = inspectH - event.machine_hours;
  // The documents cited, from "GB-201-OM-5.5" to "GB-201-OM".
  const refDocs = [...new Set((dx?.references ?? []).map((r) => r.replace(/-\d+(\.\d+)*$/, "")))];

  return (
    <section className="card" aria-live="polite">
      <div className="card-h">
        <h2>Diagnosis</h2><span className="grow" />
        <span className={`state-pill s-${event.state}`}><span className="dot" />{event.state}</span>
      </div>
      <div className="card-b">
        <div className="dx-meta">
          <span className="chip">raised {when(event.machine_hours, event.recorded_at)}</span>
          {age > 1 / 60 && <span className="chip">{duration(age)} before {live ? "now" : "the cursor"}</span>}
          <span className="chip">health index {event.hhi.toFixed(2)}</span>
          {event.point && <span className="chip">at {event.point}</span>}
          {event.unrecognised && <span className="chip warn-chip">unrecognised change</span>}
          {event.early && <span className="chip" title="Recognised with confidence; no measurement is severe yet">early stage</span>}
          {event.diagnosis_of && (
            <span className="chip" title={`The diagnosis made for ${event.diagnosis_of}: the same fault, still present`}>
              same fault as an earlier alarm
            </span>
          )}
        </div>

        {pending && (
          <div className="empty"><Loader2 size={26} className="spin" style={{ animation: "spin 1s linear infinite" }} />
            <div>Querying the knowledge graph and the manual…</div></div>
        )}

        {!pending && !ok && (
          <>
            <div className="dx-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <SearchX size={20} /> Insufficient evidence
            </div>
            <p className="dx-text">
              The engine raised this state, but the surprised features do not support any failure
              mode this machine can have above the confidence floor. The guardrail declines to name
              one rather than guess.
            </p>
          </>
        )}

        {ok && dx && (
          <>
            <div className="dx-title">{failureMode(dx.failure_mode_id)}</div>
            <div className="conf">
              <span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>confidence</span>
              <span className="bar"><i style={{ width: `${Math.round(dx.confidence * 100)}%` }} /></span>
              <b className="num">{Math.round(dx.confidence * 100)}%</b>
            </div>
            <p className="dx-text">{dx.diagnosis}</p>
          </>
        )}

        <Evidence features={event.top_surprised_features} meanings={meanings}
          diagnosed={ok && dx ? dx.failure_mode_id : undefined} unrecognised={event.unrecognised} />

        {ok && dx && (
          <>
            <div className="tabs" role="tablist">
              {([
                ["actions", `Actions (${dx.actions?.length ?? 0})`],
                ["safety", "Safety"],
                ["causes", "Causes"],
                ["sources", "Sources"],
              ] as [Tab, string][]).map(([k, label]) => (
                <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>
              ))}
            </div>
            {tab === "actions" && (
              <ol className="steps">
                {(dx.actions ?? []).map((a, i) => (
                  <li key={i}>
                    <span className="n">{i + 1}</span>
                    <div>
                      <div>{a.step}</div>
                      <div className="meta"><Wrench size={11} style={{ verticalAlign: -1 }} /> {a.tool} · {Math.round(a.duration_min)} min · {a.skill_level}</div>
                    </div>
                  </li>
                ))}
              </ol>
            )}
            {tab === "safety" && (
              <>
                <div className="note" style={{ marginBottom: 10 }}>
                  <HardHat size={13} style={{ verticalAlign: -2 }} /> Before any work. Harmonic is advisory; the plant's own protection stays authoritative.
                </div>
                <ul className="bullets">{(dx.safety ?? []).map((s, i) => <li key={i}>{s}</li>)}</ul>
              </>
            )}
            {tab === "causes" && (
              <ul className="bullets">{(dx.root_causes ?? []).map((s, i) => <li key={i}>{s}</li>)}</ul>
            )}
            {tab === "sources" && (
              <>
                <div className="refs">{(dx.references ?? []).map((r) => <code key={r}>{r}</code>)}</div>
                <p className="hint" style={{ marginTop: 10 }}>
                  <BookOpen size={12} style={{ verticalAlign: -2 }} /> Every citation was checked against what was retrieved; an invented one is rejected.
                </p>
              </>
            )}
          </>
        )}

        {event.work_order && (() => {
          const w = event.work_order;
          const d = delivery(w.delivery);
          // A repeat was added to the order already sent; an update re-sent it.
          const text = d.tone !== "ok" ? d.text
            : w.repeat ? "Already sent — this alarm was added to it"
            : w.update ? `Update sent — escalated to ${event.state}` : d.text;
          return (
            <button className={`wo-line ${d.tone}`} onClick={() => onOrder(w.id)} title="Open the work order">
              <ClipboardCheck size={15} />
              <span>Work order <b>{w.id}</b><span className="wo-state">{text}</span></span>
              <ChevronRight size={15} />
            </button>
          );
        })()}
        <Feedback event={event} faults={faults}
          onSubmit={(v, a, n) => onFeedback(event, v, a, n)} onEngage={() => onPin(event)} />

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
          {offline && (
            <span className="note"><Bot size={12} style={{ verticalAlign: -2 }} /> Offline knowledge-graph composer — it quotes the graph and the manual, it is not a language model.</span>
          )}
          {refDocs.length > 0 && equipmentId && ok && !refDocs.some((d) => d.startsWith(equipmentId)) && (
            <span className="note"><AlertTriangle size={12} style={{ verticalAlign: -2 }} /> Procedures cite the
              reference manual {refDocs.join(", ")}, not this machine's own; upload its manual from the machine menu.</span>
          )}
        </div>
      </div>
    </section>
  );
}
