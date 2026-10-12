import { useState } from "react";
import { ArrowLeft, CheckCircle2, ClipboardList, Code2, HardHat, Inbox, Loader2, Send, TriangleAlert, Wrench, X } from "lucide-react";
import { api, type HEvent, type WorkOrder } from "../api";
import { failureMode, when } from "../format";

/** How an order's delivery reads: never as a failure unless it is one. */
export function delivery(d: string | undefined): { text: string; tone: "ok" | "busy" | "bad" } {
  switch (d) {
    case "sent": return { text: "Sent to maintenance", tone: "ok" };
    case "simulated":
    case "not configured": return { text: "Sent to maintenance", tone: "ok" };
    case "pending": return { text: "Sending to maintenance…", tone: "busy" };
    case "failed": return { text: "Not delivered — the maintenance system did not answer", tone: "bad" };
    default:
      // An order recorded before repeats were flagged said so in its delivery.
      if (d?.startsWith("not sent")) return { text: "Already sent — this alarm was added to it", tone: "ok" };
      return { text: d ?? "", tone: "ok" };
  }
}

const RANK: Record<string, number> = { WARNING: 2, CRITICAL: 3, FAILED: 4 };
const localTime = (s: number | string) => new Date(typeof s === "number" ? s * 1000 : s)
  .toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

interface Props {
  /** false: closed; null: the list; an id: that order. */
  view: string | null | false;
  orders: WorkOrder[];
  webhook: string | null;
  events: HEvent[];
  onView: (v: string | null | false) => void;
  onChanged: () => void;
  toast: (text: string, err?: boolean) => void;
}

export default function WorkOrders({ view, orders, webhook, events, onView, onChanged, toast }: Props) {
  const [filter, setFilter] = useState<"open" | "all">("open");
  if (view === false) return null;
  const wo = typeof view === "string" ? orders.find((o) => o.id === view) : undefined;
  const shown = orders.filter((o) => filter === "all" || o.status === "open");
  const target = webhook ? `the maintenance system at ${webhook}` : "the demo maintenance inbox";

  return (
    <>
      <div className="scrim" onClick={() => onView(false)} />
      <aside className="drawer wo" role="dialog" aria-modal="true" aria-labelledby="wo-title">
        <div className="card-h" style={{ padding: "16px 18px" }}>
          {wo ? <button className="btn ghost icon sm" onClick={() => onView(null)} aria-label="All work orders"><ArrowLeft size={16} /></button>
            : <ClipboardList size={18} />}
          <h2 id="wo-title" style={{ fontSize: 15 }}>{wo ? wo.id : "Work orders"}</h2>
          <span className="grow" />
          <button className="btn ghost icon sm" onClick={() => onView(false)} aria-label="Close"><X size={16} /></button>
        </div>
        {wo ? <Detail wo={wo} events={events} target={target} onChanged={onChanged} toast={toast} />
          : (
            <div className="wo-body">
              <p className="hint" style={{ marginTop: 0 }}>
                <Inbox size={12} style={{ verticalAlign: -2 }} /> When a diagnosis settles on WARNING or worse,
                a job is sent to {target}{webhook ? "" : " — a stand-in for SAP PM or IBM Maximo, which take the same order through a webhook"}.
              </p>
              <div className="row" style={{ marginBottom: 10 }}>
                {(["open", "all"] as const).map((k) => (
                  <button key={k} className="chip toggle" aria-pressed={filter === k} onClick={() => setFilter(k)}>
                    {k === "open" ? `Open (${orders.filter((o) => o.status === "open").length})` : `All (${orders.length})`}
                  </button>
                ))}
              </div>
              {shown.length === 0 && <div className="empty"><CheckCircle2 size={26} />
                <div>{filter === "open" ? "No open work orders." : "No work orders yet."}</div>
                <div className="hint">One is sent when a machine reaches WARNING with a diagnosis.</div></div>}
              <ul className="wo-list">
                {shown.map((o) => (
                  <li key={o.id}>
                    <button onClick={() => onView(o.id)}>
                      <span className={`prio p${o.priority}`}>P{o.priority}</span>
                      <span className="wo-main">
                        <b>{failureMode(o.failure_mode_id)}</b>
                        <span className="hint">{o.machine} · {o.equipment_id} · {localTime(o.opened_at)}</span>
                      </span>
                      <span className={`wo-status ${o.status}`}>{o.status === "open" ? "Open" : "Done"}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
      </aside>
    </>
  );
}

function Detail({ wo, events, target, onChanged, toast }: {
  wo: WorkOrder; events: HEvent[]; target: string; onChanged: () => void; toast: (t: string, e?: boolean) => void;
}) {
  const [raw, setRaw] = useState(false);
  const [closing, setClosing] = useState(false);
  const d = delivery(wo.delivery);
  // The order's history, from the alarms it holds: the first opened it, a
  // worse state re-sent it as an update, the rest were added without a send.
  const byId = new Map(events.map((e) => [e.id, e]));
  const mine = wo.event_ids.map((id) => byId.get(id)).filter((e): e is HEvent => !!e);
  let worst = 0;
  const activity = mine.map((e, i) => {
    const r = RANK[e.state] ?? 0;
    const kind = i === 0 ? "opened" : r > worst ? "escalated" : "repeat";
    worst = Math.max(worst, r);
    return { e, kind };
  });
  const repeats = activity.filter((a) => a.kind === "repeat").length;

  const close = async () => {
    setClosing(true);
    try {
      await api.closeWorkOrder(wo.id);
      toast(`${wo.id} closed — the next alarm for this fault opens a new order`);
      onChanged();
    } catch (e) { toast((e as Error).message, true); } finally { setClosing(false); }
  };

  return (
    <div className="wo-body">
      <div className="wo-hero">
        <span className={`prio p${wo.priority}`}>P{wo.priority} · {wo.priority === 1 ? "urgent" : "high"}</span>
        <span className={`wo-status ${wo.status}`}>{wo.status === "open" ? "Open" : "Done"}</span>
        <span className={`wo-deliv ${d.tone}`}>
          {d.tone === "busy" ? <Loader2 size={12} className="spin" /> : d.tone === "bad" ? <TriangleAlert size={12} /> : <Send size={12} />}
          {d.text}</span>
      </div>
      <h3 className="wo-title">{failureMode(wo.failure_mode_id)} suspected</h3>
      <div className="hint">{wo.machine} · {wo.equipment_id} · raised at {wo.state}
        {wo.confidence != null && <> · {Math.round(wo.confidence * 100)}% confidence</>}
        {wo.remaining_life_hours != null && wo.remaining_life_hours > 0 && <> · about {Math.round(wo.remaining_life_hours)} h of life left</>}</div>
      {wo.summary && <p className="dx-text" style={{ marginTop: 10 }}>{wo.summary}</p>}

      {wo.actions.length > 0 && (<>
        <div className="wo-sec">Tasks</div>
        <ol className="steps">
          {wo.actions.map((a, i) => (
            <li key={i}><span className="n">{i + 1}</span><div><div>{a.step}</div>
              {(a.tool || a.duration_min) && <div className="meta"><Wrench size={11} style={{ verticalAlign: -1 }} /> {a.tool ?? "—"}
                {a.duration_min ? ` · ${Math.round(a.duration_min)} min` : ""}{a.skill_level ? ` · ${a.skill_level}` : ""}</div>}</div></li>
          ))}
        </ol></>)}
      {wo.safety.length > 0 && (<>
        <div className="wo-sec"><HardHat size={12} style={{ verticalAlign: -2 }} /> Safety</div>
        <ul className="bullets">{wo.safety.map((s, i) => <li key={i}>{s}</li>)}</ul></>)}
      {wo.references.length > 0 && <div className="refs" style={{ marginTop: 10 }}>{wo.references.map((r) => <code key={r}>{r}</code>)}</div>}

      <div className="wo-sec">Activity</div>
      <ul className="wo-log">
        {activity.length === 0 && <li><span className="dot" />Opened and sent to {target} · {localTime(wo.opened_at)}
          {wo.alarms > 1 && ` · ${wo.alarms} alarms on this order`}</li>}
        {activity.map(({ e, kind }) => kind === "repeat" ? null : (
          <li key={e.id}><span className={`dot s-${e.state}`} />
            {kind === "opened" ? `Opened at ${e.state} and sent to ${target}` : `Escalated to ${e.state} — update sent`}
            <span className="hint"> · {when(e.machine_hours, e.recorded_at)}</span></li>
        ))}
        {repeats > 0 && <li><span className="dot" />{repeats} repeat alarm{repeats > 1 ? "s" : ""} added to this order — not sent again, so the planner gets one job, not a stream</li>}
        {wo.status === "closed" && <li><span className="dot s-OK" />Closed — job done · {localTime(wo.updated_at)}</li>}
      </ul>

      <button className="btn sm ghost" style={{ marginTop: 12 }} onClick={() => setRaw(!raw)} aria-expanded={raw}>
        <Code2 size={13} />{raw ? "Hide" : "Show"} what was sent</button>
      {raw && <pre className="wo-raw">{JSON.stringify({
        type: "harmonic.work_order", id: wo.id, update: wo.update, equipment_id: wo.equipment_id,
        title: wo.title, priority: wo.priority, state: wo.state, failure_mode_id: wo.failure_mode_id,
        confidence: wo.confidence, summary: wo.summary, actions: wo.actions, safety: wo.safety,
        references: wo.references, remaining_life_hours: wo.remaining_life_hours, event_ids: wo.event_ids,
      }, null, 2)}</pre>}

      <p className="hint" style={{ marginTop: 12 }}>{wo.advisory}</p>
      {wo.status === "open" && (
        <div className="wo-foot">
          <span className="hint">Repair done? Close the order; a new alarm for this fault then opens a new job.</span>
          <button className="btn primary" disabled={closing} onClick={close}><CheckCircle2 size={15} />Mark job done</button>
        </div>
      )}
    </div>
  );
}
