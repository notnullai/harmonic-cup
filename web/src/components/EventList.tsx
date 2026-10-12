import { BellOff, ClipboardCheck } from "lucide-react";
import type { HEvent } from "../api";
import { failureMode, when } from "../format";

interface Props { events: HEvent[]; selectedId: string | undefined; onSelect: (e: HEvent) => void }

export default function EventList({ events, selectedId, onSelect }: Props) {
  return (
    <section className="card">
      <div className="card-h"><h2>Events</h2><span className="grow" /><span className="chip num">{events.length}</span></div>
      <div className="card-b" style={{ paddingTop: 8 }}>
        {events.length === 0 ? (
          <div className="empty"><BellOff size={26} /><div>No events raised.</div></div>
        ) : (
          <ul className="events">
            {[...events].reverse().map((e) => {
              const v = e.diagnosis_status === "ok" && e.diagnosis?.failure_mode_id
                ? failureMode(e.diagnosis.failure_mode_id)
                : e.diagnosis_status === "fallback" ? "Insufficient evidence" : "Diagnosing…";
              return (
                <li key={e.id} className={`s-${e.state}`}>
                  <button aria-current={e.id === selectedId} onClick={() => onSelect(e)}>
                    <span className="dot" />
                    <span>
                      <div className="v">{v}{e.feedback && <ClipboardCheck size={12} className="fb-mark"
                        aria-label={e.feedback.verdict === "confirmed" ? "confirmed on inspection" : "corrected on inspection"}
                        style={{ marginLeft: 6, verticalAlign: -1, color: e.feedback.verdict === "confirmed" ? "var(--ok)" : "var(--warning)" }} />}</div>
                      <div className="t num">{when(e.machine_hours, e.recorded_at)} · index {e.hhi.toFixed(2)}</div>
                    </span>
                    <span className="state-text" style={{ fontSize: 11, letterSpacing: ".05em" }}>{e.state}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
