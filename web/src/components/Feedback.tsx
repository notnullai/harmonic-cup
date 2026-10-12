import { useEffect, useState } from "react";
import { Check, ClipboardCheck, X } from "lucide-react";
import type { FeedbackRecord, HEvent } from "../api";

// What the technician found at the machine (HARM-S7-T12). Stored with the
// event, shown in the report, and exported as labelled training data: the
// only ground truth a deployed system ever gets.

const label = (k: string) =>
  k === "healthy" ? "No fault found" : k === "other" ? "Something else"
    : k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " ");

export function verdictText(fb: FeedbackRecord): string {
  if (fb.verdict === "confirmed") return "Confirmed on inspection";
  return fb.actual === "healthy" ? "No fault found on inspection" : `On inspection: ${label(fb.actual ?? "other").toLowerCase()}`;
}

interface Props {
  event: HEvent;
  faults: string[];
  onSubmit: (verdict: "confirmed" | "wrong", actual: string | null, note: string) => Promise<void>;
  // Called when someone starts answering: live mode moves the panel to the
  // newest event every few seconds, so the event being judged is pinned.
  onEngage?: () => void;
}

export default function Feedback({ event, faults, onSubmit, onEngage }: Props) {
  const [mode, setMode] = useState<"idle" | "wrong">("idle");
  const [actual, setActual] = useState("healthy");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { setMode("idle"); setNote(""); setActual("healthy"); }, [event.id]);

  const send = async (verdict: "confirmed" | "wrong") => {
    setBusy(true);
    try { await onSubmit(verdict, verdict === "wrong" ? actual : null, note); setMode("idle"); }
    finally { setBusy(false); }
  };

  const fb = event.feedback;
  if (fb) {
    return (
      <div className={`feedback done ${fb.verdict}`}>
        <ClipboardCheck size={15} />
        <div><b>{verdictText(fb)}</b>{fb.note && <div className="hint">“{fb.note}”</div>}</div>
      </div>
    );
  }
  return (
    <div className="feedback">
      <div className="fb-q"><ClipboardCheck size={15} /><b>After someone has checked the machine: was this right?</b></div>
      {mode === "idle" ? (
        <div className="row">
          <button className="btn sm" disabled={busy} onClick={() => send("confirmed")}><Check size={13} />Confirmed</button>
          <button className="btn sm" disabled={busy} onClick={() => { onEngage?.(); setMode("wrong"); }}><X size={13} />It was something else</button>
        </div>
      ) : (
        <div className="fb-form">
          <label className="hint" htmlFor={`fb-${event.id}`}>What was actually there</label>
          <select id={`fb-${event.id}`} className="input" value={actual} onChange={(e) => setActual(e.target.value)}>
            {["healthy", ...faults, "other"].map((k) => <option key={k} value={k}>{label(k)}</option>)}
          </select>
          <textarea className="input" rows={2} maxLength={1000} placeholder="Note (optional): what you saw, what you did"
            value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="row">
            <button className="btn sm primary" disabled={busy} onClick={() => send("wrong")}>Save</button>
            <button className="btn sm ghost" disabled={busy} onClick={() => setMode("idle")}>Cancel</button>
          </div>
        </div>
      )}
      <span className="hint">Answer once the machine has been inspected. Kept with the alarm and in the report, and it teaches Harmonic: judged alarms become training data for this type of machine.</span>
    </div>
  );
}
