import { useEffect, useState } from "react";
import { History, X } from "lucide-react";
import { api, type MachineRecord, type RecordingInfo, type WorkOrder } from "../api";
import { failureMode } from "../format";
import { BaselinePicker, Commission } from "./Wizard";

interface Props {
  machineId: string | null;
  /** "iso": opened to set the vibration zones; the healthy period stays. */
  mode?: "baseline" | "iso";
  onClose: () => void;
  toast: (text: string, err?: boolean) => void;
  /** Every work order; the machine's open ones can be closed with the re-learn. */
  orders: WorkOrder[];
  onOrdersClosed: () => void;
}

/**
 * Re-learn normal after maintenance (HARM-S7-T09). After a repair the machine
 * sounds different and healthy; it is commissioned again on a healthy period
 * after the work. The old brain stays in service until the new one is proven,
 * the history and events stay, and the timeline marks the change.
 */
export default function Rebaseline({ machineId, mode = "baseline", onClose, toast, orders, onOrdersClosed }: Props) {
  const [m, setM] = useState<MachineRecord | null>(null);
  const [recs, setRecs] = useState<RecordingInfo[]>([]);
  const [recKey, setRecKey] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<[number, number] | null>(null);
  const [note, setNote] = useState("");
  const [minReadings, setMinReadings] = useState(160);
  const [isoClasses, setIsoClasses] = useState<Record<string, string>>({});
  const [iso, setIso] = useState("none");
  const [job, setJob] = useState<MachineRecord | null>(null);
  const [submitting, setSubmitting] = useState(false);
  // The machine's open jobs (every point of it): re-learning after a repair
  // usually means they are done.
  const mine = orders.filter((o) => o.status === "open" && machineId
    && (o.source === machineId || o.source.startsWith(`${machineId}@`)));
  const [closeIds, setCloseIds] = useState<Set<string>>(new Set());
  useEffect(() => {
    setCloseIds(new Set(mode === "iso" ? [] : mine.map((o) => o.id)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [machineId, mode]);

  useEffect(() => {
    setM(null); setJob(null); setBaseline(null); setNote("");
    if (!machineId) return;
    api.machine(machineId).then((r) => {
      setM(r); setRecKey(r.recording); setIso(r.iso_class);
      // Setting zones keeps the period the machine already learned from.
      if (mode === "iso") { setBaseline(r.baseline_h); setNote("Vibration zones set"); }
    }).catch((e: Error) => toast(e.message, true));
    api.recordings().then((r) => setRecs(r.recordings)).catch(() => {});
    api.machines().then((c) => { setMinReadings(c.min_baseline_readings); setIsoClasses(c.iso_classes); }).catch(() => {});
  }, [machineId, mode, toast]);

  // Poll while it commissions; it ends ready with a new baseline, or ready
  // with the old one and the reason it was not taken.
  useEffect(() => {
    if (!job || !machineId || job.status !== "commissioning" && job.status !== "draft") return;
    const t = setTimeout(() => api.machine(machineId).then(setJob).catch(() => {}), 700);
    return () => clearTimeout(t);
  }, [job, machineId]);

  if (!machineId) return null;
  const isLive = !!m?.live;
  const failed = job?.rebaseline_error;
  const done = job && job.status === "ready";

  const submit = async () => {
    if (!m || !baseline) return;
    setSubmitting(true);
    try {
      setJob(await api.rebaseline(m.id, { baseline_h: baseline, recording: recKey ?? undefined, note, iso_class: iso }));
      const closing = mine.filter((o) => closeIds.has(o.id));
      if (closing.length) {
        await Promise.all(closing.map((o) => api.closeWorkOrder(o.id).catch(() => {})));
        toast(`Closed ${closing.map((o) => o.id).join(", ")}: the job is done`);
        onOrdersClosed();
      }
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="scrim" />
      <div className="dialog wizard" role="dialog" aria-modal="true" aria-labelledby="rb-title">
        <div className="wz-head">
          <History size={20} />
          <div>
            <h3 id="rb-title">{mode === "iso" ? "Set vibration zones" : "Re-learn normal"}{m ? ` — ${m.name}` : ""}</h3>
            <div className="hint">{mode === "iso"
              ? "Published ISO limits make a severe vibration level an alarm whatever the machine learned. The machine is commissioned again on the same healthy period, with the zones added; its history stays."
              : "After maintenance the machine sounds different. Pick a healthy period after the work; the machine keeps its history and the timeline marks the change."}</div>
          </div>
          <span style={{ flex: 1 }} />
          <button className="btn ghost icon sm" onClick={onClose} aria-label="Close"
            disabled={job?.status === "commissioning"}><X size={16} /></button>
        </div>

        {m && !job && (
          <div className="body wz-body">
            <p className="hint">
              Learned now: {m.baseline_h[0].toFixed(1)}–{m.baseline_h[1].toFixed(1)} h of {m.recording}
              {m.baselines?.length ? ` · re-learned ${m.baselines.length} time${m.baselines.length > 1 ? "s" : ""} before` : ""}.
            </p>
            <label className="hint">Vibration zones (ISO 10816)
              <select className="input" value={iso} onChange={(e) => setIso(e.target.value)}>
                {Object.entries(isoClasses).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
              </select>
              {iso === "none"
                ? <span className="warn-text">Without zones, absolute vibration limits are not checked.</span>
                : <span>Choose the class for the machine's size and mounting.</span>}
            </label>
            {mode !== "iso" && (isLive ? (
              <label className="hint">Recording to learn from
                <select className="input" value={recKey ?? ""} onChange={(e) => { setRecKey(e.target.value); setBaseline(null); }}>
                  {recs.map((r) => <option key={r.key} value={r.key}>{r.name}</option>)}
                </select>
                <span>Record the machine after the repair with the wizard's Live sensor tab, then pick that recording here.</span>
              </label>
            ) : (
              <p className="hint">This machine replays a recording, so it re-learns from a later period of the same recording.</p>
            ))}
            {recKey && (mode === "iso" ? (
              <details>
                <summary className="hint">Healthy period: {m.baseline_h[0].toFixed(1)}–{m.baseline_h[1].toFixed(1)} h, as learned. Change it</summary>
                <BaselinePicker recKey={recKey} channel={m.channel} min={minReadings} value={baseline} onChange={setBaseline} />
              </details>
            ) : <BaselinePicker recKey={recKey} channel={m.channel} min={minReadings} value={baseline} onChange={setBaseline} />)}
            {mode !== "iso" && mine.length > 0 && (
              <div className="wo-close">
                <div className="hint">Open work orders for this machine</div>
                {mine.map((o) => (
                  <label key={o.id} className="check">
                    <input type="checkbox" checked={closeIds.has(o.id)} onChange={(e) => {
                      const s = new Set(closeIds);
                      if (e.target.checked) s.add(o.id); else s.delete(o.id);
                      setCloseIds(s);
                    }} />
                    <span>Close <b>{o.id}</b> — {failureMode(o.failure_mode_id).toLowerCase()} repaired</span>
                  </label>
                ))}
              </div>
            )}
            <label className="hint">What was done (shown on the timeline)
              <input className="input" maxLength={200} value={note} placeholder="e.g. drive-end bearing replaced"
                onChange={(e) => setNote(e.target.value)} /></label>
            <button className="btn primary" disabled={submitting || !baseline || (mode === "iso" && iso === "none")} onClick={submit}>
              <History size={15} />{mode === "iso" ? "Set zones and commission again"
                : `Re-learn from ${baseline ? `${baseline[0].toFixed(1)}–${baseline[1].toFixed(1)} h` : "…"}`}</button>
          </div>
        )}

        {job && (
          <div className="body wz-body">
            {failed ? (
              <div className="note bad-note"><b>Not re-learned; the machine kept its brain.</b> {failed}</div>
            ) : (
              <Commission job={job} name={job.name} rec={recs.find((r) => r.key === job.recording)}
                channel={job.channel} baseline={job.baseline_h} submitting={false} onSubmit={() => {}} />
            )}
            {(done || failed) && <button className="btn" onClick={onClose}>Close</button>}
          </div>
        )}
      </div>
    </>
  );
}
