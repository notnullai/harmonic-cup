import { useEffect, useMemo, useState } from "react";
import { FileDown, X } from "lucide-react";
import { api, type Point } from "../api";
import { atOrBefore, when } from "../format";
import type { View } from "./Timeline";

interface Props {
  open: boolean;
  points: Point[];
  view: View | null;
  cursorH: number | null;
  onClose: () => void;
  onDone: (msg: string, err?: boolean) => void;
}

type Range = "view" | "all" | "custom";

export default function ExportDialog({ open, points, view, cursorH, onClose, onDone }: Props) {
  const [range, setRange] = useState<Range>("view");
  const first = points[0]?.h ?? 0;
  const last = points[points.length - 1]?.h ?? 0;
  const [from, setFrom] = useState(first);
  const [to, setTo] = useState(last);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setFrom(view?.start ?? first);
    setTo(view?.end ?? last);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const label = (h: number) => when(h, atOrBefore(points, h, (p) => p.h)?.at ?? null);
  const chosen = useMemo((): [number | undefined, number | undefined] => {
    if (range === "all") return [undefined, undefined];
    if (range === "view") return [view?.start ?? first, view?.end ?? last];
    return [Math.min(from, to), Math.max(from, to)];
  }, [range, view, from, to, first, last]);

  if (!open) return null;

  const download = async () => {
    setBusy(true);
    try {
      const r = await fetch(api.reportUrl(chosen[0], chosen[1]));
      if (!r.ok) throw new Error(`${r.status}`);
      const blob = await r.blob();
      const cd = r.headers.get("content-disposition") ?? "";
      const name = /filename="([^"]+)"/.exec(cd)?.[1] ?? "harmonic-report.pdf";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      onDone(`Report saved — ${name}`);
      onClose();
    } catch (e) {
      onDone(`The report could not be built (${(e as Error).message})`, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="exp-title">
        <div className="body">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <FileDown size={20} /><h3 id="exp-title">Export report</h3><span style={{ flex: 1 }} />
            <button className="btn ghost icon sm" onClick={onClose} aria-label="Close"><X size={16} /></button>
          </div>
          <p className="hint" style={{ margin: "8px 0 0" }}>
            A PDF with the state summary, the health trend with alarm bands, every event in the period
            and the full diagnosis — actions, safety and sources.
          </p>
          <div className="radio">
            <label><input type="radio" name="r" checked={range === "view"} onChange={() => setRange("view")} />
              <span>Visible on the timeline<small>{label(view?.start ?? first)} → {label(view?.end ?? last)}</small></span></label>
            <label><input type="radio" name="r" checked={range === "all"} onChange={() => setRange("all")} />
              <span>Everything kept<small>{label(first)} → {label(last)} · {points.length} readings</small></span></label>
            <label><input type="radio" name="r" checked={range === "custom"} onChange={() => setRange("custom")} />
              <span>Custom period, in machine hours<small>{label(Math.min(from, to))} → {label(Math.max(from, to))}</small></span></label>
          </div>
          {range === "custom" && (
            <div className="inputs">
              <label className="hint">From (h)<input type="number" step="0.1" value={from.toFixed(2)}
                onChange={(e) => setFrom(Number(e.target.value))} /></label>
              <label className="hint">To (h)<input type="number" step="0.1" value={to.toFixed(2)}
                onChange={(e) => setTo(Number(e.target.value))} /></label>
              {cursorH !== null && (
                <button className="btn sm" style={{ gridColumn: "1 / -1" }}
                  onClick={() => setTo(cursorH)}>End the period at the cursor ({label(cursorH)})</button>
              )}
            </div>
          )}
        </div>
        <div className="foot">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || points.length < 2} onClick={download}>
            <FileDown size={15} />{busy ? "Building…" : "Download PDF"}</button>
        </div>
      </div>
    </>
  );
}
