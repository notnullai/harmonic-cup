// Plain words shared by the page and the downloaded report (S10-T06): human
// times, the alarm levels, and the "how far from normal" chart. The chart is
// built as an HTML string so the page and the downloaded file draw it the
// same way; every value in it is a number the server computed (api/try_mode.py
// `_series`), and every text is escaped.

import type { Report } from "./api";

/** "40 seconds", "12 min", "1 h 21 min", "26 hours", "3.4 days" — the same
 *  rules as api/try_mode.py `human_duration`. */
export function humanDuration(h: number): string {
  const sec = Math.max(0, h * 3600);
  if (sec < 90) { const n = Math.round(sec); return `${n} second${n === 1 ? "" : "s"}`; }
  const mins = Math.round(sec / 60);
  if (mins < 60) return `${mins} min`;
  if (h < 10) { const hh = Math.floor(mins / 60), mm = mins % 60; return mm ? `${hh} h ${mm} min` : `${hh} h`; }
  if (h < 48) return `${Math.round(h)} hours`;
  return `${(h / 24).toFixed(1)} days`;
}

/** A recording's length: "45 min", "5.6 hours", "6.8 days" (api `human_span`). */
export function humanSpan(h: number): string {
  if (h < 1) return humanDuration(h);
  if (h < 48) return `${h.toFixed(1)} hours`;
  return `${(h / 24).toFixed(1)} days`;
}

/** Seconds between snapshots, as a person says them: "1 minute", "20 minutes", "1 hour". */
export function humanInterval(s: number): string {
  if (s >= 3600 && Number.isInteger(s / 3600)) { const n = s / 3600; return `${n} hour${n === 1 ? "" : "s"}`; }
  if (s >= 60 && Number.isInteger(s / 60)) { const n = s / 60; return `${n} minute${n === 1 ? "" : "s"}`; }
  return `${+s.toFixed(2)} second${s === 1 ? "" : "s"}`;
}

export const esc = (v: unknown): string => String(v ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

/** The three alarm levels, in plain words, for the reader who acts on them. */
export const LEVELS: { key: "WATCH" | "WARNING" | "CRITICAL"; name: string; means: string }[] = [
  { key: "WATCH", name: "Early sign (Watch)",
    means: "Something has started to change. Plan a check at the next convenient stop." },
  { key: "WARNING", name: "Warning",
    means: "The change is clear. Schedule an inspection soon, before it gets worse." },
  { key: "CRITICAL", name: "Critical",
    means: "The change is severe. Treat it as urgent and follow your plant's own procedure." },
];

export const UNRECOGNISED_NOTE = "An unrecognised change means the machine moved away from "
  + "its own normal in a way that doesn't match a fault Harmonic knows for the machine type "
  + "you chose, so it can say that something changed, and where, but not which fault it is.";

// --- the chart ----------------------------------------------------------------------------

/** Styles for the chart, used by the page and by the downloaded file. Colours
 *  come from --hc-* variables, which each of them sets. */
export const CHART_CSS = `
.hc{margin:0}
.hc-plot{position:relative;height:190px;border-bottom:1px solid var(--hc-axis);border-left:1px solid var(--hc-axis)}
.hc-plot svg{position:absolute;inset:0;width:100%;height:100%;display:block;overflow:visible}
.hc-learn{fill:var(--hc-learn)}
.hc-line{fill:none;stroke:var(--hc-line);stroke-width:2;vector-effect:non-scaling-stroke;stroke-linejoin:round}
.hc-alarm{stroke:var(--hc-alarm);stroke-width:1.5;stroke-dasharray:6 4;vector-effect:non-scaling-stroke}
.hc-normal{stroke:var(--hc-axis);stroke-width:1;stroke-dasharray:2 4;vector-effect:non-scaling-stroke}
.hc-first{stroke:var(--hc-first);stroke-width:2;vector-effect:non-scaling-stroke}
.hc-lab{position:absolute;font-size:11.5px;line-height:1.2;color:var(--hc-text);white-space:nowrap;pointer-events:none}
.hc-lab.l-learn{top:4px;left:6px;color:var(--hc-text)}
.hc-lab.l-alarm{right:4px;transform:translateY(-100%);color:var(--hc-alarm);font-weight:600}
.hc-lab.l-normal{right:4px;transform:translateY(-100%)}
.hc-lab.l-first{top:4px;color:var(--hc-first);font-weight:700;background:var(--hc-bg);padding:0 3px;border-radius:3px}
.hc-lab.l-clip{top:4px;right:4px}
.hc-x{display:flex;justify-content:space-between;font-size:11.5px;color:var(--hc-text);margin-top:4px}
.hc figcaption{font-size:12.5px;color:var(--hc-text);margin-top:8px;line-height:1.45}
`;

/** Hex colours for the downloaded file (the page maps --hc-* to its theme). */
export const CHART_VARS_LIGHT = "--hc-axis:#c3ccd6;--hc-learn:#e3eefb;--hc-line:#23364a;"
  + "--hc-alarm:#c0561a;--hc-first:#b3261e;--hc-text:#51606f;--hc-bg:#ffffff;";

export function chartHtml(r: Report): string {
  const c = r.chart;
  if (!c || c.points.length < 2) return "";
  const end = Math.max(c.end_h, 1e-9);
  const maxD = Math.max(...c.points.map((p) => p.d ?? 0), c.normal);
  // Up to three times the alarm level: past that the line is clipped (and
  // says how high it went), so the learning period never looks flat.
  const top = Math.min(Math.max(maxD * 1.1, c.alarm * 1.5), c.alarm * 3);
  const X = (h: number) => (h / end) * 1000;
  const Y = (d: number) => 240 - (Math.min(d, top) / top) * 240;
  const pct = (h: number) => `${((h / end) * 100).toFixed(2)}%`;
  const ypct = (d: number) => `${((Y(d) / 240) * 100).toFixed(2)}%`;
  let path = "";
  let pen = false;
  for (const p of c.points) {
    if (p.d === null) { pen = false; continue; }
    path += `${pen ? "L" : "M"}${X(p.h).toFixed(1)},${Y(p.d).toFixed(1)}`;
    pen = true;
  }
  const [b0, b1] = r.baseline_h;
  const first = c.first_alarm_h;
  const firstLeft = first !== null && first / end > 0.7;
  const firstText = r.first_alarm ? `First alarm: ${r.first_alarm.before_end} before the end` : "";
  const label = (cls: string, style: string, text: string) =>
    `<span class="hc-lab ${cls}" style="${style}">${esc(text)}</span>`;
  const svg = `<svg viewBox="0 0 1000 240" preserveAspectRatio="none" aria-hidden="true">`
    + `<rect class="hc-learn" x="${X(b0).toFixed(1)}" y="0" width="${Math.max(X(b1) - X(b0), 1).toFixed(1)}" height="240"/>`
    + `<line class="hc-normal" x1="0" x2="1000" y1="${Y(c.normal).toFixed(1)}" y2="${Y(c.normal).toFixed(1)}"/>`
    + `<line class="hc-alarm" x1="0" x2="1000" y1="${Y(c.alarm).toFixed(1)}" y2="${Y(c.alarm).toFixed(1)}"/>`
    + `<path class="hc-line" d="${path}"/>`
    + (first !== null ? `<line class="hc-first" x1="${X(first).toFixed(1)}" x2="${X(first).toFixed(1)}" y1="0" y2="240"/>` : "")
    + `</svg>`;
  const labels = label("l-learn", `left:calc(${pct(b0)} + 6px)`, "Learning normal")
    + label("l-alarm", `top:${ypct(c.alarm)}`, `Alarm level (${c.alarm}×)`)
    + label("l-normal", `top:${ypct(c.normal)}`, "Edge of normal (1×)")
    + (first !== null ? label("l-first", firstLeft
      ? `right:calc(100% - ${pct(first)} + 4px);top:22px` : `left:calc(${pct(first)} + 4px);top:22px`, "First alarm") : "")
    + (maxD > top ? label("l-clip", "", `Peaks go off the chart, up to ${Math.round(maxD)}×`) : "");
  const alt = `How far the machine was from its own normal over ${r.span_text}. `
    + (r.first_alarm ? `${firstText}.` : "No alarm.");
  const caption = "The line is how far the machine was from its own normal at each snapshot. "
    + "1× is the edge of normal: the most it moved while running normally. A change that "
    + `stays above the dashed alarm level (${c.alarm}×) for several snapshots in a row raises an early sign.`
    + (r.names_faults ? " For this machine type Harmonic can also warn below that line, when it "
      + "recognises the pattern of a known fault." : "")
    + " The shaded part is the learning period.";
  return `<figure class="hc" role="img" aria-label="${esc(alt)}"><div class="hc-plot">${svg}${labels}</div>`
    + `<div class="hc-x"><span>Start</span><span>${esc(r.span_text)}</span></div>`
    + `<figcaption>${esc(caption)}</figcaption></figure>`;
}

/** The same chart, as a standalone SVG document with literal colours (no
 *  CSS variables or classes) — for the PDF report (S10-T06), where it is
 *  parsed with `DOMParser` and drawn as vector graphics by svg2pdf.js. The
 *  text labels `chartHtml` positions with CSS are drawn separately, as
 *  jsPDF text, next to the chart. */
export function chartSvgStandalone(r: Report): string {
  const c = r.chart;
  if (!c || c.points.length < 2) return "";
  const end = Math.max(c.end_h, 1e-9);
  const maxD = Math.max(...c.points.map((p) => p.d ?? 0), c.normal);
  const top = Math.min(Math.max(maxD * 1.1, c.alarm * 1.5), c.alarm * 3);
  const X = (h: number) => (h / end) * 1000;
  const Y = (d: number) => 240 - (Math.min(d, top) / top) * 240;
  let path = "";
  let pen = false;
  for (const p of c.points) {
    if (p.d === null) { pen = false; continue; }
    path += `${pen ? "L" : "M"}${X(p.h).toFixed(1)},${Y(p.d).toFixed(1)}`;
    pen = true;
  }
  const [b0, b1] = r.baseline_h;
  const first = c.first_alarm_h;
  // The same hex colours as CHART_VARS_LIGHT (the downloaded HTML's light
  // theme), written literally: svg2pdf.js draws this SVG outside any page
  // that could supply --hc-* custom properties.
  const AXIS = "#c3ccd6", LEARN = "#e3eefb", LINE = "#23364a", ALARM = "#c0561a", FIRST = "#b3261e";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 240" width="1000" height="240">`
    + `<rect x="0" y="0" width="1000" height="240" fill="#ffffff"/>`
    + `<rect x="${X(b0).toFixed(1)}" y="0" width="${Math.max(X(b1) - X(b0), 1).toFixed(1)}" height="240" fill="${LEARN}"/>`
    + `<line x1="0" x2="1000" y1="${Y(c.normal).toFixed(1)}" y2="${Y(c.normal).toFixed(1)}" `
    + `stroke="${AXIS}" stroke-width="1" stroke-dasharray="2 4"/>`
    + `<line x1="0" x2="1000" y1="${Y(c.alarm).toFixed(1)}" y2="${Y(c.alarm).toFixed(1)}" `
    + `stroke="${ALARM}" stroke-width="1.5" stroke-dasharray="6 4"/>`
    + `<path d="${path}" fill="none" stroke="${LINE}" stroke-width="2" stroke-linejoin="round"/>`
    + (first !== null ? `<line x1="${X(first).toFixed(1)}" x2="${X(first).toFixed(1)}" y1="0" y2="240" `
      + `stroke="${FIRST}" stroke-width="2"/>` : "")
    + `<rect x="0" y="0" width="1000" height="240" fill="none" stroke="${AXIS}" stroke-width="1"/>`
    + `</svg>`;
}
