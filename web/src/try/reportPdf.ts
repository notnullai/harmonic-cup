// "Download report" as a PDF (S10-T06, owner-approved 2026-09-29): the same
// content as reportHtml.ts's HTML download, built from the same report data,
// laid out on A4 with jsPDF, wrapped and page-broken by pdfText.ts's helpers
// so a page break never falls inside a line. The chart is drawn as vector
// graphics by svg2pdf.js, from the same geometry format.ts's chartHtml uses
// (chartSvgStandalone — literal colours, no CSS variables, since this SVG is
// parsed and drawn outside any page).
//
// jspdf and svg2pdf.js are loaded only when the button is clicked (the
// dynamic imports below), so neither adds anything to the page's first
// load. Every string placed on the page comes from the report's own data:
// jsPDF's text()/textWithLink() calls write plain text, never HTML, so
// there is no injection path here — jsPDF's .html() method (which would
// parse HTML) is never called, on purpose.

import type { Report, Run } from "./api";
import { LEVELS, UNRECOGNISED_NOTE, chartSvgStandalone } from "./format";
import { wrapText } from "./pdfText";

// A4 portrait, in points (jsPDF's "pt" unit) — 595.28 x 841.89.
const PAGE_W = 595.28;
const PAGE_H = 841.89;
const MARGIN = 42;
const MUTED: [number, number, number] = [91, 104, 120];
const INK: [number, number, number] = [28, 35, 48];
const LINK: [number, number, number] = [31, 95, 191];

type JsPdfDoc = import("jspdf").jsPDF;

/** A simple top-to-bottom cursor over a jsPDF document: every write wraps
 *  its text to the page's content width (pdfText.ts's `wrapText`, measured
 *  by the real font jsPDF has loaded) and adds a page, never mid-line,
 *  whenever the next line would cross the bottom margin. */
class Writer {
  y = MARGIN;
  readonly left = MARGIN;
  readonly right = PAGE_W - MARGIN;
  readonly bottom = PAGE_H - MARGIN;

  constructor(private doc: JsPdfDoc) {}

  /** Reserves `height` at the current y, breaking to a new page first if it
   *  would cross the bottom margin. Returns the y the caller should draw at. */
  ensure(height: number): number {
    if (this.y + height > this.bottom) {
      this.doc.addPage();
      this.y = MARGIN;
    }
    return this.y;
  }

  heading(text: string): void {
    this.ensure(30);
    this.y += 12;
    this.doc.setFont("helvetica", "bold").setFontSize(12.5).setTextColor(...INK);
    this.doc.text(text, this.left, this.y);
    this.y += 8;
    this.doc.setDrawColor(210, 216, 224).line(this.left, this.y, this.right, this.y);
    this.y += 12;
  }

  para(text: string, opts: { size?: number; bold?: boolean; color?: [number, number, number];
    gap?: number } = {}): void {
    if (!text) return;
    const size = opts.size ?? 10;
    this.doc.setFont("helvetica", opts.bold ? "bold" : "normal").setFontSize(size)
      .setTextColor(...(opts.color ?? INK));
    const lineHeight = size * 1.4;
    const lines = wrapText(text, this.right - this.left, (s) => this.doc.getTextWidth(s));
    for (const line of lines) {
      this.ensure(lineHeight);
      this.doc.text(line, this.left, this.y);
      this.y += lineHeight;
    }
    this.y += opts.gap ?? 5;
  }

  bullets(items: string[], opts: { ordered?: boolean } = {}): void {
    if (!items.length) return;
    const size = 10;
    const indent = 14;
    const lineHeight = size * 1.4;
    this.doc.setFont("helvetica", "normal").setFontSize(size).setTextColor(...INK);
    items.forEach((item, i) => {
      const marker = opts.ordered ? `${i + 1}.` : "–";
      const lines = wrapText(item, this.right - this.left - indent, (s) => this.doc.getTextWidth(s));
      lines.forEach((line, j) => {
        this.ensure(lineHeight);
        if (j === 0) this.doc.text(marker, this.left, this.y);
        this.doc.text(line, this.left + indent, this.y);
        this.y += lineHeight;
      });
    });
    this.y += 5;
  }

  link(label: string, url: string): void {
    const size = 10.5;
    this.ensure(size * 1.4);
    this.doc.setFont("helvetica", "bold").setFontSize(size).setTextColor(...LINK);
    this.doc.textWithLink(label, this.left, this.y, { url });
    this.y += size * 1.4 + 6;
  }
}

const timeBetween = (r: Report): string => r.timed_by === "timestamps"
  ? "from the recording's time stamps"
  : `${r.interval_text ?? ""}${r.timed_by === "wav" ? " (back to back, from the WAV file)" : " (as chosen on upload)"}`;

const runSettings = (r: Report): string[] => [
  `Machine type: ${r.family_label}`,
  `Recording: ${r.n_snapshots} snapshots over ${r.span_text}, ${r.sample_rate_hz} readings per `
    + `second${r.n_channels > 1 ? `, channel ${r.channel} of ${r.n_channels}` : ""}`,
  `Time between snapshots: ${timeBetween(r)}`,
  `Learning period: the first ${r.baseline_text}; false alarms while learning: ${r.false_alarms_in_baseline}`,
  `Shaft speed: ${r.rpm_used} rpm${r.rpm_limited ? " (assumed)" : ""}`,
];

/** Builds the PDF and returns the jsPDF document — call `.save(filename)`
 *  in the browser, or `.output("arraybuffer")` (e.g. in Node, for a
 *  generated-file check) to get its bytes. */
export async function buildReportPdf(run: Run, r: Report, contact: string | null,
  retentionDays: number): Promise<JsPdfDoc> {
  const { jsPDF } = await import("jspdf");

  const doc = new jsPDF({ unit: "pt", format: "a4", orientation: "portrait" });
  doc.setProperties({ title: "Harmonic report" });
  const w = new Writer(doc);

  w.para(r.headline, { size: 15, bold: true, gap: 10 });
  if (r.sample) {
    w.para(`Sample recording: ${r.sample.title}. The real bearing ended with an outer-race `
      + `failure. ${r.sample.credit}`, { size: 9, color: MUTED });
  }

  w.heading("Run details");
  w.bullets(runSettings(r));

  if (r.chart && r.chart.points.length >= 2) {
    w.heading("How far from normal");
    const svgText = chartSvgStandalone(r);
    // DOMParser exists in every browser; a Node-side generator (the
    // check script) has none, so the chart is skipped there rather than
    // faking a DOM — everything else in the report still renders.
    if (svgText && typeof DOMParser !== "undefined") {
      // svg2pdf.js's own UMD build assumes a browser (it reaches for jsPDF
      // as a global), so it is only imported here, where it is actually
      // used — never in Node, where importing it unconditionally throws.
      const { svg2pdf } = await import("svg2pdf.js");
      const svgEl = new DOMParser().parseFromString(svgText, "image/svg+xml").documentElement;
      const chartWidth = w.right - w.left;
      const chartHeight = chartWidth * 0.24;
      const y0 = w.ensure(chartHeight);
      await svg2pdf(svgEl, doc, { x: w.left, y: y0, width: chartWidth, height: chartHeight });
      w.y = y0 + chartHeight + 8;
      w.para(`The line is how far the machine was from its own normal at each snapshot. 1× `
        + `is the edge of normal; a change that stays above the dashed alarm line `
        + `(${r.chart.alarm}×) for several snapshots in a row raises an early sign. The `
        + "shaded part is the learning period.", { size: 8.5, color: MUTED });
    } else {
      w.para("(the chart is not included in a report built outside a browser)",
        { size: 8.5, color: MUTED });
    }
  }

  w.heading("Alarm levels");
  for (const l of LEVELS) {
    const a = l.key === "WATCH" ? r.first_watch : l.key === "WARNING" ? r.first_warning : r.first_critical;
    w.para(l.name, { bold: true, gap: 1 });
    w.para(l.means, { size: 9, color: MUTED, gap: 1 });
    w.para(a ? `Reached ${a.before_end} before the end` : "Not reached", { size: 9.5, gap: 10 });
  }

  w.heading("What changed");
  if (r.evidence.length) {
    w.bullets(r.evidence);
    if (r.evidence_note) w.para(r.evidence_note, { size: 9, color: MUTED });
  } else {
    w.para("Nothing moved far enough from this machine's normal to report.");
  }
  const d = r.diagnosis?.response;
  if (d) {
    w.para(d.diagnosis);
    if (!r.names_faults || /nrecognised/.test(d.diagnosis)) w.para(UNRECOGNISED_NOTE, { size: 9, color: MUTED });
  }

  if (d) {
    w.heading("What to do");
    if (d.actions?.length) w.bullets(d.actions.map((a) => a.step), { ordered: true });
    if (d.safety?.length) { w.para("Safety first", { bold: true, gap: 3 }); w.bullets(d.safety); }
    if (d.root_causes?.length) { w.para("Likely causes", { bold: true, gap: 3 }); w.bullets(d.root_causes); }
    if (d.guidance_note) w.para(d.guidance_note, { size: 9, color: MUTED });
    if (d.references?.length) {
      w.para(`Cites ${d.references.join(", ")}${d.references_label ? ` (${d.references_label})` : ""}.`,
        { size: 9, color: MUTED });
    }
  }

  w.heading("What this can't tell you");
  w.bullets(r.caveats);
  w.para("Settings this result used", { bold: true, gap: 3 });
  w.bullets([...runSettings(r), ...(r.assumed ?? []).map((a) => `Assumed: ${a}`)]);

  if (contact) {
    w.heading("Run this on your whole fleet");
    w.para("Live watching, your machines' own manuals, and the full diagnosis.",
      { size: 9.5, color: MUTED, gap: 6 });
    w.link("Talk to us", contact);
  }

  w.heading("Keeping this report");
  w.para(run.consent
    ? "You let Harmonic keep this recording to improve its models."
    : `The uploaded recording is deleted from Harmonic after ${retentionDays} days.`,
  { size: 9, color: MUTED });
  w.para("Harmonic is advisory only: it never drives a machine, and your plant's own safety "
    + "system stays authoritative.", { size: 9, color: MUTED });

  return doc;
}
