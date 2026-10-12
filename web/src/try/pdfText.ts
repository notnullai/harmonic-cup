// Pure text-layout helpers for the downloaded PDF report (S10-T06, owner-
// approved 2026-09-29). No jsPDF and no DOM here on purpose: both functions
// take a plain measurement (a width, a height) rather than a jsPDF instance,
// so they can be unit-tested without a browser or a PDF library
// (scripts/verify_try_pdf.mjs) and reused unchanged from reportPdf.ts.

/** Greedy word-wrap: splits `text` into lines whose measured width (via
 *  `measure`, e.g. jsPDF's own `doc.getTextWidth`) fits within `maxWidth`. A
 *  single word wider than `maxWidth` stays on its own line rather than being
 *  split mid-word — the same trade-off jsPDF's own `splitTextToSize` makes. */
export function wrapText(text: string, maxWidth: number, measure: (s: string) => number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const candidate = line ? `${line} ${w}` : w;
    if (line && measure(candidate) > maxWidth) {
      lines.push(line);
      line = w;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}

export interface LinePlacement { page: number; y: number }

/** Places `count` same-height lines top to bottom starting at `startY` on
 *  page 0, moving to a new page (y reset to `top`) whenever the next line
 *  would cross `bottom` — so a page break always falls between two lines,
 *  never through one. `reportPdf.ts`'s own page-break check (there, against
 *  jsPDF's real cursor, since section heights vary) follows this exact rule. */
export function layoutLines(count: number, lineHeight: number, startY: number,
  top: number, bottom: number): LinePlacement[] {
  const out: LinePlacement[] = [];
  let page = 0;
  let y = startY;
  for (let i = 0; i < count; i++) {
    if (y + lineHeight > bottom) { page += 1; y = top; }
    out.push({ page, y });
    y += lineHeight;
  }
  return out;
}
