// The "Download report" file (S10-T07, reworded for S10-T06): one
// self-contained HTML page, built in the browser from the run the page
// already holds, in the page's own order: the plain answer, the chart, the
// alarm levels, what changed, what to do, what this can't tell you, and
// "talk to us". No email, no server round trip, nothing left behind once the
// visitor deletes the run. Every value is escaped: the report text is
// server-made, but this file is opened outside the app, so it trusts nothing.

import type { Report, Run } from "./api";
import { CHART_CSS, CHART_VARS_LIGHT, LEVELS, UNRECOGNISED_NOTE, chartHtml, esc } from "./format";

/** An http(s) or mailto link, or null: never `javascript:` or `data:`. */
export function safeHref(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const u = new URL(url, window.location.href);
    return ["https:", "http:", "mailto:"].includes(u.protocol) ? u.href : null;
  } catch {
    return null;
  }
}

const STATE_COLOR: Record<string, string> = { WATCH: "#9a7400", WARNING: "#c0561a", CRITICAL: "#b3261e" };

const list = (items: string[] | undefined, tag = "ul") =>
  items?.length ? `<${tag}>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</${tag}>` : "";

export function reportHtml(run: Run, r: Report, contact: string | null,
  retentionDays: number): string {
  const d = r.diagnosis?.response;
  const levels = LEVELS.map((l) => {
    const a = l.key === "WATCH" ? r.first_watch : l.key === "WARNING" ? r.first_warning : r.first_critical;
    return `<tr><td><b style="color:${STATE_COLOR[l.key]}">${esc(l.name)}</b><div class="muted">${esc(l.means)}</div></td>`
      + `<td>${a ? `<b>Reached ${esc(a.before_end)} before the end</b>` : "Not reached"}</td></tr>`;
  }).join("");
  const unrecognised = d && (!r.names_faults || /nrecognised/.test(d.diagnosis));
  const changed = (r.evidence.length
    ? `${list(r.evidence)}${r.evidence_note ? `<p class="muted">${esc(r.evidence_note)}</p>` : ""}`
    : "<p>Nothing moved far enough from this machine's normal to report.</p>")
    + (d ? `<p>${esc(d.diagnosis)}</p>${unrecognised ? `<p class="muted">${esc(UNRECOGNISED_NOTE)}</p>` : ""}` : "");
  const todo = d ? `<h2>What to do</h2>${list(d.actions?.map((a) => a.step), "ol")}
    ${d.safety?.length ? `<h3>Safety first</h3>${list(d.safety)}` : ""}
    ${d.root_causes?.length ? `<h3>Likely causes</h3>${list(d.root_causes)}` : ""}
    ${d.guidance_note ? `<p class="muted">${esc(d.guidance_note)}</p>` : ""}
    ${d.references?.length ? `<p class="muted">Cites ${esc(d.references.join(", "))}${d.references_label ? ` (${esc(d.references_label)})` : ""}.</p>` : ""}` : "";
  const spacing = r.timed_by === "timestamps" ? "from the recording's time stamps"
    : `${r.interval_text ?? ""}${r.timed_by === "wav" ? " (back to back, from the WAV file)" : " (as chosen on upload)"}`;
  const settings = [
    `Machine type: ${r.family_label}`,
    `Recording: ${r.n_snapshots} snapshots over ${r.span_text}, ${r.sample_rate_hz} readings per second${r.n_channels > 1 ? `, channel ${r.channel} of ${r.n_channels}` : ""}`,
    `Time between snapshots: ${spacing}`,
    `Learning period: the first ${r.baseline_text}; false alarms while learning: ${r.false_alarms_in_baseline}`,
    `Shaft speed: ${r.rpm_used} rpm${r.rpm_limited ? " (assumed)" : ""}`,
    ...(r.assumed ?? []).map((a) => `Assumed: ${a}`),
  ];
  const made = new Date().toISOString().slice(0, 16).replace("T", " ");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harmonic report</title>
<style>
:root{${CHART_VARS_LIGHT}}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1c2330;background:#fff;
max-width:760px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px}h3{font-size:14px;margin:14px 0 4px}
.muted{color:#5b6878;font-size:13px}table{border-collapse:collapse;width:100%;margin-top:14px}
td{padding:8px;border-bottom:1px solid #e4e7ec;vertical-align:top}
.head{font-size:18px;font-weight:600;margin:14px 0 16px;line-height:1.4}ul,ol{padding-left:20px}a{color:#1f5fbf}
details{margin-top:22px;border-top:1px solid #e4e7ec;padding-top:10px}summary{font-weight:700;cursor:pointer}
${CHART_CSS}
</style></head><body>
<h1>Harmonic report</h1>
<p class="muted">Made ${esc(made)} UTC from a recording uploaded to "Try Harmonic on your own data".</p>
<p class="head">${esc(r.headline)}</p>
${r.sample ? `<p class="muted">Sample recording: ${esc(r.sample.title)}. The real bearing ended with an outer-race failure. ${esc(r.sample.credit)}</p>` : ""}
${chartHtml(r)}
<table>${levels}</table>
<h2>What changed</h2>
${changed}
${todo}
<details><summary>What this can't tell you</summary>
${list(r.caveats)}
<h3>Settings this result used</h3>
${list(settings)}
</details>
${contact ? `<h2>Run this on your whole fleet</h2>
<p><a href="${esc(contact)}" rel="noopener noreferrer">Talk to us</a></p>` : ""}
<p class="muted">Harmonic is advisory only: it never drives a machine, and your plant's own
safety system stays authoritative. ${run.consent ? "" : `The uploaded recording is deleted from Harmonic after ${esc(retentionDays)} days.`}</p>
</body></html>
`;
}
