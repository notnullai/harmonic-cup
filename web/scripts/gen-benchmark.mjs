#!/usr/bin/env node
// HARM-S10-T10: renders docs/BENCHMARK.md to static HTML at build time, for
// the public /try/benchmark page (web/src/try/BenchmarkPage.tsx). No
// Markdown parser runs on the server, and the doc is trusted repo content
// (owner-approved 2026-09-30), not untrusted input — but the output is
// still built from a small, purpose-written converter (no dependency: none
// of marked/remark/showdown is in web/node_modules) that only understands
// the constructs this one document uses: #/##/### headings, paragraphs,
// **bold**, *italic*, `code`, [text](url) links, bare https:// URLs, ```
// fenced blocks, ordered/unordered lists (with a hanging-indent
// continuation line, as the doc uses) and GFM pipe tables.
//
// Two content adjustments happen here, not in the source doc, because they
// are about *publishing* it, not about its substance (every measurement,
// miss and caveat is carried over unchanged):
//   1. The internal "Draft — HARM-S10-T10. Not published anywhere yet."
//      marker is dropped — publishing this page is what makes it stale.
//   2. Two links to `docs/artifacts/benchmark.{md,json}` and the whole
//      "How to reproduce this" section reference repo-internal paths and a
//      runnable command (`python scripts/benchmark.py`, "open an issue").
//      The repo is private, so a visitor can't follow or run any of that:
//      TRY_IT_SPEC.md's brief says to reword these to plain text and say
//      the scripts are available on request, which is what happens below.
//
// Run by `npm run build` / `npm run typecheck` / `npm run dev` (via the
// pre* hooks in package.json). Output is generated, not committed (see
// .gitignore).

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");
const srcPath = path.join(repoRoot, "docs", "BENCHMARK.md");
const outPath = path.join(repoRoot, "web", "src", "try", "benchmarkContent.generated.ts");

const REPRODUCE_SECTION = `## How to reproduce this

Everything here comes from data files, not from us: NASA IMS and XJTU-SY as
extracted from their own archives, CWRU's \`.mat\` files from the Bearing
Data Center, and the OpenEI gearbox's healthy and broken-tooth recordings.
One runner script reads these files, applies the same commissioning rule,
noise-floor rule and alarm rule to every dataset with nothing retuned
between them, and writes the numbers this page quotes.

No dataset is downloaded by the runner itself, and no result is hand-edited
after the fact. The scripts (\`scripts/benchmark.py\`,
\`scripts/evaluate_ims.py\`, \`scripts/evaluate_xjtu.py\`,
\`scripts/evaluate_cwru.py\` and \`scripts/evaluate_gearbox_real.py\`) are
available on request, since the repository they live in is private.
`;

function applyPublishingRewrites(raw) {
  let out = raw;

  out = out.replace(
    /\*\*Draft — HARM-S10-T10\. Not published anywhere yet\.\*\*\s*/,
    "",
  );

  out = out.replace(
    /generated straight from the runner and kept alongside it, not retyped here[\s\S]*?What follows is the honest summary\./,
    "generated straight from the runner and kept alongside it, not retyped "
      + "here by hand: `docs/artifacts/benchmark.md` (machine-readable "
      + "version: `docs/artifacts/benchmark.json`), available on request, "
      + "since the repository is private. What follows is the honest summary.",
  );

  const reproIdx = out.indexOf("## How to reproduce this");
  if (reproIdx !== -1) out = out.slice(0, reproIdx) + REPRODUCE_SECTION;

  return out;
}

// --- a small Markdown -> HTML converter, only for the constructs above ----

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(text) {
  let out = escapeHtml(text);
  // Bare URLs (the licences section cites one in parentheses) before []().
  out = out.replace(/https?:\/\/[^\s)]+/g, (url) =>
    `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
  // [label](url): an external link is kept; a repo-internal path (no
  // scheme) is private, so only its label survives, as plain text.
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => (
    /^https?:\/\//.test(url)
      ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
      : label
  ));
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return out;
}

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|")) s = s.slice(0, -1);
  return s.split("|").map((c) => c.trim());
}

const TABLE_SEP = /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function convertMarkdown(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === "") { i++; continue; }

    if (line.startsWith("### ")) { blocks.push(`<h3>${inline(line.slice(4))}</h3>`); i++; continue; }
    if (line.startsWith("## ")) { blocks.push(`<h2>${inline(line.slice(3))}</h2>`); i++; continue; }
    if (line.startsWith("# ")) { blocks.push(`<h1>${inline(line.slice(2))}</h1>`); i++; continue; }

    if (line.startsWith("```")) {
      i++;
      const codeLines = [];
      while (i < lines.length && !lines[i].startsWith("```")) { codeLines.push(lines[i]); i++; }
      i++;
      blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      continue;
    }

    if (line.startsWith("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const head = splitRow(line);
      const align = splitRow(lines[i + 1]).map((c) => (
        c.endsWith(":") && c.startsWith(":") ? "center" : c.endsWith(":") ? "right"
          : c.startsWith(":") ? "left" : null
      ));
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) { rows.push(splitRow(lines[i])); i++; }
      const th = head.map((c, k) => `<th${align[k] ? ` style="text-align:${align[k]}"` : ""}>${inline(c)}</th>`).join("");
      const tb = rows.map((r) => `<tr>${r.map((c, k) => `<td${align[k] ? ` style="text-align:${align[k]}"` : ""}>${inline(c)}</td>`).join("")}</tr>`).join("");
      blocks.push(`<div class="bm-table"><table><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`);
      continue;
    }

    if (/^\d+\.\s/.test(line)) {
      const items = [];
      while (i < lines.length && (/^\d+\.\s/.test(lines[i]) || (lines[i].startsWith("   ") && lines[i].trim() !== ""))) {
        if (/^\d+\.\s/.test(lines[i])) items.push(lines[i].replace(/^\d+\.\s/, ""));
        else items[items.length - 1] += ` ${lines[i].trim()}`;
        i++;
      }
      blocks.push(`<ol>${items.map((it) => `<li>${inline(it)}</li>`).join("")}</ol>`);
      continue;
    }

    if (/^-\s/.test(line)) {
      const items = [];
      while (i < lines.length && (/^-\s/.test(lines[i]) || (lines[i].startsWith("  ") && lines[i].trim() !== ""))) {
        if (/^-\s/.test(lines[i])) items.push(lines[i].replace(/^-\s/, ""));
        else items[items.length - 1] += ` ${lines[i].trim()}`;
        i++;
      }
      blocks.push(`<ul>${items.map((it) => `<li>${inline(it)}</li>`).join("")}</ul>`);
      continue;
    }

    const para = [line];
    i++;
    while (i < lines.length && lines[i].trim() !== "" && !/^(#{1,3}\s|```|\|)/.test(lines[i])
      && !/^\d+\.\s/.test(lines[i]) && !/^-\s/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(`<p>${inline(para.join(" "))}</p>`);
  }

  return blocks.join("\n");
}

async function main() {
  const raw = await readFile(srcPath, "utf8");
  const html = convertMarkdown(applyPublishingRewrites(raw));
  const body = `// Generated by web/scripts/gen-benchmark.mjs from docs/BENCHMARK.md.\n`
    + `// Do not edit by hand: edit the source doc, or the script, and rerun.\n`
    + `export const BENCHMARK_HTML: string = ${JSON.stringify(html)};\n`;
  await mkdir(path.dirname(outPath), { recursive: true });
  await writeFile(outPath, body, "utf8");
  console.log(`wrote ${path.relative(repoRoot, outPath)} (${html.length} chars of HTML)`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
