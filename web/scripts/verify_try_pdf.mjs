#!/usr/bin/env node
// A manual check for the "Try it" PDF report (S10-T06, owner-approved
// 2026-09-29), run by hand: `node web/scripts/verify_try_pdf.mjs`.
//
// 1. A small unit test of pdfText.ts's two pure layout helpers.
// 2. Builds one PDF, in Node, from a stored report (the NASA IMS sample's
//    report, web/scripts/fixtures/ims_sample_report.json — the same file
//    the "Try it with a sample recording" button produces), and checks the
//    file exists, starts with "%PDF" and is bigger than 10 KB.
//
// No test framework: web/ has none, and this task adds none (only jspdf and
// svg2pdf.js, per the owner). esbuild is already vendored by Vite, so it is
// used here, in-process, to transpile the two .ts modules under test — no
// extra install. The chart step is skipped in Node (reportPdf.ts's own
// DOMParser feature-detect: no jsdom here either), so this run exercises
// every other section: headline, run details, levels, what changed, what to
// do, caveats, the sample credit, the "talk to us" link and the retention
// and advisory lines.

import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, "..");
const srcTry = path.join(webRoot, "src", "try");

async function bundle(entry) {
  // Bundled (so "./format" and "./pdfText" are resolved and inlined), but
  // "jspdf" and "svg2pdf.js" stay external real packages from web/node_modules
  // — the point is to exercise the actual installed libraries, not a stub.
  const out = await build({
    entryPoints: [entry],
    bundle: true,
    write: false,
    platform: "node",
    format: "esm",
    target: "node18",
    external: ["jspdf", "svg2pdf.js"],
  });
  return out.outputFiles[0].text;
}

async function loadModule(entry) {
  const code = await bundle(entry);
  const dir = await mkdtemp(path.join(webRoot, ".verify-tmp-"));
  const file = path.join(dir, "mod.mjs");
  await writeFile(file, code);
  try {
    return await import(pathToFileURL(file).href);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// --- 1. wrapText and layoutLines -----------------------------------------

async function testPdfText() {
  const { wrapText, layoutLines } = await loadModule(path.join(srcTry, "pdfText.ts"));

  // wrapText: a fixed-width measurer (one unit per character) makes the
  // wrap points exact and easy to state.
  const charWidth = (s) => s.length;
  assert.deepEqual(wrapText("", 10, charWidth), []);
  assert.deepEqual(wrapText("one two three", 100, charWidth), ["one two three"]);
  assert.deepEqual(
    wrapText("the quick brown fox jumps", 11, charWidth),
    ["the quick", "brown fox", "jumps"],
  );
  // A single word wider than maxWidth is kept whole, not split.
  assert.deepEqual(wrapText("supercalifragilisticexpialidocious", 5, charWidth),
    ["supercalifragilisticexpialidocious"]);

  // layoutLines: a page break falls between lines, never through one. With
  // lineHeight 10 and a [0, 25) band, only two 10-tall lines fit per page
  // (a third would end at 30, past the 25 bottom margin) — so five lines
  // land two, two, one across three pages.
  const placements = layoutLines(5, 10, 0, 0, 25);
  assert.deepEqual(placements, [
    { page: 0, y: 0 }, { page: 0, y: 10 },
    { page: 1, y: 0 }, { page: 1, y: 10 },
    { page: 2, y: 0 },
  ]);
  // Every line fits inside its page's [top, bottom) band.
  for (const { y } of placements) assert.ok(y + 10 <= 25);
  assert.deepEqual(layoutLines(0, 10, 0, 0, 25), []);

  console.log("pdfText: wrapText and layoutLines — all assertions passed");
}

// --- 2. one PDF, in Node, from a stored report ---------------------------

async function testGeneratedPdf() {
  const { buildReportPdf } = await loadModule(path.join(srcTry, "reportPdf.ts"));
  const fixturePath = path.join(here, "fixtures", "ims_sample_report.json");
  const run = JSON.parse(await readFile(fixturePath, "utf-8"));

  const doc = await buildReportPdf(run, run.report, "https://example.com/contact",
    7);
  const bytes = Buffer.from(doc.output("arraybuffer"));

  const outPath = path.join(here, "..", ".verify-tmp-out.pdf");
  await writeFile(outPath, bytes);
  const st = await stat(outPath);

  assert.ok(st.size > 0, "the PDF file was not written");
  const header = bytes.subarray(0, 4).toString("latin1");
  assert.equal(header, "%PDF", `expected a PDF header, got ${JSON.stringify(header)}`);
  assert.ok(st.size > 10 * 1024,
    `expected more than 10 KB, got ${st.size} bytes`);

  console.log(`reportPdf: ${outPath} — ${st.size} bytes, starts with "%PDF" — passed`);
  await rm(outPath, { force: true });
}

await testPdfText();
await testGeneratedPdf();
console.log("verify_try_pdf: all checks passed");
