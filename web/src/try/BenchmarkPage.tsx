// The public benchmark page (HARM-S10-T10): docs/BENCHMARK.md, rendered to
// HTML at build time by web/scripts/gen-benchmark.mjs (see that file for
// the exact conversion and the two publishing-only rewrites it makes). No
// Markdown parser runs here at request time, and nothing on this page comes
// from a visitor, so the generated HTML is trusted the same way any other
// build output is.
//
// Routed client-side by pathname (web/src/try/main.tsx) at /try/benchmark,
// served by api/main.py's own route for that path (the same SPA shell as
// /try/). Linked from the wizard's header (App.tsx) and links back to it.

import { BENCHMARK_HTML } from "./benchmarkContent.generated";

export default function BenchmarkPage() {
  return (
    <div className="try-app">
      <div className="try-shell">
        <div className="try-topbar">
          <a className="btn ghost sm try-startover" href="/try/">&larr; Try Harmonic on your own recording</a>
        </div>
        <article className="bm" dangerouslySetInnerHTML={{ __html: BENCHMARK_HTML }} />
        <footer className="try-footer">
          Harmonic is advisory only: your plant's own safety system stays in charge.{" "}
          <a href="/try/">Try it on your own recording &rarr;</a>
        </footer>
      </div>
    </div>
  );
}
