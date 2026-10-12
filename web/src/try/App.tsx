// "Try Harmonic on your own data" — the public, anonymous wizard
// (docs/TRY_IT_SPEC.md, api/try_mode.py), written for a plant maintenance
// manager or technician, not a vibration analyst (S10-T06). Four numbered
// steps: your recording, your machine, when it was running normally, the
// result. No sign-up, no email: the report downloads from the page, and
// "Delete now" removes the run on the server at once.
//
// The run id is the only key to a run. It lives in the URL fragment
// (#run=...), which a browser never sends to a server or in a Referer, so a
// visitor can reload or bookmark their own report for as long as it is kept.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api, type ConfigureBody, type Limits, type Overview, type Probe, type Report, type Run,
  type SampleInfo,
} from "./api";
import {
  CHART_CSS, LEVELS, UNRECOGNISED_NOTE, chartHtml, humanInterval, humanSpan,
} from "./format";
import { reportHtml, safeHref } from "./reportHtml";

type Step = "recording" | "machine" | "learn" | "running" | "report" | "deleted";
type Fmt = "auto" | "wide_csv" | "wav";

const STEPS: { key: Step; label: string }[] = [
  { key: "recording", label: "Your recording" },
  { key: "machine", label: "Your machine" },
  { key: "learn", label: "When was it running normally?" },
  { key: "report", label: "Result" },
];

// S10-T06: the heading each step focuses when it becomes current, for
// keyboard and screen-reader visitors.
const HEADING_ID: Record<Step, string> = {
  recording: "s1", machine: "s2", learn: "s3", running: "s-run", report: "s4",
  deleted: "s-deleted",
};

interface Settings {
  rate: string;                 // Hz, as typed or picked
  every: string;                // "60" | "600" | "3600" | "other" | "" (not chosen yet)
  otherValue: string;
  otherUnit: "s" | "min" | "h";
  unit: "g" | "mm/s";
  family: string;
  rpm: string;
  params: Record<string, string>;
}

interface Upload { runId: string; files: File[]; fmt: Fmt; hasTs: boolean; probe: Probe }

const RATE_PRESETS = [10000, 20000, 25600, 51200];
const EVERY_PRESETS: [string, string][] = [["60", "1 minute"], ["600", "10 minutes"], ["3600", "1 hour"]];

const STEP_KEYS: Step[] = ["recording", "machine", "learn", "running", "report", "deleted"];

function runFromHash(): string | null {
  const m = /(?:^|[#&])run=([A-Za-z0-9_-]{16,64})/.exec(window.location.hash);
  return m ? m[1] : null;
}

function stepFromHash(): Step | null {
  const m = /(?:^|[#&])s=([a-z]+)/.exec(window.location.hash);
  const v = m?.[1] as Step | undefined;
  return v && STEP_KEYS.includes(v) ? v : null;
}

/** The URL fragment: the run id (so a report can be reloaded or bookmarked,
 *  unchanged) plus, from S10-T06, the step — so the browser's own Back
 *  button moves one step back through the wizard instead of leaving the
 *  page (the state object carries the step too; the fragment is the
 *  fallback for the one history entry pushState never wrote to, the page's
 *  own first load). */
function setHash(id: string | null, step: Step, push: boolean) {
  const parts: string[] = [];
  if (id) parts.push(`run=${id}`);
  if (step !== "recording") parts.push(`s=${step}`);
  const url = window.location.pathname + window.location.search
    + (parts.length ? `#${parts.join("&")}` : "");
  if (push) window.history.pushState({ step }, "", url);
  else window.history.replaceState({ step }, "", url);
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function intervalSeconds(s: Settings): number | null {
  if (s.every === "") return null;
  if (s.every !== "other") return Number(s.every);
  const n = Number(s.otherValue);
  if (s.otherValue.trim() === "" || !Number.isFinite(n) || n <= 0) return null;
  return n * (s.otherUnit === "h" ? 3600 : s.otherUnit === "min" ? 60 : 1);
}

function freshSettings(limits: Limits): Settings {
  const keys = Object.keys(limits.families);
  return {
    rate: "", every: "", otherValue: "", otherUnit: "min", unit: "g",
    // "Other or unsure" is the family that names no fault: the safe default.
    family: keys.find((k) => limits.family_info?.[k]?.names_faults === false) ?? keys[0],
    rpm: "", params: {},
  };
}

function configureBody(s: Settings, up: Upload, limits: Limits, channel = 0): ConfigureBody {
  const famParams = limits.family_info?.[s.family]?.parameters ?? [];
  const iv = intervalSeconds(s);
  return {
    family: s.family, sample_rate_hz: up.probe.format === "wav" ? 0 : Number(s.rate),
    rpm: s.rpm.trim() === "" ? null : Number(s.rpm), unit: s.unit, format: up.fmt,
    has_timestamp: up.hasTs, channel,
    params: Object.fromEntries(famParams.filter((p) => (s.params[p.key] ?? "").trim() !== "")
      .map((p) => [p.key, Number(s.params[p.key])])),
    ...(up.probe.timed_by === "interval" && iv !== null ? { interval_s: iv } : {}),
  };
}

export default function App() {
  const [limits, setLimits] = useState<Limits | null>(null);
  const [step, setStep] = useState<Step>("recording");
  const [upload, setUpload] = useState<Upload | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [sample, setSample] = useState<SampleInfo | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  // S10-T06: every step transition that moves *forward* (or sideways, to an
  // already-completed step) pushes one history entry carrying the step and
  // the run id; the "Back" buttons on steps 2 and 3 call the browser's own
  // history.back() instead, so it and the browser's own Back control agree.
  const nav = useCallback((next: Step, id: string | null = runId) => {
    setStep(next);
    setHash(id, next, true);
  }, [runId]);

  useEffect(() => {
    api.limits().then((l) => { setLimits(l); setSettings((s) => s ?? freshSettings(l)); })
      .catch(() => setError("The page could not reach Harmonic. Check your internet connection "
        + "and reload the page."));
    const id = runFromHash();
    if (id) {
      api.run(id).then((r) => {
        setRunId(id);
        setRun(r);
        if (r.status === "done") { setStep("report"); setHash(id, "report", false); }
        else if (r.status === "queued" || r.status === "running" || r.status === "failed") {
          setStep("running"); setHash(id, "running", false);
        } else setHash(null, "recording", false);
      }).catch(() => setHash(null, "recording", false));
    }
  }, []);

  // The browser's own Back/Forward buttons: move one step back (or forward)
  // through whatever this visit's own navigation actually did.
  useEffect(() => {
    const onPop = (e: PopStateEvent) => {
      const s = (e.state as { step?: Step } | null)?.step ?? stepFromHash() ?? "recording";
      setStep(s);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0 });
    // S10-T06: focus the new step's own heading, for keyboard and
    // screen-reader visitors — the page does not silently change under them.
    document.getElementById(HEADING_ID[step])?.focus();
  }, [step]);

  const resetState = useCallback(() => {
    setRunId(null); setUpload(null); setOverview(null); setRun(null); setSample(null);
    setError(null); setBusy(null);
    if (limits) setSettings(freshSettings(limits));
  }, [limits]);

  // "Try another recording" at the end of the report, and after a delete: no
  // server-side delete (the finished run, if any, is simply left to expire
  // on its own schedule; "Start over" below is the one that deletes).
  const restart = useCallback(() => {
    resetState();
    nav("recording", null);
  }, [resetState, nav]);

  // S10-T06: "Start over", in the header, on every step. It deletes the
  // current run server-side (the same delete "Delete now" uses) and returns
  // to step 1 — unlike "Try another recording", which keeps a finished run.
  const startOver = useCallback(async () => {
    if (step !== "recording"
      && !window.confirm("Start over? Your uploaded file will be deleted.")) return;
    if (runId) { try { await api.deleteRun(runId); } catch { /* best-effort */ } }
    resetState();
    nav("recording", null);
  }, [step, runId, resetState, nav]);

  const removeRun = useCallback(async (confirmMsg: string) => {
    if (!runId) return;
    if (!window.confirm(confirmMsg)) return;
    try {
      await api.deleteRun(runId);
      setRunId(null); setUpload(null); setOverview(null); setRun(null);
      nav("deleted", null);
    } catch (e) {
      setError(errText(e));
    }
  }, [runId, nav]);

  const del = useCallback(() => removeRun("Delete this recording and its report from Harmonic "
    + "now? This cannot be undone. Download the report first if you want to keep it."),
  [removeRun]);

  // S10-T06: the running step's own "Cancel" — the same delete, cancelling
  // (already required, HARM-S10-T03 M1) mid-run.
  const cancelRun = useCallback(() => removeRun("Cancel this run and delete the recording from "
    + "Harmonic now? This cannot be undone."), [removeRun]);

  // The sample: the same upload, read, configure and run as a visitor's file,
  // with the settings filled in; it lands on step 3.
  const trySample = useCallback(async () => {
    if (!limits) return;
    setError(null);
    try {
      setBusy("Loading the sample recording…");
      const { id } = await api.newRun();
      const info = await api.sample(id);
      setBusy("Reading the sample recording…");
      const probe = await api.probe(id, "auto", false);
      const up: Upload = { runId: id, files: [], fmt: "auto", hasTs: false, probe };
      const s: Settings = {
        ...freshSettings(limits), rate: String(info.sample_rate_hz), unit: info.unit as "g",
        family: info.family, rpm: String(info.rpm),
        ...(EVERY_PRESETS.some(([v]) => Number(v) === info.interval_s)
          ? { every: String(info.interval_s) }
          : { every: "other", otherValue: String(info.interval_s / 60), otherUnit: "min" as const }),
      };
      const ov = await api.configure(id, configureBody(s, up, limits));
      setRunId(id); setUpload(up); setSettings(s); setSample(info); setOverview(ov);
      nav("learn", id);
    } catch (e) {
      setError(`The sample could not be loaded: ${errText(e)}`);
    } finally {
      setBusy(null);
    }
  }, [limits, nav]);

  // "Change settings and run again" on the result: configure this same run
  // again (allowed once it is done or failed — S10-T06) with today's
  // settings unchanged, then land on step 2 to edit them and go forward
  // again through the ordinary flow, with the same upload.
  const [changing, setChanging] = useState(false);
  const changeSettings = useCallback(async () => {
    if (!runId || !upload || !settings || !limits) return;
    setChanging(true);
    setError(null);
    try {
      const ov = await api.configure(runId, configureBody(settings, upload, limits,
        overview?.channel ?? 0));
      setOverview(ov);
      setRun(null);
      nav("machine");
    } catch (e) {
      setError(errText(e));
    } finally {
      setChanging(false);
    }
  }, [runId, upload, settings, limits, overview, nav]);

  const stepIndex = step === "recording" ? 0 : step === "machine" ? 1 : step === "learn" ? 2 : 3;

  return (
    <div className="try-app">
      <style>{CHART_CSS}</style>
      <div className="try-shell">
        {step !== "deleted" && (
          <div className="try-topbar">
            <button type="button" className="btn ghost sm try-startover" onClick={startOver}>
              Start over
            </button>
          </div>
        )}
        <header className="try-hero">
          <h1>Would Harmonic have warned you?</h1>
          <p>Upload a vibration recording from one of your machines, ideally one that later
            failed. Harmonic replays it and tells you whether it would have warned you, when,
            and about what. No sign-up.</p>
          <p className="try-benchmark-cta">
            How well does it work?{" "}
            <a href="/try/benchmark">See the benchmark on real failures</a>
          </p>
          {step === "recording" && !upload && (
            <p className="try-sample-cta">
              No file to hand?{" "}
              <button className="btn ghost sm" onClick={trySample} disabled={!!busy || !limits}>
                Try it with a sample recording
              </button>
            </p>
          )}
        </header>

        {step !== "deleted" && (
          <ol className="try-steps" aria-label="Steps">
            {STEPS.map((s, i) => {
              // S10-T06: a completed step (i < stepIndex) can be revisited —
              // the upload is kept; a later change re-configures on the way
              // forward again, which already replaces the stale overview
              // and report.
              const clickable = i < stepIndex;
              return (
                <li key={s.key} className={i < stepIndex ? "done" : i === stepIndex ? "on" : ""}
                  aria-current={i === stepIndex ? "step" : undefined}>
                  <button type="button" className="step-btn" disabled={!clickable}
                    aria-label={`Step ${i + 1}: ${s.label}${clickable ? ", go back to this step" : ""}`}
                    onClick={clickable ? () => { setError(null); nav(s.key); } : undefined}>
                    <span className="n" aria-hidden="true">{i + 1}</span>
                    <span className="t">{s.label}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        )}

        {error && (
          <div className="card try-error" role="alert">
            <div className="card-b"><b>That didn't work.</b> {error}</div>
          </div>
        )}
        {busy && step === "recording" && !upload && (
          <div className="card"><div className="card-b progress-view" aria-live="polite">
            <div className="spinner" aria-hidden="true" /><b>{busy}</b>
          </div></div>
        )}

        {step === "recording" && limits && settings && (
          <RecordingStep limits={limits} settings={settings} setSettings={setSettings}
            upload={upload} busy={busy} setBusy={setBusy} onError={setError}
            onUploaded={(up) => { setUpload(up); setRunId(up.runId); }}
            onClear={() => {
              if (upload) api.deleteRun(upload.runId).catch(() => undefined);
              setUpload(null); setRunId(null);
            }}
            onNext={() => { setError(null); nav("machine"); }} />
        )}
        {step === "machine" && limits && settings && upload && (
          <MachineStep limits={limits} settings={settings} setSettings={setSettings}
            onBack={() => window.history.back()} onError={setError}
            onNext={async () => {
              setError(null);
              const ov = await api.configure(upload.runId, configureBody(settings, upload, limits,
                overview?.channel ?? 0));
              setOverview(ov);
              nav("learn");
            }} />
        )}
        {step === "learn" && runId && overview && limits && settings && upload && (
          <LearnStep runId={runId} overview={overview} limits={limits} sample={sample}
            settings={settings} onError={setError}
            onChannel={async (channel) => {
              setOverview(await api.configure(runId, configureBody(settings, upload, limits, channel)));
            }}
            onStarted={() => { setError(null); nav("running", runId); }}
            onBack={() => window.history.back()} />
        )}
        {step === "running" && runId && (
          <RunningStep runId={runId} initial={run} onDone={(r) => { setRun(r); nav("report"); }}
            onRestart={restart} onDelete={del} onCancel={cancelRun} />
        )}
        {step === "report" && run?.report && (
          <ReportView run={run} report={run.report} limits={limits} onDelete={del}
            onRestart={restart} onChangeSettings={changeSettings} changing={changing} />
        )}
        {step === "deleted" && (
          <div className="card"><div className="card-b progress-view">
            <b id="s-deleted" tabIndex={-1}>Deleted.</b>
            <span className="muted">The recording and its report are gone from Harmonic.</span>
            <button className="btn primary" onClick={restart}>Try another recording</button>
          </div></div>
        )}

        <footer className="try-footer">
          Anonymous. Uploads are private and deleted after {limits?.retention_days ?? 7} days
          unless you choose to let us keep them (then for at most
          {" "}{limits?.consent_retention_days ?? 90} days). The diagnosis on this page is
          Harmonic's offline one; no AI model is called. Harmonic is advisory only: your plant's
          own safety system stays in charge.
        </footer>
      </div>
    </div>
  );
}

// --- step 1: your recording ------------------------------------------------------------

function fileFormat(files: File[]): { fmt: Fmt; single: boolean } {
  const ext = (f: File) => f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
  if (files.length > 0 && files.every((f) => ext(f) === ".wav")) return { fmt: "wav", single: files.length === 1 };
  const single = files.length === 1 && [".csv", ".txt"].includes(ext(files[0]));
  return { fmt: single ? "wide_csv" : "auto", single };
}

/** Whether a one-file CSV starts each row with a time stamp: an ISO date, or
 *  epoch seconds (no vibration reading is a ten-digit number). */
async function firstCellIsTime(f: File): Promise<boolean> {
  const head = await f.slice(0, 256).text();
  const cell = head.split(/[\r\n]/)[0].split(/[,;\t]/)[0].trim().replace(/^"|"$/g, "");
  if (/^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?/.test(cell)) return true;
  const n = Number(cell);
  return /^\d{9,10}(\.\d+)?$/.test(cell) && n > 1e9 && n < 4.2e9;
}

function fmtBytes(n: number): string {
  return n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} kB`;
}

function formatWords(fmt: string, files: number): string {
  if (fmt === "wide_csv") return "one table with a snapshot on each row";
  if (fmt === "wav") return files > 1 ? `${files} WAV files, one snapshot each` : "one long WAV sound file";
  return files > 1 ? `${files} files, one snapshot each` : "one data file (an array of snapshots)";
}

function RecordingStep({ limits, settings, setSettings, upload, busy, setBusy, onUploaded, onClear,
  onNext, onError }: {
  limits: Limits; settings: Settings; setSettings: (s: Settings) => void;
  upload: Upload | null; busy: string | null; setBusy: (b: string | null) => void;
  onUploaded: (u: Upload) => void; onClear: () => void; onNext: () => void;
  onError: (msg: string | null) => void;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const read = async (list: FileList | null) => {
    if (!list || list.length === 0) return;
    const files = Array.from(list).sort((a, b) => a.name.localeCompare(b.name));
    const total = files.reduce((s, f) => s + f.size, 0);
    onError(null);
    if (total > limits.max_total_mb * (1 << 20)) {
      onError(`These files add up to ${fmtBytes(total)}; the limit is ${limits.max_total_mb} MB. `
        + "Send a shorter stretch of the recording, or fewer snapshots.");
      return;
    }
    if (files.length > limits.max_files) {
      onError(`That is ${files.length} files; at most ${limits.max_files} are accepted. Send `
        + "fewer snapshots, or one table with a snapshot on each row.");
      return;
    }
    const { fmt, single } = fileFormat(files);
    try {
      const hasTs = fmt === "wide_csv" && single ? await firstCellIsTime(files[0]) : false;
      setBusy("Starting…");
      const { id } = await api.newRun();
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        const label = files.length > 1 ? `Uploading file ${i + 1} of ${files.length}` : "Uploading";
        setBusy(`${label}…`);
        await api.uploadFile(id, f, limits.upload_part_bytes, (sent) =>
          setBusy(`${label}… ${Math.round((100 * sent) / Math.max(f.size, 1))}%`));
      }
      setBusy("Reading your recording…");
      const probe = await api.probe(id, fmt, hasTs);
      onUploaded({ runId: id, files, fmt, hasTs, probe });
      if (probe.timed_by !== "interval") setSettings({ ...settings, every: "" });
    } catch (e) {
      onError(`${errText(e)} Choose a different file, or check it against the formats listed.`);
    } finally {
      setBusy(null);
    }
  };

  const p = upload?.probe;
  const isWav = p?.format === "wav";
  const rateN = Number(settings.rate);
  const rateOk = isWav || (Number.isInteger(rateN) && rateN >= limits.min_sample_rate_hz
    && rateN <= limits.max_sample_rate_hz);
  const needsInterval = p?.timed_by === "interval";
  const iv = intervalSeconds(settings);
  const ivOk = !needsInterval || (iv !== null && iv >= (limits.min_interval_s ?? 1)
    && iv <= (limits.max_interval_s ?? 2592000));
  const ratePreset = RATE_PRESETS.includes(rateN) ? settings.rate : settings.rate === "" ? "" : "other";
  const [rateOther, setRateOther] = useState(ratePreset === "other");

  return (
    <section className="card" aria-labelledby="s1">
      <div className="card-h"><h2 id="s1" tabIndex={-1}>1. Your recording</h2></div>
      <div className="card-b">
        <p className="try-why">Harmonic replays your recording as if it had been watching the
          machine live.</p>
        {!upload && (
          <>
            <div className={`dropzone${over ? " over" : ""}`} role="button" tabIndex={0}
              aria-label="Choose your recording files"
              onClick={() => !busy && input.current?.click()}
              onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && !busy) { e.preventDefault(); input.current?.click(); } }}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) read(e.dataTransfer.files); }}>
              <input ref={input} type="file" multiple accept=".csv,.txt,.npy,.wav"
                onChange={(e) => { read(e.target.files); e.target.value = ""; }} />
              <div className="big">{busy ?? "Drop your recording here, or tap to choose"}</div>
              <div className="hint">
                What works: one CSV table with a snapshot on each row (a logger's export), one
                CSV, TXT or NPY file per snapshot, one NPY array, or WAV (one long file, or one
                per snapshot). Up to {limits.max_total_mb} MB.
              </div>
            </div>
            <details className="try-more">
              <summary>What is a snapshot, and how long must the recording be?</summary>
              <p>A <b>snapshot</b> is one short burst of vibration readings (usually a second or
                two), taken again every so often: every minute, every 10 minutes, every hour.
                Harmonic needs at least {limits.min_snapshots} snapshots: it learns what normal
                looks like from the first part, then watches the rest. That is{" "}
                {humanSpan(limits.min_snapshots / 60)} of snapshots taken once a minute, or
                {" "}{humanSpan(limits.min_snapshots / 6)} at one every 10 minutes. One long WAV
                works too: at least about{" "}
                {Math.ceil((limits.min_snapshots * limits.wav_snapshot_samples) / 25600)} seconds
                at 25.6 kHz.</p>
            </details>
          </>
        )}

        {upload && p && (
          <div className="try-found">
            <div className="try-found-h">
              <b>What we found in your file</b>
              <button className="btn ghost sm" onClick={onClear}>Choose a different file</button>
            </div>
            <ul>
              <li>{upload.files.length > 0
                ? `${upload.files.length === 1 ? upload.files[0].name : `${upload.files.length} files`}, ${fmtBytes(upload.files.reduce((s, f) => s + f.size, 0))}: `
                : "The sample: "}{formatWords(p.format, p.files)}.</li>
              <li><b>{p.snapshots} snapshots</b>, each {p.samples_per_snapshot.toLocaleString()} readings long.</li>
              <li>{p.n_channels === 1 ? "One sensor channel." : `${p.n_channels} sensor channels; channel 1 is used unless you choose another on step 3.`}</li>
              <li>{p.timed_by === "timestamps"
                ? `Time stamps found: about ${humanInterval(p.interval_s ?? 0)} apart, ${humanSpan(p.span_h ?? 0)} in all.`
                : p.timed_by === "wav" ? `One continuous sound file: ${humanSpan(p.span_h ?? 0)} of running, cut into snapshots.`
                  : "No time stamps, so tell us below how far apart the snapshots were taken."}</li>
            </ul>
          </div>
        )}

        {upload && p && (
          <>
            {!isWav && (
              <fieldset className="field try-gap">
                <legend>Sampling rate <span>readings per second</span></legend>
                <div className="try-chips" role="radiogroup" aria-label="Sampling rate">
                  {RATE_PRESETS.map((r) => (
                    <button type="button" key={r} role="radio" aria-checked={!rateOther && rateN === r}
                      onClick={() => { setRateOther(false); setSettings({ ...settings, rate: String(r) }); }}>
                      {r % 1000 === 0 ? `${r / 1000} kHz` : `${(r / 1000).toFixed(1)} kHz`}
                    </button>
                  ))}
                  <button type="button" role="radio" aria-checked={rateOther}
                    onClick={() => { setRateOther(true); setSettings({ ...settings, rate: "" }); }}>Other</button>
                </div>
                {rateOther && (
                  <input className="input try-gap-s" inputMode="numeric" aria-label="Sampling rate in Hz"
                    placeholder="Readings per second (Hz), e.g. 12800" value={settings.rate}
                    onChange={(e) => setSettings({ ...settings, rate: e.target.value.trim() })}
                    aria-invalid={settings.rate !== "" && !rateOk} />
                )}
                <span className="hint">It's in your logger's export settings or its manual, often
                  called "sample rate", "sampling frequency" or "Fs". Common values are 25.6 kHz
                  (25,600 Hz), 51.2 kHz and 20 kHz.
                  {rateOk && settings.rate !== "" ? ` At this rate each snapshot covers ${(p.samples_per_snapshot / rateN).toFixed(2)} seconds.` : ""}</span>
                {settings.rate !== "" && !rateOk && (
                  <span className="try-warn">Enter a whole number between {limits.min_sample_rate_hz} and {limits.max_sample_rate_hz} Hz.</span>
                )}
              </fieldset>
            )}

            {needsInterval && (
              <fieldset className="field try-gap">
                <legend>How often was a snapshot taken?</legend>
                <div className="try-chips" role="radiogroup" aria-label="Time between snapshots">
                  {EVERY_PRESETS.map(([v, label]) => (
                    <button type="button" key={v} role="radio" aria-checked={settings.every === v}
                      onClick={() => setSettings({ ...settings, every: v })}>Every {label}</button>
                  ))}
                  <button type="button" role="radio" aria-checked={settings.every === "other"}
                    onClick={() => setSettings({ ...settings, every: "other" })}>Other</button>
                </div>
                {settings.every === "other" && (
                  <div className="row try-gap-s">
                    <input className="input" inputMode="decimal" aria-label="Time between snapshots"
                      placeholder="e.g. 30" value={settings.otherValue}
                      onChange={(e) => setSettings({ ...settings, otherValue: e.target.value.trim() })} />
                    <select className="input" aria-label="Unit" value={settings.otherUnit}
                      onChange={(e) => setSettings({ ...settings, otherUnit: e.target.value as Settings["otherUnit"] })}>
                      <option value="s">seconds</option><option value="min">minutes</option><option value="h">hours</option>
                    </select>
                  </div>
                )}
                {iv !== null && ivOk
                  ? <p className="try-total" aria-live="polite">{p.snapshots} snapshots × {humanInterval(iv)} = <b>{humanSpan(((p.snapshots - 1) * iv) / 3600)} of running</b>. If that's not how long the recording lasted, change the choice above: every time in the report depends on it.</p>
                  : <span className="hint">Every time in the report depends on this. Check your logger's schedule if you're not sure.</span>}
                {iv !== null && !ivOk && <span className="try-warn">Choose between 1 second and 30 days.</span>}
              </fieldset>
            )}

            <fieldset className="field try-gap">
              <legend>What the sensor measures</legend>
              <div className="try-chips" role="radiogroup" aria-label="Vibration unit">
                <button type="button" role="radio" aria-checked={settings.unit === "g"}
                  onClick={() => setSettings({ ...settings, unit: "g" })}>Acceleration (g)</button>
                <button type="button" role="radio" aria-checked={settings.unit === "mm/s"}
                  onClick={() => setSettings({ ...settings, unit: "mm/s" })}>Velocity (mm/s)</button>
              </div>
              <span className="hint">Most vibration sensors measure acceleration. Not sure? Keep g.</span>
            </fieldset>

            <div className="row try-actions">
              <button className="btn primary" disabled={!rateOk || settings.rate === "" && !isWav || !ivOk || !!busy}
                onClick={onNext}>Continue</button>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

// --- step 2: your machine --------------------------------------------------------------

function MachineStep({ limits, settings, setSettings, onBack, onNext, onError }: {
  limits: Limits; settings: Settings; setSettings: (s: Settings) => void;
  onBack: () => void; onNext: () => Promise<void>; onError: (msg: string | null) => void;
}) {
  const [busy, setBusy] = useState(false);
  const famParams = limits.family_info?.[settings.family]?.parameters ?? [];
  const rpmN = settings.rpm.trim() === "" ? null : Number(settings.rpm);
  const rpmOk = rpmN === null || (Number.isFinite(rpmN) && rpmN >= 60 && rpmN <= 60000);
  const paramBad = famParams.filter((p) => {
    const v = (settings.params[p.key] ?? "").trim();
    if (v === "") return false;
    const n = Number(v);
    return !(Number.isInteger(n) && n >= p.min && n <= p.max);
  });
  const go = async () => {
    setBusy(true);
    try { await onNext(); } catch (e) {
      onError(`${errText(e)} Check the settings on steps 1 and 2, or choose a different file.`);
    } finally { setBusy(false); }
  };
  return (
    <section className="card" aria-labelledby="s2">
      <div className="card-h"><h2 id="s2" tabIndex={-1}>2. Your machine</h2></div>
      <div className="card-b">
        <p className="try-why">The machine type tells Harmonic which faults to look for.</p>
        <div className="family-grid" role="radiogroup" aria-label="Machine type">
          {Object.entries(limits.families).map(([k, label]) => {
            const names = limits.family_info?.[k]?.names_faults !== false;
            return (
              <button key={k} type="button" role="radio" aria-checked={settings.family === k}
                onClick={() => setSettings({ ...settings, family: k })}>
                {label}
                <small>{names ? "Harmonic can name this type's common faults"
                  : "Any other machine, or not sure: Harmonic watches for any change but can't name the fault"}</small>
              </button>
            );
          })}
        </div>
        <p className="hint">Not sure? Choose Other: a wrong type can name a wrong fault.</p>

        <div className="field try-gap">
          <label htmlFor="try-rpm">Shaft speed in rpm <span>optional</span></label>
          <input id="try-rpm" className="input" inputMode="decimal" placeholder="e.g. 1480"
            value={settings.rpm} onChange={(e) => setSettings({ ...settings, rpm: e.target.value })}
            aria-invalid={!rpmOk} aria-describedby="try-rpm-h" />
          <span className="hint" id="try-rpm-h">From the motor's nameplate or your drive. Faults
            show up at speeds tied to the shaft's, so the real speed makes the diagnosis sharper.
            Left empty, a typical speed for the machine type is assumed and the report says so.</span>
          {!rpmOk && <span className="try-warn">Enter a number between 60 and 60000, or leave it empty.</span>}
        </div>

        {famParams.length > 0 && (
          <details className="try-more">
            <summary>More about your machine (optional)</summary>
            <div className="wz-grid try-gap-s">
              {famParams.map((p) => (
                <div className="field" key={p.key}>
                  <label htmlFor={`try-p-${p.key}`}>{p.label}</label>
                  <input id={`try-p-${p.key}`} className="input" inputMode="numeric"
                    placeholder={`typical: ${p.default}`} value={settings.params[p.key] ?? ""}
                    onChange={(e) => setSettings({ ...settings, params: { ...settings.params, [p.key]: e.target.value } })}
                    aria-invalid={paramBad.includes(p)} />
                  {paramBad.includes(p)
                    ? <span className="try-warn">A whole number from {p.min} to {p.max}, or leave it empty.</span>
                    : <span className="hint">Left empty, {p.default} is assumed and the report says so.</span>}
                </div>
              ))}
            </div>
          </details>
        )}

        <div className="row try-actions">
          <button className="btn ghost" onClick={onBack} disabled={busy}>Back</button>
          <button className="btn primary" onClick={go} disabled={busy || !rpmOk || paramBad.length > 0}>
            {busy ? "Reading your recording…" : "Continue"}
          </button>
        </div>
      </div>
    </section>
  );
}

// --- step 3: when was it running normally? -------------------------------------------------

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}

function LearnStep({ runId, overview, limits, sample, settings, onStarted, onBack, onError, onChannel }: {
  runId: string; overview: Overview; limits: Limits; sample: SampleInfo | null; settings: Settings;
  onStarted: () => void; onBack: () => void; onError: (msg: string | null) => void;
  onChannel: (channel: number) => Promise<void>;
}) {
  const hours = overview.hours;
  const n = hours.length;
  // Index of the last learning snapshot: at least the minimum commissioning
  // learns from, and never so late that nothing is left to watch.
  const lo = Math.min(Math.max(overview.min_baseline_snapshots - 1, 1), Math.max(n - 2, 1));
  const hi = Math.max(n - 2, lo);
  const clamp = (i: number) => Math.min(Math.max(i, lo), hi);
  const initial = hours.findIndex((h) => h >= overview.default_baseline_h[1]);
  const [split, setSplit] = useState(clamp(initial < 0 ? lo : initial));
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const svg = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);

  const h0 = hours[0];
  const span = Math.max(hours[n - 1] - h0, 1e-9);
  const x = (h: number) => ((h - h0) / span) * 1000;
  // Scaled to the learning period's own range (x3), so a big failure at the
  // end does not flatten the part the visitor is choosing from.
  const learnMax = Math.max(...overview.rms.slice(0, split + 1), 1e-12);
  const top = Math.max(learnMax * 3, 1e-12);
  const path = useMemo(() => overview.rms.map((v, i) =>
    `${i ? "L" : "M"}${x(hours[i]).toFixed(1)},${(200 - (Math.min(v, top) / top) * 190).toFixed(1)}`).join(""),
  [overview, top]);

  const fromPointer = (clientX: number) => {
    const r = svg.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    const h = h0 + ((clientX - r.left) / r.width) * span;
    let best = lo;
    for (let i = lo; i <= hi; i++) if (Math.abs(hours[i] - h) < Math.abs(hours[best] - h)) best = i;
    setSplit(clamp(best));
  };

  // Plain warnings: the shortest allowed period, little left to watch, or a
  // vibration level that already rises inside the period.
  const learnH = hours[split] - h0;
  const watchH = hours[n - 1] - hours[split];
  const period = overview.rms.slice(0, split + 1);
  const early = median(period.slice(0, Math.ceil(period.length / 2)));
  const late = median(period.slice(Math.floor(period.length * 0.8)));
  const rising = early > 0 && late > early * 1.3;
  const atMin = split === lo;

  const start = async () => {
    setBusy(true);
    onError(null);
    try {
      await api.start(runId, [hours[0], hours[split]], consent);
      onStarted();
    } catch (e) {
      onError(`${errText(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card" aria-labelledby="s3">
      <div className="card-h"><h2 id="s3" tabIndex={-1}>3. When was it running normally?</h2></div>
      <div className="card-b">
        {sample && (
          <div className="try-sample-note">
            <b>Sample recording: {sample.title}.</b> {sample.story} The settings are filled in
            for you: {sample.sample_rate_hz / 1000} kHz, a snapshot every{" "}
            {humanInterval(sample.interval_s)}, {sample.rpm} rpm, machine type "
            {limits.families[settings.family] ?? settings.family}" (the rig is none of the listed
            types). <span className="hint">{sample.credit}</span>
          </div>
        )}
        <p className="try-why">Pick a stretch when the machine was running fine. Harmonic learns
          "normal" from it. The first part of the recording is chosen for you.</p>
        <svg ref={svg} className="baseline-chart" viewBox="0 0 1000 200" preserveAspectRatio="none"
          role="img" aria-label={`Vibration level over the recording. The learning period is the first ${humanSpan(learnH)}.`}
          onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); fromPointer(e.clientX); }}
          onPointerMove={(e) => { if (dragging.current) fromPointer(e.clientX); }}
          onPointerUp={() => { dragging.current = false; }}>
          <rect x={0} y={0} width={x(hours[split])} height={200} className="learn-shade" />
          <path d={path} fill="none" className="learn-line" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
          <line className="baseline-handle" x1={x(hours[split])} x2={x(hours[split])} y1={0} y2={200}
            strokeWidth={3} vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="hc-x"><span>Start</span><span>{humanSpan(span)}</span></div>
        <label className="try-range-l" htmlFor="try-split">Drag the line, or use this slider, to
          set where the learning period ends</label>
        <input id="try-split" className="try-range" type="range" min={lo} max={hi} value={split}
          aria-valuetext={`Learning from the first ${humanSpan(learnH)}`}
          onChange={(e) => setSplit(Number(e.target.value))} />
        <div className="kv-grid try-gap">
          <div><div className="k">Learning normal from</div>
            <div className="v">the first {humanSpan(learnH)} ({split + 1} snapshots)</div></div>
          <div><div className="k">Then watching</div>
            <div className="v">the next {humanSpan(watchH)} ({n - split - 1} snapshots)</div></div>
        </div>
        <p className="hint">The chart shows the overall vibration level of each snapshot.</p>
        {rising && (
          <p className="try-warn" role="status">The vibration level is already rising inside
            this period, so the machine may be changing already. Anything it learns as normal it
            won't warn about. If you can, end the period earlier.</p>
        )}
        {atMin && !rising && (
          <p className="hint">This is the shortest period Harmonic can learn from
            ({overview.min_baseline_snapshots} snapshots). If the machine ran fine for longer,
            a longer period gives fewer false alarms.</p>
        )}
        {watchH < learnH / 4 && (
          <p className="try-warn" role="status">Only {humanSpan(watchH)} is left to watch after
            the learning period.</p>
        )}
        {overview.n_channels > 1 && (
          <div className="field try-gap">
            <label htmlFor="try-ch">Sensor channel to analyse</label>
            <select id="try-ch" className="input" value={overview.channel} disabled={busy}
              onChange={async (e) => {
                setBusy(true); onError(null);
                try { await onChannel(Number(e.target.value)); }
                catch (err) { onError(errText(err)); }
                finally { setBusy(false); }
              }}>
              {Array.from({ length: overview.n_channels }, (_, i) => (
                <option key={i} value={i}>Channel {i + 1} of {overview.n_channels}</option>
              ))}
            </select>
          </div>
        )}
        <label className="consent-row try-gap">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          <span>Let Harmonic keep this recording to improve its models. Left unticked, it is
            deleted after {limits.retention_days} days; ticked, it is kept for at most
            {" "}{limits.consent_retention_days} days. Either way you can delete it sooner from
            the result page.</span>
        </label>
        <div className="row try-actions">
          <button className="btn ghost" onClick={onBack} disabled={busy}>Back</button>
          <button className="btn primary" onClick={start} disabled={busy || n < 3}>
            {busy ? "Starting…" : "Run Harmonic"}
          </button>
        </div>
      </div>
    </section>
  );
}

// --- running -----------------------------------------------------------------------

const STAGES: { key: string; label: string }[] = [
  { key: "learn", label: "Learning what normal looks like…" },
  { key: "watch", label: "Watching the rest of the recording…" },
  { key: "write", label: "Writing the report…" },
];

function RunningStep({ runId, initial, onDone, onRestart, onDelete, onCancel }: {
  runId: string; initial: Run | null;
  onDone: (r: Run) => void; onRestart: () => void; onDelete: () => void; onCancel: () => void;
}) {
  const [run, setRun] = useState<Run | null>(initial);
  useEffect(() => {
    let live = true;
    let timer = 0;
    const poll = async () => {
      try {
        const r = await api.run(runId);
        if (!live) return;
        setRun(r);
        if (r.status === "done") { onDone(r); return; }
        if (r.status === "failed") return;
      } catch { /* transient; poll again */ }
      if (live) timer = window.setTimeout(poll, 1200);
    };
    poll();
    return () => { live = false; window.clearTimeout(timer); };
  }, [runId, onDone]);

  if (run?.status === "failed") {
    return (
      <section className="card"><div className="card-b progress-view" role="alert">
        <b>Harmonic could not finish this recording.</b>
        <span className="muted">{run.error}</span>
        <span className="hint">Try again with a different learning period or file. If it keeps
          failing, talk to us and we'll look at it.</span>
        <div className="row try-actions">
          <button className="btn ghost" onClick={onDelete}>Delete now</button>
          <button className="btn primary" onClick={onRestart}>Start again</button>
        </div>
      </div></section>
    );
  }
  const p = run?.progress ?? {};
  const stage = run?.status === "queued" || !p.stage ? "queue" : p.stage;
  const at = STAGES.findIndex((s) => s.key === stage);
  return (
    <section className="card" aria-labelledby="s-run">
      <div className="card-h"><h2 id="s-run" tabIndex={-1}>Running Harmonic on your recording</h2></div>
      <div className="card-b">
        <ol className="try-stages" aria-live="polite">
          {stage === "queue" && (
            <li className="on"><span className="spinner sm" aria-hidden="true" />
              Waiting for a free slot{p.ahead ? ` (${p.ahead} ahead of you)` : ""}…</li>
          )}
          {STAGES.map((s, i) => (
            <li key={s.key} className={i < at ? "done" : i === at ? "on" : ""}>
              {i < at ? <span className="tick" aria-hidden="true">✓</span>
                : i === at ? <span className="spinner sm" aria-hidden="true" />
                  : <span className="dot" aria-hidden="true" />}
              {s.label}
              {s.key === "watch" && i === at && p.total
                ? ` ${Math.round((100 * (p.done ?? 0)) / p.total)}%` : ""}
            </li>
          ))}
        </ol>
        <p className="hint">Usually a minute or less. You can keep this page open; your result
          appears here.</p>
        <div className="row try-actions">
          <button className="btn ghost" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </section>
  );
}

// --- step 4: the result -------------------------------------------------------------------

function ReportView({ run, report, limits, onDelete, onRestart, onChangeSettings, changing }: {
  run: Run; report: Report; limits: Limits | null; onDelete: () => void; onRestart: () => void;
  onChangeSettings: () => void; changing: boolean;
}) {
  const contact = safeHref(report.contact_url ?? limits?.contact_url);
  const deleteOn = new Date((run.created + (limits?.retention_days ?? 7) * 86400) * 1000);
  const d = report.diagnosis?.response;
  const chart = useMemo(() => chartHtml(report), [report]);
  const [downloading, setDownloading] = useState(false);
  const today = () => new Date().toISOString().slice(0, 10);

  const downloadHtml = () => {
    const html = reportHtml(run, report, contact, limits?.retention_days ?? 7);
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `harmonic-report-${today()}.html`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  // S10-T06: a PDF, built on the fly (jspdf and svg2pdf.js load only now,
  // not on the page's first visit). If it throws for any reason, the
  // visitor still gets their report, as the HTML file "Download report"
  // always produced.
  const downloadPdf = async () => {
    setDownloading(true);
    try {
      const { buildReportPdf } = await import("./reportPdf");
      const doc = await buildReportPdf(run, report, contact, limits?.retention_days ?? 7);
      doc.save(`harmonic-report-${today()}.pdf`);
    } catch {
      window.alert("The PDF could not be built, so the report downloaded as an HTML file instead.");
      downloadHtml();
    } finally {
      setDownloading(false);
    }
  };

  return (
    <>
      <section className="card" aria-labelledby="s4">
        <div className="card-h"><h2 id="s4" tabIndex={-1}>4. Result</h2></div>
        <div className="card-b">
          <p className="report-headline">{report.headline}</p>
          {report.sample && <p className="hint">Sample recording: {report.sample.title}. The
            real bearing ended with an outer-race failure. {report.sample.credit}</p>}
          <div className="try-chart" dangerouslySetInnerHTML={{ __html: chart }} />
          <div className="alarm-list">
            {LEVELS.map((l) => {
              const a = report[l.key === "WATCH" ? "first_watch" : l.key === "WARNING" ? "first_warning" : "first_critical"];
              return (
                <div className={`alarm-row s-${l.key}`} key={l.key}>
                  <div className="alarm-name"><span className="state-dot" aria-hidden="true" />{l.name}</div>
                  <div className="alarm-when">{a ? <b>Reached {a.before_end} before the end</b> : "Not reached"}</div>
                  <div className="alarm-means">{l.means}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="card" aria-labelledby="s-changed">
        <div className="card-h"><h2 id="s-changed">What changed</h2></div>
        <div className="card-b">
          {report.evidence.length === 0
            ? <p className="muted">Nothing moved far enough from this machine's normal to report.</p>
            : <>
                <ul className="evidence-list">{report.evidence.map((e) => <li key={e}>{e}</li>)}</ul>
                {report.evidence_note && <p className="hint try-gap-s">{report.evidence_note}</p>}
              </>}
          {d && (
            <div className="try-diag">
              <p>{d.diagnosis}</p>
              {(!report.names_faults || /nrecognised/.test(d.diagnosis)) && (
                <p className="hint">{UNRECOGNISED_NOTE}</p>
              )}
            </div>
          )}
        </div>
      </section>

      {d && (
        <section className="card" aria-labelledby="s-do">
          <div className="card-h"><h2 id="s-do">What to do</h2></div>
          <div className="card-b try-diag">
            {d.actions?.length ? <ol className="evidence-list">{d.actions.map((a) => <li key={a.step}>{a.step}</li>)}</ol> : null}
            {d.safety?.length ? <><div className="k">Safety first</div>
              <ul className="evidence-list">{d.safety.map((s) => <li key={s}>{s}</li>)}</ul></> : null}
            {d.root_causes?.length ? <><div className="k">Likely causes</div>
              <ul className="evidence-list">{d.root_causes.map((c) => <li key={c}>{c}</li>)}</ul></> : null}
            {d.guidance_note && <p className="hint">{d.guidance_note}</p>}
            {d.references?.length ? <p className="hint">Cites {d.references.join(", ")}
              {d.references_label ? ` (${d.references_label})` : ""}.</p> : null}
          </div>
        </section>
      )}

      <section className="card">
        <div className="card-b">
          <details className="try-more">
            <summary>What this can't tell you</summary>
            <ul className="caveats-list try-gap-s">{report.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
            <div className="k try-gap">Settings this result used</div>
            <ul className="caveats-list">
              <li>Machine type: {report.family_label}</li>
              <li>Recording: {report.n_snapshots} snapshots over {report.span_text}, {report.sample_rate_hz.toLocaleString()} readings per second{report.n_channels > 1 ? `, channel ${report.channel} of ${report.n_channels}` : ""}</li>
              <li>Time between snapshots: {report.timed_by === "timestamps" ? "from the recording's time stamps"
                : `${report.interval_text ?? ""}${report.timed_by === "wav" ? " (back to back, from the WAV file)" : " (as you chose)"}`}</li>
              <li>Learning period: the first {report.baseline_text}; false alarms while learning: {report.false_alarms_in_baseline}</li>
              <li>Shaft speed: {report.rpm_used} rpm{report.rpm_limited ? " (assumed)" : ""}</li>
              {report.assumed?.map((a) => <li key={a}>Assumed: {a}</li>)}
            </ul>
          </details>
        </div>
      </section>

      {contact && (
        <div className="contact-banner">
          <div><b>Run this on your whole fleet</b>
            <div className="hint">Live watching, your machines' own manuals, and the full diagnosis.</div></div>
          <a className="btn primary" href={contact} target="_blank" rel="noopener noreferrer">Talk to us</a>
        </div>
      )}

      <section className="card">
        <div className="card-b">
          <div className="row try-actions">
            <button className="btn primary" onClick={downloadPdf} disabled={downloading}>
              {downloading ? "Building the PDF…" : "Download report (PDF)"}
            </button>
            <button className="btn ghost" onClick={onChangeSettings} disabled={changing}>
              {changing ? "Changing settings…" : "Change settings and run again"}
            </button>
            <button className="btn ghost" onClick={onDelete}>Delete now</button>
            <button className="btn ghost" onClick={onRestart}>Try another recording</button>
          </div>
          <p className="hint try-gap">
            {run.consent
              ? `You let Harmonic keep this recording to improve its models; kept recordings are `
                + `deleted after ${limits?.consent_retention_days ?? 90} days. Delete now removes it sooner.`
              : `This recording and result are deleted on ${deleteOn.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })}. `
                + "Download the report to keep a copy."}
          </p>
        </div>
      </section>
    </>
  );
}

