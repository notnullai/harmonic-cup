import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import { DataZoomComponent, GridComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import {
  Check, CheckCircle2, CircleAlert, Cog, Database, FolderUp, Gauge, Loader2, Play, Server, X,
  Radio,
} from "lucide-react";
import { api, type Bearing, type ListenStatus, type MachineCatalog, type MachineRecord, type RecordingInfo } from "../api";

echarts.use([LineChart, GridComponent, TooltipComponent, DataZoomComponent, CanvasRenderer]);

const STEPS = ["Machine", "Data", "Baseline", "Commission"] as const;
const COMPONENT_LABEL: Record<string, string> = {
  bearing: "Bearings", shaft: "Shaft", foundation: "Foundation", coupling: "Coupling",
  motor: "Motor", impeller: "Impeller", seal: "Seal", gear: "Gears",
};

/** Bearing defect frequencies at the running speed, in Hz — shown so the user
 *  can check the geometry against the bearing maker's sheet. */
function defects(b: Bearing, rpm: number) {
  const fr = rpm / 60, r = (b.ball_mm / b.pitch_mm) * Math.cos((b.contact_angle_deg * Math.PI) / 180);
  return {
    BPFO: (b.n_balls / 2) * fr * (1 - r),
    BPFI: (b.n_balls / 2) * fr * (1 + r),
    BSF: (b.pitch_mm / (2 * b.ball_mm)) * fr * (1 - r * r),
  };
}

interface Props {
  open: boolean;
  onClose: () => void;
  onWatch: (machineId: string) => void;
  toast: (msg: string, err?: boolean) => void;
}

export default function Wizard({ open, onClose, onWatch, toast }: Props) {
  const [step, setStep] = useState(0);
  const [cat, setCat] = useState<MachineCatalog | null>(null);
  const [recs, setRecs] = useState<RecordingInfo[]>([]);

  // step 1
  const [name, setName] = useState("");
  const [family, setFamily] = useState("pump");
  const [params, setParams] = useState<Record<string, number>>({});
  const [rpm, setRpm] = useState(1500);
  const [unit, setUnit] = useState<"g" | "mm/s">("g");
  const [iso, setIso] = useState("none");
  const [preset, setPreset] = useState("6205");
  const [bearing, setBearing] = useState<Bearing>({ pitch_mm: 39.04, ball_mm: 7.94, n_balls: 9, contact_angle_deg: 0 });
  const [components, setComponents] = useState<string[]>(["bearing", "shaft", "foundation", "motor", "coupling"]);
  // step 2
  const [recKey, setRecKey] = useState<string | null>(null);
  const [channel, setChannel] = useState(0);
  // Measurement points (HARM-S7-T07): the channels watched, each by name.
  // The first is the one the baseline is drawn from.
  const [points, setPoints] = useState<{ name: string; channel: number }[]>([]);
  const [speed, setSpeed] = useState<"constant" | "measured" | "estimated">("constant");
  const [tab, setTab] = useState<"server" | "upload" | "live">("server");
  // A live sensor being listened to, and whether to keep watching it.
  const [liveCfg, setLiveCfg] = useState<{ id: string; broker: string; topic: string } | null>(null);
  const [keepWatching, setKeepWatching] = useState(true);
  // step 3
  const [baseline, setBaseline] = useState<[number, number] | null>(null);
  // step 4
  const [job, setJob] = useState<MachineRecord | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const refreshRecs = () => api.recordings().then((r) => setRecs(r.recordings)).catch(() => {});
  useEffect(() => {
    if (!open) return;
    api.machines().then(setCat).catch(() => {});
    refreshRecs();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && !submitting && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Poll a running commission.
  useEffect(() => {
    if (!job || job.status === "ready" || job.status === "failed") return;
    const t = setTimeout(() => api.machine(job.id).then(setJob).catch(() => {}), 700);
    return () => clearTimeout(t);
  }, [job]);

  const rec = recs.find((r) => r.key === recKey);
  const minReadings = cat?.min_baseline_readings ?? 160;

  const reset = () => {
    setStep(0); setName(""); pickFamily("pump"); setRecKey(null); setChannel(0); setPoints([]); setSpeed("constant"); setBaseline(null); setJob(null);
  };
  const close = () => { if (job?.status === "commissioning") return; reset(); onClose(); };

  if (!open) return null;

  const fam = cat?.families?.[family];
  const pickFamily = (key: string) => {
    const w = cat?.families?.[key]?.wizard;
    setFamily(key);
    if (w?.unit === "g" || w?.unit === "mm/s") setUnit(w.unit);
    if (w?.components) setComponents(w.components);
    setParams(Object.fromEntries((w?.parameters ?? []).map((p) => [p.key, p.default])));
  };
  const paramsOk = (fam?.wizard.parameters ?? []).every((p) =>
    Number.isInteger(params[p.key]) && params[p.key] >= p.min && params[p.key] <= p.max);
  const valid1 = paramsOk && name.trim().length > 0 && rpm >= 60 && bearing.pitch_mm > bearing.ball_mm && bearing.ball_mm > 0 && bearing.n_balls >= 3;
  const pointNames = points.map((p) => p.name.trim());
  const valid2 = !!rec && (points.length < 2 || (pointNames.every(Boolean) && new Set(pointNames).size === pointNames.length));

  const submit = async () => {
    if (!rec || !baseline) return;
    setSubmitting(true);
    try {
      const m = await api.createMachine({
        name: name.trim(), nominal_rpm: rpm, vib_unit: unit, iso_class: iso,
        bearing, components, recording: rec.key, channel, baseline_h: baseline, speed,
        family, params, points: points.length > 1 ? points : undefined,
        live: liveCfg && keepWatching && rec.key === `upload:${liveCfg.id}`
          ? { broker: liveCfg.broker, topic: liveCfg.topic } : undefined,
      });
      setJob(m);
      // The machine watches the topic itself now; the recording is complete.
      if (liveCfg && keepWatching) api.stopListen(liveCfg.id).catch(() => {});
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="scrim" />
      <div className="dialog wizard" role="dialog" aria-modal="true" aria-labelledby="wz-title">
        <div className="wz-head">
          <Cog size={20} />
          <div>
            <h3 id="wz-title">Commission a machine</h3>
            <div className="hint">Learn what normal looks like for one machine, from its own healthy running.</div>
          </div>
          <span style={{ flex: 1 }} />
          <button className="btn ghost icon sm" onClick={close} aria-label="Close"
            disabled={job?.status === "commissioning"}><X size={16} /></button>
        </div>

        <ol className="stepper" aria-label="Steps">
          {STEPS.map((s, i) => (
            <li key={s} className={i < step ? "done" : i === step ? "on" : ""}>
              <span className="n">{i < step ? <Check size={12} /> : i + 1}</span>{s}
            </li>
          ))}
        </ol>

        <div className="body wz-body">
          {step === 0 && (
            <div className="wz-grid">
              {cat?.families && (
                <div className="field wide">
                  <label>Machine family</label>
                  <div className="seg" role="group" aria-label="Machine family">
                    {Object.entries(cat.families).map(([k, f]) => (
                      <button key={k} aria-pressed={family === k} disabled={!f.available}
                        title={f.available ? undefined : "Its network is not built on this server"}
                        onClick={() => pickFamily(k)}>{f.label}</button>
                    ))}
                  </div>
                  <span className="hint">Chooses the trained network, the faults it can name, and the knowledge the diagnosis draws on.</span>
                </div>
              )}
              <div className="field">
                <label htmlFor="wz-name">Name</label>
                <input id="wz-name" className="input" value={name} autoFocus
                  placeholder="e.g. Cooling water pump CW-2" onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="wz-rpm">Running speed</label>
                <div className="with-unit"><input id="wz-rpm" className="input" type="number" min={60} max={60000}
                  value={rpm} onChange={(e) => setRpm(Number(e.target.value))} /><span>rpm</span></div>
              </div>
              {(fam?.wizard.parameters ?? []).map((p) => (
                <div className="field" key={p.key}>
                  <label htmlFor={`wz-p-${p.key}`}>{p.label}</label>
                  <input id={`wz-p-${p.key}`} className="input" type="number" min={p.min} max={p.max} step={1}
                    value={params[p.key] ?? p.default}
                    onChange={(e) => setParams({ ...params, [p.key]: Number(e.target.value) })} />
                </div>
              ))}
              <div className="field">
                <label>Vibration sensor</label>
                <div className="seg" role="group">
                  <button aria-pressed={unit === "g"} onClick={() => setUnit("g")}>
                    Acceleration · g</button>
                  <button aria-pressed={unit === "mm/s"} onClick={() => { setUnit("mm/s"); if (iso === "none") setIso("10816-1-II"); }}>
                    Velocity · mm/s</button>
                </div>
              </div>
              <div className="field">
                <label htmlFor="wz-iso">Alarm zones</label>
                <select id="wz-iso" className="input" value={iso}
                  onChange={(e) => setIso(e.target.value)}>
                  {Object.entries(cat?.iso_classes ?? {}).map(([k, v]) =>
                    <option key={k} value={k}>{v}</option>)}
                </select>
                <span className="hint">{unit === "g"
                  ? "Zones are velocity limits; the engine integrates the accelerometer to velocity (10–1000 Hz) to judge them."
                  : "Published limits: a reading in zone C raises WARNING whatever the learned baseline says."}</span>
                {iso === "none" && (
                  <span className="warn-text">No zones: absolute vibration limits will not be checked, so a severe
                    level is judged only against what the machine learns. Choose the class for its size and
                    mounting unless the sensor's unit or placement makes ISO 10816 meaningless here.</span>
                )}
              </div>
              <div className="field wide">
                <label htmlFor="wz-bearing">Bearing</label>
                <select id="wz-bearing" className="input" value={preset} onChange={(e) => {
                  setPreset(e.target.value);
                  const b = cat?.bearings[e.target.value];
                  if (b) setBearing({ pitch_mm: b.pitch_mm, ball_mm: b.ball_mm, n_balls: b.n_balls, contact_angle_deg: b.contact_angle_deg });
                }}>
                  {Object.entries(cat?.bearings ?? {}).map(([k, b]) => <option key={k} value={k}>{b.label}</option>)}
                  <option value="custom">Other — enter the geometry</option>
                </select>
                {cat?.bearings[preset]?.source && <span className="hint">Geometry from: {cat.bearings[preset].source}</span>}
                <div className="wz-4">
                  {([["pitch_mm", "Pitch diameter", "mm"], ["ball_mm", "Ball diameter", "mm"],
                    ["n_balls", "Balls per row", ""], ["contact_angle_deg", "Contact angle", "°"]] as const).map(([k, l, u]) => (
                    <label key={k} className="hint">{l}
                      <div className="with-unit"><input className="input" type="number" step="0.01" value={bearing[k]}
                        onChange={(e) => { setPreset("custom"); setBearing({ ...bearing, [k]: Number(e.target.value) }); }} />
                        {u && <span>{u}</span>}</div></label>
                  ))}
                </div>
                {valid1 && (() => {
                  const d = defects(bearing, rpm);
                  return <span className="hint">At {rpm} rpm: outer race {d.BPFO.toFixed(1)} Hz · inner race {d.BPFI.toFixed(1)} Hz · rolling element {d.BSF.toFixed(1)} Hz — check them against the bearing maker's sheet.</span>;
                })()}
              </div>
              <div className="field wide">
                <label>What the machine has</label>
                <div className="row">
                  {(cat?.components ?? Object.keys(COMPONENT_LABEL)).map((c) => (
                    <button key={c} className="chip toggle" aria-pressed={components.includes(c)} disabled={c === "bearing"}
                      onClick={() => setComponents(components.includes(c) ? components.filter((x) => x !== c) : [...components, c])}>
                      {components.includes(c) && <Check size={12} />}{COMPONENT_LABEL[c] ?? c}</button>
                  ))}
                </div>
                <span className="hint">The diagnosis only considers failures of parts the machine has — no cavitation without an impeller.</span>
              </div>
            </div>
          )}

          {step === 1 && (
            <>
              <div className="tabs" role="tablist">
                <button role="tab" aria-selected={tab === "server"} onClick={() => setTab("server")}><Server size={13} /> On the server</button>
                <button role="tab" aria-selected={tab === "upload"} onClick={() => setTab("upload")}><FolderUp size={13} /> Upload</button>
                <button role="tab" aria-selected={tab === "live"} onClick={() => setTab("live")}><Radio size={13} /> Live sensor</button>
              </div>
              {tab === "live" ? (
                <LiveListen cfg={liveCfg} toast={toast} keep={keepWatching} onKeep={setKeepWatching}
                  onStarted={setLiveCfg}
                  onRecording={(key) => refreshRecs().then(() => { setRecKey(key); setBaseline(null); })} />
              ) : tab === "server" ? (
                <div className="rec-list">
                  {recs.length === 0 && <div className="empty"><Database size={26} /><div>No recordings on the server yet. Upload one.</div></div>}
                  {recs.map((r) => (
                    <button key={r.key} className="rec" aria-pressed={recKey === r.key}
                      onClick={() => { setRecKey(r.key); setChannel(0); setPoints([]); setBaseline(null); setSpeed(r.has_speed ? "measured" : "constant"); }}>
                      <Database size={18} />
                      <span>
                        <b>{r.name}</b>
                        <span className="hint">{r.snapshots} snapshots over {r.span_h.toFixed(1)} h · {r.sample_rate_hz / 1000} kHz · {r.channels} channel{r.channels > 1 ? "s" : ""} · {r.unit}</span>
                        <span className="hint">{r.provenance}</span>
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <Upload unit={unit} onDone={(key) => { refreshRecs().then(() => { setRecKey(key); setTab("server"); }); }} toast={toast} />
              )}
              {rec && rec.channels > 1 && (
                <div className="field" style={{ marginTop: 14 }}>
                  <label>Measurement points</label>
                  <span className="hint">Tick every channel that watches this machine and name where it is
                    (drive end, non-drive end; horizontal, vertical). Each point gets its own engine; the
                    machine is as bad as its worst point.</span>
                  <div className="points">
                    {Array.from({ length: rec.channels }, (_, i) => {
                      const at = points.findIndex((p) => p.channel === i);
                      const on = at >= 0 || (points.length === 0 && i === channel);
                      return (
                        <div key={i} className="point-row">
                          <label><input type="checkbox" checked={on} disabled={!!liveCfg}
                            onChange={(e) => {
                              const cur = points.length ? points : [{ name: `Channel ${channel + 1}`, channel }];
                              const next = e.target.checked
                                ? [...cur, { name: `Channel ${i + 1}`, channel: i }]
                                : cur.filter((p) => p.channel !== i);
                              if (!next.length) return;
                              setPoints(next);
                              if (next[0].channel !== channel) { setChannel(next[0].channel); setBaseline(null); }
                            }} /> Channel {i + 1}</label>
                          {on && points.length > 1 && (
                            <input className="input sm" maxLength={40} value={points[at]?.name ?? ""}
                              aria-label={`Name of channel ${i + 1}`}
                              onChange={(e) => setPoints(points.map((p) => (p.channel === i ? { ...p, name: e.target.value } : p)))} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              {rec && (
                <div className="field" style={{ marginTop: 14 }}>
                  <label>Shaft speed</label>
                  <div className="seg" role="group">
                    <button aria-pressed={speed === "constant"} onClick={() => setSpeed("constant")}>Nominal · {rpm} rpm</button>
                    <button aria-pressed={speed === "measured"} disabled={!rec.has_speed} onClick={() => setSpeed("measured")}>From the recording</button>
                    <button aria-pressed={speed === "estimated"} onClick={() => setSpeed("estimated")}>Estimate from vibration</button>
                  </div>
                  <span className="hint">{speed === "measured"
                    ? "The recording's own speed per snapshot: every frequency follows the real shaft."
                    : speed === "estimated"
                    ? "Read from the loudest line near the nominal speed. Commissioning checks it is steady and refuses it if not."
                    : rec.has_speed ? "The recording has a speed column; the nominal speed ignores it." : "The machine runs at a fixed speed. No speed column in this recording."}</span>
                </div>
              )}
              {rec && rec.unit !== unit && (
                <div className="note" style={{ marginTop: 12 }}><CircleAlert size={12} style={{ verticalAlign: -2 }} /> The recording says its unit is {rec.unit}; the machine was set up for {unit}. Go back and change one of them if that is not intended.</div>
              )}
            </>
          )}

          {step === 2 && rec && (
            <BaselinePicker recKey={rec.key} channel={channel} min={minReadings}
              value={baseline} onChange={setBaseline} />
          )}

          {step === 3 && (
            <Commission job={job} name={name} rec={rec} channel={channel} baseline={baseline}
              submitting={submitting} onSubmit={submit} />
          )}
        </div>

        <div className="foot">
          {step > 0 && !job && <button className="btn" onClick={() => setStep(step - 1)}>Back</button>}
          {job?.status === "failed" && <button className="btn" onClick={() => { setJob(null); setStep(2); }}>Choose another baseline</button>}
          <span style={{ flex: 1 }} />
          {step < 3 && (
            <button className="btn primary" disabled={(step === 0 && !valid1) || (step === 1 && !valid2) || (step === 2 && !baseline)}
              onClick={() => setStep(step + 1)}>Next</button>
          )}
          {step === 3 && job?.status === "ready" && (
            <button className="btn primary" onClick={() => { const id = job.id; reset(); onClose(); onWatch(id); }}>
              <Play size={15} />Watch {job.name}</button>
          )}
        </div>
      </div>
    </>
  );
}

// --- upload ------------------------------------------------------------------------

function Upload({ unit, onDone, toast }: { unit: string; onDone: (key: string) => void; toast: Props["toast"] }) {
  const [files, setFiles] = useState<File[]>([]);
  const [rate, setRate] = useState(20000);
  const [interval, setInterval_] = useState(600);
  const [label, setLabel] = useState("");
  const [sent, setSent] = useState(0);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const add = (list: FileList | null) => {
    if (!list) return;
    const ok = Array.from(list).filter((f) => /\.(csv|txt|npy)$/i.test(f.name));
    if (ok.length < list.length) toast("Only .csv, .txt and .npy files are used", true);
    setFiles((f) => [...f, ...ok].sort((a, b) => a.name.localeCompare(b.name)));
  };

  const upload = async () => {
    setBusy(true); setSent(0);
    try {
      const { id } = await api.newUpload({ name: label || `Upload of ${files.length} files`, sample_rate_hz: rate, interval_s: interval, unit });
      for (let i = 0; i < files.length; i++) {
        await api.uploadFile(id, files[i]);
        setSent(i + 1);
      }
      toast(`Uploaded ${files.length} files`);
      onDone(`upload:${id}`);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className={`drop${drag ? " over" : ""}`} onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
        role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}>
        <FolderUp size={26} />
        <b>{files.length ? `${files.length} file${files.length > 1 ? "s" : ""} ready` : "Drop snapshot files here, or click to choose"}</b>
        <span className="hint">One .csv or .txt per snapshot (a column per channel), or one .npy of shape (snapshots, samples[, channels]). Name the files by time — 2024-05-01T10-00-00.csv — or give the interval below.</span>
        <input ref={input} type="file" multiple accept=".csv,.txt,.npy" hidden onChange={(e) => add(e.target.files)} />
      </div>
      <div className="wz-4" style={{ marginTop: 12 }}>
        <label className="hint">Name<input className="input" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="optional" /></label>
        <label className="hint">Sample rate<div className="with-unit"><input className="input" type="number" value={rate} onChange={(e) => setRate(Number(e.target.value))} /><span>Hz</span></div></label>
        <label className="hint">Between snapshots<div className="with-unit"><input className="input" type="number" value={interval} onChange={(e) => setInterval_(Number(e.target.value))} /><span>s</span></div></label>
        <label className="hint">Unit<input className="input" value={unit} disabled /></label>
      </div>
      {busy && <div className="progress" style={{ marginTop: 12 }}><i style={{ width: `${(100 * sent) / Math.max(files.length, 1)}%` }} /></div>}
      <div className="row" style={{ marginTop: 12 }}>
        <button className="btn primary" disabled={!files.length || busy || rate <= 0} onClick={upload}>
          <FolderUp size={15} />{busy ? `Uploading ${sent} / ${files.length}…` : "Upload"}</button>
        {files.length > 0 && !busy && <button className="btn" onClick={() => setFiles([])}>Clear</button>}
      </div>
    </div>
  );
}

// --- the baseline -------------------------------------------------------------------

export function BaselinePicker({ recKey, channel, min, value, onChange }: {
  recKey: string; channel: number; min: number; value: [number, number] | null; onChange: (v: [number, number]) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<{ hours: number[]; at: (string | null)[]; rms: number[]; unit: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setData(null); setErr(null);
    api.overview(recKey, channel).then(setData).catch((e: Error) => setErr(e.message));
  }, [recKey, channel]);

  // Default: the first third of the recording, and never fewer readings than
  // commissioning needs.
  useEffect(() => {
    if (!data || value) return;
    const n = data.hours.length;
    const k = Math.min(n - 1, Math.max(min + 20, Math.floor(n / 3)));
    onChange([data.hours[0], data.hours[k]]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  useEffect(() => {
    if (!el.current || !data || !value) return;
    const c = echarts.init(el.current);
    const cs = getComputedStyle(document.documentElement);
    const v = (n: string) => cs.getPropertyValue(n).trim();
    const band = (w: [number, number]) => ({
      silent: true, itemStyle: { color: "rgba(47,191,113,0.16)", borderColor: v("--ok"), borderWidth: 1 },
      label: { show: true, position: "insideTop", formatter: "baseline", color: v("--ok"), fontWeight: 600 },
      data: [[{ xAxis: w[0] }, { xAxis: w[1] }]],
    });
    const label = (h: number) => {
      let i = 0;
      while (i + 1 < data.hours.length && data.hours[i + 1] <= h) i++;
      return data.at[i] ?? `${h.toFixed(1)} h`;
    };
    c.setOption({
      animation: false,
      grid: { left: 52, right: 18, top: 18, bottom: 70 },
      tooltip: { trigger: "axis", backgroundColor: v("--surface-3"), borderColor: v("--border-strong"),
        textStyle: { color: v("--text") },
        formatter: (p: { value: [number, number] }[]) => `${label(p[0].value[0])}<br/>RMS <b>${p[0].value[1].toFixed(4)}</b> ${data.unit}` },
      // The chart always shows the whole recording; the slider drives a
      // hidden twin axis, so choosing the window shades it instead of zooming
      // the rest of the recording out of sight.
      xAxis: [
        { type: "value", min: data.hours[0], max: data.hours[data.hours.length - 1],
          axisLabel: { color: v("--chart-text"), formatter: (h: number) => {
            // A timestamp loses its year ("2004-02-12 10:32" -> "02-12 10:32");
            // a recording without time stamps keeps its "12.3 h" whole.
            const s = label(h);
            return /^\d{4}-/.test(s) ? s.slice(5) : s;
          } },
          axisLine: { lineStyle: { color: v("--chart-grid") } } },
        { type: "value", min: data.hours[0], max: data.hours[data.hours.length - 1], show: false },
      ],
      yAxis: { type: "value", min: 0, name: `overall RMS (${data.unit})`, nameTextStyle: { color: v("--chart-text"), align: "left" },
        axisLabel: { color: v("--chart-text") }, splitLine: { lineStyle: { color: v("--chart-grid") } } },
      dataZoom: [{ type: "slider", xAxisIndex: 1, filterMode: "none", startValue: value[0], endValue: value[1], height: 30, bottom: 16,
        brushSelect: true, fillerColor: "rgba(47,191,113,0.18)", borderColor: v("--chart-grid"),
        handleStyle: { color: v("--ok"), borderColor: v("--ok") }, textStyle: { color: v("--chart-text") },
        labelFormatter: (h: number) => label(h) }],
      series: [{ type: "line", showSymbol: false, data: data.hours.map((h, i) => [h, data.rms[i]]),
        lineStyle: { color: v("--accent"), width: 1.4 }, markArea: band(value) }],
    });
    c.on("datazoom", () => {
      const dz = (c.getOption().dataZoom as { startValue: number; endValue: number }[])[0];
      onChange([dz.startValue, dz.endValue]);
      c.setOption({ series: [{ markArea: band([dz.startValue, dz.endValue]) }] });
    });
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); c.dispose(); };
    // Redraw only when the data changes; the slider owns the window.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, value === null]);

  const count = useMemo(() => (data && value ? data.hours.filter((h) => h >= value[0] && h <= value[1]).length : 0), [data, value]);

  if (err) return <div className="note">Could not read the recording — {err}</div>;
  if (!data) return (
    <div>
      <p className="note" role="status">Reading the recording to draw its overall vibration level.
        A long recording can take a minute or two the first time.</p>
      <div className="skeleton" style={{ height: 300 }} />
    </div>
  );
  const enough = count >= min;
  return (
    <div>
      <p className="hint" style={{ marginTop: 0 }}>
        <b>Choose a stretch when the machine was running normally.</b> Everything inside it becomes
        "normal" to the engine — a fault in the baseline would be learned as healthy. Drag the green
        window's edges; a flat, steady stretch is what you want.
      </p>
      <div ref={el} style={{ height: 300 }} role="img" aria-label="Overall vibration over the recording, with the baseline window" />
      <div className="row" style={{ alignItems: "center", marginTop: 6 }}>
        <span className={`chip${enough ? "" : " bad"}`}>{enough ? <Check size={12} /> : <CircleAlert size={12} />}{count} readings in the baseline</span>
        <span className="hint">{enough ? `at least ${min} needed — ${min - 100 > 0 ? min - 100 : 60} to warm up, 100 to learn from` : `commissioning needs at least ${min}; widen the window`}</span>
      </div>
    </div>
  );
}

// --- commissioning -------------------------------------------------------------------

const JOB_STEPS = ["Reading the recording", "Writing the machine profile", "Measuring the noise floor",
  "Minting the brain", "Commissioning", "Checking the baseline"];

export function Commission({ job, name, rec, channel, baseline, submitting, onSubmit }: {
  job: MachineRecord | null; name: string; rec: RecordingInfo | undefined; channel: number;
  baseline: [number, number] | null; submitting: boolean; onSubmit: () => void;
}) {
  if (!job) {
    return (
      <div>
        <div className="summary">
          <div><span className="hint">Machine</span><b>{name}</b></div>
          <div><span className="hint">Data</span><b>{rec?.name}{rec && rec.channels > 1 ? ` · channel ${channel + 1}` : ""}</b></div>
          <div><span className="hint">Baseline</span><b>{baseline ? `${baseline[0].toFixed(1)} – ${baseline[1].toFixed(1)} h` : "—"}</b></div>
        </div>
        <p className="hint">Commissioning measures this machine's noise floor on the baseline, builds its brain,
          and replays the baseline through it to check that healthy running raises no alarm. It takes
          under a minute.</p>
        <button className="btn primary" disabled={submitting} onClick={onSubmit}><Gauge size={15} />Commission {name}</button>
      </div>
    );
  }
  const at = job.progress.step ?? 0;
  const r = job.report;
  return (
    <div>
      <ol className="jobsteps">
        {JOB_STEPS.map((s, i) => {
          const done = job.status === "ready" || i + 1 < at;
          const on = job.status === "commissioning" && i + 1 === at;
          const bad = job.status === "failed" && i + 1 === at;
          return (
            <li key={s} className={done ? "done" : on ? "on" : bad ? "bad" : ""}>
              {done ? <CheckCircle2 size={16} /> : on ? <Loader2 size={16} className="spinning" /> : bad ? <CircleAlert size={16} /> : <span className="dotempty" />}
              {s}
            </li>
          );
        })}
      </ol>
      {job.status === "failed" && <div className="note bad-note"><b>Not commissioned.</b> {job.error}</div>}
      {job.status === "ready" && (
        <div className="report">
          <div className="kpis3">
            <div><span className="hint">Baseline readings</span><b className="num">{r.readings}</b><span className="hint">over {r.span_h} h</span></div>
            <div><span className="hint">False alarms on the baseline</span><b className="num" style={{ color: r.false_alarms ? "var(--warning)" : "var(--ok)" }}>{r.false_alarms}</b><span className="hint">of {r.warm_readings} checked</span></div>
            {r.speed && <div><span className="hint">Speed from vibration</span><b className="num">{Math.round(r.speed.median_rpm)}</b><span className="hint">rpm, 5–95% {Math.round(r.speed.p5_rpm)}–{Math.round(r.speed.p95_rpm)}</span></div>}
            <div><span className="hint">Analysis window</span><b className="num">{r.window_samples}</b><span className="hint">samples at {(r.sample_rate_hz ?? 0) / 1000} kHz</span></div>
          </div>
          <p className={r.verdict === "ok" ? "hint" : "note"}>{r.note}</p>
          {(r.unmeasured ?? []).map((u) => <p key={u.feature} className="note">{u.reason}</p>)}
          <details>
            <summary className="hint">The noise floor measured for this machine</summary>
            <table className="floor">
              <tbody>
                {Object.entries(r.dead_zones ?? {}).map(([k, v]) => (
                  <tr key={k}><td>{k.replace(/^VIB_/, "").replace(/_/g, " ")}</td><td className="num">{v.toFixed(2)}</td></tr>
                ))}
              </tbody>
            </table>
            <span className="hint">2× the 99.9th percentile of healthy surprise per feature: below this, a change is noise.</span>
          </details>
        </div>
      )}
    </div>
  );
}


// Listen to a live MQTT sensor (HARM-S7-T08): its snapshots are recorded as
// they arrive, and the recording they grow is commissioned like any other.
function LiveListen({ cfg, toast, keep, onKeep, onStarted, onRecording }: {
  cfg: { id: string; broker: string; topic: string } | null;
  toast: Props["toast"];
  keep: boolean;
  onKeep: (v: boolean) => void;
  onStarted: (c: { id: string; broker: string; topic: string }) => void;
  onRecording: (key: string) => void;
}) {
  const [broker, setBroker] = useState(cfg?.broker ?? "localhost:1883");
  const [topic, setTopic] = useState(cfg?.topic ?? "");
  const [label, setLabel] = useState("");
  const [st, setSt] = useState<ListenStatus | null>(null);
  const [seen, setSeen] = useState(0);

  useEffect(() => {
    if (!cfg) return;
    const t = setInterval(() => api.listenStatus(cfg.id).then((s) => {
      setSt(s);
      if (s.snapshots > 0 && s.snapshots !== seen) { setSeen(s.snapshots); onRecording(s.recording); }
    }).catch(() => {}), 1000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg, seen]);

  const start = async () => {
    try {
      const s = await api.listen(broker.trim(), topic.trim(), label.trim());
      setSt(s);
      onStarted({ id: s.id, broker: s.broker, topic: s.topic });
    } catch (e) { toast((e as Error).message, true); }
  };

  return (
    <div className="live-listen">
      <div className="wz-grid">
        <div className="field"><label htmlFor="lv-broker">Broker</label>
          <input id="lv-broker" className="input" value={broker} disabled={!!cfg}
            onChange={(e) => setBroker(e.target.value)} placeholder="host:1883, or pubsub" />
          <span className="hint">An MQTT broker as host:port, or <b>pubsub</b> for Google Cloud Pub/Sub (the topic is then a topic id).</span></div>
        <div className="field"><label htmlFor="lv-topic">Topic</label>
          <input id="lv-topic" className="input" value={topic} disabled={!!cfg}
            onChange={(e) => setTopic(e.target.value)} placeholder="plant/line1/gb1/vibration" /></div>
        <div className="field wide"><label htmlFor="lv-name">Recording name</label>
          <input id="lv-name" className="input" value={label} disabled={!!cfg}
            onChange={(e) => setLabel(e.target.value)} placeholder="optional" /></div>
      </div>
      {!cfg ? (
        <button className="btn primary" disabled={!topic.trim()} onClick={start}><Radio size={14} />Start listening</button>
      ) : (
        <div className="note">
          <b>{st?.connected ? "Listening" : "Connecting…"}</b> to {cfg.topic} on {cfg.broker}
          {" · "}<b className="num">{st?.snapshots ?? 0}</b> snapshots recorded
          {st && st.rejected + st.mismatched > 0 && <> · {st.rejected + st.mismatched} refused</>}
          {st?.last_error && <div className="hint">{st.last_error}</div>}
          <div className="hint">Commissioning needs a healthy stretch of enough snapshots; they are recorded as they arrive and appear on the next step.</div>
        </div>
      )}
      <label className="check" style={{ marginTop: 10 }}>
        <input type="checkbox" checked={keep} onChange={(e) => onKeep(e.target.checked)} />
        Keep watching this sensor live after commissioning
      </label>
      <span className="hint">One JSON message per snapshot: sample_rate_hz, samples_b64 (float32) and optionally t, rpm, unit, slow — see api/adapters/mqtt.py.</span>
    </div>
  );
}
