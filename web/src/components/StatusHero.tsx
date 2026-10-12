import { useState } from "react";
import { AlertTriangle, Clock, Gauge as GaugeIcon, History, Info, Radio, SearchX } from "lucide-react";
import type { Point, SourceInfo } from "../api";
import { STATE_ADVICE, duration, fmt, stateVar, when } from "../format";

function Gauge({ value, state }: { value: number; state: string }) {
  const r = 58, c = 2 * Math.PI * r, arc = 0.75;
  const v = Math.max(0, Math.min(1, value));
  const col = stateVar(state);
  return (
    <svg width="168" height="150" viewBox="0 0 168 150" role="img"
      aria-label={`Health index ${v.toFixed(2)} of 1`}>
      <g transform="rotate(135 84 84)">
        <circle cx="84" cy="84" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="12"
          strokeDasharray={`${c * arc} ${c}`} strokeLinecap="round" />
        {[0.25, 0.5, 0.75].map((t) => (
          <circle key={t} cx="84" cy="84" r={r} fill="none" stroke="var(--surface)" strokeWidth="13"
            strokeDasharray={`1.5 ${c}`} strokeDashoffset={-c * arc * t} />
        ))}
        <circle cx="84" cy="84" r={r} fill="none" stroke={col} strokeWidth="12"
          strokeDasharray={`${c * arc * v} ${c}`} strokeLinecap="round"
          style={{ transition: "stroke-dasharray .5s ease, stroke .3s" }} />
      </g>
      <text x="84" y="86" textAnchor="middle" fontSize="32" fontWeight="800" fill="var(--text)"
        className="num">{v.toFixed(2)}</text>
      <text x="84" y="108" textAnchor="middle" fontSize="9.5" fill="var(--muted)"
        fontWeight="650" letterSpacing=".05em">HEALTH INDEX</text>
      <text x="30" y="142" fontSize="10" fill="var(--muted)">0</text>
      <text x="134" y="142" fontSize="10" fill="var(--muted)">1</text>
    </svg>
  );
}

interface Props {
  point: Point | undefined;
  latest: Point | undefined;
  source: SourceInfo | undefined;
  live: boolean;
  /** Classes the graph rules out for this machine: class -> the part it would need. */
  ruledOut: Record<string, string>;
  /** Opens "set vibration zones" for a machine that can be commissioned again. */
  onSetIso?: () => void;
}

export default function StatusHero({ point, latest, source, live, ruledOut, onSetIso }: Props) {
  const [isoWhy, setIsoWhy] = useState(false);
  if (!point) {
    return (
      <section className="card hero" aria-busy="true">
        <div className="hero-state s-NONE"><div className="skeleton" style={{ height: 90 }} /></div>
        <div className="hero-gauge"><div className="skeleton" style={{ width: 150, height: 130 }} /></div>
        <div className="hero-kpis"><div className="skeleton" style={{ height: 90, gridColumn: "1 / -1" }} /></div>
      </section>
    );
  }
  const unit = source?.vib_unit ?? "mm/s";
  // Zones come from the standard the machine was commissioned under, judged
  // on RMS velocity: the channel itself, or an accelerometer integrated by the
  // engine.
  const std = source?.standard ?? null;
  const vel = point.vel;
  const zone = std && vel != null
    ? (["A", "B", "C"][std.bounds.slice(0, 3).findIndex((b) => vel < b)] ?? "D")
    : undefined;
  const behind = latest ? latest.h - point.h : 0;
  // The engine votes over every class its network was trained on; a class
  // this machine cannot have (cavitation on a rig with no impeller) is not
  // shown as the lead. The graph applies the same rule to the diagnosis.
  const lead = point.top.find((c) => !(c.name in ruledOut));
  const skipped = point.top.filter((c) => c.name in ruledOut && c.score >= (lead?.score ?? 0));

  return (
    <section className={`card hero s-${point.state}`} aria-live="polite">
      <div className="hero-state">
        <div className="label">{live ? "Machine state · live" : "Machine state · at the cursor"}</div>
        <div className="big">{point.state}</div>
        <div className="advice">{point.early ? "Early stage — plan the repair" : STATE_ADVICE[point.state] ?? ""}</div>
        {point.early && (
          <div className="early-note" role="note">
            <b>Recognised, not yet severe.</b> The engine is sure of the fault, but no measurement
            is far enough from normal to call it critical. It turns CRITICAL when one is, or when
            vibration reaches ISO zone D.
          </div>
        )}
        {point.unr && (
          <div className="unrec" role="note">
            <SearchX size={15} />
            <span><b>Unrecognised change.</b> Several measurements are far from this machine's normal,
              and none of the faults it was trained on matches. Inspect it: the diagnosis is drawn
              from the measurements that moved.</span>
          </div>
        )}
        <div className="when">
          <span className="chip">
            {live ? <Radio size={13} /> : <History size={13} />}
            {when(point.h, point.at)}
          </span>
          {!live && behind > 0 && <span className="chip"><Clock size={13} />{duration(behind)} before live</span>}
          {!point.warm && <span className="chip">baseline warming up</span>}
        </div>
      </div>
      <div className="hero-gauge">
        <Gauge value={point.hhi} state={point.state} />
        {point.unr && <div className="gauge-note">The index counts known faults; this change is not one of them.</div>}
      </div>
      <div className="hero-kpis">
        <div className="kpi">
          <div className="k">Overall vibration</div>
          <div className="v num">{fmt(point.vib, unit)}<small>{unit}</small></div>
          <div className="sub">
            {std && zone
              ? <>{std.short} · zone <b>{zone}</b>{unit !== "mm/s" && vel != null && <> · {fmt(vel, "mm/s")} mm/s</>}</>
              : <>{vel != null && unit !== "mm/s" && <>{fmt(vel, "mm/s")} mm/s · </>}
                {onSetIso ? "no ISO zones"
                  : <button className="linkish" onClick={() => setIsoWhy(!isoWhy)} aria-expanded={isoWhy}>
                      no ISO zones <Info size={11} style={{ verticalAlign: -1 }} /></button>}</>}
          </div>
        </div>
        <div className="kpi">
          <div className="k">Remaining useful life</div>
          <div className="v num">{point.rul >= 720 ? "> 30" : (point.rul / 24).toFixed(1)}<small>days</small></div>
          <div className="sub">heuristic, from the index trend</div>
        </div>
        <div className="kpi">
          <div className="k">Duty regime</div>
          <div className="v" style={{ fontSize: 18 }}><GaugeIcon size={15} style={{ marginRight: 6, verticalAlign: -1 }} />{point.regime}</div>
          <div className="sub">baseline learned per regime</div>
        </div>
        <div className="kpi">
          <div className="k">Leading class</div>
          <div className="v" style={{ fontSize: 16, textTransform: "capitalize" }}>
            {point.unr ? "Unrecognised" : lead ? lead.name.replace(/_/g, " ") : "—"}
          </div>
          <div className="sub" title={skipped.map((c) => `${c.name.replace(/_/g, " ")}: needs ${ruledOut[c.name]}`).join("; ") || undefined}>
            engine vote · {skipped.length ? `${skipped.map((c) => c.name.replace(/_/g, " ")).join(", ")} ruled out` : "the graph decides"}
          </div>
        </div>
      </div>
      {/* A machine you commissioned can be given zones: say so, with the action.
          A built-in one cannot: explain only when asked, never as a warning
          with nothing to do. */}
      {!std && onSetIso && (
        <div className="iso-missing" role="note">
          <AlertTriangle size={15} />
          <span><b>No ISO vibration zones.</b> Absolute limits are not checked: a severe level is
            judged only against what this machine learned.</span>
          <button className="btn sm" onClick={onSetIso}>Set zones</button>
        </div>
      )}
      {!std && !onSetIso && isoWhy && (
        <div className="iso-info" role="note">
          <Info size={14} />
          <span>{source?.iso_note || "This built-in machine runs without ISO vibration zones, so absolute limits are not checked; its alarms come from what the engine learned. Zones are chosen when you commission a machine of your own."}</span>
        </div>
      )}
    </section>
  );
}
