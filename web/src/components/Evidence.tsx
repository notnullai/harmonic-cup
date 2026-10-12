import { AlertTriangle, CheckCircle2, Info } from "lucide-react";
import type { Feature, FeatureMeaning } from "../api";
import { failureMode, feature } from "../format";

interface Props {
  features: Feature[];
  /** The selected machine's features: plain name, how a change reads, and
   *  from the knowledge graph what each is typical of. */
  meanings: Record<string, FeatureMeaning>;
  /** The diagnosed failure mode, when there is one. */
  diagnosed?: string;
  unrecognised?: boolean;
}

const UNITS: Record<string, string> = { degC: "°C", "m3/s": "m³/s", "m3/h": "m³/h", "-": "" };

/** "3.2× normal" for an amplitude (the engine's surprise is a log2 ratio),
 *  "+12 °C" for a process channel, "+2.4" for a dimensionless one. */
function change(sigma: number | undefined, m: FeatureMeaning | undefined): { big: string; small: string } | null {
  if (sigma === undefined || !m) return null;
  if (m.scale === "ratio") {
    const r = 2 ** sigma;
    if (r >= 1) return { big: `${r < 10 ? r.toFixed(1) : Math.round(r)}×`, small: "normal" };
    return { big: `${Math.round(r * 100)}%`, small: "of normal" };
  }
  const u = UNITS[m.unit] ?? m.unit;
  const v = Math.abs(sigma);
  return { big: `${sigma >= 0 ? "+" : "−"}${v < 10 ? v.toFixed(1) : Math.round(v)}${u ? ` ${u}` : ""}`,
           small: sigma >= 0 ? "above normal" : "below normal" };
}

// How loud a sign is, for its colour: an amplitude doubled, or quadrupled.
// A process channel's change is in its own unit, with no common scale.
const strength = (sigma: number | undefined, m: FeatureMeaning | undefined) =>
  sigma === undefined || m?.scale !== "ratio" ? "clear"
    : Math.abs(sigma) >= 2 ? "strong" : Math.abs(sigma) >= 1 ? "clear" : "mild";

export default function Evidence({ features, meanings, diagnosed, unrecognised }: Props) {
  const items = features.slice(0, 4).map((f) => {
    const m = meanings[f.kind];
    const sym = f.direction === "down" ? m?.down : m?.up ?? m?.down;
    const points = (sym?.points_to ?? []).slice(0, 2);
    return {
      f, m, sym, points,
      label: m?.label ?? sym?.description ?? feature(f.kind),
      supports: !!diagnosed && (sym?.points_to ?? []).some((p) => p.id === diagnosed),
      delta: change(f.sigma, m),
    };
  });
  if (items.length === 0) return null;
  const agreeing = items.filter((i) => i.supports).length;
  // What the signs point to instead, most often first.
  const tally = new Map<string, number>();
  items.forEach((i) => i.points.slice(0, 1).forEach((p) => tally.set(p.id, (tally.get(p.id) ?? 0) + 1)));
  const others = [...tally.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id).slice(0, 2);

  return (
    <div className="ev" aria-label="Evidence">
      <div className="ev-h">
        <b>Why</b>
        <span className="hint">
          {unrecognised ? "These moved, but no fault this machine was trained on matches them"
            : diagnosed && agreeing ? `${agreeing} of ${items.length} sign${items.length > 1 ? "s" : ""} point to ${failureMode(diagnosed).toLowerCase()}`
            : "What moved away from this machine's normal"}
        </span>
      </div>
      {diagnosed && !unrecognised && agreeing === 0 && others.length > 0 && (
        <div className="ev-warn" role="note">
          <AlertTriangle size={14} />
          <span>None of these signs is typical of {failureMode(diagnosed).toLowerCase()}: the diagnosis rests on the
            pattern the engine learned. The signs point to <b>{others.map((o) => failureMode(o).toLowerCase()).join(" or ")}</b> —
            check for {others.length > 1 ? "these" : "it"} too.</span>
        </div>
      )}
      <ul className="ev-list">
        {items.map(({ f, m, sym, points, label, supports, delta }) => (
          <li key={f.kind} className={supports ? "sup" : ""}>
            {delta ? (
              <span className={`ev-mag ${strength(f.sigma, m)}`} title="Compared with what this machine normally shows at this load">
                <b className="num">{delta.big}</b><small>{delta.small}</small>
              </span>
            ) : <span className="ev-mag"><b>{f.direction === "down" ? "↓" : "↑"}</b></span>}
            <div className="ev-body">
              <div className="ev-name" title={[sym?.description, f.kind.replace(/^HARMONIC_FEAT_/, "")].filter(Boolean).join(" · ")}>
                {label}</div>
              <div className="ev-sub">
                {points.length
                  ? <>Typical of {points.map((p, i) => (
                      <span key={p.id}>{i > 0 && " or "}<b className={p.id === diagnosed ? "hit" : ""}>{failureMode(p.id).toLowerCase()}</b></span>
                    ))}</>
                  : "Not linked to a known fault on this machine"}
              </div>
            </div>
            {supports && <CheckCircle2 size={15} className="ev-ok" aria-label="Supports the diagnosis" />}
          </li>
        ))}
      </ul>
      <div className="hint ev-foot"><Info size={11} style={{ verticalAlign: -1 }} /> Each sign is a measurement compared
        with this machine's own normal at the same load; what it is typical of comes from the knowledge graph.</div>
    </div>
  );
}
