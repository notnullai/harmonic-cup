import type { State } from "./api";

export const STATE_ADVICE: Record<string, string> = {
  OK: "Operating normally",
  WATCH: "A trend worth watching — not a call-out",
  WARNING: "Inspect within the shift",
  CRITICAL: "Plan a shutdown",
  FAILED: "The engine reports a fault in itself",
};

export const STATES: State[] = ["OK", "WATCH", "WARNING", "CRITICAL"];

/** A CSS custom property value for a state colour. */
export function stateVar(state: string | undefined): string {
  switch (state) {
    case "OK": return "var(--ok)";
    case "WATCH": return "var(--watch)";
    case "WARNING": return "var(--warning)";
    case "CRITICAL": return "var(--critical)";
    case "FAILED": return "var(--failed)";
    default: return "var(--muted)";
  }
}

/** Machine hours as the reader knows them: the recording's own clock for a
 *  recording, machine time from the start for the simulator. */
export function when(h: number, at: string | null | undefined, short = false): string {
  if (at) return short ? at.slice(5) : at;
  const hh = Math.floor(h);
  let mm = Math.round((h - hh) * 60);
  let H = hh;
  if (mm === 60) { H += 1; mm = 0; }
  return `T+${H}:${String(mm).padStart(2, "0")}`;
}

export function duration(hours: number): string {
  const a = Math.abs(hours);
  if (a < 1 / 60) return "now";
  if (a < 1) return `${Math.round(a * 60)} min`;
  if (a < 48) return `${a.toFixed(a < 10 ? 1 : 0)} h`;
  return `${(a / 24).toFixed(1)} days`;
}

export function feature(kind: string): string {
  return kind.replace(/^HARMONIC_FEAT_/, "").replace(/^VIB_/, "").replace(/_/g, " ");
}

export function failureMode(id: string | undefined): string {
  if (!id) return "";
  // "fm.cavitation", or a family's own "gbx.fm.tooth_breakage".
  const s = id.replace(/^(?:[a-z]+\.)?fm\./, "").replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function fmt(v: number | null | undefined, unit: string): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (unit === "g") return v.toFixed(3);
  if (unit === "rpm") return v.toFixed(0);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(2);
}

/** Last element whose key is at or before `x`; the array is sorted by key. */
export function atOrBefore<T>(items: T[], x: number, key: (t: T) => number): T | undefined {
  let lo = 0, hi = items.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (key(items[mid]) <= x) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return found >= 0 ? items[found] : undefined;
}
