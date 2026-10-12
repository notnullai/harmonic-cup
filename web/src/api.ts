// The API client. Every path is relative: in production the API serves this
// app, and in development Vite forwards the same paths to it.

export type State = "OK" | "WATCH" | "WARNING" | "CRITICAL" | "FAILED";

export interface Point {
  tick: number;
  h: number; // machine hours since the source started
  at: string | null; // the recording's own timestamp, for a recording
  state: State;
  hhi: number;
  vib: number; // the channel's RMS, in its own unit
  vel: number | null; // RMS velocity in mm/s, what ISO zones are judged on
  rul: number;
  regime: string;
  warm: boolean;
  sensors: Record<string, number | null>;
  top: { name: string; score: number }[];
  /** The state stands on an unrecognised change: no trained class, no ISO zone. */
  unr?: boolean;
  /** A fault recognised early: sure, not yet severe, so held at WARNING. */
  early?: boolean;
  inj: Record<string, number>;
}

export interface Action {
  step: string;
  tool: string;
  duration_min: number;
  skill_level: string;
}

export interface Diagnosis {
  diagnosis: string;
  confidence: number;
  root_causes?: string[];
  actions?: Action[];
  safety?: string[];
  references?: string[];
  rul_hours?: number;
  failure_mode_id?: string;
}

export interface Feature {
  kind: string;
  sigma?: number;
  direction?: string;
  value?: number;
}

export interface HEvent {
  id: string;
  tick: number;
  state: State;
  hhi: number;
  rul_hours: number;
  regime: string;
  machine_hours: number;
  recorded_at: string | null;
  vib_rms: number | null;
  vib_unit: string;
  equipment_id: string;
  top_classes: { name: string; score: number }[];
  top_surprised_features: Feature[];
  diagnosis: Diagnosis | null;
  diagnosis_status: "pending" | "queued" | "running" | "ok" | "fallback" | "failed" | string;
  /** One excursion from OK back to OK; the diagnosis is made once per episode and fault. */
  episode?: string;
  /** Set when this event carries the diagnosis made for an earlier alarm. */
  diagnosis_of?: string | null;
  /** The measurement point it was raised at; "" for a one-point machine. */
  point?: string;
  /** Raised on an unrecognised change: no trained fault class matched. */
  unrecognised?: boolean;
  /** Raised on a fault recognised early: not yet severe. */
  early?: boolean;
  diagnosis_provider: string | null;
  feedback?: FeedbackRecord | null;
  /** The job this alarm opened, escalated (`update`) or was added to (`repeat`). */
  work_order?: { id: string; delivery: string; update: boolean; repeat?: boolean } | null;
}

export interface ListenStatus {
  id: string; broker: string; topic: string; connected: boolean; received: number;
  rejected: number; mismatched: number; snapshots: number; last_error: string | null;
  recording: string; sample_rate_hz: number | null; samples: number | null;
}

export interface FeedbackRecord {
  verdict: "confirmed" | "wrong";
  actual: string | null; // a class name, "healthy" for a false alarm, or "other"
  note: string;
  at: number;
}

export interface FleetMachine {
  /** The machine's vibration standard; absent when no ISO zones are set. */
  standard?: { short: string } | null;
  /** The machine this row belongs to, and its measurement point ("" for one point). */
  machine?: string; point?: string;
  key: string; label: string; kind: string; available: boolean; reason: string | null;
  running: boolean; current: boolean; stored: { events: number; feedback: number };
  name?: string; real?: boolean; equipment_id?: string; family?: string; vib_unit?: string;
  state?: State | null; hhi?: number | null; rul_hours?: number | null; vib?: number | null;
  machine_hours?: number | null; events?: number;
  last_event?: {
    id: string; state: State; machine_hours: number; top: { name: string; score: number }[];
    failure_mode_id: string | null; diagnosis: string | null; diagnosis_status: string;
    feedback: FeedbackRecord | null;
  } | null;
}

export interface SourceInfo {
  key: string;
  kind: "simulated" | "recorded" | "live";
  real: boolean;
  name: string;
  detail: string;
  supports_injection: boolean;
  vib_unit: string;
  equipment_id: string;
  badge?: string;
  outcome?: string | null;
  // The vibration standard the machine is judged by; null = no zones.
  standard?: { key: string; label: string; short: string; bounds: number[] } | null;
  /** Why a built-in machine has no zones (it cannot be given any). */
  iso_note?: string;
  family?: string;
  fault_modes?: string[]; // what can be injected, for a simulated source
  live?: { broker: string; topic: string; connected: boolean; received: number;
           rejected: number; refused: number; dropped: number; waiting: number;
           last_error: string | null };
  progress?: { snapshot: number; of: number; finished: boolean };
}

export interface Machine {
  state: State;
  iso_state: string;
  hhi: number;
  rul_hours: number;
  regime: string;
  warm: boolean;
  vib_rms_mm_s: number | null;
  vib_rms: number;
  vib_unit: string;
  recorded_at: string | null;
  tick: number;
  machine_hours: number;
  injected: Record<string, number>;
  /** A simulator's injected faults: where each grows to, and whether growth is paused. */
  fault_growth?: FaultGrowth | null;
}

export interface FaultGrowth {
  targets: Record<string, number>;
  held: boolean;
  /** Machine seconds per wall-clock second in this demo. */
  speedup: number | null;
  /** The state the fault will pause at, until it gets there. */
  pause_at: string | null;
  /** The state it paused at. */
  paused_at: string | null;
}

/** A work order as the maintenance system received it (api/workorders.py). */
export interface WorkOrder {
  id: string;
  update: boolean;
  created: string;
  source: string;
  machine: string;
  equipment_id: string;
  state: State;
  priority: number;
  title: string;
  failure_mode_id: string;
  confidence: number | null;
  summary: string | null;
  actions: { step: string; tool?: string; duration_min?: number; skill_level?: string }[];
  safety: string[];
  references: string[];
  remaining_life_hours: number | null;
  advisory: string;
  status: "open" | "closed";
  delivery: "pending" | "sent" | "failed" | "simulated" | string;
  attempts: number;
  last_error: string | null;
  event_ids: string[];
  alarms: number;
  opened_at: number;
  updated_at: number;
}

/** What a feature's movement means on the selected machine. */
export interface FeatureMeaning {
  label: string | null;
  unit: string;
  scale: "ratio" | "linear";
  up?: { description: string; points_to: { id: string; confidence: number }[] };
  down?: { description: string; points_to: { id: string; confidence: number }[] };
}

export interface Health {
  ok: boolean;
  engine: {
    available: boolean;
    reason: string | null;
    brain?: string;
    feature_count?: number;
    class_count?: number;
    commissioned?: boolean;
    nominal_rpm?: number;
    source?: SourceInfo;
    period_s?: number;
    /** Classes the graph rules out for this machine: class -> the part it would need. */
    ruled_out_classes?: Record<string, string>;
  };
  genai: { provider?: string; processed?: number; failed?: number; running?: boolean };
  /** The knowledge graph's size, whatever backend serves it (NetworkX or Neo4j). */
  kg?: { nodes: number; relations: number; backend: string | null; empty: boolean };
  machine: Machine | null;
}

export interface KGNodeRef {
  failure_mode_id: string;
  description: string;
  severity: number;
  score: number;
  symptoms: {
    symptom_id: string;
    description: string;
    feature_kind: string;
    direction: string;
    confidence: number;
  }[];
  causes: { cause_id: string; description: string; likelihood: number }[];
  actions: { action_id: string; description: string; priority: number }[];
}

export interface KGSubgraph {
  equipment: { id: string };
  candidates: KGNodeRef[];
  agreement: string;
  ruled_out?: string[];
}

export interface Sources {
  current: string | null;
  sources: Record<string, { label: string; available: boolean; reason?: string | null; kind?: string }>;
}

export interface Bearing { pitch_mm: number; ball_mm: number; n_balls: number; contact_angle_deg: number }

export interface MachineRecord {
  id: string;
  name: string;
  equipment_id: string;
  nominal_rpm: number;
  vib_unit: string;
  iso_class: string;
  bearing: Bearing;
  components: string[];
  recording: string;
  channel: number;
  baseline_h: [number, number];
  status: "draft" | "commissioning" | "ready" | "failed";
  progress: { step?: number; of?: number; label?: string };
  error: string | null;
  report: {
    readings?: number; span_h?: number; window_samples?: number; sample_rate_hz?: number;
    dead_zones?: Record<string, number>; warm_readings?: number; false_alarms?: number;
    verdict?: string; note?: string; speed_source?: string;
    speed?: { median_rpm: number; p5_rpm: number; p95_rpm: number; stable: boolean } | null;
    /** Sideband features this window cannot tell from their carrier: the engine reads them as 0. */
    unmeasured?: { feature: string; name: string; reason: string }[];
  };
  live?: { broker: string; topic: string } | null;
  /** Earlier baselines, oldest first: the machine re-learned normal after maintenance. */
  baselines?: { n: number; until: number; recording: string; baseline_h: [number, number]; note: string;
    report: { false_alarms?: number; readings?: number } }[];
  /** Why the last re-baseline did not take; the machine kept its brain. */
  rebaseline_error?: string | null;
}

/** A moment the machine's brain changed, marked on the timeline. */
export interface Mark { tick: number; h: number; at: string | null; label: string }

export interface MachineCatalog {
  machines: MachineRecord[];
  iso_classes: Record<string, string>;
  bearings: Record<string, Bearing & { label: string; source?: string }>;
  components: string[];
  min_baseline_readings: number;
  families?: Record<string, FamilyInfo>;
}

export interface FamilyInfo {
  label: string;
  available: boolean; // its trained network is built
  wizard: {
    parameters: { key: string; label: string; default: number; min: number; max: number }[];
    unit?: string;
    components?: string[];
  };
}

export interface RecordingInfo {
  key: string; name: string; provenance: string; unit: string; sample_rate_hz: number;
  channels: number; snapshots: number; span_h: number; samples: number;
  has_speed: boolean;
}

export interface MachineSpec {
  name: string; nominal_rpm: number; vib_unit: string; iso_class: string; bearing: Bearing;
  components: string[]; recording: string; channel: number; baseline_h: [number, number];
  speed: "constant" | "measured" | "estimated";
  family: string;
  params: Record<string, number>;
  live?: { broker: string; topic: string };
  /** Measurement points, each a channel with its own engine; one point when absent. */
  points?: { name: string; channel: number }[];
}

export interface ChatAnswer {
  question: string;
  answer: string;
  references: string[];
  grounded: boolean;
  provider: string;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) =>
  call<T>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

export const api = {
  health: () => call<Health>("/health"),
  history: (sinceTick: number) =>
    call<{ source: string; points: Point[]; marks?: Mark[] }>(`/history?since_tick=${sinceTick}`),
  events: () => call<{ events: HEvent[] }>("/events?limit=200"),
  kg: (id: string) => call<KGSubgraph>(`/kg/${encodeURIComponent(id)}`),
  sources: () => call<Sources>("/sources"),
  inject: (mode: string, severity: number, ramp_minutes: number, pause_at: string | null = null) =>
    post<{ injected: string }>("/inject", { mode, severity, ramp_minutes, pause_at }),
  clear: () => post<{ cleared: boolean }>("/inject/clear", {}),
  hold: (hold: boolean) => post<FaultGrowth>("/inject/hold", { hold }),
  evidence: () => call<{ equipment_id: string; family: string; features: Record<string, FeatureMeaning> }>("/evidence"),
  workOrders: () => call<{ webhook: string | null; work_orders: WorkOrder[] }>("/workorders?limit=100"),
  closeWorkOrder: (id: string) => post<{ closed: string }>(`/workorders/${encodeURIComponent(id)}/close`, {}),
  listen: (broker: string, topic: string, name: string) =>
    post<ListenStatus>("/live/listen", { broker, topic, name }),
  listenStatus: (id: string) => call<ListenStatus>(`/live/listen/${encodeURIComponent(id)}`),
  stopListen: (id: string) =>
    call<{ stopped: string }>(`/live/listen/${encodeURIComponent(id)}`, { method: "DELETE" }),
  select: (source: string) => post<{ selected: string }>("/select", { source }),
  fleet: () => call<{ current: string; machines: FleetMachine[] }>("/fleet"),
  feedback: (id: string, verdict: "confirmed" | "wrong", actual: string | null, note: string) =>
    post<{ feedback: FeedbackRecord }>(`/events/${encodeURIComponent(id)}/feedback`, { verdict, actual, note }),
  reset: (source?: string) =>
    post<{ reset: boolean; warm_after_ticks: number }>("/reset", source ? { source } : {}),
  chat: (question: string) => post<ChatAnswer>("/chat", { question }),
  machines: () => call<MachineCatalog>("/machines"),
  machine: (id: string) => call<MachineRecord>(`/machines/${encodeURIComponent(id)}`),
  rebaseline: (id: string, body: { baseline_h: [number, number]; recording?: string; channel?: number; note?: string; iso_class?: string }) =>
    post<MachineRecord>(`/machines/${encodeURIComponent(id)}/rebaseline`, body),
  createMachine: (spec: MachineSpec) => post<MachineRecord>("/machines", spec),
  deleteMachine: (id: string) =>
    call<{ deleted: string }>(`/machines/${encodeURIComponent(id)}`, { method: "DELETE" }),
  recordings: () => call<{ recordings: RecordingInfo[] }>("/recordings"),
  overview: (key: string, channel: number) =>
    call<{ hours: number[]; at: (string | null)[]; rms: number[]; unit: string }>(
      `/recordings/${encodeURIComponent(key)}/overview?channel=${channel}`),
  newUpload: (meta: { name: string; sample_rate_hz: number; interval_s: number; unit: string }) =>
    post<{ id: string }>("/uploads", meta),
  uploadManual: (machine: string, file: File) =>
    call<{ doc_id: string; sections: number }>(
      `/machines/${encodeURIComponent(machine)}/manual/${encodeURIComponent(file.name)}`,
      { method: "PUT", body: file }),
  uploadFile: (id: string, file: File) =>
    call<{ saved: string }>(`/uploads/${encodeURIComponent(id)}/${encodeURIComponent(file.name)}`,
      { method: "PUT", body: file }),
  reportUrl: (fromH?: number, toH?: number) => {
    const q = new URLSearchParams();
    if (fromH !== undefined) q.set("from_h", fromH.toFixed(4));
    if (toH !== undefined) q.set("to_h", toH.toFixed(4));
    const s = q.toString();
    return `/report.pdf${s ? `?${s}` : ""}`;
  },
};
