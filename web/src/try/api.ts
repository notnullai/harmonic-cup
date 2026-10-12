// The try-mode API client (api/try_mode.py). Every path is relative, the same
// convention the console's own api.ts uses: in production the API serves
// this page too, and in development Vite forwards these paths to it.

export interface Limits {
  max_total_mb: number;
  max_files: number;
  max_snapshots: number;
  max_samples_per_snapshot: number;
  min_sample_rate_hz: number;
  max_sample_rate_hz: number;
  families: Record<string, string>;
  family_info?: Record<string, FamilyInfo>;
  default_interval_s?: number;
  min_interval_s?: number;
  max_interval_s?: number;
  rate_limit_per_hour: number;
  retention_days: number;
  consent_retention_days: number;
  max_request_mb: number;
  upload_part_bytes: number;
  max_total_samples: number;
  contact_url: string | null;
  min_snapshots: number;
  wav_snapshot_samples: number;
}

export interface FamilyParam { key: string; label: string; default: number; min: number; max: number }

export interface FamilyInfo { names_faults: boolean; parameters: FamilyParam[] }

export type TimedBy = "timestamps" | "wav" | "interval";

export interface Overview {
  hours: number[];
  at: (string | null)[];
  rms: number[];
  unit: string;
  snapshots: number;
  sample_rate_hz: number;
  default_baseline_h: [number, number];
  rpm_used: number;
  rpm_limited: boolean;
  min_baseline_snapshots: number;
  n_channels: number;
  channel: number;
  interval_s?: number;
  timed_by?: TimedBy;
}

export interface DiagnosisResponse {
  diagnosis: string;
  confidence: number;
  root_causes?: string[];
  actions?: { step: string; tool: string; duration_min: number; skill_level: string }[];
  safety?: string[];
  guidance_note?: string;
  references?: string[];
  references_label?: string;
}

export interface Diagnosis {
  provider: string;
  top_candidate: string | null;
  response: DiagnosisResponse;
}

export interface AlarmFirst {
  hours: number; before_end_h: number; at: string | null;
  into: string; before_end: string; level: string; state?: string;
}

/** What a probe read from the upload, before anything else is asked. */
export interface Probe {
  snapshots: number;
  samples_per_snapshot: number;
  n_channels: number;
  format: string;
  files: number;
  timed_by: TimedBy;
  interval_s: number | null;
  span_h: number | null;
  sample_rate_hz: number | null;
}

export interface SampleInfo {
  file_name: string; title: string; story: string; credit: string;
  family: string; sample_rate_hz: number; rpm: number; interval_s: number; unit: string;
}

export interface Chart {
  points: { h: number; d: number | null }[];
  normal: number;
  alarm: number;
  end_h: number;
  first_alarm_h: number | null;
}

export interface Report {
  headline: string;
  chart: Chart;
  first_alarm: AlarmFirst | null;
  baseline_h: [number, number];
  baseline_text: string;
  first_watch: AlarmFirst | null;
  first_warning: AlarmFirst | null;
  first_critical: AlarmFirst | null;
  false_alarms_in_baseline: number;
  warm_readings_in_baseline: number;
  evidence: string[];
  evidence_note: string | null;
  diagnosis: Diagnosis | null;
  family: string;
  family_label: string;
  names_faults?: boolean;
  assumed?: string[];
  interval_s?: number;
  interval_text?: string;
  timed_by?: TimedBy;
  rpm_used: number;
  rpm_limited: boolean;
  unit: string;
  sample_rate_hz: number;
  n_snapshots: number;
  channel: number;
  n_channels: number;
  span_h: number;
  span_text: string;
  sample: SampleInfo | null;
  caveats: string[];
  contact_url: string | null;
}

export type RunStatus = "collecting" | "configured" | "queued" | "running" | "done" | "failed";

export interface Run {
  id: string;
  status: RunStatus;
  created: number;
  consent: boolean;
  family: string;
  unit: string;
  sample_rate_hz: number;
  rpm: number | null;
  rpm_limited: boolean;
  baseline_h: [number, number] | null;
  n_snapshots: number;
  n_channels: number;
  channel: number;
  progress: {
    step?: number; of?: number; label?: string;
    stage?: "queue" | "learn" | "watch" | "write"; ahead?: number; done?: number; total?: number;
  };
  sample?: boolean;
  error: string | null;
  report?: Report;
}

/** An error the visitor can read: the server's own plain reason, or what
 *  went wrong on the way and what to do about it. */
export class TryApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(path, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json", ...init.headers } : init?.headers,
    });
  } catch {
    throw new TryApiError("Harmonic could not be reached. Check your internet connection, "
      + "then try again.", 0);
  }
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    const d = body?.detail;
    let msg = typeof d === "string" ? d
      : Array.isArray(d) ? "Some of the settings were not understood. Check them and try again."
      : r.status === 404 ? "This run no longer exists (it was deleted, or it expired). Start again with your file."
      : r.status >= 500 ? "Harmonic hit a problem on its side. Please try again in a few minutes."
      : `The request was refused (${r.status}).`;
    throw new TryApiError(msg.charAt(0).toUpperCase() + msg.slice(1), r.status);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}

export interface ConfigureBody {
  family: string; sample_rate_hz: number; rpm?: number | null; unit: string;
  format: "auto" | "wide_csv" | "wav"; has_timestamp?: boolean; channel?: number;
  params?: Record<string, number>; interval_s?: number;
}

// A file bigger than one request may carry (Cloud Run: 32 MiB) goes up in
// ordered parts; the server appends each at its byte offset.
async function uploadFile(id: string, file: File, partBytes: number,
  onPart?: (sent: number) => void): Promise<{ saved: string; bytes: number }> {
  const url = `/try/runs/${id}/uploads/${encodeURIComponent(file.name)}`;
  let last: { saved: string; bytes: number } = { saved: file.name, bytes: 0 };
  let off = 0;
  do {
    const part = file.slice(off, off + partBytes);
    last = await req(`${url}?offset=${off}`, { method: "PUT", body: part });
    off += part.size;
    onPart?.(off);
  } while (off < file.size);
  return last;
}

export const api = {
  limits: () => req<Limits>("/try/limits"),
  newRun: () => req<{ id: string }>("/try/runs", { method: "POST" }),
  uploadFile,
  configure: (id: string, body: ConfigureBody) => req<Overview>(`/try/runs/${id}/configure`, { method: "POST", body: JSON.stringify(body) }),
  // Reads the upload and says what it holds; nothing else needs to be known yet.
  probe: (id: string, format: "auto" | "wide_csv" | "wav", has_timestamp: boolean) =>
    req<Probe>(`/try/runs/${id}/configure`, {
      method: "POST", body: JSON.stringify({ probe: true, format, has_timestamp }),
    }),
  sample: (id: string) => req<SampleInfo>(`/try/runs/${id}/sample`, { method: "POST" }),
  start: (id: string, baseline_h: [number, number], consent: boolean) =>
    req<{ status: string; queue_depth: number }>(`/try/runs/${id}/start`, {
      method: "POST", body: JSON.stringify({ baseline_h, consent }),
    }),
  run: (id: string) => req<Run>(`/try/runs/${id}`),
  deleteRun: (id: string) => req<{ deleted: string }>(`/try/runs/${id}/delete`, { method: "POST" }),
};
