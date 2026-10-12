import type { Point } from "../api";
import { fmt } from "../format";

type Sensor = { key: string; label: string; unit: string | null };

// Labels for the readings the built-in sources report. A machine that reports
// others (a fan's airflow) has them shown under their own key.
const SENSORS: Sensor[] = [
  { key: "vib_rms", label: "Vibration RMS", unit: null },
  { key: "bearing_temp_c", label: "Bearing temperature", unit: "°C" },
  { key: "motor_temp_c", label: "Motor temperature", unit: "°C" },
  { key: "discharge_bar", label: "Discharge pressure", unit: "bar" },
  { key: "suction_bar", label: "Suction pressure", unit: "bar" },
  { key: "flow_m3h", label: "Flow", unit: "m³/h" },
  { key: "current_a", label: "Motor current", unit: "A" },
  { key: "oil_temp_c", label: "Oil temperature", unit: "°C" },
  { key: "winding_temp_c", label: "Winding temperature", unit: "°C" },
  { key: "airflow_m3s", label: "Airflow", unit: "m³/s" },
  { key: "rpm", label: "Shaft speed", unit: "rpm" },
];

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return <svg aria-hidden="true" />;
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const d = values.map((v, i) =>
    `${i ? "L" : "M"}${((i / (values.length - 1)) * 100).toFixed(2)},${(28 - ((v - lo) / span) * 26).toFixed(2)}`).join("");
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true">
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

interface Props { point: Point | undefined; window: Point[]; vibUnit: string }

export default function SensorGrid({ point, window, vibUnit }: Props) {
  // Velocity is shown with the ISO zone in the machine state; not twice.
  const known = new Set([...SENSORS.map((s) => s.key), "vib_rms_mm_s"]);
  const extra: Sensor[] = Object.keys(point?.sensors ?? {}).filter((k) => !known.has(k))
    .map((k) => ({ key: k, label: k.replace(/_/g, " "), unit: "" }));
  const shown = [...SENSORS, ...extra];
  const reported = shown.filter((s) => s.key === "vib_rms" || !point?.sensors || point.sensors[s.key] !== undefined);
  const isOff = (s: Sensor) => s.key !== "vib_rms" && point?.sensors?.[s.key] === null;
  const measured = reported.filter((s) => !isOff(s));
  const missing = reported.filter(isOff);
  // Sparklines over the visible window, thinned to ~80 points.
  const step = Math.max(1, Math.floor(window.length / 80));
  const thin = window.filter((_, i) => i % step === 0);
  return (
    <section className="card">
      <div className="card-h"><h2>Sensors</h2><span className="grow" />
        <span className="hint">values at the inspected moment · trend over the visible window</span></div>
      <div className="card-b">
        <div className="sensors">
          {/* A sensor the source does not report is not shown. One it reports
              as null (a recording saying what it lacks) is named once, below,
              rather than as a tile each. */}
          {measured.map((s) => {
            const unit = s.unit ?? vibUnit;
            const v = point?.sensors?.[s.key] ?? null;
            const off = v === null || v === undefined;
            return (
              <div key={s.key} className={`sensor${off ? " off" : ""}`}>
                <div className="k"><span>{s.label}</span>{!off && <span>{unit}</span>}</div>
                {off ? <div className="v">Not measured on this machine</div>
                  : <div className="v num">{fmt(v, unit)}<small>{unit}</small></div>}
                {!off && <Spark values={thin.map((q) => q.sensors?.[s.key]).filter(
                  (x): x is number => typeof x === "number")} />}
              </div>
            );
          })}
        </div>
        {missing.length > 0 && (
          <p className="hint" style={{ margin: "10px 0 0" }}>
            Not measured on this machine: {missing.map((s) => s.label.toLowerCase()).join(", ")}.</p>
        )}
      </div>
    </section>
  );
}
