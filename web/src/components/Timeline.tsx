import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart, ScatterChart } from "echarts/charts";
import {
  AxisPointerComponent,
  DataZoomComponent,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { HEvent, Mark, Point } from "../api";
import { atOrBefore, failureMode, fmt, when } from "../format";

echarts.use([LineChart, ScatterChart, GridComponent, TooltipComponent, DataZoomComponent,
  MarkAreaComponent, MarkLineComponent, AxisPointerComponent, CanvasRenderer]);

export interface View { start: number; end: number }

interface Props {
  points: Point[];
  events: HEvent[];
  /** Where the brain changed: a re-baseline after maintenance. */
  marks: Mark[];
  unit: string;
  live: boolean;
  cursorH: number | null;
  windowH: number;
  theme: string;
  onCursor: (h: number) => void;
  onLive: (live: boolean) => void;
  onView: (v: View) => void;
  onEvent: (e: HEvent) => void;
}

function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  return {
    text: v("--chart-text"), grid: v("--chart-grid"), accent: v("--accent"), muted: v("--muted"),
    surface: v("--surface-3"), border: v("--border-strong"), fg: v("--text"),
    OK: v("--ok"), WATCH: v("--watch"), WARNING: v("--warning"), CRITICAL: v("--critical"),
    FAILED: v("--failed"),
  } as Record<string, string>;
}

const alpha = (hex: string, a: number) => {
  const m = hex.replace("#", "");
  const n = parseInt(m.length === 3 ? m.split("").map((c) => c + c).join("") : m.slice(0, 6), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

/** Contiguous runs of one non-OK state, as mark-area pairs. */
function bands(points: Point[], t: Record<string, string>) {
  const out: [object, object][] = [];
  let start = 0;
  for (let i = 1; i <= points.length; i++) {
    if (i === points.length || points[i].state !== points[start].state) {
      const st = points[start].state;
      if (st !== "OK" && t[st]) {
        out.push([
          { xAxis: points[start].h, itemStyle: { color: alpha(t[st], 0.12) } },
          { xAxis: points[Math.min(i, points.length - 1)].h },
        ]);
      }
      start = i;
    }
  }
  return out;
}

export default function Timeline(p: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const programmatic = useRef(false);
  // The window on screen, in machine hours. Written into every redraw: left
  // out, each refresh reset the zoom to the full range under the user.
  const range = useRef<[number, number] | null>(null);
  const props = useRef(p);
  props.current = p;

  // -- create once
  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: "canvas" });
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);

    // Any zoom or pan the user makes leaves live mode, and reports the view;
    // dragging back to the newest reading resumes it.
    c.on("datazoom", () => {
      if (programmatic.current) return;
      const dz = (c.getOption().dataZoom as { startValue: number; endValue: number }[])[0];
      // A plain click also emits a (throttled, late) datazoom with the window
      // unchanged; treating it as a pan put the view back on live and threw
      // away the cursor the click had just placed.
      const prev = range.current;
      if (prev && Math.abs(prev[0] - dz.startValue) < 1e-6 && Math.abs(prev[1] - dz.endValue) < 1e-6) return;
      const pts = props.current.points;
      const last = pts.length ? pts[pts.length - 1].h : 0;
      range.current = [dz.startValue, dz.endValue];
      props.current.onView({ start: dz.startValue, end: dz.endValue });
      const span = Math.max(dz.endValue - dz.startValue, 1e-6);
      props.current.onLive(last - dz.endValue < span * 0.02);
    });

    // A click (not a drag) places the cursor. Listened for on the container
    // itself: the pan handler owns ECharts' own mouse events while it is
    // active. A click on an event pin opens that event instead — the pin's
    // click arrives after the pointerup, so the cursor waits a tick and is
    // cancelled if a pin was hit.
    const host = el.current;
    let down: [number, number] | null = null;
    let pinHit = false;
    const at = (e: PointerEvent): [number, number] => {
      const r = host.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const onDown = (e: PointerEvent) => { down = at(e); pinHit = false; };
    const onUp = (e: PointerEvent) => {
      if (!down) return;
      const [x, y] = at(e);
      const moved = Math.hypot(x - down[0], y - down[1]);
      down = null;
      if (moved > 4) return;
      if (!c.containPixel({ gridIndex: 0 }, [x, y]) && !c.containPixel({ gridIndex: 1 }, [x, y])) return;
      // Converted against a grid, which takes the [x, y] pair; an axis finder
      // takes a single number and threw here.
      const [h] = c.convertFromPixel({ gridIndex: 0 }, [x, y]) as number[];
      window.setTimeout(() => { if (!pinHit) props.current.onCursor(h); }, 0);
    };
    host.addEventListener("pointerdown", onDown, true);
    host.addEventListener("pointerup", onUp, true);
    c.on("click", { seriesName: "Events" }, (params) => {
      pinHit = true;
      const id = (params.data as { id?: string }).id;
      const ev = props.current.events.find((e) => e.id === id);
      if (ev) props.current.onEvent(ev);
    });

    return () => {
      host.removeEventListener("pointerdown", onDown, true);
      host.removeEventListener("pointerup", onUp, true);
      ro.disconnect(); c.dispose(); chart.current = null;
    };
  }, []);

  // -- draw
  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    const t = tokens();
    const pts = p.points;
    const first = pts.length ? pts[0].h : 0;
    const last = pts.length ? pts[pts.length - 1].h : 1;
    if (p.live || !range.current) range.current = [Math.max(first, last - p.windowH), last];
    const [zs, ze] = range.current;

    const markLines: object[] = [0.25, 0.5, 0.75].map((y, i) => ({
      yAxis: y,
      lineStyle: { color: [t.WATCH, t.WARNING, t.CRITICAL][i], type: "dashed", width: 1, opacity: 0.55 },
      label: { show: false },
    }));
    for (const m of p.marks) {
      markLines.push({
        xAxis: m.h,
        lineStyle: { color: t.muted, type: "dashed", width: 1.5 },
        label: { show: true, position: "end", color: t.muted, fontSize: 10, formatter: m.label },
      });
    }
    if (p.cursorH !== null) {
      const at = atOrBefore(pts, p.cursorH, (q) => q.h);
      markLines.push({
        xAxis: p.cursorH,
        lineStyle: { color: t.accent, type: "solid", width: 2 },
        label: {
          show: true, position: "end", color: t.fg, backgroundColor: t.surface,
          borderColor: t.accent, borderWidth: 1, padding: [4, 7], borderRadius: 6,
          formatter: when(p.cursorH, at?.at ?? null, true), fontWeight: 600,
        },
      });
    }

    c.setOption({
      animation: false,
      // Two panels on one clock: the health index and its events on top,
      // overall vibration underneath. Sharing one axis made a calm machine's
      // vibration line sit among the alarm thresholds and read as an alarm.
      grid: [
        { left: 48, right: 20, top: 34, height: "50%" },
        { left: 48, right: 20, top: "68%", bottom: 62 },
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: {
        trigger: "axis",
        backgroundColor: t.surface, borderColor: t.border, textStyle: { color: t.fg, fontSize: 12 },
        axisPointer: { type: "line", lineStyle: { color: t.muted, type: "dashed" } },
        formatter: (items: { axisValue: number }[]) => {
          const h = items?.[0]?.axisValue;
          if (h === undefined) return "";
          const q = atOrBefore(props.current.points, h, (x) => x.h);
          if (!q) return "";
          const col = t[q.state] ?? t.muted;
          return `<div style="font-weight:650;margin-bottom:4px">${when(q.h, q.at)}</div>
            <div><span style="color:${col};font-weight:700">${q.state}</span> · health index <b>${q.hhi.toFixed(2)}</b></div>
            <div>vibration <b>${fmt(q.vib, props.current.unit)}</b> ${props.current.unit}</div>
            <div style="color:${t.muted};margin-top:4px">click to inspect this moment</div>`;
        },
      },
      xAxis: [0, 1].map((gi) => ({
        type: "value", gridIndex: gi, min: first, max: last,
        axisLine: { lineStyle: { color: t.grid } }, splitLine: { show: false },
        axisTick: { show: gi === 1 },
        axisLabel: {
          show: gi === 1, color: t.text, hideOverlap: true,
          formatter: (h: number) => {
            const q = atOrBefore(props.current.points, h, (x) => x.h);
            return when(h, q?.at ?? null, true);
          },
        },
      })),
      yAxis: [
        {
          type: "value", gridIndex: 0, min: 0, max: 1.1, interval: 0.25, name: "health index",
          nameTextStyle: { color: t.text, align: "left" },
          axisLabel: { color: t.text, formatter: (v: number) => (v > 1 ? "" : v.toFixed(2)) },
          splitLine: { lineStyle: { color: t.grid } },
        },
        {
          // From zero: auto-scaling a healthy signal blows its noise up to
          // fill the panel, and a calm machine looks like a failing one.
          type: "value", gridIndex: 1, min: 0, max: (v: { max: number }) => v.max * 1.2,
          name: `vibration (${p.unit})`, nameTextStyle: { color: t.text, align: "left" },
          splitNumber: 2,
          axisLabel: { color: t.text, formatter: (v: number) => fmt(v, props.current.unit) },
          splitLine: { lineStyle: { color: t.grid } },
        },
      ],
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1], filterMode: "none", rangeMode: ["value", "value"],
          startValue: zs, endValue: ze,
          zoomOnMouseWheel: true, moveOnMouseMove: true },
        { type: "slider", xAxisIndex: [0, 1], filterMode: "none", rangeMode: ["value", "value"],
          startValue: zs, endValue: ze,
          height: 26, bottom: 14, showDetail: false, borderColor: t.grid,
          fillerColor: alpha(t.accent, 0.12), handleStyle: { color: t.accent, borderColor: t.accent },
          moveHandleStyle: { color: t.accent }, dataBackground: {
            lineStyle: { color: t.muted, opacity: 0.6 }, areaStyle: { color: t.muted, opacity: 0.12 } },
          selectedDataBackground: { lineStyle: { color: t.accent }, areaStyle: { color: t.accent, opacity: 0.2 } },
          textStyle: { color: t.text } },
      ],
      series: [
        {
          // Silent, so only the event pins are click targets; the axis tooltip
          // does not need the lines to be hoverable.
          name: "Vibration", type: "line", xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, sampling: "lttb", silent: true,
          lineStyle: { color: t.muted, width: 1.2, opacity: 0.9 }, z: 1,
          areaStyle: { color: alpha(t.muted, 0.08) },
          markLine: { silent: true, symbol: "none", animation: false,
            data: p.cursorH === null ? [] : [{ xAxis: p.cursorH, label: { show: false },
              lineStyle: { color: t.accent, type: "solid", width: 2 } }] },
          data: pts.map((q) => [q.h, q.vib]),
        },
        {
          name: "Health index", type: "line", yAxisIndex: 0, showSymbol: false, silent: true,
          lineStyle: { color: t.accent, width: 2.2 }, z: 3,
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: alpha(t.accent, 0.35) }, { offset: 1, color: alpha(t.accent, 0.02) },
            ]),
          },
          data: pts.map((q) => [q.h, Math.min(q.hhi, 1)]),
          markArea: { silent: true, data: bands(pts, t) },
          markLine: { silent: true, symbol: "none", data: markLines, animation: false },
        },
        {
          name: "Events", type: "scatter", yAxisIndex: 0, z: 5, symbol: "pin", symbolSize: 20,
          cursor: "pointer",
          data: p.events.map((e) => ({
            id: e.id, value: [e.machine_hours, 1.06],
            itemStyle: { color: t[e.state] ?? t.muted, borderColor: t.surface, borderWidth: 1 },
          })),
          tooltip: {
            trigger: "item",
            formatter: (params: { data: { id: string } }) => {
              const e = props.current.events.find((x) => x.id === params.data.id);
              if (!e) return "";
              const dx = e.diagnosis_status === "ok" && e.diagnosis?.failure_mode_id
                ? failureMode(e.diagnosis.failure_mode_id) : e.diagnosis_status === "fallback"
                ? "insufficient evidence" : "diagnosis pending";
              return `<b>${e.state}</b> · ${when(e.machine_hours, e.recorded_at)}<br/>${dx}<br/>
                <span style="color:${t.muted}">click to open this event</span>`;
            },
          },
        },
      ],
    }, { lazyUpdate: true });

    if (p.live && pts.length > 1) p.onView({ start: zs, end: ze });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.points, p.events, p.marks, p.cursorH, p.live, p.windowH, p.theme, p.unit]);

  return <div ref={el} className="timeline" role="img"
    aria-label="Health index and vibration over time, with raised events" />;
}
