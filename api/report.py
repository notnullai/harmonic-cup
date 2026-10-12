"""The machine health report, as a PDF, for a chosen stretch of time.

    GET /report.pdf?from_h=100&to_h=164

Built from the same history and events the dashboard shows — nothing is
recomputed here, only laid out — so the report and the screen cannot
disagree. Server-side on purpose: the same PDF comes out whatever the front
end, and it can be tested.

Layout: a summary a shift supervisor can read in ten seconds, the health trend
over the period with the alarm bands, the events, and the diagnosis in full
for the most recent event of each failure mode, with its actions, safety steps
and citations. Every page carries the advisory notice: Harmonic advises, and
the plant's own protection stays in charge.
"""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from typing import Any

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

STATE_COLOURS = {
    "OK": colors.HexColor("#1f9d55"),
    "WATCH": colors.HexColor("#c99a06"),
    "WARNING": colors.HexColor("#d9640f"),
    "CRITICAL": colors.HexColor("#d1242f"),
    "FAILED": colors.HexColor("#8250df"),
}
INK = colors.HexColor("#0f1720")
MUTED = colors.HexColor("#5b6b7d")
RULE = colors.HexColor("#d5dde6")
PANEL = colors.HexColor("#f3f6f9")
ACCENT = colors.HexColor("#0a7cc1")

STATES = ("OK", "WATCH", "WARNING", "CRITICAL")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica",
                          fontSize=9.2, leading=12.5, textColor=INK, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", parent=body, fontName="Helvetica-Bold",
                                fontSize=20, leading=24, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=10.5,
                                   textColor=MUTED, leading=14),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold",
                             fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=3),
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8, leading=10.5,
                                textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.5),
        "cellb": ParagraphStyle("cellb", parent=body, fontSize=8.2, leading=10.5,
                                fontName="Helvetica-Bold"),
        "kpi": ParagraphStyle("kpi", parent=body, fontName="Helvetica-Bold",
                              fontSize=15, leading=18),
        "kpil": ParagraphStyle("kpil", parent=body, fontSize=7.5, leading=9.5,
                               textColor=MUTED),
    }


def _esc(text: Any) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def when(hours: float, recorded_at: str | None) -> str:
    """A point in time as the reader knows it: the recording's own clock for
    a recording, machine time for the simulator."""
    if recorded_at:
        return recorded_at
    h = int(hours)
    m = round((hours - h) * 60)
    if m == 60:
        h, m = h + 1, 0
    return f"T+{h:d}:{m:02d}"


def _feature(kind: str) -> str:
    return kind.removeprefix("HARMONIC_FEAT_").replace("_", " ")


# --- the trend chart -------------------------------------------------------------


def _trend(points: Sequence[Any], width: float, height: float, unit: str) -> Drawing:
    """Health index (filled, left axis) and overall vibration (line, scaled to
    its own range), over state bands. Drawn directly: a chart library would
    add a dependency for two lines."""
    d = Drawing(width, height)
    left, right, top, bottom = 34, 40, 10, 22
    pw, ph = width - left - right, height - top - bottom
    d.add(Rect(left, bottom, pw, ph, fillColor=colors.white, strokeColor=RULE,
               strokeWidth=0.6))
    if len(points) < 2:
        d.add(String(left + pw / 2, bottom + ph / 2, "not enough data in this range",
                     fontName="Helvetica", fontSize=8, fillColor=MUTED,
                     textAnchor="middle"))
        return d

    t0, t1 = points[0].machine_hours, points[-1].machine_hours
    span = max(t1 - t0, 1e-9)

    def x(h: float) -> float:
        return left + (h - t0) / span * pw

    # State bands: contiguous runs of one non-OK state.
    run_start, run_state = points[0].machine_hours, points[0].state
    for p in [*points[1:], None]:
        if p is None or p.state != run_state:
            end = points[-1].machine_hours if p is None else p.machine_hours
            if run_state != "OK" and run_state in STATE_COLOURS:
                c = STATE_COLOURS[run_state]
                d.add(Rect(x(run_start), bottom, max(x(end) - x(run_start), 0.4), ph,
                           fillColor=colors.Color(c.red, c.green, c.blue, alpha=0.13),
                           strokeColor=None))
            if p is not None:
                run_start, run_state = p.machine_hours, p.state

    # Health index thresholds and gridlines.
    for level, label in ((0.25, "WATCH"), (0.5, "WARNING"), (0.75, "CRITICAL")):
        y = bottom + level * ph
        d.add(Line(left, y, left + pw, y, strokeColor=RULE, strokeWidth=0.5,
                   strokeDashArray=[2, 2]))
        d.add(String(left - 3, y - 2.5, f"{level:.2f}", fontName="Helvetica",
                     fontSize=6.5, fillColor=MUTED, textAnchor="end"))
        d.add(String(left + 3, y + 1.5, label, fontName="Helvetica", fontSize=5.5,
                     fillColor=MUTED))

    # Vibration, scaled to its own range, drawn first so the index sits on top.
    vib = [p.vib_rms for p in points]
    vmin, vmax = min(vib), max(vib)
    vspan = max(vmax - vmin, 1e-9)
    d.add(PolyLine([c for p in points for c in
                    (x(p.machine_hours), bottom + (p.vib_rms - vmin) / vspan * ph * 0.92)],
                   strokeColor=colors.HexColor("#8a99ab"), strokeWidth=0.7))
    for frac, val in ((0.0, vmin), (0.92, vmax)):
        d.add(String(left + pw + 3, bottom + frac * ph - 2.5, f"{val:.3g}",
                     fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    d.add(String(left + pw + 3, bottom + ph - 6, unit, fontName="Helvetica",
                 fontSize=6.5, fillColor=MUTED))

    d.add(PolyLine([c for p in points for c in
                    (x(p.machine_hours), bottom + max(0.0, min(1.0, p.hhi)) * ph)],
                   strokeColor=ACCENT, strokeWidth=1.3))

    # Time axis: five labels.
    for i in range(5):
        p = points[round(i * (len(points) - 1) / 4)]
        d.add(String(x(p.machine_hours), bottom - 10, when(p.machine_hours, p.recorded_at),
                     fontName="Helvetica", fontSize=6.5, fillColor=MUTED,
                     textAnchor="middle" if 0 < i < 4 else ("start" if i == 0 else "end")))
    d.add(String(left, height - 8, "Health index (blue) and overall vibration (grey)",
                 fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return d


# --- the document ----------------------------------------------------------------


def _kpis(cells: list[tuple[str, str, colors.Color | None]], s: dict, width: float) -> Table:
    row = [[Paragraph(f'<font color="{c.hexval() if c else INK.hexval()}">{_esc(v)}</font>',
                      s["kpi"]), Paragraph(_esc(label), s["kpil"])]
           for label, v, c in cells]
    t = Table([[Table([[a], [b]], colWidths=[width / len(cells) - 6]) for a, b in row]],
              colWidths=[width / len(cells)] * len(cells))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LINEBEFORE", (1, 0), (-1, -1), 0.6, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _grid(rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), PANEL),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    t.setStyle(TableStyle(style))
    return t


def _state_cell(state: str, s: dict) -> Paragraph:
    c = STATE_COLOURS.get(state, MUTED)
    return Paragraph(f'<font color="{c.hexval()}"><b>{_esc(state)}</b></font>', s["cell"])


def _feedback_text(fb: dict[str, Any] | None) -> str:
    """What the technician found, in the report's words (HARM-S7-T12)."""
    if not fb:
        return "—"
    if fb.get("verdict") == "confirmed":
        text = "Confirmed"
    else:
        actual = str(fb.get("actual") or "other").replace("_", " ")
        text = "No fault found" if actual == "healthy" else f"Wrong: was {actual}"
    return f"{text}. {fb['note']}" if fb.get("note") else text


def build_report(*, history: Sequence[Any], events: Sequence[Any], source: dict[str, Any],
                 engine: dict[str, Any], from_h: float | None = None,
                 to_h: float | None = None) -> bytes:
    """The report for [from_h, to_h] in machine hours; the whole history when
    either end is omitted."""
    s = _styles()
    pts = [p for p in history
           if (from_h is None or p.machine_hours >= from_h)
           and (to_h is None or p.machine_hours <= to_h)]
    evs = [e for e in events
           if (from_h is None or e.machine_hours >= from_h)
           and (to_h is None or e.machine_hours <= to_h)]

    buf = io.BytesIO()
    width = A4[0] - 36 * mm
    machine = source.get("equipment_id", "machine")
    real = bool(source.get("real"))
    unit = source.get("vib_unit", "")
    generated = time.strftime("%Y-%m-%d %H:%M")

    def decorate(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(INK)
        canvas.rect(0, h - 14 * mm, w, 14 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(18 * mm, h - 9 * mm, "HARMONIC")
        canvas.setFont("Helvetica", 8.5)
        canvas.drawString(46 * mm, h - 9 * mm, "Machine health report")
        badge = "REAL RECORDING" if real else "SIMULATION"
        canvas.setFillColor(colors.HexColor("#1f9d55") if real else colors.HexColor("#c99a06"))
        bw = canvas.stringWidth(badge, "Helvetica-Bold", 7.5) + 8
        canvas.roundRect(w - 18 * mm - bw, h - 10.6 * mm, bw, 5 * mm, 2, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(w - 18 * mm - bw + 4, h - 9.2 * mm, badge)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(18 * mm, 10 * mm,
                          "Advisory only. Harmonic never drives an actuator; the plant's "
                          "own protection and safety systems stay authoritative.")
        canvas.drawRightString(w - 18 * mm, 10 * mm, f"{machine} · page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=22 * mm, bottomMargin=18 * mm,
                            title=f"Harmonic report — {machine}", author="Harmonic")
    story: list[Any] = []

    # -- title
    period = ("the whole history" if not pts else
              f"{when(pts[0].machine_hours, pts[0].recorded_at)} to "
              f"{when(pts[-1].machine_hours, pts[-1].recorded_at)}")
    story += [
        Paragraph(_esc(source.get("name", machine)), s["title"]),
        Paragraph(f"Period: <b>{_esc(period)}</b> · {len(pts)} readings · "
                  f"generated {generated}", s["subtitle"]),
        Spacer(1, 3),
        Paragraph(_esc(source.get("detail", "")), s["small"]),
        Spacer(1, 8),
    ]

    # -- summary
    if pts:
        last = pts[-1]
        peak = max(pts, key=lambda p: p.hhi)
        counts = {st: sum(p.state == st for p in pts) for st in STATES}
        first_alarm = next((p for p in pts if p.state != "OK"), None)
        story.append(_kpis([
            ("State at the end of the period", last.state, STATE_COLOURS.get(last.state)),
            ("Peak health index", f"{peak.hhi:.2f}", None),
            ("Events raised", str(len(evs)), None),
            ("First alarm in the period",
             when(first_alarm.machine_hours, first_alarm.recorded_at) if first_alarm else "none",
             STATE_COLOURS["WATCH"] if first_alarm else STATE_COLOURS["OK"]),
        ], s, width))
        story.append(Spacer(1, 4))
        share = " · ".join(f'<font color="{STATE_COLOURS[st].hexval()}"><b>{st}</b></font> '
                           f"{100 * counts[st] / len(pts):.0f}%" for st in STATES)
        story.append(Paragraph(f"Time in each state: {share}", s["small"]))
        story += [Spacer(1, 8), _trend(pts, width, 62 * mm, unit), Spacer(1, 4)]
    else:
        story.append(Paragraph("No readings fall inside this period.", s["body"]))

    # -- events
    story.append(Paragraph("Events", s["h2"]))
    if not evs:
        story.append(Paragraph("No event was raised in this period.", s["body"]))
    else:
        rows: list[list[Any]] = [[Paragraph(h, s["cellb"]) for h in
                                  ("Time", "State", "Index", "Leading evidence", "Diagnosis",
                                   "On inspection")]]
        for e in evs:
            dx = e.diagnosis or {}
            fm = dx.get("failure_mode_id") or ""
            verdict = (f"{_esc(fm.split('fm.', 1)[-1].replace('_', ' '))} "
                       f"({float(dx.get('confidence', 0)):.0%})" if fm and
                       e.diagnosis_status == "ok" else _esc(
                           {"fallback": "insufficient evidence", "pending": "pending",
                            "running": "pending"}.get(e.diagnosis_status,
                                                      e.diagnosis_status)))
            rows.append([
                Paragraph(_esc(when(e.machine_hours, e.recorded_at)), s["cell"]),
                _state_cell(e.state, s),
                Paragraph(f"{e.hhi:.2f}", s["cell"]),
                Paragraph(_esc(", ".join(_feature(f["kind"]) + (" ↑" if f.get("direction") == "up"
                                                                 else " ↓" if f.get("direction") == "down" else "")
                                         for f in e.top_surprised_features[:3])), s["cell"]),
                Paragraph(verdict, s["cell"]),
                Paragraph(_esc(_feedback_text(getattr(e, "feedback", None))), s["cell"]),
            ])
        story.append(_grid(rows, [28 * mm, 18 * mm, 12 * mm, 45 * mm, width - 133 * mm,
                                  30 * mm]))

    # -- diagnoses in full: the latest of each failure mode
    latest: dict[str, Any] = {}
    for e in evs:
        dx = e.diagnosis or {}
        if e.diagnosis_status == "ok" and dx.get("failure_mode_id"):
            latest[dx["failure_mode_id"]] = e
    if latest:
        story.append(Paragraph("Diagnosis", s["h2"]))
        provider = next(iter(latest.values())).diagnosis_provider or "unknown"
        if provider.startswith("offline"):
            story.append(Paragraph(
                "Assembled by the offline knowledge-graph composer, not a language model: "
                "it quotes the knowledge graph and the manual and does not reason beyond "
                "them. Every citation below was checked against what was retrieved.",
                s["small"]))
    for fm, e in sorted(latest.items(), key=lambda kv: -kv[1].machine_hours):
        dx = e.diagnosis
        block: list[Any] = [
            Paragraph(f"{_esc(fm.split('fm.', 1)[-1].replace('_', ' ').capitalize())} — "
                      f"confidence {float(dx.get('confidence', 0)):.0%}", s["h3"]),
            Paragraph(f"Latest at {_esc(when(e.machine_hours, e.recorded_at))}, "
                      f"state {e.state}, health index {e.hhi:.2f}.", s["small"]),
            Spacer(1, 3),
            Paragraph(_esc(dx.get("diagnosis", "")), s["body"]),
        ]
        if dx.get("root_causes"):
            block += [Spacer(1, 3), Paragraph("<b>Likely causes</b>", s["body"])]
            block += [Paragraph(f"• {_esc(c)}", s["body"]) for c in dx["root_causes"]]
        story.append(KeepTogether(block))
        if dx.get("actions"):
            rows = [[Paragraph(h, s["cellb"]) for h in ("#", "Step", "Tool", "Time", "Skill")]]
            for i, a in enumerate(dx["actions"], 1):
                rows.append([Paragraph(str(i), s["cell"]), Paragraph(_esc(a.get("step", "")), s["cell"]),
                             Paragraph(_esc(a.get("tool", "")), s["cell"]),
                             Paragraph(f"{float(a.get('duration_min', 0)):.0f} min", s["cell"]),
                             Paragraph(_esc(a.get("skill_level", "")), s["cell"])])
            story += [Spacer(1, 4), _grid(rows, [7 * mm, width - 97 * mm, 42 * mm, 18 * mm, 30 * mm])]
        if dx.get("safety"):
            story += [Spacer(1, 4), Paragraph("<b>Before any work</b>", s["body"])]
            story += [Paragraph(f"• {_esc(x)}", s["body"]) for x in dx["safety"]]
        if dx.get("references"):
            story += [Spacer(1, 3), Paragraph("Sources: " + _esc(", ".join(dx["references"])),
                                              s["small"])]
        if machine != "P-101":
            story.append(Paragraph("The procedures cite the P-101 reference manual, the only "
                                   "manual in the corpus; this machine has none.", s["small"]))

    # -- provenance
    story += [Spacer(1, 10), Paragraph("About this report", s["h3"]),
              Paragraph(
                  f"Engine brain <b>{_esc(str(engine.get('brain', '?')).rsplit('/', 1)[-1])}</b>, "
                  f"{engine.get('feature_count', '?')} features, "
                  f"{'commissioned' if engine.get('commissioned') else 'not commissioned'}. "
                  "The health state, index and events come from the fixed-point engine; "
                  "the diagnosis from the knowledge graph and the manual. This report lays "
                  "out what the dashboard shows and computes nothing of its own.",
                  s["small"])]

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buf.getvalue()
