import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";
import { Maximize2, Network } from "lucide-react";
import { api, type HEvent, type KGSubgraph } from "../api";
import { failureMode, feature } from "../format";

cytoscape.use(dagre);

const cache = new Map<string, KGSubgraph>();

const KIND = {
  symptom: "#8ec5ff",
  failure: "#ff9b94",
  cause: "#cdb4f5",
  action: "#9fe0b4",
} as const;

function elements(kg: KGSubgraph): cytoscape.ElementDefinition[] {
  const els: cytoscape.ElementDefinition[] = [];
  const seen = new Set<string>();
  const node = (id: string, label: string, kind: keyof typeof KIND, extra: Record<string, unknown> = {}) => {
    if (seen.has(id)) return;
    seen.add(id);
    els.push({ data: { id, label, kind, ...extra } });
  };
  // The leading candidate in full; the runner-up with enough to show why.
  const limits = [[4, 2, 3], [3, 1, 1]];
  kg.candidates.slice(0, 2).forEach((c, i) => {
    const [ns, nc, na] = limits[i];
    node(c.failure_mode_id, `${failureMode(c.failure_mode_id)}\n${Math.round(c.score * 100)}%`, "failure",
      { lead: i === 0 ? 1 : 0 });
    c.symptoms.slice(0, ns).forEach((s) => {
      const arrow = s.direction === "up" ? " ↑" : s.direction === "down" ? " ↓" : "";
      node(s.symptom_id, `${feature(s.feature_kind)}${arrow}`, "symptom", { tip: s.description });
      els.push({ data: { id: `${s.symptom_id}>${c.failure_mode_id}`, source: s.symptom_id,
        target: c.failure_mode_id, label: s.confidence.toFixed(2) } });
    });
    c.causes.slice(0, nc).forEach((k) => {
      node(k.cause_id, k.description, "cause");
      els.push({ data: { id: `${c.failure_mode_id}>${k.cause_id}`, source: c.failure_mode_id,
        target: k.cause_id, label: "cause", dashed: 1 } });
    });
    c.actions.slice(0, na).forEach((a) => {
      node(a.action_id, `${a.priority}. ${a.description}`, "action");
      els.push({ data: { id: `${c.failure_mode_id}>${a.action_id}`, source: c.failure_mode_id,
        target: a.action_id, label: "" } });
    });
  });
  return els;
}

interface Props { event: HEvent | undefined; theme: string }

export default function KnowledgeGraph({ event, theme }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const cy = useRef<cytoscape.Core | null>(null);
  const [kg, setKg] = useState<KGSubgraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tip, setTip] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    if (!event) { setKg(null); return; }
    const hit = cache.get(event.id);
    if (hit) { setKg(hit); return; }
    let live = true;
    api.kg(event.id).then((r) => { cache.set(event.id, r); if (live) setKg(r); })
      .catch((e: Error) => live && setError(e.message));
    return () => { live = false; };
  }, [event?.id]);

  useEffect(() => {
    if (!el.current || !kg || !kg.candidates.length) return;
    // Cytoscape draws into a box of its own, created and removed here, inside
    // a host React keeps mounted. Handed a node React also managed, its
    // teardown and React's reconciliation removed the same children, and the
    // second removal threw and took the panel down whenever the graph gave
    // way to an empty state and back.
    const box = document.createElement("div");
    box.style.cssText = "position:absolute;inset:0";
    el.current.appendChild(box);
    const cs = getComputedStyle(document.documentElement);
    const text = cs.getPropertyValue("--muted").trim();
    const g = cytoscape({
      container: box,
      elements: elements(kg),
      wheelSensitivity: 0.25,
      minZoom: 0.3,
      maxZoom: 2.5,
      style: [
        {
          selector: "node",
          style: {
            shape: "round-rectangle", label: "data(label)", "text-wrap": "wrap", "text-max-width": "150px",
            "font-size": 11, "font-family": "Inter Variable, Inter, system-ui, sans-serif",
            color: "#0b1118", "text-valign": "center", "text-halign": "center",
            width: "label", height: "label", padding: "10px", "border-width": 0,
            "background-color": (n: cytoscape.NodeSingular) => KIND[n.data("kind") as keyof typeof KIND],
          },
        },
        {
          selector: 'node[kind = "failure"]',
          style: { "font-size": 12.5, "font-weight": 700, "border-width": 2, "border-color": "#d1242f",
            "text-max-width": "170px" },
        },
        { selector: "node[lead = 0]", style: { opacity: 0.8, "border-style": "dashed" } },
        {
          selector: "edge",
          style: {
            width: 1.4, "curve-style": "bezier", "line-color": text, "target-arrow-color": text,
            "target-arrow-shape": "triangle", "arrow-scale": 0.8, label: "data(label)", "font-size": 9.5,
            color: text, "text-background-color": cs.getPropertyValue("--surface-2").trim(),
            "text-background-opacity": 1, "text-background-padding": "2px", opacity: 0.85,
          },
        },
        { selector: "edge[dashed = 1]", style: { "line-style": "dashed" } },
      ],
      layout: { name: "dagre", rankDir: "LR", nodeSep: 14, rankSep: 70, edgeSep: 8, padding: 16 } as cytoscape.LayoutOptions,
    });
    g.on("mouseover", "node", (e) => setTip(e.target.data("tip") ?? null));
    g.on("mouseout", "node", () => setTip(null));
    cy.current = g;
    const ro = new ResizeObserver(() => { g.resize(); g.fit(undefined, 16); });
    ro.observe(el.current);
    return () => { ro.disconnect(); g.destroy(); box.remove(); cy.current = null; };
  }, [kg, theme]);

  const ruled = kg?.ruled_out ?? [];
  const showGraph = !!event && !error && !!kg && kg.candidates.length > 0;
  return (
    <section className="card">
      <div className="card-h">
        <h2>Knowledge graph path</h2><span className="grow" />
        {kg?.candidates.length ? (
          <button className="btn ghost icon sm" title="Fit to view" aria-label="Fit graph to view"
            onClick={() => cy.current?.fit(undefined, 16)}><Maximize2 size={14} /></button>
        ) : null}
      </div>
      <div className="card-b">
        {/* Every state has its own key, so React never reuses one state's
            node for another; the graph host stays mounted throughout. */}
        {!event ? (
          <div key="none" className="empty"><Network size={28} /><div>The graph is queried when an event is raised.</div></div>
        ) : error ? (
          <div key="err" className="note">Graph unavailable — {error}</div>
        ) : !kg ? (
          <div key="load" className="kg skeleton" />
        ) : !kg.candidates.length ? (
          <div key="nomatch" className="empty"><Network size={28} /><div>The surprised features match no failure mode this machine can have.</div></div>
        ) : null}
        <div key="graph" ref={el} className="kg" role="img" aria-label="Evidence, failure mode, causes and actions"
          style={{ position: "relative", display: showGraph ? "block" : "none" }} />
        {showGraph && kg && (
          <div key="details">
            <div className="kg-legend">
              <span><i style={{ background: KIND.symptom }} />evidence</span>
              <span><i style={{ background: KIND.failure }} />failure mode</span>
              <span><i style={{ background: KIND.cause }} />likely cause</span>
              <span><i style={{ background: KIND.action }} />action, by priority</span>
              <span className="muted">· scroll to zoom, drag to pan</span>
            </div>
            {tip && <div className="note" style={{ marginTop: 8 }}>{tip}</div>}
            {kg.agreement.startsWith("disagree") && (
              <div className="note" style={{ marginTop: 8 }}>
                The trained class and the symptom evidence point at different failure modes; both are shown.
              </div>
            )}
            {ruled.length > 0 && (
              <div className="note" style={{ marginTop: 8 }}>
                Ruled out — no component on this machine for: {ruled.map((r) => failureMode(r).toLowerCase()).join(", ")}.
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
