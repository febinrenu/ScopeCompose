/*  The pipeline graph: fourteen stages, drawn as a serpentine circuit.
 *
 *  Nodes carry their status (pending / running / done / skipped / warn /
 *  error), their time, and a badge for LLM calls made inside them. When a
 *  stage starts, a packet of light travels the edge into it; while it runs
 *  the node breathes. Click a node to inspect it.
 */
import { PIPELINE, STAGE_LABEL, STAGE_SUB, type RunState } from "./model";

const NS = "http://www.w3.org/2000/svg";
const W = 1000, H = 236, NW = 126, NH = 76;
const ROW = [62, 176];

/** Node centres: row one left→right, row two right→left. */
function layout(): Record<string, [number, number]> {
  const pos: Record<string, [number, number]> = {};
  const per = 7, x0 = 72, dx = (W - 160) / (per - 1);
  PIPELINE.forEach((id, i) => {
    const row = i < per ? 0 : 1;
    const col = row === 0 ? i : per - 1 - (i - per);
    pos[id] = [x0 + col * dx, ROW[row]];
  });
  return pos;
}

function edgePath(a: [number, number], b: [number, number]): string {
  if (a[1] === b[1]) {
    const dir = Math.sign(b[0] - a[0]);
    return `M${a[0] + dir * NW / 2},${a[1]} L${b[0] - dir * NW / 2},${b[1]}`;
  }
  // the U-turn at the right end
  const x = a[0] + NW / 2 + 34;
  return `M${a[0] + NW / 2},${a[1]} C${x},${a[1]} ${x},${b[1]} ${b[0] + NW / 2},${b[1]}`;
}

export class PipelineGraph {
  readonly el: SVGSVGElement;
  private pos = layout();
  private nodes: Record<string, SVGGElement> = {};
  private edges: Record<string, SVGPathElement> = {};
  private seen = new Set<string>();
  selected: string | null = null;

  constructor(private onSelect: (id: string) => void) {
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "pg");
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", "Pipeline stages");
    this.el = svg;
    svg.innerHTML = `<defs>
      <radialGradient id="pgGlow"><stop offset="0" stop-color="#fff" stop-opacity="1"/><stop offset="0.35" stop-color="#7ff0ff" stop-opacity="0.9"/><stop offset="1" stop-color="#7ff0ff" stop-opacity="0"/></radialGradient>
      <pattern id="pgGrid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="rgba(244,242,236,0.04)"/></pattern>
    </defs><rect width="${W}" height="${H}" fill="url(#pgGrid)"/>`;

    PIPELINE.forEach((id, i) => {
      if (i === 0) return;
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", edgePath(this.pos[PIPELINE[i - 1]], this.pos[id]));
      p.setAttribute("class", "pg-edge");
      svg.appendChild(p);
      this.edges[id] = p;
    });

    for (const id of PIPELINE) {
      const [x, y] = this.pos[id];
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "pg-node");
      g.setAttribute("data-s", "pending");
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", `${STAGE_LABEL[id]}: pending`);
      g.setAttribute("transform", `translate(${x - NW / 2},${y - NH / 2})`);
      g.innerHTML = `<rect class="pg-halo" x="-6" y="-6" width="${NW + 12}" height="${NH + 12}" rx="14"/>
        <rect class="pg-box" width="${NW}" height="${NH}" rx="9"/>
        <text class="pg-title" x="12" y="25">${STAGE_LABEL[id]}</text>
        <text class="pg-sub" x="12" y="43">${STAGE_SUB[id] || ""}</text>
        <text class="pg-ms" x="12" y="62"></text>
        <g class="pg-llm" transform="translate(${NW - 14},0)"><circle r="10"/><text y="4" text-anchor="middle"></text></g>
        <circle class="pg-dot" cx="${NW - 14}" cy="${NH - 14}" r="4"/>`;
      g.addEventListener("click", () => this.select(id));
      g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.select(id); } });
      svg.appendChild(g);
      this.nodes[id] = g;
    }
  }

  select(id: string) {
    this.selected = id;
    for (const [k, g] of Object.entries(this.nodes)) g.classList.toggle("sel", k === id);
    this.onSelect(id);
  }

  reset() {
    this.seen.clear();
    for (const id of PIPELINE) this.paint(id, "pending", "", 0);
    Object.values(this.edges).forEach(e => e.classList.remove("lit"));
    this.el.querySelectorAll(".pg-packet").forEach(p => p.remove());
  }

  private paint(id: string, status: string, ms: string, llm: number) {
    const g = this.nodes[id];
    g.setAttribute("data-s", status);
    g.setAttribute("aria-label", `${STAGE_LABEL[id]}: ${status}${ms ? ", " + ms : ""}`);
    (g.querySelector(".pg-ms") as SVGTextElement).textContent = ms;
    const badge = g.querySelector(".pg-llm") as SVGGElement;
    badge.style.display = llm ? "" : "none";
    (badge.querySelector("text") as SVGTextElement).textContent = String(llm);
  }

  /** Bring the drawing up to the state. `animate` sends packets for stages that just started. */
  update(s: RunState, animate = true) {
    for (const id of PIPELINE) {
      const st = s.stages[id];
      if (!st) { this.paint(id, "pending", "", 0); continue; }
      const ms = st.ms !== null ? (st.ms < 1 ? "<1 ms" : st.ms < 1000 ? `${Math.round(st.ms)} ms` : `${(st.ms / 1000).toFixed(1)} s`) : st.status === "running" ? "running…" : "";
      this.paint(id, st.status, ms, st.llm);
      if (this.edges[id]) this.edges[id].classList.toggle("lit", st.status !== "pending");
      if (!this.seen.has(id) && st.status !== "pending") {
        this.seen.add(id);
        if (animate && this.edges[id]) this.packet(this.edges[id]);
      }
    }
  }

  private packet(edge: SVGPathElement) {
    const len = edge.getTotalLength();
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("r", "9");
    dot.setAttribute("class", "pg-packet");
    dot.setAttribute("fill", "url(#pgGlow)");
    this.el.appendChild(dot);
    const t0 = performance.now(), dur = 380;
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      const p = edge.getPointAtLength(e * len);
      dot.setAttribute("cx", String(p.x)); dot.setAttribute("cy", String(p.y));
      if (k < 1) requestAnimationFrame(step); else dot.remove();
    };
    requestAnimationFrame(step);
  }
}
