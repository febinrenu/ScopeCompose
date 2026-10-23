/*  Act 06: the decision at a glance.
 *
 *  A small live tree beside the reader controls:
 *
 *      question
 *       ├── branch (rule)        ── kept by ──>  Selection
 *       └── branch (exception)   ── kept by ──>  ScopeCompose
 *
 *  Every edge comes from what the engines returned: which branches selection
 *  kept, which the composition kept. The reader marker sits on the branch
 *  that is true for the reader; change the reader and it moves (GSAP Flip),
 *  and the answer that dropped their branch turns coral.
 */
import { gsap } from "gsap";
import { esc } from "../ui/dom";

// Flip is only needed once someone changes the input, so it loads on the side
// (it would otherwise be the one thing pushing the entry bundle over budget).
let Flip: typeof import("gsap/Flip").Flip | null = null;
const flipReady = import("gsap/Flip").then(m => { gsap.registerPlugin(m.Flip); Flip = m.Flip; }).catch(() => {});
void flipReady;

export interface TreeBranch { id: string; sup: string; def: boolean; cond: string; out: string }
export interface TreeModel {
  branches: TreeBranch[];
  selKept: string[];
  compKept: string[];
  selLabel: string;
  compLabel: string;
  /** the branch that is true for the reader (most specific), or null when no reader is set */
  readerAt: string | null;
  /** for the reader: did each answer keep their branch? null when no reader is set */
  selRight: boolean | null;
  compRight: boolean | null;
}

const NS = "http://www.w3.org/2000/svg";

export class BranchTree {
  private svg: SVGSVGElement;
  private body: HTMLElement;
  private model: TreeModel | null = null;
  private edgeSig = "";
  private rm = () => document.documentElement.classList.contains("rm");

  constructor(host: HTMLElement) {
    host.classList.add("rz-tree");
    host.innerHTML = `<h3>The decision at a glance</h3><div class="rt-body"><svg class="rt-edges" aria-hidden="true"></svg><div class="rt-grid"></div></div>`;
    this.svg = host.querySelector("svg")!;
    this.body = host.querySelector(".rt-grid")!;
    new ResizeObserver(() => this.drawEdges(false)).observe(host);
  }

  update(m: TreeModel) {
    const state = this.model && !this.rm() && Flip ? Flip.getState(this.body.querySelectorAll("[data-flip-id]")) : null;
    this.model = m;
    const node = (b: TreeBranch) => `
      <div class="rt-node rt-branch ${b.def ? "def" : "exc"} ${m.readerAt === b.id ? "is-reader" : ""}" data-flip-id="b-${esc(b.id)}" data-b="${esc(b.id)}">
        <span class="rt-cond">${b.def ? "Rule" : "If " + esc(b.cond)}</span>
        <span class="rt-out">${esc(b.out)}</span>
        <span class="rt-sup mono">${esc(b.sup)}</span>
        ${m.readerAt === b.id ? `<span class="rt-reader mono" data-flip-id="reader">reader</span>` : ""}
      </div>`;
    const verdict = (right: boolean | null) => right === null ? "" :
      `<span class="rt-v ${right ? "ok" : "bad"}">${right ? "right for this reader" : "wrong for this reader"}</span>`;
    this.body.innerHTML = `
      <div class="rt-row rt-q"><div class="rt-node q" data-flip-id="q" data-q>Question</div></div>
      <div class="rt-row rt-bs" style="--n:${m.branches.length}">${m.branches.map(node).join("")}</div>
      <div class="rt-row rt-as">
        <div class="rt-node ans sel ${m.selRight === false ? "wrong" : ""}" data-flip-id="sel" data-a="sel">
          <span class="rt-k mono">Selection</span><span class="rt-keep">${esc(m.selLabel)}</span>${verdict(m.selRight)}</div>
        <div class="rt-node ans comp ${m.compRight === false ? "wrong" : ""}" data-flip-id="comp" data-a="comp">
          <span class="rt-k mono">ScopeCompose</span><span class="rt-keep">${esc(m.compLabel)}</span>${verdict(m.compRight)}</div>
      </div>
      ${m.readerAt === null ? `<p class="rt-hint">Set a reader above: a marker moves onto the branch that is true for them.</p>` : ""}`;

    if (state && Flip) {
      this.svg.style.opacity = "0.25";
      Flip.from(state, {
        duration: 0.6, ease: "power3.inOut", absolute: true, nested: true,
        onComplete: () => { this.svg.style.opacity = ""; this.drawEdges(true); },
      });
      // the answer boxes flash as their verdict changes
      gsap.fromTo(this.body.querySelectorAll(".rt-v"), { opacity: 0, y: 4 }, { opacity: 1, y: 0, duration: 0.4, delay: 0.35 });
    } else {
      this.drawEdges(true);
    }
  }

  /** Edges from the DOM boxes; drawn on (stroke offset) when they change. */
  private drawEdges(animate: boolean) {
    const m = this.model;
    if (!m) return;
    const box = this.svg.getBoundingClientRect();
    if (!box.width) return;
    this.svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    const rect = (el: Element | null) => {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: r.left - box.left, y: r.top - box.top, w: r.width, h: r.height };
    };
    const q = rect(this.body.querySelector("[data-q]"));
    const sel = rect(this.body.querySelector('[data-a="sel"]'));
    const comp = rect(this.body.querySelector('[data-a="comp"]'));
    if (!q || !sel || !comp) return;

    type E = { d: string; cls: string; x?: [number, number] };
    const edges: E[] = [];
    const curve = (x0: number, y0: number, x1: number, y1: number) => {
      const my = (y0 + y1) / 2;
      return `M${x0.toFixed(1)},${y0.toFixed(1)} C${x0.toFixed(1)},${my.toFixed(1)} ${x1.toFixed(1)},${my.toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)}`;
    };
    for (const b of m.branches) {
      const r = rect(this.body.querySelector(`[data-b="${CSS.escape(b.id)}"]`));
      if (!r) continue;
      const mine = m.readerAt === b.id ? " mine" : "";
      edges.push({ d: curve(q.x + q.w / 2, q.y + q.h, r.x + r.w / 2, r.y), cls: "e-q" + mine });
      const bx = r.x + r.w / 2, by = r.y + r.h;
      const keptS = m.selKept.includes(b.id), keptC = m.compKept.includes(b.id);
      const sx = sel.x + sel.w * 0.5, cx = comp.x + comp.w * 0.5;
      edges.push({ d: curve(bx, by, sx, sel.y), cls: (keptS ? "e-sel" : "e-del") + mine, x: keptS ? undefined : [(bx + sx) / 2, (by + sel.y) / 2] });
      if (keptC) edges.push({ d: curve(bx, by, cx, comp.y), cls: "e-comp" + mine });
    }
    const sig = edges.map(e => e.cls + e.d).join("|");
    const changed = sig !== this.edgeSig;
    this.edgeSig = sig;
    this.svg.replaceChildren(...edges.flatMap(e => {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", e.d); p.setAttribute("class", e.cls);
      const out: SVGElement[] = [p];
      if (e.x) {
        // a deleted edge carries a cross where selection cut it
        const g = document.createElementNS(NS, "path");
        const [x, y] = e.x, s = 4.5;
        g.setAttribute("d", `M${x - s},${y - s} L${x + s},${y + s} M${x + s},${y - s} L${x - s},${y + s}`);
        g.setAttribute("class", "e-x");
        out.push(g);
      }
      return out;
    }));
    if (animate && changed && !this.rm()) {
      this.svg.querySelectorAll<SVGPathElement>("path:not(.e-x):not(.e-del)").forEach((p, i) => {
        const len = p.getTotalLength();
        gsap.fromTo(p, { strokeDasharray: len, strokeDashoffset: len },
          { strokeDashoffset: 0, duration: 0.7, delay: i * 0.04, ease: "power2.out", clearProps: "strokeDasharray,strokeDashoffset" });
      });
    }
  }
}
