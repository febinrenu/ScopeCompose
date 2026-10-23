/*  Portals between acts.
 *
 *  As an act scrolls out, the frame closes to a circle; at the turning point
 *  the screen is the brand mark (the rule's ring holding the exception's),
 *  with the next scene seen through it; then it opens on the next act.
 *
 *  A pure function of the story cursor, like everything else in the world,
 *  so it plays backwards exactly when you scroll up. DOM, not WebGL: it
 *  costs a repaint, works at every tier, and sits under the copy.
 */
import { ACT_FROM, stopIndex } from "../acts/keyframes";

/** Acts that are entered through a portal (the paper research act and the frontier have their own entrances). */
const PORTAL_ACTS = [1, 2, 4, 5, 7, 10];

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (a: number, b: number, x: number) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };

/** How closed the iris is, 0 (open) .. 1 (the mark), for a story cursor. Exported for tests. */
export function portalClosure(cursor: number): number {
  for (const act of PORTAL_ACTS) {
    const b = stopIndex(ACT_FROM[act]);
    const u = cursor - (b - 1);
    if (u > 0 && u < 1) return smooth(0.04, 0.44, u) * (1 - smooth(0.56, 0.96, u));
  }
  return 0;
}

export class Portal {
  private el: HTMLElement;
  private last = -1;

  constructor(parent: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "portal";
    this.el.setAttribute("aria-hidden", "true");
    this.el.innerHTML = `<i class="portal-ring"></i><i class="portal-exc"></i>`;
    parent.appendChild(this.el);
  }

  set(cursor: number, enabled: boolean) {
    const e = enabled ? portalClosure(cursor) : 0;
    if (Math.abs(e - this.last) < 1e-4) return;
    this.last = e;
    const s = this.el.style;
    if (e <= 0) { s.visibility = "hidden"; return; }
    s.visibility = "visible";
    // radius: from past the corners down to a mark-sized porthole
    const rMax = Math.hypot(innerWidth, innerHeight) / 2 + 40;
    const rMin = Math.min(innerWidth, innerHeight) * 0.11;
    const r = rMax + (rMin - rMax) * e;
    s.setProperty("--r", r.toFixed(1) + "px");
    s.setProperty("--e", e.toFixed(3));
  }
}
