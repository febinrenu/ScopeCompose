/*  A quiet custom cursor: a dot, and a ring that names what you can do.
 *  DRAG over the hero and the scope volumes; X-RAY over the failure act,
 *  where the ring becomes the lens. Mouse only -- never on touch, never in
 *  reduced motion -- and the system cursor returns over anything you can
 *  type into or press.
 */
export type CursorMode = "none" | "drag" | "xray";

export class Cursor {
  private dot = document.createElement("div");
  private ring = document.createElement("div");
  private label = document.createElement("span");
  private x = -100; private y = -100; private rx = -100; private ry = -100;
  private mode: CursorMode = "none";
  private over = false;
  private seen = false;                 // nothing shows until a mouse has moved
  private raf = 0;

  constructor() {
    this.dot.className = "cur-dot"; this.ring.className = "cur-ring";
    this.ring.appendChild(this.label);
    this.dot.setAttribute("aria-hidden", "true"); this.ring.setAttribute("aria-hidden", "true");
    this.ring.style.transform = this.dot.style.transform = "translate3d(-100px, -100px, 0)";
    document.body.append(this.ring, this.dot);
    document.documentElement.classList.add("has-cursor");
    addEventListener("pointermove", e => {
      if (e.pointerType !== "mouse") return;
      if (!this.seen) { this.seen = true; this.rx = e.clientX; this.ry = e.clientY; }
      this.x = e.clientX; this.y = e.clientY;
      const t = e.target as HTMLElement;
      this.over = !!t.closest("a, button, input, select, textarea, label, summary, [role=tab], [role=radio], .contract, .explorer, .reasoner, .lab-view");
      this.paint();
      if (!this.raf) this.raf = requestAnimationFrame(this.loop);
    }, { passive: true });
    document.addEventListener("pointerleave", () => { this.x = this.y = -100; this.paint(); });
  }

  setMode(m: CursorMode) {
    if (m === this.mode) return;
    this.mode = m;
    this.ring.dataset.mode = m;
    this.label.textContent = m === "drag" ? "drag" : m === "xray" ? "x-ray" : "";
    this.paint();
  }

  private paint() {
    const active = this.over || !this.seen || this.x < 0 ? "none" : this.mode;
    this.ring.dataset.active = active;
    this.dot.style.transform = `translate3d(${this.x}px, ${this.y}px, 0)`;
    document.documentElement.classList.toggle("cursor-hidden", active !== "none");
  }

  private loop = () => {
    this.rx += (this.x - this.rx) * 0.22; this.ry += (this.y - this.ry) * 0.22;
    this.ring.style.transform = `translate3d(${this.rx}px, ${this.ry}px, 0)`;
    this.raf = Math.abs(this.x - this.rx) + Math.abs(this.y - this.ry) > 0.3 ? requestAnimationFrame(this.loop) : 0;
  };
}
