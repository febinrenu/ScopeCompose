/*  The loader counts real steps only (engines verified, archive read, story
 *  choreographed, world built, first light) -- 000 to 100 -- then the
 *  curtain splits. It never holds the page hostage: a CSS fail-safe drops it
 *  on its own, and it is skipped quickly on repeat visits.
 */
import gsap from "gsap";

const el = () => document.getElementById("loader");
const counter = { v: 0 };

export const loader = {
  steps: 0,
  total: 5,
  status(text: string) {
    const s = document.getElementById("ldStatus");
    if (s) s.textContent = text;
  },
  step(text: string) {
    this.steps++;
    this.status(text);
    const target = Math.min(100, Math.round((this.steps / this.total) * 100));
    const p = document.getElementById("ldPct");
    gsap.to(counter, {
      v: target, duration: 0.5, ease: "power2.out",
      onUpdate: () => { if (p) p.textContent = String(Math.round(counter.v)).padStart(3, "0"); },
    });
  },
  fail(message: string) {
    this.status(message);
    const l = el();
    if (l) l.style.animation = "none";
  },
  done() {
    const l = el();
    if (!l) return;
    const p = document.getElementById("ldPct");
    gsap.to(counter, {
      v: 100, duration: 0.35, ease: "power2.out",
      onUpdate: () => { if (p) p.textContent = String(Math.round(counter.v)).padStart(3, "0"); },
      onComplete: () => {
        l.classList.add("done");
        try { sessionStorage.setItem("sc-seen", "1"); } catch { /* private mode */ }
        setTimeout(() => l.remove(), 1600);
      },
    });
  },
};
