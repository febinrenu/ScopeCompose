/*  Two small physical cues, mouse only, off in reduced motion:
 *  cards catch a soft light where the pointer is, and the header's round
 *  buttons lean a little toward it. Purely decorative; nothing depends on them.
 */
const CARDS = ".tcard, .fv, .composed-card, .vs-side, .big-stats > *, .rt-node, .lab-view .stat";
const MAGNETS = ".icon-btn, .brand, .track-foot a";

export function initHover() {
  if (!matchMedia("(hover: hover) and (pointer: fine)").matches) return;
  const rm = () => document.documentElement.classList.contains("rm");
  let lit: HTMLElement | null = null, mag: HTMLElement | null = null;

  addEventListener("pointermove", e => {
    if (e.pointerType !== "mouse" || rm()) return;
    const t = e.target as HTMLElement;
    const card = t.closest<HTMLElement>(CARDS);
    if (card !== lit) { lit?.classList.remove("lit"); lit = card; }
    if (card) {
      card.classList.add("glow", "lit");
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    }
    const m = t.closest<HTMLElement>(MAGNETS);
    if (m !== mag) { if (mag) mag.style.transform = ""; mag = m; }
    if (m) {
      m.classList.add("magnet");
      const r = m.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) * 0.28, dy = (e.clientY - (r.top + r.height / 2)) * 0.28;
      m.style.transform = `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`;
    }
  }, { passive: true });
  document.addEventListener("pointerleave", () => { lit?.classList.remove("lit"); if (mag) mag.style.transform = ""; lit = mag = null; });
}
