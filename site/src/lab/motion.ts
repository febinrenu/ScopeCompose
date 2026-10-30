/*  Motion for the lab views, the same vocabulary as the story:
 *  figures decode into their value, charts grow from their axis, diagrams
 *  draw themselves. Everything plays once, as it enters, and always ends on
 *  the exact final state. Nothing runs in reduced motion.
 */
import gsap from "gsap";
import { $$ } from "../ui/dom";
import { scrambleOnEnter } from "../motion/text";

const reduced = () => document.documentElement.classList.contains("rm")
  || matchMedia("(prefers-reduced-motion: reduce)").matches;

function onEnter(els: Element[], play: (el: Element) => void, margin = "0px 0px -12% 0px") {
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    io.unobserve(e.target);
    play(e.target);
  }), { rootMargin: margin });
  els.forEach(el => io.observe(el));
}

/** The page's headline rises in, character by character, as in the story. */
export function introduce(host: HTMLElement) {
  if (reduced()) return;
  for (const h of $$<HTMLElement>(":scope > .wrap > .display", host)) {
    gsap.from(h.children.length ? h.children : h, { yPercent: 40, opacity: 0, duration: 0.9, ease: "expo.out", stagger: 0.08 });
  }
  const rest = $$<HTMLElement>(":scope > .wrap > .lede, :scope > .wrap > .eyebrow", host);
  if (rest.length) gsap.from(rest, { opacity: 0, y: 12, duration: 0.7, ease: "power3.out", delay: 0.15, stagger: 0.06 });
}

/** Figures that are numbers (0.976, 0 / 42, 1.0000) decode as they enter. */
export function decodeFigures(host: ParentNode, selector = ".stat .v") {
  if (reduced()) return;
  for (const el of $$<HTMLElement>(selector, host)) {
    if (/^[\d\s./–·%-]+$/.test((el.textContent || "").trim())) el.setAttribute("data-scramble", "");
  }
  scrambleOnEnter(host);
}

/** Bars in an SVG chart grow up from the axis, left to right. */
export function growBars(host: ParentNode, selector: string) {
  if (reduced()) return;
  for (const svg of $$<SVGSVGElement>(selector, host)) {
    const bars = $$<SVGRectElement>("rect", svg);
    if (!bars.length) continue;
    gsap.set(bars, { scaleY: 0, transformOrigin: "50% 100%", transformBox: "fill-box" } as gsap.TweenVars);
    onEnter([svg], () => gsap.to(bars, { scaleY: 1, duration: 0.9, ease: "expo.out", stagger: { each: 0.012, from: "start" } }));
  }
}

/** Lines and paths draw themselves, then the boxes they connect rise in. */
export function drawDiagram(svg: SVGSVGElement, { lines = "path.edge, line", nodes = "g.node, g.contract" } = {}) {
  if (reduced()) return;
  const paths = $$<SVGGeometryElement>(lines, svg);
  const boxes = $$<SVGGElement>(nodes, svg);
  paths.forEach(p => { p.setAttribute("pathLength", "1"); p.style.strokeDasharray = "1"; p.style.strokeDashoffset = "1"; });
  // opacity only: a GSAP transform would replace the SVG translate() that positions each box
  gsap.set(boxes, { opacity: 0 });
  onEnter([svg], () => {
    const tl = gsap.timeline();
    tl.to(boxes, { opacity: 1, duration: 0.55, ease: "power2.out", stagger: 0.07 });
    tl.to(paths, { strokeDashoffset: 0, duration: 0.6, ease: "power2.inOut", stagger: 0.07 }, 0.15);
  }, "0px 0px -5% 0px");
}

/** Table rows settle in, a few at a time. */
export function settleRows(host: ParentNode, selector = "table.data tbody tr") {
  if (reduced()) return;
  for (const table of $$<HTMLTableElement>("table.data", host)) {
    const rows = $$<HTMLTableRowElement>("tbody tr", table).filter(r => r.matches(selector));
    if (!rows.length || rows.length > 60) continue;
    gsap.set(rows, { opacity: 0, x: -8 });
    onEnter([table], () => gsap.to(rows, { opacity: 1, x: 0, duration: 0.5, ease: "power3.out", stagger: 0.04 }));
  }
}
