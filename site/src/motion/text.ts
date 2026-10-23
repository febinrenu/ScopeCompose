/*  Type in motion.
 *
 *  - Beats: each heading is split into characters inside line masks; when its
 *    beat turns on the characters rise into place, and when it turns off they
 *    fall back -- so scrolling backwards plays every reveal in reverse.
 *  - Flowing sections: headings reveal once as they enter.
 *  - Numbers decode (ScrambleText) and always finish on their true value.
 *  - Display type leans with scroll speed and springs back to upright at rest.
 *  None of this runs in reduced motion or on narrow screens.
 */
import gsap from "gsap";
import { SplitText } from "gsap/SplitText";
import { ScrambleTextPlugin } from "gsap/ScrambleTextPlugin";
import { $$ } from "../ui/dom";

gsap.registerPlugin(SplitText, ScrambleTextPlugin);

const HEADINGS = ".display, .display-sm, .serif-q";
const timelines = new WeakMap<HTMLElement, gsap.core.Timeline>();

function reveal(el: HTMLElement): gsap.core.Timeline {
  // Screen readers get the sentence once, from a hidden copy, with a real space where a
  // <br> was; the split characters are hidden from them. (An aria-label on a <p> is not
  // allowed, which is what SplitText's own "auto" mode would add.)
  const text = el.innerText.replace(/\s+/g, " ").trim();
  const split = SplitText.create(el, { type: "lines,chars", mask: "lines", linesClass: "sl", charsClass: "sc", aria: "hidden" });
  const sr = document.createElement("span");
  sr.className = "sr-only"; sr.textContent = text;
  el.prepend(sr);
  return gsap.timeline({ paused: true }).from(split.chars, {
    yPercent: 118, rotate: 7, opacity: 0, duration: 0.95, ease: "expo.out", stagger: { each: 0.011, from: "start" },
  });
}

/** Returns a handler for beat state changes. A beat's headings are split the
 *  first time it turns on (splitting every beat at boot was main-thread time
 *  spent on copy nobody had reached yet); a beat that never turned on has
 *  nothing to reverse. */
export function prepareBeats(): (beat: HTMLElement, on: boolean) => void {
  const build = (beat: HTMLElement) => {
    const heads = $$<HTMLElement>(HEADINGS, beat);
    if (!heads.length) return null;
    const tl = gsap.timeline({ paused: true });
    heads.forEach((h, i) => tl.add(reveal(h).play(), i * 0.08));
    tl.progress(0).pause();
    timelines.set(beat, tl);
    return tl;
  };
  return (beat, on) => {
    const tl = timelines.get(beat) ?? (on ? build(beat) : null);
    if (!tl) return;
    if (on) tl.timeScale(1).play(); else tl.timeScale(2.4).reverse();
  };
}

/** Headings in flowing sections reveal once, as they enter. */
export function revealOnEnter(selector: string) {
  const els = $$<HTMLElement>(selector);
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    io.unobserve(e.target);
    const tl = timelines.get(e.target as HTMLElement) ?? reveal(e.target as HTMLElement);
    tl.play();
  }), { rootMargin: "0px 0px -10% 0px" });
  els.forEach(el => { timelines.set(el, reveal(el)); io.observe(el); });
}

/** Numbers decode as they enter, always finishing on their exact text. */
export function scrambleOnEnter(root: ParentNode = document) {
  const els = $$<HTMLElement>("[data-scramble]", root);
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    io.unobserve(e.target);
    const el = e.target as HTMLElement;
    const final = el.dataset.final ?? el.textContent ?? "";
    el.dataset.final = final;
    gsap.to(el, { duration: 1.1, ease: "none", scrambleText: { text: final, chars: "0123456789·–.", speed: 0.55, revealDelay: 0.25 } });
  }), { rootMargin: "0px 0px -8% 0px" });
  els.forEach(el => io.observe(el));
}

/** Display type leans with scroll speed. Returns a setter fed each frame. */
export function velocitySkew(): (v: number) => void {
  const root = document.documentElement;
  const proxy = { s: 0 };
  const to = gsap.quickTo(proxy, "s", { duration: 0.45, ease: "power3.out", onUpdate: () => root.style.setProperty("--skew", proxy.s.toFixed(3) + "deg") });
  let last = 0;
  return v => {
    const target = Math.abs(v) < 0.02 ? 0 : Math.max(-5, Math.min(5, -v * 6));
    if (target !== last) { last = target; to(target); }
  };
}
