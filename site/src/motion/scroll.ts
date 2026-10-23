/*  The story cursor.
 *
 *  One number drives the world: which keyframe the reader is at. Each act's
 *  section scrolls the cursor across its span of keyframes. It is computed
 *  from every section's ScrollTrigger progress on each tick -- never
 *  accumulated -- so jumps, anchors and reverse scrolling all land exactly.
 *
 *  Lenis is the only smoother, and only when motion is allowed. Native
 *  scrolling is never blocked; if Lenis fails, the page still scrolls.
 */
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { ACT_FROM, LAST_STOP, stopIndex } from "../acts/keyframes";
import { cursorFor } from "../acts/sample";
import { $$ } from "../ui/dom";

gsap.registerPlugin(ScrollTrigger);
// ScrollTrigger's own refresh on "load" is replaced by the single one below, after fonts and load.
ScrollTrigger.config({ autoRefreshEvents: "visibilitychange,DOMContentLoaded,resize" });

export interface Section {
  el: HTMLElement;
  act: number;
  name: string;
  k0: number;
  k1: number;
  st: ScrollTrigger;
  /** Fraction of the section's scroll during which its frame is pinned. */
  hold: number;
  beats: { el: HTMLElement; a: number; b: number; state: number }[];
}

export interface StoryState { cursor: number; act: number; actProgress: number }

export class StoryScroll {
  readonly sections: Section[] = [];
  lenis: Lenis | null = null;
  private listeners: ((s: StoryState) => void)[] = [];
  private beatListeners: ((beat: HTMLElement, on: boolean) => void)[] = [];
  private last: StoryState = { cursor: -1, act: -1, actProgress: -1 };
  private enabled = true;

  constructor(private rm: boolean) {}

  init() {
    const els = $$<HTMLElement>("#story > .act");
    els.forEach((el, i) => {
      const act = Number(el.dataset.act);
      const k0 = stopIndex(ACT_FROM[act]);
      // the last act has no successor: it walks all of its own stops while pinned
      // (one past the end, so the final stop is reached; sampling clamps)
      const next = els[i + 1] ? stopIndex(ACT_FROM[Number(els[i + 1].dataset.act)]) : LAST_STOP + 1;
      const sec = { el, act, name: el.dataset.name || "", k0, k1: next, hold: 0.7, beats: [] } as unknown as Section;
      this.sections.push(sec);
      // From "top at top" to "bottom at top": the pinned phase, then the
      // scroll-out. Sections are contiguous, so there is never a dead zone.
      const st = ScrollTrigger.create({
        trigger: el, start: "top top", end: "bottom top",
        onRefresh: self => { sec.hold = Math.max(0, Math.min(1, 1 - innerHeight / Math.max(1, self.end - self.start))); },
      });
      const beats = $$<HTMLElement>(".beat", el).map(b => ({
        el: b, a: Number(b.dataset.in ?? 0), b: Number(b.dataset.out ?? 1.01), state: -1,
      }));
      sec.st = st; sec.beats = beats;
    });
    this.setMotion(this.rm);
    gsap.ticker.lagSmoothing(0);
    gsap.ticker.add(this.update);
    // one refresh once the layout is final (each refresh forces a full layout)
    const loaded = document.readyState === "complete" ? Promise.resolve() : new Promise(r => addEventListener("load", r, { once: true }));
    Promise.all([document.fonts?.ready, loaded]).then(() => ScrollTrigger.refresh());
  }

  setMotion(rm: boolean) {
    this.rm = rm;
    if (!rm && !this.lenis) {
      try {
        this.lenis = new Lenis({ autoRaf: false, lerp: 0.09, wheelMultiplier: 0.95, syncTouch: false, anchors: false });
        this.lenis.on("scroll", ScrollTrigger.update);
        gsap.ticker.add(this.raf);
      } catch { this.lenis = null; }
    } else if (rm && this.lenis) {
      gsap.ticker.remove(this.raf);
      this.lenis.destroy();
      this.lenis = null;
    }
    // Section heights differ between modes (sticky frames vs flowing copy).
    requestAnimationFrame(() => ScrollTrigger.refresh());
  }

  private raf = (time: number) => { this.lenis?.raf(time * 1000); };

  onChange(fn: (s: StoryState) => void) { this.listeners.push(fn); }
  onBeat(fn: (beat: HTMLElement, on: boolean) => void) { this.beatListeners.push(fn); }

  /** Scroll speed, normalised to about -1..1, exactly 0 at rest. */
  velocity(): number {
    const v = this.lenis?.velocity ?? 0;
    return Math.abs(v) < 0.4 ? 0 : Math.max(-1, Math.min(1, v / 60));
  }

  /** Pause the story's cursor (lab views). Scrolling itself is never locked. */
  setEnabled(on: boolean) {
    this.enabled = on;
    if (on) requestAnimationFrame(() => ScrollTrigger.refresh());
  }

  /**
   * While a section is pinned, the cursor walks its own keyframes
   * (k0 .. k1-1); while it scrolls out, it travels to the next act's first
   * keyframe. So an act's world never starts changing into the next act
   * while that act's copy is still on screen.
   */
  cursorOf(sec: Section, p: number): number { return cursorFor(sec.k0, sec.k1, sec.hold, p); }

  /** Progress through the pinned phase only: what the copy's beats follow. */
  pinned(sec: Section, p = sec.st.progress) { return sec.hold > 0 ? Math.min(1, p / sec.hold) : p; }

  /** Cursor = deepest section that has begun, at its progress. */
  compute(): StoryState {
    let cursor = 0, act = 0, actProgress = 0;
    for (const s of this.sections) {
      const p = s.st.progress;
      if (s === this.sections[0] || p > 0) {
        cursor = this.cursorOf(s, p);
        act = s.act; actProgress = this.pinned(s, p);
      }
    }
    if (this.rm) cursor = Math.round(cursor);
    return { cursor, act, actProgress };
  }

  private update = () => {
    if (!this.enabled || !this.sections.length) return;
    const s = this.compute();
    if (!this.rm) this.updateBeats();
    if (s.cursor !== this.last.cursor || s.act !== this.last.act || s.actProgress !== this.last.actProgress) {
      this.last = s;
      for (const fn of this.listeners) fn(s);
    }
  };

  private updateBeats() {
    for (const sec of this.sections) {
      if (!sec.beats.length) continue;
      const p = this.pinned(sec);
      // classes are only touched when a beat changes state, so idle sections cost nothing
      for (const b of sec.beats) {
        const state = p < b.a ? 0 : p >= b.b ? 2 : 1;
        if (state === b.state) continue;
        b.state = state;
        b.el.classList.toggle("on", state === 1);
        b.el.classList.toggle("past", state === 2);
        for (const fn of this.beatListeners) fn(b.el, state === 1);
      }
    }
  }

  /** Scroll so the cursor lands on keyframe `stop` inside act `act`. */
  scrollToStop(act: number, stop: number) {
    const s = this.sections.find(x => x.act === act);
    if (!s) return;
    const inner = Math.max(1e-6, s.k1 - 1 - s.k0);
    const p = Math.max(0, Math.min(1, (stop - s.k0) / inner)) * s.hold;
    const y = s.st.start + p * (s.st.end - s.st.start);
    if (this.lenis) this.lenis.scrollTo(y, { duration: 1.4 });
    else scrollTo({ top: y, behavior: "auto" });
  }

  scrollToEl(el: HTMLElement, immediate = false) {
    if (this.lenis) this.lenis.scrollTo(el, { duration: immediate ? 0 : 1.6, immediate });
    else el.scrollIntoView({ behavior: "auto" });
  }

  refresh() { ScrollTrigger.refresh(); }
}
