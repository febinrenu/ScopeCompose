/*  ScopeCompose — boot.
 *
 *  Order matters and is deliberate:
 *    1. the verified engines and the corpus must be present (they are
 *       classic scripts that ran before this module);
 *    2. every number on the page is computed and written into the DOM, so
 *       the content is complete before any motion starts;
 *    3. the scroll system and the type motion start;
 *    4. only then, and only on capable hardware, the 3D world is fetched.
 *  If anything after step 2 fails, the page is still a complete, readable
 *  research document.
 */
import "./styles/main.css";
import gsap from "gsap";
import { assertEngine } from "./engine/bridge";
import { corpus } from "./data/corpus";
import { loader } from "./ui/loader";
import { reducedMotion, prefs, watchReducedMotion } from "./env/prefs";
import { $, $$ } from "./ui/dom";
import { StoryScroll } from "./motion/scroll";
import { prepareBeats, revealOnEnter, scrambleOnEnter, velocitySkew } from "./motion/text";
import { Nav } from "./ui/nav";
import { Cursor } from "./ui/cursor";
import { Portal } from "./ui/portal";
import { initHover } from "./ui/hover";
import { renderStory } from "./demo/story";
import { RelationExplorer, RELATIONS } from "./demo/relations";
import { renderStations } from "./demo/stations";
import { Reasoner } from "./demo/reasoner";
import { renderResearch } from "./demo/research";
import { renderFrontier } from "./demo/frontier";
import { anchorTarget, startRouter } from "./lab/router";
import { Ambience } from "./ui/audio";
import { classifyTier, probeDevice, qualityFor, type Tier } from "./env/tier";
import { sampleActs } from "./acts/sample";
import { KEYFRAMES, RELATION_STOPS, stopIndex } from "./acts/keyframes";
import { stationIndex } from "./acts/derive";
import { FEE } from "./data/examples";
import { detectorCorpusStats } from "./engine/pipeline";
import { TESTS } from "./data/facts";
import type { World } from "./scene/world";

const html = document.documentElement;

function fatal(err: unknown) {
  const msg = err instanceof Error ? err.message : String(err);
  loader.fail("could not start: " + msg);
  const p = document.createElement("p");
  p.setAttribute("role", "alert");
  p.style.cssText = "position:relative;z-index:50;margin:90px 20px 0;padding:14px 16px;border:1px solid #ff6a5c;color:#ff6a5c;font:14px/1.5 system-ui";
  p.textContent = msg;
  document.body.prepend(p);
  setTimeout(() => loader.done(), 1500);
}

/** Give the browser a turn between chunks of boot work, so no task runs long. */
const yieldToMain = () => new Promise<void>(r => {
  const s = (globalThis as unknown as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
  if (s?.yield) s.yield().then(r); else setTimeout(r, 0);
});

/** Five real benchmark passages for the retrieval cards (the ranking drawn is illustrative). */
function retrievalCards() {
  const seen = new Set<string>();
  const out: { id: string; text: string }[] = [];
  for (const r of corpus()) for (const p of r.p) {
    if (out.length >= 5 || seen.has(p.t) || !/\b(fee|fees|charge|charged)\b/i.test(p.t) || p.t.length < 40) continue;
    seen.add(p.t);
    const t = p.t.length > 110 ? p.t.slice(0, 108).replace(/\s+\S*$/, "") + "…" : p.t;
    out.push({ id: `${r.id} · ${p.id}`, text: t });
  }
  return out;
}

async function main() {
  const motion = () => !reducedMotion() && innerWidth > 760;
  loader.total = 6;
  loader.step("verifying the engines");
  assertEngine();
  corpus();

  // ---- 2. content: every figure computed before any motion ---------------
  loader.step("reading the archive");
  const papers = $$<HTMLElement>(".paper");
  const underBar = new Set<Element>();
  const barIO = new IntersectionObserver(es => {
    es.forEach(e => e.isIntersecting ? underBar.add(e.target) : underBar.delete(e.target));
    html.classList.toggle("bar-paper", underBar.size > 0 && !html.classList.contains("lab-on"));
  }, { rootMargin: "0px 0px -94% 0px" });
  papers.forEach(p => barIO.observe(p));
  const storyFacts = renderStory();
  const stationEls = renderStations();
  await yieldToMain();
  new Reasoner($("#reasoner")!);
  await yieldToMain();
  renderResearch();
  await yieldToMain();
  renderFrontier();
  await yieldToMain();
  const tests = $("#colTests");
  if (tests) tests.textContent = `${TESTS.value} Python tests passing at the last recorded run`;

  // ---- 3. scroll and type motion -----------------------------------------
  let rm = reducedMotion();
  const story = new StoryScroll(rm);
  if (motion()) {
    const onBeat = prepareBeats();
    story.onBeat(onBeat);
    revealOnEnter(".flow > .wrap > .display");
    scrambleOnEnter();
  } else {
    $$(".reveal").forEach(el => el.classList.add("in"));
  }
  story.init();
  loader.step("choreographing the story");

  let world: World | null = null;
  let cursor = 0;
  let worldPaused = false;
  let act = 0;
  const ambience = new Ambience();
  const skew = motion() ? velocitySkew() : () => {};
  const cur = !rm && matchMedia("(pointer: fine)").matches && !prefs.still ? new Cursor() : null;

  const relations = new RelationExplorer(rel => {
    if (rm) { cursor = RELATION_STOPS[rel]; world?.setCursor(cursor); applyScene(cursor); }
    else story.scrollToStop(4, RELATION_STOPS[rel]);
  });

  const nav = new Nav(story, href => {
    const t = anchorTarget(href);
    if (t) story.scrollToEl(t);
  });

  const track = $("#stations");
  const trackProg = $("#trackProg");
  let lastStation = -1, lastAct = -1, lastY = scrollY;

  function applyScene(c: number) {
    const p = sampleActs(c);
    html.style.setProperty("--scene-alpha", (p.sceneAlpha * (1 - p.dim * 0.5)).toFixed(3));
    return p;
  }

  const portal = new Portal($("#world-wrap")!);
  initHover();
  story.onChange(s => {
    cursor = s.cursor; act = s.act;
    portal.set(s.cursor, !rm);
    nav.update(s);
    world?.setCursor(s.cursor);
    const p = applyScene(s.cursor);
    html.dataset.act = String(s.act);
    // act 04: the panel follows the geometry
    if (s.act === 4) {
      const k = Math.max(0, Math.min(3, Math.round(s.cursor - RELATION_STOPS.refinement)));
      relations.follow(RELATIONS[k]);
    }
    // act 05: the horizontal journey -- the track is a function of the cursor
    if (track && (s.act === 5 || lastStation !== -1)) {
      const f = s.act === 5 ? p.flow : s.act > 5 ? 1 : 0;
      const card = stationEls[0];
      const stride = card ? card.offsetWidth + 24 : 0;
      if (!rm && innerWidth > 760) track.style.transform = `translate3d(${(-f * 6 * stride).toFixed(1)}px, 0, 0)`;
      if (trackProg) trackProg.style.transform = `scaleX(${f.toFixed(3)})`;
      const idx = s.act === 5 ? stationIndex(p.flow) : -1;
      if (idx !== lastStation) { stationEls.forEach((el, i) => el.classList.toggle("on", i === idx)); lastStation = idx; }
    }
    cur?.setMode(s.act === 0 || s.act === 4 ? "drag" : s.act === 3 ? "xray" : "none");
    if (s.act !== lastAct) { if (lastAct !== -1) ambience.tick(s.act === 2); lastAct = s.act; }
    ambience.velocity(scrollY - lastY); lastY = scrollY;
  });

  // scroll speed bends the frame and leans the type; both settle to zero at rest
  gsap.ticker.add(() => { const v = story.velocity(); skew(v); world?.setVelocity(v); });

  // ---- routing ------------------------------------------------------------
  startRouter({
    enterLab: () => { story.setEnabled(false); world?.pause(true); worldPaused = true; },
    leaveLab: () => { story.setEnabled(true); world?.pause(false); worldPaused = false; },
    anchor: target => story.scrollToEl(target),
  });

  // ---- sound --------------------------------------------------------------
  const sb = $<HTMLButtonElement>("#soundBtn")!;
  sb.addEventListener("click", async () => {
    const on = await ambience.toggle();
    sb.setAttribute("aria-pressed", String(on));
    sb.setAttribute("aria-label", on ? "Sound on. Turn ambient sound off" : "Sound off. Turn ambient sound on");
  });

  // ---- 4. the world ---------------------------------------------------------
  let tier: Tier = 0;
  const forced = prefs.quality;
  if (forced !== "off") {
    const probe = probeDevice(document.createElement("canvas"), { antialias: false });
    tier = classifyTier(probe.info);
    probe.gl?.getExtension("WEBGL_lose_context")?.loseContext();
    if (forced === "low") tier = probe.info.webgl2 ? 1 : 0;
    if (forced === "mid") tier = probe.info.webgl2 ? 2 : 0;
    if (forced === "high") tier = probe.info.webgl2 ? 3 : 0;
  }
  html.dataset.tier = String(tier);

  if (tier > 0) {
    loader.step("building the world");
    try {
      const { bootWorld } = await import("./scene/boot");
      const crack = KEYFRAMES[stopIndex("collision.crack")];
      const det = detectorCorpusStats();
      world = await bootWorld({
        canvas: $<HTMLCanvasElement>("#world")!,
        labels: $("#labels")!,
        quality: qualityFor(tier),
        still: prefs.still,
        reducedMotion: rm,
        nodeCount: corpus().length,
        cards: retrievalCards(),
        passageA: { tag: "Passage A · general rule · illustrative", text: "“" + FEE.p[0].t + "”", meta: `${FEE.p[0].id} · ${FEE.p[0].st?.replace(/_/g, " ")}`, accent: "#7ff0ff" },
        passageB: { tag: "Passage B · conditional exception · illustrative", text: "“" + FEE.p[1].t + "”", meta: `${FEE.p[1].id} · ${FEE.p[1].st?.replace(/_/g, " ")}`, accent: "#ffb45c" },
        frontier: det.points,
        burnFrom: { x: crack.bX, y: crack.bY, s: crack.bS },
        labelText: {
          kept: `kept · ${storyFacts.winner} · ${storyFacts.winnerType}`,
          dropped: `about to be discarded · ${storyFacts.loserType}`,
          missing: "missing · the premium-tier branch",
          rule: "rule · applicability region",
          condition: "condition · applicability region",
          bm25: "BM25 · lexical",
          dense: "dense · semantic",
          conflict: "conflict threshold 0.50",
          floor: "escalation floor 0.35",
        },
        onLost: () => html.classList.add("gl-lost"),
        onRestored: () => html.classList.remove("gl-lost"),
      });
      world.setCursor(cursor);
      if (worldPaused) world.pause(true);
      gsap.ticker.add(t => { world?.tick(t); });
      await world.firstFrame;
      loader.step("first light");
      html.classList.add("gl", "gl-ready");
      // pointer: stirs the light, steers the x-ray lens; drag turns the glass
      if (!rm && !prefs.still && qualityFor(tier).pointer) {
        const ndc = (e: PointerEvent) => ({ x: e.clientX / innerWidth * 2 - 1, y: -(e.clientY / innerHeight * 2 - 1) });
        addEventListener("pointermove", e => { if (e.pointerType === "mouse") world?.setPointer(ndc(e)); }, { passive: true });
        document.addEventListener("pointerleave", () => world?.setPointer(null));
        addEventListener("pointerdown", e => {
          const t = e.target as HTMLElement;
          if ((act !== 0 && act !== 4) || t.closest("a, button, input, select, textarea, label, summary, [role=tab], .geo, .truth")) return;
          world?.setDrag(true, ndc(e).x);
        });
        addEventListener("pointerup", () => world?.setDrag(false));
      }
      document.addEventListener("visibilitychange", () => world?.pause(document.hidden || worldPaused));
      // Past the closing act the world is hidden behind the colophon: stop drawing it.
      const col = $("#colophon");
      if (col) new IntersectionObserver(es => {
        world?.pause(es[0].intersectionRatio > 0.85 || worldPaused);
      }, { threshold: [0, 0.85, 1] }).observe(col);
      if (prefs.debug) {
        const { mountHud } = await import("./debug/hud");
        mountHud(() => world?.stats() ?? null, tier, () => cursor, () => world?.loseContextForTest());
      }
    } catch (err) {
      console.error("3D world unavailable; continuing without it.", err);
      html.classList.add("no-gl");
    }
  } else {
    html.classList.add("no-gl");
  }
  finishIntro(rm);

  watchReducedMotion(next => {
    rm = next;
    story.setMotion(next);
    world?.setReducedMotion(next);
  });
}

/** The curtain lifts; without the world, the DOM wordmark rises instead. */
function finishIntro(rm: boolean) {
  loader.done();
  if (rm) { $$(".reveal").forEach(el => el.classList.add("in")); return; }
  const delay = html.classList.contains("seen") ? 0.15 : 0.45;
  if (!html.classList.contains("gl-ready")) {
    gsap.fromTo($$<HTMLElement>(".mega-line > span"), { "--rise": "108%" }, { "--rise": "0%", duration: 1.5, ease: "expo.out", stagger: 0.12, delay });
  }
  setTimeout(() => $$(".reveal").forEach(el => el.classList.add("in")), (delay + 0.6) * 1000);
}

function start() {
  main().catch(fatal);
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
else start();
