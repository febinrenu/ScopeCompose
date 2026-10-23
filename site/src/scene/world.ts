/*  One renderer, one scene, one camera -- for the whole story.
 *
 *  Colour is managed end to end: an opaque canvas, linear HDR inside, bloom
 *  only on what is meant to glow, NEUTRAL tone mapping out (by the post
 *  chain at tiers 2–3, by the renderer at tier 1).
 *
 *  The world renders on demand: when the story cursor moves, the window
 *  resizes, the pointer moves, or (on capable hardware) for ambient drift.
 *  In strict mode (`?still`, tier 1, reduced motion) time is frozen and the
 *  pointer ignored, so a given cursor always gives the identical frame.
 */
import {
  NeutralToneMapping, NoToneMapping, PerspectiveCamera, Plane, Raycaster, Scene, Vector2, Vector3, WebGLRenderer,
  SRGBColorSpace, Color, type Texture,
} from "three";
import { sampleActs } from "../acts/sample";
import type { WorldParams } from "../acts/keyframes";
import { aberration, focusDistance, shake } from "../acts/derive";
import type { Quality } from "../env/tier";
import type { Actor, FrameContext } from "./core/actor";
import { buildEnvironment } from "./render/env";
import { Particles } from "./actors/particles";
import { Hero } from "./actors/hero";
import { Retrieval, type CardData } from "./actors/retrieval";
import { Slabs } from "./actors/slabs";
import { ScopeVolumes } from "./actors/scopeSdf";
import { Engine } from "./actors/engine";
import { Frontier, type FrontierPoint } from "./actors/frontier";
import type { Targets, Node3 } from "./targets";
import type { PanelText } from "./textures";
import type { Slab } from "./fracture";
import type { Post } from "./render/post";
import { HEX } from "./core/color";

export interface WorldInput {
  canvas: HTMLCanvasElement;
  labels: HTMLElement;
  quality: Quality;
  still: boolean;
  reducedMotion: boolean;
  targets: Targets;
  nodes: Node3[];
  cards: CardData[];
  passageA: PanelText;
  passageB: PanelText;
  frontier: FrontierPoint[];
  /** B's pose at the moment it shatters: the shards launch from here. */
  burnFrom: Slab;
  labelText: Record<"kept" | "dropped" | "missing" | "rule" | "condition" | "bm25" | "dense" | "conflict" | "floor", string>;
  makePost: ((r: WebGLRenderer, s: Scene, c: PerspectiveCamera, o: { tier: number; msaa: boolean }) => Post) | null;
  onLost: () => void;
  onRestored: () => void;
}

/** An element's box in the viewport, ignoring its own transform (its offset parent's box plus its offsets). */
function layoutRect(el: HTMLElement): DOMRect {
  const op = el.offsetParent as HTMLElement | null;
  const base = op ? op.getBoundingClientRect() : new DOMRect(-scrollX, -scrollY, 0, 0);
  return new DOMRect(base.left + el.offsetLeft - (op?.scrollLeft ?? 0), base.top + el.offsetTop - (op?.scrollTop ?? 0), el.offsetWidth, el.offsetHeight);
}

/** Drag orbit: soft limit (rad), fastest throw (rad/s), and the spring that brings it home. */
const DRAG_LIMIT = 1.25, MAX_THROW = 3.5, SPRING_K = 4.2, SPRING_C = 2.6;

interface Label { el: HTMLElement; at: (p: WorldParams) => Vector3 | null; alpha: (p: WorldParams) => number }

export class World {
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new PerspectiveCamera(35, 1, 0.1, 160);
  private env: { texture: Texture; dispose: () => void };
  private actors: Actor[] = [];
  private particles: Particles;
  private slabs: Slabs;
  private scope: ScopeVolumes;
  private retrieval: Retrieval;
  private frontier: Frontier;
  private post: Post | null = null;
  private labels: Label[] = [];
  private params: WorldParams;
  private cursor = 0;
  private dirty = true;
  private paused = false;
  private lastAmbient = 0;
  private time = 0;
  private lastTick = 0;
  private velocity = 0;
  private rm: boolean;
  private w = 1; private h = 1;
  private res = new Vector2(1, 1);
  private pointerNdc: Vector2 | null = null;
  /** grab-to-orbit: yaw (rad), its velocity, and the last pointer sample while held */
  private drag = { on: false, x: 0, t: 0, yaw: 0, vel: 0 };
  private ray = new Raycaster();
  private plane = new Plane(new Vector3(0, 0, 1), 0);
  private samples: number[] = [];
  private steps = 0;
  private lostCount = 0;
  private frame = 0;
  private ro: ResizeObserver;
  private copyRects: DOMRect[] = [];
  private firstResolve: (() => void) | null = null;
  readonly firstFrame = new Promise<void>(r => (this.firstResolve = r));

  constructor(private input: WorldInput) {
    const q = input.quality;
    this.rm = input.reducedMotion;
    this.renderer = new WebGLRenderer({
      canvas: input.canvas, antialias: !q.post && q.tier >= 2, alpha: false, powerPreference: "high-performance", stencil: false,
    });
    this.renderer.setClearColor(new Color(HEX.ink), 1);
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = q.post ? NoToneMapping : NeutralToneMapping;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, q.maxDpr));
    // count every pass of a frame (the composer renders several), reset once per frame in tick()
    this.renderer.info.autoReset = false;
    // Checking shader logs reads each program back synchronously, which blocks the main
    // thread on the GPU and defeats parallel compilation. Only under ?debug.
    this.renderer.debug.checkShaderErrors = /[?&]debug\b/.test(location.search);

    this.env = buildEnvironment(this.renderer);

    this.particles = new Particles(input.targets);
    const hero = new Hero(this.env.texture, q.transmission);
    this.retrieval = new Retrieval(input.nodes, input.cards);
    this.slabs = new Slabs(input.passageA, input.passageB, this.env.texture, q.transmission, q.shards, q.embers, input.burnFrom);
    this.scope = new ScopeVolumes(q.steps);
    const engine = new Engine(this.env.texture, q.transmission);
    this.frontier = new Frontier(input.frontier, this.env.texture);
    this.actors = [this.particles, hero, this.retrieval, this.slabs, this.scope, engine, this.frontier];
    for (const a of this.actors) this.scene.add(a.group);

    if (input.makePost) {
      const msaa = Math.min(devicePixelRatio || 1, q.maxDpr) <= 1.25 && q.tier >= 3;
      this.post = input.makePost(this.renderer, this.scene, this.camera, { tier: q.dof ? 3 : 2, msaa });
    }

    this.params = sampleActs(0);
    this.buildLabels();

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(input.canvas);
    this.resize();

    // ?debug: expose the world for inspection from the console
    if (/[?&]debug\b/.test(location.search)) (window as unknown as { __world?: World }).__world = this;

    input.canvas.addEventListener("webglcontextlost", this.onLost, false);
    input.canvas.addEventListener("webglcontextrestored", this.onRestored, false);
  }

  /* ------------------------------------------------------------- labels */

  private label(text: string, tone: string, at: Label["at"], alpha: Label["alpha"]) {
    const el = document.createElement("div");
    el.className = "wl " + tone;
    el.textContent = text;
    el.style.opacity = "0";
    this.input.labels.appendChild(el);
    this.labels.push({ el, at, alpha });
  }

  private buildLabels() {
    const t = this.input.labelText;
    this.label(t.kept, "cyan", p => this.slabs.anchors(p).a, p => p.duel * p.select * (1 - p.socket) * (1 - p.compose) * (1 - p.burn * 0.5));
    this.label(t.dropped, "coral", p => this.slabs.anchors(p).b, p => p.duel * p.crack * Math.max(0, 1 - p.burn * 6) * (1 - p.socket));
    this.label(t.missing, "coral", p => this.slabs.anchors(p).b, p => p.socket * (1 - p.compose) * (1 - p.scope));
    // the scope labels belong to the geometry act; they leave as soon as the shards start home
    const geo = (p: WorldParams) => p.scope * (1 - p.dim) * Math.max(0, 1 - p.compose * 4) * (1 - p.duel);
    this.label(t.rule, "cyan", p => this.scope.anchors(p).a, geo);
    this.label(t.condition, "amber", p => this.scope.anchors(p).b, geo);
    this.label(t.bm25, "cyan", () => this.retrieval.beamMid.lexical, p => p.bm25 * (1 - p.cards));
    this.label(t.dense, "violet", () => this.retrieval.beamMid.dense, p => p.dense * (1 - p.cards));
    this.label(t.conflict, "coral", () => this.frontier.anchors()[0], p => p.frontier);
    this.label(t.floor, "amber", () => this.frontier.anchors()[1], p => p.frontier);
  }

  /** Copy blocks of the visible act: labels never sit on top of text.
   *  Layout boxes, not transformed ones: a beat sliding in must not change the
   *  answer, or a label's visibility would depend on when the frame was drawn. */
  private readCopyRects() {
    this.copyRects = Array.from(document.querySelectorAll<HTMLElement>("#story .beat.on, #story .act-tag, #story .geo, #story .truth, #story .fv, #story .track, .act-frontier .wrap > *"))
      .map(layoutRect).filter(r => r.width > 0 && r.bottom > 0 && r.top < this.h);
  }

  private placeLabels() {
    const v = new Vector3();
    let read = false;
    for (const l of this.labels) {
      const a = this.paused ? 0 : l.alpha(this.params);
      if (a < 0.02) { if (l.el.style.opacity !== "0") l.el.style.opacity = "0"; continue; }
      if (!read) { this.readCopyRects(); read = true; }
      const at = l.at(this.params);
      if (!at) continue;
      v.copy(at).project(this.camera);
      const lw = l.el.offsetWidth + 14, lh = 18;
      // kept on screen: a label is pushed in from the edges rather than cut off
      const x = Math.min(this.w - lw - 12, Math.max(12, (v.x + 1) / 2 * this.w)), y = Math.max(64, (1 - v.y) / 2 * this.h);
      const hit = v.z > 1 || this.copyRects.some(r => x < r.right && x + lw > r.left && y - lh < r.bottom && y + lh > r.top);
      l.el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
      l.el.style.opacity = hit ? "0" : a.toFixed(3);
    }
  }

  /* ------------------------------------------------------------- control */

  setCursor(c: number) {
    if (Math.abs(c - this.cursor) < 1e-5) return;
    this.cursor = c;
    this.params = sampleActs(c);
    this.dirty = true;
  }

  /** Pointer in normalised device coordinates (y up), or null when it leaves. */
  setPointer(ndc: { x: number; y: number } | null) {
    if (this.input.still || this.rm || !this.input.quality.pointer) return;
    this.pointerNdc = ndc ? (this.pointerNdc ?? new Vector2()).set(ndc.x, ndc.y) : null;
    const d = this.drag;
    if (d.on && ndc) {
      const now = performance.now() / 1000, dt = Math.max(1 / 240, now - d.t);
      // past the soft limit the glass resists, like pulling against a spring
      const give = Math.abs(d.yaw) > DRAG_LIMIT ? 0.3 : 1;
      const dy = (ndc.x - d.x) * 1.8 * give;
      d.yaw += dy;
      d.vel = d.vel * 0.6 + (dy / dt) * 0.4;     // smoothed, so a flick reads as a flick
      d.x = ndc.x; d.t = now;
    }
    this.dirty = true;
  }

  /** Grab-to-orbit (hero and scope acts). On release the orbit keeps the throw's
   *  momentum and swings home on a lightly underdamped spring. */
  setDrag(on: boolean, ndcX = 0) {
    if (this.input.still || this.rm) return;
    const d = this.drag;
    d.on = on;
    if (on) { d.x = ndcX; d.t = performance.now() / 1000; d.vel = 0; }
    // held still before letting go: no throw
    else d.vel = performance.now() / 1000 - d.t > 0.1 ? 0 : Math.max(-MAX_THROW, Math.min(MAX_THROW, d.vel));
    this.dirty = true;
  }

  setVelocity(v: number) {
    const nv = this.input.still || this.rm ? 0 : v;
    if (nv !== this.velocity) { this.velocity = nv; this.dirty = true; }
  }

  setReducedMotion(rm: boolean) { this.rm = rm; this.dirty = true; }

  pause(p: boolean) {
    this.paused = p;
    if (p) for (const l of this.labels) l.el.style.opacity = "0";
    this.dirty = true;
  }

  get sceneAlpha() { return this.params.sceneAlpha * (1 - this.params.dim * 0.5); }

  private resize() {
    const c = this.input.canvas;
    this.w = c.clientWidth || innerWidth; this.h = c.clientHeight || innerHeight;
    this.renderer.setSize(this.w, this.h, false);
    const px = this.renderer.getPixelRatio();
    this.res.set(this.w * px, this.h * px);
    this.post?.setSize(this.w, this.h);
    const aspect = this.w / this.h;
    this.camera.aspect = aspect;
    this.camera.fov = aspect < 1 ? Math.min(62, 35 / Math.pow(aspect, 0.75)) : 35;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  /* ------------------------------------------------------------- frame */

  tick(nowSec: number): boolean {
    if (this.paused || (this.lostCount > 0 && this.renderer.getContext().isContextLost())) return false;
    const q = this.input.quality;
    const strict = this.input.still || this.rm;
    const ambient = !strict && q.ambientFps > 0;
    const dt = this.lastTick ? nowSec - this.lastTick : 0;
    this.lastTick = nowSec;
    let render = this.dirty;
    if (ambient && nowSec - this.lastAmbient >= 1 / q.ambientFps) { render = true; this.lastAmbient = nowSec; }
    const d = this.drag;
    if (!d.on && (Math.abs(d.yaw) > 1e-4 || Math.abs(d.vel) > 1e-4)) {
      // momentum, then home: a damped spring, integrated in small steps so a long frame stays stable
      for (let left = Math.min(dt, 0.1); left > 0; left -= 1 / 120) {
        const h = Math.min(left, 1 / 120);
        d.vel += (-SPRING_K * d.yaw - SPRING_C * d.vel) * h;
        d.yaw += d.vel * h;
      }
      if (Math.abs(d.yaw) < 1e-4 && Math.abs(d.vel) < 1e-3) { d.yaw = 0; d.vel = 0; }
      render = true;
    } else if (d.on) render = true;
    if (!render) return false;
    if (ambient) this.time += Math.min(dt, 0.1);

    const p = this.params;
    const aspect = this.w / this.h;
    const pf = aspect < 0.9 ? 1 : aspect < 1.3 ? 0.55 : 0;
    const shift = (p.subX - p.tgtX) * pf;
    const [sx, sy] = shake(p);
    // orbit the subject by the drag yaw
    const tx = p.tgtX + shift, ty = p.tgtY, tz = p.tgtZ;
    // squarer screens dolly back a little so a frame composed at 16:10 still fits across
    const dolly = aspect >= 0.9 && aspect < 1.6 ? 1 + (1.6 / aspect - 1) * 0.85 : 1;
    let cx = (p.camX + shift - tx) * dolly, cz = (p.camZ + (aspect < 0.9 ? 1.2 : 0) - tz) * dolly;
    const cyaw = Math.cos(d.yaw), syaw = Math.sin(d.yaw);
    [cx, cz] = [cx * cyaw - cz * syaw, cx * syaw + cz * cyaw];
    this.camera.position.set(tx + cx + sx, ty + (p.camY - ty) * dolly + sy, tz + cz);
    this.camera.lookAt(tx, ty, tz);
    this.camera.updateMatrixWorld();

    // pointer on the subject plane; the x-ray lens follows it or parks on the socket
    let pointer: Vector3 | null = null;
    if (this.pointerNdc) {
      this.ray.setFromCamera(this.pointerNdc, this.camera);
      this.plane.constant = -tz;
      pointer = this.ray.ray.intersectPlane(this.plane, new Vector3());
    }
    const lens = { x: 0, y: 0, r: 0.3 };
    if (this.pointerNdc) { lens.x = this.pointerNdc.x; lens.y = this.pointerNdc.y; }
    else { const s = this.slabs.socketCentre(p).project(this.camera); lens.x = s.x; lens.y = s.y; }

    const ctx: FrameContext = {
      t: this.time, ambient: ambient ? 1 : 0, pointer, lens, resolution: this.res,
      px: this.renderer.getPixelRatio(), camera: this.camera, quality: q, env: this.env.texture,
      spin: { yaw: d.yaw, vel: d.on ? 0 : d.vel },
    };
    for (const a of this.actors) a.update(p, ctx);

    const t0 = performance.now();
    this.renderer.info.reset();
    if (this.post) {
      this.post.set({ bloom: p.bloom, dof: p.dof, focus: focusDistance(p) * dolly, vign: p.vign, ca: aberration(p), vel: this.velocity, exposure: p.exposure });
      this.post.render();
    } else {
      this.renderer.toneMappingExposure = p.exposure;
      this.renderer.render(this.scene, this.camera);
    }
    this.placeLabels();
    this.govern(dt, performance.now() - t0);
    this.dirty = Math.abs(this.velocity) > 1e-4;      // keep drawing while the frame is warped
    this.frame++;
    document.documentElement.dataset.frame = String(this.frame);
    if (this.firstResolve) { this.firstResolve(); this.firstResolve = null; }
    return true;
  }

  /** The lens, for the DOM ring: where it is on screen (NDC). */
  lensNdc() { return this.pointerNdc; }

  /** One-way step-down ladder when frames run long. */
  private govern(dt: number, _cpu: number) {
    // ?still promises a frame that depends on the scroll position alone, so quality never moves
    if (this.input.still || this.steps >= 3 || dt <= 0 || dt > 0.25) return;
    this.samples.push(dt * 1000);
    if (this.samples.length < 120) return;
    const s = this.samples.slice().sort((a, b) => a - b);
    const p90 = s[Math.floor(s.length * 0.9)];
    this.samples = [];
    if (p90 <= 24) return;
    this.steps++;
    if (this.steps === 1) { this.renderer.setPixelRatio(1); this.resize(); }
    if (this.steps === 2) { this.post?.dropDof(); this.particles.setDensity(0.5); }
    if (this.steps === 3) { this.particles.setDensity(0.25); this.scope.setSteps(24); }
    document.documentElement.dataset.governor = "step-" + this.steps;
  }

  async warm() {
    // compile every program up front, so nothing stalls mid-scroll
    const vis = this.actors.map(a => a.group.visible);
    for (const a of this.actors) a.group.visible = true;
    // With post-processing the scene renders into a linear half-float buffer, which selects
    // different program variants from the canvas: compile for the target actually used.
    const rt = this.renderer.getRenderTarget();
    if (this.post) this.renderer.setRenderTarget(this.post.target);
    try { await this.renderer.compileAsync(this.scene, this.camera); } catch { /* compile on first draw instead */ }
    this.renderer.setRenderTarget(rt);
    this.actors.forEach((a, i) => (a.group.visible = vis[i]));
  }

  stats() {
    const i = this.renderer.info;
    return { calls: i.render.calls, triangles: i.render.triangles, points: i.render.points, px: this.renderer.getPixelRatio(), stepped: this.steps > 0 };
  }

  loseContextForTest() {
    const ext = this.renderer.getContext().getExtension("WEBGL_lose_context");
    if (!ext) return;
    ext.loseContext();
    setTimeout(() => ext.restoreContext(), 600);
  }

  private onLost = (e: Event) => {
    e.preventDefault();
    this.lostCount++;
    this.paused = true;
    this.input.onLost();
  };

  private onRestored = () => {
    if (this.lostCount >= 2) return;           // a second loss keeps the poster for the session
    // The reflection map was a render target: its pixels died with the context.
    const old = this.env.texture;
    this.env = buildEnvironment(this.renderer);
    this.scene.traverse(o => {
      const m = (o as unknown as { material?: { envMap?: Texture | null; needsUpdate: boolean } }).material;
      if (m && m.envMap === old) { m.envMap = this.env.texture; m.needsUpdate = true; }
    });
    old.dispose();
    this.paused = false;
    this.dirty = true;
    this.input.onRestored();
  };

  dispose() {
    this.ro.disconnect();
    this.input.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.input.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    for (const a of this.actors) a.dispose();
    for (const l of this.labels) l.el.remove();
    this.post?.dispose();
    this.env.dispose();
    this.renderer.dispose();
  }
}
