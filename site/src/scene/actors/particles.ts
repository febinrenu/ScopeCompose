/*  The particle field: one cloud of light that becomes, in turn, the
 *  wordmark, the shells around the glass, a constellation of real passages,
 *  a starfield, the applicability volumes, and the closing lattice.
 *
 *  Stateless: every frame each particle's position is recomputed from its
 *  target shapes, the weights in WorldParams, a bounded swirl and the
 *  pointer. Nothing is integrated across frames, so the same cursor always
 *  gives the same field, and context loss loses nothing.
 */
import { AdditiveBlending, BufferAttribute, BufferGeometry, Points, ShaderMaterial, Vector3, Vector4 } from "three";
import type { Actor, FrameContext } from "../core/actor";
import type { WorldParams } from "../../acts/keyframes";
import type { Targets } from "../targets";

/** World width the wordmark is built at, and the opening camera distance it is framed from. */
export const WORD_WIDTH = 10.5;
const WORD_VIEW_DIST = 9.5;

const VERT = /* glsl */ `
  attribute vec3 aWord; attribute vec3 aMark; attribute vec3 aNet;
  attribute vec3 aField; attribute vec3 aShell; attribute vec3 aGrid;
  attribute vec4 aSeed;
  uniform vec3 uW0; uniform vec3 uW1;
  uniform float uT; uniform float uTurb; uniform float uPx; uniform float uSize;
  uniform vec4 uSA; uniform vec4 uSB;
  uniform vec3 uPtr; uniform float uPtrK; uniform float uHeat; uniform float uWordFit;
  varying vec3 vCol; varying float vA;

  vec3 swirl(vec3 p, float t){
    return vec3(sin(p.y * 1.3 + t) + cos(p.z * 1.7 - t * 0.7),
                sin(p.z * 1.1 - t * 0.8) + cos(p.x * 1.5 + t * 0.6),
                sin(p.x * 1.2 + t * 0.5) + cos(p.y * 1.4 - t));
  }

  void main(){
    float roleB = step(0.62, aSeed.w);
    vec3 shellPos = mix(uSA.xyz + aShell * uSA.w, uSB.xyz + aShell * uSB.w, roleB);
    // per-particle stagger: each leaves its shape at its own moment
    float e = 1.0 + aSeed.x * 1.8;
    float w0 = pow(uW0.x, e), w1 = pow(uW0.y, e), w2 = pow(uW0.z, e);
    float w3 = pow(uW1.x, e), w4 = pow(uW1.y, e), w5 = pow(uW1.z, e);
    float sum = max(1e-4, w0 + w1 + w2 + w3 + w4 + w5);
    vec3 base = (aWord * uWordFit * w0 + aMark * w1 + aNet * w2 + aField * w3 + shellPos * w4 + aGrid * w5) / sum;
    float dom = max(max(max(w0, w1), max(w2, w3)), max(w4, w5)) / sum;
    // the swirl peaks mid-morph and vanishes on every keyframe
    float amp = uTurb + (1.0 - dom) * 3.2;
    vec3 p = base + swirl(base * 0.55 + aSeed.y * 9.0, uT * 0.22 + aSeed.x * 6.0) * 0.11 * amp;
    // the pointer parts the field like wind
    vec2 d = p.xy - uPtr.xy;
    float dd = length(d);
    p.xy += normalize(d + 1e-4) * uPtrK * pow(max(0.0, 1.0 - dd / 1.5), 2.0) * 0.9;

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;

    vec3 cyan = vec3(0.45, 0.95, 1.0), ivory = vec3(1.0, 0.97, 0.92), amber = vec3(1.0, 0.62, 0.25);
    vec3 cWord = mix(ivory, cyan, aSeed.y * 0.6);
    vec3 cMark = mix(cyan, amber, step(0.74, aSeed.w));
    vec3 cNet = mix(cyan, ivory, aSeed.y * 0.5);
    vec3 cField = ivory * 0.6;
    vec3 cShell = mix(cyan, amber, roleB);
    vec3 cGrid = mix(cyan, mix(amber, cyan, step(0.07, aSeed.w)), step(aSeed.w, 0.2));
    vec3 col = (cWord * w0 * 1.5 + cMark * w1 * 1.9 + cNet * w2 * 1.6 + cField * w3 * 0.55 + cShell * w4 * 1.4 + cGrid * w5 * 1.5) / sum;
    col = mix(col, amber * 1.3, uHeat * 0.6);
    vCol = col * aSeed.z;
    float near = smoothstep(0.25, 1.4, -mv.z);
    vA = near * smoothstep(48.0, 12.0, -mv.z);
    gl_PointSize = clamp(uPx * uSize * (0.55 + aSeed.z * 1.25) * 3.2 / max(0.5, -mv.z), 0.0, 24.0);
  }
`;
const FRAG = /* glsl */ `
  uniform float uAlpha;
  varying vec3 vCol; varying float vA;
  void main(){
    float d = length(gl_PointCoord - 0.5);
    float g = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vCol * g * g * vA * uAlpha, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export class Particles implements Actor {
  readonly group: Points;
  private mat: ShaderMaterial;
  private n: number;
  private ptr = new Vector3(999, 999, 0);
  private ptrK = 0;

  constructor(t: Targets) {
    this.n = t.n;
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(t.field, 3));
    g.setAttribute("aWord", new BufferAttribute(t.word, 3));
    g.setAttribute("aMark", new BufferAttribute(t.mark, 3));
    g.setAttribute("aNet", new BufferAttribute(t.net, 3));
    g.setAttribute("aField", new BufferAttribute(t.field, 3));
    g.setAttribute("aShell", new BufferAttribute(t.shell, 3));
    g.setAttribute("aGrid", new BufferAttribute(t.grid, 3));
    g.setAttribute("aSeed", new BufferAttribute(t.seed, 4));
    this.mat = new ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG,
      uniforms: {
        uW0: { value: new Vector3(1, 0, 0) }, uW1: { value: new Vector3(0, 0, 0) },
        uT: { value: 0 }, uTurb: { value: 0 }, uPx: { value: 1 }, uSize: { value: 1 },
        uSA: { value: new Vector4() }, uSB: { value: new Vector4() },
        uPtr: { value: this.ptr }, uPtrK: { value: 0 }, uHeat: { value: 0 }, uAlpha: { value: 1 }, uWordFit: { value: 1 },
      },
      // Additive, and in the opaque list (transparent: false) so glass refracts it.
      transparent: false, depthWrite: false, blending: AdditiveBlending,
    });
    this.group = new Points(g, this.mat);
    this.group.frustumCulled = false;
  }

  /** Governor step-down: draw a fraction of the points, no reallocation. */
  setDensity(frac: number) { this.group.geometry.setDrawRange(0, Math.floor(this.n * frac)); }

  update(p: WorldParams, ctx: FrameContext) {
    const u = this.mat.uniforms;
    const a = p.pAlpha * (1 - p.dim * 0.45);
    this.group.visible = a > 0.003;
    if (!this.group.visible) return;
    (u.uW0.value as Vector3).set(p.pWord, p.pMark, p.pNet);
    (u.uW1.value as Vector3).set(p.pField, p.pShell, p.pGrid);
    u.uT.value = ctx.t; u.uTurb.value = p.pTurb * ctx.ambient; u.uPx.value = ctx.px;
    // density compensation: fewer particles draw a little larger
    u.uSize.value = Math.sqrt(262144 / this.n) * 0.55 + 0.45;
    (u.uSA.value as Vector4).set(p.sAx, p.sAy, 0, p.sAr * 1.04);
    (u.uSB.value as Vector4).set(p.sBx, p.sBy, 0, p.sBr * 1.06);
    // the pointer only stirs the wordmark and the shells, and eases in and out
    const want = ctx.pointer && ctx.ambient ? (p.pWord + p.pMark + p.pShell * 0.6) : 0;
    this.ptrK += (want - this.ptrK) * 0.15;
    if (ctx.pointer) this.ptr.copy(ctx.pointer);
    u.uPtrK.value = this.ptrK;
    u.uHeat.value = p.pHeat;
    u.uAlpha.value = a;
    // the wordmark fits the frame as seen from the opening camera, on any screen
    const cam = ctx.camera;
    const halfW = Math.tan((cam.fov * Math.PI) / 360) * WORD_VIEW_DIST * cam.aspect;
    u.uWordFit.value = Math.min(1, (2 * halfW * 0.92) / WORD_WIDTH);
  }

  dispose() { this.group.geometry.dispose(); this.mat.dispose(); }
}
