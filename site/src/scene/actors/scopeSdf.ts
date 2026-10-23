/*  Act 04 / 07: applicability regions as liquid glass.
 *
 *  Raymarched inside a tight bounding box. The rule's region (A, cyan) and
 *  the condition's (B, amber) are signed-distance spheres combined by the
 *  relation:
 *    refinement  B nested in A, glowing through it
 *    disjoint    apart
 *    redundant   smooth-min: they melt into one shape, the colours merge
 *    opposed     where both hold, a coral lens pulses -- the real conflict
 *  Overlap of applicability and disagreement of outcome are separate
 *  uniforms, because the project decides them separately.
 */
import { BackSide, BoxGeometry, Mesh, ShaderMaterial, Vector3, Vector4 } from "three";
import type { Actor, FrameContext } from "../core/actor";
import type { WorldParams } from "../../acts/keyframes";

const VERT = /* glsl */ `
  varying vec3 vWorld;
  void main(){
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;

const FRAG = /* glsl */ `
  uniform vec4 uA; uniform vec4 uB;
  uniform float uFuse; uniform float uOppose; uniform float uNest; uniform float uAlpha; uniform float uT;
  uniform vec3 uBoxMin; uniform vec3 uBoxMax; uniform int uSteps;
  varying vec3 vWorld;

  float smin(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
  float dA(vec3 p){ return length(p - uA.xyz) - uA.w; }
  float dB(vec3 p){ return length(p - uB.xyz) - uB.w; }
  float field(vec3 p){
    float a = dA(p), b = dB(p);
    return mix(min(a, b), smin(a, b, 0.65), uFuse);
  }
  vec3 nrm(vec3 p){
    vec2 e = vec2(0.002, 0.0);
    return normalize(vec3(field(p + e.xyy) - field(p - e.xyy), field(p + e.yxy) - field(p - e.yxy), field(p + e.yyx) - field(p - e.yyx)));
  }
  vec2 box(vec3 ro, vec3 rd){
    vec3 inv = 1.0 / rd;
    vec3 t0 = (uBoxMin - ro) * inv, t1 = (uBoxMax - ro) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
  }
  // a cheap studio for reflections: cool top, cyan left, amber right
  vec3 studio(vec3 r){
    vec3 c = vec3(0.02, 0.025, 0.04);
    c += vec3(0.55, 0.95, 1.0) * pow(max(0.0, -r.x), 6.0) * 2.0;
    c += vec3(1.0, 0.62, 0.28) * pow(max(0.0, r.x), 6.0) * 1.6;
    c += vec3(0.85, 0.9, 1.0) * pow(max(0.0, r.y), 10.0) * 1.8;
    return c;
  }

  void main(){
    vec3 ro = cameraPosition;
    vec3 rd = normalize(vWorld - ro);
    vec2 tb = box(ro, rd);
    float t = max(tb.x, 0.0), tEnd = tb.y;
    if (tEnd <= t) discard;
    vec3 cyan = vec3(0.45, 0.95, 1.0), amber = vec3(1.0, 0.6, 0.24), coral = vec3(1.0, 0.36, 0.3);

    // sphere-trace to the surface
    float hit = -1.0;
    for (int i = 0; i < 64; i++){
      if (i >= uSteps) break;
      vec3 p = ro + rd * t;
      float d = field(p);
      if (d < 0.0015){ hit = t; break; }
      t += max(d * 0.9, 0.004);
      if (t > tEnd) break;
    }
    if (hit < 0.0) discard;

    vec3 p = ro + rd * hit;
    vec3 n = nrm(p);
    float a = dA(p), b = dB(p);
    float wB = mix(step(b, a), smoothstep(-0.25, 0.25, a - b), uFuse);
    vec3 tint = mix(cyan, amber, wB * (1.0 - uFuse * 0.55));
    float fres = pow(1.0 - max(0.0, dot(n, -rd)), 3.0);
    vec3 refl = studio(reflect(rd, n));

    // march through the interior: thickness of A, of B, and of both
    float thA = 0.0, thB = 0.0, thAB = 0.0;
    float span = min(tEnd - hit, (uA.w + uB.w) * 2.2);
    float dt = span / 24.0;
    for (int i = 0; i < 24; i++){
      vec3 q = ro + rd * (hit + dt * (float(i) + 0.5));
      float ia = step(dA(q), 0.0), ib = step(dB(q), 0.0);
      thA += ia * dt; thB += ib * dt; thAB += ia * ib * dt;
    }
    vec3 col = refl * (0.6 + fres) + tint * fres * 2.6;
    col += cyan * thA * 0.32 * (1.0 - uFuse * 0.3);
    // B seen through A glows warm -- the exception inside the rule
    col += amber * thB * (0.55 + uNest * 0.9) * (1.0 - uFuse * 0.5);
    // where both hold and the outcomes disagree: a coral lens, pulsing
    float pulse = 0.8 + 0.2 * sin(uT * 2.6);
    col += coral * thAB * uOppose * 5.0 * pulse;
    // redundancy merges into one calm cyan body
    col = mix(col, cyan * (0.5 + fres * 2.4) + refl * 0.6 + cyan * (thA + thB) * 0.22, uFuse * 0.55);
    float alpha = clamp(0.25 + fres * 0.9 + (thA + thB) * 0.25, 0.0, 1.0) * uAlpha;
    gl_FragColor = vec4(col * uAlpha, alpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

export class ScopeVolumes implements Actor {
  readonly group: Mesh;
  private mat: ShaderMaterial;

  constructor(steps: number) {
    this.mat = new ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG,
      uniforms: {
        uA: { value: new Vector4() }, uB: { value: new Vector4() },
        uFuse: { value: 0 }, uOppose: { value: 0 }, uNest: { value: 0 }, uAlpha: { value: 0 }, uT: { value: 0 },
        uBoxMin: { value: new Vector3() }, uBoxMax: { value: new Vector3() }, uSteps: { value: steps },
      },
      // back faces, so the box still renders when the camera is inside it
      side: BackSide, transparent: true, depthWrite: false,
    });
    this.group = new Mesh(new BoxGeometry(1, 1, 1), this.mat);
    this.group.frustumCulled = false;
  }

  setSteps(n: number) { this.mat.uniforms.uSteps.value = n; }

  update(p: WorldParams, ctx: FrameContext) {
    const v = p.scope * (1 - p.dim * 0.7);
    this.group.visible = v > 0.003;
    if (!this.group.visible) return;
    const u = this.mat.uniforms;
    (u.uA.value as Vector4).set(p.sAx, p.sAy, 0, p.sAr);
    (u.uB.value as Vector4).set(p.sBx, p.sBy, 0, p.sBr);
    u.uFuse.value = p.fuse; u.uOppose.value = p.oppose; u.uNest.value = p.nest;
    u.uAlpha.value = v; u.uT.value = ctx.t;
    const m = 0.5;
    const min = new Vector3(Math.min(p.sAx - p.sAr, p.sBx - p.sBr) - m, Math.min(p.sAy - p.sAr, p.sBy - p.sBr) - m, -Math.max(p.sAr, p.sBr) - m);
    const max = new Vector3(Math.max(p.sAx + p.sAr, p.sBx + p.sBr) + m, Math.max(p.sAy + p.sAr, p.sBy + p.sBr) + m, Math.max(p.sAr, p.sBr) + m);
    (u.uBoxMin.value as Vector3).copy(min); (u.uBoxMax.value as Vector3).copy(max);
    this.group.position.set((min.x + max.x) / 2, (min.y + max.y) / 2, (min.z + max.z) / 2);
    this.group.scale.set(max.x - min.x, max.y - min.y, max.z - min.z);
  }

  /** World-space anchors for the projected labels. */
  anchors(p: WorldParams) {
    return { a: new Vector3(p.sAx - p.sAr * 0.7, p.sAy + p.sAr * 1.05, 0), b: new Vector3(p.sBx - p.sBr * 0.4, p.sBy - p.sBr * 1.25, 0) };
  }

  dispose() { this.group.geometry.dispose(); this.mat.dispose(); }
}
