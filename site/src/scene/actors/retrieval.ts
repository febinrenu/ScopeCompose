/*  Act 01: a query searching the constellation.
 *  The constellation (particles, one cluster per corpus instance) is drawn
 *  by the particle field; this actor adds the query comet, the two search
 *  beams -- lexical BM25 in cyan, dense in violet -- the fusion that ignites
 *  the nodes both agree on, and five cards pulled forward with real text.
 */
import {
  AdditiveBlending, BufferAttribute, BufferGeometry, Group, LineBasicMaterial, LineLoop, Mesh, MeshBasicMaterial,
  PlaneGeometry, Points, QuadraticBezierCurve3, ShaderMaterial, TubeGeometry, Vector3,
} from "three";
import type { Actor, FrameContext } from "../core/actor";
import type { WorldParams } from "../../acts/keyframes";
import type { Node3 } from "../targets";
import { HEX, lin } from "../core/color";
import { cardTexture } from "../textures";

const Q0 = new Vector3(0, 0.5, 7.5), Q1 = new Vector3(0, 0.05, -5.5);
const clamp = (x: number) => Math.min(1, Math.max(0, x));
const smooth = (x: number) => { const t = clamp(x); return t * t * (3 - 2 * t); };

const BEAM_VERT = /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const BEAM_FRAG = /* glsl */ `
  uniform vec3 uCol; uniform float uProg; uniform float uA;
  varying vec2 vUv;
  void main(){
    float body = smoothstep(uProg + 0.001, uProg - 0.02, vUv.x);
    float head = smoothstep(0.06, 0.0, abs(vUv.x - uProg)) * step(0.001, uProg) * step(uProg, 0.999);
    float edge = 1.0 - abs(vUv.y - 0.5) * 2.0;
    gl_FragColor = vec4(uCol * (body * 0.55 + head * 3.0) * edge * uA, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

export interface CardData { id: string; text: string }

export class Retrieval implements Actor {
  readonly group = new Group();
  private beams: { mesh: Mesh; mat: ShaderMaterial; lexical: boolean; i: number }[] = [];
  private head: Points;
  private sparks: Points;
  private cards: { g: Group; from: Vector3; to: Vector3; rot: number; k: number }[] = [];
  private seat = new Vector3();
  readonly beamMid: { lexical: Vector3; dense: Vector3 };

  constructor(nodes: Node3[], cards: CardData[]) {
    // Deterministic picks: the nodes nearest the query's destination.
    const order = nodes.map((n, i) => ({ i, d: Math.hypot(n.x - Q1.x, n.y - Q1.y, (n.z - Q1.z) * 0.6) }))
      .filter(o => nodes[o.i].z < -3).sort((a, b) => a.d - b.d).map(o => o.i);
    const bm25 = order.slice(0, 6), dense = order.slice(2, 8);
    const rrf = [order[2], order[3], order[4], order[5], order[0]];
    const at = (i: number) => new Vector3(nodes[i].x, nodes[i].y, nodes[i].z);

    const mkBeam = (to: Vector3, lexical: boolean, i: number) => {
      const mid = Q1.clone().lerp(to, 0.5).add(new Vector3(0, lexical ? 1.1 : -0.9, 0.6));
      const curve = new QuadraticBezierCurve3(Q1.clone(), mid, to);
      const mat = new ShaderMaterial({
        vertexShader: BEAM_VERT, fragmentShader: BEAM_FRAG,
        uniforms: { uCol: { value: lin(lexical ? HEX.cyan : HEX.violet, 3) }, uProg: { value: 0 }, uA: { value: 1 } },
        transparent: true, depthWrite: false, blending: AdditiveBlending,
      });
      const mesh = new Mesh(new TubeGeometry(curve, 48, 0.014, 6, false), mat);
      mesh.frustumCulled = false;
      this.group.add(mesh);
      this.beams.push({ mesh, mat, lexical, i });
      return curve;
    };
    const lexCurves = bm25.map((n, i) => mkBeam(at(n), true, i));
    const denCurves = dense.map((n, i) => mkBeam(at(n), false, i));
    this.beamMid = { lexical: lexCurves[0].getPoint(0.55), dense: denCurves[denCurves.length - 1].getPoint(0.55) };

    // query head and its trail
    const trail = 32, hg = new BufferGeometry(), hk = new Float32Array(trail);
    for (let i = 0; i < trail; i++) hk[i] = i / (trail - 1);
    hg.setAttribute("position", new BufferAttribute(new Float32Array(trail * 3), 3));
    hg.setAttribute("aK", new BufferAttribute(hk, 1));
    this.head = new Points(hg, new ShaderMaterial({
      uniforms: { uQ: { value: 0 }, uA: { value: 0 }, uPx: { value: 1 }, uQ0: { value: Q0 }, uQ1: { value: Q1 } },
      vertexShader: /* glsl */ `
        attribute float aK; uniform float uQ; uniform float uPx; uniform vec3 uQ0; uniform vec3 uQ1; varying float vA;
        void main(){
          float s = clamp(uQ - aK * 0.14, 0.0, 1.0);
          vec4 mv = modelViewMatrix * vec4(mix(uQ0, uQ1, s), 1.0);
          gl_Position = projectionMatrix * mv;
          vA = (1.0 - aK);
          gl_PointSize = uPx * mix(46.0, 4.0, aK) / max(0.7, -mv.z * 0.13);
        }`,
      fragmentShader: /* glsl */ `
        uniform float uA; varying float vA;
        void main(){ float d = length(gl_PointCoord - 0.5); float g = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(vec3(2.6, 3.0, 3.2) * g * g * vA * uA, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: AdditiveBlending,
    }));
    this.head.frustumCulled = false;
    this.group.add(this.head);

    // ignition at the fused picks
    const sg = new BufferGeometry();
    sg.setAttribute("position", new BufferAttribute(new Float32Array(rrf.flatMap(i => [nodes[i].x, nodes[i].y, nodes[i].z])), 3));
    this.sparks = new Points(sg, new ShaderMaterial({
      uniforms: { uA: { value: 0 }, uPx: { value: 1 } },
      vertexShader: /* glsl */ `uniform float uPx; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = uPx * 140.0 / max(1.0, -mv.z); }`,
      fragmentShader: /* glsl */ `uniform float uA; void main(){ float d = length(gl_PointCoord - 0.5); float g = smoothstep(0.5, 0.0, d);
        gl_FragColor = vec4(vec3(2.2, 3.4, 3.6) * pow(g, 3.0) * uA, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: AdditiveBlending,
    }));
    this.sparks.frustumCulled = false;
    this.group.add(this.sparks);

    // the five cards
    const W = 1.5, H = W / 1.5;
    cards.slice(0, 5).forEach((c, k) => {
      const g = new Group();
      const face = new Mesh(new PlaneGeometry(W, H), new MeshBasicMaterial({ map: cardTexture(c.id, c.text, k + 1), transparent: true }));
      const edge = new LineLoop(new BufferGeometry().setFromPoints([
        new Vector3(-W / 2, -H / 2, 0.002), new Vector3(W / 2, -H / 2, 0.002), new Vector3(W / 2, H / 2, 0.002), new Vector3(-W / 2, H / 2, 0.002),
      ]), new LineBasicMaterial({ color: lin(HEX.cyan, 2.6), transparent: true }));
      g.add(face, edge);
      this.group.add(g);
      const x = (k - 2) * 1.56;
      this.cards.push({ g, from: at(rrf[k]), to: new Vector3(x, -0.72 - Math.abs(k - 2) * 0.06, 2.4 - Math.abs(k - 2) * 0.35), rot: -x * 0.09, k });
    });
  }

  update(p: WorldParams, ctx: FrameContext) {
    const on = Math.max(p.query, p.bm25, p.dense, p.rrf, p.cards);
    this.group.visible = on > 0.003;
    if (!this.group.visible) return;
    const hm = this.head.material as ShaderMaterial;
    hm.uniforms.uQ.value = p.query; hm.uniforms.uPx.value = ctx.px;
    hm.uniforms.uA.value = p.query > 0.001 ? 1 - p.cards * 0.8 : 0;
    for (const b of this.beams) {
      const prog = b.lexical ? p.bm25 : p.dense;
      const stag = smooth(prog * 1.4 - b.i * 0.07);
      b.mat.uniforms.uProg.value = stag;
      b.mat.uniforms.uA.value = (1 - p.cards * 0.85);
      b.mesh.visible = stag > 0.001;
    }
    const sm = this.sparks.material as ShaderMaterial;
    sm.uniforms.uA.value = p.rrf * (1 - p.cards * 0.6); sm.uniforms.uPx.value = ctx.px;
    // the arc fits the viewport: narrower on squarer screens, a vertical stack in portrait
    const aspect = ctx.resolution.x / ctx.resolution.y;
    const fit = Math.min(1, Math.max(0.62, aspect / 1.4));     // the camera also dollies back below 16:10
    this.cards.forEach((c, k) => {
      const t = smooth(p.cards * 1.7 - k * 0.16);
      c.g.visible = t > 0.001;
      if (aspect < 0.9) this.seat.set(0, 0.9 - c.k * 1.08, 2.4);
      else this.seat.set(c.to.x * fit, c.to.y, c.to.z);
      c.g.position.lerpVectors(c.from, this.seat, t);
      c.g.scale.setScalar((0.12 + 0.88 * t) * (aspect < 0.9 ? 1 : Math.max(0.8, fit)));
      c.g.rotation.y = aspect < 0.9 ? 0 : c.rot * t;
      ((c.g.children[0] as Mesh).material as MeshBasicMaterial).opacity = Math.min(1, t * 1.4);
    });
  }

  dispose() {
    this.group.traverse(o => {
      const m = o as Mesh;
      m.geometry?.dispose();
      const mat = m.material as MeshBasicMaterial | undefined;
      mat?.map?.dispose(); mat?.dispose?.();
    });
  }
}
