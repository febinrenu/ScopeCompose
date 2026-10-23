/*  The two passages of the story, as glass monoliths.
 *
 *  Act 02: they meet; a credibility scan picks A; a stillness with one amber
 *  caret; a crack runs through B; B shatters into Voronoi shards that tumble
 *  on seeded ballistic arcs, shedding embers.
 *  Act 03: A stands alone on a pedestal; B is only a dashed socket, and an
 *  x-ray lens reveals its ghost.
 *  Act 07: every shard flies to its own seat in B's new place inside the
 *  rule, and B reassembles.
 *
 *  All of it is a function of WorldParams (fracture.ts holds the pure maths).
 */
import {
  AdditiveBlending, BoxGeometry, BufferAttribute, BufferGeometry, DataTexture, EdgesGeometry, FloatType, Group,
  LineBasicMaterial, LineDashedMaterial, LineLoop, LineSegments, Mesh, MeshBasicMaterial, NearestFilter, PlaneGeometry,
  Points, RGBAFormat, RingGeometry, ShaderMaterial, Vector2, Vector3, type MeshPhysicalMaterial, type Texture,
} from "three";
import type { Actor, FrameContext } from "../core/actor";
import type { WorldParams } from "../../acts/keyframes";
import { HEX, lin } from "../core/color";
import { makeGlass } from "../materials/glass";
import { passageTexture, type PanelText } from "../textures";
import { shardHeat, shardPose, voronoiCells, type Cell, type Slab as SlabPose } from "../fracture";
import { rng } from "../core/rand";

export const SLAB_W = 3.2, SLAB_H = 1.25, SLAB_D = 0.14;

class Monolith {
  readonly g = new Group();
  readonly face: Mesh;
  readonly glass: Mesh;
  readonly rim: LineSegments;
  readonly faceMat: MeshBasicMaterial;
  constructor(tex: Texture, accent: string, env: Texture, transmission: boolean) {
    this.glass = new Mesh(new BoxGeometry(SLAB_W, SLAB_H, SLAB_D), makeGlass(accent === HEX.amber ? "amber" : "cyan", env, transmission));
    this.faceMat = new MeshBasicMaterial({ map: tex, transparent: true });
    this.face = new Mesh(new PlaneGeometry(SLAB_W - 0.06, SLAB_H - 0.06), this.faceMat);
    this.face.position.z = SLAB_D / 2 + 0.004;
    this.rim = new LineSegments(new EdgesGeometry(new BoxGeometry(SLAB_W, SLAB_H, SLAB_D)),
      new LineBasicMaterial({ color: lin(accent, 2.4), transparent: true, blending: AdditiveBlending, depthWrite: false }));
    this.g.add(this.glass, this.face, this.rim);
  }
  set(x: number, y: number, s: number, alpha: number, rimK: number) {
    this.g.position.set(x, y, 0); this.g.scale.setScalar(s);
    this.g.visible = alpha > 0.003;
    this.faceMat.opacity = alpha;
    (this.rim.material as LineBasicMaterial).opacity = alpha * rimK;
    const gm = this.glass.material as MeshPhysicalMaterial;
    if (gm.transparent) gm.opacity = Math.min(gm.opacity, 1) * 0 + (gm.userData.base ??= gm.opacity) * alpha;
  }
}

const SHARD_VERT = /* glsl */ `
  attribute float aShard; attribute vec3 aLocal; attribute float aFront;
  uniform sampler2D uPose; uniform float uN;
  varying vec2 vUv; varying float vFront; varying float vHeat; varying vec3 vN;
  vec3 rot(vec4 q, vec3 v){ return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
  void main(){
    float i = aShard;
    vec4 a = texture2D(uPose, vec2((i * 3.0 + 0.5) / (uN * 3.0), 0.5));
    vec4 q = texture2D(uPose, vec2((i * 3.0 + 1.5) / (uN * 3.0), 0.5));
    vec4 h = texture2D(uPose, vec2((i * 3.0 + 2.5) / (uN * 3.0), 0.5));
    vec3 p = rot(q, aLocal * a.w) + a.xyz;
    vUv = uv; vFront = aFront; vHeat = h.x; vN = normalize(rot(q, normal));
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
  }`;
const SHARD_FRAG = /* glsl */ `
  uniform sampler2D uTex; uniform float uAlpha;
  varying vec2 vUv; varying float vFront; varying float vHeat; varying vec3 vN;
  void main(){
    vec3 amber = vec3(1.0, 0.62, 0.26);
    vec4 t = texture2D(uTex, vUv);
    float rim = pow(1.0 - abs(vN.z), 2.0);
    vec3 face = t.rgb * 0.95 + amber * 0.05;
    vec3 side = amber * (0.6 + rim * 1.4);
    vec3 col = mix(side, face, vFront);
    col += amber * vHeat * 3.0;              // hot as it leaves
    gl_FragColor = vec4(col, uAlpha);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

const GHOST_FRAG = /* glsl */ `
  uniform sampler2D uTex; uniform vec3 uLens; uniform vec2 uRes; uniform float uA; uniform float uT;
  varying vec2 vUv;
  void main(){
    vec2 ndc = gl_FragCoord.xy / uRes * 2.0 - 1.0;
    float aspect = uRes.x / uRes.y;
    float d = length(vec2((ndc.x - uLens.x) * aspect, ndc.y - uLens.y));
    float inside = smoothstep(uLens.z, uLens.z - 0.02, d);
    if (inside <= 0.0) discard;
    vec4 t = texture2D(uTex, vUv);
    float lum = dot(t.rgb, vec3(0.3, 0.59, 0.11));
    float scan = 0.82 + 0.18 * sin(gl_FragCoord.y * 1.4 + uT * 6.0);
    vec3 xr = vec3(0.55, 0.85, 1.0) * (0.25 + lum * 2.4) * scan;
    float edge = smoothstep(0.012, 0.0, min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y)));
    xr += vec3(0.7, 0.95, 1.0) * edge * 2.5;
    gl_FragColor = vec4(xr * inside * uA, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

export class Slabs implements Actor {
  readonly group = new Group();
  private A: Monolith;
  private B: Monolith;
  private cells: Cell[];
  private shardMesh: Mesh;
  private shardMat: ShaderMaterial;
  private pose: DataTexture;
  private poseData: Float32Array;
  private crack: LineSegments;
  private crackMat: ShaderMaterial;
  private embers: Points;
  private emberMat: ShaderMaterial;
  private caret: Mesh;
  private caretUV: [number, number];
  private scan: Mesh;
  private socket: LineLoop;
  private ghost: Mesh;
  private ghostMat: ShaderMaterial;
  private pedestal: Mesh;
  private burnFrom: SlabPose;

  constructor(a: PanelText, b: PanelText, env: Texture, transmission: boolean, shardCount: number, emberCount: number, burnFrom: SlabPose) {
    const ta = passageTexture(a), tb = passageTexture(b);
    this.A = new Monolith(ta.tex, HEX.cyan, env, transmission);
    this.B = new Monolith(tb.tex, HEX.amber, env, transmission);
    this.caretUV = tb.caret;
    this.burnFrom = burnFrom;
    this.group.add(this.A.g, this.B.g);

    // caret: one amber bar at the end of the exception
    this.caret = new Mesh(new PlaneGeometry(0.018, 0.13), new MeshBasicMaterial({ color: lin(HEX.amber, 4), transparent: true }));
    this.caret.position.set((this.caretUV[0] - 0.5) * (SLAB_W - 0.06), (this.caretUV[1] - 0.5) * (SLAB_H - 0.06), SLAB_D / 2 + 0.01);
    this.B.g.add(this.caret);

    // the credibility scan: a bar of light sweeping both passages during selection
    this.scan = new Mesh(new PlaneGeometry(0.04, SLAB_H * 1.25), new MeshBasicMaterial({ color: lin(HEX.cyan, 5), transparent: true, blending: AdditiveBlending, depthWrite: false }));
    this.group.add(this.scan);

    // ---- fracture
    this.cells = voronoiCells(shardCount, SLAB_W, SLAB_H, 19);
    const n = this.cells.length;
    const pos: number[] = [], loc: number[] = [], uvs: number[] = [], idx: number[] = [], front: number[] = [], nrm: number[] = [];
    const crackPos: number[] = [], crackOrd: number[] = [];
    const hz = SLAB_D / 2;
    this.cells.forEach((c, i) => {
      const [cx, cy] = c.centroid, P = c.poly;
      const push = (x: number, y: number, z: number, isFront: number, nx: number, ny: number, nz: number) => {
        pos.push(x, y, z); loc.push(x - cx, y - cy, z); idx.push(i); front.push(isFront); nrm.push(nx, ny, nz);
        uvs.push(x / SLAB_W + 0.5, y / SLAB_H + 0.5);
      };
      for (let k = 1; k < P.length - 1; k++) {           // front (fan) and back
        push(P[0][0], P[0][1], hz, 1, 0, 0, 1); push(P[k][0], P[k][1], hz, 1, 0, 0, 1); push(P[k + 1][0], P[k + 1][1], hz, 1, 0, 0, 1);
        push(P[0][0], P[0][1], -hz, 0, 0, 0, -1); push(P[k + 1][0], P[k + 1][1], -hz, 0, 0, 0, -1); push(P[k][0], P[k][1], -hz, 0, 0, 0, -1);
      }
      for (let k = 0; k < P.length; k++) {                 // sides
        const [x0, y0] = P[k], [x1, y1] = P[(k + 1) % P.length];
        const nx = y1 - y0, ny = -(x1 - x0), l = Math.hypot(nx, ny) || 1;
        push(x0, y0, hz, 0, nx / l, ny / l, 0); push(x0, y0, -hz, 0, nx / l, ny / l, 0); push(x1, y1, -hz, 0, nx / l, ny / l, 0);
        push(x0, y0, hz, 0, nx / l, ny / l, 0); push(x1, y1, -hz, 0, nx / l, ny / l, 0); push(x1, y1, hz, 0, nx / l, ny / l, 0);
        // crack line on the face, revealed from the right edge first
        crackPos.push(x0, y0, hz + 0.006, x1, y1, hz + 0.006);
        const o = 1 - ((x0 + x1) / 2 / SLAB_W + 0.5);
        crackOrd.push(o, o);
      }
    });
    const sg = new BufferGeometry();
    sg.setAttribute("position", new BufferAttribute(new Float32Array(pos), 3));
    sg.setAttribute("aLocal", new BufferAttribute(new Float32Array(loc), 3));
    sg.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
    sg.setAttribute("normal", new BufferAttribute(new Float32Array(nrm), 3));
    sg.setAttribute("aShard", new BufferAttribute(new Float32Array(idx), 1));
    sg.setAttribute("aFront", new BufferAttribute(new Float32Array(front), 1));
    this.poseData = new Float32Array(n * 3 * 4);
    this.pose = new DataTexture(this.poseData, n * 3, 1, RGBAFormat, FloatType);
    this.pose.minFilter = this.pose.magFilter = NearestFilter;
    this.pose.needsUpdate = true;
    this.shardMat = new ShaderMaterial({
      vertexShader: SHARD_VERT, fragmentShader: SHARD_FRAG,
      uniforms: { uPose: { value: this.pose }, uN: { value: n }, uTex: { value: tb.tex }, uAlpha: { value: 1 } },
      transparent: true,
    });
    this.shardMesh = new Mesh(sg, this.shardMat);
    this.shardMesh.frustumCulled = false;
    this.group.add(this.shardMesh);

    const cg = new BufferGeometry();
    cg.setAttribute("position", new BufferAttribute(new Float32Array(crackPos), 3));
    cg.setAttribute("aOrd", new BufferAttribute(new Float32Array(crackOrd), 1));
    this.crackMat = new ShaderMaterial({
      uniforms: { uCrack: { value: 0 }, uT: { value: 0 } },
      vertexShader: /* glsl */ `attribute float aOrd; varying float vOrd; void main(){ vOrd = aOrd; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `uniform float uCrack; uniform float uT; varying float vOrd;
        void main(){ float on = smoothstep(vOrd, vOrd - 0.06, uCrack * 1.1 - 0.05);
          float flick = 0.75 + 0.25 * sin(uT * 31.0 + vOrd * 40.0);
          gl_FragColor = vec4(vec3(1.0, 0.66, 0.3) * 5.0 * on * flick, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: AdditiveBlending,
    });
    this.crack = new LineSegments(cg, this.crackMat);
    this.B.g.add(this.crack);

    // embers: sparks thrown off each shard as it leaves
    const r = rng(77);
    const eg = new BufferGeometry(), eIdx = new Float32Array(emberCount), eDir = new Float32Array(emberCount * 3), ePos = new Float32Array(emberCount * 3);
    for (let i = 0; i < emberCount; i++) {
      eIdx[i] = Math.floor(r() * n);
      eDir[i * 3] = (r() - 0.3) * 1.6; eDir[i * 3 + 1] = (r() - 0.2) * 1.4; eDir[i * 3 + 2] = (r() - 0.1) * 1.4;
    }
    eg.setAttribute("position", new BufferAttribute(ePos, 3));
    eg.setAttribute("aShard", new BufferAttribute(eIdx, 1));
    eg.setAttribute("aDir", new BufferAttribute(eDir, 3));
    this.emberMat = new ShaderMaterial({
      uniforms: { uPose: { value: this.pose }, uN: { value: n }, uPx: { value: 1 }, uA: { value: 1 } },
      vertexShader: /* glsl */ `
        attribute float aShard; attribute vec3 aDir; uniform sampler2D uPose; uniform float uN; uniform float uPx;
        varying float vA;
        void main(){
          vec4 a = texture2D(uPose, vec2((aShard * 3.0 + 0.5) / (uN * 3.0), 0.5));
          vec4 h = texture2D(uPose, vec2((aShard * 3.0 + 2.5) / (uN * 3.0), 0.5));
          float life = 1.0 - h.x;                                   // 0 at launch, 1 when spent
          float launched = h.y;
          vec3 p = a.xyz + aDir * life * 0.9 + vec3(0.0, -0.6, 0.0) * life * life;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          vA = launched * (1.0 - life) ;
          gl_PointSize = uPx * (14.0 + 26.0 * (1.0 - life)) / max(0.8, -mv.z);
        }`,
      fragmentShader: /* glsl */ `uniform float uA; varying float vA;
        void main(){ float d = length(gl_PointCoord - 0.5); float g = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(vec3(3.2, 1.7, 0.7) * g * g * vA * uA, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: AdditiveBlending,
    });
    this.embers = new Points(eg, this.emberMat);
    this.embers.frustumCulled = false;
    this.group.add(this.embers);

    // Act 03: the socket, the x-ray ghost, the pedestal
    const sw = SLAB_W / 2, sh = SLAB_H / 2;
    this.socket = new LineLoop(new BufferGeometry().setFromPoints([
      new Vector3(-sw, -sh, 0), new Vector3(sw, -sh, 0), new Vector3(sw, sh, 0), new Vector3(-sw, sh, 0),
    ]), new LineDashedMaterial({ color: lin(HEX.coral, 2.5), dashSize: 0.09, gapSize: 0.07, transparent: true, depthWrite: false }));
    this.socket.computeLineDistances();
    this.group.add(this.socket);
    this.ghostMat = new ShaderMaterial({
      vertexShader: /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: GHOST_FRAG,
      uniforms: { uTex: { value: tb.tex }, uLens: { value: new Vector3(0, 0, 0.2) }, uRes: { value: new Vector2(1, 1) }, uA: { value: 0 }, uT: { value: 0 } },
      transparent: true, depthWrite: false, blending: AdditiveBlending,
    });
    this.ghost = new Mesh(new PlaneGeometry(SLAB_W, SLAB_H), this.ghostMat);
    this.group.add(this.ghost);
    this.pedestal = new Mesh(new RingGeometry(1.05, 1.12, 128), new MeshBasicMaterial({ color: lin(HEX.cyan, 3), transparent: true, blending: AdditiveBlending, depthWrite: false }));
    this.pedestal.rotation.x = -Math.PI / 2;
    this.group.add(this.pedestal);
  }

  /** World-space anchors for the projected labels. */
  anchors(p: WorldParams) {
    return {
      a: new Vector3(p.aX - SLAB_W / 2 * p.aS, p.aY + (SLAB_H / 2 + 0.22) * p.aS, 0),
      b: new Vector3(p.bX - SLAB_W / 2 * p.bS, p.bY + (SLAB_H / 2 + 0.22) * p.bS, 0),
    };
  }

  update(p: WorldParams, ctx: FrameContext) {
    const v = p.duel;
    this.group.visible = v > 0.003 || p.socket > 0.003;
    if (!this.group.visible) return;
    const shattered = p.burn > 0.0005 && p.restore < 0.999;

    // A: the rule. Brightens as it wins.
    this.A.set(p.aX, p.aY, p.aS, v, 1 + p.select * 1.2);
    // B: the exception. Intact until the break; again once restored.
    const bAlpha = shattered ? 0 : v * (p.burn > 0.0005 ? p.restore : 1) * (1 - p.select * 0.3 * (1 - p.restore));
    this.B.set(p.bX, p.bY, p.bS, bAlpha, 1);
    this.caret.visible = p.caret > 0.01 && !shattered;
    (this.caret.material as MeshBasicMaterial).opacity = p.caret * (ctx.ambient ? (Math.sin(ctx.t * 6.6) > 0 ? 1 : 0.15) : 1);

    // the credibility scan sweeps left to right while selection decides
    const sel = p.select;
    this.scan.visible = sel > 0.02 && sel < 0.98;
    const sx = (p.aX - SLAB_W * 0.55) + ((p.bX + SLAB_W * 0.55) - (p.aX - SLAB_W * 0.55)) * sel;
    this.scan.position.set(sx, (p.aY + p.bY) / 2, 0.2);
    (this.scan.material as MeshBasicMaterial).opacity = Math.sin(sel * Math.PI);

    // crack
    this.crack.visible = p.crack > 0.01 && !shattered;
    this.crackMat.uniforms.uCrack.value = p.crack;
    this.crackMat.uniforms.uT.value = ctx.ambient ? ctx.t : 0;

    // shards
    this.shardMesh.visible = shattered && v > 0.003;
    this.embers.visible = this.shardMesh.visible;
    if (this.shardMesh.visible) {
      const to: SlabPose = { x: p.bX, y: p.bY, s: p.bS };
      const d = this.poseData;
      this.cells.forEach((c, i) => {
        const q = shardPose(c, SLAB_W, p.burn, p.compose, this.burnFrom, to);
        const h = shardHeat(c, SLAB_W, p.burn);
        d.set([q.x, q.y, q.z, q.s, q.qx, q.qy, q.qz, q.qw, h * (1 - p.compose), h > 0 || p.burn >= 0.999 ? 1 : 0, 0, 0], i * 12);
      });
      this.pose.needsUpdate = true;
      // in the failure act the debris recedes: faint, so it never crosses the copy
      const recede = 1 - p.socket * 0.88 * (1 - p.compose);
      this.shardMat.uniforms.uAlpha.value = v * (1 - p.restore) * recede;
      this.shardMat.depthWrite = this.shardMat.uniforms.uAlpha.value > 0.98;   // faint debris must not hide what is behind it
      this.emberMat.uniforms.uPx.value = ctx.px;
      this.emberMat.uniforms.uA.value = v * (1 - p.compose) * recede;
    }

    // Act 03
    const so = p.socket * (1 - p.compose);
    this.socket.visible = so > 0.01;
    this.socket.position.set(p.bX, p.bY, 0); this.socket.scale.setScalar(p.bS);
    (this.socket.material as LineDashedMaterial).opacity = so;
    this.ghost.visible = p.xray > 0.01 && shattered;
    this.ghost.position.set(p.bX, p.bY, 0.02); this.ghost.scale.setScalar(p.bS);
    const gu = this.ghostMat.uniforms;
    gu.uA.value = p.xray; gu.uT.value = ctx.ambient ? ctx.t : 0;
    (gu.uLens.value as Vector3).set(ctx.lens.x, ctx.lens.y, ctx.lens.r);
    (gu.uRes.value as Vector2).copy(ctx.resolution);
    this.pedestal.visible = p.socket > 0.01;
    this.pedestal.position.set(p.aX, p.aY - SLAB_H / 2 * p.aS - 0.32, 0.4);
    this.pedestal.scale.setScalar(p.aS * 1.6);
    (this.pedestal.material as MeshBasicMaterial).opacity = p.socket * 0.9;
  }

  /** Socket centre, for parking the lens when there is no pointer. */
  socketCentre(p: WorldParams) { return new Vector3(p.bX, p.bY, 0); }

  dispose() {
    this.group.traverse(o => {
      const m = o as Mesh;
      m.geometry?.dispose();
      const mat = m.material as MeshBasicMaterial | undefined;
      mat?.map?.dispose(); mat?.dispose?.();
    });
    this.pose.dispose();
  }
}
