/*  Act 05: the engine as a machine you travel along.
 *  Seven stations on one rail, a contract gate between A4 and B1, and a
 *  packet of evidence carried along by `flow`. Each station's micro-scene
 *  is lit by stationActivation(flow, i) -- a pure function.
 */
import {
  AdditiveBlending, BoxGeometry, BufferGeometry, CylinderGeometry, EdgesGeometry, Group, Line, LineBasicMaterial,
  LineSegments, Mesh, MeshBasicMaterial, PlaneGeometry, ShaderMaterial, SphereGeometry, TorusGeometry, Vector3, type Texture,
} from "three";
import type { Actor, FrameContext } from "../core/actor";
import { STATION_X, type WorldParams } from "../../acts/keyframes";
import { stationActivation } from "../../acts/derive";
import { HEX, lin } from "../core/color";
import { makeGlass } from "../materials/glass";
import { labelTexture, passageTexture } from "../textures";

const Y = 0.55;
const add = (o: { opacity: number }) => o;
const glow = (hex: string, k: number) => new MeshBasicMaterial({ color: lin(hex, k), transparent: true, blending: AdditiveBlending, depthWrite: false });
const line = (pts: Vector3[], hex: string, k: number) =>
  new Line(new BufferGeometry().setFromPoints(pts), new LineBasicMaterial({ color: lin(hex, k), transparent: true, blending: AdditiveBlending, depthWrite: false }));

const FRINGE = /* glsl */ `
  uniform float uK; uniform float uT; uniform float uA; varying vec2 vUv;
  void main(){
    float f = sin(vUv.x * 60.0 + uT * 3.0) * sin(vUv.y * 9.0 - uT * 1.5);
    float band = smoothstep(0.2, 1.0, abs(f)) * uK;
    vec3 c = mix(vec3(0.45, 0.95, 1.0), vec3(1.0, 0.62, 0.26), vUv.x) * band * 2.2;
    gl_FragColor = vec4(c * uA, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;
const REVEAL = /* glsl */ `
  uniform sampler2D uTex; uniform float uScan; uniform float uA; varying vec2 vUv;
  void main(){
    vec4 t = texture2D(uTex, vUv);
    float shown = step(vUv.x, uScan);
    float laser = smoothstep(0.012, 0.0, abs(vUv.x - uScan)) * step(0.001, uScan) * step(uScan, 0.999);
    vec3 dim = t.rgb * 0.18;
    vec3 col = mix(dim, t.rgb, shown) + vec3(1.0, 0.35, 0.25) * laser * 5.0;
    gl_FragColor = vec4(col * uA, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

export class Engine implements Actor {
  readonly group = new Group();
  private st: Group[] = [];
  private mats: { opacity: number }[][] = [];
  private packet: Mesh;
  private fringe: ShaderMaterial;
  private reveal: ShaderMaterial;
  private prismBeams: Line[] = [];
  private rings: Mesh[] = [];
  private blocks: Mesh[] = [];
  private typer: ShaderMaterial;
  private gate: Group;
  private prism!: Mesh;

  constructor(env: Texture, transmission: boolean) {
    const rail = line([new Vector3(STATION_X[0] - 2, Y - 0.62, 0), new Vector3(STATION_X[6] + 2, Y - 0.62, 0)], HEX.ivory, 0.35);
    this.group.add(rail);
    STATION_X.forEach(() => { const g = new Group(); this.st.push(g); this.mats.push([]); this.group.add(g); });
    this.st.forEach((g, i) => g.position.set(STATION_X[i], Y, 0));

    // A1 retrieval: a small constellation, two beams converging
    {
      const g = this.st[0];
      for (let k = 0; k < 18; k++) {
        const a = k / 18 * Math.PI * 2;
        const m = new Mesh(new SphereGeometry(0.025, 8, 6), glow(HEX.ivory, 2.2));
        m.position.set(Math.cos(a) * 0.75, Math.sin(a * 1.7) * 0.32, Math.sin(a) * 0.3);
        g.add(m); this.mats[0].push(add(m.material as MeshBasicMaterial));
      }
      const l1 = line([new Vector3(-0.9, 0.4, 0), new Vector3(0, 0, 0)], HEX.cyan, 3);
      const l2 = line([new Vector3(-0.9, -0.4, 0), new Vector3(0, 0, 0)], HEX.violet, 3);
      g.add(l1, l2); this.mats[0].push(l1.material as LineBasicMaterial, l2.material as LineBasicMaterial);
    }
    // A2 detection: two passages as bars, interference between them
    {
      const g = this.st[1];
      const b1 = new Mesh(new BoxGeometry(0.06, 0.9, 0.06), glow(HEX.cyan, 2.5)); b1.position.x = -0.55;
      const b2 = new Mesh(new BoxGeometry(0.06, 0.9, 0.06), glow(HEX.amber, 2.5)); b2.position.x = 0.55;
      this.fringe = new ShaderMaterial({ vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);}`,
        fragmentShader: FRINGE, uniforms: { uK: { value: 0 }, uT: { value: 0 }, uA: { value: 0 } }, transparent: true, blending: AdditiveBlending, depthWrite: false });
      const pl = new Mesh(new PlaneGeometry(1.0, 0.85), this.fringe);
      g.add(b1, b2, pl); this.mats[1].push(b1.material as MeshBasicMaterial, b2.material as MeshBasicMaterial);
    }
    // A3 typing: a dispersive prism splitting one beam into five classes
    {
      const g = this.st[2];
      const prism = new Mesh(new CylinderGeometry(0.42, 0.42, 0.9, 3), makeGlass("prism", env, transmission));
      prism.rotation.z = Math.PI / 2; prism.rotation.y = Math.PI / 6;
      g.add(prism); this.prism = prism;
      const inBeam = line([new Vector3(-1.3, 0.15, 0), new Vector3(-0.2, 0.02, 0)], HEX.ivory, 4);
      g.add(inBeam); this.mats[2].push(inBeam.material as LineBasicMaterial);
      const classes: [string, number][] = [["#6b7280", 1.4], [HEX.ivory, 3], [HEX.violet, 3], [HEX.mint, 3], [HEX.amber, 5]];
      classes.forEach(([c, k], j) => {
        const b = line([new Vector3(0.2, 0, 0), new Vector3(1.4, -0.42 + j * 0.21, 0.1 * j)], c, k);
        g.add(b); this.prismBeams.push(b); this.mats[2].push(b.material as LineBasicMaterial);
      });
    }
    // A4 scope: two rings, nested and turning
    {
      const g = this.st[3];
      const r1 = new Mesh(new TorusGeometry(0.62, 0.012, 8, 160), glow(HEX.cyan, 3.2));
      const r2 = new Mesh(new TorusGeometry(0.3, 0.012, 8, 120), glow(HEX.amber, 3.6));
      r2.position.set(0.22, -0.12, 0);
      g.add(r1, r2); this.rings.push(r1, r2); this.mats[3].push(r1.material as MeshBasicMaterial, r2.material as MeshBasicMaterial);
    }
    // the contract gate
    {
      this.gate = new Group();
      const fr = new LineSegments(new EdgesGeometry(new PlaneGeometry(1.1, 1.5)), new LineBasicMaterial({ color: lin(HEX.ivory, 3.5), transparent: true, blending: AdditiveBlending }));
      // the plate hangs inside the frame, so it never rises into the page header
      const lab = new Mesh(new PlaneGeometry(1.0, 0.25), new MeshBasicMaterial({ map: labelTexture("CONTRACT", "v1.2.0 · frozen JSON"), transparent: true }));
      lab.position.y = 0.52;
      this.gate.add(fr, lab);
      this.gate.position.set((STATION_X[3] + STATION_X[4]) / 2, Y, 0);
      this.group.add(this.gate);
    }
    // B1 probing: a laser scan reveals a condition the direct ask missed
    {
      const g = this.st[4];
      const tex = passageTexture({ tag: "hidden condition · revealed", text: "…for Global Traveller account holders", meta: "grounding gate: entailed ✓", accent: HEX.amber }, 1.6 / 0.62).tex;
      this.reveal = new ShaderMaterial({ vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);}`,
        fragmentShader: REVEAL, uniforms: { uTex: { value: tex }, uScan: { value: 0 }, uA: { value: 0 } }, transparent: true });
      g.add(new Mesh(new PlaneGeometry(1.6, 0.62), this.reveal));
    }
    // B2 composition: blocks snapping together
    {
      const g = this.st[5];
      for (let k = 0; k < 6; k++) {
        const m = new Mesh(new BoxGeometry(0.22, 0.22, 0.22), k < 4 ? glow(HEX.cyan, 1.8) : glow(HEX.amber, 2.6));
        g.add(m); this.blocks.push(m); this.mats[5].push(m.material as MeshBasicMaterial);
      }
    }
    // B3 generation: the answer types itself, citations glowing
    {
      const g = this.st[6];
      const tex = passageTexture({ tag: "scoped answer · per p0 · per p1", text: "3% fee by default; waived for premium-tier cardholders.", meta: "faithfulness: passes", accent: HEX.cyan }, 1.7 / 0.66).tex;
      this.typer = new ShaderMaterial({ vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);}`,
        fragmentShader: REVEAL.replace("vec3(1.0, 0.35, 0.25)", "vec3(0.5, 0.95, 1.0)"), uniforms: { uTex: { value: tex }, uScan: { value: 0 }, uA: { value: 0 } }, transparent: true });
      g.add(new Mesh(new PlaneGeometry(1.7, 0.66), this.typer));
    }
    this.packet = new Mesh(new SphereGeometry(0.06, 16, 12), glow("#ffffff", 6));
    this.group.add(this.packet);
  }

  update(p: WorldParams, ctx: FrameContext) {
    const v = p.stations * (1 - p.dim * 0.7);
    this.group.visible = v > 0.003;
    if (!this.group.visible) return;
    const t = ctx.t;
    const act = (i: number) => stationActivation(p.flow, i);
    this.st.forEach((g, i) => {
      const a = act(i);
      const k = v * (0.22 + 0.78 * a);
      this.mats[i].forEach(m => (m.opacity = k));
      g.scale.setScalar(0.85 + 0.15 * a);
    });
    this.fringe.uniforms.uK.value = act(1); this.fringe.uniforms.uT.value = t; this.fringe.uniforms.uA.value = v;
    // glass cannot fade with the rest of a dim station, so it shows only near its own stop
    this.prism.visible = act(2) > 0.12;
    this.prismBeams.forEach((b, j) => ((b.material as LineBasicMaterial).opacity = v * act(2) * (j === 4 ? 1 : 0.7)));
    this.rings[0].rotation.set(0.6 + t * 0.3, t * 0.5, 0); this.rings[1].rotation.set(-0.4 - t * 0.4, t * 0.7, 0);
    this.reveal.uniforms.uScan.value = Math.min(1, act(4) * 1.25); this.reveal.uniforms.uA.value = v * (0.35 + 0.65 * act(4));
    const asm = act(5);
    this.blocks.forEach((b, k) => {
      const home = new Vector3((k % 3 - 1) * 0.23, (Math.floor(k / 3) - 0.5) * 0.23, 0);
      const out = new Vector3(Math.cos(k * 1.9) * 0.9, Math.sin(k * 2.3) * 0.6, Math.sin(k) * 0.4);
      b.position.lerpVectors(out, home, asm);
      b.rotation.set((1 - asm) * k, (1 - asm) * k * 0.7, 0);
    });
    this.typer.uniforms.uScan.value = Math.min(1, act(6) * 1.2); this.typer.uniforms.uA.value = v * (0.35 + 0.65 * act(6));
    this.gate.children.forEach(c => {
      const m = (c as Mesh).material as MeshBasicMaterial; m.opacity = v * (0.5 + 0.5 * Math.max(act(3), act(4)));
    });
    const x0 = STATION_X[0] - 1.2, x1 = STATION_X[6] + 1.2;
    this.packet.position.set(x0 + (x1 - x0) * p.flow, Y - 0.62, 0.05);
    (this.packet.material as MeshBasicMaterial).opacity = v;
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
