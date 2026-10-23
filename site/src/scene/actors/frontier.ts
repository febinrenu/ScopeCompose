/*  Act 09: the stage-1 detector's REAL score for every one of the 128 corpus
 *  instances, one bar each, against its two thresholds. The scores are
 *  computed by screenPair() in the main bundle and handed over as numbers --
 *  the scene never touches the engine. Bars that clear the conflict
 *  threshold glow; everything else is quiet glass.
 */
import {
  AdditiveBlending, BoxGeometry, BufferGeometry, Color, DoubleSide, Group, InstancedMesh, LineBasicMaterial, LineDashedMaterial,
  LineLoop, Matrix4, Mesh, MeshBasicMaterial, MeshPhysicalMaterial, PlaneGeometry, Vector3, type Texture,
} from "three";
import type { Actor, FrameContext } from "../core/actor";
import type { WorldParams } from "../../acts/keyframes";
import { HEX, lin } from "../core/color";

export interface FrontierPoint { score: number; ty: string; fired: boolean }

const H = 3.2, SP = 0.42, Y0 = -0.6;

export class Frontier implements Actor {
  readonly group = new Group();
  private bars: InstancedMesh;
  private glowBars: InstancedMesh;
  private heights: number[] = [];
  private slots: Vector3[] = [];
  private fired: boolean[] = [];
  private conflict: LineLoop;
  private floor: LineLoop;
  private sheet: Mesh;
  private m = new Matrix4();
  readonly cols = 16;
  readonly rows: number;

  constructor(points: FrontierPoint[], env: Texture) {
    const n = points.length;
    this.rows = Math.ceil(n / this.cols);
    const geo = new BoxGeometry(0.17, 1, 0.17); geo.translate(0, 0.5, 0);
    this.bars = new InstancedMesh(geo, new MeshPhysicalMaterial({
      envMap: env, envMapIntensity: 1.1, roughness: 0.18, metalness: 0.1, clearcoat: 1, transparent: true, opacity: 0,
    }), n);
    this.glowBars = new InstancedMesh(geo, new MeshBasicMaterial({ color: lin(HEX.cyan, 3.6), transparent: true, opacity: 0 }), n);
    points.forEach((pt, i) => {
      const c = i % this.cols, r = Math.floor(i / this.cols);
      this.slots.push(new Vector3((c - (this.cols - 1) / 2) * SP, Y0, (r - (this.rows - 1) / 2) * SP));
      this.heights.push(Math.max(0.02, pt.score) * H);
      this.fired.push(pt.fired);
      const col = pt.ty === "conditional" ? HEX.amber : pt.ty === "no_conflict" ? "#4a5163" : "#d9d6cd";
      this.bars.setColorAt(i, new Color(col));
    });
    this.bars.frustumCulled = this.glowBars.frustumCulled = false;
    const w = this.cols * SP + 0.6, d = this.rows * SP + 0.6;
    const rect = (y: number) => new BufferGeometry().setFromPoints([
      new Vector3(-w / 2, y, -d / 2), new Vector3(w / 2, y, -d / 2), new Vector3(w / 2, y, d / 2), new Vector3(-w / 2, y, d / 2),
    ]);
    this.conflict = new LineLoop(rect(Y0 + 0.5 * H), new LineBasicMaterial({ color: lin(HEX.coral, 3.5), transparent: true, opacity: 0 }));
    this.floor = new LineLoop(rect(Y0 + 0.35 * H), new LineDashedMaterial({ color: lin(HEX.amber, 2.6), dashSize: 0.1, gapSize: 0.08, transparent: true, opacity: 0 }));
    this.floor.computeLineDistances();
    this.sheet = new Mesh(new PlaneGeometry(w, d), new MeshBasicMaterial({ color: lin(HEX.coral, 0.5), transparent: true, opacity: 0, side: DoubleSide, blending: AdditiveBlending, depthWrite: false }));
    this.sheet.rotation.x = -Math.PI / 2; this.sheet.position.y = Y0 + 0.5 * H;
    this.group.add(this.bars, this.glowBars, this.conflict, this.floor, this.sheet);
    this.group.rotation.y = -0.35;
  }

  anchors() {
    const w = this.cols * SP + 0.6, d = this.rows * SP + 0.6;
    this.group.updateMatrixWorld();
    return [new Vector3(-w / 2, Y0 + 0.5 * H + 0.08, -d / 2), new Vector3(-w / 2, Y0 + 0.35 * H + 0.08, -d / 2)]
      .map(v => v.applyMatrix4(this.group.matrixWorld));
  }

  update(p: WorldParams, _ctx: FrameContext) {
    const vis = p.frontier;
    this.group.visible = vis > 0.003;
    if (!this.group.visible) return;
    const grow = Math.min(1, vis * 1.2);
    for (let i = 0; i < this.heights.length; i++) {
      // each bar rises in turn, front rows first
      const g = Math.min(1, Math.max(0, grow * 1.6 - (i / this.heights.length) * 0.6));
      this.m.makeScale(1, Math.max(0.0001, this.heights[i] * g), 1);
      this.m.setPosition(this.slots[i]);
      this.bars.setMatrixAt(i, this.m);
      if (!this.fired[i]) this.m.makeScale(0, 0, 0);
      else { this.m.makeScale(1.12, Math.max(0.0001, this.heights[i] * g), 1.12); this.m.setPosition(this.slots[i]); }
      this.glowBars.setMatrixAt(i, this.m);
    }
    this.bars.instanceMatrix.needsUpdate = true;
    this.glowBars.instanceMatrix.needsUpdate = true;
    (this.bars.material as MeshPhysicalMaterial).opacity = vis;
    (this.glowBars.material as MeshBasicMaterial).opacity = vis;
    (this.conflict.material as LineBasicMaterial).opacity = vis;
    (this.floor.material as LineDashedMaterial).opacity = vis * 0.9;
    (this.sheet.material as MeshBasicMaterial).opacity = vis * 0.18;
  }

  dispose() {
    this.bars.geometry.dispose(); (this.bars.material as MeshPhysicalMaterial).dispose();
    (this.glowBars.material as MeshBasicMaterial).dispose();
    for (const l of [this.conflict, this.floor]) { l.geometry.dispose(); (l.material as LineBasicMaterial).dispose(); }
    this.sheet.geometry.dispose(); (this.sheet.material as MeshBasicMaterial).dispose();
  }
}
