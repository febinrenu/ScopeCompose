/*  Act 00: the brand mark as an object.
 *  A clear glass sphere (the rule) holding a smaller amber one (the
 *  exception) -- the mark in the corner of the page, made physical. The
 *  particle wordmark behind it bends through the glass.
 */
import { AdditiveBlending, Group, Mesh, MeshBasicMaterial, SphereGeometry, TorusGeometry, type Texture } from "three";
import type { Actor, FrameContext } from "../core/actor";
import type { WorldParams } from "../../acts/keyframes";
import { makeGlass } from "../materials/glass";
import { HERO_CENTRE, HERO_EXC, HERO_EXC_R, HERO_RULE_R } from "../targets";
import { HEX, lin } from "../core/color";

export class Hero implements Actor {
  readonly group = new Group();
  private rule: Mesh;
  private exc: Mesh;
  private ring: Mesh;
  private ring2: Mesh;
  private core: Mesh;

  constructor(env: Texture, transmission: boolean) {
    this.rule = new Mesh(new SphereGeometry(HERO_RULE_R, 128, 96), makeGlass("clear", env, transmission));
    this.exc = new Mesh(new SphereGeometry(HERO_EXC_R, 96, 72), makeGlass("amber", env, transmission));
    // the exception's warm core: a small HDR heart that blooms through the glass
    this.core = new Mesh(new SphereGeometry(HERO_EXC_R * 0.42, 32, 24), new MeshBasicMaterial({ color: lin(HEX.amber, 3.2) }));
    this.ring = new Mesh(new TorusGeometry(HERO_RULE_R * 1.004, 0.006, 8, 256),
      new MeshBasicMaterial({ color: lin(HEX.cyan, 4), transparent: true, blending: AdditiveBlending, depthWrite: false }));
    this.ring2 = new Mesh(new TorusGeometry(HERO_EXC_R * 1.01, 0.004, 8, 160),
      new MeshBasicMaterial({ color: lin(HEX.amber, 4), transparent: true, blending: AdditiveBlending, depthWrite: false }));
    this.ring.rotation.x = Math.PI / 2;
    this.group.add(this.rule, this.exc, this.core, this.ring, this.ring2);
    this.group.position.set(...HERO_CENTRE);
  }

  update(p: WorldParams, ctx: FrameContext) {
    const v = p.hero;
    this.group.visible = v > 0.003;
    if (!this.group.visible) return;
    const s = 0.35 + 0.65 * v;
    this.group.scale.setScalar(s);
    const t = ctx.t;
    // dragging turns the glass itself, further than the camera orbits
    this.group.rotation.y = t * 0.12 + p.heroOpen * 0.6 + ctx.spin.yaw * 1.4;
    this.group.rotation.x = Math.sin(t * 0.17) * 0.08;
    // the exception sits inside the rule, offset; heroOpen lets it drift outward
    const ox = HERO_EXC[0] - HERO_CENTRE[0], oy = HERO_EXC[1] - HERO_CENTRE[1], oz = HERO_EXC[2] - HERO_CENTRE[2];
    const k = 1 + p.heroOpen * 0.8;
    this.exc.position.set(ox * k, oy * k, oz * k);
    this.core.position.copy(this.exc.position);
    this.ring2.position.copy(this.exc.position);
    // the exception's ring lags a throw, like something loose inside the glass
    this.ring2.rotation.set(Math.PI / 2 + Math.sin(t * 0.3) * 0.3 - ctx.spin.vel * 0.12, t * 0.4, ctx.spin.vel * 0.08);
    this.ring.rotation.z = t * 0.05;
    (this.ring.material as MeshBasicMaterial).opacity = v;
    (this.ring2.material as MeshBasicMaterial).opacity = v;
  }

  dispose() {
    for (const m of [this.rule, this.exc, this.core, this.ring, this.ring2]) {
      m.geometry.dispose(); (m.material as MeshBasicMaterial).dispose();
    }
  }
}
