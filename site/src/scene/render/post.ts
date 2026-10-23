/*  Post-processing (tiers 2–3), loaded as its own chunk.
 *
 *    RenderPass (HDR, half float)
 *    -> [DepthOfField (tier 3)] + Bloom      threshold 1.0: only HDR glows
 *    -> Lens                                 velocity warp, aberration, exposure
 *    -> Vignette + Tone mapping (NEUTRAL)    keeps ivory type and brand colours true
 *    -> SMAA                                 when MSAA is off
 *
 *  Every uniform comes from WorldParams (plus scroll velocity, which snaps
 *  to exactly zero at rest).
 */
import { HalfFloatType, Uniform, type PerspectiveCamera, type Scene, type WebGLRenderer } from "three";
import {
  BloomEffect, DepthOfFieldEffect, Effect, EffectAttribute, EffectComposer, EffectPass, RenderPass,
  SMAAEffect, ToneMappingEffect, ToneMappingMode, VignetteEffect,
} from "postprocessing";

const LENS = /* glsl */ `
  uniform float uVel; uniform float uCA; uniform float uExposure;
  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor){
    vec2 c = uv - 0.5;
    // scroll speed stretches the frame toward its edges
    vec2 w = uv + vec2(0.0, c.y * abs(c.y) * uVel * 0.35);
    float r2 = dot(c, c);
    vec2 dir = c * (uCA * 0.0055 + abs(uVel) * 0.012) * (0.25 + r2 * 2.4);
    float r = texture2D(inputBuffer, w + dir).r;
    vec2 gb = texture2D(inputBuffer, w).ga;
    float b = texture2D(inputBuffer, w - dir).b;
    outputColor = vec4(vec3(r, gb.x, b) * uExposure, inputColor.a);
  }`;

class LensEffect extends Effect {
  constructor() {
    super("LensEffect", LENS, {
      attributes: EffectAttribute.CONVOLUTION,
      uniforms: new Map<string, Uniform>([["uVel", new Uniform(0)], ["uCA", new Uniform(0.12)], ["uExposure", new Uniform(1)]]),
    });
  }
}

export interface PostParams { bloom: number; dof: number; focus: number; vign: number; ca: number; vel: number; exposure: number }

export class Post {
  private composer: EffectComposer;
  private bloom: BloomEffect;
  private dof: DepthOfFieldEffect | null;
  private lens: LensEffect;
  private vign: VignetteEffect;

  constructor(renderer: WebGLRenderer, scene: Scene, camera: PerspectiveCamera, opts: { tier: number; msaa: boolean }) {
    this.composer = new EffectComposer(renderer, { frameBufferType: HalfFloatType, multisampling: opts.msaa ? 4 : 0 });
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new BloomEffect({
      mipmapBlur: true, luminanceThreshold: 1.0, luminanceSmoothing: 0.3, intensity: 1.0, radius: 0.78,
      levels: opts.tier >= 3 ? 7 : 5, resolutionScale: opts.tier >= 3 ? 1 : 0.5,
    });
    this.dof = opts.tier >= 3 ? new DepthOfFieldEffect(camera, { focusDistance: 9, focusRange: 3.5, bokehScale: 2, resolutionScale: 0.5 }) : null;
    this.composer.addPass(new EffectPass(camera, ...(this.dof ? [this.dof, this.bloom] : [this.bloom])));
    this.lens = new LensEffect();
    this.composer.addPass(new EffectPass(camera, this.lens));
    this.vign = new VignetteEffect({ offset: 0.42, darkness: 0.33 });
    this.composer.addPass(new EffectPass(camera, this.vign, new ToneMappingEffect({ mode: ToneMappingMode.NEUTRAL })));
    if (!opts.msaa) this.composer.addPass(new EffectPass(camera, new SMAAEffect()));
  }

  set(p: PostParams) {
    this.bloom.intensity = p.bloom;
    if (this.dof) {
      this.dof.cocMaterial.focusDistance = p.focus;
      // blur ramps from zero at the focus distance, so the range is generous:
      // the subject's whole depth stays sharp, the far field still melts
      this.dof.cocMaterial.focusRange = Math.max(2.4, p.focus * 0.55);
      this.dof.bokehScale = p.dof * 2.6;
    }
    this.lens.uniforms.get("uVel")!.value = p.vel;
    this.lens.uniforms.get("uCA")!.value = p.ca;
    this.lens.uniforms.get("uExposure")!.value = p.exposure;
    // a gentle vignette: it frames the stage without darkening text near the edges
    this.vign.darkness = p.vign * 0.6;
  }

  /** Governor step-down: drop depth of field. */
  dropDof() { if (this.dof) { this.dof.bokehScale = 0; this.dof = null; } }

  /** The buffer the scene is rendered into, so shaders can be compiled for it ahead of time. */
  get target() { return this.composer.inputBuffer; }

  setSize(w: number, h: number) { this.composer.setSize(w, h, false); }
  render() { this.composer.render(); }
  dispose() { this.composer.dispose(); }
}
