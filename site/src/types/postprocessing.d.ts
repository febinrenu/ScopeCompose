/*  postprocessing 6.39 ships without its .d.ts files (the package.json points
 *  at build/types/, which is not in the tarball). This declares only the
 *  surface the site uses, so the type checker still guards our calls.
 */
declare module "postprocessing" {
  import type { Camera, Scene, Texture, Uniform, WebGLRenderer, WebGLRenderTarget, TextureDataType } from "three";

  export class Pass {
    enabled: boolean;
    renderToScreen: boolean;
    setSize(width: number, height: number): void;
    dispose(): void;
  }
  export class EffectComposer {
    constructor(renderer: WebGLRenderer, options?: { frameBufferType?: TextureDataType; multisampling?: number; depthBuffer?: boolean; stencilBuffer?: boolean });
    inputBuffer: WebGLRenderTarget;
    addPass(pass: Pass, index?: number): void;
    removePass(pass: Pass): void;
    render(deltaTime?: number): void;
    setSize(width?: number, height?: number, updateStyle?: boolean): void;
    setRenderer(renderer: WebGLRenderer): void;
    multisampling: number;
    dispose(): void;
  }
  export class RenderPass extends Pass {
    constructor(scene: Scene, camera: Camera);
  }
  export const BlendFunction: Record<string, number>;
  export const EffectAttribute: { NONE: number; DEPTH: number; CONVOLUTION: number };
  export class Effect {
    constructor(name: string, fragmentShader: string, options?: {
      attributes?: number; blendFunction?: number; defines?: Map<string, string>;
      uniforms?: Map<string, Uniform>; vertexShader?: string;
    });
    uniforms: Map<string, Uniform>;
    blendMode: { opacity: Uniform };
    dispose(): void;
  }
  export class EffectPass extends Pass {
    constructor(camera: Camera, ...effects: Effect[]);
  }
  export class BloomEffect extends Effect {
    constructor(options?: {
      intensity?: number; luminanceThreshold?: number; luminanceSmoothing?: number;
      mipmapBlur?: boolean; radius?: number; levels?: number; resolutionScale?: number;
    });
    intensity: number;
    luminanceMaterial: { threshold: number; smoothing: number };
  }
  export class DepthOfFieldEffect extends Effect {
    constructor(camera: Camera, options?: { focusDistance?: number; focusRange?: number; bokehScale?: number; resolutionScale?: number });
    bokehScale: number;
    cocMaterial: { focusDistance: number; focusRange: number };
  }
  export class VignetteEffect extends Effect {
    constructor(options?: { offset?: number; darkness?: number });
    offset: number;
    darkness: number;
  }
  export const ToneMappingMode: Record<string, number>;
  export class ToneMappingEffect extends Effect {
    constructor(options?: { mode?: number });
  }
  export const SMAAPreset: Record<string, number>;
  export class SMAAEffect extends Effect {
    constructor(options?: { preset?: number });
  }
  export type { Texture };
}
