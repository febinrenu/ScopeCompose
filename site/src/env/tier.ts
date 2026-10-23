/*  Quality tiers.
 *
 *  classifyTier() is pure -- it maps what the device reports to a tier, and
 *  is tested on a table of real renderer strings. probeDevice() is the only
 *  part that touches the browser. Unknown hardware lands on the middle tier
 *  and the runtime governor corrects it if frames run long.
 */

export type Tier = 0 | 1 | 2 | 3;

export interface DeviceInfo {
  webgl2: boolean;
  renderer: string;
  cores: number;
  memory: number | null;
  coarse: boolean;
  viewportW: number;
  saveData: boolean;
}

export interface Quality {
  tier: Tier;
  maxDpr: number;
  /** GPU particles in the field (a power of 2). */
  particles: number;
  shards: number;
  embers: number;
  /** Raymarch steps for the applicability volumes. */
  steps: number;
  /** Post-processing (bloom, lens, tone mapping); depth of field only at tier 3. */
  post: boolean;
  dof: boolean;
  /** Real transmission for glass, else reflective "fake glass". */
  transmission: boolean;
  /** Ambient drift frame rate; 0 renders only when something changes. */
  ambientFps: number;
  pointer: boolean;
}

const SOFTWARE = /swiftshader|llvmpipe|software|basic render|mesa offscreen|microsoft basic/i;
const WEAK = /mali-[gt][1-5]\d|adreno \(tm\) [1-5]\d\d|powervr|intel.*(hd|uhd) graphics( \d{3})?\b(?!.*(iris|arc))|gma/i;
const STRONG = /apple m\d|rtx|radeon rx|radeon pro|\barc\b|geforce gtx 1[0-9]{3}|quadro|tesla|a\d{3,4}\b/i;

export function classifyTier(d: DeviceInfo): Tier {
  if (!d.webgl2) return 0;
  if (SOFTWARE.test(d.renderer)) return 0;
  if (d.saveData && (d.memory ?? 8) <= 2) return 0;
  if (WEAK.test(d.renderer) || d.cores <= 4 || (d.memory !== null && d.memory <= 4) || d.saveData) return 1;
  let t: Tier = STRONG.test(d.renderer) ? 3 : 2;
  // Touch devices with small screens get the middle tier at most: battery
  // matters more than the last 10% of particles there.
  if (d.coarse && d.viewportW < 900 && t > 2) t = 2;
  return t;
}

export function qualityFor(tier: Tier): Quality {
  switch (tier) {
    case 0: return { tier, maxDpr: 1, particles: 0, shards: 0, embers: 0, steps: 0, post: false, dof: false, transmission: false, ambientFps: 0, pointer: false };
    case 1: return { tier, maxDpr: 1, particles: 16384, shards: 40, embers: 400, steps: 24, post: false, dof: false, transmission: false, ambientFps: 0, pointer: false };
    case 2: return { tier, maxDpr: 1.5, particles: 65536, shards: 80, embers: 1200, steps: 36, post: true, dof: false, transmission: true, ambientFps: 30, pointer: true };
    case 3: return { tier, maxDpr: 1.75, particles: 262144, shards: 120, embers: 2400, steps: 56, post: true, dof: true, transmission: true, ambientFps: 60, pointer: true };
  }
}

/** Reads the device. Returns the context it created so the renderer can reuse it. */
export function probeDevice(canvas: HTMLCanvasElement, opts: { antialias: boolean }): { info: DeviceInfo; gl: WebGL2RenderingContext | null } {
  let gl: WebGL2RenderingContext | null = null;
  try {
    gl = canvas.getContext("webgl2", {
      antialias: opts.antialias, alpha: true, premultipliedAlpha: true,
      powerPreference: "high-performance", preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;
  } catch { gl = null; }
  let renderer = "";
  if (gl) {
    // RENDERER is unmasked in current browsers; the debug extension is the fallback.
    renderer = String(gl.getParameter(gl.RENDERER) || "");
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    if (ext && (!renderer || /webkit|mozilla/i.test(renderer))) {
      renderer = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || renderer);
    }
  }
  const nav = navigator as Navigator & { deviceMemory?: number; connection?: { saveData?: boolean } };
  return {
    gl,
    info: {
      webgl2: !!gl,
      renderer,
      cores: navigator.hardwareConcurrency || 4,
      memory: typeof nav.deviceMemory === "number" ? nav.deviceMemory : null,
      coarse: matchMedia("(pointer: coarse)").matches,
      viewportW: innerWidth,
      saveData: Boolean(nav.connection?.saveData),
    },
  };
}
