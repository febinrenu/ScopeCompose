import type { Object3D, PerspectiveCamera, Texture, Vector2, Vector3 } from "three";
import type { WorldParams } from "../../acts/keyframes";
import type { Quality } from "../../env/tier";

/** What every actor gets each frame. Actors are pure functions of (params, ctx). */
export interface FrameContext {
  /** Ambient time in seconds; frozen at 0 in strict ("still") mode. */
  t: number;
  /** 1 when ambient motion is allowed, 0 in strict mode. */
  ambient: number;
  /** The pointer on the subject plane, in world space, or null. */
  pointer: Vector3 | null;
  /** The x-ray lens in normalised device coordinates (x, y), radius in NDC y units. */
  lens: { x: number; y: number; r: number };
  resolution: Vector2;
  px: number;
  camera: PerspectiveCamera;
  quality: Quality;
  env: Texture;
  /** The drag orbit: yaw (rad) and its angular velocity (rad/s); both 0 in strict mode. */
  spin: { yaw: number; vel: number };
}

export interface Actor {
  readonly group: Object3D;
  update(p: WorldParams, ctx: FrameContext): void;
  dispose(): void;
}
