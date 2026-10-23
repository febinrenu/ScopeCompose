/*  sampleActs(cursor) -> WorldParams. Pure: same cursor, same frame.
 *
 *  Between stops, parameters ease with a held plateau at each end, so the
 *  world rests on each keyframe for a moment instead of sliding through it.
 *  The camera uses a Catmull-Rom spline through the stops, so its path
 *  curves through the world instead of zig-zagging between waypoints.
 */
import { KEYFRAMES, PARAM_KEYS, type WorldParams } from "./keyframes";

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));

/** Smoothstep with a plateau: holds at 0 for the first `hold`, at 1 for the last. */
export function plateau(t: number, hold = 0.18): number {
  const u = clamp((t - hold) / (1 - 2 * hold));
  return u * u * (3 - 2 * u);
}

function catmull(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

const CAMERA = new Set(["camX", "camY", "camZ", "tgtX", "tgtY", "tgtZ", "subX"]);

export function sampleActs(cursor: number, frames: WorldParams[] = KEYFRAMES): WorldParams {
  const last = frames.length - 1;
  const c = clamp(Number.isFinite(cursor) ? cursor : 0, 0, last);
  const i = Math.min(Math.floor(c), last - 1 < 0 ? 0 : last - 1);
  const f = c - i;
  if (last === 0) return { ...frames[0] };
  const a = frames[i], b = frames[i + 1];
  const e = plateau(f);
  const out = {} as WorldParams;
  for (const k of PARAM_KEYS) {
    if (CAMERA.has(k)) {
      const p0 = frames[Math.max(0, i - 1)][k], p3 = frames[Math.min(last, i + 2)][k];
      out[k] = catmull(p0, a[k], b[k], p3, e);
    } else {
      out[k] = a[k] + (b[k] - a[k]) * e;
    }
  }
  return out;
}

/**
 * Where a section's scroll progress puts the cursor. While the section is
 * pinned (p <= hold) the cursor walks the act's own keyframes k0 .. k1-1;
 * while it scrolls out it travels on to the next act's first keyframe k1.
 * So an act's world never starts turning into the next act while that act's
 * copy is still on screen.
 */
export function cursorFor(k0: number, k1: number, hold: number, p: number): number {
  const inner = Math.max(0, k1 - 1 - k0);
  const q = clamp(p);
  if (hold >= 1 || q <= hold) return k0 + (hold > 0 ? Math.min(1, q / hold) : 1) * inner;
  return k1 - 1 + (q - hold) / (1 - hold);
}
