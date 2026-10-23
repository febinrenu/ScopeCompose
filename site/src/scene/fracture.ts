/*  The shatter, as pure functions.
 *
 *  voronoiCells(): the slab's face cut into convex cells by half-plane
 *  clipping (no dependency). shardPose(): where a shard is, as a function of
 *  burn / compose / restore only -- never of time -- so scrolling back flies
 *  every shard home along exactly the path it left on.
 */
import { rng } from "./core/rand";

export type Pt = [number, number];

/** Clip convex polygon `poly` to the half-plane closer to `a` than to `b`. */
function clipBisector(poly: Pt[], a: Pt, b: Pt): Pt[] {
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
  const nx = b[0] - a[0], ny = b[1] - a[1];
  const side = (p: Pt) => (p[0] - mx) * nx + (p[1] - my) * ny;   // < 0: a's side
  const out: Pt[] = [];
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const sp = side(p), sq = side(q);
    if (sp <= 0) out.push(p);
    if ((sp < 0 && sq > 0) || (sp > 0 && sq < 0)) {
      const t = sp / (sp - sq);
      out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
    }
  }
  return out;
}

export interface Cell { poly: Pt[]; centroid: Pt; area: number; seed: number }

/** Voronoi cells of `n` seeded sites inside the w×h rectangle centred on 0. */
export function voronoiCells(n: number, w: number, h: number, seed = 7): Cell[] {
  const r = rng(seed);
  // Denser near the right edge, where the exception breaks first.
  const sites: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const u = Math.pow(r(), 0.75);
    sites.push([(u - 0.5) * w, (r() - 0.5) * h]);
  }
  const rect: Pt[] = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]];
  const cells: Cell[] = [];
  sites.forEach((s, i) => {
    let poly = rect;
    for (let j = 0; j < sites.length && poly.length; j++) if (j !== i) poly = clipBisector(poly, s, sites[j]);
    if (poly.length < 3) return;
    let area = 0, cx = 0, cy = 0;
    for (let k = 0; k < poly.length; k++) {
      const [x0, y0] = poly[k], [x1, y1] = poly[(k + 1) % poly.length];
      const c = x0 * y1 - x1 * y0;
      area += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c;
    }
    area /= 2;
    if (Math.abs(area) < 1e-6) return;
    cells.push({ poly: area < 0 ? poly.slice().reverse() : poly, centroid: [cx / (6 * area), cy / (6 * area)], area: Math.abs(area), seed: r() });
  });
  return cells;
}

export interface Pose { x: number; y: number; z: number; qx: number; qy: number; qz: number; qw: number; s: number }

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (x: number) => { const t = clamp(x); return t * t * (3 - 2 * t); };

function axisAngle(ax: number, ay: number, az: number, ang: number): [number, number, number, number] {
  const l = Math.hypot(ax, ay, az) || 1, s = Math.sin(ang / 2);
  return [ax / l * s, ay / l * s, az / l * s, Math.cos(ang / 2)];
}

function slerpToIdentity(q: [number, number, number, number], t: number): [number, number, number, number] {
  // nlerp is ample for a settling shard
  const x = q[0] * (1 - t), y = q[1] * (1 - t), z = q[2] * (1 - t), w = q[3] * (1 - t) + t;
  const l = Math.hypot(x, y, z, w) || 1;
  return [x / l, y / l, z / l, w / l];
}

export interface Slab { x: number; y: number; s: number }

/**
 * Where shard `cell` is.
 *  - `from`: the slab's pose when it shattered (a fixed keyframe).
 *  - burn 0..1: the break travels right-to-left; each shard flies on a seeded ballistic arc.
 *  - compose 0..1: each shard flies to its own seat in the exception's new place `to`.
 */
export function shardPose(cell: Cell, w: number, burn: number, compose: number, from: Slab, to: Slab): Pose {
  const [cx, cy] = cell.centroid;
  const r = rng(Math.floor(cell.seed * 1e9) + 3);
  const delay = (1 - (cx / w + 0.5)) * 0.5 + r() * 0.1;
  const t = clamp((burn * 1.35 - delay) / 0.75);
  const T = t * 1.6;
  // outward from the break point (the right edge, middle height), with lift and depth
  const dx = cx - w * 0.45, dy = cy;
  const dl = Math.hypot(dx, dy) || 1;
  const sp = 1.1 + r() * 1.6;
  const vx = (dx / dl) * sp * 0.6 + 0.35 + r() * 0.6;
  const vy = (dy / dl) * sp * 0.5 + 0.55 + r() * 0.7;
  const vz = 0.6 + r() * 1.8;
  const g = -0.9;
  const fx = from.x + cx * from.s + vx * T;
  const fy = from.y + cy * from.s + vy * T + 0.5 * g * T * T;
  const fz = vz * T;
  const spin = t * (2.5 + r() * 7);
  const qf = axisAngle(r() - 0.5, r() - 0.5, r() - 0.5, spin);

  // composition: staggered return to the seat in the new slab
  const c = smooth((compose * 1.25 - r() * 0.25) / 1.0);
  const sx = to.x + cx * to.s, sy = to.y + cy * to.s;
  const q = slerpToIdentity(qf, c);
  const s = from.s + (to.s - from.s) * c;
  return {
    x: fx + (sx - fx) * c, y: fy + (sy - fy) * c, z: fz * (1 - c),
    qx: q[0], qy: q[1], qz: q[2], qw: q[3], s,
  };
}

/** Launch progress of a shard, for its glow (hot as it leaves). */
export function shardHeat(cell: Cell, w: number, burn: number): number {
  const [cx] = cell.centroid;
  const r = rng(Math.floor(cell.seed * 1e9) + 3);
  const delay = (1 - (cx / w + 0.5)) * 0.5 + r() * 0.1;
  const t = clamp((burn * 1.35 - delay) / 0.75);
  return t > 0 ? 1 - t : 0;
}
