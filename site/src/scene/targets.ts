/*  The shapes the particle field can take, as plain arrays.
 *
 *  Pure and seeded: the same inputs give the same shapes, so the particle
 *  field is a function of the story cursor and nothing else. The word's
 *  outline is sampled from text drawn by the caller (a canvas in the page,
 *  synthetic points in tests).
 */
import { gauss, rng } from "./core/rand";

export const HERO_CENTRE: [number, number, number] = [1.8, 0, 0];
export const HERO_RULE_R = 2.05;
export const HERO_EXC: [number, number, number] = [2.4, -0.4, 0.55];
export const HERO_EXC_R = 0.78;

export interface Node3 { x: number; y: number; z: number }

/** One node per corpus instance, spread through a deep volume. */
export function netNodes(count: number, seed = 11): Node3[] {
  const r = rng(seed);
  const out: Node3[] = [];
  for (let i = 0; i < count; i++) {
    const z = -1 - Math.pow(r(), 0.8) * 22;
    const spread = 1 + (-z) * 0.32;
    out.push({ x: (r() - 0.5) * 2 * (5 + spread * 1.1), y: (r() - 0.5) * 2 * (2.4 + spread * 0.45), z });
  }
  return out;
}

export interface Targets {
  n: number;
  word: Float32Array; mark: Float32Array; net: Float32Array;
  field: Float32Array; shell: Float32Array; grid: Float32Array;
  /** x: random, y: random, z: brightness, w: role (shell A < 0.62 <= B) */
  seed: Float32Array;
}

/**
 * @param n        particle count
 * @param wordPts  [x0,y0,x1,y1,...] outline points of the wordmark, in world units
 * @param nodes    the constellation (one per corpus instance)
 */
export function buildTargets(n: number, wordPts: Float32Array, nodes: Node3[], seed = 3): Targets {
  const g = slices(n, wordPts, nodes, seed);
  let step = g.next();
  while (!step.done) step = g.next();
  return step.value;
}

/** The same shapes, built in slices with the main thread handed back between them
 *  (262k particles in one go was a long task at boot). Identical output. */
export async function buildTargetsAsync(n: number, wordPts: Float32Array, nodes: Node3[], seed = 3, yieldTo: () => Promise<void>): Promise<Targets> {
  const g = slices(n, wordPts, nodes, seed);
  let step = g.next();
  while (!step.done) { await yieldTo(); step = g.next(); }
  return step.value;
}

const SLICE = 8192;      // about 15 ms a slice on a mid phone: under the long-task line

function* slices(n: number, wordPts: Float32Array, nodes: Node3[], seed: number): Generator<void, Targets> {
  const r = rng(seed);
  const word = new Float32Array(n * 3), mark = new Float32Array(n * 3), net = new Float32Array(n * 3);
  const field = new Float32Array(n * 3), shell = new Float32Array(n * 3), grid = new Float32Array(n * 3);
  const sd = new Float32Array(n * 4);
  const wp = Math.max(1, wordPts.length / 2);
  const put = (a: Float32Array, i: number, x: number, y: number, z: number) => { a[i * 3] = x; a[i * 3 + 1] = y; a[i * 3 + 2] = z; };

  for (let i = 0; i < n; i++) {
    const s0 = r(), s1 = r(), s2 = r(), role = r();
    sd[i * 4] = s0; sd[i * 4 + 1] = s1; sd[i * 4 + 2] = 0.35 + 0.65 * Math.pow(s2, 3); sd[i * 4 + 3] = role;

    // word: the outline, with a little depth so it reads as a field of light
    const k = Math.floor(r() * wp) * 2;
    put(word, i, (wordPts[k] ?? 0) + gauss(r) * 0.012, (wordPts[k + 1] ?? 0) + gauss(r) * 0.012, -3.2 + gauss(r) * 0.08);

    // mark: two glowing shells hugging the glass sculpture
    {
      const u = r() * 2 - 1, th = r() * Math.PI * 2, rr = Math.sqrt(1 - u * u);
      const onExc = role > 0.74;
      const c = onExc ? HERO_EXC : HERO_CENTRE, R = (onExc ? HERO_EXC_R : HERO_RULE_R) * (1.06 + Math.abs(gauss(r)) * 0.05);
      put(mark, i, c[0] + rr * Math.cos(th) * R, c[1] + u * R, c[2] + rr * Math.sin(th) * R);
    }

    // net: clusters at each node, and filaments between related nodes
    {
      const a = Math.floor(r() * nodes.length), A = nodes[a];
      if (r() < 0.66) {
        put(net, i, A.x + gauss(r) * 0.14, A.y + gauss(r) * 0.14, A.z + gauss(r) * 0.14);
      } else {
        const B = nodes[(a * 7 + 3) % nodes.length], t = r();
        put(net, i, A.x + (B.x - A.x) * t + gauss(r) * 0.02, A.y + (B.y - A.y) * t + gauss(r) * 0.02, A.z + (B.z - A.z) * t);
      }
    }

    // field: deep, quiet starfield
    put(field, i, (r() - 0.5) * 34, (r() - 0.5) * 20, -26 + r() * 30);

    // shell: a unit direction; the shader places it on volume A or B from the scope parameters
    {
      const u = r() * 2 - 1, th = r() * Math.PI * 2, rr = Math.sqrt(1 - u * u);
      const thick = 1 + gauss(r) * 0.025;
      put(shell, i, rr * Math.cos(th) * thick, u * thick, rr * Math.sin(th) * thick);
    }

    // grid: a lattice of light, and the brand mark forming in the sky
    if (role < 0.2) {
      const inner = r() < 0.32, th = r() * Math.PI * 2;
      const R = inner ? 1.15 : 3.1, cx = inner ? 0.95 : 0, cy = inner ? -0.55 : 0;
      put(grid, i, cx + Math.cos(th) * R + gauss(r) * 0.02, 6.2 + cy + Math.sin(th) * R + gauss(r) * 0.02, -9);
    } else {
      const along = r() < 0.5, lane = Math.round((r() - 0.5) * 28) * 1.0, t = (r() - 0.5) * 30;
      const x = along ? t : lane, z = along ? lane * 0.8 - 8 : t * 0.75 - 8;
      put(grid, i, x, -2 + Math.sin(x * 0.35) * 0.12 + Math.cos(z * 0.4) * 0.12, z);
    }
    if ((i + 1) % SLICE === 0) yield;
  }
  return { n, word, mark, net, field, shell, grid, seed: sd };
}
