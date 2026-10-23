/*  Seeded randomness. Every "random" placement in the world comes from here,
 *  so the world is identical across reloads and the reversibility check can
 *  compare frames exactly.
 */

/** mulberry32: small, fast, good enough for placement. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A stable hash of an integer to [0, 1). */
export function hash1(n: number): number {
  let x = Math.imul((n | 0) ^ 0x9E3779B9, 0x85EBCA6B);
  x ^= x >>> 13; x = Math.imul(x, 0xC2B2AE35); x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

/** Gaussian sample via Box–Muller, from a seeded generator. */
export function gauss(r: () => number): number {
  const u = Math.max(1e-9, r()), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
