/*  The world's script.
 *
 *  The 3D world is a pure function of one number, the story cursor. Each
 *  keyframe is a COMPLETE set of world parameters (built by carrying the
 *  previous keyframe forward), and sampleActs() interpolates between
 *  neighbours. Nothing accumulates, so scrolling back lands on exactly the
 *  frame you saw on the way down.
 *
 *  Each act's DOM section maps onto a span of keyframes (ACT_FROM).
 */

export const PARAM_KEYS = [
  // camera rail; subX = the act's subject, which portrait screens re-centre on
  "camX", "camY", "camZ", "tgtX", "tgtY", "tgtZ", "subX",
  // grade: exposure, bloom intensity, depth-of-field amount, vignette
  "exposure", "bloom", "dof", "vign",
  // ambience
  "sceneAlpha", "dim",
  // Act 00: the glass sculpture of the brand mark
  "hero", "heroOpen",
  // GPU particles: weights of each target shape, and their look
  "pWord", "pMark", "pNet", "pField", "pShell", "pGrid", "pAlpha", "pHeat", "pTurb",
  // Act 01: retrieval
  "query", "bm25", "dense", "rrf", "cards",
  // Acts 02/03/07: the two glass monoliths, the crack, the shatter, the return
  "duel", "aX", "aY", "aS", "bX", "bY", "bS", "select", "caret", "crack", "burn", "compose", "restore",
  "socket", "xray",
  // Act 04: applicability volumes (centre x/y and radius for rule A and condition B)
  "scope", "sAx", "sAy", "sAr", "sBx", "sBy", "sBr", "oppose", "fuse", "nest",
  // Act 05: the engine
  "stations", "flow",
  // Act 09: the detector's real scores
  "frontier",
] as const;

export type ParamKey = typeof PARAM_KEYS[number];
export type WorldParams = Record<ParamKey, number>;

const BASE: WorldParams = {
  camX: 0, camY: 0, camZ: 9.5, tgtX: 0, tgtY: 0, tgtZ: 0, subX: 1.9,
  exposure: 1, bloom: 1, dof: 0.35, vign: 0.55,
  sceneAlpha: 1, dim: 0,
  hero: 1, heroOpen: 0,
  pWord: 1, pMark: 0, pNet: 0, pField: 0, pShell: 0, pGrid: 0, pAlpha: 1, pHeat: 0, pTurb: 0.15,
  query: 0, bm25: 0, dense: 0, rrf: 0, cards: 0,
  duel: 0, aX: -1.7, aY: -0.35, aS: 0.92, bX: 1.7, bY: -0.35, bS: 0.92,
  select: 0, caret: 0, crack: 0, burn: 0, compose: 0, restore: 0,
  socket: 0, xray: 0,
  scope: 0, sAx: 1.75, sAy: 1.2, sAr: 1.05, sBx: 2.1, sBy: 0.98, sBr: 0.44, oppose: 0, fuse: 0, nest: 0,
  stations: 0, flow: 0,
  frontier: 0,
};

type Patch = Partial<WorldParams>;

/** The seven engine stations sit along x; the camera stops in front of each. */
export const STATION_X = [-6, -4, -2, 0, 2.8, 4.8, 6.8];
export const STATION_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"] as const;

/** Named stops, in order. Each patch is applied on top of the previous stop. */
const STOPS: [name: string, patch: Patch][] = [
  // ---- 00 Arrival: the glass mark in front of a wordmark made of light
  ["arrival.hold", {}],
  // the wordmark inhales into the sculpture; the camera leans in
  ["arrival.inhale", { pWord: 0, pMark: 1, pTurb: 0.25, camX: 1.3, camY: 0.15, camZ: 5.6, tgtX: 1.8, tgtY: 0, dof: 0.5, heroOpen: 0.15 }],

  // ---- 01 Retrieval: through the glass into a constellation of real passages
  ["retrieval.archive", { hero: 0, heroOpen: 0, pMark: 0, pNet: 1, pTurb: 0.1, subX: 0,
    camX: 0, camY: 0.6, camZ: 13, tgtX: 0, tgtY: 0, tgtZ: -9, query: 0.12, dof: 0.25 }],
  ["retrieval.beams", { camY: 0.4, camZ: 11.5, tgtZ: -8, query: 1, bm25: 1, dense: 1 }],
  ["retrieval.cards", { camY: 0.25, camZ: 10.6, tgtX: 0, tgtY: 0, tgtZ: 0, rrf: 1, cards: 1, dof: 0.9, pAlpha: 0.7 }],

  // ---- 02 Collision: two glass monoliths; selection; stillness; crack; shatter
  ["collision.approach", { subX: 0.6, cards: 0, rrf: 0, bm25: 0, dense: 0, query: 0, pNet: 0, pField: 1, pAlpha: 0.55,
    camX: 0.3, camY: 0.15, camZ: 7.8, tgtX: 0.3, tgtY: -0.1, tgtZ: 0, dof: 0.6,
    duel: 1, aX: -1.75, aY: -0.4, aS: 0.92, bX: 1.95, bY: -0.4, bS: 0.92 }],
  ["collision.select", { camZ: 7.4, aX: -1.1, aY: -0.35, aS: 1.04, bX: 2.35, bY: -0.5, bS: 0.8, select: 1 }],
  ["collision.still", { caret: 1, pAlpha: 0.25, bloom: 0.6, vign: 0.85, pTurb: 0, camZ: 7.1 }],
  ["collision.crack", { caret: 0, crack: 1, pHeat: 0.6, bloom: 1.2 }],
  ["collision.shatter", { burn: 1, pAlpha: 0.6, pHeat: 0.3, bloom: 1.5, vign: 0.6, camZ: 7.6, dof: 0.75 }],

  // ---- 03 Invisible failure: the rule alone, and an x-ray for the missing branch
  ["failure.xray", { subX: 2.2, camX: 2.3, camY: 0.65, camZ: 7.6, tgtX: 1.8, tgtY: 0.05, tgtZ: 0, socket: 1, xray: 1,
    aX: 0.9, aY: -0.02, aS: 0.86, bX: 3.55, bY: -0.02, bS: 0.86, bloom: 1, pHeat: 0, pAlpha: 0.4, dof: 0.5 }],

  // ---- 04 Geometry: liquid applicability volumes, four relations
  ["scope.refinement", { subX: 1.9, camX: 0, camY: 0.6, camZ: 8.6, tgtX: 0, tgtY: 0, tgtZ: 0,
    duel: 0, socket: 0, xray: 0, pField: 0, pShell: 1, pAlpha: 0.55, pTurb: 0.05,
    scope: 1, sAx: 1.75, sAy: 1.2, sAr: 1.05, sBx: 2.1, sBy: 0.98, sBr: 0.44, oppose: 0, fuse: 0, nest: 1, dof: 0.3 }],
  ["scope.disjoint", { sAx: 0.75, sAy: 1.2, sAr: 0.78, sBx: 2.95, sBy: 1.2, sBr: 0.78, nest: 0 }],
  ["scope.redundant", { sAx: 1.75, sAy: 1.2, sAr: 1.0, sBx: 1.98, sBy: 1.08, sBr: 0.84, fuse: 1 }],
  ["scope.opposed", { sAx: 1.15, sAy: 1.2, sAr: 0.92, sBx: 2.4, sBy: 1.2, sBr: 0.92, fuse: 0, oppose: 1 }],

  // ---- 05 Engine: a journey along seven stations (one stop each)
  ...STATION_X.map((x, i): [string, Patch] => [`engine.${STATION_IDS[i]}`, {
    // the station sits upper right, clear of the headline and of the card track below
    subX: x, camX: x - 0.6, camY: 0.45, camZ: 4.9, tgtX: x - 0.95, tgtY: 0, tgtZ: 0,
    scope: 0, oppose: 0, pShell: 0, pField: 1, pAlpha: 0.35, stations: 1, flow: i / 6, dof: 0.55,
  }]),

  // ---- 06 Demo: the world dims to a slow field
  ["demo", { subX: 0, camX: 0, camY: 0.4, camZ: 11, tgtX: 0, tgtY: 0, tgtZ: 0, stations: 0, dim: 1, pAlpha: 0.4, dof: 0.2 }],

  // ---- 07 Composition: the shards fly home, inside the rule's region
  ["compose.return", { subX: 2.8, camX: 0.1, camY: 0.9, camZ: 8.4, tgtX: 0.6, tgtY: 0.6, tgtZ: 0, dim: 0, duel: 1, scope: 0.6, nest: 1,
    pField: 0, pShell: 1, pAlpha: 0.45,
    sAx: 2.65, sAy: 0.95, sAr: 1.28, sBx: 3.2, sBy: 0.62, sBr: 0.5, oppose: 0, fuse: 0,
    aX: 2.4, aY: 1.2, aS: 0.5, bX: 3.2, bY: 0.62, bS: 0.27, compose: 0.55, select: 0, restore: 0.15, bloom: 1.2 }],
  ["compose.done", { camX: 0.9, camY: 1.2, camZ: 8.0, tgtX: 0.7, tgtY: 0.75, scope: 1, compose: 1, restore: 1, dof: 0.45 }],

  // ---- 08 Research: paper surface; the world rests
  ["research", { sceneAlpha: 0 }],

  // ---- 09 Frontier: the detector's real scores as glass bars
  ["frontier", { subX: 0, sceneAlpha: 1, camX: 0, camY: 4.2, camZ: 10.5, tgtX: 0, tgtY: 0.8, tgtZ: 0, duel: 0, scope: 0, compose: 0,
    pShell: 0, pField: 1, pAlpha: 0.35, frontier: 1, dof: 0.35 }],

  // ---- 10 Closing: everything settles into a lattice of light, and the mark forms in the sky
  ["closing.lattice", { camX: 0, camY: 7.5, camZ: 11.5, tgtX: 0, tgtY: -1.5, tgtZ: -5, frontier: 0,
    pField: 0, pGrid: 1, pAlpha: 1, dof: 0.2, bloom: 1.3 }],
  ["closing.end", { camY: 13, camZ: 19, tgtY: 3.2, tgtZ: -7 }],      // the mark in the sky comes into frame
];

export const STOP_NAMES = STOPS.map(s => s[0]);

export const KEYFRAMES: WorldParams[] = (() => {
  const out: WorldParams[] = [];
  let cur: WorldParams = { ...BASE };
  for (const [, patch] of STOPS) {
    cur = { ...cur, ...patch };
    out.push(cur);
  }
  return out;
})();

export const stopIndex = (name: string): number => {
  const i = STOP_NAMES.indexOf(name);
  if (i < 0) throw new Error("unknown stop " + name);
  return i;
};

/** Each act's section scrolls the cursor from `from` to the next act's `from`. */
export const ACT_FROM: Record<number, string> = {
  0: "arrival.hold",
  1: "retrieval.archive",
  2: "collision.approach",
  3: "failure.xray",
  4: "scope.refinement",
  5: "engine.A1",
  6: "demo",
  7: "compose.return",
  8: "research",
  9: "frontier",
  10: "closing.lattice",
};

/** The final stop the last act scrolls to. */
export const LAST_STOP = STOP_NAMES.length - 1;

/** Stops where scrolling should linger (the four relations). */
export const RELATION_STOPS: Record<string, number> = {
  refinement: stopIndex("scope.refinement"),
  disjoint: stopIndex("scope.disjoint"),
  redundant: stopIndex("scope.redundant"),
  opposed: stopIndex("scope.opposed"),
};
