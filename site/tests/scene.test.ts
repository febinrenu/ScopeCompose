/*  The pure parts of the scene: the shatter, the particle targets, and the
 *  values derived from WorldParams. All of them must be functions of their
 *  inputs alone, so the story stays reversible.
 */
import { describe, expect, it } from "vitest";
import { shardHeat, shardPose, voronoiCells } from "../src/scene/fracture";
import { buildTargets, netNodes } from "../src/scene/targets";
import { aberration, CARD_FOCUS, flash, focusDistance, shake, stationActivation, stationIndex } from "../src/acts/derive";
import { KEYFRAMES, STATION_IDS, STOP_NAMES } from "../src/acts/keyframes";
import { sampleActs } from "../src/acts/sample";

const W = 3.2, H = 1.25;

describe("fracture", () => {
  const cells = voronoiCells(80, W, H, 19);

  it("tiles the slab: cells cover it exactly, with no overlap", () => {
    const total = cells.reduce((s, c) => s + c.area, 0);
    expect(total).toBeCloseTo(W * H, 6);
    expect(cells.length).toBeGreaterThan(70);
  });

  it("puts every centroid inside the slab", () => {
    for (const c of cells) {
      expect(Math.abs(c.centroid[0])).toBeLessThanOrEqual(W / 2);
      expect(Math.abs(c.centroid[1])).toBeLessThanOrEqual(H / 2);
    }
  });

  it("is seeded: the same seed cuts the same cells", () => {
    expect(voronoiCells(80, W, H, 19)).toEqual(cells);
    expect(voronoiCells(80, W, H, 20)).not.toEqual(cells);
  });

  const from = { x: 2.35, y: -0.5, s: 0.8 }, to = { x: 3.2, y: 0.62, s: 0.27 };

  it("holds every shard in its place before the break", () => {
    for (const c of cells) {
      const q = shardPose(c, W, 0, 0, from, to);
      expect(q.x).toBeCloseTo(from.x + c.centroid[0] * from.s, 9);
      expect(q.y).toBeCloseTo(from.y + c.centroid[1] * from.s, 9);
      expect(q.z).toBe(0);
      expect(q.qw).toBeCloseTo(1, 9);
      expect(shardHeat(c, W, 0)).toBe(0);
    }
  });

  it("seats every shard in the composed slab when composition completes", () => {
    for (const c of cells) {
      const q = shardPose(c, W, 1, 1, from, to);
      expect(q.x).toBeCloseTo(to.x + c.centroid[0] * to.s, 9);
      expect(q.y).toBeCloseTo(to.y + c.centroid[1] * to.s, 9);
      expect(q.z).toBeCloseTo(0, 9);
      expect(q.qw).toBeCloseTo(1, 9);
      expect(q.s).toBeCloseTo(to.s, 9);
      expect(shardHeat(c, W, 1)).toBe(0);
    }
  });

  it("is a function of burn and compose only (scrolling back retraces the flight)", () => {
    for (const b of [0.1, 0.37, 0.8]) {
      const a = cells.map(c => shardPose(c, W, b, 0.2, from, to));
      const again = cells.map(c => shardPose(c, W, b, 0.2, from, to));
      expect(again).toEqual(a);
    }
  });

  it("breaks from the right edge first", () => {
    const right = cells.filter(c => c.centroid[0] > W * 0.3);
    const left = cells.filter(c => c.centroid[0] < -W * 0.3);
    const moved = (cs: typeof cells) => cs.filter(c => shardHeat(c, W, 0.25) > 0).length / cs.length;
    expect(moved(right)).toBeGreaterThan(moved(left));
  });
});

describe("particle targets", () => {
  const word = new Float32Array([-1, 0, 0, 0, 1, 0.5]);
  const nodes = netNodes(128);

  it("gives one node per corpus instance, all behind the camera's subject plane", () => {
    expect(nodes).toHaveLength(128);
    for (const n of nodes) expect(n.z).toBeLessThan(0);
  });

  it("fills every shape for every particle, with finite values", () => {
    const t = buildTargets(2048, word, nodes);
    expect(t.n).toBe(2048);
    for (const a of [t.word, t.mark, t.net, t.field, t.shell, t.grid]) {
      expect(a).toHaveLength(2048 * 3);
      expect(Array.from(a).every(Number.isFinite)).toBe(true);
    }
    expect(t.seed).toHaveLength(2048 * 4);
  });

  it("is seeded: the same inputs give the same field", () => {
    expect(buildTargets(512, word, nodes)).toEqual(buildTargets(512, word, nodes));
  });
});

describe("derived values", () => {
  const at = (name: string) => KEYFRAMES[STOP_NAMES.indexOf(name)];

  it("flashes only once the break has begun", () => {
    // regression: a symmetric bump around burn 0.06 used to leave half a flash at burn 0,
    // which put chromatic aberration and shake on every frame of the collision
    expect(flash(at("collision.select"))).toBe(0);
    expect(shake(at("collision.select"))).toEqual([0, 0]);
    expect(flash({ ...at("collision.crack"), burn: 0.06 })).toBeCloseTo(1, 9);
    expect(flash(at("collision.shatter"))).toBe(0);
  });

  it("keeps aberration at zero wherever the frame rests on text", () => {
    for (const name of STOP_NAMES) {
      const p = at(name);
      if (p.crack === 0 || p.burn === 1) expect(aberration(p)).toBe(0);
    }
  });

  it("lights exactly the station the journey has reached", () => {
    STATION_IDS.forEach((_, i) => {
      const flow = i / (STATION_IDS.length - 1);
      expect(stationActivation(flow, i)).toBe(1);
      expect(stationIndex(flow)).toBe(i);
      if (i > 0) expect(stationActivation(flow, i - 1)).toBe(0);
    });
  });

  it("focuses on the subject, and on the retrieved cards while they are up", () => {
    const p = at("retrieval.beams");
    expect(focusDistance(p)).toBeCloseTo(Math.hypot(p.camX - p.tgtX, p.camY - p.tgtY, p.camZ - p.tgtZ), 9);
    const c = at("retrieval.cards");
    expect(focusDistance(c)).toBeCloseTo(Math.hypot(c.camX - CARD_FOCUS[0], c.camY - CARD_FOCUS[1], c.camZ - CARD_FOCUS[2]), 9);
    // and never jumps: it moves continuously through the act
    let prev = focusDistance(sampleActs(STOP_NAMES.indexOf("retrieval.beams")));
    for (let k = STOP_NAMES.indexOf("retrieval.beams"); k <= STOP_NAMES.indexOf("retrieval.cards"); k += 0.01) {
      const f = focusDistance(sampleActs(k));
      expect(Math.abs(f - prev)).toBeLessThan(0.5);      // the refocus spans ~11 units; a jump would show here
      prev = f;
    }
  });
});

describe("particle targets, built in slices", () => {
  it("gives exactly the same field as the one-shot build", async () => {
    const { buildTargetsAsync } = await import("../src/scene/targets");
    const word = new Float32Array([-1, 0, 0, 0, 1, 0.5]);
    const nodes = netNodes(64);
    let yields = 0;
    const a = await buildTargetsAsync(40000, word, nodes, 3, async () => { yields++; });
    expect(yields).toBe(4);
    expect(a).toEqual(buildTargets(40000, word, nodes, 3));
  });
});
