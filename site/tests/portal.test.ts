/*  Portals: a pure function of the story cursor, open at every stop. */
import { describe, expect, it } from "vitest";
import { portalClosure } from "../src/ui/portal";
import { ACT_FROM, LAST_STOP, stopIndex } from "../src/acts/keyframes";

describe("portals", () => {
  it("is fully open on every stop, so no act ever rests behind the iris", () => {
    for (let k = 0; k <= LAST_STOP; k++) expect(portalClosure(k)).toBe(0);
  });

  it("closes to the mark halfway into a portal act, and nowhere else", () => {
    for (const act of [1, 2, 4, 5, 7, 10]) expect(portalClosure(stopIndex(ACT_FROM[act]) - 0.5)).toBe(1);
    // the paper research act and the frontier keep their own entrances
    for (const act of [3, 6, 8, 9]) expect(portalClosure(stopIndex(ACT_FROM[act]) - 0.5)).toBe(0);
  });

  it("is continuous: scrolling never makes the iris jump", () => {
    let prev = 0;
    for (let c = 0; c <= LAST_STOP; c += 0.002) {
      const e = portalClosure(c);
      expect(Math.abs(e - prev)).toBeLessThan(0.05);
      prev = e;
    }
  });
});
