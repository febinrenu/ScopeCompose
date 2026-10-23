import { describe, expect, it } from "vitest";
import { ACT_FROM, KEYFRAMES, LAST_STOP, PARAM_KEYS, STOP_NAMES, stopIndex } from "../src/acts/keyframes";
import { cursorFor, plateau, sampleActs } from "../src/acts/sample";
import { classifyTier, qualityFor, type DeviceInfo } from "../src/env/tier";

describe("keyframes", () => {
  it("are complete: every stop carries every parameter as a finite number", () => {
    for (const [i, kf] of KEYFRAMES.entries()) {
      for (const k of PARAM_KEYS) expect(Number.isFinite(kf[k]), `${STOP_NAMES[i]}.${k}`).toBe(true);
      expect(Object.keys(kf).sort()).toEqual([...PARAM_KEYS].sort());
    }
  });

  it("every act starts on a real stop, in order", () => {
    const starts = Object.keys(ACT_FROM).map(Number).sort((a, b) => a - b).map(a => stopIndex(ACT_FROM[a]));
    expect(starts).toEqual([...starts].sort((a, b) => a - b));
    expect(LAST_STOP).toBe(KEYFRAMES.length - 1);
  });
});

describe("sampleActs", () => {
  it("lands exactly on each keyframe at integer cursors", () => {
    KEYFRAMES.forEach((kf, i) => {
      const s = sampleActs(i);
      for (const k of PARAM_KEYS) expect(s[k], `${STOP_NAMES[i]}.${k}`).toBeCloseTo(kf[k], 9);
    });
  });

  it("is pure: the frame depends on the cursor alone, not on the path taken", () => {
    const a = sampleActs(7.37);
    sampleActs(19.2); sampleActs(0.5); sampleActs(3.9);
    expect(sampleActs(7.37)).toEqual(a);
  });

  it("is continuous: no jump anywhere along the story", () => {
    for (let c = 0; c < LAST_STOP - 0.001; c += 0.01) {
      const a = sampleActs(c), b = sampleActs(c + 0.001);
      for (const k of PARAM_KEYS) expect(Math.abs(a[k] - b[k]), `${k} @ ${c.toFixed(2)}`).toBeLessThan(0.25);
    }
  });

  it("clamps out-of-range and non-finite cursors", () => {
    expect(sampleActs(-4)).toEqual(sampleActs(0));
    expect(sampleActs(1e9)).toEqual(sampleActs(LAST_STOP));
    expect(sampleActs(NaN)).toEqual(sampleActs(0));
  });

  it("holds at both ends of each segment", () => {
    expect(plateau(0.1)).toBe(0);
    expect(plateau(0.9)).toBe(1);
  });
});

describe("cursorFor: scroll progress to cursor", () => {
  it("walks the act's own stops while pinned, then crosses to the next act", () => {
    // act with stops 4..7, next act at 8, pinned for 75% of its scroll
    expect(cursorFor(4, 8, 0.75, 0)).toBe(4);
    expect(cursorFor(4, 8, 0.75, 0.75)).toBe(7);
    expect(cursorFor(4, 8, 0.75, 1)).toBe(8);
    expect(cursorFor(4, 8, 0.75, 0.875)).toBeCloseTo(7.5, 9);
  });

  it("holds a single-stop act still while pinned", () => {
    expect(cursorFor(8, 9, 0.6, 0.3)).toBe(8);
    expect(cursorFor(8, 9, 0.6, 0.6)).toBe(8);
    expect(cursorFor(8, 9, 0.6, 0.8)).toBeCloseTo(8.5, 9);
  });

  it("is monotonic in progress", () => {
    let last = -1;
    for (let p = 0; p <= 1.0001; p += 0.01) {
      const c = cursorFor(9, 13, 0.8, p);
      expect(c).toBeGreaterThanOrEqual(last);
      last = c;
    }
  });
});

describe("quality tiers", () => {
  const base: DeviceInfo = { webgl2: true, renderer: "", cores: 8, memory: 8, coarse: false, viewportW: 1440, saveData: false };
  const cases: [Partial<DeviceInfo>, number][] = [
    [{ webgl2: false }, 0],
    [{ renderer: "Google SwiftShader" }, 0],
    [{ renderer: "llvmpipe (LLVM 15.0.7, 256 bits)" }, 0],
    [{ renderer: "ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11)" }, 0],
    [{ saveData: true, memory: 2 }, 0],
    [{ renderer: "Mali-G52 MC2" }, 1],
    [{ renderer: "Adreno (TM) 506" }, 1],
    [{ renderer: "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11)" }, 1],
    [{ cores: 4 }, 1],
    [{ memory: 4 }, 1],
    [{ renderer: "Apple GPU" }, 2],
    [{ renderer: "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics Direct3D11)" }, 2],
    [{ renderer: "ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Laptop GPU Direct3D11)" }, 3],
    [{ renderer: "Apple M2 Pro" }, 3],
    [{ renderer: "AMD Radeon RX 7800 XT" }, 3],
    [{ renderer: "ANGLE (NVIDIA GeForce RTX 3070)", coarse: true, viewportW: 820 }, 2],
  ];
  it.each(cases)("%o -> tier %i", (over, want) => {
    expect(classifyTier({ ...base, ...over })).toBe(want);
  });

  it("tier 0 loads no particles at all", () => {
    expect(qualityFor(0).particles + qualityFor(0).shards + qualityFor(0).embers).toBe(0);
  });
});
