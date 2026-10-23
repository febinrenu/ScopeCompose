import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { resolve, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { assertEngine, engine } from "../src/engine/bridge";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "..", "src");

describe("bridge", () => {
  it("finds every engine symbol the page needs", () => {
    expect(() => assertEngine()).not.toThrow();
  });

  it("exposes the engines frozen, as plain pass-throughs", () => {
    const E = engine();
    expect(Object.isFrozen(E)).toBe(true);
    for (const [k, v] of Object.entries(E)) {
      if (!["REL", "ROUTE", "ESCALATE_LOW", "ESCALATE_HIGH", "RETRIEVAL_BASELINES"].includes(k)) {
        expect(typeof v, k).toBe("function");
      }
    }
    expect(E.ESCALATE_LOW).toBe(0.35);
    expect(E.ROUTE.opposed[0]).toBe("select");
  });
});

/** The engine names that may only be touched through bridge.ts. */
const ENGINE_NAMES = [
  "compare", "combine", "routeFor", "satisfies", "pairedSize", "outcomes_match", "similarity",
  "containment", "judgeTextual", "rates", "wilson", "mcnemar", "holmAdjust", "heuristicNLI",
  "screenPair", "mostCredible", "classifyExceptionPair", "composeBranches", "renderTemplate",
  "checkFaithfulness", "branchesLost", "REL", "ROUTE", "RETRIEVAL_BASELINES", "ESCALATE_LOW", "ESCALATE_HIGH",
];

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap(f => {
    const p = resolve(dir, f);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith(".ts") ? [p] : [];
  });
}

describe("boundary", () => {
  it("names engine globals only in bridge.ts", () => {
    const offenders: string[] = [];
    for (const file of walk(src)) {
      const rel = relative(src, file).replace(/\\/g, "/");
      if (rel === "engine/bridge.ts" || rel === "engine/engine.d.ts") continue;
      const code = readFileSync(file, "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "")   // comments
        .replace(/`(?:\\[\s\S]|[^`\\])*`|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, '""');  // strings
      for (const n of ENGINE_NAMES) {
        // a bare reference: not a property access (".compare"), not a key ("compare:")
        const re = new RegExp(`(?<![\\w.$])${n}(?![\\w$])(?!\\s*:)`, "g");
        if (re.test(code)) offenders.push(`${rel}: ${n}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("never imports the engine files as modules", () => {
    for (const file of walk(src)) {
      expect(readFileSync(file, "utf8"), file).not.toMatch(/from\s+["'][^"']*public\/engine/);
    }
  });
});
