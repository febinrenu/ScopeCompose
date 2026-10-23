/*  The TS adapters, replayed against the recorded Python answers.
 *
 *  verify_engines.js proves the JS engines match the Python. This proves the
 *  site's orchestration of those engines -- the code that used to be spread
 *  across app.js, bench.js and byo.js -- reproduces the same decisions.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  deriveRelation, detectorCorpusStats, proveInvariance, retrievalLoss, RESOLVERS, runChain, scoreResolvers, screen,
} from "../src/engine/pipeline";
import { annotated, byId, corpus } from "../src/data/corpus";
import { RELATION_EXAMPLES, FEE } from "../src/data/examples";
import { engine } from "../src/engine/bridge";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(resolve(here, "..", "tools", "compare_reference.json"), "utf8"));

describe("A4 relation, as the site derives it", () => {
  it("agrees with the gold relation except on the two known analyser-vs-gold disagreements", () => {
    // These are genuine: the analyser (with Python's recorded NLI verdict) and
    // the annotation disagree, and the page says so wherever they appear
    // ("the analyser disagrees"). wp1_imm_opp_055 is the authored opposed case
    // the project README already flags as needing verification. Pinned so a
    // change in either direction is a deliberate review, not a silent drift.
    const rows = corpus().filter(r => (r.b || []).length === 2 && typeof r.oa === "boolean" && r.rel);
    expect(rows.length).toBeGreaterThan(30);
    const wrong = rows.filter(r => deriveRelation(r, r.b!).relation !== r.rel)
      .map(r => `${r.id}: gold ${r.rel}, analyser ${deriveRelation(r, r.b!).relation}`);
    expect(wrong).toMatchInlineSnapshot(`
      [
        "wp1_fin_imp_019: gold refinement, analyser redundant",
        "wp1_imm_opp_055: gold opposed, analyser redundant",
      ]
    `);
  });

  it("is order-invariant, and not vacuously so", () => {
    const inv = proveInvariance();
    expect(inv.violations).toBe(0);
    expect(inv.passRate).toBe(1);
    expect(inv.nonPositional).toBeGreaterThan(0);
  });
});

describe("oracle chain", () => {
  it("renders exactly the answer the Python renders, for every instance", () => {
    const oracle = RESOLVERS.find(r => r.key === "oracle")!;
    let n = 0;
    for (const c of ref.oracle) {
      const rec = byId(c.inst)!;
      expect(oracle.answer!(rec), c.inst).toBe(c.answer);
      n++;
    }
    expect(n).toBe(ref.oracle.length);
  });
});

describe("textual judging through the resolvers", () => {
  it("reproduces Python's judgements for selection, concat and gold", () => {
    const scored = scoreResolvers();
    const by = (k: string) => scored.find(s => s.key === k)!;
    for (const c of ref.textual) {
      const key = c.sys === "selection" ? "selection" : c.sys === "concat" ? "concat" : "gold";
      expect(by(key).perInstance.get(c.inst)!.j, `${c.inst}/${c.sys}`).toEqual(c.j);
    }
  });

  it("keeps the bench's headline figures (a data change must be a deliberate copy review)", () => {
    const pr = Object.fromEntries(scoreResolvers().map(s => [s.key, Number(s.PR.toFixed(3))]));
    expect(pr).toMatchInlineSnapshot(`
      {
        "asruns": 0.446,
        "concat": 0.855,
        "gold": 0.699,
        "oracle": 0.976,
        "selection": 0.542,
      }
    `);
  });

  it("keeps the framing badges on the oracle and concatenation columns", () => {
    expect(RESOLVERS.find(r => r.key === "oracle")!.badge?.text).toMatch(/upper bound/i);
    expect(RESOLVERS.find(r => r.key === "concat")!.badge?.text).toMatch(/gameable/i);
    expect(RESOLVERS.find(r => r.key === "asruns")!.judge).toBeTypeOf("function");
  });
});

describe("retrieval baselines", () => {
  it("lose exactly the branches the Python loses", () => {
    const want: Record<string, number> = {};
    for (const c of ref.retrieval) want[c.sys] = (want[c.sys] || 0) + c.lost.length;
    const got = Object.fromEntries(retrievalLoss().map(r => [r.key, r.lost]));
    expect(got).toEqual(want);
  });
});

describe("detector, as the page reports it", () => {
  it("agrees with the recorded stage-1 scores", () => {
    for (const c of ref.detector) {
      const rec = byId(c.inst)!;
      expect(screen(rec).score).toBeCloseTo(c.score, 9);
    }
  });

  it("computes precision and recall over the whole corpus", () => {
    const d = detectorCorpusStats();
    expect(d.n).toBe(128);
    expect(d.fired).toBe(6);
    expect(Number(d.precision.toFixed(3))).toBe(0.833);
    expect(Number(d.recall.toFixed(3))).toBe(0.064);
  });
});

describe("illustrative examples run through the real engine", () => {
  it("each relation example yields the relation it was written to show", () => {
    for (const [rel, ex] of Object.entries(RELATION_EXAMPLES)) {
      expect(runChain(ex, ex.b).relation, rel).toBe(rel);
    }
  });

  it("the story's fee example: selection drops the exception, composition keeps it", () => {
    const E = engine();
    const sel = E.renderTemplate(E.composeBranches(FEE, FEE.b, "select"));
    expect(E.rates(E.judgeTextual(sel, FEE.b)).SR).toBe(0.5);
    const comp = runChain(FEE, FEE.b);
    expect(comp.action).toBe("compose");
    expect(comp.faith.ok).toBe(true);
    expect(E.rates(E.judgeTextual(comp.answer, FEE.b)).SR).toBe(0);
  });

  it("annotated subset is the denominator the page states", () => {
    expect(annotated().length).toBe(42);
  });
});
