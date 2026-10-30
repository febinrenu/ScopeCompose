/*  The studio's two pure parts: the event reducer, and the browser runner.
 *
 *  The reducer is fed real recordings made by the engine server
 *  (public/data/runs, written by `python -m service.record`), so a change to
 *  the server's event shape that the console cannot read fails here.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { apply, emptyState, fold, latest, PIPELINE, type RunEvent } from "../src/studio/model";
import { localRun, states } from "../src/studio/local";
import { runChain, selectionOf } from "../src/engine/pipeline";
import { branchy, byId } from "../src/data/corpus";

const here = dirname(fileURLToPath(import.meta.url));
const recording = (name: string) =>
  JSON.parse(readFileSync(resolve(here, "..", "public", "data", "runs", `${name}.json`), "utf8")).events as RunEvent[];

describe("event reducer, on the server's own recordings", () => {
  for (const name of ["fee-override", "fee-as-run", "card-passages", "fee-live"]) {
    it(`${name}: every pipeline stage ends, in order, and the run is done`, () => {
      const events = recording(name);
      const s = fold(events);
      expect(s.status).toBe("done");
      for (const id of PIPELINE) expect(s.stages[id]?.status, id).toMatch(/done|skipped|warn/);
      expect(s.order.filter(id => (PIPELINE as readonly string[]).includes(id))).toEqual([...PIPELINE]);
      expect(latest(s, "generate", "answer")?.text).toBeTruthy();
      expect(s.contract.length).toBeGreaterThanOrEqual(2);
    });
  }

  it("is a pure fold: folding twice, or in two halves, gives the same state", () => {
    const events = recording("fee-override");
    const a = fold(events), b = fold(events);
    expect(JSON.stringify(b)).toBe(JSON.stringify(a));
    const half = emptyState();
    events.slice(0, 30).forEach(e => apply(half, e));
    events.slice(30).forEach(e => apply(half, e));
    expect(JSON.stringify(half)).toBe(JSON.stringify(a));
  });

  it("scrubbing: folding up to a seq shows exactly the stages begun by then", () => {
    const events = recording("fee-override");
    const at = events.find(e => e.stage === "scope" && e.kind === "start")!.seq;
    const s = fold(events, at);
    expect(s.stages.scope.status).toBe("running");
    expect(s.stages.compose).toBeUndefined();
  });

  it("counts every LLM call of a live run, with its tokens", () => {
    const s = fold(recording("fee-live"));
    expect(s.counters.calls).toBe(s.llm.length);
    expect(s.counters.calls).toBeGreaterThan(0);
    expect(s.counters.tokens).toBe(s.llm.reduce((t, c) => t + c.prompt_tokens + c.completion_tokens, 0));
  });

  it("the override is never silent: the policy stage warns and badges it", () => {
    const s = fold(recording("fee-override"));
    expect(s.stages.policy.status).toBe("warn");
    expect(s.summary.override.kind).toBe("gold");
    const as = fold(recording("fee-as-run"));
    expect(as.summary.override).toBeNull();
    expect(as.summary.resolution).toBe("pass_through");
  });
});

describe("browser runner", () => {
  it("composes exactly what the verified engines compose, on every record with two or more gold branches", () => {
    for (const rec of branchy()) {
      const s = fold(localRun(rec, { policy: "override", reader: {} }));
      const answer = latest(s, "generate", "answer")?.text;
      if (rec.ty === "no_conflict") continue;
      const chain = runChain({ ...rec, ty: rec.ty }, rec.b!);
      expect(answer, rec.id).toBe(chain.answer);
    }
  });

  it("emits the same stage sequence as the engine server", () => {
    const s = fold(localRun(byId("wp1_fin_imp_003")!, { policy: "override", reader: {} }));
    expect(s.order.filter(id => (PIPELINE as readonly string[]).includes(id))).toEqual([...PIPELINE]);
    expect(s.status).toBe("done");
  });

  it("says what it could not run, instead of pretending", () => {
    const s = fold(localRun(byId("wp1_fin_imp_003")!, { policy: "override", reader: {} }));
    expect(latest(s, "extract", "skipped")?.reason).toMatch(/engine server/);
    expect(latest(s, "retrieve", "skipped")?.reason).toMatch(/engine server/);
    expect(latest(s, "branches", "branches")?.origin).toMatch(/oracle/);
  });

  it("the reader verdict uses the engine's matcher, with its polarity veto", () => {
    const rec = byId("wp1_fin_imp_003")!;
    const sel = selectionOf(rec, rec.b!);
    expect(states("no monthly fee applies", sel.answer)).toBe(false);
    const s = fold(localRun(rec, { policy: "override", reader: { scheme: "graduate", years_since_registration: 1 } }));
    const v = Object.fromEntries(latest(s, "reader", "verdicts")!.verdicts.map((x: any) => [x.system, x.right]));
    expect(v.scopecompose).toBe(true);
    expect(v.selection).toBe(false);
  });
});
