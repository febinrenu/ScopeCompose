/*  A run with no server: the browser's verified engines, emitting the same
 *  events as the engine server.
 *
 *  What runs here is real (the ports `tools/verify_engines.js` checks against
 *  the Python on every build): the stage-1 detector, the scope algebra and
 *  four-way relation, routing, the composition operator, the renderer, the
 *  faithfulness check and the outcome matcher. What needs the server says so
 *  and is skipped: hybrid retrieval, the NLI model, the LLM stages, and B1
 *  extraction -- so a browser run composes from the record's gold branches,
 *  and is labelled "oracle" wherever that matters.
 */
import { engine } from "../engine/bridge";
import { deriveRelation, readerDims, runChain, screen, selectionOf, inside } from "../engine/pipeline";
import type { Branch, CorpusRecord } from "../engine/types";
import type { RunEvent } from "./model";

const CLAUSE = /(?<=[.;!?])\s+|;\s*/;

/** Does any clause of `text` state `outcome`? The engine's matcher, with its number and polarity veto. */
export function states(outcome: string, text: string): boolean {
  const E = engine();
  return text.split(CLAUSE).map(c => c.trim()).filter(Boolean).some(c => E.outcomes_match(outcome, c));
}

function judge(text: string, gold: Branch[]): string[] {
  const E = engine();
  const clauses = text.split(CLAUSE).map(c => c.trim()).filter(Boolean);
  return gold.map(b => states(b.out, text) ? "preserved"
    : clauses.some(c => E.similarity(b.out, c) >= 0.25) ? "distorted" : "suppressed");
}

function rateRow(j: string[]) {
  const n = j.length, pres = j.filter(x => x === "preserved").length, supp = j.filter(x => x === "suppressed").length;
  return { n, preserved: pres, suppressed: supp, distorted: n - pres - supp,
           PR: n ? pres / n : null, SR: n ? supp / n : null, judgements: j };
}

const branchView = (b: Branch, origin = "") => ({
  id: b.id, default: b.def, condition: b.def ? null : (b.cond || b.desc), outcome: b.out,
  applies_to: b.def ? "all cases" : b.desc, passage: b.sup ?? null, explicitness: b.exp || "explicit",
  attributes: (b.at || []).map(a => ({ name: a.n, kind: a.k, values: a.v, min_value: a.lo ?? undefined, max_value: a.hi ?? undefined })),
  origin,
});

export interface LocalOptions { policy: "as_run" | "override"; reader: Record<string, unknown> }

export function localRun(rec: CorpusRecord, opts: LocalOptions): RunEvent[] {
  const runId = `browser_${Date.now().toString(36)}`;
  const events: RunEvent[] = [];
  const t0 = performance.now();
  let seq = 0;
  const ev = (stage: string, kind: RunEvent["kind"], title: string, payload: Record<string, any> = {}) =>
    events.push({ run_id: runId, seq: seq++, t_ms: performance.now() - t0, stage, kind, title, payload });
  const run = (id: string, title: string, fn: () => void) => {
    ev(id, "start", title);
    const t = performance.now();
    fn();
    ev(id, "end", title, { ms: performance.now() - t });
  };
  const request = { source: "benchmark", record_id: rec.id, query: rec.q, mode: "browser", detector_policy: opts.policy, reader: opts.reader };
  ev("run", "start", "run", { kind: "run", request });

  const gold = rec.b || [];
  let conflict = false;
  let override: Record<string, any> | null = null;

  run("setup", "Load the pipeline", () => {
    ev("setup", "data", "components", { live: false, components: {
      engines: "verified JavaScript ports of the Python (checked on every build)",
      nli: "heuristic NLI (offline fixture, not a model)",
      stage1_scorer: "hand-weighted rule scorer (no trained head)",
      stage2: "off: needs the engine server's LLM",
      extraction: "off: needs the engine server (gold branches are used instead)",
      generation: "template",
    }, thresholds: { low: engine().ESCALATE_LOW, high: engine().ESCALATE_HIGH, decide: 0.5, tuned: false } });
    ev("setup", "log", "browser run: the engine server is not connected", { level: "warn" });
  });

  run("ingest", "Gather the documents", () => {
    ev("ingest", "data", "benchmark record", { id: rec.id, query: rec.q, domain: rec.dom, construction: rec.con,
      separation: rec.sep, split: rec.sp, gold_type: rec.ty, gold_relation: rec.rel, gold_branches: gold.length });
    ev("ingest", "log", `benchmark record ${rec.id} (${rec.con}, ${rec.sp} split)`);
  });

  run("retrieve", "Retrieve passages", () => {
    ev("retrieve", "data", "decision", { mode: "oracle",
      note: "Hybrid retrieval (BM25 + Contriever) runs on the engine server. A browser run answers from the instance's own passages, as every experiment does." });
    ev("retrieve", "data", "passages", { passages: rec.p.map(p => ({ id: p.id, text: p.t, source_type: p.st || "unknown", date: p.dt ?? null, url: p.u ?? null })) });
    ev("retrieve", "data", "skipped", { reason: "retrieval runs on the engine server" });
  });

  run("detect", "Detect conflicts (stage 1)", () => {
    const s = screen(rec);
    conflict = s.isConflict;
    ev("detect", "data", `${rec.p[0].id} × ${rec.p[1].id}`, { pair: [rec.p[0].id, rec.p[1].id], nli: s.nli,
      features: s.features, score: s.score, is_conflict: s.isConflict, escalate: s.needsEscalation });
    ev("detect", "log", `1 pair scored locally; ${s.needsEscalation ? 1 : 0} in the uncertain band`);
    ev("detect", "data", "verdicts", { pairs: [{ pair: [rec.p[0].id, rec.p[1].id], is_conflict: s.isConflict,
      type: s.isConflict ? "factual" : "no_conflict", relation: null, confidence: s.confidence, decided_by: "stage1_local" }] });
  });

  run("type", "Classify the conflict (A3)", () => {
    const t = conflict ? "factual" : "no_conflict";
    ev("type", "log", `p0 × p1: ${t} (stage1_local) — the trained typing head needs the engine server`);
    ev("type", "data", "types", { pairs: [{ pair: ["p0", "p1"], is_conflict: conflict, type: t, decided_by: "stage1_local" }] });
  });

  run("policy", "Detector-miss policy", () => {
    if (conflict) ev("policy", "data", "no override needed", { conflicts: 1 });
    else if (opts.policy === "as_run") ev("policy", "warn", "No conflict detected: answering with the general rule (as-run)", { policy: "as_run" });
    else if (rec.ty === "no_conflict") ev("policy", "data", "gold agrees: no conflict", { policy: "override", applied: false });
    else {
      override = { kind: "gold", pair: ["p0", "p1"], type: rec.ty, provisional_relation: rec.rel };
      ev("policy", "warn", "Detector missed: continuing as if detected", { policy: "override", ...override, badge: "oracle: gold label" });
      conflict = true;
    }
  });

  const usable = gold.length >= 2;
  let relation: string | null = null;
  run("scope", "Scope relation (A4)", () => {
    if (!conflict || !usable) { ev("scope", "data", "skipped", { reason: !conflict ? "no conflict to analyse" : "this record has no gold branches to compare" }); return; }
    ev("scope", "data", "descriptors", { descriptors: gold.map(b => ({ passage: b.sup, outcome: b.out,
      applies_to: b.def ? "all cases" : b.desc, default: b.def, source: "gold",
      attributes: (b.at || []).map(a => ({ name: a.n, kind: a.k, values: a.v })) })) });
    const d = deriveRelation(rec, gold);
    relation = d.relation;
    const def = gold.find(b => b.id === d.defaultBranch), exc = gold.find(b => b.id === d.exceptionBranch);
    ev("scope", "data", "p0 × p1", { pair: ["p0", "p1"], relation: d.relation, set_relation: d.canonical,
      outcomes_agree: d.agree, evidence: "attributes (gold applicability, set algebra)", confidence: 1, rationale: d.rationale,
      default_passage: def?.sup ?? null, exception_passage: exc?.sup ?? null, gold_relation: rec.rel });
    ev("scope", "log", `p0 × p1: ${d.relation} — ${d.rationale}`);
  });

  run("extract", "Extract conditions (B1)", () => {
    ev("extract", "data", "skipped", { reason: "B1 extraction needs the engine server (LLM proposer and NLI grounding gate)" });
  });

  let branches: Branch[] = [];
  run("branches", "Assemble branches", () => {
    branches = usable ? gold : gold.slice(0, 1);
    const origin = usable ? "oracle: gold branches (browser run)" : "this record has no gold branches";
    ev("branches", "data", "branches", { origin, branches: branches.map(b => branchView(b)), dims: readerDims(branches).map(d => ({
      name: d.name, kind: d.kind, values: d.values, min: d.lo ?? undefined, max: d.hi ?? undefined })) });
  });

  let answer = "";
  let kept: string[] = [];
  let chain: ReturnType<typeof runChain> | null = null;
  run("compose", "Compose (B2)", () => {
    if (!branches.length) { ev("compose", "data", "skipped", { reason: "nothing to compose" }); return; }
    // a missed conflict (as-run) is answered as no conflict: the general rule alone
    const ty = !conflict ? "no_conflict" : rec.ty === "no_conflict" ? "conditional" : rec.ty;
    chain = runChain({ ...rec, ty }, branches);
    answer = chain.answer;
    kept = chain.composed.branches.map(b => b.id);
    ev("compose", "data", "composition", { resolution: chain.composed.resolution, reason: chain.composed.reason,
      flags: chain.composed.flags, selected_passage: chain.composed.selectedPassage ?? null, merged_pairs: chain.composed.mergedPairs || 0,
      kept, route: { action: chain.action, reason: chain.routeReason, flag: null } });
    ev("compose", "log", `${chain.composed.resolution}: ${chain.composed.reason}`);
  });
  run("generate", "Generate the answer (B3)", () => {
    if (!chain) { ev("generate", "data", "skipped", { reason: "nothing composed" }); return; }
    ev("generate", "data", "answer", { text: answer, method: "template", api_calls: 0,
      citations: [...new Set(chain.composed.branches.map(b => b.sup).filter(Boolean))] });
  });
  run("verify", "Verify faithfulness", () => {
    if (!chain) { ev("verify", "data", "skipped", { reason: "no answer" }); return; }
    const f = chain.faith;
    ev("verify", "data", "faithfulness", { ok: f.ok, missing_branches: f.missingBranches, missing_attribution: f.missingAttribution,
      unsupported_numbers: f.unsupportedNumbers, adjudicated: f.adjudicated, grounded: true });
  });

  const arena: Record<string, any>[] = [{ system: "scopecompose", name: "ScopeCompose", text: answer, kept, method: "template" }];
  run("arena", "Baselines on the same input", () => {
    arena.push({ system: "concat", name: "Standard RAG (all passages, unresolved)", text: rec.p.map(p => p.t).join(" "),
                 kept: branches.map(b => b.id), method: "concatenation" });
    if (branches.length) {
      const sel = selectionOf(rec, branches);
      arena.push({ system: "selection", name: `Selection (keeps ${sel.winner || "—"}, most credible)`, text: sel.answer,
                   kept: sel.composed.branches.map(b => b.id), method: "credibility selection" });
    }
    ev("arena", "data", "answers", { systems: arena });
  });

  run("reader", "Who is this answer right for?", () => {
    const dims = readerDims(branches);
    const set = Object.values(opts.reader).some(v => v !== "" && v !== null && v !== "__other__");
    if (!set) { ev("reader", "data", "skipped", { reason: "no reader set" }); return; }
    const inn = inside(branches, opts.reader);
    const target = branches.find((b, i) => !b.def && inn[i]) || branches.find(b => b.def) || null;
    ev("reader", "data", "verdicts", { reader: opts.reader, dims, branch: target?.id ?? null,
      membership: Object.fromEntries(branches.map((b, i) => [b.id, inn[i]])),
      verdicts: arena.map(s => ({ system: s.system, right: target ? states(target.out, s.text) : null })) });
  });

  run("score", "Score against gold", () => {
    if (!gold.length) { ev("score", "data", "skipped", { reason: "no gold branches for this input" }); return; }
    ev("score", "data", "preservation", { gold_branches: gold.length, gold_answer: rec.sa ?? null,
      rows: arena.map(s => ({ system: s.system, name: s.name, instrument: "textual (clause matcher, number and polarity veto)", ...rateRow(judge(s.text, gold)) })) });
  });

  ev("run", "end", "done", { summary: { query: rec.q, mode: "browser", source: "benchmark", answer, relation, override,
    resolution: (events.find(e => e.stage === "compose" && e.title === "composition")?.payload.resolution) ?? null } });
  return events;
}
