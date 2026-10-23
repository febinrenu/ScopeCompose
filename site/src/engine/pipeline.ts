/*  Composite adapters over the verified engines.
 *
 *  Every decision below is made by a call through `engine()`. What lives here
 *  is orchestration the old page repeated in three files (app.js, bench.js,
 *  byo.js) -- and in one case repeated inconsistently: the hero picked the
 *  default branch by array position, the bench by passage id. Only the bench
 *  version is order-invariant (the prover below catches the other), so it is
 *  the one kept.
 */
import { engine } from "./bridge";
import type {
  Action, Branch, Composed, CorpusRecord, Faithfulness, Interval, Judgement,
  PassageHolder, Rates, Relation, ScreenResult, SetRel,
} from "./types";
import { annotated, corpus } from "../data/corpus";

const app = (b: Branch) => ({ def: !!b.def, at: b.at });

/* ------------------------------------------------------------------------ */
/*  A4 -- the relation for one ordering of an instance's branches            */
/* ------------------------------------------------------------------------ */

export interface Derived {
  setRel: SetRel;
  /** SUBSET/SUPERSET restated by role: the exception sits inside the default. */
  canonical: SetRel;
  agree: boolean;
  agreeSource: "NLI, recorded from the pipeline" | "lexical approximation";
  relation: Relation;
  rationale: string;
  defaultBranch: string;
  defaultPassage: string | null | undefined;
  exceptionBranch: string;
}

/**
 * Mirrors ScopeAnalyser.analyse: pairs enumerated from passages sorted by id,
 * roles from the applicability relation, never from position.
 *
 * Outcome agreement: the pipeline decides it with a bidirectional NLI model
 * the browser cannot run. Where Python's verdict was recorded at build time
 * (`oa`) it is used and labelled; otherwise a lexical test stands in and is
 * labelled as the approximation it is.
 */
export function deriveRelation(rec: Pick<CorpusRecord, "oa">, presented: Branch[]): Derived {
  const E = engine();
  const branches = presented.slice().sort((a, b) =>
    String(a.sup || "").localeCompare(String(b.sup || "")) || String(a.id).localeCompare(String(b.id)));

  let i0 = 0, i1 = 1;
  const defAt = branches.findIndex(b => b.def);
  if (defAt >= 0) { i0 = defAt; i1 = branches.findIndex((_b, k) => k !== defAt); }

  const setRel = E.compare(app(branches[i0]), app(branches[i1]));
  const haveReal = typeof rec.oa === "boolean";
  const agree = haveReal
    ? (rec.oa as boolean)
    : branches[i0].out.trim().toLowerCase() === branches[i1].out.trim().toLowerCase();
  const [relation, rationale] = E.combine(setRel, agree);
  const canonical: SetRel = setRel === E.REL.SUPERSET ? E.REL.SUBSET : setRel;

  let defIdx: number;
  if (setRel === E.REL.SUBSET) defIdx = i1;
  else if (setRel === E.REL.SUPERSET) defIdx = i0;
  else defIdx = (branches[i0].sup || "") <= (branches[i1].sup || "") ? i0 : i1;

  return {
    setRel, canonical, agree,
    agreeSource: haveReal ? "NLI, recorded from the pipeline" : "lexical approximation",
    relation, rationale,
    defaultBranch: branches[defIdx].id,
    defaultPassage: branches[defIdx].sup,
    exceptionBranch: branches[defIdx === i0 ? i1 : i0].id,
  };
}

/** Routing with the multi-exception override of contract/routing.py::route_with_flags. */
export interface Routed {
  derived: Derived;
  action: Action;
  reason: string;
  multiException: boolean;
}

export function routeRecord(rec: CorpusRecord): Routed {
  const E = engine();
  const bs = rec.b || [];
  const derived = deriveRelation(rec, bs);
  let [action, reason] = E.ROUTE[derived.relation] || (["select", ""] as [Action, string]);
  const nExc = bs.filter(b => !b.def).length;
  const multiException = nExc > 1 && action === "compose";
  if (multiException) {
    action = "select";
    reason = "two exceptions on one rule: second-order structure, flagged and routed to "
      + "selection rather than silently mis-composed";
  }
  return { derived, action, reason, multiException };
}

/** Permute every annotated instance and check the A4 decision does not move. */
export function proveInvariance(rows: CorpusRecord[] = annotated()) {
  let permutations = 0, violations = 0, nonPositional = 0;
  for (const rec of rows) {
    const bs = rec.b || [];
    if (bs.length < 2) continue;
    const base = deriveRelation(rec, bs);
    const reversed = bs.slice().reverse();
    const other = deriveRelation(rec, reversed);
    permutations++;
    const keys = ["relation", "canonical", "agree", "defaultBranch", "exceptionBranch"] as const;
    if (keys.some(k => base[k] !== other[k])) violations++;
    // The non-vacuity counter: a system that called the first-listed passage
    // the default would score a perfect pass rate and zero here.
    if (other.defaultPassage && other.defaultPassage !== reversed[0].sup) nonPositional++;
  }
  return { permutations, violations, nonPositional, passRate: permutations ? 1 - violations / permutations : 1 };
}

/* ------------------------------------------------------------------------ */
/*  A2 -- stage 1, as it runs offline                                        */
/* ------------------------------------------------------------------------ */

export function screen(rec: PassageHolder): ScreenResult {
  const [p0, p1] = rec.p;
  return engine().screenPair(p0.t, p1.t);
}

/** Stage-1 precision and recall over the whole corpus, computed not quoted (once). */
let _det: ReturnType<typeof computeDetectorCorpusStats> | null = null;
export function detectorCorpusStats() { return (_det ??= computeDetectorCorpusStats()); }

function computeDetectorCorpusStats() {
  let fired = 0, tp = 0, positives = 0, max = 0;
  const points: { id: string; score: number; ty: string; fired: boolean }[] = [];
  for (const rec of corpus()) {
    const d = screen(rec);
    const conflict = rec.ty !== "no_conflict";
    positives += conflict ? 1 : 0;
    fired += d.isConflict ? 1 : 0;
    tp += d.isConflict && conflict ? 1 : 0;
    max = Math.max(max, d.score);
    points.push({ id: rec.id, score: d.score, ty: rec.ty, fired: d.isConflict });
  }
  const conditional = corpus().filter(r => r.ty === "conditional");
  const conditionalFound = conditional.filter(r => screen(r).isConflict).length;
  return {
    n: corpus().length, fired, tp, positives, max, points,
    precision: fired ? tp / fired : 0,
    recall: positives ? tp / positives : 0,
    conditional: conditional.length, conditionalFound,
  };
}

/** Stage 1 on the branch-bearing subset: the bench's headline gap. */
export function detectorAnnotatedStats() {
  let fired = 0, escalated = 0, max = 0;
  const scores: number[] = [];
  for (const rec of annotated()) {
    const d = screen(rec);
    scores.push(d.score);
    max = Math.max(max, d.score);
    fired += d.isConflict ? 1 : 0;
    escalated += d.needsEscalation ? 1 : 0;
  }
  return { fired, escalated, max, scores, n: annotated().length };
}

/* ------------------------------------------------------------------------ */
/*  The bench's five resolvers                                               */
/* ------------------------------------------------------------------------ */

export interface Resolver {
  key: "selection" | "asruns" | "concat" | "oracle" | "gold";
  name: string;
  sub: string;
  /** Framing that must travel with the number. Typed, so a rewrite cannot drop it. */
  badge: { tone: "warn" | "bad"; text: string } | null;
  note: string;
  answer?: (rec: CorpusRecord) => string;
  judge?: (rec: CorpusRecord) => Judgement[];
  answerNote?: string;
}

export const RESOLVERS: Resolver[] = [
  {
    key: "selection", name: "Selection", sub: "prior work", badge: null,
    note: "The corpus's own selection answer: one passage wins and the other is discarded. "
      + "This is what retrieval-augmented generation does today.",
    answer: rec => rec.sel || "",
  },
  {
    key: "asruns", name: "ScopeCompose", sub: "as it actually runs",
    badge: { tone: "warn", text: "routing-level" },
    note: "Stage 1 screens the pair live, and on this subset it returns no conflict, so nothing "
      + "routes to compose. Scored exactly as run_decisive scores the pipeline arm: a branch "
      + "survives if it is the default, or if its pair composed. No text is generated, so no "
      + "gold branch text is read, and distortion is not measurable at this step.",
    // Deliberately NOT renderTemplate(composeBranches(rec, rec.b, ...)): feeding the gold
    // branches in and scoring the render against those same branches reads the answer key
    // twice. The pipeline arm in run_decisive.py decides survival from routing alone.
    judge: rec => {
      const composed = screen(rec).isConflict && engine().routeFor(rec.ty, rec.rel) === "compose";
      return (rec.b || []).map(b => (b.def || composed) ? "preserved" : "suppressed");
    },
    answerNote: "no text is emitted: the detector found no conflict, so the pipeline "
      + "contributes nothing beyond ordinary retrieval",
  },
  {
    key: "concat", name: "Full context", sub: "both passages, no structure",
    badge: { tone: "bad", text: "gameable" },
    note: "Paste everything and let the reader sort it out. It scores above the gold answer, "
      + "which is a finding about the metric, not about this resolver. Its answer length is "
      + "shown beside it for that reason.",
    answer: rec => rec.p.map(x => x.t).join(" "),
  },
  {
    key: "oracle", name: "ScopeCompose", sub: "routing oracle",
    badge: { tone: "warn", text: "ORACLE — upper bound" },
    note: "Given the gold conflict label and the gold branches, the composition operator routes, "
      + "composes and renders. Both inputs are the answer key, because B1 extraction has no "
      + "offline path. This is a ceiling on the operator, not a measurement of the system.",
    answer: rec => {
      const E = engine();
      return E.renderTemplate(E.composeBranches(rec, rec.b || [], E.routeFor(rec.ty, rec.rel)));
    },
  },
  {
    key: "gold", name: "Gold answer", sub: "the scorer's ceiling", badge: null,
    note: "The human-written scoped answer, scored by the same judge. It does not reach 1.0, which "
      + "bounds what any resolver can score on this metric.",
    answer: rec => rec.sa || "",
  },
];

export interface Scored extends Resolver, Rates {
  meanChars: number;
  ci: Interval;
  perInstance: Map<string, { text: string; j: Judgement[] }>;
}

let _scored: Scored[] | null = null;
export function scoreResolvers(): Scored[] {
  if (_scored) return _scored;
  const E = engine();
  const rows = annotated();
  _scored = RESOLVERS.map(r => {
    const judgements: Judgement[] = [];
    let chars = 0;
    const perInstance = new Map<string, { text: string; j: Judgement[] }>();
    for (const rec of rows) {
      const text = r.answer ? (r.answer(rec) || "") : "";
      chars += text.length;
      const j = r.judge ? r.judge(rec) : E.judgeTextual(text, rec.b || []);
      perInstance.set(rec.id, { text, j });
      judgements.push(...j);
    }
    const s = E.rates(judgements);
    return { ...r, ...s, meanChars: Math.round(chars / rows.length), perInstance, ci: E.wilson(s.preserved, s.n) };
  });
  return _scored;
}

/* ------------------------------------------------------------------------ */
/*  Retrieval-side baselines (public/engine/baselines.js)                    */
/* ------------------------------------------------------------------------ */

export const RETRIEVERS = [
  { key: "standard_rag", name: "StandardRAG", sub: "keeps everything",
    note: "The control. k is unset, so nothing is dropped and nothing can be lost." },
  { key: "rerank_top1", name: "RerankTop1", sub: "selection",
    note: "Keeps the top-ranked passage and discards the rest. Retrieval rank is the credibility "
      + "order, so this is what selection-based RAG does." },
  { key: "nli_filter", name: "NLIFilter", sub: "contradiction-aware",
    note: "Drops the passage carrying the most summed contradiction, if it clears 0.5. The careful "
      + "engineer's answer, and it still destroys branches." },
] as const;

export function retrievalLoss() {
  const E = engine();
  const rows = annotated();
  return RETRIEVERS.map(r => {
    let lost = 0, total = 0, affected = 0;
    for (const rec of rows) {
      const gone = E.branchesLost(rec, E.RETRIEVAL_BASELINES[r.key](rec)).length;
      lost += gone; total += (rec.b || []).length;
      if (gone) affected++;
    }
    return { ...r, lost, total, affected, instances: rows.length, rate: total ? lost / total : 0 };
  });
}

/* ------------------------------------------------------------------------ */
/*  The full chain on two branches -- the reasoner and bring-your-own        */
/* ------------------------------------------------------------------------ */

export interface ChainResult {
  setRel: SetRel;
  agree: boolean;
  agreeSource: string;
  relation: Relation;
  rationale: string;
  action: Action;
  routeReason: string;
  composed: Composed;
  answer: string;
  faith: Faithfulness;
  selectedPassage: string | null;
  multiException: boolean;
}

/**
 * compare -> outcome agreement -> combine -> route -> compose -> render ->
 * faithfulness. Outcome agreement uses the recorded NLI verdict when the
 * record carries one, else the engine's lexical outcomes_match.
 */
export function runChain(record: PassageHolder & { ty?: string; oa?: boolean }, branches: Branch[]): ChainResult {
  const E = engine();
  const type = record.ty || "conditional";
  let setRel: SetRel, agree: boolean, agreeSource: string, relation: Relation, rationale: string;
  if (branches.length >= 2) {
    const d = deriveRelation({ oa: record.oa }, branches);
    setRel = d.setRel; relation = d.relation; rationale = d.rationale;
    if (typeof record.oa === "boolean") { agree = d.agree; agreeSource = d.agreeSource; }
    else {
      // bring-your-own text: the engine's outcome matcher, not exact string equality
      const def = branches.find(b => b.id === d.defaultBranch)!;
      const exc = branches.find(b => b.id === d.exceptionBranch)!;
      agree = E.outcomes_match(def.out, exc.out);
      agreeSource = "lexical outcome matcher (the pipeline uses NLI here)";
      [relation, rationale] = E.combine(setRel, agree);
    }
  } else {
    setRel = E.REL.UNKNOWN; agree = false; agreeSource = "n/a";
    [relation, rationale] = E.combine(setRel, false);
  }
  let action: Action = E.routeFor(type, type === "conditional" ? relation : null);
  let routeReason = type === "conditional" ? (E.ROUTE[relation]?.[1] || "") : "routed by conflict type";
  const nExc = branches.filter(b => !b.def).length;
  const multiException = nExc > 1 && action === "compose";
  // The operator itself applies the nested/crossed override; this flag is only
  // reported, never used to pre-empt it.
  const composed = E.composeBranches(record, branches, action);
  if (composed.flags.nested || composed.flags.crossed) {
    routeReason = composed.reason;
  }
  const answer = E.renderTemplate(composed);
  const faith = E.checkFaithfulness(answer, composed.branches);
  if (composed.resolution === "selected" && action === "compose") action = "select";
  return {
    setRel, agree, agreeSource, relation, rationale, action, routeReason,
    composed, answer, faith, selectedPassage: composed.selectedPassage ?? null, multiException,
  };
}

/** Selection's answer for the same record: the deterministic credibility fallback. */
export function selectionOf(record: PassageHolder, branches: Branch[]) {
  const E = engine();
  const composed = E.composeBranches(record, branches, "select");
  return { composed, answer: E.renderTemplate(composed), winner: composed.selectedPassage ?? null };
}

/** Score an answer's branches with the deterministic textual judge. */
export function audit(answer: string, gold: Branch[]): { judgements: Judgement[]; rates: Rates } {
  const E = engine();
  const judgements = E.judgeTextual(answer, gold);
  return { judgements, rates: E.rates(judgements) };
}

/* ------------------------------------------------------------------------ */
/*  Readers: which branch does a person fall inside?                         */
/* ------------------------------------------------------------------------ */

export interface Dim { name: string; kind: string; values: string[]; lo: number | null; hi: number | null }

export function readerDims(branches: Branch[]): Dim[] {
  const dims = new Map<string, Dim>();
  for (const b of branches) for (const a of b.at || []) {
    const k = String(a.n).trim().toLowerCase();
    if (!dims.has(k)) dims.set(k, { name: k, kind: a.k, values: [], lo: null, hi: null });
    const d = dims.get(k)!;
    for (const v of a.v || []) { const s = String(v).trim().toLowerCase(); if (!d.values.includes(s)) d.values.push(s); }
    if (a.b !== undefined && a.b !== null) { const s = String(a.b).toLowerCase(); if (!d.values.includes(s)) d.values.push(s); }
    if (a.lo !== undefined && a.lo !== null) d.lo = d.lo === null ? a.lo : Math.min(d.lo, a.lo);
    if (a.hi !== undefined && a.hi !== null) d.hi = d.hi === null ? a.hi : Math.max(d.hi, a.hi);
  }
  return [...dims.values()];
}

export function inside(branches: Branch[], reader: Record<string, unknown>): boolean[] {
  const E = engine();
  return branches.map(b => E.satisfies(app(b), reader));
}

export function power(effect: number, discordantRate: number) {
  const rows = annotated();
  const bpi = rows.reduce((s, r) => s + (r.b || []).length, 0) / rows.length;
  return { ...engine().pairedSize(effect, discordantRate, bpi), bpi, have: rows.length };
}
