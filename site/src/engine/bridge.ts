/*  The one door into the verified engines.
 *
 *  Nothing in src/ may name an engine global except this file (enforced by
 *  tests/boundary.test.ts). Everything here is a pass-through: no logic, no
 *  wrapping that could change a result. If a computation is needed that the
 *  engines do not provide, it belongs in pipeline.ts and must be composed of
 *  calls through `engine`, so the verified code stays the only code that
 *  decides anything.
 */
const REQUIRED = [
  "compare", "combine", "routeFor", "satisfies", "pairedSize",
  "outcomes_match", "similarity", "containment", "judgeTextual", "rates",
  "wilson", "mcnemar", "holmAdjust",
  "heuristicNLI", "screenPair",
  "mostCredible", "classifyExceptionPair", "composeBranches", "renderTemplate", "checkFaithfulness",
  "branchesLost",
] as const;

/** Throws once, naming every missing symbol, if the engine scripts did not load. */
export function assertEngine(): void {
  const missing: string[] = [];
  // typeof on an undeclared binding is safe; a bare reference would throw.
  const probes: Record<string, string> = {
    compare: typeof compare, combine: typeof combine, routeFor: typeof routeFor,
    satisfies: typeof satisfies, pairedSize: typeof pairedSize,
    outcomes_match: typeof outcomes_match, similarity: typeof similarity,
    containment: typeof containment, judgeTextual: typeof judgeTextual, rates: typeof rates,
    wilson: typeof wilson, mcnemar: typeof mcnemar, holmAdjust: typeof holmAdjust,
    heuristicNLI: typeof heuristicNLI, screenPair: typeof screenPair,
    mostCredible: typeof mostCredible, classifyExceptionPair: typeof classifyExceptionPair,
    composeBranches: typeof composeBranches, renderTemplate: typeof renderTemplate,
    checkFaithfulness: typeof checkFaithfulness, branchesLost: typeof branchesLost,
  };
  for (const name of REQUIRED) if (probes[name] !== "function") missing.push(name);
  if (typeof REL !== "object") missing.push("REL");
  if (typeof ROUTE !== "object") missing.push("ROUTE");
  if (typeof RETRIEVAL_BASELINES !== "object") missing.push("RETRIEVAL_BASELINES");
  if (missing.length) {
    throw new Error("Engine scripts did not load: missing " + missing.join(", ")
      + ". They are served from public/engine/ and must load before the module bundle.");
  }
}

let cached: Readonly<EngineApi> | null = null;

export interface EngineApi {
  REL: typeof REL;
  ROUTE: typeof ROUTE;
  ESCALATE_LOW: number;
  ESCALATE_HIGH: number;
  RETRIEVAL_BASELINES: typeof RETRIEVAL_BASELINES;
  compare: typeof compare;
  combine: typeof combine;
  routeFor: typeof routeFor;
  satisfies: typeof satisfies;
  pairedSize: typeof pairedSize;
  outcomes_match: typeof outcomes_match;
  similarity: typeof similarity;
  containment: typeof containment;
  judgeTextual: typeof judgeTextual;
  rates: typeof rates;
  wilson: typeof wilson;
  mcnemar: typeof mcnemar;
  holmAdjust: typeof holmAdjust;
  heuristicNLI: typeof heuristicNLI;
  screenPair: typeof screenPair;
  mostCredible: typeof mostCredible;
  classifyExceptionPair: typeof classifyExceptionPair;
  composeBranches: typeof composeBranches;
  renderTemplate: typeof renderTemplate;
  checkFaithfulness: typeof checkFaithfulness;
  branchesLost: typeof branchesLost;
}

/** The engines, frozen. Resolved lazily so importing this module never throws. */
export function engine(): Readonly<EngineApi> {
  if (cached) return cached;
  assertEngine();
  cached = Object.freeze({
    REL, ROUTE, ESCALATE_LOW, ESCALATE_HIGH, RETRIEVAL_BASELINES,
    compare, combine, routeFor, satisfies, pairedSize,
    outcomes_match, similarity, containment, judgeTextual, rates,
    wilson, mcnemar, holmAdjust,
    heuristicNLI, screenPair,
    mostCredible, classifyExceptionPair, composeBranches, renderTemplate, checkFaithfulness,
    branchesLost,
  });
  return cached;
}

/** The corpus the generated public/data/corpus.js assigned. */
export function corpusGlobal(): unknown {
  return typeof window !== "undefined" ? window.SC_CORPUS : (globalThis as { SC_CORPUS?: unknown }).SC_CORPUS;
}
