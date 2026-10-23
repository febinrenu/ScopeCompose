/*  The browser engines, as the type checker sees them.
 *
 *  public/engine/*.js are classic scripts. Their top-level `function`
 *  declarations become window properties, but their top-level `const`s
 *  (REL, ROUTE, BJ, RESOLUTION, RETRIEVAL_BASELINES) are global LEXICAL
 *  bindings: reachable by bare name, absent from `window`. So these are
 *  declared as globals and only ever referenced by bare name -- and only from
 *  src/engine/bridge.ts, which a test enforces.
 */
import type {
  Applicability, Action, Branch, Composed, CorpusRecord, Faithfulness, Interval,
  Judgement, McNemar, NLI, PassageHolder, PowerResult, Rates, Relation, ScreenResult, SetRel,
} from "./types";

declare global {
  // engines.js
  const REL: {
    SUBSET: "subset"; SUPERSET: "superset"; EQUAL: "equal";
    DISJOINT: "disjoint"; OVERLAPPING: "overlapping"; UNKNOWN: "unknown";
  };
  const ROUTE: Record<Relation, [Action, string]>;
  function compare(a: Applicability, b: Applicability): SetRel;
  function combine(setRel: SetRel, outcomesAgree: boolean): [Relation, string];
  function routeFor(type: string, relation: Relation | null | undefined): Action;
  function satisfies(app: Applicability, persona: Record<string, unknown>): boolean;
  function pairedSize(effect: number, discordantRate: number, branchesPerInstance: number): PowerResult;

  // match.js
  function outcomes_match(gold: string, pred: string): boolean;
  function similarity(a: string, b: string): number;
  function containment(gold: string, pred: string): number;
  function judgeTextual(answerText: string, goldBranches: Branch[]): Judgement[];
  function rates(judgements: Judgement[]): Rates;

  // stats.js
  function wilson(successes: number, n: number, alpha?: number): Interval;
  function mcnemar(correctA: boolean[], correctB: boolean[]): McNemar;
  function holmAdjust(pValues: Record<string, number>, alpha?: number): Record<string, { adjusted: number; rejected: boolean }>;

  // detector.js
  function heuristicNLI(premise: string, hypothesis: string): NLI;
  function screenPair(textI: string, textJ: string): ScreenResult;
  const ESCALATE_LOW: number;
  const ESCALATE_HIGH: number;

  // compose.js
  function mostCredible(passages: PassageHolder["p"]): string | null;
  function classifyExceptionPair(a: Branch, b: Branch): [("nested" | "crossed" | null), string];
  function composeBranches(record: PassageHolder, branches: Branch[], action: Action): Composed;
  function renderTemplate(composed: Composed): string;
  function checkFaithfulness(text: string, branches: Branch[]): Faithfulness;

  // baselines.js
  const RETRIEVAL_BASELINES: Record<"standard_rag" | "rerank_top1" | "nli_filter", (rec: PassageHolder) => string[]>;
  function branchesLost(rec: PassageHolder, kept: Iterable<string>): string[];

  interface Window { SC_CORPUS?: CorpusRecord[] }
}

export {};
