/*  Shapes of the generated corpus and of the engine's results.
 *
 *  The short keys mirror build_data.py::_FIELDS. If that map changes, this
 *  file changes with it -- the golden test replays the Python reference
 *  through these types, so a drift shows up as a failing test, not a blank.
 */

export type AttrKind = "categorical" | "ordered_tier" | "boolean" | "numeric_range" | "free_text";

export interface Attr {
  n: string;
  k: AttrKind;
  v?: (string | number)[];
  lo?: number | null;
  hi?: number | null;
  b?: boolean | null;
  t?: string;
}

export interface Applicability {
  def: boolean;
  at?: Attr[];
}

export interface Branch {
  id: string;
  out: string;
  desc: string;
  def: boolean;
  sup?: string | null;
  cond?: string;
  at?: Attr[];
  exp?: "explicit" | "implicit" | string;
}

export interface Passage {
  id: string;
  t: string;
  u?: string;
  d?: string;
  dt?: string;
  st?: string;
}

export type ConflictType = "no_conflict" | "factual" | "temporal" | "opinion" | "conditional";
export type Relation = "refinement" | "disjoint" | "redundant" | "opposed";
export type SetRel = "subset" | "superset" | "equal" | "disjoint" | "overlapping" | "unknown";
export type Action = "pass_through" | "prior_work" | "compose" | "merge" | "select";
export type Judgement = "preserved" | "suppressed" | "distorted";
export type Resolution = "composed" | "merged" | "selected" | "prior_work" | "pass_through";

export interface CorpusRecord {
  id: string;
  q: string;
  dom: string;
  con: "natural" | "split";
  sep: "cross_document" | "synthetic_split" | "same_guide" | null;
  sp: "train" | "dev" | "test";
  ty: ConflictType;
  rel: Relation | null;
  prov: string;
  p: Passage[];
  b?: Branch[];
  oa?: boolean;
  sa?: string;
  sel?: string;
}

export interface NLI { entailment: number; neutral: number; contradiction: number }

export interface PairFeatures {
  exception_cues_i: number; exception_cues_j: number;
  exception_cues_max: number; exception_cues_asymmetry: number;
  restriction_cues_i: number; restriction_cues_j: number; restriction_asymmetry: number;
  qualifiers_i: number; qualifiers_j: number; specificity_asymmetry: number;
  length_ratio: number; numeric_overlap: number; numeric_clash: number; year_clash: number;
  temporal_cues_max: number; opinion_cues_max: number; token_jaccard: number;
}

export interface ScreenResult {
  score: number;
  nli: NLI;
  features: PairFeatures;
  isConflict: boolean;
  confidence: number;
  needsEscalation: boolean;
}

export interface Composed {
  resolution: Resolution;
  branches: Branch[];
  flags: { nested: boolean; crossed: boolean };
  reason: string;
  selectedPassage?: string | null;
  mergedPairs?: number;
}

export interface Faithfulness {
  missingBranches: string[];
  missingAttribution: string[];
  unsupportedNumbers: string[];
  adjudicated: boolean;
  ok: boolean;
}

export interface Rates {
  PR: number; SR: number; DR: number; n: number;
  preserved: number; suppressed: number; distorted: number;
}

export interface Interval { point: number; low: number; high: number; n: number; method: string }

export interface McNemar {
  b: number; c: number; bothRight: number; bothWrong: number;
  p: number; method: string; nDiscordant: number; n: number;
}

export interface PowerResult { branches: number; instances: number; discordant: number }

/** Minimal record shape the selection fallback and baselines read. */
export interface PassageHolder { p: Passage[]; b?: Branch[] }
