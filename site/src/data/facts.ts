/*  Figures the page states that the browser cannot compute.
 *
 *  Everything else on the page is computed at load from the corpus and the
 *  engines. What is here comes from an experiment artifact or a findings
 *  document, and every entry names its source -- the page shows that source
 *  beside the number. Nothing in this file is a target, an estimate, or a
 *  number that "should" hold.
 */
import decisive from "../../../benchmark/data/decisive.json";
import schema from "../../../contract/schema/query_record-v1.2.0.json";

export interface Fact {
  value: string;
  label: string;
  source: string;
  caveat?: string;
}

/* ---- the decisive experiment: read from the artifact itself ------------ */
const base = decisive.baselines[0];
export const DECISIVE = {
  instances: decisive.pipeline.instances,
  branches: decisive.pipeline.gold_branches,
  pipelinePR: decisive.pipeline.PR,
  pipelineCI: [decisive.pipeline.PR_ci.ci_low, decisive.pipeline.PR_ci.ci_high] as const,
  baselineName: base.name,
  baselinePR: base.PR,
  baselineCI: [base.PR_ci?.ci_low ?? NaN, base.PR_ci?.ci_high ?? NaN] as const,
  baselineDistortion: base.distortion,
  diff: decisive.comparison.pr_difference.point,
  diffCI: [decisive.comparison.pr_difference.ci_low, decisive.comparison.pr_difference.ci_high] as const,
  mcnemarP: decisive.comparison.mcnemar.p_value,
  mcnemarMethod: decisive.comparison.mcnemar.method,
  discordant: decisive.comparison.mcnemar.b + decisive.comparison.mcnemar.c,
  splits: Object.values(decisive.provenance.splits_in_corpus as Record<string, string>)
    .reduce((m, s) => ((m[s] = (m[s] || 0) + 1), m), {} as Record<string, number>),
  seed: decisive.provenance.seed,
  gitSha: String(decisive.provenance.git_sha).slice(0, 7),
  splitWarning: decisive.provenance.split_warning as string,
  note: decisive.provenance.note as string,
  source: "benchmark/data/decisive.json",
};

/* ---- the interface contract: read from the exported JSON schema --------- */
export const CONTRACT_SCHEMA = schema as unknown as {
  title?: string;
  properties?: Record<string, unknown>;
  $defs?: Record<string, unknown>;
  required?: string[];
};
export const CONTRACT_VERSION = "1.2.0";
export const CONTRACT_SOURCE = "contract/schema/query_record-v1.2.0.json · contract/models.py:24";

/* ---- findings documents ------------------------------------------------- */
export const TIER1_PILOT: Fact[] = [
  { value: "38", label: "candidate pairings in the first mining pilot (17 gov.uk, 21 UK banking)",
    source: "docs/wp1_tier1_findings.md:5" },
  { value: "81% / 29%", label: "pairings that were genuinely separate documents: banking 17/21, gov.uk 5/17",
    source: "docs/wp1_tier1_findings.md:193-224" },
  { value: "≈0.08", label: "plausible Tier-1 conditional pairs per curated pairing, first pilot (2026-09-24)",
    source: "docs/wp1_tier1_findings.md:31",
    caveat: "A yield under targeted search, not a prevalence. The code refuses to compute prevalence from this corpus." },
  { value: "0.34", label: "candidates per pairing after re-pairing, over 91 gov.uk pairings (31 candidates)",
    source: "docs/wp1_tier1_findings.md:556-569",
    caveat: "Also a targeted-search yield. Quoted with its date because it revised the first figure." },
  { value: "29% → 80%", label: "distinct-document rate, guide sections (5/17) vs guide against its rules appendix (12/15)",
    source: "docs/wp1_tier1_findings.md" },
];

export const KAPPA: Fact[] = [
  { value: "1.000", label: "Cohen's κ, five-class conflict type · n = 39 · pₑ = 0.540",
    source: "docs/kappa_pilots.md:691-696 (pilot 9)",
    caveat: "Pilot 9 only, never the pooled figure. The bootstrap interval [1.000, 1.000] is an artifact: every resample of a sample with no disagreement is also perfect." },
  { value: "1.000", label: "Cohen's κ, four-way scope relation · n = 14 · pₑ = 0.541",
    source: "docs/kappa_pilots.md:691-696 (pilot 9)" },
  { value: "0.326", label: "κ at pilot 1, before the annotation manual's decision procedure",
    source: "docs/kappa_pilots.md" },
];

export const PROBE = {
  cases: 54,
  verified: "2026-09-25",
  source: "benchmark/wp1_probe_set.py · docs/wp1_probe_verification.md",
  rule: "GO if implicit-recall gain ≥ 0.10 and HCR cost ≤ 0.10",
  ruleSource: "experiments/wp1_probe/run_probe.py:205",
};

export const BENCHMARK_TARGET = { total: 300, split: "60 / 180 / 60", source: "docs/wp1_tier1_findings.md:330-338" };

export const JUDGE = {
  bar: 0.89,
  barSource: "metrics/preservation.py:398-440 (validate_judge; Cattan et al. precedent)",
  status: "implemented, not yet validated against human labels",
};

export const LONG_CONTEXT = {
  model: "openai/gpt-oss-120b (Groq, open weights)",
  source: "baselines/structured_long_context.py · config/models.yaml:94",
};

/** Filled by an actual `pytest` run during verification -- see site/README.md. */
export const TESTS: Fact = {
  value: "520",
  label: "Python tests passing (520 passed, 0 failed)",
  source: "python -m pytest tests · run 2026-10-22 on branch site-immersive",
};

/** Filled by an actual run of the Python order-invariance gate. */
export const ORDER_GATE = {
  value: "1.0000",
  label: "order-invariance pass rate, Python gate",
  detail: "60 records sampled, 72 permutations, 72 pair decisions, 66 non-positional roles",
  source: "python -m scope.order_invariance --sample 60 --heuristic-nli · run 2026-10-22",
};
