# Implementation plan v3: from the ScopeCompose pilot to a submittable paper

**Companion to `02-proposal-v7.md`. Written 2026-10-30 against `ScopeCompose/` at commit `7438728`, then revised after the second-round review (`04-second-round-review-and-responses.md`).**

This plan replaces `implementation-plan-two-person-v2.md`. It is written as **edits to what already exists**, not as a new build. About 80% of the infrastructure stays as it is. What changes is three things:

- what gets measured;
- what data it is measured on;
- which component the paper is about.

Every work item lists:

- **Owner.** A is the front end: retrieval, detection, scope, mining, contexts and release. B is the back end: metrics, baselines, composition, generation and validation.
- **Priority.** P0: the paper cannot be submitted without it. P1: a reviewer will ask for it. P2: it strengthens the paper.
- **Concrete files.**
- **An acceptance test.**

> **Two kinds of ID appear below.**
> - **Work-item IDs** are hyphenated: M-1, D-1, B-1, S-1, E-1.
> - **Unhyphenated IDs** refer to the proposal's detectors (D0–D7), systems (B0–B8, O1–O3), strata (S1–S5), experiments (E1–E8) and hypotheses (H1–H4, H-T, H-C, H-F, V1–V3).

---

## 0. Ground rules for the next 16 weeks

1. **Keep what already works.** The 520 existing tests must stay green. These are the project's best assets:
   - contract versioning;
   - the API cache and cost log;
   - provenance on every judged number;
   - the κ tooling and its pass-plausibility checks;
   - the order-invariance gate.
2. **Freeze `site/` and the Studio (`service/`) until submission**, except for bug fixes. They are viva and demo material, but they produce no evidence. The last four weeks of commits went mostly into them while the measurement critical path did not move. The Studio's persona mapping (`service/branches.py: reader_dims, branch_for_reader`) **is** reused below.
3. **No headline number from the 54 LLM-authored probes (S5).** They are dev and diagnostic data only. Where the paper cites pilot findings measured on them, it says so.
4. **No evaluated system or detector touches test construction.** That covers QNFA, D3 and every backbone in mining, negative screening and branch pre-annotation. These roles go to evaluation-only models:
   - Qwen ≈ 27B on Groq;
   - Llama-3.1-8B-Instruct, local.
5. **One scorer for everything.** Every system's final answer is scored by the same code. The pilot's routing-level scoring (`run_decisive.score_pipeline_routing`) is retired from headline use.
6. **No test-set run before the pre-registration is tagged.** Nothing runs on test data until `prereg/` is frozen and tagged (week 7).
7. **Pin every model.** Every LLM-produced artefact records the exact model ID and date. Cached outputs are released, because Groq's model availability changes without notice.

---

## 1. What changes, at a glance

| Area | Today | v3 target | Review item |
|---|---|---|---|
| Unit of contribution | The resolution operator | Detection (QNFA), the measurement of harm, the benchmark | F1, F8 |
| Headline metric | Containment PR, which is gamed by verbosity | **CCA-all** (reader-based, non-backbone reader), plus scoped precision and length, plus per-branch CCA | F2, R2-3 |
| Judge | Implemented, never validated | Deterministic first, then an enumerate-then-verify judge (non-evaluated model), validated against two humans and an external annotator | F2 |
| Benchmark unit | 2-sentence pairs | K = 6 chunk-level contexts, one chunker, screened negatives, personas, strata S1–S5 | F3 |
| Test construction | Mixed | Mined and screened **without** evaluated detectors; natural groups re-split | R2-2, R2-5 |
| Branch annotation | 42 LLM-authored instances | All real-text conditional instances (≈ 320), pre-annotated by a non-evaluated model, corrected, 30% from-scratch second pass, 15% external | F3, R2-2 |
| Rare relations | 2 `redundant`, 2 `opposed` | ≥ 20 each, as human-verified contrast sets (S4) | F3 |
| Baselines | rerank-top1, NLI-filter, structured LC | Adds ConflictRAG-style (fidelity-checked), CARE-style, taxonomy-aware, free-text, structured + verify, vanilla | F4, R2-7 |
| Detector | Heuristic stage 1, recall 0.064 | QNFA, distilled DeBERTa and a cascade; 8 detectors with defined scores | F1, R2-4 |
| Backbones | gpt-oss-120b only | Qwen2.5-7B (local), gpt-oss-20b, gpt-oss-120b; frames extracted by each cell's own backbone | F6, R2-6 |
| Statistics | Branch-level Wilson, McNemar | Instance-level exact McNemar, BCa cluster bootstrap, Holm + gatekeeping, bias-reduced GEE as sensitivity, simulation power, pre-registration | F7, R2-11 |
| Licence | Not cleared | OGL for gov.uk; URL + offsets + hash for bank text | F9 |

---

## 2. Contract v1.3.0 (week 1, A leads, B signs off) — P0

The changes are additive, so the minor version is bumped and all v1.2.0 records stay loadable. `contract/mock.py`, the JSON Schema export and the tests are updated **in the same commit** (`CONVENTIONS.md`).

### 2.1 `contract/models.py`

```python
class Stratum(str, Enum):           # DERIVED from separation + provenance, never from `construction`
    NATURAL_XDOC = "natural_xdoc"             # S1: distinct content_id
    NATURAL_SAME_GUIDE = "natural_same_guide" # S2: one content_id (pilot records are construction=split!)
    REAL_SPLIT = "real_split"                 # S3
    CONTRAST = "contrast"                     # S4
    AUTHORED_PROBE = "authored_probe"         # S5: dev-only

class Opposition(str, Enum):        # sub-label of ScopeRelation.OPPOSED
    EQUAL = "equal"                 # same scope -> selection is correct
    CROSSING = "crossing"           # overlap, non-nested -> contested
    UNKNOWN = "unknown"             # undecidable -> contested (policy), select (ablation)

class Passage(BaseModel):            # + optional
    license: str | None = None       # "OGL-3.0" | "copyright-pointer"
    char_span: tuple[int, int] | None = None
    sha256: str | None = None
    chunk_id: str | None = None      # from the single shared chunker

class ConflictPair(BaseModel):       # + optional
    opposition: Opposition | None = None
    score: float | None = None       # every detector must emit a score (proposal §5.4)

class QueryRecord(BaseModel):        # + optional
    stratum: Stratum | None = None
    granularity: Literal["sentence", "chunk"] = "sentence"
```

### 2.2 `contract/gold.py`

```python
class OutcomeValue(BaseModel):
    kind: Literal["numeric", "boolean", "categorical", "text"]
    value: float | bool | str
    unit: str | None = None

class Branch(BaseModel):             # + optional
    outcome_value: OutcomeValue | None = None

class Persona(BaseModel):
    persona_id: str
    attributes: dict[str, str | float | bool]
    statement: str                   # paraphrased from SOURCE PASSAGE wording, not attribute ids
    gold_branch_id: str              # via service.branches.branch_for_reader
    gold_outcome: OutcomeValue
    hand_written: bool = False
    verified_by: str | None = None

class GoldInstance(BaseModel):       # + optional
    stratum: Stratum | None = None
    gold_passage_ids: list[str] = []
    personas: list[Persona] = []
    mined_by: Literal["retrieval+verifier", "llm_proposer", "lexical", "qnfa",
                      "constructed", "authored"] | None = None
    v7_relabelled_from: str | None = None   # e.g. "conditional/disjoint" -> no_conflict
```

### 2.3 New contracts

- **`contract/frames.py`.** `NormativeFrame` holds:
  - `passage_id`, `variable`, `applicability`;
  - `outcome`, `outcome_value`, `modality`;
  - `span`, which must be verbatim.
- **`contract/system_output.py`.** `SystemAnswer` holds:
  - `instance_id`, `system_id`, `backbone`;
  - `setting`, `persona_id`;
  - `answer_text`, `short_answer`, `branches`;
  - `calls`, `prompt_tokens`, `completion_tokens`, `latency_s`, `cache_hits`;
  - `fell_back`, `prompt_sha`, `run_id`, `reasoning_effort`.

### 2.4 One-off v7 relabelling script: `benchmark/relabel_v7.py` (new)

The script relabels pilot instances deterministically, per proposal §4.2:

| Pilot label | v7 label |
|---|---|
| `conditional/disjoint` | `no_conflict` (complementary) |
| `conditional/redundant` | `no_conflict` (redundant) |
| `opposed` | `opposition` set to `equal` or `crossing` from attributes, else `unknown` |

It writes `v7_relabelled_from` on each instance it changes.

**Acceptance:**
- The report reproduces **18 natural conditional conflicts** (14 S1, 4 S2).
- `pytest -q` is green.
- The old corpus still validates.

---

## 3. Work streams

### WS-M — Metrics, reader and validity battery (owner B; weeks 1–3) — **P0, first**

| # | File | Change | Acceptance test |
|---|---|---|---|
| M-1 | `metrics/values.py` (new) | `normalise(text) -> OutcomeValue` and `values_match(a, b)`. Reuses `branch_match.numbers_conflict`, `polarity_conflict` and `_strip_quantifiers`. Handles currency, percentages, durations and yes/no. | ≥ 60 unit cases, e.g. "£2.50" ≡ "2.50 per transaction"; "no charge" ≡ "fee-free" |
| M-2 | `metrics/case_accuracy.py` (new) | `reader_cca(answers, gold, reader)`: CCA-all (primary), CCA-E, CCA-D, over-generalisation. Persona mode uses deterministic matching on `short_answer`. Clarification-only answers are flagged and rate-reported. | See M-7 |
| M-3 | `api_budget/client.py`, `config/models.yaml` | New tier **READER**: `ollama:llama3.1:8b-instruct` with `num_ctx` set explicitly and logged. **No silent fallback.** If the reader is unavailable, the run stops. New tier **EVAL**: Qwen ≈ 27B (Groq), for the verifier, screener, pre-annotator and judge. Qwen-27B is removed from `SECOND_BACKBONE`. | `scripts/smoke_test.py --tier READER --tier EVAL` |
| M-4 | `metrics/claims.py` (new) | Sentence and clause splitting (on ";", "but", "except", "unless"). Labels: aligned / misattached / extraneous / unsupported. Deterministic first; EVAL judge on the residue. | Deterministic coverage reported |
| M-5 | `metrics/scoped.py` (new) | Scoped precision, branch recall, attachment error, fabricated-exception rate, unsupported rate, length, and single-reference AC (descriptive). | — |
| M-6 | `metrics/judge.py` (new; refactors `preservation._judge_with_llm`) | Direct judge and enumerate-then-verify judge, one gold branch per call. A failure is scored SUPPRESSED, as now. `validate_judge` is extended with accuracy, κ, per-class recall and Kendall τ of system rankings. | Fixture of 20 items |
| M-7 | `experiments/validity_battery.py` (new) | Seven outputs per validation instance: gold, selection, concat-gold, concat-K, gold+padding, unattached hedge, null. Applies the **per-metric** V1 criteria from the proposal, plus V2's adversarial NOT_DETERMINED recall. Runs on dev-built validation data, **not** S5 alone. | Prints a pass/fail per criterion. **The criteria are fixed before the first run and never edited to pass.** |
| M-8 | `metrics/cluster.py` (new) | `mcnemar_instances(a, b)` (exact; wraps `stats.mcnemar` at instance level); `bca_cluster_bootstrap`; `did_bootstrap` (risk-difference interaction); `gee_bias_reduced(df, formula, groups)` via statsmodels GEE `cov_type="bias_reduced"`; `signflip_permutation`. | Checked against statsmodels and hand calculations |
| M-9 | `metrics/power_sim.py` (new; extends `power.py`) | Simulates instance-level paired binary outcomes at dev-observed discordance and returns the MDD at Holm α. Its output goes into the pre-registration. | Reproduces the proposal §6.7 table within ±1 point |
| M-10 | `experiments/run_decisive.py`, `metrics/preservation.py` | Retire routing-level scoring from headline output (`--legacy-routing` keeps the pilot reproducible). Rename `hallucinated` to `nonexistent_citation`, keeping back-compatible keys. | Old tests updated deliberately |

### WS-D — Detection: QNFA, distilled detector, cascade (owner A; weeks 1–7) — **P0**

| # | File | Change | Acceptance test |
|---|---|---|---|
| D-1 | `detection/frames.py` (new) | `FrameExtractor(backbone=...)`: one batched call per query, cached on `(query, passages, prompt_sha, model)`, with a JSON schema. Non-verbatim spans get one re-ask and are then dropped and counted. Outcomes are normalised by M-1. **Backbone-parameterised**, so O2 and O3 use their own cell's model. | Frame yield and dropped-span rate on the pilot. Outcome-value agreement with gold branches on the 42 probes. |
| D-2 | `detection/align.py` (new) | `DecisionVariableAligner`: Contriever cosine on `variable` strings, plus a query-relevance check, plus an LLM tie-break only inside a dev-tuned band. | Alignment recall of gold conditional pairs on dev |
| D-3 | `detection/variants/base.py`, `detection/stage1.py` | Add `classify_in_context(record, candidate)`, defaulting to `classify`. Thread `query` into `PairCandidate` (default `""`). **Every variant must return a score.** | Existing variants untouched; delegation test |
| D-4 | `detection/variants/qnfa.py` (new) | Detector D5: frames → alignment → `not values_match` → `ScopeAnalyser` relation → type. Score = alignment similarity × mismatch confidence. Sets `opposition`. | In `compare` |
| D-5 | `detection/variants/conflictrag_mlp.py` (new) | Detector D2: frozen `all-MiniLM-L6-v2` pair features → MLP, with LLM fallback below τ = 0.7. **Trained on ConflictQA-derived pairs per ConflictRAG's description (three types + none), not on our labels.** Data built by `S-9`. Used inside B3. | Trains on CPU in < 5 min; fidelity accuracy reported (S-9) |
| D-6 | `detection/variants/llm_pair.py` (new) | Detector D3: zero-shot pair typing using our taxonomy definitions, plus a verbal confidence elicited in the same call, as its score. Fixed extractor model, gpt-oss-20b. | Prompt hash recorded |
| D-7 | `detection/variants/finetuned.py` | Detectors D4 / D6: input `[q] [SEP] d_i [SEP] d_j`, **seq 512, pair-aware truncation**. Two heads: type (3-way) and relation (5-way). Class weights. `fit()` takes gold plus silver with a source weight. Uses the `conservative_6gb` profile, with batch 4 × accumulation 8 at seq 512. | Peak VRAM < 6 GB (`torch.cuda.max_memory_allocated`) |
| D-8 | `detection/distill.py` (new) | Silver labels from QNFA over mined pairs from **train/dev document groups only**. A query is synthesised from the rule passage's decision variable, then recorded. Positives need QNFA score ≥ 0.8; negatives are same-provider pairs. Target 3–6k. | Asserts zero overlap with test groups via the `splits.py` components |
| D-9 | `detection/cascade.py` (new); `detection/threshold.py` | D6 handles pairs outside a band; QNFA handles pairs inside it. The threshold sweep targets F1 and tokens and writes the cost–F1 curve. **Uses a chunk-level dev set.** | Curve on dev; band written with `tuned_on` |
| D-10 | `detection/stage1.py` | **Do not change `_rule_score`.** It is detector D0, the documented failure. Register it as `legacy_lexical`. | Pilot P/R reproduced (0.833 / 0.064) |
| D-11 | `detection/variants/compare.py` | Reports F1 at dev-fixed points, P, R, PR-AUC (score-based), the conditional / same-scope cell, tokens, calls and latency. Two candidate sets: **natural hard-candidate (primary)** and all-within-context. | Table for D0–D7 on both sets |
| D-12 | `scope/relation.py` | Emit `opposition`. Add `policy_unknown: "contested" \| "select"`, default `"contested"`. | Order-invariance gate still passes at 1.0000 |
| D-13 | `experiments/qnfa_ablations.py` (new) | E2b ablations: query-free frames; whole-passage alignment; entailment-only relation; string outcomes. Dev runs before the freeze; test runs on a 120-instance subset after it. | Ablation table |

**Week-5 detection gate (G4).** QNFA F1 on the **chunk-level** dev natural hard set must be ≥ 0.6.
- **If not:** diagnose with the D-1 error categories (variable mismatch, normalisation, applicability) and fix the largest category once.
- **If it is still below 0.4:** H2 is at risk, and the paper re-centres on RQ1/RQ2 (proposal §11).

### WS-B — Benchmark to test targets (A: mining/contexts/release; B: personas/branches; joint annotation; weeks 1–7) — **P0**

**Targets** (proposal §6.7), all in the test split:

| Component | Target |
|---|---|
| Real-text conditional instances (S1–S3) | ≥ 200, of which ≥ 60 are S1+S2 |
| Controls | ≥ 120 |
| S4 contrast instances | ≥ 40 |
| Personas | ≥ 1 per region |
| Dev | ≈ 100 real-text instances built like test |

| # | File | Change | Acceptance test |
|---|---|---|---|
| B-1 | `benchmark/strata.py` (new); `benchmark/splits.py` | Derive `stratum` from `separation` and provenance. S5 is forced to dev. **Re-split natural document groups before the freeze** (archive the old assignment in `benchmark/data/splits_archive_2026-10.json`; record that no model was tuned on it). New groups go to test until the targets are met. Quota report per stratum. | 0 leakage; S5-in-test = 0; reproducible seed |
| B-2 | `benchmark/mining/retrieval_miner.py` (new) — **test mining** | For each rule-bearing sentence (one with an outcome), BM25 + Contriever retrieves cross-document chunks. The **EVAL verifier** (Qwen ≈ 27B, own prompt) filters them; humans verify. Tags `mined_by="retrieval+verifier"`. | Verified yield per 100 candidates, tracked weekly |
| B-3 | `benchmark/mining/frame_miner.py` (new) — train/dev only | QNFA-assisted candidate proposal. Tags `mined_by="qnfa"`. **Refuses to write to test groups.** | Assertion test |
| B-4 | `benchmark/mining/fetch_govuk.py` | More guides: Universal Credit, Child Benefit, council tax, student finance, driving licences. **Store `document_type` and `public_updated_at`** for B3's authority and recency criteria. | ≥ 150 new pairings by week 4; metadata coverage reported |
| B-5 | `benchmark/chunker.py` (new) and `benchmark/contexts.py` (new) | **One sentence-aligned chunker** (≈ 120–160 words) applied to every source document. Gold passage = the chunk containing the gold sentence. K = 6: gold chunks + RRF hard negatives from the same provider. The EVAL model screens negatives; flagged negatives are **human-adjudicated, never silently removed**. A human audits a random 20% of contexts and the residual rate is reported. Retriever ranks are kept for the end-to-end setting, and a sentence-level variant is emitted too. | 100% gold inclusion (oracle setting) |
| B-6 | `benchmark/personas.py` (new) | Personas per region via `reader_dims` / `branch_for_reader`. Statements are paraphrased by the EVAL model **from source-passage wording**, then human-verified. `hand_written=True` where attributes are untyped. Reports the lexical overlap of personas with system outputs. | Every conditional instance has ≥ 1 persona per region |
| B-7 | `benchmark/mining/contrast.py` (new; extends `relation_author.py`) | Minimal edits of real instances to create `redundant` and crossing `opposed`. Each is verified by `compare` + `values_match`; failures are discarded, never relabelled. | ≥ 25 of each authored → ≥ 20 of each verified |
| B-8 | `benchmark/annotation/cli.py` | Three new modes. `branches`: correct the EVAL pre-annotation, recording correction distance. `personas`: verify. `second-pass`: a stratified 30%, blind, **from scratch**. A `--annotator external` path for the 15% external sample. Display is blinded to `mined_by`. | Pass-plausibility checks extended |
| B-9 | `benchmark/annotation/agreement.py` | Branch-level F1 between annotators; Krippendorff's α on persona outcomes; κ per axis **per stratum**, with informative-n, p_e and composition; an author vs external comparison. | Renders the paper's agreement table |
| B-10 | `benchmark/release/` (new) | `make_release.py`: gov.uk text with OGL attribution; bank passages as `{url, char_span, sha256}`; `rehydrate.py` verifies hashes and reports drift; cached outputs; manifests. | Round-trip on 10 bank passages |
| B-11 | `benchmark/datasheet.md` | Generated by `corpus_plan --datasheet`: strata, provenance, licence, agreement, snapshot dates, limitations. | No hand-typed counts |
| B-12 | `benchmark/external/conditionalqa_probe.py` (new) — **P1, gated** | Week 3: sample 50 ConditionalQA items and measure the share that maps to refinement structure with both outcomes stated; check the dataset licence. If the share is ≥ 40% and the licence permits, build the cross-passage variant (E8). | Gate G-CQA recorded |

### WS-S — Systems and baselines (owner B; A for detector plumbing; weeks 2–7) — **P0**

| # | File | Change | Acceptance test |
|---|---|---|---|
| S-1 | `baselines/registry.py` (new) | `run(system_id, record, backbone, setting, persona=None) -> SystemAnswer`. The same context, instructions, `max_tokens` and **one uniform `reasoning_effort`** for every system. The setting is decided in week 1 from the measured budget (G1), then frozen. | All 12 systems run on 5 mock records offline |
| S-2 | `baselines/conflictrag_topsis.py` (new) | B3, labelled **"ConflictRAG-style"** everywhere. **Routing:** D-5 type → factual = Entropy-TOPSIS; temporal = most recent date; opinion = multi-perspective prompt. **TOPSIS criteria:** authority (ordinal over enriched `source_type`), recency (enriched dates), relevance (retriever score), specificity (count of applicability constraints, or a lexical proxy), consistency (mean NLI agreement); entropy weights per query. Then generation over the selected passage(s). Flag `--no-specificity` for the ablation. | Hand-computed TOPSIS test on 3 passages; every unspecified choice listed in an appendix |
| S-3 | `baselines/prompting.py` (new) | B0, B4 (CARE-style), B5 (CONFLICTS taxonomy-aware: type first, then type-specific behaviour), B6, B7 (start from `structured_long_context.py`; adapt the schema and instructions from the released NormBench repository where applicable; deviations listed), B8 (B7 + self-verify). Prompts live in `baselines/prompts/*.txt`; SHAs go in the pre-registration. | Prompt files frozen |
| S-4 | `baselines/retrieval_side.py` | B1 and B2 generate answers over the passages they keep. | `SystemAnswer` produced |
| S-5 | `composition/operator.py` — **P1** | `partition_compose(frames)`: Venn regions over typed constraints (`scope.attributes`), most-specific-wins, `contested` for incomparable disagreements. Legacy `compose()` is the untyped fallback; nested/crossed flags are still emitted. | Property tests: order invariance; refinement → 2 regions; nested chain; crossed → contested |
| S-6 | `generation/scoped_answer.py` | `LLMConstrainedGenerator`: coverage check (every branch `outcome_value` present, via M-1) and faithfulness check; one retry, then template fallback (logged); persona mode emits `short_answer`. | 0 coverage failures on gold structures |
| S-7 | `pipelines/scopecompose.py` (new) | O1, O2, O3 per proposal §5.8. **Frames come from the cell's backbone.** Each repair in O3 is logged so it can be ablated. Per-stage traces reuse `service/events.py`. | Traces render |
| S-8 | `extraction/probing.py` — **P1** | Run `experiments/wp1_probe --live` **now** on S5 and record `verdict()`. Wire Contrastive Scope Probing into O2/O3 only if the verdict is GO. | Verdict recorded (G5) |
| S-9 | `experiments/fidelity_conflictrag.py` (new) — **P0** | Rebuild ConflictRAG's training and evaluation pairs from Xie et al.'s ConflictQA (ICLR 2024) as their paper describes. Train D-5 and report its accuracy next to their published figure. Also search for released code and write to the authors (if code is released, use it and drop the reimplementation). | Accuracy with CI; gap discussed in the paper |
| S-10 | `experiments/preference_study.py` (new) — **P2** | 60 blind pairwise judgements (O3 vs the H3 comparator, same backbone) by the external annotator, on clarity and actionability. | Win rate with Wilson CI |

### WS-E — Experiments, statistics, pre-registration (joint; weeks 6–14) — **P0**

| # | File | Change |
|---|---|---|
| E-1 | `prereg/preregistration.md` (new) | Fixes everything the confirmatory analysis depends on:<br>- H1–H3 (Holm) and H4 (gated);<br>- H-T, H-C, H-F;<br>- V1–V3 with numeric criteria;<br>- the primary endpoint (CCA-all);<br>- the analysis populations (real-text S1–S3 for H1/H3/H4; natural S1+S2 for H2);<br>- the H3 comparator (B7 or B8, chosen on dev);<br>- the reader;<br>- operating points;<br>- the M-9 power output;<br>- exclusion rules;<br>- the analysis code path. |
| E-2 | `prereg/freeze.py` (new) | Writes `prereg/manifest.json`: SHA-256 of `test.jsonl`, prompts, thresholds, reader, comparator, model IDs (from `discover_models.py`), and the reasoning setting. Then `git tag prereg-v1`. |
| E-3 | `experiments/run_matrix.py` (new) | Runs over (system × backbone × setting × instance) from `experiments/matrix.yaml` with P0/P1/P2. Resumable, append-only `runs/<run_id>/answers.jsonl`. Stops cleanly at a daily cap. Records `fell_back`; **never mixes models within a cell**. |
| E-4 | `scripts/scheduler.py` (new) | Daily runner (Windows Task Scheduler). Drains each Groq model's cap in priority order, then local jobs. Has a **heartbeat file**, sleep prevention (`powercfg /requestsoverride` or the `SetThreadExecutionState` API), API-key expiry checks, and an email-free status file the team checks daily. **Designed to run unattended over the December break.** |
| E-5 | `experiments/score_matrix.py` (new) | Computes all metrics into `results/<run_id>/*.json` and LaTeX/Markdown tables. Every pooled number is shown with its per-stratum breakdown. |
| E-6 | `experiments/oracle_cascade.py` (new) | O2 with gold substituted stage by stage, on the 7B and 20b backbones; Δ per stage plus an order-averaged version. |
| E-7 | `experiments/analysis/hypotheses.py` (new) | Runs V1–V3 first; then H1–H3 (exact McNemar / BCa bootstrap, Holm), then H4 if H3 is rejected; then the secondaries. Writes `results/hypotheses.json`. |
| E-8 | `experiments/stability.py` (new) | Two repeats × 50 instances for API systems; per-instance agreement. Descriptive only, with no override rule. |
| E-9 | `experiments/metric_validity_study.py` (new) | Samples 200 reader, 300 claim and 200 branch decisions, stratified and **blinded to system identity**, and exports them to the annotation CLI. |

### WS-V — Human validation study (owner B, both annotate; weeks 9–11) — **P0**

- 700 decisions are labelled blind by both members, then adjudicated.
- The external annotator labels 100 of them.
- Every automatic number in the paper rests on this.

### WS-R — Reproducibility and release (owner A; weeks 12–16) — P1

- Add `reproducibility.py --from-cache`, which regenerates every table offline from released outputs.
- The bundle contains:
  - the code at the release tag;
  - XScope (with licence handling, B-10);
  - cached outputs;
  - manifests;
  - the pre-registration;
  - the κ pilot history.
- Mint a Zenodo DOI at submission.

### WS-W — Writing (joint; weeks 12–16) — P0

The paper is ≈ 9,000 words plus appendices, following proposal §§1–12. Checklist:

- [ ] Every pilot figure is stated with its population (the 42 LLM-authored probes, or all 128 instances).
- [ ] The exploratory decisive run is disclosed.
- [ ] Defeasible NLI is cited as *Findings*.
- [ ] CONFLICTS and ConflictRAG are cited as arXiv.
- [ ] ConflictRAG's mechanism is described precisely.
- [ ] The two ConflictQA datasets are disambiguated.
- [ ] TCR is described as context–memory.
- [ ] NormBench Level 3+ terminology is used.
- [ ] The SG-DT limitations sentence is quoted.
- [ ] MDCR, Xie & Du, InsLogicBench, TransplantQA and CitizenQuery-UK are cited and differentiated.
- [ ] The Gokul et al. naming clash is addressed.
- [ ] κ is reported with informative-n and p_e.
- [ ] Every 2026 arXiv ID is re-verified against its abstract page in the week before submission.

---

## 4. Compute and API budget

### 4.1 Inputs

| Quantity | Value |
|---|---|
| Groq free tier (`config/models.yaml`) | 200K tokens/day and 1,000 requests/day **per model** |
| Groq models | gpt-oss-20b and gpt-oss-120b (backbones); Qwen ≈ 27B (EVAL) |
| Local | Qwen2.5-7B (backbone) and Llama-3.1-8B (reader) via Ollama, ≈ 5 GB VRAM each, **never run at the same time on one GPU**. Unlimited, ≈ 6–12 s per short call on an RTX 4060 8 GB, ≈ 2× slower on 6 GB. |
| Pilot passage length | 17.7 words mean. Chunk-level passages are ≈ 200 tokens, so K = 6 ≈ 1.2K context tokens. |

### 4.2 Per-call estimates — **measured and replaced in week 1 (gate G1)**

| Call type | Estimate |
|---|---:|
| Plain generation (B0–B6), reasoning setting included | ≈ 1.5K |
| Structured parse (B7) | ≈ 3.5K |
| B8 (2 calls) | ≈ 7K |
| O3 (frames + parse + verbalise) | ≈ 7K |

### 4.3 The matrix (API window: weeks 8–14, Dec 21 to Feb 7 = 49 days)

| Pri | Model | Work | Tokens | Days |
|---|---|---|---:|---:|
| **P0** | gpt-oss-120b | E1/E3: B0, B3, B4, B5, B7, B8, O3 on 200 real-text conditional (under-specified) | 4.7M | 24 |
| **P0** | gpt-oss-120b | H-C: B0, B3 on ≈ 400 personas | 1.2M | 6 |
| P1 | gpt-oss-120b | B7, O3 on 120 controls (fabricated-exception rate) | 1.3M | 6 |
| P1 | gpt-oss-120b | E-8 stability (B7, O3; 2 × 50) | 1.0M | 5 |
| | **gpt-oss-120b total + 10% retries** | | **≈ 9.0M** | **≈ 45 / 49** |
| **P0** | gpt-oss-20b | E2: frames for test (≈ 320 queries) + D3 on the natural hard set | 1.0M | 5 |
| **P0** | gpt-oss-20b | E3: B0, B3, B7, B8, O3 on 200 conditional | 4.1M | 21 |
| P1 | gpt-oss-20b | D3 on a 60-context all-pairs subset | 0.9M | 5 |
| P1 | gpt-oss-20b | E2b ablations on a 120-instance subset | 1.0M | 5 |
| | **gpt-oss-20b total + 10%** | | **≈ 7.7M** | **≈ 39 / 49** |
| **P0** | Qwen ≈ 27B (EVAL) | Claim judge on residual claims, stratified 40% of answers | 2.3M | 12 |
| P0 | Qwen ≈ 27B | Reader cross-check (300 decisions) | 0.2M | 1 |
| P2 | Free frontier tier, if available | B7, B8, O3 on 200 conditional | — | tier limits |

**Before the freeze (weeks 2–7), on the EVAL model:**

| Work | Tokens |
|---|---:|
| Mining verification (≈ 3,000 short candidates) | ≈ 1.2M |
| Negative screening | ≈ 0.8M |
| Branch pre-annotation (≈ 320 × 3K) | ≈ 0.9M |
| Persona paraphrase | ≈ 0.2M |
| **Total** | **≈ 3.1M ≈ 16 days** |

Silver labels and dev QNFA runs use gpt-oss-20b, ≈ 2–3M over weeks 2–6. Everything is cached.

**Local work (free), weeks 8–13:**

| Work | Calls | GPU time |
|---|---:|---:|
| 7B backbone: all 12 systems on 200 conditional + 160 controls/contrast, plus personas, plus E4 oracle cascade and E6 robustness | ≈ 15k | ≈ 40–50 h |
| Reader (Llama-3.1-8B) | ≈ 11k short calls | ≈ 20–25 h |

That is ≈ 70 GPU-hours, split across both machines.

**If G1 measures more than the estimates, cut in this order:**

1. E-8 stability → move to 20b.
2. Controls on 120b → run on 7B and 20b only.
3. Persona H-C on 120b → move to 20b.
4. Drop K from 6 to 4.
5. B4 and B5 on 120b → move to 7B.

**Never cut:** B0, B3, B7, B8 or O3 on 120b on the 200 conditional instances. Those cells are H1 and H3.

**Extend `scripts/budget_estimate.py`** to read `experiments/matrix.yaml` and measured token means, and to print this table. It already serialises tiers that share one model.

### 4.4 Local GPU jobs

| Job | VRAM | Time (4060) |
|---|---|---|
| Chunking + Contriever embeddings of the pool | ≈ 1.5 GB | minutes |
| NLI scoring of within-context pairs | ≈ 1.5 GB | < 1 h |
| DeBERTa-v3-base fine-tune, seq 512, batch 4 × 8 | ≈ 5.5 GB | ≈ 2–3 h |
| Ollama 7B backbone / 8B reader | ≈ 5 GB each | see above |

---

## 5. Annotation budget (realistic, measured paces from the κ pilots plus margin)

| Task | Items | Each | Who | Hours |
|---|---:|---:|---|---:|
| Verify mined test candidates (chunk pairs) | ≈ 500 | 2 min | A | 16.7 |
| Type + relation, double blind (new S1–S4 and controls) | ≈ 450 × 2 | 1 min | A, B | 15.0 |
| Correct branch pre-annotation | ≈ 320 | 5 min | A | 26.7 |
| From-scratch second pass (30%) | ≈ 95 | 8 min | B | 12.7 |
| Verify personas | ≈ 700 | 1 min | B | 11.7 |
| Author + verify contrast sets | ≈ 50 | 10 min | A + B | 8.3 |
| Adjudicate flagged negatives + 20% context audit | — | — | A | 8.2 |
| Adjudicate disagreements | ≈ 70 | 3 min | A + B | 3.5 |
| Validation study (WS-V) | 700 × 2 | 40 s | A, B | 15.6 |
| **Authors total** | | | | **≈ 118 h** (≈ 5.9 h/person/week over weeks 2–11) |
| External annotator (15% of instances on 3 axes, + 100 validation decisions) | | | external | ≈ 12 h |

This is the single largest execution risk, because it falls in exam season. The pre-agreed cuts, in order:

1. Dev branch annotation down to 60 instances.
2. Second pass down to 20%.
3. Contrast sets down to 15 per relation (S4 is diagnostic).
4. Persona verification of S3 down to a 50% sample, with the unverified rate reported.

---

## 6. Schedule (week 1 = Monday 2 Nov 2026)

| Wk | Dates | A | B | Gate / deliverable |
|---|---|---|---|---|
| 1 | 2–8 Nov | Contract v1.3.0; `relabel_v7.py`; chunker; D-1 prototype; **measure tokens per call** | M-1–M-3, M-10; reader and EVAL tiers; start S-9 data build | **G1** (tokens); 18 natural conflicts reproduced |
| 2 | 9–15 Nov | D-2, D-3, D-4 on dev; B-4 fetch with metadata | M-4–M-7; validity battery on pilot-derived dev | **G2 (V1) first read** |
| 3 | 16–22 Nov | B-2 retrieval miner + verifier; D-5 / D-6 | M-8, M-9; S-1, **S-2**; S-9 fidelity run; B-12 CQA probe | **G-CQA**; fidelity number |
| 4 | 23–29 Nov | Mining sprint; B-1 re-split; B-5 contexts | S-3 prompting suite; B-6 personas | **G-YIELD** (verified natural yield projects to ≥ 60 by week 7) |
| 5 | 30 Nov–6 Dec | D-8 silver; D-7 fine-tune; annotation | S-6, S-7; **S-8 `run_probe --live`** | **G4** detection (chunk-level dev); **G5** CSP |
| 6 | 7–13 Dec | D-9, D-11, D-13 (dev); annotation; negative adjudication | Branch correction; B-7 contrast; second pass | ≥ 70% of test targets annotated |
| 7 | 14–20 Dec | Finish annotation; B-1 audit; E-2 | Finish personas; S-5 (P1); E-1; comparator chosen on dev | **G6**; **freeze + `prereg-v1` (Sun 20 Dec)** |
| 8–9 | 21 Dec–3 Jan | E-4 scheduler unattended: P0 Groq + local matrix | — (break) | Heartbeat checked daily; matrix ≥ 35% |
| 10 | 4–10 Jan | Monitor; E-6 oracle cascade (local) | E-9 export; **WS-V** starts | — |
| 11 | 11–17 Jan | D-results tables | WS-V done; V2/V3 | **G7 / G8** |
| 12 | 18–24 Jan | P1 runs; E-5 scoring of completed P0 cells | Begin E-7 on completed cells | — |
| 13 | 25–31 Jan | Write §5–6, §9 detection | Write §7, §9 resolution | — |
| 14 | 1–7 Feb | API window closes; final E-5 / E-7; release (B-10, WS-R) | Write §1–3, §11–12 | **All P0 results final** |
| 15 | 8–14 Feb | Internal strict-review pass (doc 04 checklist) | Same | Revised draft |
| 16 | 15–21 Feb | Supervisor review; Zenodo; submit | Same | **Submission** |

---

## 7. Decision gates (pre-committed)

| Gate | When | Criterion | If it fails |
|---|---|---|---|
| G1 tokens | Week 1 | Measured means ≤ §4.2 + 25% | Apply the §4.3 cut order; fix one reasoning setting for all systems |
| G2 validity (V1) | Week 2 | Per-metric criteria as pre-written | Fix M-4/M-5 **implementation bugs only**. Criteria are not edited. If CCA fails "gold ≥ concat", report it. |
| G-CQA | Week 3 | ≥ 40% mappable and the licence permits | Drop E8; say so |
| G-YIELD | Week 4 | Projected natural S1+S2 test ≥ 60 | Continue mining to week 7. If still short, H2 runs on what exists with its MDD stated, and a reviewer-facing note is added. |
| G4 detection | Week 5 | QNFA F1 ≥ 0.6 on chunk-level dev | One fix round. Below 0.4, re-centre on RQ1/RQ2 |
| G5 CSP | Week 5 | `verdict()` = GO | CSP reported as a negative component; excluded from O2/O3 |
| G6 annotation | Week 7 | Real-text conditional test ≥ 160 (MDD ≈ 10–15) | Below 160, slip the freeze one week, accepting a 42-day API window and cutting P1 |
| G7 reader (V2) | Week 11 | ≥ 0.90 agreement; adversarial NOT_DETERMINED recall ≥ 0.90; null ≥ 0.95 | **No reader switch.** The persona setting (judge-free) becomes the headline, and the reader result is reported as a finding. |
| G8 judge (V3) | Week 11 | ≥ 0.80 | Scoped precision is reported only on the human-labelled subset |

---

## 8. Definition of done (submission checklist)

- [ ] `prereg-v1` predates every test answer file (git tag plus run manifests).
- [ ] No evaluated detector or backbone appears in any test-construction provenance field.
- [ ] H1–H3 are Holm-adjusted; H4 is gated; V1–V3 are reported before them; the secondaries are labelled.
- [ ] Every pooled number has a per-stratum breakdown; S4 and S5 never appear in pooled tables.
- [ ] Reader and judge validity are reported against humans, including NOT_DETERMINED recall and the external annotator.
- [ ] The validity battery table (7 outputs × all metrics) is in the paper.
- [ ] The ConflictRAG-style fidelity result and implementation choices are in an appendix.
- [ ] The oracle-cascade figure; the cost Pareto figure; per-system tokens and calls.
- [ ] Datasheet, manual, κ history, licence statement, snapshot dates.
- [ ] Release bundle with `rehydrate.py`; offline regeneration from cache.
- [ ] Limitations written before the discussion section.
- [ ] All pre-existing and new tests green; the order-invariance gate at 1.0000.
- [ ] Every 2026 arXiv citation re-verified.

---

## 9. Execution risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Natural S1+S2 yield short of 60 | Medium | High (H2 power) | Re-split + retrieval mining from week 3; G-YIELD; H2's MDD stated honestly |
| Groq changes models or limits mid-run | Medium | High | Pinned IDs; cache; a replaced model finishes on its successor and is reported; cells are never mixed |
| Measured tokens per call exceed estimates | High | Medium | G1 and the cut order; P0 cells protected |
| Annotation slips in exam season | High | High | Front-loaded weeks 2–7; pre-agreed cuts (§5); EVAL pre-annotation |
| Laptop scheduler stops over the break | Medium | High | Heartbeat, sleep prevention, both machines, daily remote check |
| Silver circularity inflates the distilled detector | Medium | Medium | Silver from train/dev groups only; H2 on natural test mined without QNFA |
| ConflictRAG fidelity gap | Medium | Medium | S-9 is P0; use their code if released; reported as "ConflictRAG-style" |
| Time drifts back to the site or Studio | Medium | High | Ground rule 0.2 |
