# Implementation plan v4: tiered, staged, and written as edits to ScopeCompose

This plan is the companion to `02-proposal-v8.md` and supersedes plan v3.

v4 keeps v3's ground rules, contract v1.3.0 and work-stream structure, and changes four things:

- **Tiers.** Every item is now **Essential / Important / Optional**, and nothing in a lower tier starts until its gate passes.
- **New work** from the third-round review:
  - decision-variable signatures and unification;
  - the Variable-Alignment Challenge Set (VACS);
  - CCA with evidence spans, NLI entailment and counterfactual personas;
  - B7 on the released NormBench pipeline, with a fidelity check;
  - a taxonomy-coverage study on MDCR, ConfRAG and CONFLICTS;
  - gov.uk Search API mining for S1.
- **Targets.** S1 now carries the cross-document claims on its own: ≥ 80 test instances, floor 60.
- **Budgets.** Both are re-derived: **≈ 8.1M tokens on gpt-oss-120b**, and **≈ 130–140 author-hours** of annotation, with pre-agreed cuts.

**IDs.** Hyphenated IDs (M-1, D-1, …) are work items. Unhyphenated IDs (D0–D7, B0–B8, O1–O3, S1–S5, E1–E10, H*, V*) refer to the proposal.

---

## 0. Ground rules (unchanged from v3, restated briefly)

1. Keep the 520 tests green, plus the contract discipline, cache, provenance, κ tooling and order-invariance gate.
2. Freeze `site/` and the Studio until submission.
3. S5 is dev-only.
4. **No evaluated system or detector touches test construction.** That means no QNFA, no D3 and no backbone. Evaluation-only models: Qwen ≈ 27B (`EVAL`) and Llama-3.1-8B (`READER`).
5. One scorer for every system.
6. No test-set run before `prereg-v1`.
7. Pin model IDs, and release the cached outputs.
8. **New:** S3 never substitutes for S1 in any cross-document claim (proposal §3.1).

---

## 1. Tiers at a glance

| Tier | What it contains | Starts |
|---|---|---|
| **Essential** | Contract v1.3.0 + relabel; CCA-strict (reader + evidence + NLI) + validity battery + entailment audit; signatures + QNFA + VACS; B0, B3 (+ fidelity), B7 (NormBench + fidelity), B8, O3; S1/S3 benchmark to target; E1, E2 (D0–D5), E2c, E4, E5 | Week 1 |
| **Important** | Distilled detector + cascade (D6/D7); gpt-oss-20b backbone (E3, H3-S3, H4); B5 (H-T); O1/O2; taxonomy-coverage study (E9); robustness (E6); QNFA ablations (E2b); cost (E7) | After G1, G4 and G-YIELD pass |
| **Optional** | ConditionalQA transfer (E8); B1/B2/B4/B6 beyond the 7B backbone; preference study; frontier check; full ablation grid; all systems × all backbones | Only if Essential is complete by week 11 |

---

## 2. Contract v1.3.0 — Essential (week 1)

This is as in plan v3 §2, with two additions:

```python
# contract/frames.py
class Signature(BaseModel):
    domain: Literal["banking", "immigration"]
    slots: dict[str, str]            # normalised values from qnfa/schema/<domain>.yaml
    untyped_slots: dict[str, str] = {}   # free-text values outside the vocabulary
    schema_version: str

class NormativeFrame(BaseModel):     # v3 fields + signature + hedged modality
    signature: Signature
    modality: Literal["obligation","permission","prohibition","statement","hedged"]

# contract/gold.py
class PersonaPair(BaseModel):        # counterfactual pairs (proposal §6.5)
    base_persona_id: str
    flipped_persona_id: str
    flip: Literal["relevant", "irrelevant"]
    attribute: str
    gold_changes: bool
```

`benchmark/relabel_v7.py` also emits `opposition ∈ {crossing, same_scope, unknown}`, so crossing and same-scope pairs are never conflated.

**Acceptance:** the relabel reports **18** natural conditional conflicts (14 in S1, 4 in S2).

---

## 3. Work streams

### WS-M — Metrics, reader, validity (owner B)

| # | Tier | File | Change | Acceptance |
|---|---|---|---|---|
| M-1 | E | `metrics/values.py` | Outcome normalisation and matching (unchanged from v3). **Adds a modality detector**: "may", "might", "could", "in some cases", "subject to" → `hedged`. | ≥ 60 value cases + 30 modality cases |
| M-2 | E | `metrics/case_accuracy.py` | **CCA-strict.** The reader returns `{outcome, evidence, condition_used}`. A determination counts only if all four hold: (1) `evidence` is a verbatim substring of the answer; (2) the local NLI cross-encoder (DeBERTa-v3, already in repo) judges that `evidence + persona` entails `"the outcome is <outcome>"` at or above a dev-calibrated τ; (3) the span is not hedged (M-1); (4) the value matches gold. **CCA-lenient** drops (1)–(3). Outputs CCA-all, CCA-E, CCA-D, over-generalisation, clarification rate. | M-7 battery passes V1 |
| M-3 | E | `api_budget`, `config/models.yaml` | READER tier = Ollama `llama3.1:8b-instruct` with `num_ctx` set and logged, **no fallback**. EVAL tier = Qwen ≈ 27B. Qwen is removed from the backbone tiers. | `smoke_test --tier READER --tier EVAL` |
| M-4 – M-6 | E | `metrics/claims.py`, `scoped.py`, `judge.py` | As in v3: scoped precision, attachment, fabricated exceptions, enumerate-then-verify judge, extended `validate_judge`. | v3 acceptance |
| M-7 | E | `experiments/validity_battery.py` | **Eight** outputs per validation instance: gold, selection, concat-gold, concat-K, padded gold, unattached hedge, **modality hedge**, null. Prints each V1 criterion for each metric. **Criteria are fixed before the first run.** | Runs on dev-built validation data |
| M-8, M-9 | E | `metrics/cluster.py`, `metrics/power_sim.py` | As in v3. `power_sim` reports MDD for S1 at Holm α and for S3 at α = 0.05. | Reproduces proposal §6.7 within ±1 |
| M-10 | E | `run_decisive.py`, `preservation.py` | Retire routing-level scoring from headlines (v3). | — |
| **M-11** | E | `metrics/counterfactual.py` (new) | **Reader-level:** irrelevant-flip change rate and relevant-flip change rate (V2). **System-level:** counterfactual consistency, i.e. the share of pairs where the reader, reading the *system's* answer, changes or holds the outcome as gold says it should. | Fixture with hand-built pairs |
| **M-12** | E | `metrics/coverage_confrag.py` (new) | Reimplements ConfRAG's answer coverage (keyword-substring matching of gold clusters, one-to-one) and applies it to our answers, using branch outcome keywords as judge keywords, for H-K. The reimplementation is documented against their paper (§3, App. D.2). | On the battery: unattached hedge ≥ gold (expected) |

### WS-D — QNFA, signatures, detection (owner A)

| # | Tier | File | Change | Acceptance |
|---|---|---|---|---|
| **D-14** | E | `qnfa/schema/banking.yaml`, `qnfa/schema/immigration.yaml`, `detection/signature.py` (new) | Slot inventory per domain (`event`, `object`, `charge_kind` / `decision_kind`, `qualifiers.*`); value vocabularies with a **generality taxonomy** (e.g. `any_foreign_txn ⊒ cash_withdrawal ⊒ atm_withdrawal`); `unify(σ1, σ2) → equivalent / subsumes(dir) / distinct / uncertain`; `fold_into_applicability()` for subsumption. Schemas are versioned; schema edits after the freeze are forbidden. | Unit tests, including the proposal §5.3 example (three fee variables), on property: unify is symmetric for equivalent/distinct and antisymmetric for subsumes |
| D-1 | E | `detection/frames.py` | One batched call per query and per **cell backbone**. Emits frames with `signature`; values normalised by the schema vocabulary; out-of-vocabulary values go to `untyped_slots`. Verbatim span check (one re-ask, then the frame is dropped and counted). | Frame yield, dropped-span rate and out-of-vocabulary rate on dev |
| D-2 | E | `detection/align.py` | **Replaced:** alignment is `signature.unify`. LLM tie-break only for `uncertain`, with an abstain option. Contriever cosine survives only as an ablation (E2b) and as a VACS baseline. | VACS (D-15) |
| D-3 | E | `detection/variants/base.py`, `stage1.py` | `classify_in_context`, `query` threaded into `PairCandidate`, mandatory score (v3). | Delegation test |
| D-4 | E | `detection/variants/qnfa.py` | Detector D5, per proposal §5.4: unify → commensurability check → fold → `ScopeAnalyser` relation (crossing and same-scope separated) → calibrated score (isotonic on dev; ECE reported). | In `compare` |
| D-5 | E | `detection/variants/conflictrag_mlp.py` | Detector D2, trained on ConflictQA-derived data (S-9). | Fidelity number |
| D-6 | E | `detection/variants/llm_pair.py` | Detector D3: zero-shot pair typing with a verbal confidence. | — |
| D-7 | I | `detection/variants/finetuned.py` | Detectors D4/D6: seq 512, pair-aware truncation. **Three heads:** alignment (2-way), type (conditional / same-scope / crossing / none), relation (5-way). | Peak VRAM < 6 GB |
| D-8 | I | `detection/distill.py` | Silver labels from **train/dev groups only**, using a synthesised query from the rule signature. Kept only if calibrated score ≥ 0.8 or ≤ 0.2. Negatives 3:1. **Hard negatives:** distinct-signature same-provider pairs; high-BM25 distinct-signature pairs; redundant/disjoint. Temperature-scaled on dev. | Zero overlap with test groups (asserted) |
| D-9 | I | `detection/cascade.py`, `threshold.py` | Cascade D6 → QNFA; cost–F1 curve on **chunk-level** dev. | Curve |
| D-10 | E | `stage1.py` | Do not change. This is detector D0 (`legacy_lexical`). | Reproduces P 0.833 / R 0.064 |
| D-11 | E | `variants/compare.py` | F1 at dev-fixed points, P, R, PR-AUC, the 4-way type confusion matrix, cost. **Primary set = natural S1 hard candidates**, including high-lexical different-variable negatives. Adds an embedding-cosine + logistic-regression baseline. | Table for D0–D7 |
| D-12 | E | `scope/relation.py` | `opposition ∈ {crossing, same_scope, unknown}`; `policy_unknown` defaults to `"contested"`; the **same-scope evidence-sufficiency rule** selects only if the TOPSIS margin exceeds a dev-tuned δ, and otherwise presents both sources. | Order-invariance gate at 1.0000 |
| D-13 | I | `experiments/qnfa_ablations.py` | E2b: no query anchoring; string-cosine instead of unification; entailment-only relation; no commensurability check. | Ablation table |
| **D-15** | E | `benchmark/vacs.py`, `experiments/vacs_eval.py` (new) | **VACS.** 160–200 passage pairs from **train/dev groups**, stratified into 4 quadrants: (low / high BM25 similarity) × (same / different variable). Sampling for high-lexical / different-variable and low-lexical / same-variable uses BM25 + EVAL proposals; humans label equivalent / subsumes / distinct (double-annotated on 50%). Compares whole-passage cosine, variable-string cosine, NLI, LLM direct (gpt-oss-20b) and QNFA unification. Reports per-quadrant accuracy (H-V). | κ on VACS labels; per-quadrant table |

**G4 detection gate (week 5).** QNFA F1 must be ≥ 0.6 on the chunk-level dev natural hard set, **and** QNFA must beat variable-string cosine on VACS.
- If either fails: one fix round on the top error category (schema gap, normalisation, applicability).
- If F1 is then still below 0.4: H2 is at risk, and the paper re-centres on RQ1/RQ2.

### WS-B — XScope (A: mining/contexts/release; B: personas/branches; joint annotation)

| # | Tier | File | Change | Acceptance |
|---|---|---|---|---|
| B-1 | E | `strata.py`, `splits.py` | Derive strata, keep S5 in dev, re-split natural groups (archive the old split), quota report per stratum. **S1 test target 80, floor 60.** | 0 leakage |
| B-2 | E | `mining/retrieval_miner.py` | Rule-bearing chunk → retrieve cross-document chunks → EVAL verifier → human. `mined_by="retrieval+verifier"`. **Weekly yield report on verified S1 per 100 candidates.** | Yield tracked |
| **B-4b** | E | `mining/govuk_search.py` (new) | Harvest distinct content items per topic through the gov.uk **Search API** (filter by `document_type`; e.g. a guide plus detailed guidance plus the relevant Immigration Rules appendix). Records `content_id`, `document_type` and `public_updated_at`. This is the main lever for raising S1 yield, because it finds genuinely separate content items rather than parts of one guide. | ≥ 300 distinct-item pairings by week 4 |
| B-3 | I | `mining/frame_miner.py` | QNFA-assisted mining, **train/dev only**. Refuses test groups. | Assertion |
| B-4 | E | `mining/fetch_govuk.py` | Metadata enrichment for B3's TOPSIS. | Coverage reported |
| B-5 | E | `chunker.py`, `contexts.py` | One chunker; K = 6; negatives screened by EVAL; flagged negatives adjudicated by a human, never deleted; 20% audit. | 100% gold inclusion |
| B-6 | E | `personas.py` | Personas per region, paraphrased from source wording, human-verified. **Plus counterfactual pairs** (one relevant flip, one irrelevant flip per conditional instance), with `gold_changes` derived from `branch_for_reader`. | Every conditional instance has ≥ 1 persona per region + 2 pairs |
| B-7 | E | `mining/contrast.py` | S4: ≥ 20 redundant, ≥ 20 **crossing** (verified by `compare` + `values_match`). | — |
| B-8, B-9 | E | `annotation/cli.py`, `agreement.py` | Modes: `branches`, `personas` (now including pairs), `second-pass` (from scratch), `vacs`, `coverage`, `entailment-audit` (blind to system and reader output), `--annotator external`. Agreement per axis per stratum; author vs external. | Pass-plausibility checks extended |
| B-10, B-11 | I | `release/`, `datasheet.md` | As in v3, plus schemas, VACS and coverage-study labels. | Round trip |
| B-12 | O | `external/conditionalqa_probe.py` | ConditionalQA transfer feasibility (gated, as in v3). | G-CQA |
| **B-13** | I | `benchmark/external/coverage_study.py` (new) | **Taxonomy-coverage study (E9).** Load the released data: CONFLICTS (`google-research-datasets/rag_conflicts`), ConfRAG (`github.com/XaiverYuan/ConfRAG`), MDCR (ACL Anthology data attachment). Sample 40–60 each: MDCR document pairs, ConfRAG contradictory questions, CONFLICTS "complementary" instances. Annotate with our frame scheme (signature unification, relation); 30% double-annotated. Report the relation distribution and refinement share, with Wilson CIs. **Check each dataset's licence before use, and record it.** | Distribution table for §2.7 |

### WS-S — Systems and baselines (owner B)

| # | Tier | File | Change | Acceptance |
|---|---|---|---|---|
| S-1 | E | `baselines/registry.py` | Uniform context, instructions, `max_tokens` and reasoning setting (frozen at G1). | All Essential systems run on mocks |
| S-2 | E | `baselines/conflictrag_topsis.py` | B3, called "ConflictRAG-style" (as in v3). Plus `--no-specificity`, and the same-scope sufficiency rule shared with D-12. | Hand-computed TOPSIS test |
| **S-3a** | E | `baselines/sgdt_normbench.py` (new) + `third_party/normbench/` (vendored at a pinned commit, CC BY 4.0, attribution kept) | **B7** = NormBench `article2unit2structure` stage-1 prompt and parser, called through our `api_budget` client (OpenAI-compatible, so Groq works). **Multi-passage adaptation:** passages concatenated as `[p_i] <text>` units; spans must cite `p_i`. Their validator enforces exclusion guards. The SG-DT is compiled to branches and verbalised exactly as in O2. Every adaptation is listed in `baselines/SGDT_ADAPTATIONS.md`. | Runs on 5 dev instances |
| **S-3b** | E | `experiments/fidelity_normbench.py` (new) | Run *their* pipeline and *their* evaluator on 100 NormBench English-slice items, with our closest backbone (gpt-oss-120b). Compare with their reported metrics for the nearest model class, then repeat with our multi-passage wrapper on the same items, presented as single passages, to show the wrapper adds no loss. | Two fidelity tables (G-NB) |
| S-3c | E/I | `baselines/prompting.py` | B0 (E), B8 = B7 + self-verify (E), B5 (I), B4/B6 (O). | Prompts frozen in prereg |
| S-4 | O | `retrieval_side.py` | B1/B2 generate answers (7B only unless budget allows). | — |
| S-5 | E | `composition/operator.py` | `partition_compose` per **proposal Proposition 2**: sign-vector regions; per-attribute satisfiability; drop empty regions; discard (and log) frames whose own constraints are unsatisfiable; label regions `assigned` / `contested` / `uncovered` / `indeterminate` (untyped). **Property test for Proposition 3 (no deletion)** under randomly corrupted relation labels. | Property tests: partition, soundness, order invariance, no deletion |
| S-6 | E | `generation/scoped_answer.py` | Coverage + faithfulness checks; persona mode (v3). Indeterminate and contested regions are verbalised with verbatim conditions and both sources. | 0 coverage failures on gold structures |
| S-7 | E | `pipelines/scopecompose.py` | O3 (E); O1, O2 (I). Frames from the cell backbone; repairs logged. | Traces |
| S-8 | I | `extraction/probing.py` | `run_probe --live` on S5 now; CSP enters O2/O3 only if GO. | G5 |
| S-9 | E | `experiments/fidelity_conflictrag.py` | D2 trained on ConflictQA-derived pairs; accuracy compared with the paper's figure. Search for, or request, the authors' code. | G-CR |
| S-10 | O | `preference_study.py` | As in v3. | — |

### WS-E — Experiments, statistics, pre-registration (joint)

These items are as in v3 (E-1 to E-9). Changes:

- **E-1 (pre-registration)** now fixes:
  - the analysis populations: **S1** for H1–H3; **S3** for H3-S3 and H4;
  - the **S1 narrowing rule** (below 60, cross-document claims become exploratory);
  - the reader;
  - the NLI entailment threshold for CCA-strict;
  - the signature schemas (hash);
  - the VACS split;
  - the H3 comparator;
  - the power simulation output.
- **E-7** runs, in order:
  1. V1–V3;
  2. Holm over H1–H3 on S1;
  3. H3-S3 at α = 0.05, then H4 if H3-S3 is rejected;
  4. the secondary hypotheses (H-T, H-C, H-F, H-V, H-K).
- **E-9** (validation study) now includes the **human entailment audit** of 300 reader decisions, blind to the system and to the reader's output. The external annotator covers 100 of them.
- **New: E-10, `experiments/staging.py`.** This reads the gates file and refuses to launch any Important or Optional run while its gate is open. Without this, the tiers are only advisory.

### WS-V, WS-R, WS-W

These are as in v3. The writing checklist adds:

- the §2.7 six-dimension table, filled from each paper and from the E9 data;
- the MDCR formal mapping paragraph;
- TCR cited as AAAI 2026;
- the WWW 2026 title confirmed against the ACM DL;
- the SG-DT adaptations appendix;
- the NormBench fidelity table.

---

## 4. Compute and API budget

The inputs are as in v3: Groq gives 200K tokens/day per model; local models are unlimited. Per-call estimates are **measured at G1**.

| Call type | Estimate |
|---|---:|
| B0 / B3 / B5 | 1.5K |
| **B7 (SG-DT prompt is longer)** | **4.0K** |
| B8 | 7.5K |
| O3 | 7.5K |
| Frames | 2K per query |

### 4.1 API window: weeks 8–14 (21 Dec – 7 Feb, 49 days)

| Tier | Model | Work | Tokens | Days |
|---|---|---|---:|---:|
| **E** | gpt-oss-120b | B0, B3, B7, B8, O3 × (S1 80 + S3 140) | 4.8M | 24 |
| **E** | gpt-oss-120b | B0, B3 persona setting on S1 (≈ 180 personas) | 0.5M | 3 |
| I | gpt-oss-120b | B7, O3 on 80 controls | 0.9M | 5 |
| I | gpt-oss-120b | Stability, 2 × 50 (B7, O3) | 1.1M | 6 |
| | **120b total + 10%** | | **≈ 8.1M** | **≈ 41 / 49** |
| **E** | gpt-oss-20b | E2: frames for test + D3 on the S1 hard set | 1.0M | 5 |
| I | gpt-oss-20b | E3: B0, B3, B7, O3 × (S1 + S3) | 3.2M | 16 |
| I | gpt-oss-20b | E2b ablations (120 instances) | 1.0M | 5 |
| | **20b total + 10%** | | **≈ 5.7M** | **≈ 29 / 49** |
| **E** | Qwen ≈ 27B (EVAL) | Claim judge (residuals, 40% sample) | 2.3M | 12 |

### 4.2 Before the freeze (weeks 1–7)

**EVAL model (Qwen ≈ 27B):**

| Work | Tokens |
|---|---:|
| Mining verification (≈ 5,000 short candidates, raised for S1 ≥ 80) | ≈ 2.0M |
| Screening | ≈ 0.8M |
| Branch pre-annotation | ≈ 0.9M |
| Persona and counterfactual paraphrase | ≈ 0.3M |
| VACS proposals | ≈ 0.2M |
| **Total** | **≈ 4.2M ≈ 21 days**, inside the 42-day window |

**gpt-oss-20b:**

| Work | Tokens |
|---|---:|
| Dev QNFA + silver | 2–3M |
| NormBench fidelity (S-3b: 100 items × 2 runs) | ≈ 0.8M |
| VACS LLM-direct baseline | ≈ 0.2M |

**gpt-oss-120b:** the fidelity run at 100 items costs ≈ 0.4M.

### 4.3 Local (free)

| Work | Volume | GPU time |
|---|---|---|
| 7B backbone: Essential systems on all test strata, then Optional systems | ≈ 10k calls | ≈ 30 h |
| Reader | ≈ 12k calls | ≈ 25 h |
| NLI entailment checks for CCA-strict | ≈ 12k pairs | < 1 h |
| DeBERTa fine-tune | 1 run | ≈ 3 h |

The reader and the 7B backbone never share a GPU at the same time.

### 4.4 Cut order if G1 measures more than the estimates

1. Stability → 20b.
2. Controls on 120b → 7B/20b.
3. E2b on 20b → dev only.
4. K = 6 → 4.
5. The S3 share of the 120b matrix → 100 instances.

**Never cut** B0, B3, B7, B8 and O3 on gpt-oss-120b on S1. Those cells are H1 and H3.

---

## 5. Annotation budget

| Task | Items | Each | Who | Hours | Tier |
|---|---:|---:|---|---:|---|
| Verify mined S1/S2 test candidates | ≈ 800 | 2 min | A + B | 26.7 | E |
| Type + relation, double blind | ≈ 500 × 2 | 1 min | A, B | 16.7 | E |
| Correct branch pre-annotation | ≈ 300 | 5 min | A | 25.0 | E |
| From-scratch second pass (20%) | ≈ 65 | 8 min | B | 8.7 | E |
| Verify personas + counterfactual pairs | ≈ 1,000 | 45 s | B | 12.5 | E |
| Contrast sets (S4) | ≈ 50 | 10 min | A + B | 8.3 | E |
| Negative adjudication + 20% context audit | — | — | A | 8.2 | E |
| Disagreement adjudication | ≈ 80 | 3 min | A + B | 4.0 | E |
| VACS (180 pairs; 50% double) | 270 | 1.5 min | A, B | 6.8 | E |
| Validation: entailment audit 300 + claims 300 + branches 200, all × 2 | 1,600 | 40 s | A, B | 17.8 | E |
| **Essential subtotal** | | | | **≈ 135 h** | |
| Taxonomy-coverage study (150 items; 30% double) | 195 | 3 min | A, B | 9.8 | I |
| External annotator (15% of instances on 3 axes; 100 audit decisions) | | | external | ≈ 12 | E |

**≈ 135 Essential author-hours is ≈ 6.75 h per person per week over weeks 2–11.** This is the largest risk in the plan. The cuts are pre-agreed and applied in this order:

1. Dev branch annotation to 60 instances (−6 h).
2. S4 to 15 per relation (−2.5 h).
3. Persona verification of S3 to a 50% sample (−4 h).
4. VACS to 160 pairs (−1 h).
5. The S3 test target to 120 (−4 h). This raises S3's MDD by about 1 point.

**S1 mining verification is never cut.** It is the evidence the reviewer asked for.

---

## 6. Schedule (week 1 = Monday 2 Nov 2026)

| Wk | A | B | Gate |
|---|---|---|---|
| 1 | Contract v1.3.0 + relabel; chunker; **D-14 schemas draft**; B-4b Search API harvester; download NormBench / MDCR / ConfRAG / CONFLICTS data and **check licences** | M-1 – M-3; reader + EVAL tiers; **measure tokens per call (incl. B7 SG-DT)**; S-9 data build | **G1** |
| 2 | D-1 frames with signatures; D-2 unify; B-2 retrieval miner | M-2 CCA-strict + M-7 battery on pilot-derived dev; M-11; S-3a NormBench wrapper | G2 (V1, first read) |
| 3 | D-4 QNFA on dev; **D-15 VACS sampling + labelling**; mining sprint | **S-3b NormBench fidelity**; S-2 B3; S-9 fidelity | **G-NB**, **G-CR** |
| 4 | Mining sprint; B-1 re-split; B-5 contexts | B-6 personas + counterfactual pairs; S-5 partition operator | **G-YIELD-1** (projection: S1 ≥ 80 by week 7) |
| 5 | D-11 detector comparison on dev; VACS eval; annotation | S-6, S-7 (O3); S-8 probe | **G4** (dev F1 ≥ 0.6 and VACS beats cosine); G5 |
| 6 | (I) D-7, D-8, D-9 if G4 passes; annotation; negative adjudication | Branch correction; S4; second pass | **G-YIELD-2** (S1 ≥ 60 confirmed, else narrowing rule) |
| 7 | Finish annotation; B-1 audit; E-2 freeze | Personas; E-1 pre-registration (populations, narrowing rule, schemas hash, NLI τ, comparator) | **G6; freeze + `prereg-v1` (Sun 20 Dec)** |
| 8–9 | Scheduler unattended: Essential cells on 120b/20b + local | — (break) | Heartbeat daily |
| 10 | E-6 oracle cascade (local, S1 + S3); (I) B-13 coverage study | E-9 validation study begins (entailment audit) | — |
| 11 | D results; (I) E2b, E6 | Validation done | **G7/G8**; **G-OPT** (start Optional?) |
| 12–14 | Remaining cells; release | E-5 / E-7 | All Essential results final (7 Feb) |
| 15–16 | Strict-review pass; supervisor; Zenodo; submit | Same | Submission ≈ 19 Feb 2027 |

---

## 7. Gates

| Gate | Week | Criterion | If it fails |
|---|---|---|---|
| G1 | 1 | Measured tokens per call ≤ estimate + 25% | Apply the §4.4 cut order |
| G2 (V1) | 2 | Per-metric criteria as pre-written | Fix implementation bugs only. Criteria are never edited. |
| **G-NB** | 3 | Our run of NormBench's pipeline is within 10 points of their reported structure metric for the nearest model class, **and** the wrapper loses ≤ 3 points | Investigate and report. If the gap persists, B7 is reported as "SG-DT-style (NormBench pipeline, open-weight)", with the gap in an appendix. |
| **G-CR** | 3 | D2 is within 10 points of ConflictRAG's reported detector accuracy | Report the gap; B3 remains "ConflictRAG-style" |
| **G-YIELD-1** | 4 | Projected S1 test ≥ 80 | Increase B-4b harvesting; move B time from S4 to mining |
| G4 | 5 | QNFA dev F1 ≥ 0.6 **and** beats variable-string cosine on VACS | One fix round; if F1 < 0.4, re-centre on RQ1/RQ2 |
| G5 | 5 | CSP `verdict()` = GO | CSP is not used in O2/O3 |
| **G-YIELD-2** | 6 | S1 test ≥ 60 confirmed | **Narrowing rule:** H1–H3 become exploratory; the title and abstract narrow (proposal §3.1) |
| G6 | 7 | S3 ≥ 120 and S1 ≥ 60 | Slip the freeze one week and cut Important API work |
| G7 (V2) | 11 | Reader precision ≥ 0.95 / recall ≥ 0.85 vs the entailment audit; counterfactual and null criteria met | No reader switch; the persona setting becomes the headline |
| G8 (V3) | 11 | Judge ≥ 0.80 | Scoped precision on the human subset only |
| **G-OPT** | 11 | All Essential cells complete | Optional work is not started |

---

## 8. Definition of done

Everything in plan v3 §8, plus:

- [ ] Every cross-document claim in the paper cites S1 numbers only, and S3 is described as a mechanism diagnostic.
- [ ] The QNFA definition, schemas, unification rules, the VACS results and the training details (labels, splits, balance, hard negatives, calibration with ECE) are in the paper or its appendix.
- [ ] CCA-strict and CCA-lenient are both reported, together with the entailment-audit precision and recall and the counterfactual pass rates.
- [ ] The NormBench fidelity table and the SG-DT adaptations list are included.
- [ ] The §2.7 comparison table is filled, and the E9 coverage results (MDCR, ConfRAG, CONFLICTS) are reported, if run.
- [ ] Crossing pairs are never counted as composition successes. Same-scope selection respects the sufficiency rule.
- [ ] Propositions 2–3 have their property tests linked in the repository.

---

## 9. Execution risks (updated)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| S1 yield below 80 or 60 | **High** | High | B-4b Search API harvesting; G-YIELD-1/2; pre-declared narrowing rule |
| Signature schemas miss real variation (high out-of-vocabulary rate) | Medium | High | Out-of-vocabulary values become untyped, then `uncertain`, then the LLM tie-break. The OOV rate is reported. One schema revision is allowed before the freeze, none after. |
| NormBench pipeline does not transfer to open weights or to multi-passage input | Medium | Medium | G-NB; report the gap; B8 remains a strong comparator |
| Annotation overruns in exam season | High | High | Tiered annotation; cuts in §5; S1 verification protected |
| The reader leaks world knowledge | Medium | High | CCA-strict evidence + NLI; counterfactual pairs; entailment audit (G7) |
| Groq volatility | Medium | High | Pinned IDs, cache, no cell mixing |
| Licence of external datasets (MDCR, ConfRAG, CONFLICTS) for the coverage study | Low–medium | Low | Analysis only, labels released without source text if required; checked in week 1 |
