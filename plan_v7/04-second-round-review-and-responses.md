# Second-round strict review of v7 / plan v3, and how each point was addressed

This document is the second of two review rounds:

- **Round 1** (`01-critical-review-and-gap-analysis.md`) reviewed v6 together with the repository.
- **Round 2** reviewed the first draft of v7 and the plan. It combined:
  1. an **independent reviewer pass**, run without sight of our reasoning and given only the documents and the repository;
  2. our own re-read against the repository data.

Every point below has been fixed in the current `02-proposal-v7.md` and `03-implementation-plan-v3.md`, unless it is marked **residual**.

**Round-2 verdict on the first v7 draft:** *major revision, and probably rejection at IP&M if executed as written.* The ideas held up. The validity design did not:

- the method under test helped build its own benchmark;
- the primary metric was validated on the authors' own data, using a reader that was also a backbone;
- the detection result would have been measured on contexts filtered by the same detector.

None of this needed new ideas. All of it needed redesign before the freeze.

---

## 1. Facts the reviewer raised, checked against the repo

| Claim | Checked | Consequence |
|---|---|---|
| All 42 branch-bearing pilot instances are LLM-authored probes (S5) | **True** (`source_url = example.invalid`) | Every pilot figure is now stated with its population (abstract, §1.3) |
| The pilot `conditional` type includes 10 `disjoint` and 2 `redundant` pairs | **True** (53 refinement, 10 disjoint, 2 redundant, 2 opposed = 67) | v7 §4.2 relabels them deterministically; `relabel_v7.py` (plan §2.4) |
| Natural conditional conflicts under v7 definitions | **18** (14 cross-document, 4 same-guide) | Stated in §4.2 and §6.2 |
| …and only **1** of them is in the current test split (16 in train) | **True** | Natural groups are re-split before the freeze (§6.3, plan B-1); all ≥ 60 natural test items must be mined |
| 83 of 256 passages have `source_type: unknown`; 148 of 256 have no date | **True** | Without metadata enrichment, the ConflictRAG-style TOPSIS would degenerate. Enrichment from gov.uk `document_type` / `public_updated_at` added (plan B-4) |
| All 15 same-guide records carry `construction: split` | **True** | Stratum is derived from `separation`, never from `construction` (plan §2.1, B-1) |
| All 128 records have `disagreed: false` | **True** | κ = 1.000 is now reported with informative-n and p_e. An external annotator labels 15% (§6.6). |
| The exploratory decisive run (−12.5 points) was not mentioned | **True** | Now disclosed in §1.3 |

---

## 2. Major concerns and responses

| # | Concern | Response | Where |
|---|---|---|---|
| R2-1 | Pilot evidence was drawn from data the proposal excludes, and populations were mixed | Each figure is stated with its population; the decisive run is disclosed; labels are relabelled under v7 | v7 Abstract, §1.3, §4.2 |
| R2-2 | **Circularity.** QNFA mined the benchmark; QNFA/D3 filtered hard negatives; a backbone pre-annotated gold; personas reused attribute names | Test construction uses **only non-evaluated models**: Qwen ≈ 27B (EVAL) and humans. QNFA-assisted mining is limited to train/dev. Flagged negatives are human-adjudicated, never deleted, with a random 20% audit. Persona statements are paraphrased from source wording, and their overlap with system outputs is reported. | v7 §6.3–6.6; plan ground rule 4, B-2/B-3/B-5/B-6 |
| R2-3 | **CCA validity.** Concatenation is determinate, so CCA cannot penalise it. The reader was a backbone. G7 allowed switching readers after the freeze. | V1 now has **per-metric** criteria: CCA "gold ≥ concat"; scoped precision "gold > concat"; hedge ≤ 0.05. CCA is always reported with precision and length. The reader is Llama-3.1-8B, outside every evaluated family, and is pre-registered. Adversarial NOT_DETERMINED recall is tested. **No reader switch**: if the reader fails, the persona setting becomes the headline. | v7 §3 (V1–V2), §7.2; plan M-3, M-7, G7 |
| R2-4 | **Hypotheses ill-posed.** H1 was near-tautological and depended on the specificity criterion. CCA-E was the wrong endpoint. H2 was tested against 0, with recall@P undefined for discrete detectors. H3 pooled across an interaction. H4 was on a logit scale and confounded by model family. | Primary endpoint is now **CCA-all** (every region correct), which penalises selection whichever branch it picks. Specificity is ablated. H2 uses **F1 at operating points fixed on dev**, and every detector must emit a score. The 15-point margin is assessed by a lower confidence bound. H3 is per primary backbone against the **better of B7 and B8**. H4 is **gpt-oss-20b vs 120b** (same family), on the **risk-difference** scale, gated on H3. | v7 §3, §5.4, §10; plan D-3, E-1 |
| R2-5 | **Power and strata inconsistent.** "Never pooled" contradicted an MDD of ≈ 10. The power model assumed branch-level independence. | Analysis populations are now explicit: real-text S1–S3 for H1/H3/H4, with every pooled figure broken down by stratum; natural S1+S2 for H2. Instance-level McNemar MDD table: ≈ 9–13 points at n = 200 (Holm α). The natural subset alone is ≈ 16–24 points and **descriptive**. Simulation-based power analysis (M-9) goes into the pre-registration. | v7 §6.7, §10; plan M-9 |
| R2-6 | **Baseline fairness.** O3 used a 20b extractor in a 7B cell; O3 has more compute; reasoning settings were unequal. | Frames are extracted by **the cell's own backbone**. The H3 comparator is the better of B7 and B8 (B8 is closer in compute), with tokens per system reported. **One uniform reasoning setting** for every system, fixed at G1. | v7 §5.1, §8.2; plan S-1, S-7, D-1 |
| R2-7 | **The ConflictRAG reimplementation was not credible.** The detector's training data was unclear; criteria were invented; metadata was missing. | The D2 detector is trained the way ConflictRAG describes (ConflictQA-derived; three types plus none), **not on our labels**. The fidelity check (S-9) is now **P0**, and we ask for or search for released code. Metadata is enriched. Specificity is ablated. The system is labelled "ConflictRAG-style" throughout. | v7 §8.1; plan D-5, S-2, S-9, B-4 |
| R2-8 | **Novelty survives only narrowly.** S2 resembles chunked ConditionalQA; disjoint pairs are CONFLICTS' "complementary". | The failure is now named **cross-document exception suppression**, i.e. the multi-document case the SG-DT paper excludes in its own words. Disjoint pairs are mapped to "complementary" explicitly. A new secondary hypothesis, **H-T**, tests whether CONFLICTS-style prompting handles disjoint pairs but suppresses refinement pairs; this is the empirical evidence for the taxonomy gap. A cross-passage ConditionalQA set (E8) is P1, gated on a week-3 mapping and licence check. It is not P0, because ConditionalQA's conditions are often requirements without a stated alternative outcome. | v7 §1.1, §2.1, §3, §9 E8; plan B-12 |
| R2-9 | **Inconsistencies.** "Nine resolvers" vs 12 systems; the E1 scope vs the plan; controls only on 7B; Qwen-27B listed as a backbone yet cut in P1; the stratum derivation; the persona/annotation order; the WS-V hours; the judge-family sentence. | Wording is now "nine baselines + three variants". The E1 and E3 system lists match the plan matrix. Controls on 120b are P1, cut only in the stated order. Qwen-27B is **EVAL-only**, not a backbone. Stratum comes from `separation`. The annotation budget includes WS-V. Judges are never from an evaluated family. | v7 §8–9; plan §4.3, §5 |
| R2-10 | **The budget did not hold.** Structured outputs are 3–5K tokens; stability, ablations, D3 all-pairs, retries and silver-query synthesis were missing; annotation paces were too optimistic. | Per-call estimates are now by call type and **measured at G1**. The API window is extended to **49 days** (weeks 8–14). The matrix includes stability, ablations on a subset, D3 on the hard set plus a 60-context subset, and 10% retries. There is an explicit cut order, and the H1/H3 cells are protected. Silver uses synthesised queries. Annotation is re-estimated at ≈ 118 author-hours plus ≈ 12 external, with pre-agreed cuts. | plan §4–5 |
| R2-11 | **Statistics.** Too few clusters for GEE; unspecified one-sided bootstrap; Holm vs gatekeeping; H2 margin; uninformative TOST; the "2 SD" override; κ from authors only. | Primary tests: **exact instance-level McNemar** (H1, H3) and a **BCa cluster bootstrap** (H2). Holm over H1–H3, with fixed-sequence gatekeeping for H4. GEE with **bias-reduced (Mancl–DeRouen)** errors and sign-flip permutation as sensitivity analyses. The H2 margin is assessed by a lower confidence bound. TOST is removed (single-reference AC is descriptive). The 2-SD rule is removed. An external annotator covers 15% of instances. | v7 §10; plan M-8, E-7 |

---

## 3. Minor concerns, all addressed

| Concern | Response |
|---|---|
| DeBERTa seq 384 truncates two chunks | seq 512 with pair-aware truncation (plan D-7) |
| Ollama `num_ctx` is a silent truncation risk | Set explicitly and logged for every local model (plan M-3) |
| Gold-centred chunks are a positional artefact | One sentence-aligned chunker for every document (v7 §6.4; plan B-5) |
| G4 tuned on the sentence-level pilot | Chunk-level dev set (plan D-9, G4) |
| Clarifying questions penalised unevenly | Reported separately, as a rate, with a sensitivity analysis that excludes them (v7 §7.1) |
| Unverified 2026 arXiv IDs | Two that could not be verified (the conflict-typing cascade paper; "Don't Pick a Winner") were **removed**. The SG-DT limitations quote was **verified in the PDF**. NeurIPS 2026 is confirmed for arXiv:2608.31016. All IDs are re-checked in the week before submission (plan WS-W). |
| The reader probe is not Fox et al.'s intervention | Wording changed to "the same principle", with no claim of being their design (v7 §7.2) |
| Proposition 2 presented as a contribution | Called elementary; its role is to locate the difficulty (v7 §1.4, §4.3) |
| A 5-class type head with 2 and 4 examples in some classes | 3-way type head plus a 5-way relation head (v7 §5.6; plan D-7) |
| WS-V not blinded | Blinded to system identity (plan E-9) |
| Unattended runs over the break | Heartbeat, sleep prevention, key checks, both machines (plan E-4) |
| US stretch goal | **Dropped** (v7 §6.1, §12) |
| Bayesian sensitivity priors unspecified | Bayesian model dropped; bias-reduced GEE plus permutation instead |
| Share of hand-written personas | Reported (`hand_written` flag; v7 §6.5) |

---

## 4. Residual risks no document can remove

These are stated so that nobody, including you, mistakes a strong plan for a guaranteed result.

1. **Natural yield.** At least 60 natural S1+S2 test conflicts have to be newly mined and verified in about five weeks. If yield is low, H2 runs at a larger MDD, and the paper must say so (G-YIELD).
2. **H3 may fail.** A strong 120B model with a good structured prompt may preserve exceptions as well as the hybrid. The paper is designed to remain publishable in that case (v7 §11), but its headline then changes.
3. **Annotation load in exam season.** About 118 author-hours is real work. The cuts are pre-agreed, but each one costs some evidence.
4. **Free-tier volatility.** Groq can change models or limits. Caching and pinning protect what is already done, not what is not yet run.
5. **The external annotator** is a person who has to say yes. Line one up in week 1.

---

## 5. Expected reviewer outcome after these changes

Assuming the plan is executed and the P0 cells complete, the pre-registered review checklist in plan §8 then reads green, and the remaining objections a Q1/Q2 reviewer could raise are of the kind settled in a minor revision:

- scope: English, UK, open-weight models;
- the size of the natural subset;
- requests for a frontier-model check.

Our estimate for a well-matched journal (Information Processing & Management, Knowledge-Based Systems, NLP Journal) is **minor-to-major revision on first decision**, with acceptance likely after one round. The most probable first-round request is *"add a frontier model"*, which is why the optional free-tier check is kept in the plan.

A broader-remit Q2 journal (Journal of Intelligent Information Systems, Applied Intelligence) has a higher first-decision acceptance probability for the same paper.

No plan can guarantee acceptance. What this one guarantees is that every likely reason to reject is either fixed or explicitly reported.
