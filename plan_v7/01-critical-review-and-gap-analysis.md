# Critical review and gap analysis: proposal v6, implementation plan v2, and the ScopeCompose repository as of 2026-10-30

This is written the way a demanding Q1/Q2 reviewer would write it, after reading both documents **and** the code and data in `ScopeCompose/`. A reviewer of the eventual paper will see only the paper. They will still find every problem below, because each one shows up in the numbers.

Two documents follow from this one:

- `02-proposal-v7.md` is the rewritten proposal.
- `03-implementation-plan-v3.md` is the plan, written as concrete edits to the existing repository.

---

## 0. Executive verdict

| Artefact | Verdict if submitted today | Why, in one line |
|---|---|---|
| Proposal v6, as a proposal | Major revision | It is well positioned and honestly hedged, but it is built around the wrong bottleneck, and its evaluation instrument is broken in ways the team has since measured. |
| The paper v6 would produce from the current repo | **Reject** | The central claim is untested end to end. The one exploratory test points the wrong way (−12.5 pts, n = 12 instances). The headline metric can be gamed. Every branch-scored number is computed on 42 LLM-authored passages. |
| The engineering in `ScopeCompose/` | Unusually strong | 25k lines, 520 tests, a frozen versioned contract, provenance on every judged number, κ refused for non-independent passes, power analysis, and order invariance proven. Few student projects, and not every published one, have this. |

The gap is not effort. It is **where the effort went**: the instrument is built, but the measurement that would make it a paper has not been taken, and v6 aims that measurement at the wrong place.

The team's own records already contain most of the diagnosis: `docs/journal_readiness_audit.md`, `docs/detector_operator_gap.md` and `docs/wp0_positioning_memo.md`. This review adds:

1. An outside-reviewer reading of what those findings mean for the paper's claim.
2. New literature from May–Oct 2026 that v6 does not handle.
3. A concrete path to a paper that survives review.

---

## 1. Measured state of the project (verified against the repo, not the plan)

All figures were recomputed from `benchmark/data/corpus.jsonl`, `benchmark/data/decisive.json` and `docs/*.md` on 2026-10-30.

| Quantity | Value | Consequence |
|---|---|---|
| Corpus size | 128 instances, **every one exactly 2 passages** | The task as built is passage-*pair* classification, not RAG over K retrieved passages. The "K = 5 hybrid retrieval" in the plan is never exercised by the benchmark. |
| Conflict-type labels | conditional 67, no_conflict 50, temporal 5, factual 4, opinion 2 | The factual, temporal and opinion cells (11 instances combined) cannot support per-class claims. |
| Separation strata | cross-document 59, synthetic split 54, same-guide 15 | — |
| **Natural conditional conflicts** (the paper's subject) | **25** (19 cross-document + 6 same-guide) | The phenomenon the title is about is represented by 25 instances. |
| Instances with branch annotation | **42, all LLM-authored probe passages** (`source_url = example.invalid`) | **Every PR/SR/HCR/SCR number in the project is computed on synthetic text written by a model.** None is computed on real cross-document data. |
| Four-way relation cells | refinement 53, disjoint 10, **redundant 2, opposed 2** | Two of the four relations are effectively unmeasured. Nine pilots and 91 pairings found neither in published policy. |
| Annotation provenance | 74 double-annotated (κ = 1.000 at p_e ≈ 0.54, pilot 9) / 54 model-proposed + one human | The two must never be pooled. The κ = 1.000 needs careful presentation: informative-n, p_e, composition. |
| Stage-1 detector (as run) | precision 0.833, **recall 0.064**; **2 of 67** conditional conflicts recovered; **fires on 0 of 42** branch-bearing instances | End to end, the pipeline almost never reaches its own composition operator. |
| Composition operator, given gold routing and branches | PR **0.976** (81/83) | The operator is not the bottleneck. Given gold inputs it is close to a lookup table, so a reviewer will ask what is novel about it. |
| PR of verbatim concatenation vs gold scoped answer | **0.855 vs 0.699** | **PR is gamed by verbosity.** It cannot separate a composed answer from a dumped one, which is the paper's core comparison. |
| Exploratory "decisive" run | pipeline PR 0.500 vs structured long-context 0.625; Δ = −12.5 [−37.5, +12.5]; McNemar p = 0.51; 12 instances / 24 branches; LLM-authored; straddles all three splits | Underpowered and confounded. Arms are scored differently: routing-level for the pipeline, `align()` for the baseline. **It is not evidence for the thesis.** |
| Selection baselines actually implemented | rerank-top1, NLI-filter, standard RAG | **ConflictRAG-style credibility selection is not implemented.** The paper argues against it but never runs it. CoT, taxonomy-aware and CARE-style baselines are also missing. |
| Branch loss from selection (retrieval side) | rerank-top1 loses 41/83 branches (40/42 instances); NLI-filter 5/83 | A strong, citable effect. The rerank number is near-tautological; the NLI-filter number is the interesting one. |
| LLM judge | implemented, **never validated** against human labels | Recent work (Aug 2026, §3) shows LLM judges are close to chance at detecting *omissions*, which is exactly what SR measures. |
| Models | Groq free tier (gpt-oss-120b/20b, a Qwen 27B), local Qwen2.5-7B via Ollama; **no frontier model, zero spend** | A "pipeline beats structured LLM" result is open to the objection "weak backbone". v6's decisive experiment can falsify the claim but cannot confirm it (README §5 says so). |
| Tests | 520 passing | — |

**The one-paragraph reading a reviewer will arrive at:** *"The authors propose replacing selection with composition. Their own measurements show:*

- *composition works trivially when the conflict is already known;*
- *their detector finds 3% of conflicts;*
- *their primary metric rewards copying the retrieved passages;*
- *their only head-to-head test shows a structured prompt doing better;*
- *every branch-level number comes from synthetic text.*

*The benchmark is small and mostly constructed. The interesting finding — that detection, not resolution, is the bottleneck — is not what the paper claims."*

That paragraph is the thing to prevent.

---

## 2. Fatal and major issues

Numbered so v7 and the implementation plan can respond item by item. "Fatal" means a competent reviewer would reject on it alone.

### F1 — The claimed contribution is in the wrong place (fatal)
v6 makes the resolution operator "the unit of novelty". The measurements say otherwise:

- The operator reaches 0.976 PR given gold inputs. Given the right frames, it is a deterministic table: compose / merge / select.
- Detection is at 6.4% recall.

A reviewer will say the novel part is trivial and the hard part is unsolved. **The paper must move its centre of gravity:**

- to **detecting** conditional conflicts across separately retrieved passages,
- to **measuring the harm** of suppression without a gameable metric,
- to **locating** where end-to-end failure occurs.

The operator stays, as the necessary but simple last step.

### F2 — The evaluation instrument is invalid as specified (fatal)

- PR is containment with no precision side, so verbatim concatenation (0.855) beats the human gold answer (0.699).
- HCR counts only citations of non-existent passages.
- SCR only applies to `no_conflict` instances, and the branch-annotated set has none.
- The judge is unvalidated.
- The 2026 literature shows LLM judges detect omissions at roughly 0.50–0.63 paired discrimination (Fox et al., arXiv:2608.31016). Planted over-inclusions are detected several times more often than omissions (arXiv:2608.01000).

**A suppression metric scored by a standard LLM judge is therefore biased toward under-counting suppression,** which is the direction that hides the paper's own effect. v6 has no defence against this.

### F3 — The benchmark does not test the setting in the title (fatal)

- All instances are 2-passage pairs. There are no distractor passages, no ranking, and no K.
- Branch annotation exists only on LLM-authored text.
- The headline phenomenon — natural cross-document rule/exception — has **25 instances** and **zero** with branch annotation.
- Redundant and opposed have 2 each.

A paper about "conflict-aware RAG" whose benchmark has no retrieval context and no real-text branch annotation will be called a pair-classification study on synthetic data.

### F4 — The comparison the paper argues against is not run (major)
ConflictRAG's resolution policy is the foil in every section of v6. Its pieces are:

- type routing,
- Entropy-TOPSIS credibility selection for factual conflicts,
- recency for temporal conflicts,
- multi-perspective synthesis for opinion conflicts.

None of these is reimplemented. `rerank_top1` is a proxy that "takes retrieval order as credibility". Reviewers will not accept a proxy for the named target system. Missing as well:

- a conflict-driven summarization baseline (CARE-RAG, arXiv:2507.01281);
- the taxonomy-aware prompting of Cattan et al.;
- a strong structured-prompt-plus-self-verification baseline, which prevents a strawman objection.

### F5 — Novelty positioning has new holes (major)
v6 positions against SG-DT, ConditionalQA, Defeasible NLI and CONFLICTS. It misses:

| Missing work | Why it matters |
|---|---|
| **MDCR** (Chen et al., Findings EMNLP 2024, arXiv:2406.11784) | Multi-document conditional reasoning that models *relationships between conditions across documents* — conflicting, equivalent, inclusive. **This overlaps the four-way relation.** It is unaddressed, and a reviewer who knows it will treat that as a novelty failure. |
| Xie & Du (arXiv:2604.17340) | Neuro-symbolic detection of conflicts between rules in *separate* clinical guidelines, with a taxonomy of rule interactions, framed as a RAG-reliability problem. |
| InsLogicBench (ACL 2026) | Defeasible insurance-policy logic; LLMs "struggle with exception clauses". |
| TransplantQA / HERO-QA (arXiv:2605.29084) | Inter-source relation audit; 75% of pairs "complementary (differing in detail or scope)", <0.1% contradictory. **Supports** the premise, but must be cited. |
| CitizenQuery-UK (arXiv:2602.04064) | 22k gov.uk citizen queries with FActScore-style evaluation. **Same data source as this benchmark**, so the difference must be stated. ConditionalQA is also built from UK government pages. |
| Naming clashes | Gokul et al. (arXiv:2504.00180) use "conditional contradictions" for a different phenomenon. Van & Tran (arXiv:2610.36169), "Don't Pick a Winner", is a near-collision with "Beyond Source Selection". |

The SG-DT distinction can now be made in the authors' own words (verified in the PDF, §7): *"NormBench is scoped to intra-provision defeasible scope (not precedent, multi-document synthesis, or open-textured standards)."* v6 does not use this, its strongest available sentence.

### F6 — The "decisive experiment" is mis-specified (major)
v6 frames one comparison — the operator pipeline against a structured long-context LLM — as decisive. There are three problems:

1. **On open weights it is one-directional.** It can falsify the claim but not confirm it (README §5).
2. **The two arms were scored by different code paths**, so even the exploratory number is not a valid comparison (`detector_operator_gap.md` §4).
3. **It tests the wrong question.** Whether a pipeline beats an LLM is a model-tier question. The scientifically interesting questions are:
   - Does explicit scope structure help *a given model*? That needs a same-backbone comparison.
   - Does it help *more as models get smaller*? That needs a scale interaction.
   - Does it reduce harm to the users the exception protects? That needs case-level evaluation.

### F7 — Statistical design is not journal-grade (major)

- Branches are nested in instances, but CIs treat branches as independent.
- No multiplicity control across the many comparisons.
- No pre-registration, even though the team already behaves as if pre-registered.
- No inter-run variance estimate for API models, which are non-deterministic at temperature 0.
- The power table exists (`metrics/power.py`: 137 instances for a 10-point difference), but no experiment has been sized to it.

### F8 — The contribution list is diffuse (major)
v6 carries all of the following:

- five conflict classes,
- four relations,
- nested/crossed flags,
- Contrastive Scope Probing,
- three detector architectures,
- an LLM judge,
- Tier 1/Tier 2,
- a decisive experiment.

The evidence supports almost none of these yet. A journal paper with eight half-measured components reads as a project report. **v7 must cut**: fewer claims, each measured.

### F9 — External validity and release (major)

- UK only, English only, two domains.
- Bank T&C text is ordinary copyright (audit §8), so the banking half cannot be released as text.
- Groq free-tier model IDs change without notice, so model versions must be pinned and outputs released for reproducibility.

### F10 — Writing and framing (moderate, but it decides reviewer mood)
v6 is long, defensive and self-referential: "this proposal stands or falls…", "the decisive test…", repeated hedging about WP0. That works for a proposal panel, not for a journal. A journal paper states:

- the problem,
- the gap,
- three or four contributions,
- the evidence for each,
- the limitations, once.

---

## 3. Minor issues (each cheap, each a reviewer comment if left)

1. **Fix two citation venues.** Defeasible NLI is *Findings* of EMNLP 2020, not the main conference. CONFLICTS (arXiv:2506.08500) has no verified peer-reviewed venue, so cite it as arXiv.
2. **Cite ConflictRAG as an arXiv preprint.** Its IEEE SMC 2026 acceptance is unverified. Describe its mechanism precisely: factual → Entropy-TOPSIS selection; temporal → recency; opinion → multi-perspective synthesis.
3. **Disambiguate "ConflictQA".** Xie et al. (ICLR 2024) and Zhao et al. (arXiv:2604.11209) both use the name.
4. **TCR is context–memory, not inter-source** (WP0 memo §5.7). v6 still describes it loosely.
5. **"Order invariance is provable" is trivially true by construction.** Present it as a property with a one-line argument plus the permutation test, not as a result.
6. **The 37% Level-3+ figure from NormBench** must be cited in its authors' terminology (WP0 memo §5.4).
7. **The four relation names are old** (specificity, lex specialis, rule-base verification). v6 already concedes this, but it still headlines them. v7 should headline the *operationalisation over retrieved text*, and `opposed` as the case specificity does not settle.
8. **"Hallucinated-Condition Rate" over-claims.** As implemented it measures citation of non-existent passages, so rename it or re-implement it.
9. **Datasheet tier counts** must be generated, never typed. The repo already does this; keep it.
10. **Keep the authored-probe instances out of the test split.** They are useful as a diagnostic dev set, never as headline evidence.

---

## 4. What is genuinely strong (and must be visible in the paper)

1. **Two publishable findings already exist, independent of the main claim:**
   - **Tier-1 scarcity, with its mechanism.** gov.uk "guides" are one content item with many URLs, which inflates naive multi-document counts about 3×. Opposed and redundant pairs are absent from published policy because they are, respectively, a drafting error and redundant to write.
   - **Detection, not resolution, is the bottleneck,** with an identified mechanism: lexical-topicality gating suppresses exactly the rule/exception pairs, because they are written by different teams with different vocabulary.
2. **The selection-harm effect is real and large on the retrieval side.** Rerank-top1 deletes 49% of gold branches. A contradiction-aware NLI filter, which a careful engineer would actually ship, still deletes 6%.
3. **Annotation methodology is rigorous.** Nine κ pilots, with failures diagnosed:
   - pilot 2 was voided by a missing field;
   - pilot 5's κ = 1.000 was *rejected* on a pass-plausibility check;
   - κ is refused for non-independent passes.

   That is an exemplary methods story.
4. **Reproducibility infrastructure** — seeded runs, manifests with git SHA and model tiers, cached responses, a frozen versioned contract — is above the publication norm.
5. **Pre-committed negative-result handling in code**: `verdict()`, `prevalence_estimate()` raising, and provenance on judged numbers.

v7 keeps all five and builds the paper around 1 and 2.

---

## 5. Scorecard

The scores rate the paper a reviewer would receive, not the effort. Columns:

- **v6 now**: v6 as written, plus the repo's current evidence.
- **v7 expected**: v7 as designed, if executed to plan.

| Criterion | v6 now | v7 expected | What moves it |
|---|---:|---:|---|
| Novelty (defensible) | 5/10 | 8/10 | Detection-centred contribution, verified SG-DT/MDCR gap, harm metric |
| Technical soundness | 4/10 | 8.5/10 | Precision-aware metrics, judge-free persona accuracy, same-backbone comparisons, oracle cascade |
| Benchmark quality | 4/10 | 8/10 | Real-text branch annotation, K-passage contexts, contrast sets, licence-clean release |
| Experimental design & statistics | 4/10 | 9/10 | Pre-registration, cluster bootstrap, mixed effects, Holm correction, power-sized test set |
| Baselines | 3/10 | 8.5/10 | Fidelity-checked ConflictRAG-style selector, CARE-style, CONFLICTS taxonomy-aware, structured + verify, 3 backbones with same-backbone frames |
| Clarity / focus | 6/10 | 8.5/10 | 4 contributions instead of 8 |
| Reproducibility | 9/10 | 9.5/10 | Already strong; add pinned models and released outputs |
| **Overall** | **~4.5 → Reject** | **~8.5 → Accept / minor revision at a Q1–Q2 IR/NLP journal** | — |

No document can guarantee acceptance. A paper is accepted when its evidence supports its claims and a reviewer cannot find an unaddressed objection. v7 is designed to meet both conditions, provided the measurements are actually taken at the stated sizes. The risk that remains is empirical, not presentational, and v7 says what the paper becomes under each outcome (v7 §11).

---

## 6. Response map: where v7 and plan v3 address each issue

| Issue | v7 section | Plan v3 work item |
|---|---|---|
| F1 contribution location | §1, §3 (C1–C4), RQ2–RQ3 | WS-D (detection), E-6 (oracle cascade) |
| F2 metric validity | §7 (case-complete accuracy via a non-backbone reader, scoped precision, attachment, validity battery, enumerate-then-verify judge) | WS-M, WS-V |
| F3 benchmark realism | §6 (strata, K-passage contexts, personas, real-text branches, contrast sets) | WS-B |
| F4 missing baselines | §8 | WS-S |
| F5 novelty holes | §2 | WS-W (writing), WP0 addendum |
| F6 decisive experiment | §9 (same-backbone, scale interaction, persona harm) | WS-E |
| F7 statistics | §6.7, §10 | WS-E, M-8, M-9, `prereg/` |
| F8 diffuse contributions | §3 (four contributions; CSP, nested/crossed and the 5-class taxonomy demoted to components or appendix) | — |
| F9 validity / licence | §6.6, §12 | WS-B (licence), WS-R (release) |
| F10 framing | whole document (journal register) | WS-W |
| Minor 1–10 | §2, §5, §7, §12 | WS-W checklist |
