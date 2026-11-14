# Third-round review (verdict: major revision, 7.8/10): point-by-point responses

The third review endorses the reframing. It asks for five things:

- deeper prior-art work;
- separation of the natural cross-document (S1) evidence;
- a rigorous QNFA;
- a CCA that is harder to game;
- a realistic, tiered schedule.

Every point is addressed in `02-proposal-v8.md` (P§) and `03-implementation-plan-v4.md` (plan §), except where it is marked **residual**.

Several new citations were verified against their primary pages before use:

| Paper | Verified details |
|---|---|
| ConfRAG | ACL 2026, pp. 303–322; metric definitions read in the PDF; code and data at `XaiverYuan/ConfRAG` |
| Ragability | LREC 2026, pp. 2323–2333 |
| TCR | Published at **AAAI 2026** (not only arXiv); now corrected |
| NormBench | Code and data released under CC BY 4.0 |
| CONFLICTS | Data released as `google-research-datasets/rag_conflicts` |
| MDCR | Task definition read in full |
| WWW 2026 Conflict-Aware RAG | Title **not** confirmable by search. It is cited by DOI, with a confirm-before-submission flag. |

---

## §3A — MDCR is a closer competitor

**Response.** We now state the relationship in our formalism instead of asserting a difference (P§2.2):

- **MDCR documents each govern their own outcome.** Each document covers one scholarship or one job.
- **MDCR's relations hold between conditions.** Its conflicting, equivalent and inclusive relations sit between conditions of *different* outcomes.
- **In frame terms, these are our applicability set-relations restricted to distinct decision variables:**

  | MDCR relation | Our set relation |
  |---|---|
  | conflicting | disjoint |
  | equivalent | equal |
  | inclusive | subset |

- **XScope adds the outcome dimension on a shared variable.** Defeat, and therefore suppression, can only occur there.
- **No MDCR pair can be a refinement under our definitions.** MDCR can also be solved without ever deciding whether two passages govern the same outcome. That decision is exactly what QNFA does.

This is framed as a **prediction, tested empirically**. The taxonomy-coverage study (P§6.8, E9; plan B-13) annotates 40–60 MDCR document pairs with our frame scheme and reports the relation distribution. We predict ≈ 0% same-variable refinement.

## §3B — ConfRAG is a missing benchmark comparison

**Response.** ConfRAG is added and examined at the level of its metrics (P§2.1).

- Its *answer coverage* matches a gold cluster when any judge keyword appears as a **substring** of a predicted answer. Their own Figure-1 example gets full credit when one answer contains "Yes" and another contains "No".
- So a response that lists both outcomes without attaching them to anyone is fully covered.
- Its unit is the viewpoint cluster, not an applicability-scoped branch.

**What we added:**

1. **The six-dimension comparison table the reviewer asked for** (P§2.7). It covers eight works, including ConfRAG, CONFLICTS, Ragability, MDCR, ConditionalQA, SG-DT and InsLogicBench.
2. **A reimplementation of ConfRAG-style answer coverage** (plan M-12). This supports a new secondary hypothesis, **H-K**: coverage scores unattached hedges at least as high as gold, while CCA-all scores them ≤ 0.05.
3. **60 ConfRAG contradictory questions annotated with our scheme** (E9). If a non-trivial share is refinement, suppression occurs in open-web RAG too, and ConfRAG neither labels nor scores it.

## §3C — SG-DT; B7 must be faithful

**Response.** NormBench is released (`github.com/AlexJJJChen/NormBench`, CC BY 4.0) with its stage-1 SG-DT pipeline, validator and evaluator. The pipeline uses an OpenAI-compatible client, so it runs on Groq.

| Plan item | Change |
|---|---|
| **S-3a** | B7 now *is* that pipeline (vendored at a pinned commit, with attribution). The multi-passage adaptation is passage-ID-prefixed units, with exclusion guards enforced by their validator. Every change is listed in `SGDT_ADAPTATIONS.md`. |
| **S-3b** (Essential) | A fidelity check runs their pipeline and evaluator on 100 NormBench English items and compares the result with their reported numbers. It also checks that our multi-passage wrapper loses ≤ 3 points. |
| **G-NB** | The gate decides how B7 is labelled if a gap remains. |

## §4 — CCA safeguards against gap-filling

**Response.** All three tests the reviewer proposed are adopted, plus one more (P§7.2, §6.5, §7.5; plan M-2, M-11, E-9).

1. **Answer-only evidence.** The reader must return a **verbatim span from the answer**. A local NLI cross-encoder must judge that span plus the persona as entailing the outcome. **Hedged modality does not determine an outcome:** "may", "might", "could" and "in some cases" are excluded. This is **CCA-strict**, the primary metric; CCA-lenient is reported alongside. The reviewer's example ("Premium customers *may* receive fee waivers") is scored NOT_DETERMINED.
2. **Counterfactual persona pairs.** Each conditional instance gets one relevant flip (the outcome must change) and one irrelevant flip (it must not). These pairs:
   - validate the reader (V2: irrelevant-flip change rate ≤ 5%; relevant-flip change rate ≥ 90%);
   - also give a system-level *counterfactual consistency* score.
3. **Human entailment audit.** 300 reader decisions are judged by two blind annotators. The question is *"does the answer, by itself, establish this person's outcome?"*, not whether the reader guessed right. The external annotator covers 100. V2 now requires **determination precision ≥ 0.95**, because a false determination inflates CCA, and recall ≥ 0.85.
4. *(Added)* **A modality-hedge item** joins the validity battery, which now has eight outputs.

## §5 — Formalisation

| Point | Change |
|---|---|
| "The only case where selection is right" is too strong | Rewritten (P§4.2): same-scope conflicts are *the principal case* in which credibility- or recency-based selection is appropriate, *subject to evidence sufficiency and source reliability*; otherwise abstain or present both. Implemented as a sufficiency rule: selection only if the TOPSIS margin exceeds a dev-tuned δ (plan D-12). |
| Crossing is not default-plus-exception | Crossing is now its **own relation class**. It is never counted as a composition success. A conditional conflict is now **refinement only**. |
| Partition conditions are imprecise | Proposition 2 is restated over a typed constraint language with per-attribute decidable satisfiability. It specifies how the operator treats internally inconsistent frames (discarded and logged), empty regions (dropped), regions with no applicable frame (*uncovered*), and untyped constraints (*indeterminate*, with verbatim conditions and no inferred outcome). |
| *(Added)* | **Proposition 3 (no deletion).** Under the default policy, the operator never drops a frame, even when relation decisions are wrong. Errors become misattachments, which CCA penalises, not silent suppression. A property test with corrupted relation labels checks this (plan S-5). |

## §6 — QNFA specification

| Question | Answer |
|---|---|
| 1. What is QNFA? | **Query-anchored Normative Frame Alignment**, now defined with an algorithm (P§5.1, §5.4). The novel part is stated explicitly: (a) a query-anchored, *structured decision-variable signature* as the unit of comparison; (b) a three-valued **unification** test (equivalent / subsumes / distinct) with a separate *uncertain* state and an abstaining tie-break; (c) an outcome-commensurability check; (d) distillation of the whole decision. LLM frame extraction on its own is acknowledged as not new. |
| 2. Why better than semantic similarity? | This is tested directly by **VACS**, the Variable-Alignment Challenge Set (P§7.4; plan D-15). It has 160–200 human-labelled pairs in four quadrants: low/high lexical similarity × same/different variable. QNFA is compared with whole-passage cosine, variable-string cosine, NLI and direct LLM judgement. **H-V** predicts the largest gains in the two off-diagonal quadrants. G4 now requires QNFA to beat cosine on VACS. |
| 3. Ambiguous variables | Signatures have typed slots with a generality taxonomy. The reviewer's three fee variables are worked through in P§5.3. Withdrawal fee vs FX charge are *distinct*, because `charge_kind` differs and the charges apply cumulatively. The FX charge *subsumes* the overseas ATM conversion fee, which makes it a candidate refinement. A similarity threshold cannot make these distinctions; unification can. |
| 4. How does the distilled model learn? | Now specified (P§5.5; plan D-7, D-8): exact labels, three heads, document-group splits, silver drawn only from train/dev groups and filtered by calibrated score, 3:1 balance plus class weights, three kinds of hard negative (including high-lexical different-variable pairs), temperature scaling with ECE, and comparison against an **embedding cosine + logistic regression baseline**. |

**Residual:** signature schemas are hand-designed per domain. This is listed as a limitation, and E2b measures what the schemas contribute.

## §7 — Strata

**Response.** Adopted in full (P§3.1, §6.2).

- **S1** carries every cross-document claim. **S2** is secondary. **S3** is a resolution-mechanism diagnostic, with its claims explicitly scoped. S4 is a stress test. S5 is dev only.
- H1–H3 are now confirmatory on **S1**. H3-S3 and H4 form a separate mechanism family on S3.
- **S1 has its own minimum:** ≥ 80 in test, floor 60.
- **Below 60, the narrowing rule applies:** H1–H3 become exploratory, and the title and abstract narrow. S3 never substitutes for S1.

**What makes the target reachable:** gov.uk Search API harvesting of distinct content items (plan B-4b), and two yield gates (G-YIELD-1 in week 4, G-YIELD-2 in week 6).

**Residual:** the pilot has 14 S1 instances, and only about one is in the current test split. Reaching 80 means verifying roughly 800 new candidates. This is the plan's highest risk, and it is budgeted (26.7 h) and protected from cuts.

## §8 — Additional benchmarks

Ragability, ConfRAG, the WWW 2026 Conflict-Aware RAG paper and InsLogicBench are all now in P§2.1–2.3 and in the comparison table.

The WWW 2026 title could not be confirmed by search. It is cited by DOI and flagged for confirmation against the ACM DL.

## §9 — Feasibility and scale

**Response.** Adopted, using the reviewer's tiers almost verbatim (P§9; plan §1).

| Tier | Contents |
|---|---|
| **Essential** | B0, B3, B7, B8, O3; QNFA vs the strongest detector; VACS; CCA validation; oracle cascade |
| **Important** | Distilled model, the 20b backbone and scale, B5, robustness, ablations, the coverage study |
| **Optional** | ConditionalQA transfer, the full grid, the preference study, a frontier model |

Important work starts only after G1, G4 and G-YIELD pass. Optional work starts only if Essential is complete by week 11. A staging script enforces the gates (plan E-10). No longer planned: 12 systems × 3 backbones.

**The budgets were re-derived:**

| Resource | Use | Capacity |
|---|---|---|
| gpt-oss-120b | ≈ 8.1M tokens | 41 of 49 days |
| gpt-oss-20b | ≈ 5.7M tokens | 29 of 49 days |
| Essential annotation | ≈ 135 author-hours | — |

The annotation cuts are pre-agreed.

## §10–12 — Summary assessment

We agree with the reviewer's reading. The paper's strength is the *combination* of four elements:

- cross-document exception suppression as a measurable RAG failure;
- QNFA;
- XScope with its S1 stratum;
- evidence-grounded CCA.

The resolution operator is no longer presented as the main novelty. P§1.4 says explicitly that the operator is not novel, and maps each contribution to the claim and the evidence it supports.

---

## Remaining risks (unchanged in kind, sharper in measure)

1. **S1 yield.** This is now the single deciding risk for the cross-document claims, with a pre-declared fallback.
2. **H3 on S1 is powered only for ≈ 14–21-point differences.** A smaller true effect will appear as inconclusive on S1 while being established as a mechanism on S3. The paper is written to say exactly that.
3. **About 135 Essential author-hours land in exam season.**
4. **Open weights only,** unless a free frontier tier is available at run time.

Accepting these risks is a choice, not an oversight. Each has a gate, a fallback and a sentence already written for the paper.
