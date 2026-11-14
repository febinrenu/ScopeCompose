# v8 package: response to the third-round review (major revision, 7.8/10)

## What is here

| File | What it is |
|---|---|
| `02-proposal-v8.md` | The full proposal, revised. It supersedes `../v7/02-proposal-v7.md`. |
| `03-implementation-plan-v4.md` | The tiered plan. It builds on `../v7/03-implementation-plan-v3.md`, and every v3 section it does not restate still applies. |
| `05-third-round-review-and-responses.md` | A point-by-point response to the review. |

Still valid from `../v7/`:

- `01-critical-review-and-gap-analysis.md` (the repository audit);
- `04-second-round-review-and-responses.md`.

## The five changes, in one line each

1. **Prior art.**
   - MDCR is mapped formally: its relations are our applicability algebra restricted to *different* decisions, so it can never contain a refinement. A coverage study tests that prediction.
   - ConfRAG, Ragability and the WWW 2026 paper are added.
   - A six-dimension comparison covers eight works.
2. **S1 stands alone.**
   - Every cross-document claim rests on natural cross-document pairs only: ≥ 80 in test, floor 60.
   - Below 60, the claims narrow by a pre-declared rule.
   - Split passages (S3) are a mechanism diagnostic only.
3. **QNFA is specified.**
   - It works on structured decision-variable signatures with a unification test (equivalent / subsumes / distinct / uncertain).
   - The three ambiguous fee variables are worked through.
   - VACS, a 4-quadrant alignment set, tests QNFA directly against semantic similarity.
   - The distillation details are complete.
4. **CCA is hard to game.**
   - The reader must quote a verbatim span that NLI-entails the outcome.
   - Hedged modality never counts.
   - Counterfactual persona pairs test the reader.
   - A human entailment audit requires precision ≥ 0.95.
5. **Tiers.**
   - Essential first: B0, B3, B7, B8, O3; QNFA vs the strongest detector; VACS; validity; oracle cascade.
   - Important and Optional work starts only through gates, enforced by a staging script.

Also changed:

- B7 now *is* the released NormBench SG-DT pipeline, with a fidelity check.
- TCR is cited at AAAI 2026.
- The selection claim is softened.
- Crossing is its own relation.
- Proposition 2 is made precise, and Proposition 3 (no deletion) is added.

## New week-1 actions (in addition to `../v7/00-README.md`)

1. **Download and check the licences of** NormBench, MDCR (ACL Anthology attachment), ConfRAG (`XaiverYuan/ConfRAG`) and CONFLICTS (`google-research-datasets/rag_conflicts`).
2. **Draft the two signature schemas** (banking, immigration) and the unification tests.
3. **Build the gov.uk Search API harvester.** It is the main lever for S1 yield.
4. **Measure tokens per call for the NormBench B7 prompt.** It is longer than the old structured prompt, and it sets gate G1.
5. **Confirm the WWW 2026 paper's exact title** in the ACM Digital Library.

## Honest bottom line

The research direction is the same as v7, but now in a form that survives this review's objections.

The decisive uncertainty is **S1 yield**, not design. Verifying about 800 candidates in five weeks is the single most important piece of work. If S1 ends below 60, the paper is still publishable, but under the narrowed claim.
