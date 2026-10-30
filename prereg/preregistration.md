# Pre-registration

Status: **draft, not yet frozen.** It is frozen by `python -m prereg.freeze` followed by a commit and the
`prereg-v1` tag. Until the tag exists, `experiments.staging.assert_frozen` refuses every test-split run.
Anything written here after the tag is a deviation and must be reported as one.

## 1. Question

Do conflict-aware RAG designs that *select* between retrieved passages suppress valid exceptions that are
stated in a separately retrievable document, and can an explicit scope-composition step recover them?
The population of interest is natural cross-document rule/exception pairs (stratum **S1**).

## 2. Confirmatory hypotheses (S1 only)

Tested with the **exact** McNemar test (`mcnemar(..., force_exact=True)`) on case-complete accuracy,
CCA-all strict, Holm-adjusted across H1-H3. The unit of analysis is the instance.

| | Hypothesis |
|---|---|
| **H1 harm** | B3 (ConflictRAG-style, oracle conflict detection) has lower CCA-all than B0 (full context) on the primary backbone |
| **H2 detection** | QNFA's conditional-class F1 on the S1 hard-candidate set, at a dev-fixed operating point, exceeds the best non-QNFA detector; assessed by a one-sided 95% BCa lower bound against 15 points |
| **H3 resolution** | O3 has higher CCA-all than the H3 comparator (the better of B7c and B8, **chosen on dev and recorded in the manifest**) on the primary backbone |

Mechanism family (S3, separate alpha = 0.05; H4 gated on H3-S3): H3-S3 and H4 are **not run** in the
seven-day schedule (the 20b scale study is cut) and are reported as not run.

Secondary, unadjusted, with confidence intervals: H-V (signature unification beats string/passage
similarity on VACS), H-K (keyword-coverage scores unattached hedges at least as high as gold, CCA-all at
most 0.05), H-C and H-F where run.

## 3. Validity gates (checked first, on dev data)

Fixed in code before any run (`metrics/validity.py`, `metrics/counterfactual.py`; hashed in the manifest).

* **V1** -- legacy containment PR: concatenation >= gold (a replication of the pilot); CCA-all strict: gold >=
  concatenation, unattached hedge <= 0.05, modality hedge <= 0.10, null <= 0.05.
* **V2** -- reader sensitivity: irrelevant flips change the reader's output in <= 5% of pairs, relevant flips in
  >= 90%; against the human entailment audit, determination precision >= 0.95 and recall >= 0.85.
* **V3** -- claim judge >= 0.80 on human-labelled claims; otherwise scoped precision is reported on the human
  subset only.

A failed gate is a result, not a reason to edit the criterion. If V2 fails the persona setting (deterministic,
judge-free) becomes the headline and the reader result is reported as a methods finding.

## 4. Narrowing rule

The pilot has **14** natural cross-document conditional conflicts (S1) and **4** same-guide (S2), not 18 in S1.
If S1 in the test split is below **60** the confirmatory family H1-H3 is reported as **exploratory** and the
title and abstract narrow to "exception suppression in RAG, with exploratory natural cross-document evidence".
S3 (real-text splits) never substitutes for S1 in any cross-document claim. The manifest records whether the
rule applies; it is computed from the data, not written by hand.

## 5. Systems and backbones

B0, B3, B7c, B8, O3. Primary backbone: `openai/gpt-oss-120b` (Groq free tier). Second, local backbone:
Qwen2.5-7B (Q4) via Ollama, which runs the full matrix and is the no-API reproduction path. Evaluation-only models
(never evaluated as systems): READER (local Llama-3.1-8B) and EVAL (Groq Qwen). Every tier is pinned; a rate limit
stops a cell and never substitutes another model.

## 6. Deviations from proposal v8, stated now

1. **B7 is split.** The released NormBench pipeline's prompt is about 18-27k tokens per call, above the free
   tier's 8k tokens-per-minute limit. **B7c**, a compact structured parse, is reported as "SG-DT-style" and is the
   comparator on both backbones; the released pipeline runs on the local backbone only, if at all.
2. **B3 receives gold conflict groups.** Its published detector is not reproduced; this makes B3's behaviour an
   upper bound for a selection-based resolver. It is labelled "ConflictRAG-style, oracle detection".
3. **S1 is far below the target of 80.** Expected, and handled by the narrowing rule above.
4. **No fine-tuned or distilled detector** (about 18 gold positives cannot train one), no external annotator,
   no coverage study, no scale study. The authors annotate; this is a stated limitation.
5. **The exact MDD figures** differ from the proposal's approximate table by at most one point in either direction.
6. **Mining and test construction use only the EVAL model.** QNFA, D3 and every backbone are excluded from mining,
   verification and pre-annotation of test data.

## 7. What the paper claims under each outcome

| Outcome | Claim |
|---|---|
| H1-H3 hold on S1 | Published selection designs cause measurable, metric-invisible cross-document harm; signature unification repairs detection; explicit scope checking beats the best structured prompting |
| H1 and H2 hold, H3 inconclusive | Harm is real and detection is fixable; the resolution advantage is not established on natural pairs. Stated plainly |
| H1 holds, H2 fails | Harm is real; recognising cross-document defeat is the open problem. A resource and evaluation paper |
| **H1 fails (B3 mostly presents both sources)** | A selection-based resolver with a sufficiency rule does not suppress most exceptions on these pairs; the harm is then concentrated in B0-style full-context generation and in the metrics that cannot see it. Reported with the no-specificity ablation (see `docs/v7_decisions.md` D3.1) |
| S1 below 60 | Every cross-document claim is exploratory; title and abstract narrow |
| A validity gate fails | The persona setting becomes the headline; the reader result is a methods finding |

## 8. Reporting rules

Every number carries its population and its model. Strata are never pooled in a headline. Run-to-run variance is
reported descriptively. Every pilot figure quoted carries the instances it was computed on. Nothing is reported
that the claims ledger cannot trace to a script and a cached output.
