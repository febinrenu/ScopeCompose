# Documents

## The study

The documents for the paper's study live next to the code they describe:

| Document | What it holds |
|---|---|
| [`../README.md`](../README.md) | the study, its findings, the repository layout, and how to reproduce every result and build the paper |
| [`../study/ANALYSIS_PLAN.md`](../study/ANALYSIS_PLAN.md) | the analysis plan, committed before the evaluation, with dated deviations |
| [`../paper/response_to_reviewers.md`](../paper/response_to_reviewers.md) | seven rounds of review and the response to each point |
| [`../paper/references_verified.md`](../paper/references_verified.md) | every reference checked against its primary source, and the claim it may support |
| [`../PROGRESS.md`](../PROGRESS.md) | the dated history of the work |

## This folder: the earlier ScopeCompose design

The files here belong to the project's earlier design, *Beyond Source Selection*. That design built a pipeline
to detect and compose conditional knowledge conflicts, and it was set aside on 2026-11-14 (README,
Section 12). They are kept as the record of that work and are not part of the paper's study.

| Document | Content |
|---|---|
| `wp0_positioning_memo.md` | prior-art review and positioning of the earlier design |
| `wp1_tier1_findings.md` | mining pilot: how rare natural cross-document rule/exception pairs are |
| `wp1_probe_verification.md` | human verification of the 54 written probe cases |
| `kappa_pilots.md` | inter-annotator agreement pilots for the earlier annotation scheme |
| `detector_operator_gap.md` | measured gap between the conflict detector and the composition operator |
| `v7_decisions.md` | append-only decision log of the v7/v8 design |
| `journal_readiness_audit.md` | an audit of the earlier design against reviewer expectations |
| `datasheet_generated.md` | generated datasheet of the earlier 128-instance corpus |

The corpus documented by these files (`benchmark/data/corpus.jsonl`) is the one the study uses for its
out-of-distribution check of the completion scorer (`study/ood.py`).

Related legacy material elsewhere:
- `plan_v7/`: earlier proposals and review rounds;
- `benchmark/manual/`: the annotation manual;
- `review.txt`: the probe-verification sheet.
