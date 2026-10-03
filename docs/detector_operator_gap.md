# The detector–operator gap

**Status: measured. This changes what the paper argues.**
**Date: 2026-10-03**

Building a browser demonstration of the pipeline meant running every stage
over the whole annotated corpus rather than over the twelve instances the
decisive experiment used. The results are not what the project expected, and
they are more useful than what it expected.

> **The composition operator preserves 97.6% of gold branches when it is told
> a conflict exists. The detector in front of it finds 0 of 42. The gap
> between those two numbers is the remaining research problem.**

This project was designed around the hypothesis that composing branches
without mangling them would be the hard part. It is not. Detection is.

---

## 1. The detector finds almost nothing

`Stage1Filter` with `heuristic_nli=True` and no trained head — the
configuration `run_decisive.py` actually uses — over all 128 instances:

| | |
|---|---:|
| precision | **0.833** |
| recall | **0.064** |
| F1 | 0.119 |
| conditional conflicts recovered | **2 / 67 (3.0%)** |
| TP / FP / FN / TN | 5 / 1 / 73 / 49 |

On the **42 branch-bearing instances** — the ones every branch-scored metric
is computed over — it fires **0 times**, with a maximum score of **0.2206**
against an escalation band of `[0.35, 0.75]`.

**Two consequences worth stating separately.**

Stage 2 is unreachable. Escalation requires a score inside the band, and
nothing on the annotated set comes within 0.13 of the floor. Supplying an API
key changes nothing: the LLM judge never runs, because nothing is ever
referred to it.

It is not random. Precision 0.833 says that when it fires it is usually
right; recall 0.064 says it almost never fires. That is a specific, fixable
failure, not noise — and the mechanism is identifiable:

```
score *= 0.35 + 0.65 * min(1.0, token_jaccard * 3.0)
```

The topicality multiplier damps any pair whose passages are worded
differently. A general rule and its carve-out are written by different teams
for different audiences and share little vocabulary, so the gate suppresses
exactly the class the project exists to detect.

`detection/stage1.py` already says `_rule_score` is "a scaffold rather than a
model". This quantifies it on real data.

## 2. The operator is not the problem

Given gold routing and gold branches, `CompositionOperator` → `render_template`
over the same 42 instances:

| | |
|---|---:|
| Preservation Rate | **0.9759** (81 / 83 branches) |
| Suppression Rate | 0.0241 |
| Distortion Rate | 0.0000 |
| resolutions | composed 37, merged 2, selected 3 |

**This is an upper bound, not a result.** Both inputs are the answer key.
B1 (`extraction/probing.py`) has no offline path, so there is no way to feed
the operator branches it derived itself. What the number establishes is that
the operator is not the bottleneck: the ceiling above it is high, and the
system is nowhere near it.

## 3. Preservation Rate is gameable by verbosity

Scoring four resolvers with the project's own textual judging path:

| resolver | PR | mean answer length |
|---|---:|---:|
| selection (`selection_answer`) | 0.542 | 69 |
| ScopeCompose, as it runs (routing-level) | 0.446 | — |
| **full-context concatenation** | **0.855** | 184 |
| ScopeCompose, routing oracle | 0.976 | 142 |
| `gold_scoped_answer` | 0.699 | 163 |

**Concatenating both passages verbatim scores above the human-written gold
answer.** PR is `containment(branch.outcome, answer) >= 0.6` with no length
normalisation, and the suite has no precision-side counterweight: HCR catches
only invented *figures*, and SCR applies only to `no_conflict` instances, of
which the annotated 42 have none.

So the metric rewards saying more, without limit, and the ceiling it implies
(0.699, from the gold answer) is beaten by the least intelligent possible
resolver.

**The suite needs a precision term** — a penalty for content not traceable to
any gold branch, or a length-normalised variant of PR. Until it has one, PR
alone cannot adjudicate between a composed answer and a concatenated one,
which is precisely the comparison the paper wants to make.

This is a defect in our own instrument, found by our own instrument. It is
better reported than discovered.

## 4. Three of the pipeline arm's five metrics cannot be non-zero

`run_decisive.score_pipeline_routing` decides survival at the routing step:

```python
survives = branch.is_default or composed
```

It increments `preserved` or `suppressed` and nothing else. `score.distorted`
and `score.hallucinated` are never assigned anywhere in that function, and
`no_conflict_instances` is 0 so SCR divides an empty denominator.

So the pipeline arm's distortion 0.000, HCR 0.000 and SCR 0.000 in
`decisive.json` are properties of the scoring code, not of the pipeline. The
baseline arm's 0.333 distortion comes from a real generated answer put
through `align()`. **The two arms cannot be compared on those three metrics**,
and the earlier reading of "the pipeline preserves fewer branches but does not
corrupt the ones it keeps" does not survive contact with the code.

They should be reported as `n/a — not measurable at the routing step`.

## 5. Retrieval destroys the branch before any resolver runs

The three implemented baselines in `baselines/retrieval_side.py`, over 42
instances and 83 gold branches. A branch is lost when its supporting passage
is dropped:

| system | branches lost | instances affected |
|---|---:|---:|
| `standard_rag` (control) | 0 / 83 | 0 / 42 |
| **`rerank_top1`** — selection | **41 / 83 (49.4%)** | **40 / 42** |
| `nli_filter` | 5 / 83 (6.0%) | 5 / 42 |

`rerank_top1` keeping only the top-ranked passage must drop branches supported
by the second one, so the mechanism is near-tautological and the paper should
say so. The result is the magnitude: **roughly half of every gold branch in
the benchmark, in all but two instances.**

`nli_filter` is the row that should worry a practitioner. A
contradiction-aware filter — what a careful engineer would actually build to
avoid this problem — still destroys five branches.

## 6. The decisive run used what is now test data

`decisive.json` recorded five rates and nothing else: no instance ids, no data
path, no seed, no commit, no per-branch outcomes. The subset had to be
recovered from git history and the response cache. It is
`wp1_probe.jsonl --limit 12`, i.e. `wp1_fin_imp_001`–`010`,
`wp1_imm_imp_011`–`012`.

Those twelve were selected by file order, before splits existed. Under the
assignment now in `benchmark/data/corpus.jsonl` they straddle all three:
**5 test, 4 train, 3 dev.**

The headline experiment therefore corresponds to no defined split and is
majority held-out data. The re-run at 137+ instances must be on a defined
split, and this run should be described as exploratory.

`decisive.json` now carries full provenance and all 24 per-branch outcomes.

## 7. A dead feature, now alive

`detection/features.py` defined `_YEAR_RE = r"\b(19|20)\d{2}\b"` with a
**capturing** group. `re.findall` returns the group rather than the match, so
every year collapsed to `"19"` or `"20"`: 2019 and 2024 compared equal, and
`year_clash` could only fire across the century boundary. No real policy pair
crosses it.

The feature was therefore contributing its 0.10 weight to nothing on all 128
instances. Fixed to a non-capturing group, with a regression test. The effect
is small and real — one more true positive, precision 0.800 → 0.833, recall
0.051 → 0.064.

---

## What this means for the paper

The contribution as originally framed — a composition operator that preserves
branches selection would delete — is **supported at the operator level and
unsupported end to end**, and the reason is now precisely located.

Three honest framings are available, and they are not mutually exclusive:

1. **The benchmark and the metric suite are the contribution.** They are what
   found this. Report the operator ceiling, the detector floor, and the gap,
   and let the gap be the stated open problem.
2. **Detection of conditional conflict is the harder sub-problem, and this is
   the first measurement of how much harder.** 3% recall from a lexical
   scorer, with an identified mechanism (the topicality gate) and an obvious
   next step (train the head, or route on scope-attribute extraction rather
   than surface overlap).
3. **The metric suite needs a precision term before it can adjudicate
   anything**, and §3 above is the evidence.

What is *not* available is the original claim, measured end to end, on this
detector.

## Reproducing

```bash
python -m experiments.run_decisive --data benchmark/data/wp1_probe.jsonl \
    --limit 12 --pace 0 --seed 0 --json benchmark/data/decisive.json
python -m scope.order_invariance --data benchmark/data/corpus.jsonl \
    --sample 128 --no-llm --heuristic-nli
python site/tools/make_reference.py && node site/tools/verify_engines.js
```

Every figure in §1–§5 is also computed live, in the browser, at
`site/` → `#/bench`, by ports verified against the Python above.
