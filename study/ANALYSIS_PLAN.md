# Analysis plan: condition loss in chunked RAG

Written 2026-11-14, before any system was run on the evaluation questions.
Committed to git before the first evaluation cell; the commit hash is the
record of what was decided in advance. Anything below that later changes is
listed under *Deviations*, with the reason, and the paper reports it.

## Question sets (fixed: `study/splits.json`, seed 0)

* **Tuning**: every question on 40 randomly drawn ConditionalQA training pages
  (209 questions; 205 after removing the four used as worked examples in the
  prompt). Prompts, thresholds and operator settings are chosen here only.
* **Evaluation**: from all other pages (train and dev splits of ConditionalQA
  v1.0; the test split's answers are not public): all 560 conditional
  questions, all 101 unanswerable questions, and 400 unconditional questions
  sampled at random. 1,061 questions on 416 pages.
* **Contrast**: all 500 RAMDocs questions.

## Fixed pipeline settings

* Chunking: whole HTML elements packed greedily to 150 words, page title and
  section heading prefixed for retrieval (`study/cqa.py`).
* Retrieval: BM25 + Contriever, reciprocal-rank fusion, over all 6,365
  chunks of all 652 pages (`study/retrieve.py`). Default context: top 8.
* Generator: one prompt, one JSON output contract (answer, evidence lines,
  condition lines), four worked examples from the tuning set, temperature 0
  (`study/reader.py`). The prompt hash is recorded in every cell file.

## Metrics

* **Primary: F1 with conditions** (`F1c`), per question, computed by
  ConditionalQA's official `evaluate.py`, unchanged.
* Secondary: answer F1 and EM (conditions ignored), EM with conditions,
  element-level condition precision and recall, share of answers citing any
  condition, tokens and model calls per question.
* RAMDocs: strict accuracy (every gold answer present, no wrong answer
  present, after SQuAD normalisation), gold-answer recall, wrong-answer rate.

## Models

Full matrix: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, local, Q4_K_M,
temperature 0, seed 0. Replication on a stratified subsample (150 conditional,
100 unconditional, 50 unanswerable, drawn with seed 0 from the evaluation set)
on gpt-oss-20b, Qwen 27B and gpt-oss-120b via Groq. Hosted results are reported
as replication of direction, not as confirmatory tests.

## Inference

* Unit of resampling: the **page** (document). Questions on one page share
  chunks and retrieval behaviour.
* Paired differences: mean difference with a BCa bootstrap interval over
  pages (2,000 resamples) and a page-level sign-flip permutation p-value
  (20,000 flips).
* Family-wise error: Holm over the confirmatory family below, separately per
  model, at alpha = 0.05.

## Confirmatory family (per local model, evaluation set)

| ID | Comparison | Population | Direction tested |
|----|------------|------------|------------------|
| C1 | `gold` vs `gold_nocondel` (condition elements deleted) | conditional | two-sided; validity check that F1c measures condition use |
| C2 | `gold` vs `gold_nocondchk` (condition-only chunks removed) | conditional questions with at least one condition-only chunk | two-sided |
| C3 | `goldpad8` vs `top8` (the retrieval gap at equal budget) | conditional | two-sided |
| C4 | `rr8to3` vs `top8` | conditional | two-sided |
| C5 | `nli` vs `top8` | conditional | two-sided |
| C6 | `crs` vs `top8` | conditional | two-sided |
| C7 | `qc` vs `top8` | conditional | two-sided |
| C8 | `qc` vs `top8` | unconditional | non-inferiority, margin 2.0 F1c points: the lower 95% bound of (qc - top8) must exceed -0.02 |

C8 is not in the Holm family (it is a one-sided bound, not a p-value).

## Exploratory (reported, not tested for confirmation)

* Retrieval: answer-chunk vs condition-chunk recall at k = 3..50, three
  rankers, three chunk sizes, two chunking strategies.
* `grade` (per-passage relevance grading) and `select` (listwise selection),
  local models only.
* `adj` (adjacent chunks), `page` (whole page), `qc0` (untrained scorer),
  `top3/5/12/20`.
* The 88 evaluation questions whose gold contains both "yes" and "no": the
  rule-and-exception subset.
* RAMDocs: every reducer vs full context; oracle discard; the operator-by-
  regime contrast (change on ConditionalQA conditional vs change on RAMDocs
  questions with misinformation).
* Error taxonomy of condition failures, computed from cell files: condition
  omitted, wrong condition cited, condition cited where gold has none, answer
  wrong.
* Out-of-distribution: qualifier scores on the project's own 128-instance
  corpus (refinement vs no-conflict pairs).

## Gates

* G2: if `gold` F1c on conditional questions is below 15 for both local
  models, the stage-attribution comparisons move to gpt-oss-120b on the
  hosted subsample and the local results are reported as a floor effect.
* Remedy claims are made only from C7/C8; if C7 is not significant, the paper
  reports that and does not claim the remedy works.

## Deviations

None yet.
