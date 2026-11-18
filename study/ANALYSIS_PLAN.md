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

## Decisions taken on the tuning set (2026-11-14, before any evaluation run)

**Generator prompt.** Four wordings of the condition instruction were run with the `gold`
context on the tuning set (205 questions; hosted models on a fixed 60-question sample,
`study/probe.py`). F1c on conditional / unconditional questions:

| Prompt | Instruction for "conditions" | Qwen-7B | Llama-8B | Qwen3.8-27B (probe) |
|---|---|---|---|---|
| v1 | lines stating unresolved requirements (no evidence field) | 12.5 / 6.2 | 14.0 / 12.0 | - |
| v2 | "usually []", evidence field added | 7.6 / 42.1 | - | 12.7 / 82.2 |
| v3 | "list every one of them" | 8.2 / 16.9 | - | - |
| v4 | the benchmark's definition: a requirement that must be satisfied for the answer to be true and that the scenario does not mention | 9.8 / 31.0 | 12.6 / 55.1 | 13.8 / 62.2 |

Under the official metric a stated condition scores zero whenever the gold has none, and
77% of questions have none, so the wording that discourages conditions (v2) scores best
overall. Choosing by overall score would select a generator that cannot lose a condition
because it rarely states one, which biases every stage comparison towards no effect.
**v4 is frozen**: it states the benchmark's own definition of a condition and steers in
neither direction. All four tuning results are reported in the paper.

**Gate G2 is triggered.** `gold` F1c on conditional tuning questions is below 15 for both
local models (9.8 and 12.6). As specified, the generation-level stage comparisons (C1-C3,
C7) are also run on the hosted models over the hosted subsample, and the local results
are reported with the floor effect stated. Even Qwen3.8-27B recovers under 7% of gold
condition elements from gold context, so the generation stage is reported as the
dominant point of loss, with this caveat: the metric credits a condition only when the
exact element is cited.

**Added secondary metric (deterministic).** *Condition noticed*: the share of gold
condition elements present in the context that the generator cites anywhere (as evidence
or as a condition). It separates "the condition was in context and ignored" from "it was
used but not marked as a condition".

**Local matrix, trimmed.** `top5`, `top12`, `top20` (retrieval depth is analysed without
generation) and per-passage `grade` on ConditionalQA (nine calls per question) are dropped
from the local generation matrix. Everything in the confirmatory family is kept.

**Hosted cells.** Per hosted model: `gold` and `top8` and `qc` on the 300-question
subsample; `gold_nocondchk` on its 150 conditional questions.

## Deviations

1. **Hosted serving providers (2026-11-15, after 118 hosted answers, before any hosted
   analysis).** Groq's per-model daily cap made the hosted replication a two-week run. The
   two gpt-oss models are open weights served by other providers, so when Groq's quota for
   one is spent, the same model is called on NVIDIA's API (gpt-oss-20b) or Ollama's cloud
   (both), with identical settings (temperature 0, reasoning effort low, same prompt and
   token limit). Every answer records the provider in `served_by`; the paper reports the
   share served by each and the gold-context F1c by provider. Qwen 27B is served only by
   Groq and is unchanged. No system, question, metric or test changes.
2. **Hosted models (2026-11-15, same day, before any hosted analysis).** Ollama's cloud
   turned out to be paid-only for this account, leaving gpt-oss-120b and Qwen 27B on Groq
   alone (about twelve more days each). Both are replaced in the hosted replication by
   models NVIDIA serves: Gemma-4-31B (`google/gemma-4-31b-it`, dense) and Mistral-Large-4
   (`mistralai/mistral-large-4-1t-a52b`, mixture of experts, 52B active), with the frozen
   prompt, temperature 0 and no other change. The ladder becomes three model families
   instead of two. gpt-oss-120b had answered no hosted question; Qwen 27B had answered 9 of
   them, which are not analysed. Qwen 27B's tuning-set probe (Table 2) is unaffected. The
   hosted results were replication of direction before this change and remain so. NVIDIA
   requests go through a primary key and, if it is limited or failing, a standby key
   (`served_by` says `nvidia` or `nvidia2`); both reach the same hosted model.
3. **Third hosted rung (2026-11-15, before any hosted analysis).** Mistral-Large-4 reasons
   before answering, and on NVIDIA's endpoint its reasoning used up the 3,072-token limit with
   no answer on 55 of its first 86 questions; a low reasoning effort still used 2,959 tokens
   on one of them, reasoning cannot be switched off, and longer limits drop the connection. Its
   answers are not analysed. The rung is Nemotron-3-Ultra (`nvidia/nemotron-3-ultra-550b-a55b`,
   550B mixture of experts, 55B active) with its reasoning mode off, chosen before any of its
   answers were scored because it was the largest model the endpoint served that answered in
   seconds and returned a parseable answer on the two questions tried.
4. **Diagnostics added in revision (2026-11-15, after every confirmatory result was computed).**
   In answer to reviewers, four exploratory analyses were added; none changes or replaces a
   planned test, and all are reported as exploratory: (i) a condition-verification second pass
   over the frozen generator's answers (`study/revision.py`, cells `<system>_verify`); (ii) oracle
   completion (`qcoracle`); (iii) `top8` and `gold` with section-aware chunks end to end, with fixed
   70-word chunking as a size-matched separation control; (iv) every gpt-oss-20b gold-context
   question served by Groq asked again on NVIDIA's API. Also added, from existing outputs only:
   the decomposition of gold conditions in context (stated / evidence only / not cited), the
   completion candidate pool's ceiling, and the scorer's threshold sweep on the evaluation set.
   In the third review round, also from existing outputs: an audit of exact element matching
   (`report.metric_audit`), which credits a missed gold condition when the generator stated another
   element restating it (token F1 >= 0.5, and a loose bound at 1/3), on the gold-context cells.
   In the fourth round, also from existing outputs: the local models' C2 and E1 contrasts restricted to
   the hosted subsample's questions (`confirm.confirmatory`, key `on_hosted`), so that local and hosted
   generators are compared on identical populations.
   Two Qwen2.5-7B `page` answers aborted by the server as repetition loops are scored as
   unanswered, the same rule as for truncated output.
5. **Analyses added in the sixth review round (2026-11-18, after every earlier result was
   computed).** All exploratory and unadjusted; none changes or replaces a planned test.
   (i) *Prompt robustness* (`study/prompts.py`): with the gold context and with the condition-only
   chunks removed (C2's ablation), the hosted subsample is answered under five condition-aware
   prompting strategies besides the frozen v4 wording (direct condition extraction, explicit
   answer-condition pairing, chain-of-condition reasoning, scenario-condition applicability checks,
   structured evidence first), by Qwen2.5-7B, Llama-3.1-8B and Nemotron-3-Ultra. The five prompts
   reuse the frozen lines, worked examples and output contract, were written before any of them was
   run on evaluation questions, and were checked on 12 tuning questions for valid JSON only; none
   was tuned. (ii) *Context-preserving retrieval designs* (`study/baselines.py`): sentence window,
   parent-child, hierarchical (page then chunk) retrieval and query expansion with condition cue
   words, at the top 8's word budget, measured model-free and end to end with both local
   generators on the evaluation set. (iii) *Clean threshold protocol* for completion
   (`study.qualifier clean`, system `qcclean`): each fold's scorer is trained on the other four
   evaluation folds only and its threshold is chosen on the tuning questions under that same
   scorer, so no evaluation label shapes any threshold; the pre-registered C7 and C8 stay as run.
   (iv) *Semantic condition matching* (`study/semantic.py`): a missed gold condition is credited
   when a stated non-gold condition and it entail each other under the study's NLI model (and, as a
   lenient bound, when either entails the other).
