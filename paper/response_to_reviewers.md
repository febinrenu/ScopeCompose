# Response to reviewers

**Manuscript:** Where Do the Conditions Go? Stage-wise Attribution of Condition Loss in Chunked Retrieval-Augmented Generation

We thank both reviewers for careful and constructive reports. We have revised the manuscript throughout. Every number in the paper is still produced by script from the released model outputs (`study/report.py` → `paper/numbers.tex`); nothing was typed by hand. The analyses added in revision were not in the pre-registered plan. They are recorded as deviation 4 in `study/ANALYSIS_PLAN.md` and reported as exploratory.

Below, reviewer comments are summarised in bold and our responses follow. Section names refer to the revised manuscript.

---

## Reviewer 1

**1.1 The indicator in the definition of F1 with conditions renders as `|F[...]`.**
Fixed. It is now typeset as $\mathbf{1}[\hat C=\emptyset]$; the broken glyph came from a blackboard-bold font missing under our build engine.

**1.2 Inconsistent notation for condition-only chunks ($\mathcal{K}_C$, $\mathcal{H}_C$, $\mathcal{X}_C$) and broken set unions in Eqs. 1–2.**
Fixed. The notation is unified as $\mathcal{K}_C(q)$ throughout, and the unions are typeset as $\bigcup_i C_i$ and $\bigcup_i (C_i \cup E_i)$. As with 1.1, the stray symbols came from fonts the build engine lacked.

**1.3 "Qwen3.8-27B" looks like an unusual model name.**
The name is correct: it is the provider's model identifier (`qwen/qwen3.8-27b`), now given in the text. The model was used only in prompt tuning on the separate tuning fold, never in the evaluation, and the text now says so. Every evaluated model is now given with its exact provider identifier and published checkpoint name, including Gemma-4-31B (`google/gemma-4-31b-it` on the NVIDIA API, published as `google/gemma-4-31B-it`).

**1.4 Mixing Groq and NVIDIA for gpt-oss-20b may introduce hidden variation.**
We ran a direct check. Every gold-context question that Groq had answered (128) was asked again on NVIDIA with the same prompt and settings. The two providers returned the same answer set for 67.2% of the questions. F1 with conditions on the 123 conditional ones was 16.5 on Groq and 13.8 on NVIDIA, a difference of −2.7 points (95% CI −7.7 to +1.7; p = 0.25).

The section on larger generators now reports this check. It states that we find no systematic provider difference but cannot exclude one of a few points, and that the gpt-oss-20b figures carry this extra variance. Every answer in the released outputs records the provider that served it.

**1.5 RAMDocs is introduced abruptly, and its synthetic-misinformation caveat is buried.**
The RAMDocs paragraph in the method section now opens with its limits: synthetic misinformation, several valid answers for same-name entities, and no training split. It frames RAMDocs explicitly as a secondary diagnostic, not a primary evaluation. The abstract and conclusion use the same framing.

**1.6 Speculate on why the completion scorer fails to learn.**
We go further than speculation, with new analyses in "Why context completion does not help":
- **The candidate pool is not the problem.** 71.5% of the condition chunks that the top five miss lie on the pages completion searches.
- **The threshold is not the problem.** Choosing it on the evaluation questions themselves, the most favourable case, raises the trained scorer's F1 only to 0.09 (0.11 for the untrained reranker).
- **Even a perfect scorer changes little.** A new oracle variant adds exactly the missed condition chunks, raising the share of gold conditions in context from 72.7% to 89.1%. Yet F1 with conditions on conditional questions changes by only +0.5 points for Qwen2.5-7B (95% CI −0.8 to +1.7) and +2.4 for Llama-3.1-8B (+1.1 to +4.2). Nemotron-3-Ultra's condition recall does not move (15.6% and 15.9%). [Gemma-4-31B to be added.]
- **Three task properties explain the weak learning:** few positive pairs per fold, near-identical vocabulary between condition chunks and their neighbours, and the dependence of "is a condition" on the scenario.

---

## Reviewer 2

**Issue 1. The C2 effects in the text (+9.0, +9.5, +9.5) disagree with Table 5.**
We traced the discrepancy to its source. The numbers were not wrong, but the populations were not stated. The text's effects are paired differences on the 61 hosted conditional questions that have a condition-only chunk, the population on which C2 is defined. Table 5's columns averaged over all 150 conditional questions.

Every comparison was recomputed from the saved outputs. Table 5 now has separate rows for the C2 subset, with its own n, and every table caption states its population. Both the prose and the tables are generated from the same results file, so they cannot drift apart again.

**Issue 2. "Generation loss" conflates several failures.**
We added a decomposition (new subsection "Inside the generation loss" and Table 7). It uses the line-numbered output format to classify every gold condition in context as one of:
- stated as a condition;
- cited only as supporting evidence (a classification error);
- not cited at all.

We also report the share of stated conditions that are not gold conditions, and how many of those are lines that state the answer itself.

The dominant failure is non-citation. With gold context, the local generators leave 63.5% (Qwen2.5-7B) and 62.7% (Llama-3.1-8B) of the gold conditions in context uncited; even Nemotron-3-Ultra leaves 50.4%.

We also ran the reviewer's Experiment A in the form of a **condition-verification pass**. A second call to the same model receives its own answers and the same context and lists, for each answer, the lines that state an unresolved requirement. The result separates "noticing" from "discriminating":
- On conditional questions, condition recall roughly doubles at similar precision: 21.0% → 44.0% for Qwen2.5-7B and 18.2% → 33.0% for Llama-3.1-8B. F1 with conditions rises by +2.9 points (95% CI +0.8 to +4.7) and +3.5 (+0.7 to +6.3).
- On unconditional questions it attaches a condition to 77.8% and 96.8% of answers (39.0% and 26.5% before), and F1 with conditions falls by 28.1 and 49.8 points.
- 41.1% and 66.4% of the lines it cites there state the answer itself, which the instruction explicitly excludes.

The largest generator behaves the same way. With gold context, Nemotron-3-Ultra's recall rises from 32.1% to 58.4%, but its precision falls from 55.2% to 35.7%, so F1 with conditions changes by only +2.2 points (95% CI −1.0 to +5.9). Its unconditional answers given a condition rise from 17% to 70%, and their F1 with conditions falls by 38.1 points.

Inattention is therefore only part of the loss: the models find conditions when prompted, but cannot judge which ones the scenario leaves open. [Gemma-4-31B verification results to be added.]

**Issue 3. The official metric can reward stating no conditions; add supplementary metrics.**
We keep the official metric as primary and now report alongside it:
- condition precision and recall;
- exact match with conditions (joint answer-and-condition correctness);
- the share of unconditional answers given any condition (the false-condition rate);
- the share of stated conditions that are not gold (spurious conditions), and how many of those restate the answer.

The verification pass shows concretely why these are needed. On F1 with conditions over all questions its gain and its loss partly cancel, while the supplementary metrics show both.

**Issue 4. Generalisation beyond UK government guidance.**
We could not add a second independently annotated corpus within this revision. As the reviewer suggests for that case, we narrowed the claims throughout:
- The abstract states that all texts are English UK guidance.
- The findings in the introduction are qualified "on English UK guidance".
- The limitations section discusses how healthcare, financial and legal documents may structure conditions differently.

No claim of general applicability remains.

**Issue 5. Qualifier-aware completion is not convincing as a method.**
We agree, and reframed it as a pre-registered hypothesis with a negative result, not a proposed solution. Specifically:
- The method section states its page-local candidate ceiling.
- The completion-ceiling analysis separates candidate-generation recall (71.5% of missed condition chunks are in the pool) from ranking precision (threshold sweep).
- The class imbalance is discussed.
- The reviewer's oracle candidate selector is now included as `completion (oracle)`. Even with a perfect scorer, the end-to-end gain is at most a few points: +0.5 (not significant) for Qwen2.5-7B, +2.4 for Llama-3.1-8B, and nothing detectable for Nemotron-3-Ultra. Completion's ceiling is set by generation, which leaves most supplied conditions unstated. [Gemma-4-31B to be added.]

**Issue 6. Interpretation of the conflict-handling experiments.**
- (A) The ConflictRAG-style resolver is labelled a partial reproduction throughout. The method section lists the three criteria implemented (source authority, recency, specificity) and the two omitted (relevance, consistency, which the original extracts with a language model). It also explains how their omission could matter, and the paper no longer makes any claim about the original system.
- (B) The method section now states that an NLI contradiction judgement says only that two passages cannot both hold as stated: not which is authoritative, and not whether one qualifies the other.
- (C) RAMDocs conclusions are narrowed to the tested configuration (see 1.5).

**Issue 7. Incomplete baselines (retrieval and chunking).**
- **Retrieval:** BM25-only and Contriever-only retrieval are now reported beside the hybrid.
- **Chunking:** we added a fixed 70-word chunking, size-matched to the section-aware chunks, as the control the reviewer asks for in 5.7. With it, section-aware chunks separate the conditions of 68.5% of conditional questions against 86.1% for size-matched fixed chunks. Structure, not size, reduces separation.
- **Section-aware chunks end to end:** we also ran them through the pipeline, with both local generators. Condition-chunk recall@8 is lower with section-aware chunks (66.5% vs 74.5%), so better co-location does not come free. End to end the two effects nearly cancel. With the top 8, F1 with conditions on conditional questions changes by −0.3 points (95% CI −1.7 to +1.1) for Qwen2.5-7B and +1.6 (+0.1 to +3.1) for Llama-3.1-8B. With the gold chunks re-cut into sections, neither change is distinguishable from zero. Structure-aware chunking is therefore at best a small gain while generation leaves most supplied conditions unstated.
- **Not added:** parent-document and sentence-level retrieval. The size ladder (70, 100, 150, 300 words) and the section-aware variant span the granularity question, and adding both would have exceeded the journal's 25-page limit.

**5.1 Taxonomy of applicability conditions.**
The method section now describes the forms conditions take in ConditionalQA, with the benchmark's definition, and states that all forms are treated as one category, as the benchmark's annotation does. We report how often questions carry several conditions and several answers.

**5.2 Annotation quality.**
We now report:
- the benchmark's agreement methodology and human ceiling;
- the distribution of conditions per question (mean 2.58; 58.2% with two or more);
- the share of questions with several answers (49.6%).

We did not carry out a manual audit of 100 questions. The conclusions rest on the benchmark's published annotation, and the limitations section discusses how annotation noise bounds the condition recall we report.

**5.3 Generation protocol.**
The method section now gives the full protocol:
- decoding settings (temperature 0) and the output and input token limits;
- the JSON contract and how malformed or truncated outputs are scored (as unanswered, recorded as deviation 4);
- retries and the handling of multiple answers;
- chat templates, and that every system receives the same evidence ordering and line numbering.

The full prompts are in the repository (`study/reader.py`, `study/revision.py`).

**5.4 The model comparison is not a scaling experiment.**
Agreed. All language implying a parameter-scaling relationship has been removed. The larger-generators section states that the models differ in architecture, training and serving, so the comparison is not a controlled scaling result.

**5.5 More granular retrieval reporting.**
Beside chunk-level recall we now report, at k = 8:
- recall of all gold chunks;
- recall of all conditions of a question;
- the probability that a condition is lost given that the answer was retrieved;
- condition recall for BM25 and Contriever alone as well as for the hybrid.

**5.5 (second) Transfer corpus description.**
A new paragraph in the method section describes the corpus:
- its source (gov.uk and UK bank pages);
- its size;
- how refinement and non-conflict pairs are defined.

Each instance was labelled independently by two annotators following a written annotation manual. Their labels agree on every instance, so no adjudication was needed. The corpus is released with the code. The results call the check indicative only, given the small number of refinement pairs.

**5.6 Unanswerable questions.**
A new paragraph reports abstention on the 101 unanswerable questions under each evidence policy. The two local generators behave in opposite ways (Qwen2.5-7B abstains on most, Llama-3.1-8B on almost none), and no evidence policy changes this much.

**5.7 Size-matched chunking control.** Added; see Issue 7.

**5.8 Verifiable analysis plan.**
The repository is public at https://github.com/febinrenu/ScopeCompose. The analysis plan was committed in `b435681`, and the tuning decisions were frozen in `1ed2e83`, both before the first evaluation run; the commit timestamps are public. The repository also contains:
- the splits, model configurations and complete prompts;
- every model output, and the completion scorer's scores;
- the statistical scripts and the deviations log.

The cross-fitting and threshold protocol is now described step by step in the method section. One point the reviewer raises deserves an explicit answer. The model that scores the tuning questions was trained on evaluation labels, but it never scores an evaluation question; it only sets the threshold. The threshold sweep then shows that no threshold, including one chosen on the evaluation questions, makes the scorer accurate, so the threshold choice cannot explain the result.

**Literature (6.1–6.4).**
- The reranker is now identified by its exact checkpoint, `BAAI/bge-reranker-base`, alongside the BGE technical report.
- The comparison with Chain of Condition is expanded. Chain of Condition improves how a reader handles conditions it is given, mainly with gold passages. This paper asks whether the pipeline delivers them at all, and where it loses them.
- The novelty claims are stated relative to the work they differ from.

**Writing and presentation (Section 7).**
- Section 3 is retitled "Methodology and Experimental Design".
- The abstract now gives the evaluation scale, the models and the main limitations.
- Claims are softened as suggested: "The lever is the generator, not the context" is gone. The conclusion now states that conditions are lost at every stage, that generation is the largest loss in the configurations tested, and that losing an isolated condition chunk costs every generator; it limits its claims to the tested setting.
- Every results table states its population in its caption; the figures are numbered floats.

**Placeholders and publication readiness (Section 9).**
- **Resolved:** the repository URL, analysis-plan commits and all result placeholders.
- **Supplied by the authors at submission:** author names, affiliations, emails, biographies, and confirmation of the funding and conflict-of-interest statements.
- **AI disclosure:** retained, as the journal's policy requires.

---

## Second round (Reviewer 2, updated report)

We thank the reviewer for the second reading, and for withdrawing the numerical-inconsistency point after checking the C† populations. The three experiments the reviewer flagged as outstanding are now complete, for every generator they were run on. The target journal is the *Journal of Trends in Computer Science and Smart Technology* (IRO Journals), which limits papers to 25 pages and 25 references. Some responses below are shaped by those limits.

**Outstanding experiments.**
- **Verification pass:** results for Qwen2.5-7B, Llama-3.1-8B and Nemotron-3-Ultra are in "Inside the generation loss". [Gemma-4-31B to be added.]
- **Section-aware chunks end to end:** results for both local generators are in the same subsection.
- **Oracle completion:** results for Qwen2.5-7B, Llama-3.1-8B and Nemotron-3-Ultra are in "Why context completion does not help". [Gemma-4-31B to be added.]

None of the three changes the paper's conclusions, and each sharpens one:
- A dedicated pass finds more conditions but cannot tell when none applies.
- A perfect completion scorer gains at most a few points.
- Structure-aware chunking is at best a small gain end to end.

The abstract, the introduction's fourth finding and the conclusion now state these results.

**Issue 1. Stage contributions are not directly comparable.**
Agreed. The conclusion now says that generation shows the largest *observed* deficit under gold evidence, while chunking and retrieval add losses upstream. It states explicitly that the stages are measured on different scales (separation, recall, deletion, statement), so the ranking is of observed deficits, not an additive decomposition of the end-to-end loss. The abstract no longer says generation "accounts for" the loss.

**Issue 2. Separate answer correctness, condition correctness, spurious and omitted conditions.**
Table 7 now gives, for every generator with gold context:
- conditions stated, cited only as evidence, or omitted;
- spurious conditions, and the share of those that restate the answer;
- answer F1 ignoring conditions;
- exact match with conditions (joint correctness);
- the share of unconditional answers that name any condition (false conditions).

All of these come from the existing outputs. They show that the generators get much of the answer right while the conditions are lost.

On exact element matching: our generators cite conditions by line number, so paraphrase can never be penalised. Exact matching can still penalise citing a different element that restates the same requirement. The metrics section now says so, and that the reported condition recall is therefore a lower bound on what a generator recognised.

On 2.1 (understood but omitted, versus never recognised): we agree that line citations cannot reveal internal understanding. The verification pass is the closest test we can make. Asked directly, the generators recognise many more of the conditions, which suggests that part of the omission is not a failure to recognise.

**Issue 3. The transfer corpus.**
The method section now gives:
- the numbers of positive and negative pairs (18 refinement and 56 non-conflict natural pairs, from 84 pages);
- the scoring protocol: each pair is scored in both directions by the scorer that sets the threshold, and the higher score counts;
- that the written probe pairs predate this study, and that the scorer was never tuned on any part of the corpus.

The results now report a Hanley–McNeil 95% interval for each AUC: 0.36 (0.22 to 0.50) on natural pairs and 0.25 (0.10 to 0.40) on probe pairs. We accordingly say the scorer does *not rank refinement pairs above* non-conflict pairs, rather than that it ranks them below. The check remains an exploratory diagnostic, as the reviewer suggests.

**Issue 4. RAMDocs conclusions.**
The conclusion now uses the reviewer's formulation. Across the two regimes, the tested operators do not reliably select evidence by its role in making an answer correct. The different construction and metrics of ConditionalQA and RAMDocs rule out comparing their scores directly.

**Issue 5. Operator implementation details.**
The method section now states the consequential parameters:
- **Reranker:** checkpoint and revision; it keeps the best 3.
- **NLI filter:** checkpoint and revision. It scores every pair of the eight passages in both directions with a contradiction-probability threshold of 0.5, and always drops the lower-ranked passage, so ties are resolved by retrieval rank and the top passage always survives.
- **ConflictRAG-style resolver:** resolves the same NLI-detected conflicts by credibility.
- **Relevance grader:** its criterion is stated, and it keeps the top passage if none is judged useful.
- **Failures:** an unparseable grade or selection keeps the passage.
- **Budget:** every operator works within the same eight-chunk budget and can only remove.

The resolver remains labelled a partial reproduction, and no claim is made about the full ConflictRAG method.

**Issue 6. Provider equivalence for gpt-oss-20b.**
The provider check is reported in full:
- **Parsing:** both providers returned valid JSON for all 128 questions, with no truncated or degenerate output.
- **Unanswered:** 7 on Groq and 4 on NVIDIA.
- **Conditions stated:** 151 and 128.
- **Same answer set:** 67.2% of questions.
- **Paired difference** in F1 with conditions: −2.7 points (95% CI −7.7 to +1.7).

We pool the two providers with this qualification stated in the paper: no systematic difference is detectable, one of a few points cannot be excluded, and the gpt-oss-20b figures carry this extra variance.

**Issue 7. Statistical reporting.**
- Every effect is given as a point estimate with its 95% interval, in the text or in the table beside it (Tables 4 and 5); p-values accompany intervals, never replace them.
- Exploratory analyses (the hosted comparisons and every diagnostic added in revision) are labelled as such where they are reported and in the limitations.
- The larger-generators section now says that with 150 conditional questions (61 for C2) the intervals are wide, and that these comparisons are read for direction only.

**Issue 8. Reranker and NLI citations.**
The journal's 25-reference limit is reached, so we identify both models unambiguously in the text:
- reranker: Hugging Face checkpoint `BAAI/bge-reranker-base`, revision `2cfc18c`;
- NLI model: `cross-encoder/nli-deberta-v3-base`, revision `6c749ce`.

The C-Pack reference is kept only as background for the BGE family, which is what it supports. The NLI model is a DeBERTa-v3 cross-encoder and keeps the DeBERTa-v3 citation.

**Issue 9. Reproducibility audit.**
- **Truncation:** of the 39,106 local generations, only 23 had an input long enough to leave less than the output budget of the 8,192-token window. The paper reports this, so input truncation is negligible; the hosted models' windows are far larger than any input.
- **Provider:** every answer records the provider that served it.
- **Scripts:** the verification pass and oracle completion have their own scripts (`study/revision.py`, and the `qcoracle` system in `study/qualifier.py`).
- **Regeneration:** every table and macro is regenerated from the raw outputs by `python -m study.report`.

**Presentation.**
- **Table 5:** the text now points explicitly to its last two rows for the C† comparison, and notes that differences are computed from unrounded scores, so they can differ slightly from differences of the rounded entries (e.g. 9.6 versus 9.5).
- **Figure 2:** set with larger labels and values.
- **Limitations:** split into experimental limitations and external validity.
- **Title:** retained, as the reviewer recommends.
- **Condition types (2.2):** ConditionalQA does not label condition types, and a reliable breakdown would need new annotation. We describe the forms conditions take and state that the analysis does not distinguish them.
