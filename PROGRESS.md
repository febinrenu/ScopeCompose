# Progress log

Session-by-session record of what was actually done, newest first. `README.md`
is the plan; this file is the history.

**Append an entry before ending any working session.** The reasoning behind a
decision is what gets lost over 22 weeks, not the code — so write down *why*,
not just *what*. The entry format is in `CONVENTIONS.md`.

---

## Sessions

### 2026-09-25 (later) - Corpus composition settled. Contract v1.2.0.

Task 3 closed. The WP1 gating question now has a decision attached, and the
decision is encoded rather than left in prose.

**The target: 300 instances**

| row | target | role |
|---|---|---|
| natural cross-document | 60 (20%) | evidence for the natural multi-document setting |
| synthetic split | 180 (60%) | main controlled benchmark |
| same-guide retrieval split | 60 (20%) | real retrieval separation within one document |

The 150-instance Tier-1 requirement is **dropped**. At ~0.08 plausible pairs
per curated pairing it would need ~1,900 pairings for no proportional
scientific return. Tier 1 is now evidence for the natural setting, not the main
corpus.

**Done**
- `benchmark/corpus_plan.py` - the target as executable code, with progress
  reporting: `python -m benchmark.corpus_plan --data <corpus.jsonl>`.
- `contract.models.Separation` - new optional enum
  (`cross_document | synthetic_split | same_guide`) on both `QueryRecord` and
  `GoldInstance`.
- `metrics.classification.split_for_reporting()` - the three-way split the
  paper reports. `split_by_construction()` keeps the old two-way behaviour.
- Probe set and mock generator both now label their separation.
- 13 tests added (460 -> 473). Gate still PASS at 1.0000.

**Decisions**

*Same-guide needed a field, not just an intention.* The agreed plan was "store
as Tier 2, report separately" to avoid schema risk. Storing it was free;
**reporting it separately was not** -- `split_by_construction` partitions on
`construction`, so a same-guide instance stored as `split` would have been
silently folded into Tier 2 by every table, and the distinction would have
existed only in prose. The additive optional `separation` field is the
mechanism. `Construction` stays a two-way tag, so nothing that routes or
validates on it changes.

*Contract v1.1.0 -> v1.2.0.* Additive, optional, backwards compatible -- every
existing instance carries `separation=None` and still validates. Schema
regenerated; mock updated in the same commit per the freeze rule.

*Unlabelled instances fall back to synthetic split, not Tier 1.* An unlabelled
instance promoted to Tier 1 would inflate the claim that matters most, so the
fallback goes the conservative way and the count of unlabelled instances is
reported rather than hidden.

*Tier 2 is stratified, not counted.* `TIER2_STRATA` sets floors across all four
relations, both explicitness values, seven condition devices, and distractors.
A test demonstrates the failure mode directly: 180 instances that are all
refinements meet the count target and still leave the plan incomplete, because
the refinement/opposed boundary the paper reports would be untested.

*The pilot yield cannot become a prevalence rate.* 38 **curated** pairings
across two domains supports a design decision, not a population claim.
`prevalence_estimate()` exists only to raise, with the two sentences -- the one
that is supportable and the one that is not -- written into the error. Cheaper
than remembering.

*LLM proposer is the primary discovery mechanism for WP2.* On the banking
corpus the lexical miner found 1 candidate against the proposer's 4, of which 3
held up. `tier1_pilot.py` now prints this at the end of every run, so the
decision is visible where it applies rather than only in a document.

**Next up**
- Remaining human tasks: second annotator (kappa pilot), SG-DT sections 3-4,
  closed-venue literature sweep.
- Open from the probe verification: `OPPOSED` has one instance, thin coverage
  of the relation hardest to tell from refinement.

---

### 2026-09-25 - WP1 probe set human-verified. Gating item closed.

All 54 cases reviewed case by case by a human. **50 accepted as proposed, 3
changed, 1 dropped.** The set is 53 instances and is usable as gold.

This closes the last gating item that was blocking Member B's feasibility probe.

**Changed**
- `wp1_imm_imp_016` - re-encoded from two branches to **three**: two years
  generally, five with settled status, four for Swiss nationals holding settled
  status. The `Case` schema gained `sub_condition` / `sub_outcome` /
  `sub_applicability` / `sub_attrs` to express a branch nested inside an
  exception. The instance now carries `gold_multi_exception_flags.nested=True`
  and B2 routes it to **selection with the flag**, which is the correct
  first-order behaviour -- exception-to-exception precedence is out of scope and
  is flagged, not guessed.
- `wp1_imm_imp_026` - refinement -> **disjoint**. The text never establishes
  that someone who has lived in the UK for six months is a subset of those
  resident in a listed country, and refinement requires that nesting. Re-encoded
  with both branches separately scoped and no default, per v1.1.0.
- `wp1_imm_opp_054` - **dropped**. p0 concerns "the skill requirement", p1 "the
  requirements of this route" -- different propositions, so a 46-year-old
  graduate satisfies both and there was never a contradiction. Dropped rather
  than reworded: changing p0 would have produced a valid case, but a different
  one from the case reviewed, and editing evidence to fit a label is the wrong
  habit to build into a benchmark.

**A fourth matcher bug, exposed by the 016 re-encoding**
Encoding three branches immediately disagreed with B2, which read *"the period
is five years"* and *"the period is four years"* as the **same outcome**.
`numbers_conflict` extracted digits only, and policy text spells small
quantities out constantly. The two strings share "period" and "years"; the only
tokens that differed carried the whole meaning.

Now handles spelled cardinals including compounds, so "twenty-eight" and "28"
resolve to one figure rather than conflicting. Gold and B2 now agree on `016`.

This is the third bug of the same shape after negation-as-stopword and
passage-ids-as-figures: **two outcomes differing in one small token that carries
all the meaning.** Worth watching for a fourth.

**Provenance and the kappa guard**
Every instance now carries `annotator_a="model-proposed"` and
`annotator_b="human-verified-2026-09-25"`. Both are kept: the first records that
a model drafted the case, which is exactly why this set can never enter a
Cohen's kappa -- the two passes share a starting point, so their agreement would
measure the proposal rather than the task.

`MODEL_ANNOTATORS` was missing the hyphenated spelling `"model-proposed"`, which
is the exact string this repo writes. The guard would have let the project's own
model-proposed labels straight through. Fixed, and a test now pins the refusal
against that literal string.

`test_probe_instances_are_marked_unverified` failed, as designed -- its docstring
said that if it ever failed with a real annotator name, the set had been verified
and the assertion should be changed deliberately. Replaced with a test that
checks both fields are present, plus the new kappa-refusal test.

**Open limitation**
`OPPOSED` now has **one** instance (`wp1_fin_opp_053`). Thin coverage of the
relation hardest to tell from refinement, which is the confusion cell the paper
reports. Worth authoring a replacement before the probe runs; it can be drafted
and verified the same way these were.

**Next up**
- Remaining human tasks are unchanged: second annotator for the kappa pilot,
  supervisor sign-off on the Tier 1/Tier 2 split, SG-DT sections 3-4, and the
  closed-venue literature sweep.

---

### 2026-09-24 (later still) - Member B built: B1, B2, B3 and the preservation metrics. Four scoring bugs found on the way.

The pipeline now runs end to end. Building B exercised the shared scoring code
against real structures for the first time, and it broke in four places -- every
one of them silently, and three of them in the direction of a better-looking
number.

**Done**
- `extraction/probing.py` (B1) - Contrastive Scope Probing plus the grounding
  gate. Contrastive questions rather than open ones, because a question with a
  presupposition is harder to answer vacuously. The gate scores
  `passage |= (condition AND outcome)` locally and its rejection rate is
  reported: a gate that rejects nothing is not a gate. Includes the cue-matching
  fallback as the baseline the probe must beat.
- `composition/operator.py` (B2) - the composition operator. Routes on A4's
  relation, merges restatements rather than branching them, and flags `nested`
  and `crossed` separately instead of composing second-order structure.
- `generation/scoped_answer.py` (B3) - scoped answers with per-branch
  attribution. Default renderer uses **no model at all**; `check_faithfulness()`
  verifies afterwards that every branch reached the answer, that no figure was
  invented, and that the generator did not rank sources.
- `metrics/preservation.py` - PR / SR / HCR / SCR implemented against the
  frozen signatures, broken out by tier. The LLM-judge path raises rather than
  silently working, because proposal 6.1 requires validation against human
  labels first.
- 36 tests added (423 -> 459). Order-invariance gate still PASS at 1.0000.

**Four bugs in the shared scoring code**

1. **Negation was a stopword.** `"no"` and `"not"` were in `_STOP`, so
   `"no fee applies"` and `"a fee applies"` scored a Jaccard of **1.00** --
   exact opposites judged identical. A system emitting the reverse of a gold
   branch was scored as preserving it. This sits in `metrics/branch_match.py`,
   which the decisive experiment scores on. Fixed by removing them from the
   stoplist and adding `polarity_conflict()` as a hard veto beside
   `numbers_conflict()`.
2. **Two `BranchJudgement` enums.** `align()` returned `branch_match`'s and the
   scorer compared against `preservation`'s, so every `is` check was False and a
   perfectly preserved run scored **PR = 0.0** with all branches counted
   distorted. Now one enum, imported.
3. **Passage ids read as invented figures.** `(per p0)` matched the number
   regex, so every correctly-attributed answer was flagged for inventing a
   figure -- including the template renderer's, which cannot invent anything.
   That the deterministic path failed its own check is what exposed it.
4. **Decimals split as clause boundaries.** `[;.]` tore `"a 2.75% fee applies"`
   into `"a 2"` and `"75% fee applies"`, so any answer quoting a decimal was
   reported as dropping its own branches.

Plus a false positive from the fix to (1): `"no more than 20 hours"` is a
quantifier bound, not a negation, and it made a permission read as a
prohibition. Comparative quantifier phrases are now stripped before polarity is
counted.

**Decisions**
- *`PASS_THROUGH` and the flagged path keep only the branches the answer
  actually states.* Both previously carried every branch through while emitting
  an answer that stated one or none. On `PASS_THROUGH` that mattered most: a
  missed conditional conflict is exactly the suppression the project measures,
  and crediting it with full preservation hid the detector's own failures behind
  the scorer. End-to-end PR on mock fell from an inflated **0.97** to **0.58**
  against **0.73** with oracle routing -- the 0.15 gap is the detector's real
  contribution.
- *Generation defaults to the template renderer, no model.* It cannot drop a
  branch or invent a figure. The model earns its place only where fluency is
  being measured, and the two are reported as separate systems.
- *The LLM judge is unreachable by default.* Passing a client raises. An
  unvalidated judge that quietly works would get used.
- *`validate_judge` reports rank correlation as well as agreement.* A judge can
  agree on 85% of branches and still mis-rank two systems if its errors
  concentrate in one -- and the ranking is what a conclusion rests on.

**Decisive experiment re-run after the matcher fix**
Baseline PR 58.3% -> 62.5%, difference -8.3% -> **-12.5% [-37.5%, +12.5%]**,
McNemar p = 0.51, 9 discordant branches. **Conclusion unchanged**: the interval
contains zero and the test has almost no power at this size. Both directions of
the negation fix moved numbers -- negation-to-negation now matches better,
negation-to-affirmation is now vetoed.

**Verified**
- A -> B end to end on mock: A1-A4 output feeds B2-B3 without adaptation, 25/25
  answers faithful, PR/SR separate composition (0.78) from pure selection
  (0.49) as the design requires.

**Next up**
- Everything remaining is data, not code: the 54 probe cases need human
  verification (3 flagged in `docs/wp1_probe_verification.md`), WP0's
  closed-venue sweep, WP2 annotation, and the kappa pilot, which needs a second
  person.

---

### 2026-09-24 (later) - Annotation tooling, probe verification, WP0 first pass, and a schema bug that inverted four test cases

Cleared the remaining Member A work and the manual tasks. The verification pass
found a defect that had been silently inverting the four cases most important to
the scope-relation claim.

**Done**
- `benchmark/annotation/` — the labelling tool. Was empty; it blocked annotation,
  which is the project's critical path.
  - `store.py` — per-annotator files, blind by construction. There is deliberately
    no "load everything" method.
  - `agreement.py` — Cohen's kappa per axis, with bootstrap CI and the
    disagreement cells.
  - `cli.py` — `annotate` (blind pass), `adjudicate` (review model proposals),
    `agreement`, `status`.
- `docs/wp1_probe_verification.md` — adversarial pass over all 54 probe cases.
- `docs/wp0_positioning_memo.md` — completed from skeleton to a first-pass
  deliverable with a verdict.
- 23 tests added (400 -> 423). Order-invariance gate still PASS at 1.0000.

**Found: four probe cases contradicted their own labels**
`wp1_fin_dis_051`, `wp1_imm_dis_052`, `wp1_fin_opp_053`, `wp1_imm_opp_054` each
declared `disjoint` or `opposed` while carrying a branch structure that says
`refinement`. Mechanically checkable: `compare(exception, default) = subset` in
all four.

Root cause was in the schema. `GoldInstance` required **exactly one** default
branch, whose applicability is by definition the whole case space — which makes
every other branch a subset of it, and nested-with-differing-outcomes *is*
refinement. The rule encoded the refinement shape and imposed it on every
relation. These four cases exist specifically to test telling refinement from
the rest, so they were testing the opposite of what they were for.

**The same bug was in `contract/mock.py`**, and worse: `_build_opposed` used
scopes `first-year holders` and `first-year holders at designated institutions`
— genuinely nested, so the template was a refinement mislabelled as opposed,
training the detector to call refinements opposed. Now uses crossing scopes
(first-year x part-time).

**Decisions**
- *Contract v1.0.0 -> v1.1.0: "exactly one default branch" relaxed to "at most
  one".* A default is still required for `refinement` and `redundant`, which
  genuinely are the default-plus-carve-out shape. Backwards-compatible (existing
  instances still validate) and the wire format is unchanged, so the bump is
  minor.
- *Kappa is refused, in code, when it could not measure inter-annotator
  agreement* — the same person twice, or anything model-generated. It renders a
  plausible number either way, and a fabricated kappa is unrecoverable once
  submitted.
- *A degenerate batch reports "undefined", not 1.0.* Both annotators using one
  category gives 0/0; reporting 1.0 would present a batch that tested nothing as
  flawless annotation.
- *Adjudicated labels carry `basis="model"`.* Reviewing a proposal is ~4x faster
  than labelling cold and carries automation bias. Real human judgement, but not
  the same evidence, and never pooled silently.
- *`Explicitness` docstring corrected.* It said "condition stated or inferred";
  the operational test is absence of a strong exception cue. Those differ — in
  "Where the statement balance is settled in full...", the condition is stated
  and only the override is not. Write-up must say "recovery of unmarked
  exceptions", never "implicit condition recovery".

**WP0 verdict: bridging claim survives, narrowed. Confidence moderate.**
- All 9 citations verified against arXiv / ACL Anthology. None fabricated.
- SG-DT takes an already-identified provision and never models competition
  between separately retrieved sources, so the retrieval-stage distinction
  holds. **This rests on an abstract plus a partial PDF read, and one automated
  read of that PDF returned wrong answers about the paper's own terminology.**
  A human must confirm against SG-DT §3-§4.
- Four citation discrepancies to fix: the "37%" NormBench figure is unverified;
  TCR is context-memory not inter-source; ConflictRAG's "Entropy-TOPSIS" is
  unconfirmed; SG-DT's second pathology ("Auditability Trap") was missing.
- Five works found that the memo lacked. CARE-RAG (arXiv:2507.01281,
  "conflict-driven summarization") is the nearest neighbour and must be cited
  before a reviewer raises it.
- **Closed-venue sweep (KR, TPLP, AAAI, AIJ, JAR) is NOT discharged** and is
  where the residual risk sits: classical KR has reasoned about
  specificity-based override since Reiter 1980.

**Not done, and cannot be**
- *The kappa pilot.* It measures whether two independent humans converge. Both
  label sets from one source measures self-consistency instead. Needs a second
  person.
- *Bulk annotation as gold.* Labels generated by the same model as the detector
  make the evaluation circular. `adjudicate` exists to cut the human cost from
  ~20 hours to ~3, not to remove the human.

**Next up**
- Member B's modules (`extraction/`, `composition/`, `generation/`) are still
  8-line stubs and are the largest remaining build.

---

### 2026-09-24 - Cost/latency instrumentation, error analysis, and a key that should not have been committed

Closes the last two measurement gaps against the proposal, and catches a
credential sitting in a tracked file on the way to publishing the repo.

**Done**
- `experiments/run_cost_latency.py` — per-query cost and latency broken down by
  stage, per proposal §6.1. `--compare-stage2` prices the escalation itself by
  running the same records with stage 2 disabled and differencing.
- `experiments/error_analysis.py` — the qualitative pass §5.2 promises on the
  confused instances. Errors are grouped by the *feature pattern that drove
  them*, so a systematic weakness shows up as a cluster rather than as a list
  of anecdotes. Emits an annotatable markdown sheet for the manual read.
- `tests/test_cost_latency.py` (10) and `tests/test_error_analysis.py` (14).
  Suite now **400 passing**; order-invariance gate still **PASS at 1.0000**.

**Found**
- **`.env.example` contained the real Groq key**, not a placeholder — tracked,
  and present in commit `ca0ec7d`. The key had already been rotated, so the
  exposure is historical rather than live, but it would have been published to
  a public remote. Replaced with `gsk_your_key_here` and scrubbed from history
  in the same rewrite that removed the commit trailers. `.env` itself was never
  tracked (`.gitignore:2`), which is why this survived earlier checks — the
  check confirmed `.env` was ignored and never looked at `.env.example`.

**Decisions**
- *Cost/latency reports median and p95, not just the mean.* The distribution is
  bimodal — most queries resolve in stage 1, a few escalate to an API call — so
  the mean sits between the two modes and describes no query that actually ran.
- *The bimodal warning needs an absolute floor, not just a ratio.* Offline,
  both modes are sub-millisecond and the p95/median ratio between two samples
  of timer noise runs past 50x. The first version duly announced a bimodal
  distribution on a run where nothing took any time. `BIMODAL_FLOOR_S = 0.010`.
- *A fully cached run is reported as such.* Warm-path cost is zero, which is
  true and useless for planning; without the caveat it reads as "the system is
  free to run" when what happened is that somebody ran it before.
- *Errors are grouped by feature signature rather than by surface text.* An
  error grouped by what drove it points at a fix; grouped by its wording it
  points at an anecdote.

**Verified**
- On 12 mock records the error analysis found a genuine cluster: 57% of the
  `conditional -> no_conflict` misses share the signature
  `no-cue / no-numeric-clash / nli-neutral / scope-asymmetry` — cueless
  exceptions that the heuristic NLI reads as unrelated. That is the harness
  working, not a result: mock data with heuristic NLI, so the 83% error rate is
  a fixture property and is reported as one.
- One test in the first draft (`test_records_absent_from_gold_...`) passed for
  the wrong reason — mock ids derive from the template name, so two seeds
  produce identical ids and nothing was ever skipped. Rewritten to rename the
  ids explicitly and assert the denominator excludes them.

**Blocked / open**
- Unchanged and all on the data side: the 54 probe cases need human
  verification, WP0 literature review, WP2 annotation (250–350 instances), and
  the κ pilot with two annotators. No code blocks on any of these.

**Next up**
- Published to the remote. Next substantive step is WP2 annotation — the
  infrastructure is ahead of the data, and every remaining headline number is
  waiting on labelled instances rather than on more code.

---

### 2026-09-24 (final) - The decisive experiment exists, and its first result was fake

Proposal 6.3's structured long-context baseline now runs end to end. The
important part of this session is that **the first number it produced was
wrong in the flattering direction**, and both causes are now tested against.

**Done**

- `baselines/structured_long_context.py` - a long-context model given every
  passage at once, asked for the gold branch schema. Two variants: `structured`
  (6.3's strong form) and `freetext`, the latter to separate the pipeline's
  architectural benefit from the benefit of merely requiring structured output.
- `metrics/branch_match.py` - deterministic gold-to-predicted branch
  alignment, so the experiment is scoreable without Member B's LLM judge.
- `experiments/run_decisive.py` - head-to-head with Wilson intervals, paired
  bootstrap difference, and McNemar.
- 376 tests pass (up from 354), 22 of them on this code specifically.

**The first result, and why it was fake**

    pipeline 100.0%  vs  baseline 33.3%   difference +66.7%   p = 3e-05

Spectacular, and wrong at both ends.

1. **The pipeline was scored with GOLD labels.** `to_query_record(
   include_gold_pairs=True)` hands routing the correct conflict type, so it
   routes perfectly by construction. That is an ORACLE pipeline compared
   against a baseline doing its own detection - it flatters the pipeline by
   exactly the detector's error rate.
2. **The matcher was too strict.** Gold outcomes are terse annotations; model
   outputs are verbose prose. Symmetric Jaccard scored gold "a 2.50
   per-transaction charge applies" against "You will be charged a 2.50 fee per
   cash withdrawal..." at **0.19** and called two identical outcomes a
   distortion.

**After fixing both**

    pipeline 50.0%  vs  baseline 58.3%   difference -8.3% [-29.2%, +12.5%]   p = 0.73

The baseline is nominally AHEAD and the interval contains zero. With only 8
discordant branches the test has almost no power, so this settles nothing yet -
but it is the honest reading, and it is the opposite of what the broken version
said.

**Fixes**

- `score_pipeline_routing(oracle=...)` defaults to running the REAL detector.
  Oracle mode still exists, is labelled "oracle routing" in every output, and
  its docstring says plainly that comparing it to a self-detecting baseline is
  the easiest way to manufacture a result here.
- Matching now combines three signals: numeric disagreement as a hard veto,
  symmetric Jaccard OR asymmetric containment, and optional NLI entailment.
  Containment handles the verbose-prediction case; **NLI is required for true
  paraphrase**, and a test records that containment alone does not rescue the
  original failing example.

**What the renderer now does**

Prints the falsifying outcome as prominently as a positive one, per 6.5; warns
when fewer than 10 branches are discordant; and states the open-weights
asymmetry inline - a pipeline win is weak evidence, a baseline win is decisive.

**Still not journal-ready**

Zero annotated instances, no kappa, WP0 not started. Every number above is on
12 hand-built unverified instances with the offline detector. The machinery is
right; the data is not there.

---

### 2026-09-24 (final) - Statistical rigour layer, and a broken invariance guarantee

Answering "is this journal-ready?" honestly: no, and the largest closable gap
was that **every number in the repo was N=1 with no interval**. That is now
fixed. The data gap remains and is not closable by code.

**A real defect found while auditing**

`interaction_features` claimed order-symmetry by sorting the two embeddings
into a canonical position on their element sum. Two different embeddings can
share a sum, and on that tie the sort is arbitrary and the vector genuinely
differs under swap. **The bug survived its own unit test** because the test
used embeddings with unequal sums.

It matters more than a normal defect: order invariance is a CLAIMED DESIGN
PROPERTY of this pipeline, and one that holds "almost always" is not a
property. Now `[e_i + e_j; |e_i - e_j|; e_i * e_j]` - every term symmetric, so
invariance holds by construction. Added a tie-case test and a randomised
property test across 3/64/384 dimensions.

**New: `metrics/stats.py`**

- **Wilson intervals** for proportions. Chosen over the normal approximation
  because this project's rates sit at the boundaries, where the normal
  interval misbehaves: 72/72 returns [1.0, 1.0] and claims certainty from 72
  observations.
- **Bootstrap intervals** for F1, macro-F1 and the leak rates, resampled over
  **instances, not pairs** - pairs within a record share passages, and
  resampling them independently understates every interval.
- **`bootstrap_difference`** resamples both systems together, preserving the
  pairing. Checking whether two separate intervals overlap is a weaker and
  different test, and a common mistake.
- **McNemar** with the exact binomial below 25 discordant items, because on a
  few hundred instances the disagreement count is often single digits and
  chi-square is unreliable there.
- **Holm correction**, because three variants means three comparisons and
  reporting the one that cleared 0.05 is how a null result becomes a finding.

**New: `reproducibility.py`**

`seed_everything()` covers all four random sources (random, numpy, torch CPU
and CUDA, PYTHONHASHSEED) plus cuDNN determinism. `run_manifest()` records git
commit and dirty state, contract version, hardware profile, tier map and
whether the threshold band was ever tuned - a seed alone does not reproduce a
number if the checkpoint moved underneath it.

**What the intervals immediately revealed**

On the three-variant comparison:

    cond->fact  0.0% [0.0%, 16.8%]
    accuracy  100.0% [89.3%, 100.0%]

A perfect zero leak rate - consistent with up to **16.8%** leakage, because
there were only 19 conditional instances. That is the entire argument for this
layer: the point estimate said "flawless", the interval says "we have not
measured enough to tell".

Also: `lexical_nli vs structured_entailment`, McNemar p = 1.5e-05, Holm-adjusted
p = 1.5e-05 - that ranking is real, and now demonstrably so rather than
asserted from a leaderboard.

**Wired in**

- `compare.py` records per-item correctness and prints Holm-corrected pairwise
  McNemar, flagging underpowered comparisons explicitly.
- `run_pipeline.py --evaluate` prints the provenance block and bootstrap
  intervals for every headline metric.
- 354 tests pass (up from 319), 35 of them on the statistics itself.

**One note on method**

My first hand-computed Wilson reference was wrong, and the test caught the
docstring rather than the code. Corrected, with the derivation written out so
the reference is checkable rather than asserted.

**Still not journal-ready, and not closable by code**

Zero annotated instances. No kappa. WP0 not started. The structured
long-context baseline still does not exist. Every number above is on mock or
hand-built data.

---

### 2026-09-24 (final) - WP1 batch 2: the estimate was optimistic, and that matters

**Done**

- Fetched and audited 10 more banking pairings: Santander, Nationwide,
  Starling, Monzo. Three Nationwide pairings were unfetchable - their
  `robots.txt` disallows `/current-accounts/` wholesale.
- Ran both miners on the 7 usable pairings.
- Consolidated `docs/wp1_tier1_findings.md` across all 38 pairings.
- 316 tests pass.

**The finding that changed the plan**

| batch | pairings | distinct docs | lexical | LLM | plausible on inspection |
|---|---|---|---|---|---|
| gov.uk | 17 | 5 (29%) | 4 | not run | unverified |
| banking 1 | 11 | 10 (91%) | 1 | 4 | **3** |
| banking 2 | 10 | 7 (70%) | 0 | 2 | **0** |
| **total** | **38** | **22 (58%)** | **5** | **6** | **3** |

Batch 2 produced two LLM proposals and both fail on reading: Santander pairs
two dated fee changes (temporal, not conditional), Monzo pairs two statements
about the same scope with inconsistent outcomes (closer to a contradiction).

**Batch 1 was the lucky batch.** An estimate built on it alone - roughly 0.4
plausible pairs per pairing, "~375 curated pairs for the floor" - was
optimistic by about five times. Across all 38 pairings the real rate is about
**0.08 per pairing**, which puts the 150-instance floor at ~1,900 pairings.

**Revised recommendation: Tier 1 cannot carry the corpus.** Set a Tier-1 target
the evidence supports (20-40 instances, reported as the naturally occurring
subset) and let Tier 2 carry the volume. A small, honestly labelled Tier-1 set
is worth more than a large blended count.

**Why the pairs are scarce even when documents are separate**

Banking scores 81% on document separation and still yields almost nothing. The
cross-document content is a product page saying what the account does and a
charges page saying what it costs - **complementary information**, the DRAGged
into Conflicts category section 4 already distinguishes this project from. The
genuine conditional structure sits inside one document, because an author
writing one clause states the carve-out next to the rule. Splitting a rule from
its exception across documents is simply not how organisations write.

That is a finding about the phenomenon, not only about the mining method, and
it belongs in the paper: it explains why the Tier 1 / Tier 2 construction is
necessary rather than merely convenient.

**Note to self on estimating**

I reported "~375 curated pairs" after one batch. One more batch moved it to
~1,900. Single-batch yield estimates on small samples are worth very little,
and the honest move would have been to label the first figure provisional.

---

### 2026-09-24 (later still) - WP1 domain 2: banking, and the proposer swap

**Two findings that change how WP2 should run.** Full write-up in
`docs/wp1_tier1_findings.md`.

**Done**

- `benchmark/mining/fetch_web.py` - general fetcher for HTML and PDF, with
  per-host `robots.txt` checked before every request and content-hash identity
  (commercial sites expose no `content_id`).
- `benchmark/mining/llm_proposer.py` - the proposer the proposal actually
  specifies, compared head-to-head against the lexical miner.
- 11 banking pairings fetched across NatWest, HSBC, Barclays, Lloyds. One URL
  was an origin-server hostname that `robots.txt` disallows; skipped and
  reported rather than fetched.
- 316 tests pass. Whole WP1 exercise cost 0.00 USD and 157K tokens.

**Finding 1 - banking is the far better Tier-1 domain**

| | gov.uk | banking |
|---|---|---|
| distinct-document pairings | 5 / 17 (29%) | **10 / 11 (91%)** |

Banks publish the rule and the charge in separate documents. gov.uk publishes
one guide with parts. Make financial terms the primary Tier-1 source.

**Finding 2 - the lexical miner was under-implementing the proposal**

Section 6.2 says *a model* proposes candidate pairs. The miner was cue words
plus word overlap. On identical documents:

| method | cross-document pairs |
|---|---|
| lexical | 1 |
| LLM proposer | **4** |

Three of four look genuinely right, including an implicit one: *"you won't pay
any non-sterling transaction fees when paying with your card"* against *"Not
valid for ATM withdrawals."* A product page and a charges schedule share few
content words, so a bag-of-words bridge fails even where the pairing is obvious
to a reader.

**A tuning result worth keeping**

Shrinking the proposer's input window from 9,000 to 5,500 chars per document
fixed a rate-limit problem and made the output WORSE - it stopped finding
fee-waiver pairs and started returning October price changes, because
truncation had cut the section where the rules live. Pacing the calls is the
right lever; truncation is not. `PACE_S = 22` now carries the TPM limit and the
window stays wide.

**Two more bugs, both caught by running it**

- The console crashed on a non-breaking hyphen (cp1252) *after* the work was
  done but *before* the JSON was written - the worst place to fail. Output is
  now sanitised for display and the JSON is written first.
- Four of ten pairings had failed on TPM exhaustion because the client's
  exponential backoff tops out well short of the 60-second rate-limit window.
  Backoff cannot clear a per-minute cap; pacing can.

**Net WP1 position**

Tier-1 pairs are scarce in both domains - the section 8 risk is confirmed.
Commit to the Tier 1 / Tier 2 split. But banking plus an LLM proposer makes the
150-instance floor plausible at roughly 375 curated document pairs, against
~1,200 documents for gov.uk. That should be planned, not assumed.

---

### 2026-09-24 (later) — WP1 Tier-1 gate ANSWERED on real gov.uk data

**The gating question is closed: Tier-1 pairs are scarce. Commit to the
Tier 1 / Tier 2 split.** Full write-up in `docs/wp1_tier1_findings.md`.

**Done**

- `benchmark/mining/fetch_govuk.py` — fetches through the **gov.uk Content
  API** rather than scraping. The API returns a `content_id`, and that field
  turned out to decide the whole question.
- Fetched 34 URLs / 17 candidate pairings / 32 documents, audited each pair,
  ran the miner, swept the threshold.
- `docs/wp1_tier1_findings.md` — the gating deliverable.
- 316 tests pass (up from 303).

**Three findings**

1. **Most "pairs" are one document wearing two URLs.** Only **5 of 17**
   pairings are genuinely distinct content items. Ten are two *parts* of one
   gov.uk guide — `/student-visa` and `/student-visa/money` share a
   `content_id`. URL-based mining would have counted those as Tier 1 and
   inflated the naturally-occurring claim about 3x.

2. **There is a third category the Tier 1 / Tier 2 binary does not cover.**
   "Two parts of one guide" is not Tier 1 (one document, one author) and not
   Tier 2 (nothing was artificially split — a RAG chunker really does index
   them apart). Needs its own reporting; do not silently absorb it into either.

3. **Yield: 4 Tier-1 candidates from 32 documents (0.12/doc).** The 150-floor
   would need ~1,200 documents. Across a threshold sweep the count moves 2→21,
   but the **Tier-1 share stays 5–18%** — same-document pairs outnumber Tier-1
   pairs by 5x to 50x at every setting. The share is the robust finding; the
   count is tunable.

**Four harness bugs, found because the first run's answer was obviously wrong**

The first run reported **939 Tier-1 candidates, 29.34 per document, "proceed
with Tier 1 as the primary corpus"**. It had passed its synthetic demo cleanly.

- **Cross product** — every rule sentence paired with every exception-looking
  sentence in the provider. Fixed with a topic-overlap requirement and a cap
  per rule.
- **Sentence splitting** — gov.uk headings have no terminal punctuation, so
  they glued to the next sentence. Now splits on newlines, with a length floor
  and a navigation filter.
- **Weak cues treated as exception markers** — `subject to`, `where the`,
  `if the` appear constantly in this register. Strong cues only now.
- **Five regex word boundaries were literal backspace bytes** (``), written
  by a shell heredoc that interpreted the escape. Those patterns silently never
  matched. Repaired; a repo-wide scan found no other affected file.

None of the four changed the conclusion, but the first run would have produced
exactly the inflated number the honesty guardrail exists to prevent. Worth
remembering that a harness passing a synthetic demo says little about how it
behaves on real prose.

**The caveat that could still move the answer**

All 34 URLs are gov.uk immigration. The proposal specifies **two** domains and
**financial terms are untested**. Banks publish separate product pages,
standalone fee schedules and dated amendment notices — a more Tier-1-friendly
structure than a government guide with parts. Testing that domain is the
cheapest remaining action that could change the corpus plan.

---

### 2026-09-24 — Ablation runner, zeroth-review deck, probe triage

Everything remaining that did not require Member A, Member B, or real data.

**Done**

- **`experiments/run_ablations.py`** — the two named ablations that had no
  runner. The headline one (route conditional → factual) is Member A's alone:
  it measures *routing*, so it needs no LLM judge and no generation step.
  Relabel every conditional pair as factual — what a four-class detector emits
  — and those pairs route to selection, which keeps one passage. Result on the
  probe set: **95.0% of exception branches lost** at the routing step. That is
  the argument for the conditional class existing, and it holds without the
  metric suite. Plus Tier-1/Tier-2 and explicit/implicit breakdowns.
- **Zeroth-review deck, slides 4–8** — `docs/zeroth_review_member_a.html`,
  published as an artifact. Each card is one slide, sized to screenshot into
  PowerPoint or print to PDF one page per slide. Every figure traces to a
  command in the repo. The honesty guardrail is on the slide, not buried:
  the deck says outright which numbers are structural results and which are
  fixture artifacts awaiting the corpus.
- **Probe review triage** — `--review` now surfaces **14 genuine judgement
  calls** ahead of the other 40 cases, each with the specific boundary it sits
  on and an accept/relabel line. Verification attention is finite; a reviewer
  who runs out of patience should run out of it on the easy cases.
- 6 more tests for the ablation runner. **Suite: 297 → 303.**

**Note on the 95% figure**

Measured on the 54-case probe set, which is conditional-heavy by construction.
The denominator is exception branches only, so the rate is not inflated by the
class balance — but it is still hand-built data, and the deck labels it as
such. Re-run on the WP2 corpus before it appears in the paper.

**Now genuinely blocked on Member A / Member B / real data**

| # | Item | What unblocks it |
|---|---|---|
| 1 | Verify 54 probe cases | `--review`, start with the 14 flagged |
| 2 | Contract sign-off | send `contract/schema/query_record-v1.0.0.json` to B |
| 3 | Rotate the Groq key | console → revoke → reissue → update `.env` |
| 4 | WP0 literature review | reading; the three anchors are in `docs/wp0_positioning_memo.md` |
| 5 | Tier-1 mining | **~20 real provider URLs** — then the harness runs |
| 6 | WP2 annotation | two annotators |
| 7 | kappa pilot | two annotators |
| 8 | Deck slides 1–3 | co-authored with Member B |

---

### 2026-09-22 (later still) — Closed the debt from today's own work

An audit of what I had built during the day, not of the original plan. Four
issues, all self-inflicted, all now fixed.

**Done**

1. **Tests for the six untested modules.** `tests/test_tooling.py`, 51 tests
   covering `threshold.py`, `train.py`, `run_baselines.py`,
   `run_retrieval_eval.py`, `wp1_probe_set.py` and `budget_estimate.py`.
   Everything written that day had shipped with zero coverage while the core
   pipeline had 246 tests. **Suite: 246 -> 297.**

2. **Stale trained heads are now rejected instead of silently used.**
   `models/stage1_head.pkl` had been fitted before the cue-list split changed
   what `exception_cues_*` *counts*. The vector shape was unchanged, so it
   loaded fine and scored nonsense — the worst kind of failure, because
   nothing surfaces it. `PairHead` now records a `feature_signature()` hashing
   the feature names together with every cue vocabulary; `load()` refuses a
   mismatch with a retrain command, and `Stage1Filter` warns loudly rather
   than falling back in silence. The stale file was deleted.

3. **`--write` now exists.** `threshold.py` documented a flag that was never
   implemented, so tuning produced numbers with no path back into config. It
   now writes the chosen band to `config/hardware.yaml`, records `tuned_on`,
   and **refuses to write a band tuned on a saturated curve**. `Stage1Filter`
   reads its band from config, so a tuned band is actually used.

4. **README documents the tooling.** Eight of nine new tools were missing from
   the plan of record — the file Member B reads. Added, grouped by purpose,
   with the correct running order (train the head *before* tuning the band;
   the sweep is meaningless against the near-chance rule fallback). Status
   legend rewritten: several rows are DONE for the *harness* while the run
   itself is PENDING real data, and saying so is the point.

**Bugs found while doing it**

- `train.py` was auto-loading an existing head while fitting a new one, so the
  "rule-based baseline" it reported was not the rule-based baseline.
- My first `--write` patch silently failed to apply: the argparse flag landed
  but the handler did not, reproducing the exact documented-but-unimplemented
  bug I was fixing. Caught by running it rather than trusting the patch.
- Running `--write` on mock data marked the committed config `tuned: true`.
  Reverted. `test_shipped_thresholds_are_marked_untuned` now guards it —
  a band fitted to templated data is worse than no band, because it looks
  legitimate.

**State**

| Check | Result |
|---|---|
| Test suite | **297 passed** |
| Order-invariance gate | **PASS, 1.0000** |
| WP1 probe set | 54/54 valid |
| Shipped escalation band | `tuned: false` — placeholder, as it should be |

**Still open — all non-code, all Member A's**

1. Verify the 54 probe cases → unblocks Member B
2. Contract sign-off from Member B → blocks everything
3. Rotate the Groq key
4. WP0 literature review — *gating*
5. Tier-1 mining on ~20 real documents — *gating*, harness ready, no documents
6. WP2 annotation, 250–350 instances — largest time cost
7. kappa pilot with Member B
8. Zeroth review deck, slides 4–8

Two tools produce fixture artifacts until real data exists, and say so
themselves: the threshold curve saturates on templates, and exception recall
is floored by near-duplicate mock passages.

---

### 2026-09-22 (later) — WP1 probe set drafted (54 cases, awaiting Member A verification)

**Done**

- `benchmark/wp1_probe_set.py` — 54 hand-built cases for Member B's WP1
  feasibility probe, emitted as validated `GoldInstance` JSONL at
  `benchmark/data/wp1_probe.jsonl`.

  | Group | n | Purpose |
  |---|---|---|
  | Implicit conditions | 28 | the probe's core — what Contrastive Scope Probing must recover |
  | Explicit conditions | 8 | control arm; explicit/implicit are reported separately |
  | Distractors | 14 | factual, temporal, opinion, no-conflict, redundant — the SCR denominator |
  | Disjoint / opposed | 4 | so the set covers all four scope relations |

- Every instance carries `annotator_a="model-proposed"`, `annotator_b=None`.
  **This is not gold data until Member A signs off.** That is the workflow the
  proposal specifies (model proposes, human verifies), and reviewing is far
  faster than authoring.
- `python -m benchmark.wp1_probe_set --review` prints a verification sheet with
  a four-point checklist per case and a `why` note saying exactly what the
  reader must infer.

**Design notes**

- The 28 implicit cases each carry the narrowing by a *different device* —
  product name, scheme membership, tier, amount threshold, jurisdiction, time
  window, customer status, sponsor attribute, recharacterisation, negated
  sub-condition. A method that handles only one device is exposed rather than
  flattered.
- Deliberate boundary cases for Member A to adjudicate: `imp_008` (origination
  date — conditional, NOT temporal, since both texts remain in force);
  `imp_019` and `imp_020` (surface contradictions that are conditional, not
  factual); `fin_red_049` (nested scope, agreeing outcome — the single most
  important distractor, since proposing a condition there fabricates a branch).
- Prose is written for this benchmark, not copied from any provider: realistic
  in shape, fictional in content, so the set carries no licensing question.
  Tagged `construction="split"` so it can never be counted toward Tier 1.

**Two problems the tooling caught in the draft**

1. **Four "implicit" cases contained exception cues.** One (`imp_009`,
   "exempt from") was genuinely mislabelled and has been reworded. The other
   three were false positives from an overly broad cue list — see below.
   `check_implicitness()` now runs inside `build_all()`, so a mislabelled case
   cannot be emitted.

2. **`contract.validate` reported no `disjoint` or `opposed` instances**, with
   the note that a fixture missing a relation cannot catch confusions involving
   it. Four cases added; all four relations are now present.

**A real defect in the detector, found while checking the probe set**

`EXCEPTION_CUES` lumped strong exception markers together with generic
conditional prose — `"where the"`, `"if the"`, `"subject to"`. Those fire on
almost every sentence in this domain, so `exception_cues_max`, one of the
strongest signals for the conditional class, was partly measuring "this text
describes a rule" rather than "this text states an exception". The noise landed
directly on the factual/conditional cell the detector is judged on. Now split
into `STRONG_EXCEPTION_CUES` and `WEAK_CONDITION_CUES`; the feature counts
strong cues only.

**Measured**

| Check | Result |
|---|---|
| Probe set validation | **54/54 valid**, all 4 scope relations |
| Implicit/explicit split | 0 mislabelled (enforced at build time) |
| Test suite | **246 passed** |
| Baselines on probe prose | nli_filter **26.8%** branch loss (vs 15.8% on templates) |

That last row is worth noting: realistic varied phrasing is harder than the
templates, which is exactly why the probe set exists.

**Next for Member A**

1. `python -m benchmark.wp1_probe_set --review` and verify all 54. Correct the
   Python source, not the JSONL, then regenerate.
2. Hand the verified JSONL to Member B to start the probe.
3. Remaining non-code items unchanged: contract sign-off, rotate the key, WP0,
   Tier-1 mining on real documents, WP2 annotation, kappa pilot, review deck.

---

### 2026-09-22 — Closed Member A's six code gaps

All six items from the completeness audit. Four of them surfaced real bugs.

**Done**

- **`detection/threshold.py`** (gap 1) — sweeps the stage-1/stage-2 escalation
  band and reports the cost/accuracy curve with a Pareto front. Selection
  requires an explicit constraint (`--max-escalation` or `--min-accuracy`),
  because without one the answer is always "escalate everything", which is what
  the two-stage design exists to avoid. Detects and refuses to tune on a
  saturated curve.
- **`detection/train.py`** (gap 3) — trains and persists the stage-1 head,
  compares it against the rule-based fallback on a held-out split, and refuses
  to save a head that does not beat the fallback. `Stage1Filter` auto-loads it.
- **`experiments/run_baselines.py`** (gap 2) — scores the three retrieval-side
  baselines on **branch loss**, not answer correctness: did the exception
  passage survive to generation? Prints worked examples.
- **Variant (ii) now trains** (gap 4) — two blockers fixed, see below. The
  three-architecture comparison is finally three-way.
- **`experiments/run_retrieval_eval.py`** (gap 5) — recall@k and, more
  importantly, **exception recall@k**, comparing BM25 against hybrid.
- **`scripts/eval_descriptors.py`** (gap 6) — measures the typed-attribute rate
  directly.

**Bugs found by running these**

1. **Stage 1 was at chance.** The threshold sweep showed stage-1 accuracy of
   **0.500 at zero escalation** with the rule-based fallback. The earlier
   "70% of pairs resolved locally at zero API cost" was therefore measuring
   escalation rate, not correctness -- it was resolving them *wrongly*. After
   training the head: 0.460 -> 1.000 held-out. (1.000 is itself a fixture
   artifact; the mock templates are regular enough to memorise.)

2. **fp16 master weights broke fine-tuning.** Variant (ii) died with
   "Attempting to unscale FP16 gradients": some checkpoints carry a
   half-precision dtype in their config which `from_pretrained` honours, so
   parameters arrived as fp16 while GradScaler expects fp32 masters with fp16
   activations. Fixed with an explicit `.float()`, which is version-agnostic
   where the dtype kwarg name is not. Also needed `tiktoken` + `sentencepiece`
   for the DeBERTa-v3 tokenizer.

3. **A contradictory ConflictPair could be constructed.** When stage 1 said
   "conflict" but the A3 classifier returned `no_conflict`, the pipeline built
   `is_conflict=True, type=no_conflict` -- which the contract rejects. Latent
   until the trained head changed stage-1's verdicts and exposed it. The
   classifier now wins: it answers *how* two passages disagree, and "not
   actually" is a legitimate answer.

4. **`compare()` deferred a decidable case.** When two applicability sets
   constrain *different* dimensions (`account_kind` vs `age`) it returned
   UNKNOWN and fell through to entailment. That case is provably OVERLAPPING:
   they necessarily intersect, and neither contains the other because each
   leaves the other's dimension unconstrained. Since UNKNOWN falls back to
   `opposed`, this was also costing composability on merely cross-cutting
   pairs. **Live effect: exact attribute decisions 0% -> 44%, and `compose`
   fired 4x instead of 1x on the same input.**

**Measured**

| Check | Result |
|---|---|
| Test suite | **246 passed** |
| Order-invariance gate | **PASS, 1.0000**, 66/72 non-positional |
| Typed-attribute rate (live) | **82%** on representative passages |
| Baseline branch loss | rerank_top1 **55.4%**, nli_filter **15.8%**, standard_rag 0% |
| Three-variant comparison | lexical_nli 0.988 acc, combined leak 0.010; other two flagged DEGENERATE |

**The suppression demo now runs**, and it is the slide:

    query:      Do I pay a fee on cross-border transfers?
    nli_filter: dropped p1 as inconsistent (contradiction score 1.000)
    should say: ...the fee is waived for premium-tier cardholders (per p1).
    will say:   Cross-border transfers incur a 3.5% fee.

**Two numbers that are fixture artifacts, not results**

- Exception recall reads 28.6%, but **65% of mock passages are near-duplicates**
  of each other, so pooled retrieval is being asked an unanswerable question.
  The tool now detects this and says so before the number can be misread.
- Stage-1 held-out accuracy of 1.000, and the saturated threshold curve, are
  both the templates being memorised.

Both need re-running on the WP2 corpus. Neither is quotable now.

**Still open for Member A** — none of it code:

1. Contract sign-off from Member B (blocks everything)
2. Rotate the Groq key
3. WP0 literature review, A's slice — **gating**
4. ~50 labelled cases for B's probe — **gating**
5. Tier-1 mining on ~20 real documents — harness ready, zero documents — **gating**
6. WP2 annotation of 250-350 instances — the largest single time cost
7. kappa pilot with B on refinement-vs-opposed
8. Zeroth review deck, slides 4-8

---

### 2026-09-21 (later) — Groq wired up live; four provider findings

Key added, all four tiers verified against the real API, and the rate limits
turned into a planning artifact rather than a surprise.

**Done**

- `.env` created with the Groq key (gitignored; verified untracked).
- **Tier map set from the real free-tier limits** (`config/models.yaml`), with
  those limits stored as data so tooling can reason about them.
- `scripts/smoke_test.py` — live check: every tier reachable and non-empty,
  structured JSON parses, truncation fails loudly, cache serves the second
  call free. **All green.**
- `scripts/budget_estimate.py` — multiplies pipeline call volume against the
  daily caps and reports wall-clock days per model.
- `api_budget` hardened for reasoning models: `reasoning_effort` passthrough,
  `TruncatedResponseError`, per-model capability guard.
- **Live end-to-end run** (5 instances, real DeBERTa on GPU + real Groq calls):
  detection resolved **80% of pairs locally at zero API cost**, stage 2
  escalated one, A4 ran live, routing produced one COMPOSE.
- 245 tests pass.

**Findings — all four came from running the thing, not from reading docs**

1. **Groq rate limits are PER MODEL, not per account.** Putting every tier on
   gpt-oss-120b caps the project at 200K tokens/day. Spreading BULK, JUDGE and
   SECOND_BACKBONE across three models gives **1.1M/day — 5.5x**. This is the
   single reason the tier map looks the way it does; it is not a quality
   compromise.

2. **gpt-oss is a reasoning model, and its reasoning tokens consume
   `max_tokens` before any answer appears.** At `max_tokens=20` it returned
   `finish_reason='length'` with *empty content*. The old backend would have
   returned `''`, every downstream JSON parse would have quietly fallen back,
   and a whole corpus run would have produced plausible-looking empty
   extractions. Now raises `TruncatedResponseError` on any `length` finish —
   partial output too, since JSON cut mid-string is garbage rather than an
   obvious error. `reasoning_effort` is set per tier (`low` for bulk, `medium`
   for load-bearing).

3. **allam-2-7b rejected as the overflow model**, despite having 2.5x the
   daily token budget of anything else. Two disqualifiers, both measured:
   it returns HTTP 400 on `reasoning_effort`, and it answers English prompts
   in Arabic by default. Mixed-language extractions on financial and
   immigration prose would be worse than a rate-limit stall and far harder to
   notice. Overflow is now qwen, which has spare quota.

4. **`groq/compound` has no daily token cap but is unusable as the decisive
   baseline.** It is an agentic system with built-in web search — it could
   look up the real policy instead of parsing the passages it was given, which
   would silently invalidate the one comparison the central claim rests on.
   The unlimited quota is a trap here. LONG_CONTEXT stays on gpt-oss-120b.

   Separately: `llama-prompt-guard-2-22m/86m` have the highest request limits
   on the account and are **not chat models** — they are prompt-injection
   classifiers. `canopylabs/orpheus-*` are text-to-speech. All excluded.

**The schedule problem, and the fix**

`python scripts/budget_estimate.py` on the 300-instance target, 12 configs:

| Plan | Wall clock |
|---|---|
| Naive (per-branch judging, all ablations on full corpus) | **115 days** |
| `--judge-batched --ablation-instances 100` | **41 days** |

WP4 is roughly 42 days. The fitted plan fits with **essentially zero slack**.
The metric judge is 65% of the total, so the two levers that matter are:

- **batch the judge per instance**, not per branch — the passages and rubric
  are re-sent for every branch otherwise, roughly doubling the cost for
  identical judgements;
- **run non-headline ablations on a stratified subsample**, headline
  configurations on the full corpus. Ordinary practice; report it as such.

Re-run the estimator before committing to any full-corpus pass. The cache
makes re-runs free, so the number above is for a cold run only.

**Decisions**

- `BULK` → `openai/gpt-oss-20b`, not allam. allam has 7x the request budget,
  but BULK feeds the factual/conditional decision — the paper's headline
  number — and buying throughput with quality there is a bad trade.
- `JUDGE` / `LONG_CONTEXT` → `openai/gpt-oss-120b` (they share one 200K/day
  cap, so their days add; the estimator now says so explicitly).
- `SECOND_BACKBONE` → `qwen/qwen3.8-27b`. Different family from gpt-oss, so
  the RQ3 robustness ablation is meaningful, and it carries its own 200K/day.
- **Rotate the API key.** It was pasted into a chat transcript, so treat it as
  public regardless of what the transcript is used for.

**Next up**

1. Contract sign-off from Member B — still the gating item.
2. Rotate the Groq key; update `.env`.
3. Tell Member B to batch the metric judge per instance from the start. It is
   a 2x budget difference and much cheaper to design in than to retrofit.
4. A4 currently decides 100% of relations by entailment rather than typed
   attributes on mock data — the LLM extractor is not emitting typed
   attributes for these passages. Worth tuning the descriptor prompt on real
   text, since attribute decisions are exact and entailment ones are not.
5. Start WP0 (gating, reading not code).
6. Collect ~20 real documents for the Tier-1 yield trial.

---

### 2026-09-21 — Repository bootstrap: contract, API layer, and all of Member A

**Done**

- **Scaffold.** Created `conflict-rag/` following §7 of the implementation
  plan. Python 3.11 venv (not the system 3.13 — parts of the ML stack lag
  behind it), `.gitignore` covering `.env` / venv / model caches / cache DB /
  working data, `.env.example`, `requirements.txt`, `pyproject.toml` with
  pytest config and `slow` / `network` markers.
- **`README.md`** — the plan of record: problem statement, A/B ownership split,
  the frozen contract, hardware and provider decisions, module build order with
  status, WP0–WP5 timeline, setup instructions, named risks.
- **`CONVENTIONS.md`** — repo conventions: the session-log rule, the contract freeze
  rule, hardware rules, API rules, and the research-integrity rules carried
  over from §12 of the implementation plan.
- **`contract/`** (Week-1 joint deliverable, A leads) — Pydantic v2 wire schema
  (`models.py`), gold annotation schema (`gold.py`), the routing table as
  executable code (`routing.py`), a seeded deterministic mock generator
  (`mock.py`), JSON Schema export, and a validator CLI. `CONTRACT_VERSION =
  1.0.0`, stamped on every record.
- **`api_budget/`** (Week-1 joint deliverable) — SQLite request cache keyed on
  everything that can change a response, provider-agnostic client over the
  `openai` SDK with configurable `base_url` (Groq + Ollama + any future
  provider on one code path), tiered model routing, JSONL cost log with a
  report CLI, and a deterministic offline mock backend.
- **`config/`** — `hardware.yaml` with three profiles (6 GB floor, 8 GB opt-in,
  CPU-only for CI) and `models.yaml` mapping logical tiers to models.
  `scripts/discover_models.py` populates the tier map from Groq's live model
  list rather than hardcoding IDs.
- **A1 `retrieval/`** — BM25 (`rank_bm25`) + dense (Contriever, fp16, local) +
  reciprocal-rank fusion at K = 5, over a passage store carrying source type,
  date and `document_id`.
- **A2 `detection/`** — two-stage detector. Stage 1: local DeBERTa-v3 NLI
  cross-encoder + hand-authored lexical features + a learned head over
  embedding interaction features. Stage 2: selective cached LLM escalation for
  the uncertain band only.
- **A3 `detection/variants/`** — all three architectures: lexical+NLI (with a
  coefficient-table `explain()`), fine-tuned DeBERTa-v3-base (multi-task: five
  classes + four-way relation), and structured entailment (the cross-encoder
  called twice and combined by the A4 rule). Plus `compare.py`, which reports
  the factual/conditional confusion cell as the headline.
- **A4 `scope/`** — the four-way relation analyser. Typed attribute algebra
  (`attributes.py`) for exact subset/disjoint decisions over tiers, ranges and
  categories; LLM + rule-based descriptor extraction; and `relation.py`, which
  decides applicability and outcome agreement **separately** and combines them.
- **`scope/order_invariance.py`** — the permutation harness, wired to exit
  non-zero so it can gate CI.
- **`baselines/retrieval_side.py`** — standard RAG, rerank-top-1, and the
  NLI-filter (the demonstrative one: on a conditional conflict it deletes
  exactly the exception branch).
- **`metrics/`** — A's scorers (binary detection, five-class with the named
  confusion cells, four-way scope with the four named cells, tier splitting)
  plus typed stubs for B's PR/SR/HCR/SCR with the agreed signatures.
- **Member B landing zone** — `extraction/`, `composition/`, `generation/`
  each hold an ownership README listing what goes there, what already exists to
  build against, which metrics B owns, and B's WP1 go/no-go.
- **WP support** — `benchmark/mining/tier1_pilot.py` (the ~20-document Tier-1
  yield trial, with a recommendation that changes based on measured yield),
  `docs/wp0_positioning_memo.md` (template with the three confirmed prior-art
  anchors and the venue checklist), `benchmark/manual/annotation_manual.md`
  (draft, including the refinement-vs-opposed decision procedure).
- **Tests** — 242 passing, offline, zero API spend, ~3 seconds.

**Verified by running**

| Check | Result |
|---|---|
| `pytest -m "not slow"` | 242 passed |
| Mock determinism (same seed, two runs) | byte-identical SHA-256 |
| Mock coverage at n=30 | all 5 conflict types, all 4 scope relations, nested + crossed, explicit + implicit |
| Contract validation of generated records | 60/60 valid |
| **`scope.order_invariance --sample 60`** | **PASS, 1.0000, 66/72 non-positional roles** |
| `detection.variants.compare` | runs, emits the three-variant table |
| `experiments.run_pipeline --evaluate` | runs A1→A4, emits a valid record and all scorers |
| `benchmark.mining.tier1_pilot --demo` | runs, produces a yield-based recommendation |
| torch CUDA | 2.6.0+cu124, RTX 4060 Laptop, 8.59 GB, `cuda.is_available() == True` |
| `pytest -m slow` (dense retrieval on GPU) | passed — Contriever downloads, encodes, and fuses with BM25 |
| Full suite incl. slow | **244 passed, 1 skipped** |

The one skip is `test_second_backbone_is_a_different_family_from_judge`, which
correctly skips until `SECOND_BACKBONE` is assigned.

**Decisions**

- **Provider: Groq**, zero frontier-API spend, local Ollama/Qwen 2.5 7B Q4 as
  the rate-limit fallback. *Why:* the user's constraint. The consequence is
  written into README §5 as a named risk rather than buried — see below.
- **The open-weights asymmetry.** At K = 5 the structured long-context
  baseline's input is only ~2–5K tokens, so context length was never the
  binding constraint; capability is. That makes the decisive experiment
  interpretable in only one direction: if the open-weights baseline *matches or
  beats* the pipeline, that is valid falsification and still publishable; if it
  *loses*, a reviewer attributes the gap to model tier. The experiment is still
  worth running — it can falsify the claim but cannot confirm it. This must go
  in the paper. Model IDs are config, not code, so one line swaps in a frontier
  model if credits appear.
- **Not using DeepSeek-R1-8B for bulk work.** It is a reasoning distill that
  emits long `<think>` traces — slow and awkward to parse across thousands of
  structured-extraction calls. Qwen 2.5 7B Instruct is the better local bulk
  model. R1-distill stays available as an extraction-model-size sweep row.
- **6 GB VRAM floor, 8 GB opt-in.** This machine has 8 GB; Member B's is
  unconfirmed. Both profiles train at the *same* effective batch size (32), so
  the two machines produce the same model from one config — otherwise the
  comparison between them would be meaningless.
- **Local GPU never runs a generative model** in the reported pipeline. It runs
  cross-encoders and embeddings. Putting a Q4 7B resident on the same 8 GB card
  would contend with the DeBERTa fine-tune and serialise the whole dev loop.
- **`set_relation` is stored canonically** (as the exception relative to the
  default), not as `compare()`'s raw argument-relative output. *Why:* found by
  a failing test — the raw value mirrors SUBSET↔SUPERSET under argument swap,
  which is semantically correct but would have made the order-invariance
  harness report a false violation on every single refinement pair.
- **A `redundant` gold instance resolves to exactly one merged branch**, not
  two. *Why:* found by the schema validator rejecting the mock generator's
  output. Recording a second branch would assert an exception that does not
  exist — the fabricated-branch failure that separating `redundant` from
  `refinement` exists to prevent.
- **Stage 1 alone never emits a `conditional` label.** It establishes only
  *that* two passages disagree, not *how*. An unearned conditional label routes
  to composition and can fabricate a branch, so the honest default is
  `factual`.
- **An undecidable scope relation falls back to `opposed`** (→ selection)
  rather than optimistically composing. Composing on unestablished scope
  invents a branch; selection is what prior work does and is the safe default.
- **Model: `openai/gpt-oss-120b`** (chosen by Member A, set late in the
  session) for `JUDGE` and `LONG_CONTEXT` — the two load-bearing tiers. It is
  among the strongest open-weights models Groq serves, which is the right call
  for the metric judge and the decisive baseline.
  - `BULK` is currently also pointed at it **only so nothing is unresolved**.
    This is the one tier that should move to a smaller model: bulk steps are
    thousands of calls and will exhaust a free-tier daily token cap long
    before the load-bearing steps do. Run `scripts/discover_models.py` and
    reassign.
  - `SECOND_BACKBONE` deliberately left **unset**. It must be a *different
    family* from JUDGE — another gpt-oss model would make the robustness
    ablation (RQ3, "architectural rather than model-specific") vacuous. A test
    enforces this and skips until it is assigned.

**Blocked / open**

1. **Contract sign-off from Member B.** `contract/schema/query_record-v1.0.0.json`
   is ready to send. Nothing should be built against a contract B has not
   agreed to.
2. **No Groq key on disk yet — THE remaining blocker for live runs.**
   Everything above ran offline. `.env` does not exist; only `.env.example`
   does, and `GROQ_API_KEY` is not in the process or user environment either.
   Create `.env` with the key, then run `python scripts/discover_models.py`.
   Until then every live call raises a provider-not-configured error.
3. **Member B's GPU size is unknown.** The 6 GB floor is assumed. If their card
   is 4 GB, `config/hardware.yaml` needs a fourth profile.
4. **`FineTunedDetector` has never been trained.** The code is written and fits
   the memory budget on paper; it has not been run. It is the one A3 variant
   with no rule-based fallback.
5. **`SECOND_BACKBONE` unassigned.** Needs a non-gpt-oss model of comparable
   size, chosen after seeing Groq's real catalogue.
6. **Supervisor should hear about the zero-spend consequence** before WP4, with
   the asymmetry argument. It is a framing decision, not only a budget one.

**Next up**

1. Send `contract/schema/query_record-v1.0.0.json` to Member B and get the
   contract frozen. Everything downstream depends on it.
2. Create `.env` with `GROQ_API_KEY`, run `python scripts/discover_models.py`,
   then make one live `BULK` call and confirm the second identical call is
   served from cache at zero tokens. Reassign `BULK` to a smaller model and
   pick a non-gpt-oss `SECOND_BACKBONE` from the discovered catalogue.
3. Train `FineTunedDetector` on mock data — not for accuracy, but to confirm it
   fits in 8 GB at the stated batch size and that the training loop runs.
4. **Start WP0.** It is gating and it is reading, not code. Begin with the three
   confirmed anchors in `docs/wp0_positioning_memo.md`.
5. Collect ~20 real documents and run `benchmark.mining.tier1_pilot` for the
   actual Tier-1 yield. Due end of week 4.
