# v7/v8 implementation decisions (append-only)

Each entry: what was decided, the evidence, and what it changes in the paper. Newest last.
The plan of record is `plan_v7/` (proposal v8, implementation plan v4) as cut down to a 7-day,
zero-cost schedule (week-one plan: contract, metrics, QNFA core, S1 mining; then experiments and release).

---

## 2026-10-31 D1

### D1.1 The natural conditional-conflict count is 18 (14 S1 + 4 S2), not 19

The 7-day plan was drafted with "19 natural conditional instances" from an audit that counted
every natural cross-document instance labelled `conditional`. Five of those are `disjoint`,
which proposal v8 section 4.2 reclassifies as **no conflict**. `benchmark/relabel_v7.py` derives
the v7 label from the frozen gold and reproduces plan v4's figure: **14 in S1, 4 in S2, 18 total**.

*Consequence.* S1 starts from 14, not 19, and only 1 of those was in the old test split. Reaching
the floor of 60 needs about 46 newly verified S1 instances, roughly 3.3x the mining the 7-day
plan assumed. The narrowing rule (proposal section 3.1) is therefore the expected outcome, not a
contingency, and is computed from the data by `benchmark.strata.quota_report`.

### D1.2 Derive, never rewrite, the frozen gold

`corpus.jsonl` stays at contract v1.2.0. The two-annotator kappa pilots were run on the old labels,
so relabelling is a derived sidecar (`relabel_v7.py`), and the stratified re-split archives the old
assignment (`corpus.pre_v7_split.jsonl`) instead of overwriting it. Contract v1.3.0 is purely
additive, so v1.2.0 records still validate (tested against all 128).

### D1.3 Strata live outside the contract

`Stratum` is derived by `benchmark/strata.py`, not stored in `GoldInstance`. A stored stratum could
disagree with the data it was derived from; a derived one cannot. This departs from plan v3 section
2.1 (which put `stratum` on `QueryRecord`) deliberately.

### D1.4 B7 cannot run as released on the free tier

NormBench's stage-1 prompt (`stage1_major_premise_extraction.md`, commit `bd56aff`, CC BY 4.0) is
1,196 lines, ~12k CJK characters plus ~24k other characters: **an estimated 18-27k tokens per call
before the passages are added**. Groq's free tier allows 8,000 tokens per minute on gpt-oss-120b, so
a single request exceeds the per-minute limit. This is a hard limit, not a budget shortfall.

*Decision.* B7 is split:

* **B7-full** -- the released NormBench pipeline, unmodified except for multi-passage input -- runs
  on the **local** backbone only (num_ctx raised for the long prompt), with the fidelity check run
  locally.
* **B7-compact** -- the repo's existing SG-DT-style structured prompt, which fits the free tier --
  runs on both backbones, and is the comparator for the confirmatory cells on gpt-oss-120b.

*Paper consequence.* The confirmatory H3 comparator on the primary backbone is B7-compact/B8-compact,
and must be called "SG-DT-style", not "the NormBench pipeline". B7-full is reported on the local
backbone with its limits stated. v4 estimated B7 at 4.0k tokens per call; that figure described a
prompt that does not exist.

### D1.5 Model downloads are bandwidth-limited

Ollama pulls ran at ~60-200 KB/s (total, regardless of parallelism): `llama3.1:8b` (4.9 GB) takes
~7 h and `qwen2.5:7b` (4.7 GB) the same, so they are pulled sequentially. Wiring and tests use the
already-installed `llama3.2` / `qwen3.5:4b` as stand-ins; **no reported number may use a stand-in**.

### D1.6 Pinned tiers

`READER`, `EVAL`, `BACKBONE_PRIMARY`, `BACKBONE_LOCAL` carry `fallback: false`: a rate limit raises
instead of silently handing the call to another model. A cell answered by a different model than the
one named in the paper is a different number. Tested in `tests/test_tiers_v7.py`.

### D1.7 Reader context is pinned in a Modelfile

Ollama's OpenAI-compatible endpoint ignores `num_ctx`, so the reader and local backbone are derived
models (`config/ollama/*.Modelfile`) with `num_ctx`, `temperature 0`, `seed 0` set and named in the
tier config. A reader whose context silently truncated an answer would score a system down for text
it never saw.

### D1.8 Figures are compared by unit

`branch_match.numbers_conflict` treats *any* shared figure as agreement, so "1270 for 28 days"
matched "1270 for 14 days". For CCA that is the dangerous direction (a reader with the wrong period
scores correct). `metrics/values.figures_conflict` compares figures per unit (days against days,
amount against amount). Found by the value-matching tests, not by inspection.

### D1.9 "May" is classified by what follows it

`detect_modality` treats "may apply" as a permission when the subject is a person ("applicants may
apply") and a hedge when the subject is a charge or rule ("a fee may apply"); "may receive/be/qualify"
is a hedge; "may not <action>" is a prohibition. Deliberately conservative: a false hedge costs
recall, a false determination inflates CCA, and recall is bounded separately by gate V2.
"Normally/usually/generally" are not hedges (gold answers use them to mark the default branch).

### D2.1 The pre-registered McNemar test must be forced exact

`metrics.stats.mcnemar` silently switches to a continuity-corrected chi-square at 25 or more
discordant items. At 20 vs 5 that gives p = 0.0051; the exact test gives 0.0041. The analysis plan
(proposal v8 section 10) and the power calculation both assume the exact test, and an S1 of about 50
instances can easily reach 25 discordant pairs. Added `force_exact=True`; **every confirmatory
comparison must pass it**. The default is unchanged so earlier reported numbers still reproduce.
Found because the power module's own exact p-value disagreed with the repo's, not by inspection.

### D2.2 Foreignness lives in `location`, not in the event

Proposal v8 section 5.3's worked example uses `event: any_foreign_currency_txn`. A cross-cutting
event class cannot be ordered by generality against `cash_withdrawal`, so the schemas put foreignness
in the `location` slot (`abroad`). The three worked fee variables come out exactly as the proposal
states (rows 1/2 distinct, row 2 subsumes row 3, rows 1/3 distinct); tested in `tests/test_signature.py`.

### D2.3 Unification gains an `OVERLAP` outcome

Proposal v8 is silent on crossing generality (cash withdrawals anywhere vs any transaction abroad).
Dropping such pairs loses a real class (an ATM fee and a foreign-transaction fee both reach a foreign
ATM withdrawal); calling them `UNCERTAIN` sends a decidable case to an LLM. `OVERLAP` is folded like
`SUBSUMES`, and the scope algebra then sees a crossing pair. Free text can never produce `DISTINCT`.

### D2.4 Mining that can reach a test set uses the EVAL tier only

`llm_proposer` was hard-wired to `BULK` = gpt-oss-20b, which is also evaluated detector D3. Plan v4
ground rule 4 forbids evaluated systems in test construction, so `--tier EVAL` (Qwen) was added and is
used for the S1 harvest. Cost: ~3,050 tokens per pairing, so ~65 pairings per day of Qwen quota; the
218-pairing harvest is a ~3-day job and is run best-first.

### D2.5 Exact MDD differs from the proposal's table by at most one point, in either direction

`metrics/power_sim.py` computes exact McNemar power (no simulation noise). It reproduces the proposal's
section 6.7 table within 1.0 point in every cell; it is not uniformly larger (n=60, d=0.15: 15 vs 16).
An earlier docstring claimed "up to one point larger"; a test caught the overstatement.

### D2.6 Both Ollama pulls failed once with EOF

The first Llama pull ended in `max retries exceeded` at 13%; the task reported "completed" only
because the shell wrapper exited 0. Re-run with an exit-status retry loop (`ollama pull` resumes).
Lesson recorded: a background task's exit code is the wrapper's, not the command's.

### D3.1 Entropy-TOPSIS on two passages is degenerate: margins are ~0 or ~1

With two alternatives, entropy weighting puts essentially all weight on the criteria that
discriminate, so closeness is either an exact tie or near-extreme. A typical rule/exception pair
(authority favours the rule; recency and specificity favour the exception) comes out as a **tie
(margin 0)**, so B3's sufficiency rule makes it *present both sources* rather than delete one.
Ablating specificity flips the same pair to "select the newest" (margin 0.989). Consequence for the
paper: whether B3 suppresses an exception is an empirical, per-pair outcome, not a given, and H1 may
fail honestly if B3 mostly presents both. The pre-registered B3 uses the specificity criterion and
the default delta; the no-specificity ablation is reported beside it. Found while fixing test
fixtures that assumed a margin nobody had computed.

### D3.2 B3 gets an oracle conflict signal

B3's published detector is trained on other data and is not reproduced. A weak detector would let B3
quietly degrade into B0. Given gold conflict groups B3 selects as often as it ever would, so any
suppression measured is the upper bound for a selection-based resolver. B3 is reported as
"ConflictRAG-style, with oracle detection".

### D3.3 O3 must verbalise each decision separately

First O3 draft merged the branches of every co-referential group into one list. The shared verbaliser
then chose one default and rendered a *different decision's* outcome as that rule's exception
("A 3% fee applies. However, if premium-tier cardholders, no fee applies.", for a flat fee and a
percentage fee). That is the misattachment the design exists to prevent, and a loose `or` assertion in
the first test hid it. Found by printing the actual output. Fixed by verbalising per component, with
hedged frames and parse-only repairs rendered in their own sections; the test now asserts the exact text.
