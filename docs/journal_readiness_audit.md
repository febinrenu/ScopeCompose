# Journal readiness audit — 2026-10-02

An honest assessment against what a Q2 reviewer checks, written from the code
and data as they actually are rather than from the plan.

**Headline: the instrument is built and the measurement has not been taken.**
Every component the proposal specifies exists and is tested. No experiment has
run on real data at a size that could support a conclusion. That is the gap,
and it is not a gap more engineering closes.

> **Superseded in part, 2026-10-03.** A stage-level measurement has since been
> taken over all 42 annotated instances, and it locates the gap precisely: the
> operator preserves 0.976 given a conflict, the detector finds 0 of 42, and
> Preservation Rate turns out to be gameable by verbosity. See the addendum
> before the critical path, and `docs/detector_operator_gap.md`. The headline
> above is still right that more engineering on the *operator* closes nothing.

---

## Fixed in this pass

### 1. There was no test split — and the existing ones leaked

The corpus was 89 `dev`, 39 `train`, **zero `test`**. Every threshold tuned and
every choice made so far was made on data that would also have served as the
evaluation set.

Worse, **18 source documents spanned dev and train**. 128 instances draw on 186
documents and `youth-mobility#overview` alone appears in five, so independent
assignment put the same gov.uk guide on both sides. A system that memorised its
wording would score on the held-out side for having seen the other, every
number would be inflated, and nothing in any output would have shown it.

`benchmark/splits.py` assigns by **document group**, using connected components
over shared documents — an instance citing two documents links them, so the
relation is transitive and components, not documents, are the indivisible unit.
Reassigned: **77 / 26 / 25**, zero leakage.

> The cost is stated rather than hidden: groups are indivisible, so exact
> proportions are unreachable and the audit reports where assignment actually
> landed.

### 2. No power analysis, on a project whose results keep spanning zero

The decisive experiment reported **−12.5% [−37.5%, +12.5%]** on 24 branches
with 9 discordant. The κ pilots repeatedly cleared a threshold on intervals
reaching 0.000. Every one of those reports was honest about being
underpowered — and none of them said what size *would* be enough.

`metrics/power.py` answers it. Validated against published values (z(0.975) =
1.9600, and the textbook two-proportion case at 93 per group).

| to detect | discordant needed | branches | instances |
|---|---:|---:|---:|
| 20 points | 22 | 63 | **32** |
| 15 points | 41 | 118 | **60** |
| 10 points | 94 | 269 | **137** |
| 5 points | 383 | 1095 | 555 |

**At the current 83 branches the smallest detectable difference is about 17%.**
Anything below that returns an interval spanning zero however the experiment is
run.

That is the number to put in the methods section, and it sizes the corpus by
argument rather than by when the annotators got tired.

> One finding worth carrying into planning, because it is the opposite of the
> intuition: a *lower* discordant rate needs *fewer* branches. The effect is
> concentrated in a smaller subset, so it is easier to see. Two systems that
> mostly agree but differ sharply where they differ are cheaper to distinguish
> than two that disagree everywhere by a little. A test asserting the
> intuitive direction failed, and the model was right.

---

## Open — and these decide acceptance

### 3. The central claim is unsupported, and currently points the wrong way

The one run of the experiment §6.5 designates as able to falsify the claim:

> pipeline **50.0%** vs structured long-context baseline **62.5%**,
> difference −12.5%, interval spanning zero, 9 discordant branches.

Underpowered, so it concludes nothing. But it is not evidence *for* the thesis,
and at 12 instances it never could have been.

**This is the single largest risk and it is not a code problem.** Re-run at
137+ instances per the table above. Decide with the supervisor *before* the
number arrives what the paper says if it stays negative — §6.5 already commits
to reporting that as prominently as a positive, and a measured null on a
well-specified claim, with a benchmark and metric suite as the contribution,
is a publishable paper. (The suite is **not** validated — the LLM judge has
never been run against human labels, and PR needs a precision term before it
can adjudicate a PR table at all. See the addendum.) Deciding after the number arrives is how
a project ends up quietly dropping a component.

### 4. The headline metrics have a 42-instance denominator

128 instances, but **only 42 carry branch structure**, and PR/SR/HCR/SCR score
branches. The other 86 have Member A's labels and nothing of Member B's.

Until that is annotated, the four headline metrics rest on 42 instances, 54 of
which are the probe set.

### 5. Two classes of annotation evidence, which must never be pooled

| provenance | instances |
|---|---:|
| two independent annotators | 74 |
| model-proposed + one adjudicator | 54 |

The κ of 1.000 was measured on the first class. The 54 probe instances were
never double-annotated — a model proposed and one person verified.

Reporting a single κ across all 128 would misstate the evidence. The audit
reports provenance per split for exactly this reason, and the paper needs the
same breakdown.

### 6. κ = 1.000 needs careful presentation

It is real: 36 informative instances, two plausible passes, varied label
distribution. It is also the kind of number a reviewer stops at.

Report it as **κ = 1.000 on n informative instances, p_e = 0.46–0.54, with
composition disclosed** — never the pooled figure that includes 33 unanimous
distractors, and never the `[1.000, 1.000]` interval, which is a bootstrap
artifact (every resample of a sample with no disagreement is also perfect).

### 7. The four-way relation is still barely populated

2 `redundant` and 2 `opposed` in 128 instances, plus 4 authored today and
unverified. Nine pilots and 91 pairings produced zero of either, because
published policy does not contain them.

The relation is the project's central contribution and cannot be reported as
measured until those cells are filled.

### 8. Source-document licensing is not cleared

The datasheet says so plainly: *"Source-document licensing must be cleared
during WP2 — do not assume it."* It has not been.

gov.uk content is generally Open Government Licence, which is permissive with
attribution. **Bank terms and conditions are ordinary copyright.** A benchmark
release that redistributes passage text from NatWest, HSBC, Monzo, Starling
and Santander is a different proposition from one that redistributes gov.uk.

Options, in rough order of safety: release gov.uk text with OGL attribution and
distribute the banking half as URLs plus offsets rather than text; seek
permission; or restrict the banking subset. **Resolve before release, not
after** — and it may affect what the paper can claim about domain coverage.

### 9. No limitations section exists

A Q2 submission needs one, and writing it is the cheapest way to find what else
is missing. The candidates are already known: open-weights-only evaluation,
Tier-1 scarcity, the same-guide category, the deterministic scorer standing in
for a validated judge, two domains, UK-only, English-only, first-order
conflicts only.

Four more, added 2026-10-03 and each with a number attached:

- **Detection recall is 0.064.** The end-to-end claim cannot be tested on this
  detector at any corpus size. State it as a limitation, not a footnote.
- **Preservation Rate has no precision term** and is maximised by verbosity;
  concatenation scores 0.855 against the gold answer's 0.699.
- **Every branch-scored number is measured on 42 authored synthetic splits.**
  Zero mined instances carry branch annotation, so none of the branch metrics
  has yet been computed on the cross-document case the project is about.
- **The one decisive run straddles all three splits** and is majority test
  data. It is exploratory.

### 10. The LLM judge is implemented but unvalidated

`validate_judge` exists and `judge_branches` now has a client path. The
validation itself — judge against human labels, Cattan et al.'s 0.89 as the
bar — has not been run, and §6.1 requires it before any judged number is
reported. `PreservationScores.judged_by` enforces disclosure rather than
silence, which is the right fallback but not a substitute.

---

## What is genuinely strong

Worth stating, because the list above is all deficits:

- **Two publishable findings already exist**, independent of the main claim:
  the Tier-1 scarcity result with its mechanism, and the κ-pilot diagnosis of
  manual defects — including a batch voided by a missing query field, which is
  a methodologically interesting failure.
- **The negative-result handling is pre-committed**, in code rather than
  intention: `verdict()` names the fallback, the decisive experiment prints the
  falsifying outcome as prominently as a positive, and `prevalence_estimate()`
  raises rather than returning a number the design cannot support.
- **Reproducibility is unusually good** — seeded runs, manifests with git SHA
  and model tiers, a frozen versioned contract, 518 tests.
- **The guards against self-deception are real**: κ refused for non-independent
  passes, pass-plausibility checking that caught an 80-second annotation run,
  provenance on every judged number, degenerate-batch detection.

That last category is what distinguishes this from a project that merely
produced numbers, and it should be visible in the paper rather than buried in
the repository.

---

## Addendum, 2026-10-03 — the gap is not where this audit assumed

Running every stage over all 42 annotated instances, rather than the twelve
the decisive experiment used, moved the problem. Full write-up in
`docs/detector_operator_gap.md`; the three items that change this audit:

**The detector, not the operator, is the bottleneck.** Given gold routing and
gold branches the composition operator preserves **0.976** of gold branches.
The detector in front of it fires on **0 of 42**, topping out at 0.2206
against an escalation floor of 0.35 — so stage 2 is unreachable with or
without an API key. Over all 128: precision 0.833, recall 0.064, 2 of 67
conditional conflicts recovered. §3 below said the central claim "points the
wrong way"; it is now clear *where*.

**Preservation Rate is gameable by verbosity.** Concatenating both passages
scores PR 0.855, above the gold answer's 0.699, because PR is containment
with no length normalisation and the suite has no precision-side term. This
invalidates PR-only comparisons between a composed answer and a verbose one —
which is the comparison the paper wants. **Add a precision term before
reporting any PR table.** This supersedes the "validated metric suite"
language used elsewhere.

**The decisive run used what is now test data.** Its twelve instances were
picked by file order before splits existed and now straddle 5 test / 4 train /
3 dev. It must be described as exploratory, and the re-run must be on a
defined split. `decisive.json` now carries full provenance and all 24
per-branch outcomes; previously it carried neither.

Also corrected since this audit: three of the pipeline arm's five metrics
(distortion, HCR, SCR) are structurally incapable of being non-zero and must
read `n/a`; `year_clash` was a dead feature (capturing-group bug) and is now
fixed with a regression test.

---

## The critical path

1. **Annotate branch structure** on the 86 instances lacking it — unblocks every
   headline metric.
2. **Add a precision term to the metric suite.** Nothing else on this list is
   worth measuring until PR can tell a composed answer from a concatenated
   one. New, and now first among the measurement items.
3. **Verify the authored `redundant`/`opposed` cases**, then author enough to
   reach the floor of 20 each.
4. **Make detection work, or reframe around it.** Train the head, or route on
   extracted scope attributes instead of surface overlap. At 3% recall the
   end-to-end claim cannot be tested at any corpus size.
5. **`run_probe --live`** — Member B's go/no-go, answerable now.
6. **Re-run the decisive experiment at 137+ instances on a defined split**,
   per the power table.
7. **Validate the judge** against human labels, or commit to the deterministic
   scorer and justify it.
8. **Clear licensing** before any release.
9. **Write the limitations section.**

Items 1 and 3 are annotation. Items 5 to 7 are runs. Items 2 and 4 are builds,
and they were not on this list before the measurement above.
