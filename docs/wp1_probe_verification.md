# WP1 probe set — verification pass

**Status: HUMAN-VERIFIED, 2026-09-25.** All 54 cases reviewed case by case.
Three changed, one dropped, 50 accepted as proposed. The set is 53 instances and
is usable as gold.

This document records two passes: the model pass that found the structural
defects (sections 1-5), and the human pass that resolved them (section 7).

The cases were model-proposed. Sections 1-6 record an adversarial model pass
that found four structural defects — mechanical contradictions between a case's
label and its own branch structure, checkable against the schema rather than
against judgement. Section 7 records the human pass that reviewed all 54 case by
case and resolved the three it could not settle.

The model pass could not have replaced the human one: reviewer and proposer
share a failure mode, so a case both got wrong in the same direction would have
survived unchallenged. It did, however, catch things a reader would not — the
`016` re-encoding exposed a matcher bug that had been silently equating "five
years" with "four years".

---

## Summary

| | count |
|---|---|
| cases reviewed | 54 |
| **accepted as proposed** | **50** |
| **changed** | **3** (`016`, `026`, plus the §1 re-encodings) |
| **dropped** | **1** (`054`) |
| final set | **53** |
| definitional issue affecting the whole implicit subset | see §3 |

---

## 1. The structural defect: four cases contradicted themselves

`wp1_fin_dis_051`, `wp1_imm_dis_052`, `wp1_fin_opp_053`, `wp1_imm_opp_054`.

Each declared a relation of `disjoint` or `opposed` while carrying a branch
structure that says `refinement`. Both cannot be true.

The mechanism: every case encoded its first branch as the **default**, whose
applicability is by definition the whole case space. That makes the second
branch necessarily a *subset* of the first — and nested scopes with differing
outcomes is the definition of refinement. Checked mechanically:

```
wp1_fin_opp_053: compare(exception, default) = subset   # label says opposed
wp1_imm_opp_054: compare(exception, default) = subset   # label says opposed
wp1_fin_dis_051: compare(exception, default) = subset   # label says disjoint
wp1_imm_dis_052: compare(exception, default) = subset   # label says disjoint
```

These four are precisely the cases that exist to test whether a system can tell
refinement from the other three relations. Encoded this way they tested the
opposite of what they were for, and a system that correctly called them
`opposed` would have been scored wrong.

**Root cause was in the schema, not the cases.** `GoldInstance` required
*exactly one* default branch. That encodes the refinement shape — an
unconditioned general rule plus a carve-out — and silently imposes it on every
relation. For an opposed pair neither branch is a default: "Reward account
holders get lounge access" is already restricted to Reward holders.

**Fix (contract v1.0.0 → v1.1.0):** *at most* one default branch; a default is
still required for `refinement` and `redundant`, which genuinely are the
default-plus-carve-out shape. The four cases were re-encoded with real scopes on
both branches, and now verify:

```
wp1_fin_dis_051 [disjoint] compare = disjoint      'instant access savings' vs 'fixed-term bonds'
wp1_imm_dis_052 [disjoint] compare = disjoint      'visitor visa holders'   vs 'student visa holders'
wp1_fin_opp_053 [opposed]  compare = overlapping   'Reward account holders' vs 'accounts opened from Jan 2024'
wp1_imm_opp_054 [opposed]  compare = overlapping   'degree-level qualification' vs 'applicants over 45'
```

The relaxation is backwards-compatible — existing instances with exactly one
default still validate — so the bump is minor and the wire format
(`QueryRecord`) is unchanged.

### The same bug was in the mock generator

Found by checking whether the defect was local. It was not:

- `_build_disjoint` marked its first branch default → scored as refinement.
- `_build_opposed` was worse. Its two scopes were `first-year holders` and
  `first-year holders at designated institutions` — **genuinely nested**, so the
  template was a refinement mislabelled as opposed, independent of the default
  question. It was training the detector to call refinements opposed.

`_build_opposed` now uses crossing scopes (first-year × part-time): a first-year
full-time holder is in the first only, a part-time holder in a later year in the
second only, and a first-year part-time holder in both with the outcomes
disagreeing. That is what opposed means.

---

## 2. One case rejected: `wp1_imm_opp_054`

Label `opposed`. **The two passages do not actually conflict.**

> p0: "Applicants holding a recognised degree-level qualification meet the
> **skill requirement**."
> p1: "Applicants aged over 45 at the date of application do not meet **the
> requirements of this route**."

These are different predicates. The skill requirement is one of the route's
requirements, not all of them. A 46-year-old graduate meets the skill
requirement *and* fails the route — both statements true, no contradiction. The
apparent opposition is an artifact of p0 saying "the skill requirement" and p1
saying "the requirements".

It has been re-encoded structurally (§1) and the branch outcome tightened to
"the skill requirement is met", but **the case still needs rewording or
dropping**: as written it is closer to a no-conflict distractor than to an
opposed pair. Flagged for the human pass rather than silently rewritten, because
choosing the replacement wording is a content decision.

---

## 3. The implicit/explicit axis measures something narrower than its name

Not a defect in any individual case — a definitional mismatch that affects how
the headline number should be described.

`Explicitness` was documented as "whether a condition is **stated or must be
inferred**". The operational test is the absence of a strong exception cue
("except", "unless", "only if", "does not apply to"). Those are not the same
property, and the gap is wide:

> `wp1_fin_imp_005` — "**Where** the statement balance is settled in full by the
> payment due date, purchases benefit from up to 56 days of interest-free credit."

The condition is stated plainly. What is unstated is that it *overrides* the
general rule that interest accrues from the transaction date. Under the
documented definition this is explicit; under the operational one it is
implicit. Same for `wp1_imm_imp_021`, which uses the identical "Where the…"
construction.

Surveying the set, nearly every passage states its own condition somewhere. What
varies is whether the passage **announces itself as a carve-out**.

**Resolution:** the operational definition is the defensible one — inferring the
exception relationship is the genuinely hard step, and it is what Contrastive
Scope Probing addresses. The docstring has been corrected to match. The
consequence for the write-up is a wording constraint:

- ❌ "implicit condition recovery" — overclaims; the conditions are mostly stated
- ✅ "recovery of unmarked exceptions" / "cases where the exception relationship
  is not signalled"

Getting this wrong in the paper would be an overclaim a reviewer could check in
five minutes by reading three examples.

---

## 4. Case-by-case: the 14 high-risk cases

| case | flagged concern | verdict |
|---|---|---|
| `wp1_fin_imp_008` | conditional vs temporal | **confirmed conditional.** Both rules are simultaneously in force; which applies turns on *loan origination date*, a property of the case. Temporal conflict is when a newer rule supersedes an older one — not so here. |
| `wp1_fin_imp_019` | conditional vs factual | **confirmed conditional / refinement.** "Non-refundable" vs "returned in full" reads contradictory only until the 14-day scope is applied. Nested, outcomes differ. |
| `wp1_fin_imp_020` | conditional vs factual | **confirmed conditional / refinement.** 85,000 vs 1m are different quantities under different scopes, not one quantity asserted twice. Matches real FSCS temporary-high-balance rules. |
| `wp1_fin_red_049` | redundant vs refinement | **confirmed redundant.** Nested scope, identical outcome (2.99%), resolves to one merged branch — and does, in the encoding. The most consequential case in the set, and it is right. |
| `wp1_imm_red_050` | redundant vs refinement | **confirmed redundant.** Same shape, one merged branch. |
| `wp1_fin_opp_053` | opposed vs refinement | **relation confirmed, encoding fixed (§1).** Reward × Jan-2024 genuinely cross. |
| `wp1_imm_opp_054` | opposed vs refinement | **REJECTED (§2).** Different predicates; not a conflict. |
| `wp1_fin_imp_009` | implicit vs explicit | **confirmed implicit.** "Our Young Saver commitment covers customers aged 18 to 24, whose accounts operate without a minimum balance" — a positive product statement carrying no exception marker. |
| `wp1_fin_imp_005` | implicit vs explicit | **confirmed implicit under the operational definition; see §3** for what that definition actually means. |
| `wp1_imm_imp_021` | implicit vs explicit | **confirmed, same ruling as 005.** Decided once and applied to both, as the flag asked. |
| `wp1_fin_imp_025` | reversed polarity | **confirmed refinement** *within the query frame* ("charges for using my card in the UK"). Outside that frame quasi-cash and sterling-UK merely overlap. The frame is doing real work — see §5. |
| `wp1_imm_imp_026` | nested default | **ESCALATE.** The default is itself conditioned ("resident in a listed country") and the exception ("living in the UK for six months") is close to *disjoint* from it, not nested. Refinement is not clearly right. Needs a human. |
| `wp1_imm_imp_016` | sub-structure | **confirmed refinement, but under-specified.** Settled status → 5 years, Swiss → 4, is a three-branch structure encoded as two. A system correctly emitting three branches would be penalised for an unmatched prediction. Either split it or set the nested multi-exception flag. |
| `wp1_fin_imp_007` | partial scope | **confirmed refinement.** The scope is a slice of the balance (first 500), not a class of customer, and both rates apply to one account at once. A genuinely good hard case. |

---

## 5. A convention that needs writing down

36 of the 40 refinement cases encode the default branch with the descriptor
`"all cases"`, when the rule passage is usually restricted — `wp1_fin_imp_008`'s
p0 covers fixed-rate loans settled early, not all loans.

This is defensible: the default is the default **within the query frame**, and
the query ("Is a fee charged for early repayment?") already restricts the
universe. It is also invisible, and it is the same assumption that produced the
§1 contradiction when carried into relations where it does not hold.

**Action:** state it in the annotation manual — *the default branch's
applicability is the whole case space **as restricted by the query**, not the
whole world* — so annotators apply it deliberately rather than by imitation, and
so the four-way relation is judged inside that frame consistently.

---

## 6. What still needs a human, ranked

1. **`wp1_imm_opp_054`** — reword or drop (§2). It is currently not a conflict.
2. **`wp1_imm_imp_026`** — refinement vs disjoint; the flag was right that this
   is ambiguous, and I could not resolve it (§4).
3. **`wp1_imm_imp_016`** — decide two branches or three (§4).
4. **The §3 wording constraint** — confirm before the paper describes the
   implicit subset.
5. **The §5 convention** — confirm and write into the manual.
6. **The other 40 cases** — do they read naturally to someone who knows UK
   banking and immigration rules? That is the question this pass cannot answer,
   and it is the one that decides whether the set is usable.

Items 1–3 are three cases. Item 6 is the real work, and with the structural
defects cleared it should be a read-through rather than an audit.


---

## 7. Human verification — decisions, 2026-09-25

All 54 cases reviewed. **50 accepted as proposed, 3 changed, 1 dropped.**

### Changed

**`wp1_imm_imp_016` — refinement, re-encoded as THREE branches**

There are three distinct outcomes, not two: two years generally, five years
with settled status, four years for Swiss nationals and their family members
holding settled status. The Swiss branch is nested inside the settled-status
branch and disagrees with it.

Two consequences, both applied:

- The `Case` schema gained `sub_condition` / `sub_outcome` /
  `sub_applicability` / `sub_attrs` so a third branch can be expressed at all.
- The instance now carries `gold_multi_exception_flags.nested = True`, and B2
  routes it to **selection with the nested flag** rather than composing it.
  That is the correct first-order behaviour: exception-to-exception precedence
  is outside this project's scope and is flagged, not guessed. Gold and B2 now
  agree on the flag.

**`wp1_imm_imp_026` — refinement → DISJOINT**

The text never establishes that someone who has lived in the UK for the six
preceding months is a *subset* of those resident in a listed country. In
practice the two populations barely meet. Refinement requires nesting, and the
passages do not support it.

Re-encoded with both branches separately scoped and **no default**, per the
v1.1.0 schema rule. `compare()` now returns `disjoint`, matching the label.

**`wp1_imm_opp_054` — DROPPED**

The two passages never contradicted each other. p0 concerns satisfaction of
*the skill requirement*; p1 concerns satisfaction of *the requirements of the
route*. A 46-year-old graduate meets the first and fails the second, both true
at once.

Dropped rather than reworded. Changing p0 to say "the requirements of this
route" would have produced a valid case, but a **different** case from the one
reviewed — and editing the evidence to fit the label is the wrong habit to
build into a benchmark.

### A bug the re-encoding exposed

Encoding `016` as three branches immediately disagreed with B2: the operator
read "the period is five years" and "the period is four years" as the *same*
outcome, so it never flagged the nesting.

`numbers_conflict` only extracted **digits**, and policy text spells small
quantities out constantly — "a continuous period of two years", "within 14
days". The two outcomes share "period" and "years", and the only tokens that
differed carried the entire meaning. Now handles spelled cardinals, including
compounds ("twenty-eight" and "28" resolve to the same figure rather than
conflicting).

This is the third bug of its kind in the matcher, after negation-as-stopword
and passage-ids-as-figures. The pattern is consistent: **two outcomes that
differ in one small token that carries all the meaning.**

### Open limitation

`OPPOSED` now has **one instance** (`wp1_fin_opp_053`). That is thin coverage of
the relation hardest to tell from refinement, and the confusion cell the paper
reports. Authoring a replacement is worthwhile before the probe is run — a
candidate can be drafted and verified the same way these were.

### Provenance

Every instance now carries `annotator_a="model-proposed"` and
`annotator_b="human-verified-2026-09-25"`. Both are kept deliberately: the first
records that a model drafted the case, which is why this set can **never** enter
a Cohen's kappa. The two passes share a starting point, so their agreement would
measure the proposal rather than the task. `benchmark.annotation.agreement`
refuses it, and a test pins that refusal against the exact string used here.
