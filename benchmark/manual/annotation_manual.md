# Annotation manual

**Status: draft by Member A. Member B must co-author before annotation begins.**

Both members label, so both must agree this document first. The
refinement-vs-opposed procedure in §4 is the part the proposal flags as the
hardest judgement in the project — draft that section together, not separately.

---

## 1. Protocol

Adopted from the highest-rigor precedent in this literature (Cattan et al.,
arXiv:2506.08500, 2025):

1. **Two annotators label independently.** No discussion, no shared notes.
2. **Disagreements are resolved by discussion** between the two.
3. **A third expert annotator performs a final review pass.**

Procedure-ambiguous refinement-vs-opposed cases go to the third reviewer
**rather than to a coin flip**. If the two of you cannot decide by the procedure
in §4, that is a signal the procedure needs improving — record the case.

### Inter-annotator agreement

Report Cohen's kappa **separately** for:

- the five-class conflict-type label (Member A's axis)
- the four-way scope-relation label (Member A's axis)
- branch structure (Member B's axis)

Never report one blended kappa. The scope-relation label is expected to be the
harder of A's two, and blending would hide exactly that.

---

## 2. Division of labour

| Label | Owner |
|---|---|
| five-class conflict type | A |
| four-way scope relation | A |
| distractor flag | A |
| branch structure `[(condition, outcome, applicability)]` | B |
| `construction` tag (Tier 1 natural / Tier 2 split) | B |
| gold scoped answer | B |
| selection answer reference | B |

Both annotate into `contract/gold.py`'s `GoldInstance`. The schema validates as
you go, so a structurally impossible annotation fails at entry rather than in
week 14.

---

## 3. The five-class conflict type

### Ask this first, before you look at the list

> **Does the second passage change the answer for anyone covered by the first?**

If **yes**, the pair is a conflict of some kind. Go to the list below and work
out which kind. **It does not matter that both passages are true**, and it does
not matter that they never contradict each other — a passage that narrows,
qualifies or carves out part of another *changes the answer* for the cases it
covers, and that is what this benchmark exists to capture.

If **no** — the passages simply do not interact — it is `no_conflict`.

This question is first because the most common labelling error is to reach
`no_conflict` too early. In the κ pilot of 2026-09-25 it accounted for 7 of 23
disagreements, the largest single cell, and **three of those seven contained an
explicit "unless" or "only if"**. Both annotators were reading carefully; the
list below was ordered in a way that invited the error.

Worked through:

> p0: "Credit interest is paid at 1.5% AER on balances up to 5,000."
> p1: "Interest is payable **only if** at least two direct debits are active in
> the calendar month."
>
> Do these contradict? No — both are true.
> Does p1 change the answer for anyone covered by p0? **Yes.** An account with
> one direct debit earns nothing, and p0 alone would tell that customer they
> earn 1.5%.
> → a conflict, and (both true, different scopes) → **`conditional`**.

> p0: "Cash advances are available up to the cash limit shown on your statement."
> p1: "Contactless payments are limited to 100 per transaction."
>
> Does p1 change the answer for anyone covered by p0? **No.** Cash advances and
> contactless payments are different things; neither qualifies the other.
> → **`no_conflict`**.

---

Now the five labels. Work through them **after** answering the question above.

**`no_conflict`** — The passages do not interact. Different subjects, or the
same subject with nothing at stake between them.

> ⚠️ "They agree" is **not** sufficient for `no_conflict`. A general rule and its
> exception agree in the sense that both are true. What makes them a conflict is
> that the second changes the outcome for part of what the first covers. If you
> are about to choose `no_conflict` for a pair where one passage qualifies the
> other, choose `conditional` instead.

**`temporal`** — They disagree **because one has replaced the other**. The rule
itself changed, and the older version is no longer in force.

> ⚠️ **A date is not enough.** Time expressions appear constantly in conditional
> instances as a *property of the case* rather than a *version of the rule*.
> This was the second-largest source of disagreement in the pilot.
>
> Ask: **is the earlier statement still in force for anybody?**
>
> - "With effect from 1 January 2022, overdrafts are 35.0% EAR" vs "With effect
>   from 1 March 2025, overdrafts are 39.9% EAR" → the 2022 rate is dead.
>   Nobody pays it now. → **`temporal`**.
> - "Loans taken out **on or after 1 April 2024** may be repaid at any time" →
>   loans taken out before that date are *still governed by the old terms*. Both
>   rules are live; the date selects which case you are in. → **`conditional`**.
> - "Closed **within 14 days**", "outside the country for **two years**",
>   "under 70 **at the start of the trip**" → durations and thresholds, not
>   versions. → **`conditional`**.
>
> Rule of thumb: a superseded rule has a *publication* date attached to the
> rule. A conditional rule has a date attached to *the case*.

**`opinion`** — They express differing judgements rather than differing facts.
"Most advisers consider X the simplest option" vs "practitioners regard X as
restrictive." Neither is checkable.

> ⚠️ **If either passage contains a figure the two disagree on, it is not
> `opinion`.** "The fee is 1,500" against "the fee is 1,846" is checkable, so it
> is `factual` — whatever else is going on in the sentence. Two pilot labels
> went astray here, and `3` (temporal) and `4` (opinion) are adjacent keys.

**`factual`** — Both passages make the same kind of claim about the same
situation, and they disagree. **One of them is wrong.** A 3% fee and a 5% fee
for the same transactions, same time, same customers.

**`conditional`** — **Both passages are TRUE, but they apply to different
scopes.** One states a general rule, the other a narrower case with a different
outcome.

### The test that separates factual from conditional

> **Can both passages be true at the same time, for different cases?**
>
> If yes → `conditional`. If no → `factual`.

This is the hardest and most consequential distinction in the manual, and it is
the one the paper reports as its own confusion cell. Apply the test literally.
Do not reason from which passage sounds more authoritative — credibility is
irrelevant to this label.

Worked example:

> A: "International transactions incur a 3% fee."
> B: "Fees are waived for premium-tier cardholders."
>
> Can both be true at once? **Yes** — for a standard cardholder A holds; for a
> premium cardholder B holds. → `conditional`.

> A: "International transactions incur a 3% fee."
> B: "International transactions incur a 5% fee."
>
> Can both be true at once? **No** — same transactions, same customers, two
> different numbers. → `factual`.

---

## 4. The four-way scope relation

**Only for `conditional` pairs (and borderline `factual` ones).**

Decide **two things separately**. Do not collapse them — this is the whole
design:

### Scopes are judged inside the query frame

Before step 1: **the universe is the set of cases the query is about, not the
whole world.** The default branch's applicability is "everything" *as the query
restricts it*.

For "Is a fee charged for early repayment?" against "An early repayment charge
applies where a fixed-rate loan is settled before the end of its term", the
rule passage is the default — even though it is visibly restricted to
fixed-rate loans settled early — because the query has already narrowed the
universe to early repayment.

Apply this deliberately, and check it still holds before you use it. Carried
into a relation where it does *not* hold, it silently forces the refinement
shape: if one branch covers the whole universe, every other branch is nested
inside it, and nested-with-differing-outcomes is refinement by definition. Four
cases in the WP1 probe set were encoded that way and came out labelled
`opposed` or `disjoint` while their structure said `refinement`.

**A branch is only the default if nothing in the query frame falls outside it.**
Where two branches each cover part of the frame — `disjoint` and `opposed` —
there is no default, and the schema permits an instance with none.

### Step 1 — Do the applicability sets overlap?

Write down who or what each passage applies to. Then:

- **No overlap at all** → `disjoint`. Stop here; outcome does not matter,
  because no case falls under both.
- **One is entirely inside the other** → nested. Go to step 2.
- **They intersect but neither contains the other** → overlapping. Go to step 2.

### Step 2 — Where they overlap, do the outcomes agree?

| Step 1 | Outcomes AGREE | Outcomes DISAGREE |
|---|---|---|
| no overlap | `disjoint` | `disjoint` |
| nested | **`redundant`** | **`refinement`** |
| overlapping | **`redundant`** | **`opposed`** |

### The four labels, in words

**`refinement`** — One scope is strictly inside the other AND the outcomes
differ there. A genuine exception: it narrows the default without contesting it
outside that narrower scope. *The fee/premium-card case.* → composed.

**`disjoint`** — The scopes do not intersect. Two unrelated rules that happened
to be retrieved together. → composed trivially.

**`redundant`** — The scopes overlap or nest, but the outcomes **agree**
throughout the overlap. A restatement, not an exception. → merged into one
branch.

> **This is the label most often got wrong, and the error is expensive.** A
> passage that repeats the general rule for a specific group has nested scope,
> exactly like a real exception. Only the outcome tells them apart. Labelling it
> `refinement` makes the system compose a branch that does not exist.
>
> "International transactions incur a 3% fee" + "Premium cardholders are charged
> 3% on international transactions, in line with the standard schedule" is
> `redundant`, not `refinement`. Same outcome.

**`opposed`** — The scopes overlap, **neither contains the other**, and the
outcomes disagree on the overlap. A real contradiction. → falls back to
selection.

### The refinement-vs-opposed decision procedure

When step 1 is unclear, ask:

> **Can the exception's condition co-occur with a case that FAILS the default's
> applicability?**
>
> - **No** — every case satisfying the exception also satisfies the default →
>   the exception's scope is nested inside the default's → `refinement`
>   (if outcomes differ on the overlap).
> - **Yes** — some cases satisfy one but not the other → the scopes are not
>   nested → re-examine as `disjoint`, `redundant`, or `opposed` depending on
>   outcome agreement.

If you still cannot decide, **route it to the third reviewer**. Do not guess.
Ambiguous cases that get coin-flipped are worse than ambiguous cases that get
escalated, because they enter the corpus with false confidence and cannot be
found again.

---

## 5. Distractors are mandatory

Every batch must include:

- genuine **factual** conflicts
- genuine **temporal** conflicts
- **disjoint** non-conflicting pairs
- **redundant** restatements
- **no-conflict** instances

Without them, the Spurious-Condition Rate has no denominator and the
factual/conditional confusion cell cannot be measured at all. A corpus of only
positive examples cannot test the thing the paper claims.

---

## 6. Tier 1 vs. Tier 2 (Member B tags this, both should understand it)

**Tier 1 — `construction: natural`.** The general rule and its exception were
found in **separately retrievable documents**: a provider's primary terms page
and a separate amendment, FAQ, or category page. This is the primary evidence
for the multi-source claim that distinguishes this benchmark from ConditionalQA.

**Tier 2 — `construction: split`.** A single-document rule/exception pair
deliberately split across two synthetic "documents". Legitimate and useful — it
isolates the resolution-operator question from the mining question — but it is
**not** evidence of natural multi-source occurrence.

Record `document_id` on every passage. The schema checks that a `natural`
instance's passages do not all share one document, so a mislabelled Tier-2
instance fails loudly rather than quietly inflating the headline count.

**Never blend the two tiers in a headline number.**

---

## 7. Common mistakes

| Mistake | Why it is wrong |
|---|---|
| Labelling by which source seems more trustworthy | Credibility is irrelevant to the conflict type. That is the whole point — selection by credibility is the failure being measured. |
| Calling a restatement a `refinement` | Fabricates a branch. Check the **outcome**, not just the scope. |
| Calling a real contradiction `conditional` | Routes to composition, which presents two incompatible claims as both true. |
| Treating retrieval order as meaningful | Branch roles come from applicability, not order. The wider scope is the default, whichever passage came first. |
| Averaging explicit and implicit conditions | They are reported separately. Implicit recovery is what the method is judged on. |
| Recording an exception with no supporting passage | An ungrounded exception is what the grounding gate exists to reject. It cannot be gold. |

---

## 8. Before you start

```bash
python -m contract.mock --n 20 --seed 0 -o practice.jsonl --coverage
```

Generates practice instances covering all five types, all four relations, and
both multi-exception patterns. Label them independently, compare, and calibrate
on these before touching real data. The pilot (~30 instances) exists to measure
kappa on the refinement-vs-opposed boundary specifically — if it is low, fix
this manual before annotating the corpus, not after.
