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

Work through these in order.

**`no_conflict`** — The passages are about different subjects, or they agree.
Most retrieved pairs are this. Do not force a conflict label onto a pair that is
merely adjacent in topic.

**`temporal`** — They disagree *because they describe different points in time*.
One has been superseded. Look for dates, "effective", "as of", "no longer".
A temporal conflict has a right answer: the current one.

**`opinion`** — They express differing judgements rather than differing facts.
"Most advisers consider X the simplest option" vs. "practitioners regard X as
restrictive." Neither is checkable.

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
