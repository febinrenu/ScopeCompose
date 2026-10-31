# Review guide: the five things only people can decide

This guide covers the fast review loop (`python -m benchmark.review`). It complements
`annotation_manual.md` (conflict type, scope relation, the refinement-vs-opposed procedure), which still
governs the *definitions*. Here: what each screen asks, how to decide in seconds, and the traps.

**Time is the constraint** (about 25-40 hours between two people), so every screen is one key press, the
time per decision is recorded, and nothing is ever "accept all". Do the tasks in this order; if time runs
out, cut from the bottom (never below the S1 floor in step 1).

| # | Kind | Question | Budget | Cut order |
|---|---|---|---:|---|
| 1 | `candidates` | Is this a genuine rule + exception from two documents? | ~2 min each | never below 150 |
| 2 | `branches` | Is the proposed branch structure right? | ~5 min each | - |
| 3 | `personas` | Does each statement put the person in the right branch? | ~1 min each | - |
| 4 | `vacs` | Do these two passages govern the same decision? | ~1.5 min each | to 100 pairs |
| 5 | `entailment` | Does this answer, by itself, establish the claim? | ~0.4 min each | to 150 |

## Ground rules (all five)

1. **Work alone.** Two annotators write to two files and never see each other's verdicts. Agreement is
   computed afterwards; the tool refuses the same person twice.
2. **Use your own name** as `--annotator`. A model name is refused by the agreement code.
3. **Judge from the text on the screen only.** No searching, no outside knowledge, no "I know how this
   scheme works". The benchmark asks what the *passages* establish.
4. **Skip is allowed, guessing is not.** `skip` means the item is missing data; it is never counted as a
   label. If you skip more than one in ten, tell the other annotator: the item may be broken.
5. **Models propose, you decide.** Pre-annotated items carry provenance
   (`branches=model-proposed` until a person reviews them). Your verdict is what makes a label gold.
6. **Take breaks.** Fatigue shows up as drift: the tool records time per decision, and a run of
   2-second accepts is read as a problem, not as speed.

Run it like this (one terminal per annotator, resumable at any time):

```
python -m benchmark.review --kind candidates review --items <file> --annotator <you> [--limit 50]
python -m benchmark.review --kind candidates status <you>
python -m benchmark.review --kind candidates agreement <annotator_a> <annotator_b>
```

---

## 1. `candidates`: is it a real exception, from two documents?

You see a RULE (document id), an EXCEPTION (document id), the model's reason and confidence.

**Accept** only if all three hold:

- the exception *narrows* the rule: it applies to some group that the rule also covers (a subset, not a
  different topic);
- the **outcome differs** for that group (a different figure, yes versus no, a different duty);
- the two sentences come from **different documents**, and the exception is not simply the rule restated.

**Reject**, and pick the reason that is actually true:

| reason | when |
|---|---|
| `not_an_exception` | the second sentence does not narrow the first; unrelated topics, or a different group with no overlap |
| `same_outcome` | it narrows the group but the outcome is the same ("students and everyone else both pay 3%") |
| `not_cross_document` | both sentences really come from one document, or two parts of one guide |
| `unclear` | you cannot tell what either sentence means in isolation |

Traps:

- **"Usually" is not an exception.** "Fees are normally 3%" and "the fee may be 3.5%" differ in figure but
  neither narrows a group; reject as `not_an_exception`.
- **A different region of one number line is not a group** unless the passage says who falls in it.
- **Two documents about different schemes** with similar wording (two banks' fee schedules) are not a rule
  and an exception; they are two rules. Reject as `not_an_exception`.
- The model is confident about plenty of bad pairs. Confidence is a sort order, not evidence.

## 2. `branches`: the gold structure for an accepted pair

You see the query and each branch: `default` or `exception`, its condition, its outcome and the passage
that supports it.

**Accept** if: exactly one default (no condition), each exception's *condition* names the group in words a
person could check, each *outcome* is what the passage says for that group, and the supporting passage
really says it.

**Edit** (type the corrected text; Enter keeps the model's) when the structure is right but a string is
wrong. Edit the *outcome* to what the passage states for that group, not what you would conclude; edit the
*condition* so it reads as a checkable fact about the person ("the holder has a premium-tier card"), never as
a conclusion ("they are exempt").

**Reject** if the structure is wrong: the default and exception are swapped, a branch is invented, or the
passage does not support the outcome. Pick `unclear` if it cannot be repaired from the screen.

Traps: outcomes must be **value-bearing** (a figure, a yes/no, a period), because case-complete accuracy
matches them as values; "a fee may apply" is not an outcome.

Write each outcome as a short, plain statement a reader could repeat in different words. The matcher treats
these as the same outcome, so they are safe: "a 3% fee applies" / "3% fee is incurred"; "no fee applies" / "the
fee is waived"; "work is allowed" / "work is permitted"; "you do not need to pay the surcharge" / "the surcharge
is not payable". It cannot compare outcomes that carry a qualifier or that are not statements: prefer "work is
allowed" to "work in most jobs is allowed", and "work is not permitted" to "no employment". When the exception
changes only a figure or a period, put the figure in the outcome ("the period is four years"). A condition naming two things ("premium and
over 65") is one branch with one conjunctive condition, not two branches.

## 3. `personas`: statements that put a person in a branch

Each persona is a first-person statement ("I hold a standard card and I am sending a transfer") that must
let a careful reader place the person in exactly one branch: a **default** person, one person per
**exception**, and an **irrelevant** statement that mentions an attribute that matters to *neither* branch.

**Accept** if each statement (a) clearly belongs to its branch from the statement alone, (b) does not name
the outcome, (c) does not contain the exception's trigger when it is the default person, and (d) the
irrelevant one really changes nothing.

**Edit** the offending statement; do not paraphrase the ones that are fine.

**Reject** if a statement cannot be repaired (it is ambiguous between branches).

The validator problems printed under a persona are model-checked hints (for example, a default statement
that contains the exception's keywords). They are advice, not a verdict.

Why this matters: the counterfactual pair (same answer, two people) is what shows whether a reader is using
the answer or its own world knowledge. A leaky statement quietly makes the test easy.

## 4. `vacs`: do two passages govern the same decision?

You see A and B. Choose:

| label | meaning |
|---|---|
| `equivalent` | same decision variable at the same generality: one passage could replace the other |
| `a_broader` | A states the more general rule (the decision B governs is a special case of A's) |
| `b_broader` | the reverse |
| `distinct` | different decision, even if the words overlap (a late-payment fee versus a transfer fee) |

Ask "what question does each passage answer?" before comparing wording. Passages that share vocabulary but
answer different questions are the whole point of this test, so be suspicious of word overlap, and of
passages that *sound* alike but name different quantities. If A is about "a fee" and B about "an annual
fee", B is the narrower decision: `a_broader`.

## 5. `entailment`: the reader audit (gate V2)

The most careful screen, and the fastest. You see a QUESTION, an ANSWER, a PERSON and a CLAIM. You are
**blind** to which system wrote the answer and to what the reader said.

> Does the ANSWER, by itself, establish the CLAIM for this person?

- **yes** if the answer states the outcome for a group the person is in, plainly (including through the
  general rule when nothing in the answer carves the person out).
- **no** if the answer lists outcomes without saying for whom, hedges ("may", "could"), says nothing, or
  gives the person the wrong group's outcome.
- **unsure** if the answer's meaning is genuinely ambiguous. Unsure is excluded from the truth rather than
  guessed.

Do **not** use outside knowledge. An answer that says "a 3% fee applies" says nothing about a premium
cardholder even if you know premium cards are usually exempt. Judge as if the answer were the only
document in the world.

This audit sets the entailment threshold and decides whether strict case-complete accuracy may be reported
as the headline. Its precision bar (0.95) is high on purpose: a false "yes" from the pipeline inflates every
number in the paper.

---

## Adjudication

After both annotators finish a kind, run `agreement`. For every disagreement, discuss and record the
resolution; a case still undecided after discussion goes to the third reviewer, not to a coin flip.
Report kappa separately per kind. Record how many hours each person spent (`status`), so "annotation hours"
in the datasheet is a measurement.

---

## After reviewing: getting verdicts into the corpus

Review items for the current corpus are ready in `benchmark/data/review_items/`
(`candidates.json`, `branches.jsonl`, `personas.jsonl`). Only a person's `accept` or `edit` changes the data.

```
python -m benchmark.human_loop apply-branches --items benchmark/data/review_items/branches.jsonl \
    --annotator <you> [--second <other>] --out benchmark/data/review_items/branches.applied.jsonl
python -m benchmark.human_loop apply-personas --data benchmark/data/corpus.jsonl \
    --items benchmark/data/review_items/personas.jsonl --annotator <you> --out <applied.jsonl>
python -m benchmark.human_loop merge --data benchmark/data/corpus.jsonl --updates <applied.jsonl> --out <new corpus>
```

The first annotator's edits are the ones applied. A second annotator can only hold an item back: if they have
not decided it, or rejected it, the item is listed as disputed and not applied. An edited persona set is
re-checked and refused (with the reason) if it breaks a rule, for example by stating the outcome. `merge` never
overwrites its input. Apply branches before personas: personas are generated from reviewed branches.
