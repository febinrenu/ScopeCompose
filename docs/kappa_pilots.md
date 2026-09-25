# κ pilots

## Pilot 1 — 2026-09-25

**Annotators:** johann, febin. **Batch:** the 54 WP1 probe instances.

| axis | κ | band | n | 95% CI | verdict |
|---|---:|---|---:|---|---|
| five-class conflict type | **0.326** | fair | 54 | [0.168, 0.484] | **fails 0.61** |
| four-way scope relation | **0.775** | substantial | 23 | [0.395, 1.000] | passes, **provisionally** |

**Do not start bulk annotation yet.** The manual has been revised in response;
the pilot must be re-run on a fresh batch.

The scope relation — expected to be the harder axis — is the one that held. The
five-class type failed, and it failed in a concentrated, diagnosable way, which
is the outcome a pilot is for. Finding this on 54 instances cost an afternoon.
Finding it on 300 would have cost the corpus.

---

## 1. The largest cell was caused by the manual's ordering

`conditional → no_conflict`, **7 of 23 disagreements**. johann said conditional;
febin said no conflict.

Three of the seven contain an **explicit** exception marker:

> `exp_030` — "Credit interest is paid at 1.5% AER" / "Interest is payable
> **only if** at least two direct debits are active"
>
> `exp_031` — "Cash advances are available up to the cash limit" / "Cash
> advances are **not permitted unless** the account has been open 60 days"
>
> `exp_036` — "Worldwide travel insurance is included" / "Cover is **not
> provided unless** the account holder is under 70"

These are the *easiest* cases in the set. A conceptual failure on the hard
boundary would not touch them, so the cause is not that conditional-vs-factual
is subtle. It is the manual.

**The mechanism.** §3 said "work through these in order" and listed
`no_conflict` first:

> **`no_conflict`** — The passages are about different subjects, **or they
> agree**. Most retrieved pairs are this. **Do not force a conflict label** onto
> a pair that is merely adjacent in topic.

For "interest is paid at 1.5%" against "interest is payable only if two direct
debits are active", both statements *are* true, they *do* agree in the ordinary
sense, and the reader has just been warned not to force a conflict label. They
stop at the first entry and never reach `conditional`.

The discriminating test existed only for factual-vs-conditional, further down
the page — a test the annotator never gets to.

**This is the project's central conceptual move, and the manual assumed it
rather than stating it:** a passage that qualifies another is a conflict *even
though nothing contradicts*. That sentence was nowhere in the document.

**Fix applied.** §3 now opens with the discriminating question, before the list:

> **Does the second passage change the answer for anyone covered by the first?**

with both worked directions, and an explicit warning on the `no_conflict` entry
that "they agree" is not sufficient.

---

## 2. `temporal` fires on any time expression

Six disagreements involve `temporal`, and the pattern is that **a date or
duration anywhere in the text pulls the label**, regardless of what the date is
doing:

| case | the time expression | what it actually is |
|---|---|---|
| `imp_008` | "loans taken out on or after 1 April 2024" | a property of the **case** |
| `imp_019` | "closed within 14 days" | a duration condition |
| `imp_016` | "two years / five years / four years" | duration thresholds |
| `opp_053` | "accounts opened from January 2024" | a property of the **case** |

The old manual entry invited this — it said *"Look for dates, 'effective', 'as
of', 'no longer'"*, which is a keyword instruction, and keywords are exactly
what misfires here.

**Both annotators fell into it**, in opposite directions: febin called `imp_008`,
`imp_019` and `imp_016` temporal; johann called `opp_053` temporal. So this is
the manual, not one reader.

**Fix applied.** The `temporal` entry now turns on a question rather than
keywords:

> **Is the earlier statement still in force for anybody?**

with the rule of thumb that a superseded rule has a date attached to *the rule*,
while a conditional rule has a date attached to *the case*.

---

## 3. Two labels look like a keypress, not a judgement

`imm_fact_040` ("the application fee is 1,500" / "is 1,846") and `imm_temp_042`
(salary threshold 26,200 / 38,700) were both labelled **`opinion`** by johann.

`opinion` means neither passage is checkable. Two passages quoting figures that
disagree are checkable by definition, so the label is not merely unlikely, it is
impossible. On the menu, `3` is temporal and `4` is opinion.

**Fix applied in the tool, not the manual.** The CLI now detects a numeric
disagreement between passages and, if the annotator picks `opinion`, says so and
offers factual / temporal / opinion-anyway. It reuses `numbers_conflict` from
the metric rather than a second regex, so the guard and the scorer cannot drift.
Verified: fires on both cases above, silent on `imp_001`, which has no numeric
disagreement.

---

## 4. Two things about the pilot itself

**Pace.** The medians differ by more than 3x, and the distributions differ by
more than the medians suggest:

| | median | p25 | p75 | under 10s | under 5s |
|---|---:|---:|---:|---:|---:|
| febin | 5.5s | 4.4s | 8.2s | **41 / 54** | **18 / 54** |
| johann | 20.3s | 18.2s | 30.8s | 3 / 54 | 3 / 54 |

(johann's *mean* of 171s is meaningless — one instance sat open for two hours,
plainly a break. The median is the statistic to read.)

Fast is not the same as careless, and the manual defects above are real
independent of who was reading. But 18 instances judged in under five seconds is
difficult to reconcile with reading a query, two passages, and applying a
two-step test to them — and 41 of 54 under ten seconds is a consistent pattern
rather than a few easy cases.

Treat this as a plausible contributor of comparable size to the manual defects,
not a footnote. The re-run should be paced deliberately: there is no prize for
finishing, and a label produced in three seconds costs more later than it saves
now.

**The batch was not fresh for one annotator.** johann verified all 54 of these
cases during the probe-set review earlier the same week. His pass was blind to
febin's labels, so κ remains a valid inter-annotator statistic — but he was
re-labelling material he had personally adjudicated days earlier, while febin
came to it cold.

Two consequences:

1. **The 85.2% vs 55.6% agreement-with-gold figures are not comparable** and
   must not be read as one annotator being more accurate. johann signed off that
   gold; agreeing with it is close to circular.
2. The asymmetry — one annotator with prior exposure, one without — could itself
   depress κ, independently of the manual.

**The re-run must use a batch neither annotator has seen.** The natural choice is
the first ~30 instances of the WP2 corpus as it is built.

---

## 5. What was changed

| change | where |
|---|---|
| discriminating question moved before the label list | `annotation_manual.md` §3 |
| `no_conflict` warning: "they agree" is not sufficient | §3 |
| `temporal` rewritten around in-force-for-anybody, not keywords | §3 |
| `opinion` warning: a disputed figure is checkable | §3 |
| numeric-disagreement guard on the `opinion` key | `annotation/cli.py` |
| caution when κ clears the bar but its interval does not | `annotation/agreement.py` |

The last one came out of this pilot too. The scope axis reported 0.775
("substantial") and *cleared* the bar, on 23 instances with an interval running
down to 0.395. The tool now says so rather than letting a point estimate carry a
conclusion its interval does not support.

---

## 6. Next

1. Both annotators re-read §3. It is the part that changed.
2. Re-run on ~30 instances **new to both**.
3. Expect the type axis to clear 0.61. If it does not, the remaining
   disagreement is conceptual rather than editorial, and the next move is the
   third-reviewer route in §1 rather than another manual edit.

The scope axis should be re-measured on more instances regardless — 23 is too
few to conclude from, whichever way it lands.


---

# Pilot 2 — 2026-09-25. Void: my error, not a result.

| axis | κ | n | verdict |
|---|---:|---:|---|
| five-class conflict type | **−0.013** | 19 | **void** |
| four-way scope relation | 1.000 | 3 | too few to mean anything |

**The batch had no queries.** The `query` field of all 19 instances read
`[to be written by the annotator]`, and I never built a way for an annotator to
write one. Both annotators judged 19 passage pairs with **no question attached**.

Every rule in the manual is relative to a question. §3 asks whether the second
passage *changes the answer for anyone covered by the first* — the answer to
what? §4 judges scopes *inside the query frame*. With no query there is no
frame, so each annotator supplied their own, and they supplied different ones.

Take `wp2_0004`:

> p0: "You may be able to apply for settlement once you've been in the UK for
> 3 years."
> p1: "You must have a new endorsement that shows you've met the requirements
> for growing your business"

Under *"can I apply for settlement after 3 years?"* p1 adds a requirement →
conditional. Under *"what do I need for settlement?"* they are complementary →
no conflict. Both readings are correct. Nothing in the instance chose between
them.

**The evidence that this is framing and not disagreement** is in the direction
of the cells. Pilot 1 was one-directional — 7 `conditional → no_conflict`, none
the other way, which is two readers applying different rules consistently.
Pilot 2 ran **both ways**: 9 one direction, 5 the other. That is not a reading
difference; that is two people framing each instance independently.

So pilot 2 says nothing about the manual revisions, and nothing about whether
the task is harder on real text than on the probe set. Both remain untested.

## A second error, which hid the first

The reported figure was **n = 73**, not 19 — 54 pilot-1 instances plus 19
pilot-2 ones, pooled. `agreement` compared whole annotator passes, and passes
accumulate across batches.

The pooled number was 0.245, close enough to pilot 1's 0.326 to read as "no
improvement". The per-batch reality was −0.013, and pilot 1's 54 instances were
dragging it upward. A failing batch was hidden behind a passing one.

## What changed

| fix | where |
|---|---|
| queries generated for every instance, barred from hinting at the relationship | `mining/build_batch.py` |
| an instance with no query is **refused**, not labelled | `annotation/cli.py` |
| `agreement --prefix wp3_` scopes to one batch | `annotation/cli.py` |
| without `--prefix`, the header says "ALL batches pooled" | `annotation/cli.py` |

Query generation is a model call, and that is a considered choice rather than a
convenience. A question is not a label: it fixes what is being asked without
saying what the answer is, and what κ needs is that **both annotators share one
frame**. The prompt is barred from words that leak a relationship, and the
filter rejected 1 of 19 for trying.

## Pilot 3 — ready, not yet run

`benchmark/data/pilot3_batch.jsonl`. Same 19 mined instances, now with queries:

> `wp3_0000` — "Do I need to pay the healthcare surcharge?"
> `wp3_0001` — "Can I extend my graduate visa?"
> `wp3_0003` — "Do I need to prove income for visa application?"

18 labellable; one refused by the leak filter and skipped by the tool.

Run with `--prefix wp3_`. **Pilot 2's labels stay on disk** — they are evidence
of what unqueried instances produce, and deleting them would erase the record
of the mistake.
