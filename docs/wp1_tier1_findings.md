# WP1 — Tier-1 mining pilot: findings

**Status: complete. Gating checkpoint answered.**
**Date: 2026-09-24 · Member A · 32 documents, 17 candidate pairings, all gov.uk**

---

## The question

> How often does a general rule and its valid exception actually live in two
> **separately retrievable documents**?

This gates the corpus design. "Naturally occurring, multi-source" is the claim
that distinguishes this benchmark from ConditionalQA, which is a
single-document task. If providers overwhelmingly state a rule and its
carve-out in one document, the framing cannot be satisfied at the 250-instance
target and the corpus has to be built as an explicit Tier 1 / Tier 2 mix.

## Answer

**Tier-1 pairs are scarce. Commit to the Tier 1 / Tier 2 split.**

The proposal's §8 scarcity risk is confirmed, not avoided.

---

## Method

34 URLs across 17 candidate pairings, fetched through the **gov.uk Content
API** (`https://www.gov.uk/api/content/<path>`) rather than scraped. The API
is the publisher's own interface, returns clean structured text, and — the
reason it matters here — returns a `content_id`.

Reproduce with:

```bash
python -m benchmark.mining.fetch_govuk --pairs benchmark/data/govuk_pairs.txt \
    -o benchmark/data/govuk_docs.jsonl
python -m benchmark.mining.tier1_pilot --docs benchmark/data/govuk_docs.jsonl \
    --sensitivity
```

---

## Finding 1 — most "pairs" are one document wearing two URLs

| Verdict | Count | Meaning |
|---|---|---|
| **distinct** | **5 / 17** | two different content items — genuine Tier-1 candidates |
| same guide | 10 / 17 | two *parts* of one gov.uk guide |
| broken | 2 / 17 | one 404, one redirect to the other member of the pair |

A gov.uk **guide** is one CMS document with many parts, each with its own URL.
`/student-visa` and `/student-visa/money` look like two documents in a browser
and share a single `content_id`. Ten of the seventeen pairings were this.

**This would have been invisible to URL-based mining.** Comparing URLs counts
guide parts as two documents and inflates the naturally-occurring claim by
roughly 3×. The miner now compares `content_id`, and the fetcher records it
precisely so this check is possible.

The five genuine pairs: `student_rules`, `graduate`, `innovator_settlement`,
`youth_mobility_ballot`, `global_talent_tech`.

## Finding 2 — a third category the Tier 1 / Tier 2 binary does not cover

"Two parts of one guide" is neither Tier 1 nor Tier 2:

- **Not Tier 1** — one document, one author, one editorial act.
- **Not Tier 2** — nothing was artificially split. A RAG chunker indexes the
  parts separately and a retriever can genuinely surface one without the other,
  which is the failure mode this project studies.

It sits between them and deserves reporting as its own thing. Folding it into
Tier 1 overstates natural occurrence; folding it into Tier 2 understates how
real the retrieval separation is. `CandidatePair.same_guide_parts` tracks it.

**Recommendation:** add a third `construction` value in a future contract
version, or report it as a labelled subset of Tier 2. Do not silently absorb it
into either.

## Finding 3 — the yield, and how much it depends on a threshold I chose

At the default topic-overlap threshold: **4 Tier-1 candidates from 32
documents = 0.12 per document.** Reaching the 150-instance Tier-1 floor at that
rate needs roughly 1,200 documents, which is not realistic in WP2.

Because the threshold is a judgement call, here is the full sweep:

| overlap threshold | candidates | Tier 1 | same doc | Tier-1 share | per doc |
|---|---|---|---|---|---|
| 0.30 | 24 | 2 | 22 | 8.3% | 0.06 |
| 0.25 | 44 | 3 | 41 | 6.8% | 0.09 |
| 0.20 | 60 | 3 | 57 | 5.0% | 0.09 |
| **0.18** | **74** | **4** | **70** | **5.4%** | **0.12** |
| 0.15 | 83 | 4 | 79 | 4.8% | 0.12 |
| 0.12 | 97 | 7 | 90 | 7.2% | 0.22 |
| 0.10 | 108 | 10 | 98 | 9.3% | 0.31 |
| 0.08 | 113 | 15 | 98 | 13.3% | 0.47 |
| 0.05 | 120 | 21 | 99 | 17.5% | 0.66 |

**The count is tunable (2–21). The share is not (5–18%).** At every setting the
same-document pairs outnumber the Tier-1 pairs by between five and fifty times.
That ratio is the robust finding, and it does not depend on where the threshold
is put.

Even at the loosest, noisiest setting — 0.66 Tier-1 candidates per document —
the floor needs ~230 documents, and those candidates are *unverified
candidates*, not instances.

---

## Three harness bugs found and fixed on the way

The first run of the pilot reported **939 Tier-1 candidates and 29.34 per
document**, with a recommendation to proceed with Tier 1 as the primary corpus.
That number was wrong, and it is worth recording why, because the harness had
passed its synthetic demo cleanly.

1. **Cross product.** Every rule sentence was paired with every
   exception-looking sentence from the same provider. On a 63,000-character
   statutory appendix that is thousands of combinations, and the samples showed
   one rule sentence repeated against dozens of unrelated "exceptions". Fixed
   with a topic-overlap requirement and a cap per rule.
2. **Sentence splitting.** gov.uk puts headings on their own line with no
   terminal punctuation, so splitting on punctuation glued them to the
   following sentence (*"When to apply When you can apply depends on…"*). Now
   splits on newlines too, with a length floor and a navigation filter — *"the
   register of licensed student sponsors can be found at www.gov.uk/…"* was
   being counted as an exception because it contains the word "student".
3. **Weak cues counted as exception markers.** `"subject to"`, `"where the"`,
   `"if the"` appear in almost every sentence of this register. Now only strong
   cues (`except`, `unless`, `does not apply`, …) mark an exception.

A fourth, unrelated: five regex word boundaries in the miner had been written
as literal backspace bytes (`\x08`) by a shell heredoc, so those patterns
silently never matched. Repaired, and a repo-wide scan confirmed no other file
was affected.

**None of this changed the conclusion** — after all four fixes the answer is
still "Tier-1 pairs are scarce" — but the first run would have produced exactly
the inflated number the project's honesty guardrail exists to prevent.

---

## Caveats

1. **One domain.** All 34 URLs are gov.uk immigration. The proposal specifies
   **two** domains, and financial terms are untested. Banks may well yield
   better: they publish separate product pages, standalone fee schedules and
   dated amendment notices, which is a more Tier-1-friendly structure than a
   government guide with parts. **Worth testing before finalising the corpus
   plan** — it could change the mix.
2. **Candidates, not instances.** The 4 are unverified. A human has to confirm
   each is a genuine rule/exception pair before it counts.
3. **One publisher.** gov.uk has a house style. Other publishers structure
   differently, and the 5/17 distinct-document ratio is partly a fact about
   gov.uk's CMS rather than about policy documents generally.
4. **High recall by design.** The miner over-generates for human verification.
   Precision is deliberately not the target.

---

## What this changes

1. **Commit to the Tier 1 / Tier 2 split now.** It is a named construction
   method in the proposal, not a fallback, and the pilot says it is needed.
   State the composition explicitly in the paper.
2. **Do not let Tier 2 carry the naturally-occurring claim.** Report the two
   tiers separately in every headline number, as §6.2 requires.
3. **Test the financial-terms domain before fixing the corpus plan.** This is
   the cheapest remaining action that could still move the answer.
4. **Decide how to report same-guide pairs.** They are a real third category
   and the honest options are a new `construction` value or a labelled subset
   of Tier 2 — not silence.
5. **Revisit the 150-instance Tier-1 floor.** On this evidence it is not
   reachable from gov.uk alone within WP2.
