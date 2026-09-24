# WP1 — Tier-1 mining pilot: findings

**Status: complete. Gating checkpoint answered.**
**Date: 2026-09-24 · Member A**
**Two domains: 32 gov.uk documents / 17 pairings, and 21 banking documents / 11 pairings.**

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

**Tier-1 pairs are scarce in both domains. Commit to the Tier 1 / Tier 2
split.** The proposal's §8 scarcity risk is confirmed, not avoided.

Two refinements the pilot added, both actionable:

1. **Banking is the better Tier-1 domain by a wide margin** — 91% of banking
   pairings are genuinely separate documents against 29% for gov.uk. Make
   financial terms the primary Tier-1 source.
2. **The lexical miner was under-implementing the proposal.** §6.2 specifies
   that *a model* proposes candidate pairs. Swapping the cue-and-overlap miner
   for an LLM proposer on the same documents took cross-document pairs from
   1 to 4, and three of those four look genuinely right.

Neither refinement rescues Tier-1 abundance. Both change how WP2 should run.

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

## Domain 2 — financial terms (the comparison that mattered)

Same method, 11 pairings across NatWest, HSBC, Barclays and Lloyds: product
pages plus separate charges schedules, and four PDFs. `robots.txt` is checked
per host before every fetch; one URL pointed at an origin-server hostname that
disallows crawling and was skipped rather than fetched.

### Document separation — banking wins decisively

| | gov.uk | banking |
|---|---|---|
| distinct-document pairings | **5 / 17 (29%)** | **10 / 11 (91%)** |
| same document | 10 / 17 | 0 / 11 |
| broken or blocked | 2 / 17 | 1 / 11 |

Banks genuinely publish the rule and the charge in separate documents — a
product page and a rates-and-charges page, or a terms PDF and a fee-information
PDF. gov.uk publishes one guide with parts. That is a structural difference,
and it is the strongest argument for making financial terms the primary Tier-1
domain.

### Pair density — neither domain is abundant

Separation is necessary, not sufficient. The documents are separate; the
question is whether a *rule* in one has a *scope-narrowing exception* in the
other.

| method | cross-document pairs found (11 banking pairings) |
|---|---|
| lexical miner (cues + word overlap) | 1 |
| LLM proposer | **4** |

Three of the four look genuinely right on inspection:

- *"Removing foreign transaction fees for Reward and Premier Reward debit
  cards"* against *"Non-Sterling Transaction Fee 2.75% of transaction"* — the
  fee applies generally and is waived for two named tiers.
- *"You can only have one Reward Platinum account in your sole name"* against
  *"You can have more than one joint Reward Platinum account…"* — a clean
  refinement on account-holding capacity.
- *"With a Reward Silver account, you won't pay any non-sterling transaction
  fees when paying with your card"* against *"Not valid for ATM withdrawals."*
  — an **implicit** narrowing, exactly the hard case Contrastive Scope Probing
  targets.

The fourth pairs a real rule against a fragment of a rates table and is noise.

### Why the lexical miner missed them

A product page and a charges schedule are written in different registers and
share few content words, so a bag-of-words bridge fails even where the pairing
is obvious to a reader. The proposal always specified a model proposer; the
lexical miner was the under-implementation, and this measured the gap rather
than assuming it.

**One tuning note worth keeping.** Shrinking the model's input window from
9,000 to 5,500 characters per document fixed a rate-limit problem and made the
results *worse* — the proposer stopped finding fee-waiver pairs and started
returning October price changes, because truncation had removed the section
where the rules live. Pacing the calls is the right lever; truncation is not.

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

1. **Candidates, not instances.** Every pair above is unverified. A human
   confirms each before it counts.
2. **Small samples.** 17 and 11 pairings — enough to answer a go/no-go, not
   enough to estimate a yield precisely.
3. **Four banks, one government.** Publishers have house styles, and the
   91%-versus-29% separation gap is partly a fact about CMS architecture
   rather than about policy documents generally.
4. **High recall by design.** Both miners over-generate for human
   verification. Precision is deliberately not the target.
5. **The LLM proposer ran on the BULK tier** — the smallest model in the tier
   map. A stronger model would likely propose more and better candidates, so
   4 is a floor, not a ceiling.

---

## What this changes

1. **Commit to the Tier 1 / Tier 2 split now.** It is a named construction
   method in the proposal, not a fallback, and the pilot says it is needed.
   State the composition explicitly in the paper.
2. **Do not let Tier 2 carry the naturally-occurring claim.** Report the two
   tiers separately in every headline number, as §6.2 requires.
3. **Make financial terms the primary Tier-1 domain.** 91% document separation
   against gov.uk's 29%. Keep immigration as the second domain the proposal
   requires, but do not expect Tier-1 volume from it.
4. **Switch WP2 mining to the LLM proposer.** Four times the lexical miner on
   identical documents, and it is what §6.2 specified all along.
   `benchmark/mining/llm_proposer.py` is written, paced and cached.
5. **Decide how to report same-guide pairs.** A real third category: either a
   new `construction` value or a labelled subset of Tier 2 — not silence.
6. **Revisit the 150-instance Tier-1 floor.** At roughly 0.4 plausible pairs
   per curated pairing, reaching 150 needs on the order of 375 hand-picked
   document pairs. Tight, and far more realistic than gov.uk's ~1,200 — but it
   should be planned, not assumed.
