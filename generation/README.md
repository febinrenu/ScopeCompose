# `generation/` — Member B

**Owner: Member B.** Member A must not implement this.

## What goes here

**B3: conflict-aware generation.** Verbalize the resolved branch set into a
scoped natural-language answer with per-branch source attribution:

> "The fee applies by default, but is waived for premium-tier cardholders
> (per p1)."

## The one rule

**The model explains the resolved structure; it does not decide which branch is
true.** Truth was settled upstream — by A4's scope relation and B2's
composition. If the generator is given room to adjudicate, it will, and then the
pipeline's careful branch preservation gets quietly undone at the last step by a
model picking a favourite.

## What already exists for you

- `api_budget` — call `complete(messages=..., tier=Tier.JUDGE,
  step="b3_generation")`. Cached and logged.
- `contract/gold.py` — the `Branch` structure you are verbalizing, and
  `gold_scoped_answer` / `selection_answer` on `GoldInstance` as references.

## Metrics you own

PR, SR, SCR, and answer faithfulness (entailment of the generated answer against
the resolved branch structure).

## A note on the model tier

This project runs at zero frontier-API spend, so generation uses the strongest
open-weights model available on Groq. That is a stated limitation, not a hidden
one — see README section 5. A local model may be used for cheap dev-loop
iteration, but a dev-loop model is never the reported system, and any automatic
fallback to one is recorded in the cost log so it appears in the write-up.
