# WP1 extraction feasibility probe — Member B

**Gating. Due end of week 4, not week 14.**

Prototype Contrastive Scope Probing on ~50 hand-built implicit-condition
instances, benchmarked against a ConditionalQA-style condition-selection
baseline, and run the local-NLI-vs-LLM-judge entailment ablation on the same
set.

## The go/no-go question

Does Contrastive Scope Probing beat direct zero-shot extraction by a useful
margin, at an acceptable Hallucinated-Condition Rate?

An accuracy gain that comes with a high fabrication rate is **not** a success.
The grounding gate exists precisely to make that trade-off visible, so report
both numbers together or neither.

## If the answer is no

The fallback, agreed in advance rather than improvised: anchor B's method
contribution on the **grounding gate and the metric suite** rather than the
probe, and report implicit extraction as an open problem with a measured
ceiling. A measured ceiling is a publishable result; a quietly dropped
component is not.

## Getting started with no data

```bash
python -m contract.mock --n 50 --seed 0 -o pilot.jsonl --coverage
```

The mock generator emits implicit-condition instances (`*_refinement_implicit`
templates) alongside explicit ones, so the probe harness can be built and
tested before the 50 hand-built cases exist. Swap in the real instances when
they are ready — the schema is the same.
