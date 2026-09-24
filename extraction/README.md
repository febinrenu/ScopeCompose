# `extraction/`

## What goes here

**B1: Contrastive Scope Probing**, plus the grounding gate.

Explicit conditions ("for premium-tier cardholders…") reduce to semantic
parsing — one API call per instance. Implicit conditions, where a passage
silently narrows applicability without saying so, are the hard case and the
core of B's contribution:

```
CONTRASTIVE SCOPE PROBING(base_rule, other_passages):
  1. rule, outcome  <- extract (rule -> outcome) from the base passage      [API]
  2. Q <- counterfactual_questions(rule, outcome)                           [API]
  3. candidates <- []
     for p in other_passages:
        for q in Q:
           a <- propose_candidate_condition(q, p)                           [API]
           if a is not null:
              s <- NLI(premise=p, hypothesis=a.condition AND a.outcome)     [local NLI,
                                                                             or API judge
                                                                             as an ablation]
              candidates.append((a, s))
  4. # GROUNDING GATE (required; never disabled outside its own ablation)
     admitted <- [a for (a, s) in candidates if s >= tau_ground]
     rejected_rate <- 1 - |admitted| / max(1, |candidates|)
  5. return admitted, rejected_rate
```

## Implementation

`probing.py` implements this. `ContrastiveScopeProbe.probe_record()` runs the
full loop; `gate()` is the grounding gate and is never disabled outside its own
ablation.

The rule-based fallback (`use_llm=False`) recovers explicitly-marked conditions
only, by design. It lives in the same class as the probe rather than in a
separate ablation file so the comparison is hard to avoid running: if the probe
does not beat cue matching on unmarked conditions, it has not earned its API
cost.

## What this builds on

- **The contract** — `contract/models.py`. `QueryRecord` is your input.
  `contract/routing.py` has the routing table as executable code, so you do not
  need to re-derive which relation composes and which selects.
- **A mock generator** — `python -m contract.mock --n 50 --seed 0 -o mock.jsonl`
  produces deterministic, schema-valid records covering all five conflict types
  and all four scope relations, plus nested and crossed cases. You can build and
  test the whole of B without waiting for A's detector.
- **Gold branch structures** — `contract/gold.py` defines `Branch`,
  `Applicability`, and `ScopeAttribute`. The mock generator fills them in, so
  your composition operator has something to assemble on day one.
- **The API layer** — `api_budget`. Call `complete(messages=..., tier=Tier.BULK,
  step="b1_counterfactual_questions")`. Cached, logged, rate-limit-aware. Never
  call a provider SDK directly; that bypasses the cache and the cost log.
- **A local NLI cross-encoder** — `detection.nli.get_nli()` returns a shared
  DeBERTa-v3 cross-encoder for the grounding gate. It is the same instance A2
  uses, which is what keeps the whole local stack inside the 6 GB VRAM floor.
  Pass `heuristic=True` for a no-download stand-in while developing.

## Metrics

Condition-extraction accuracy (explicit and implicit reported **separately**,
never averaged — implicit recovery is what the method is judged on), the
grounding-gate rejection rate, and HCR.

`metrics/preservation.py` has typed stubs with the agreed signatures for
PR/SR/HCR/SCR and the judge validation. Fill in the bodies; do not change the
signatures without telling Member A, since the shared harness imports them.

## The ablation that is a real decision, not a detail

**Local NLI scorer vs. LLM-as-judge entailment.** The local cross-encoder is
essentially free per call. The LLM judge is higher quality but spends API
budget on every single candidate, and this project runs on a free tier. Run
both on the ~50-instance WP1 pilot, report the accuracy/cost trade-off
explicitly, and default to the local scorer for full-benchmark runs unless the
probe shows a gap large enough to justify the spend.

## Your WP1 go/no-go

Due **end of week 4**, not week 14. Two questions:

1. Does Contrastive Scope Probing beat direct zero-shot extraction by a useful
   margin, at an acceptable HCR? If not, the fallback is to anchor B's method
   contribution on the grounding gate and the metric suite, and report implicit
   extraction as an open problem with a measured ceiling.
2. Does the Tier-1 mining pilot (A is running the ~20-document trial) find
   enough genuinely multi-source pairs? If not, you and A jointly commit to the
   Tier 1 / Tier 2 split rather than discovering the shortfall mid-WP2.
