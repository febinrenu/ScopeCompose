# conflict-rag

**Beyond Source Selection: Detecting and Resolving Conditional Knowledge Conflicts in Retrieval-Augmented Generation**

Implementation repository for the two-member final-year project. This file is the **plan of record**:
what we are building, why, how it is split, and in what order. For a log of what has actually been
done, see [`PROGRESS.md`](PROGRESS.md).

Source documents (one directory up):

- `conditional-conflict-rag-proposal-v6.md` — research proposal, Revision 6
- `implementation-plan-two-person-v2.md` — the two-person build plan

Section references below (§5.4, §6.3, …) point into those two documents.

---

## 1. The problem

RAG retrieves several passages and assumes they can be reconciled into one answer. Conflict-aware
RAG systems detect disagreements, classify them (factual / temporal / opinion), and resolve factual
ones by **selecting the most credible source**.

That is the wrong operation for a large and common class of disagreement:

> **Passage A (general rule):** "International transactions incur a 3% fee."
>
> **Passage B (exception):** "Fees are waived for premium-tier cardholders."

These do not contradict each other. Both are true, under **different scopes**. A selection-based
system detects a conflict, ranks the two sources, emits the winner — and silently deletes the other
branch along with the condition that governs it. If the user holds a premium card, the
"most credible" answer is wrong for them.

**The failure is invisible to standard evaluation.** Answer-correctness scores the emitted answer
against a reference; it has no way to register that *a valid, separately-scoped branch of the truth
was dropped*. A system can post strong correctness numbers while systematically erasing exceptions —
exactly the behaviour that is most dangerous in the rule-governed domains where conflict-aware RAG
matters most.

**What we build instead.** A pipeline that detects conditional conflicts as a first-class type,
extracts the governing conditions, and **composes** a scoped answer preserving every valid branch —
plus the metric suite (**Suppression Rate**) that makes the deletion measurable, and a benchmark of
naturally-occurring cases to measure it on.

---

## 2. Ownership split

Two members, one paper, meeting at one frozen JSON contract.

```
                     +----------------------------------------------+
                     |            MEMBER A  (Front-End)             |
                     |  Conflict Analysis & Scope-Relation Decision |
  query ---> A1 Hybrid Retrieval --> A2 Pairwise Conflict Detection  |
                     |        A3 Five-Class Classification          |
                     |           (3 architectures, compared) ------>|
                     |        A4 Four-Way Scope-Relation Analysis   |
                     |           refinement / disjoint /            |
                     |           redundant / opposed                |
                     |           + order-invariance check           |
                     +---------------------+------------------------+
                                           |  INTERFACE CONTRACT
                                           v
                     +----------------------------------------------+
                     |            MEMBER B  (Back-End)              |
                     |   Condition Extraction, Composition, Gen     |
                     | B1 Contrastive Scope Probing + grounding gate|
                     | B2 Composition Operator                      |
                     |    (nested / crossed multi-exception flags)  |
                     | B3 Conflict-Aware Generation --> scoped answer|
                     +----------------------------------------------+
```

| | Member A | Member B |
|---|---|---|
| **Module title** | *Retrieval-Coupled Detection and Scope-Relation Analysis of Conditional Knowledge Conflicts in RAG* | *Contrastive Scope Probing and Compositional Resolution of Conditional Knowledge Conflicts in RAG* |
| **Owns** | `retrieval/` `detection/` `scope/` | `extraction/` `composition/` `generation/` |
| **Metrics** | binary detection F1; five-class accuracy + per-type F1; **factual/conditional confusion cell**; four-way scope-relation accuracy; order-invariance pass rate | condition-extraction accuracy (explicit/implicit separately); grounding-gate rejection rate; **PR / SR / HCR / SCR**; nested & crossed flag rates; metric-judge validation vs. human labels |
| **Baselines** | standard RAG, rerank, NLI-filter | credibility selection, CoT, full-context, taxonomy-aware, **structured long-context** |

**Joint:** the contract, the API budget layer, the benchmark, the evaluation harness, end-to-end
answer correctness, and the structured long-context baseline (the decisive experiment).

---

## 3. The interface contract — freeze it, then build against it

One JSON record per query. This is the only thing both members must agree on; everything else can
change independently.

```json
{
  "query_id": "fin_00142",
  "query": "Do I pay an international transaction fee?",
  "domain": "financial_terms",
  "construction": "natural",
  "passages": [
    {"id": "p0", "text": "...", "source_type": "official_policy", "date": "2024-03"},
    {"id": "p1", "text": "...", "source_type": "product_terms",   "date": "2025-01"}
  ],
  "conflict_pairs": [
    {"doc_i": "p0", "doc_j": "p1", "is_conflict": true,
     "type": "conditional", "scope_relation": "refinement", "confidence": 0.86}
  ],
  "multi_exception_flags": {"nested": false, "crossed": false}
}
```

Three fields carry the weight:

- **`construction`** — `natural` (Tier 1, mined from genuinely separate documents) or `split`
  (Tier 2, a single-document rule/exception pair split across two synthetic documents). Tier 1 and
  Tier 2 results are **never blended** in headline reporting. The tag rides along on every record so
  the metrics library can always break results out.
- **`scope_relation`** — four-way: `refinement | disjoint | redundant | opposed`. This is Member A's
  hardest output and the one Member B's composition depends on. Confusing `redundant` with
  `refinement` makes B compose two branches that should have merged; confusing `refinement` with
  `opposed` makes B fall back to selection on a real exception. Both directions get their own
  confusion-matrix cell in reporting.
- **`multi_exception_flags`** — populated by **B**, not A, because it requires comparing extracted
  branches to each other rather than to the default.

**Routing logic B applies:**

| type | scope_relation | action |
|---|---|---|
| `conditional` | `refinement` or `disjoint` | compose |
| `conditional` | `redundant` | merge into one branch |
| `conditional` | `opposed` | genuine conflict, fall back to selection |
| `factual` / `temporal` / `opinion` | — | matching prior-work strategy |

Then: if two or more surviving exception branches are pairwise overlapping, B sets `nested` or
`crossed` and routes to selection with the flag recorded.

> **Freeze rule.** Once Member B signs off, any schema change requires bumping `CONTRACT_VERSION`
> **and** updating `contract/mock.py` in the same commit. See `CLAUDE.md`.

---

## 4. Hardware and provider decisions

The original plan assumed an RTX 3050 with 4–6 GB VRAM. This machine is an **RTX 4060 Laptop, 8 GB**
(16 GB RAM, 20 logical cores). Shared code therefore targets a **6 GB floor** so it runs unmodified
on Member B's machine, with an opt-in 8 GB profile in `config/hardware.yaml`.

### What runs locally (the GPU's job: classification and scoring, never generation)

| Component | Model class | Settings |
|---|---|---|
| Dense retrieval embeddings (A1) | Contriever, base-size | fp16, batch 32, seq <= 256 |
| Stage-1 conflict filter (A2) | DeBERTa-v3-base MNLI cross-encoder | fp16, batch 16, seq <= 256 |
| Detector variant (i) lexical+NLI | reuses Stage-1 encoder + small head | trains in minutes |
| Detector variant (ii) fine-tuned | DeBERTa-v3-base, end-to-end | fp16, batch 8 x grad-accum 4, seq <= 384, 3–5 epochs |
| Detector variant (iii) structured | same encoder, called twice | as above |
| Grounding-gate NLI scorer (B1) | DeBERTa-v3-class cross-encoder | fp16 |

### What runs via API

**Provider: Groq** (OpenAI-compatible endpoint), with local **Ollama / Qwen 2.5 7B Instruct Q4_K_M**
as an offline and rate-limit fallback. Both are reached through one `openai`-SDK code path with a
configurable `base_url`, so swapping providers is a config change, not a code change.

| Step | Tier | Notes |
|---|---|---|
| Stage-2 conflict judgement (A2) | `BULK` | only low-confidence pairs escalate |
| Applicability-descriptor extraction (A4) | `BULK` | structured JSON output |
| Counterfactual question generation (B1) | `BULK` | |
| Candidate-condition proposal (B1) | `BULK` | |
| Metric judge, PR/SR/HCR/SCR | `JUDGE` | validated against human labels before trusted |
| Final scoped-answer generation (B3) | `JUDGE` | |
| Structured long-context baseline | `LONG_CONTEXT` | the decisive experiment |
| Second-backbone robustness | `SECOND_BACKBONE` | a different model family |

Model IDs are **not hardcoded** — `scripts/discover_models.py` queries Groq's `/openai/v1/models`
and writes `config/models.yaml`, so the tier map reflects what the account can actually reach.

### Budget-control tactics (non-negotiable)

1. **Cache every LLM call** keyed on `(provider, model, messages, temperature, max_tokens, …)`.
   The second run of any script is free. Built in Week 1, not bolted on later.
2. **Tier by step, not by habit.** Bulk steps get the fast cheap model; load-bearing steps get the
   strongest available.
3. **Run every new component on the ~50-instance WP1 pilot first.** Only scale to the full benchmark
   once it is stable.
4. **Log every call** (model, step, tokens, cache hit, backend actually used) and review weekly.

---

## 5. Named risk: the zero-spend constraint and the open-weights asymmetry

**Decision: no frontier-API spend. Everything runs on Groq-hosted open weights or locally.**

This is a real constraint with a real consequence, and it must be stated in the paper rather than
discovered by a reviewer.

Proposal §6.3 specifies the **structured long-context baseline** — a strong model given the full
concatenated passage set and asked for an explicit SG-DT-style branch parse, scored with our own
PR/SR/HCR/SCR — as the paper's decisive experiment. §6.5 says outright that if that baseline matches
our Preservation Rate, the central architectural claim **does not hold as stated**.

Two observations:

- **Context length was never the binding constraint.** At K = 5 passages the baseline's input is only
  ~2–5K tokens. Every candidate model fits it comfortably. What open weights cost us is *parsing
  capability*, not window size.
- **The result is therefore interpretable in only one direction:**

  | Outcome | Interpretation |
  |---|---|
  | Open-weights baseline **matches or beats** the pipeline | **Decisive falsification.** Fully valid, still publishable — a weak baseline winning is strong evidence against us. |
  | Open-weights baseline **loses** | **Weak evidence.** A reviewer will attribute the gap to model tier rather than architecture. |

So the experiment stays worth running: it can falsify the claim but cannot confirm it. The write-up
must say this plainly. `config/models.yaml` keeps the tier map as configuration precisely so a single
line swaps in a frontier model if credits ever materialise.

The same caveat applies, more mildly, to the **metric judge**, which Cattan et al. validated at 0.89
accuracy against human labels. Our judge gets the same validation treatment — and if the open-weights
judge fails to track human labels, that is a finding to report, not a number to bury.

**Other named risks** (proposal §8), owners in brackets:

- **Novelty** [joint] — Chen et al. (SG-DT / NormBench, KDD 2026) named the same failure mode first,
  in a different setting. Mitigation is the structured baseline, not positioning prose. WP0 gates.
- **Tier-1 scarcity** [joint] — genuinely multi-source rule/exception pairs may be rare. Mitigation:
  the named Tier 1 / Tier 2 split plus a WP1 mining pilot that measures real yield before committing.
- **Extraction ceiling** [B] — if Contrastive Scope Probing does not beat direct extraction by a
  useful margin, the method contribution shifts to the grounding gate and metric suite.
- **Annotation boundary** [joint] — refinement-vs-opposed is the hardest judgement; explicit decision
  procedure, kappa reported for that axis specifically, third-reviewer adjudication.
- **Integration drift** [joint] — mitigated by the frozen contract and the shared mock generator.

---

## 6. Module build order

Status legend: **DONE** (built, tested, verified by running) / **STUBBED**
(interface fixed, body is Member B's) / **PENDING** (needs data or a human).

A status that lies is worse than no status, so these are kept honest: several
say DONE for the *harness* while the run itself is PENDING real data.

### Phase 0–2 — foundations (Week 1, must come first)

| Module | Owner | What | Status |
|---|---|---|---|
| `contract/` | A leads | Pydantic schema, gold-instance schema, routing table, **seeded mock generator**, JSON Schema export, validator CLI | DONE |
| `api_budget/` | joint | SQLite request cache, tiered provider-agnostic client, cost log, offline mock backend | DONE |
| `config/` | joint | `hardware.yaml` (6 GB / 8 GB profiles), `models.yaml` (tier map) | DONE |

The mock generator is what lets Member B build before A's detector exists. The cache is what keeps
development iteration free instead of metered. Both are due end of Week 1.

### Phase 3 — Member A's modules

| Module | What | Status |
|---|---|---|
| `retrieval/` **A1** | BM25 (`rank_bm25`) + dense (Contriever, fp16 local) fused by reciprocal-rank fusion, K = 5. Kept deliberately identical to prior systems so comparisons isolate analysis, not retrieval. | DONE |
| `detection/` **A2** | Two-stage. Stage 1: local DeBERTa-v3-base NLI cross-encoder + learned head over embedding interaction features, resolving most of the 10 pairs per query at zero API cost. Stage 2: only pairs below tau_c escalate to a cached `BULK` call. | DONE |
| `detection/variants/` **A3** | Three architectures, compared head-to-head: (i) lexical cues + NLI features, (ii) fine-tuned DeBERTa-v3-base, (iii) structured entailment (condition- and outcome-entailment scored separately, combined by the A4 rule). **The reported headline is the factual/conditional confusion cell** — that is what decides which variant is "the" detector. | DONE |
| `scope/` **A4** | Four-way relation. Extract an applicability descriptor and outcome per claim, decide the set relation (subset / disjoint / overlapping-non-nested) via entailment plus a typed attribute algebra for ranges, tiers and categories, **and separately** decide outcome agreement on the overlap. Both determinations are required — conflating "do scopes overlap" with "is this a conflict" is the exact bug Revision 4/5 fixed. | DONE |
| `scope/order_invariance.py` | Permutation harness: reshuffle passage order and assert the relation, the branch roles, and downstream scores are unchanged. Branch roles derive **solely** from the applicability-and-outcome relation, never from retrieval order — whichever branch has the superset applicability is the default, full stop. | DONE |

### Phase 4 — baselines and scorers

| Module | Owner | What | Status |
|---|---|---|---|
| `baselines/standard_rag.py` | A | concatenate top-K | DONE |
| `baselines/rerank.py` | A | single top passage | DONE |
| `baselines/nli_filter.py` | A | drop the passage judged inconsistent — the vivid, directly demonstrable suppression case | DONE |
| `metrics/` | B leads | A contributes the detection, classification, scope and order-invariance scorers; B owns PR/SR/HCR/SCR and the judge validation | A's part DONE, B's stubbed |

### Tooling (built alongside the modules)

| Tool | What it answers | Status |
|---|---|---|
| `scripts/discover_models.py` | what does my account actually serve? | DONE |
| `scripts/smoke_test.py` | is every tier reachable, does JSON parse, does the cache work? | DONE |
| `scripts/budget_estimate.py` | does this run fit the daily caps, and in how many days? | DONE |
| `scripts/eval_descriptors.py` | how often does A4 decide scope arithmetically rather than by entailment? | DONE |
| `detection/train.py` | fit and persist the stage-1 head; does it beat the rule fallback? | DONE |
| `detection/threshold.py` | what escalation band, at what cost/accuracy trade-off? | DONE, **needs real data** |
| `experiments/run_baselines.py` | do the baselines still have the exception passage at generation time? | DONE |
| `experiments/run_retrieval_eval.py` | recall@k, and **exception** recall@k | DONE, **needs real data** |
| `benchmark/wp1_probe_set.py` | 54 probe cases for Member B | DONE, **needs verification** |

Two of these currently produce fixture artifacts rather than results, and say
so in their own output: the threshold curve saturates on templated data, and
exception recall is floored by near-duplicate mock passages. Both need the WP2
corpus before anything they print is quotable.

### Phase 5 — gating work packages

| Item | Owner | What | Status |
|---|---|---|---|
| `docs/wp0_positioning_memo.md` | joint | **WP0, gating.** Three confirmed anchors to work outward from: SG-DT / NormBench (Chen et al., arXiv:2606.08932, KDD 2026), ConditionalQA (Sun et al., ACL 2022, arXiv:2110.06884), Defeasible NLI (Rudinger et al., EMNLP 2020). Plus direct searches of KR, IJCAI, TPLP, AAAI, AIJ, JAR and the ACL Anthology, which general web search does not reach. | Template ready, review pending |
| `benchmark/mining/tier1_pilot.py` | A | **WP1, gating.** ~20-document Tier-1 mining trial to measure real yield of genuinely multi-source pairs before committing to the 250-instance target. | Harness DONE, run pending |
| `experiments/wp1_probe/` | B | **WP1, gating.** ~50-instance Contrastive Scope Probing feasibility probe + the NLI-vs-LLM-judge ablation. | Stubbed for B |
| `benchmark/manual/annotation_manual.md` | joint | Includes the explicit refinement-vs-opposed decision procedure and worked examples. | Draft ready, co-author pending |

Both gating items must close by **end of Week 4** — not Week 14. Either can reshape the paper, and
both are cheap now and expensive late.

---

## 7. Timeline (maps to WP0–WP5)

| Weeks | Work | A | B |
|---|---|---|---|
| 1 | Freeze contract, mock generator, API cost cache | lead | co |
| 1–3 | **WP0 prior-art review (gating)** | RAG / conflict-aware lit | SG-DT, ConditionalQA, Defeasible NLI, classical defeasible lit |
| 2–4 | **WP1 feasibility probe + Tier-1 mining pilot (gating, go/no-go)** | ~50 labelled cases; ~20-doc mining trial | extraction probe; NLI-vs-judge ablation |
| 4–11 | Benchmark construction (Tier 1 + Tier 2, three-stage annotation, datasheet) | type / scope / distractor labels | branch / scoped-answer labels / `construction` tag |
| 6–13 | Module implementation | retrieval, detection x3, scope, order-invariance | extraction, composition, generation |
| 13–19 | Baselines, experiments, all ablations incl. **structured long-context** | owned metrics | owned metrics |
| 17–22 | Writing + benchmark/harness release | leads detection/scope sections | leads extraction/eval sections |

---

## 8. Setup

```bash
# Python 3.11 (not 3.13 — parts of the ML stack lag behind it)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell

# PyTorch with CUDA first, then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

copy .env.example .env            # then put your GROQ_API_KEY in .env  (never commit it)
python scripts/discover_models.py # writes config/models.yaml from your account's real model list
```

Verify the GPU is actually being used — if this prints `False`, the CPU-only wheel got installed and
nothing below is worth running until it is fixed:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Running things

**Everyday**

```bash
pytest                                                   # full suite, offline, zero API spend
python -m contract.mock --n 50 --seed 0 -o mock.jsonl    # deterministic mock records for Member B
python -m contract.validate mock.jsonl                   # validate any JSONL against the contract
python -m contract.export_schema                         # regenerate contract/schema/v1.0.0.json
python -m experiments.run_pipeline --data mock.jsonl     # end-to-end A1->A4 (add --live for real calls)
```

**The correctness gate** — run this before any commit that touches `scope/`.
It exits non-zero on failure, so it can gate CI, and it must pass at 1.0000:

```bash
python -m scope.order_invariance --sample 60 --heuristic-nli
```

**Provider and budget** — run the smoke test after any change to
`config/models.yaml`, and the estimator before any full-corpus pass:

```bash
python scripts/discover_models.py --show      # what your account actually serves
python scripts/smoke_test.py --all-tiers      # every tier reachable, JSON parses, cache works
python scripts/budget_estimate.py --judge-batched --ablation-instances 100
python -m api_budget.report                   # weekly spend review
```

**Training and tuning** — in this order; the threshold sweep is meaningless
against the untrained rule-based fallback, which scores near chance:

```bash
python -m detection.train --data train.jsonl              # fit + persist the stage-1 head
python -m detection.threshold --data dev.jsonl --min-accuracy 0.95 --write
```

`--write` stores the chosen band in `config/hardware.yaml` and marks it
`tuned: true`. Until then the shipped band is a **placeholder**, and any
escalation rate quoted from it is describing two invented constants.

**Evaluation**

```bash
python -m detection.variants.compare --data gold.jsonl    # three-architecture comparison
python -m experiments.run_baselines --data gold.jsonl --show-examples 2
python -m experiments.run_retrieval_eval --data gold.jsonl --k 5
python scripts/eval_descriptors.py                        # A4 typed-attribute rate
```

**Benchmark**

```bash
python -m benchmark.wp1_probe_set --review                # WP1 verification sheet
python -m benchmark.wp1_probe_set -o benchmark/data/wp1_probe.jsonl
python -m benchmark.mining.tier1_pilot --docs docs.jsonl  # WP1 Tier-1 yield trial
```

---

## 9. Repository layout

```
conflict-rag/
├── README.md              # this file — the plan of record
├── PROGRESS.md            # session log: what has actually been done
├── CLAUDE.md              # conventions for AI-assisted sessions in this repo
├── config/                # hardware.yaml (VRAM profiles), models.yaml (tier map)
├── contract/              # interface schema + mock generator        [Week 1, joint, A leads]
├── api_budget/            # request cache, cost log, tiered client   [Week 1, joint]
├── retrieval/             # A1                                       [A]
├── detection/             # A2, A3 (three architectures)             [A]
├── scope/                 # A4 (four-way relation, order-invariance) [A]
├── extraction/            # B1 Contrastive Scope Probing             [B]
├── composition/           # B2 composition operator                  [B]
├── generation/            # B3 conflict-aware generation             [B]
├── metrics/               # shared scoring library                   [B lead, A contributes]
├── baselines/             # A: standard, rerank, nli-filter
│                          # B: credibility, cot, full-context, taxonomy-aware, structured-long-context
├── benchmark/             # data, annotation tooling, manual, datasheet [joint]
├── experiments/           # end-to-end runners, ablations            [joint]
├── scripts/               # setup and maintenance utilities
├── docs/                  # WP0 positioning memo, design notes
└── tests/                 # pytest
```

---

## 10. One honesty guardrail

Carried over in spirit from §12 of the implementation plan, because it applies to every slide and
every commit:

> **Do not put target percentages into slides or the paper as if they were results.** Every number is
> a hypothesis until the experiment runs, and the design is meant to be *able to disprove the central
> claim*. The strength of this project is a sharp, evidenced failure analysis, an architectural
> distinction that is **tested rather than argued**, and a released benchmark — not a grand framing.
> The genuinely open items — the WP0 prior-art result, the WP1 extraction feasibility and Tier-1
> mining yield, and above all whether the structured long-context baseline wins or loses — are the
> honest risks, and naming them in a review is a strength, not a weakness.
