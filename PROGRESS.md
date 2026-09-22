# Progress log

Session-by-session record of what was actually done, newest first. `README.md`
is the plan; this file is the history.

**Append an entry before ending any working session.** The reasoning behind a
decision is what gets lost over 22 weeks, not the code — so write down *why*,
not just *what*. The entry format is in `CLAUDE.md`.

---

## Sessions

### 2026-09-22 — Closed Member A's six code gaps

All six items from the completeness audit. Four of them surfaced real bugs.

**Done**

- **`detection/threshold.py`** (gap 1) — sweeps the stage-1/stage-2 escalation
  band and reports the cost/accuracy curve with a Pareto front. Selection
  requires an explicit constraint (`--max-escalation` or `--min-accuracy`),
  because without one the answer is always "escalate everything", which is what
  the two-stage design exists to avoid. Detects and refuses to tune on a
  saturated curve.
- **`detection/train.py`** (gap 3) — trains and persists the stage-1 head,
  compares it against the rule-based fallback on a held-out split, and refuses
  to save a head that does not beat the fallback. `Stage1Filter` auto-loads it.
- **`experiments/run_baselines.py`** (gap 2) — scores the three retrieval-side
  baselines on **branch loss**, not answer correctness: did the exception
  passage survive to generation? Prints worked examples.
- **Variant (ii) now trains** (gap 4) — two blockers fixed, see below. The
  three-architecture comparison is finally three-way.
- **`experiments/run_retrieval_eval.py`** (gap 5) — recall@k and, more
  importantly, **exception recall@k**, comparing BM25 against hybrid.
- **`scripts/eval_descriptors.py`** (gap 6) — measures the typed-attribute rate
  directly.

**Bugs found by running these**

1. **Stage 1 was at chance.** The threshold sweep showed stage-1 accuracy of
   **0.500 at zero escalation** with the rule-based fallback. The earlier
   "70% of pairs resolved locally at zero API cost" was therefore measuring
   escalation rate, not correctness -- it was resolving them *wrongly*. After
   training the head: 0.460 -> 1.000 held-out. (1.000 is itself a fixture
   artifact; the mock templates are regular enough to memorise.)

2. **fp16 master weights broke fine-tuning.** Variant (ii) died with
   "Attempting to unscale FP16 gradients": some checkpoints carry a
   half-precision dtype in their config which `from_pretrained` honours, so
   parameters arrived as fp16 while GradScaler expects fp32 masters with fp16
   activations. Fixed with an explicit `.float()`, which is version-agnostic
   where the dtype kwarg name is not. Also needed `tiktoken` + `sentencepiece`
   for the DeBERTa-v3 tokenizer.

3. **A contradictory ConflictPair could be constructed.** When stage 1 said
   "conflict" but the A3 classifier returned `no_conflict`, the pipeline built
   `is_conflict=True, type=no_conflict` -- which the contract rejects. Latent
   until the trained head changed stage-1's verdicts and exposed it. The
   classifier now wins: it answers *how* two passages disagree, and "not
   actually" is a legitimate answer.

4. **`compare()` deferred a decidable case.** When two applicability sets
   constrain *different* dimensions (`account_kind` vs `age`) it returned
   UNKNOWN and fell through to entailment. That case is provably OVERLAPPING:
   they necessarily intersect, and neither contains the other because each
   leaves the other's dimension unconstrained. Since UNKNOWN falls back to
   `opposed`, this was also costing composability on merely cross-cutting
   pairs. **Live effect: exact attribute decisions 0% -> 44%, and `compose`
   fired 4x instead of 1x on the same input.**

**Measured**

| Check | Result |
|---|---|
| Test suite | **246 passed** |
| Order-invariance gate | **PASS, 1.0000**, 66/72 non-positional |
| Typed-attribute rate (live) | **82%** on representative passages |
| Baseline branch loss | rerank_top1 **55.4%**, nli_filter **15.8%**, standard_rag 0% |
| Three-variant comparison | lexical_nli 0.988 acc, combined leak 0.010; other two flagged DEGENERATE |

**The suppression demo now runs**, and it is the slide:

    query:      Do I pay a fee on cross-border transfers?
    nli_filter: dropped p1 as inconsistent (contradiction score 1.000)
    should say: ...the fee is waived for premium-tier cardholders (per p1).
    will say:   Cross-border transfers incur a 3.5% fee.

**Two numbers that are fixture artifacts, not results**

- Exception recall reads 28.6%, but **65% of mock passages are near-duplicates**
  of each other, so pooled retrieval is being asked an unanswerable question.
  The tool now detects this and says so before the number can be misread.
- Stage-1 held-out accuracy of 1.000, and the saturated threshold curve, are
  both the templates being memorised.

Both need re-running on the WP2 corpus. Neither is quotable now.

**Still open for Member A** — none of it code:

1. Contract sign-off from Member B (blocks everything)
2. Rotate the Groq key
3. WP0 literature review, A's slice — **gating**
4. ~50 labelled cases for B's probe — **gating**
5. Tier-1 mining on ~20 real documents — harness ready, zero documents — **gating**
6. WP2 annotation of 250-350 instances — the largest single time cost
7. kappa pilot with B on refinement-vs-opposed
8. Zeroth review deck, slides 4-8

---

### 2026-09-21 (later) — Groq wired up live; four provider findings

Key added, all four tiers verified against the real API, and the rate limits
turned into a planning artifact rather than a surprise.

**Done**

- `.env` created with the Groq key (gitignored; verified untracked).
- **Tier map set from the real free-tier limits** (`config/models.yaml`), with
  those limits stored as data so tooling can reason about them.
- `scripts/smoke_test.py` — live check: every tier reachable and non-empty,
  structured JSON parses, truncation fails loudly, cache serves the second
  call free. **All green.**
- `scripts/budget_estimate.py` — multiplies pipeline call volume against the
  daily caps and reports wall-clock days per model.
- `api_budget` hardened for reasoning models: `reasoning_effort` passthrough,
  `TruncatedResponseError`, per-model capability guard.
- **Live end-to-end run** (5 instances, real DeBERTa on GPU + real Groq calls):
  detection resolved **80% of pairs locally at zero API cost**, stage 2
  escalated one, A4 ran live, routing produced one COMPOSE.
- 245 tests pass.

**Findings — all four came from running the thing, not from reading docs**

1. **Groq rate limits are PER MODEL, not per account.** Putting every tier on
   gpt-oss-120b caps the project at 200K tokens/day. Spreading BULK, JUDGE and
   SECOND_BACKBONE across three models gives **1.1M/day — 5.5x**. This is the
   single reason the tier map looks the way it does; it is not a quality
   compromise.

2. **gpt-oss is a reasoning model, and its reasoning tokens consume
   `max_tokens` before any answer appears.** At `max_tokens=20` it returned
   `finish_reason='length'` with *empty content*. The old backend would have
   returned `''`, every downstream JSON parse would have quietly fallen back,
   and a whole corpus run would have produced plausible-looking empty
   extractions. Now raises `TruncatedResponseError` on any `length` finish —
   partial output too, since JSON cut mid-string is garbage rather than an
   obvious error. `reasoning_effort` is set per tier (`low` for bulk, `medium`
   for load-bearing).

3. **allam-2-7b rejected as the overflow model**, despite having 2.5x the
   daily token budget of anything else. Two disqualifiers, both measured:
   it returns HTTP 400 on `reasoning_effort`, and it answers English prompts
   in Arabic by default. Mixed-language extractions on financial and
   immigration prose would be worse than a rate-limit stall and far harder to
   notice. Overflow is now qwen, which has spare quota.

4. **`groq/compound` has no daily token cap but is unusable as the decisive
   baseline.** It is an agentic system with built-in web search — it could
   look up the real policy instead of parsing the passages it was given, which
   would silently invalidate the one comparison the central claim rests on.
   The unlimited quota is a trap here. LONG_CONTEXT stays on gpt-oss-120b.

   Separately: `llama-prompt-guard-2-22m/86m` have the highest request limits
   on the account and are **not chat models** — they are prompt-injection
   classifiers. `canopylabs/orpheus-*` are text-to-speech. All excluded.

**The schedule problem, and the fix**

`python scripts/budget_estimate.py` on the 300-instance target, 12 configs:

| Plan | Wall clock |
|---|---|
| Naive (per-branch judging, all ablations on full corpus) | **115 days** |
| `--judge-batched --ablation-instances 100` | **41 days** |

WP4 is roughly 42 days. The fitted plan fits with **essentially zero slack**.
The metric judge is 65% of the total, so the two levers that matter are:

- **batch the judge per instance**, not per branch — the passages and rubric
  are re-sent for every branch otherwise, roughly doubling the cost for
  identical judgements;
- **run non-headline ablations on a stratified subsample**, headline
  configurations on the full corpus. Ordinary practice; report it as such.

Re-run the estimator before committing to any full-corpus pass. The cache
makes re-runs free, so the number above is for a cold run only.

**Decisions**

- `BULK` → `openai/gpt-oss-20b`, not allam. allam has 7x the request budget,
  but BULK feeds the factual/conditional decision — the paper's headline
  number — and buying throughput with quality there is a bad trade.
- `JUDGE` / `LONG_CONTEXT` → `openai/gpt-oss-120b` (they share one 200K/day
  cap, so their days add; the estimator now says so explicitly).
- `SECOND_BACKBONE` → `qwen/qwen3.8-27b`. Different family from gpt-oss, so
  the RQ3 robustness ablation is meaningful, and it carries its own 200K/day.
- **Rotate the API key.** It was pasted into a chat transcript, so treat it as
  public regardless of what the transcript is used for.

**Next up**

1. Contract sign-off from Member B — still the gating item.
2. Rotate the Groq key; update `.env`.
3. Tell Member B to batch the metric judge per instance from the start. It is
   a 2x budget difference and much cheaper to design in than to retrofit.
4. A4 currently decides 100% of relations by entailment rather than typed
   attributes on mock data — the LLM extractor is not emitting typed
   attributes for these passages. Worth tuning the descriptor prompt on real
   text, since attribute decisions are exact and entailment ones are not.
5. Start WP0 (gating, reading not code).
6. Collect ~20 real documents for the Tier-1 yield trial.

---

### 2026-09-21 — Repository bootstrap: contract, API layer, and all of Member A

**Done**

- **Scaffold.** Created `conflict-rag/` following §7 of the implementation
  plan. Python 3.11 venv (not the system 3.13 — parts of the ML stack lag
  behind it), `.gitignore` covering `.env` / venv / model caches / cache DB /
  working data, `.env.example`, `requirements.txt`, `pyproject.toml` with
  pytest config and `slow` / `network` markers.
- **`README.md`** — the plan of record: problem statement, A/B ownership split,
  the frozen contract, hardware and provider decisions, module build order with
  status, WP0–WP5 timeline, setup instructions, named risks.
- **`CLAUDE.md`** — repo conventions: the session-log rule, the contract freeze
  rule, hardware rules, API rules, and the research-integrity rules carried
  over from §12 of the implementation plan.
- **`contract/`** (Week-1 joint deliverable, A leads) — Pydantic v2 wire schema
  (`models.py`), gold annotation schema (`gold.py`), the routing table as
  executable code (`routing.py`), a seeded deterministic mock generator
  (`mock.py`), JSON Schema export, and a validator CLI. `CONTRACT_VERSION =
  1.0.0`, stamped on every record.
- **`api_budget/`** (Week-1 joint deliverable) — SQLite request cache keyed on
  everything that can change a response, provider-agnostic client over the
  `openai` SDK with configurable `base_url` (Groq + Ollama + any future
  provider on one code path), tiered model routing, JSONL cost log with a
  report CLI, and a deterministic offline mock backend.
- **`config/`** — `hardware.yaml` with three profiles (6 GB floor, 8 GB opt-in,
  CPU-only for CI) and `models.yaml` mapping logical tiers to models.
  `scripts/discover_models.py` populates the tier map from Groq's live model
  list rather than hardcoding IDs.
- **A1 `retrieval/`** — BM25 (`rank_bm25`) + dense (Contriever, fp16, local) +
  reciprocal-rank fusion at K = 5, over a passage store carrying source type,
  date and `document_id`.
- **A2 `detection/`** — two-stage detector. Stage 1: local DeBERTa-v3 NLI
  cross-encoder + hand-authored lexical features + a learned head over
  embedding interaction features. Stage 2: selective cached LLM escalation for
  the uncertain band only.
- **A3 `detection/variants/`** — all three architectures: lexical+NLI (with a
  coefficient-table `explain()`), fine-tuned DeBERTa-v3-base (multi-task: five
  classes + four-way relation), and structured entailment (the cross-encoder
  called twice and combined by the A4 rule). Plus `compare.py`, which reports
  the factual/conditional confusion cell as the headline.
- **A4 `scope/`** — the four-way relation analyser. Typed attribute algebra
  (`attributes.py`) for exact subset/disjoint decisions over tiers, ranges and
  categories; LLM + rule-based descriptor extraction; and `relation.py`, which
  decides applicability and outcome agreement **separately** and combines them.
- **`scope/order_invariance.py`** — the permutation harness, wired to exit
  non-zero so it can gate CI.
- **`baselines/retrieval_side.py`** — standard RAG, rerank-top-1, and the
  NLI-filter (the demonstrative one: on a conditional conflict it deletes
  exactly the exception branch).
- **`metrics/`** — A's scorers (binary detection, five-class with the named
  confusion cells, four-way scope with the four named cells, tier splitting)
  plus typed stubs for B's PR/SR/HCR/SCR with the agreed signatures.
- **Member B landing zone** — `extraction/`, `composition/`, `generation/`
  each hold an ownership README listing what goes there, what already exists to
  build against, which metrics B owns, and B's WP1 go/no-go.
- **WP support** — `benchmark/mining/tier1_pilot.py` (the ~20-document Tier-1
  yield trial, with a recommendation that changes based on measured yield),
  `docs/wp0_positioning_memo.md` (template with the three confirmed prior-art
  anchors and the venue checklist), `benchmark/manual/annotation_manual.md`
  (draft, including the refinement-vs-opposed decision procedure).
- **Tests** — 242 passing, offline, zero API spend, ~3 seconds.

**Verified by running**

| Check | Result |
|---|---|
| `pytest -m "not slow"` | 242 passed |
| Mock determinism (same seed, two runs) | byte-identical SHA-256 |
| Mock coverage at n=30 | all 5 conflict types, all 4 scope relations, nested + crossed, explicit + implicit |
| Contract validation of generated records | 60/60 valid |
| **`scope.order_invariance --sample 60`** | **PASS, 1.0000, 66/72 non-positional roles** |
| `detection.variants.compare` | runs, emits the three-variant table |
| `experiments.run_pipeline --evaluate` | runs A1→A4, emits a valid record and all scorers |
| `benchmark.mining.tier1_pilot --demo` | runs, produces a yield-based recommendation |
| torch CUDA | 2.6.0+cu124, RTX 4060 Laptop, 8.59 GB, `cuda.is_available() == True` |
| `pytest -m slow` (dense retrieval on GPU) | passed — Contriever downloads, encodes, and fuses with BM25 |
| Full suite incl. slow | **244 passed, 1 skipped** |

The one skip is `test_second_backbone_is_a_different_family_from_judge`, which
correctly skips until `SECOND_BACKBONE` is assigned.

**Decisions**

- **Provider: Groq**, zero frontier-API spend, local Ollama/Qwen 2.5 7B Q4 as
  the rate-limit fallback. *Why:* the user's constraint. The consequence is
  written into README §5 as a named risk rather than buried — see below.
- **The open-weights asymmetry.** At K = 5 the structured long-context
  baseline's input is only ~2–5K tokens, so context length was never the
  binding constraint; capability is. That makes the decisive experiment
  interpretable in only one direction: if the open-weights baseline *matches or
  beats* the pipeline, that is valid falsification and still publishable; if it
  *loses*, a reviewer attributes the gap to model tier. The experiment is still
  worth running — it can falsify the claim but cannot confirm it. This must go
  in the paper. Model IDs are config, not code, so one line swaps in a frontier
  model if credits appear.
- **Not using DeepSeek-R1-8B for bulk work.** It is a reasoning distill that
  emits long `<think>` traces — slow and awkward to parse across thousands of
  structured-extraction calls. Qwen 2.5 7B Instruct is the better local bulk
  model. R1-distill stays available as an extraction-model-size sweep row.
- **6 GB VRAM floor, 8 GB opt-in.** This machine has 8 GB; Member B's is
  unconfirmed. Both profiles train at the *same* effective batch size (32), so
  the two machines produce the same model from one config — otherwise the
  comparison between them would be meaningless.
- **Local GPU never runs a generative model** in the reported pipeline. It runs
  cross-encoders and embeddings. Putting a Q4 7B resident on the same 8 GB card
  would contend with the DeBERTa fine-tune and serialise the whole dev loop.
- **`set_relation` is stored canonically** (as the exception relative to the
  default), not as `compare()`'s raw argument-relative output. *Why:* found by
  a failing test — the raw value mirrors SUBSET↔SUPERSET under argument swap,
  which is semantically correct but would have made the order-invariance
  harness report a false violation on every single refinement pair.
- **A `redundant` gold instance resolves to exactly one merged branch**, not
  two. *Why:* found by the schema validator rejecting the mock generator's
  output. Recording a second branch would assert an exception that does not
  exist — the fabricated-branch failure that separating `redundant` from
  `refinement` exists to prevent.
- **Stage 1 alone never emits a `conditional` label.** It establishes only
  *that* two passages disagree, not *how*. An unearned conditional label routes
  to composition and can fabricate a branch, so the honest default is
  `factual`.
- **An undecidable scope relation falls back to `opposed`** (→ selection)
  rather than optimistically composing. Composing on unestablished scope
  invents a branch; selection is what prior work does and is the safe default.
- **Model: `openai/gpt-oss-120b`** (chosen by Member A, set late in the
  session) for `JUDGE` and `LONG_CONTEXT` — the two load-bearing tiers. It is
  among the strongest open-weights models Groq serves, which is the right call
  for the metric judge and the decisive baseline.
  - `BULK` is currently also pointed at it **only so nothing is unresolved**.
    This is the one tier that should move to a smaller model: bulk steps are
    thousands of calls and will exhaust a free-tier daily token cap long
    before the load-bearing steps do. Run `scripts/discover_models.py` and
    reassign.
  - `SECOND_BACKBONE` deliberately left **unset**. It must be a *different
    family* from JUDGE — another gpt-oss model would make the robustness
    ablation (RQ3, "architectural rather than model-specific") vacuous. A test
    enforces this and skips until it is assigned.

**Blocked / open**

1. **Contract sign-off from Member B.** `contract/schema/query_record-v1.0.0.json`
   is ready to send. Nothing should be built against a contract B has not
   agreed to.
2. **No Groq key on disk yet — THE remaining blocker for live runs.**
   Everything above ran offline. `.env` does not exist; only `.env.example`
   does, and `GROQ_API_KEY` is not in the process or user environment either.
   Create `.env` with the key, then run `python scripts/discover_models.py`.
   Until then every live call raises a provider-not-configured error.
3. **Member B's GPU size is unknown.** The 6 GB floor is assumed. If their card
   is 4 GB, `config/hardware.yaml` needs a fourth profile.
4. **`FineTunedDetector` has never been trained.** The code is written and fits
   the memory budget on paper; it has not been run. It is the one A3 variant
   with no rule-based fallback.
5. **`SECOND_BACKBONE` unassigned.** Needs a non-gpt-oss model of comparable
   size, chosen after seeing Groq's real catalogue.
6. **Supervisor should hear about the zero-spend consequence** before WP4, with
   the asymmetry argument. It is a framing decision, not only a budget one.

**Next up**

1. Send `contract/schema/query_record-v1.0.0.json` to Member B and get the
   contract frozen. Everything downstream depends on it.
2. Create `.env` with `GROQ_API_KEY`, run `python scripts/discover_models.py`,
   then make one live `BULK` call and confirm the second identical call is
   served from cache at zero tokens. Reassign `BULK` to a smaller model and
   pick a non-gpt-oss `SECOND_BACKBONE` from the discovered catalogue.
3. Train `FineTunedDetector` on mock data — not for accuracy, but to confirm it
   fits in 8 GB at the stated batch size and that the training loop runs.
4. **Start WP0.** It is gating and it is reading, not code. Begin with the three
   confirmed anchors in `docs/wp0_positioning_memo.md`.
5. Collect ~20 real documents and run `benchmark.mining.tier1_pilot` for the
   actual Tier-1 yield. Due end of week 4.
