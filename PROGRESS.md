# Progress log

Session-by-session record of what was actually done, newest first. `README.md`
is the plan; this file is the history.

**Append an entry before ending any working session.** The reasoning behind a
decision is what gets lost over 22 weeks, not the code — so write down *why*,
not just *what*. The entry format is in `CLAUDE.md`.

---

## Sessions

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

**Blocked / open**

1. **Contract sign-off from Member B.** `contract/schema/query_record-v1.0.0.json`
   is ready to send. Nothing should be built against a contract B has not
   agreed to.
2. **No Groq key in `.env` yet.** Everything above ran offline. Copy
   `.env.example` to `.env`, add `GROQ_API_KEY`, then run
   `python scripts/discover_models.py` to populate the tier map. Until then all
   four tiers are unresolved and any live call raises `TierNotConfigured`.
3. **Member B's GPU size is unknown.** The 6 GB floor is assumed. If their card
   is 4 GB, `config/hardware.yaml` needs a fourth profile.
4. **`FineTunedDetector` has never been trained.** The code is written and fits
   the memory budget on paper; it has not been run. It is the one A3 variant
   with no rule-based fallback.
5. **Dense retrieval GPU path not yet confirmed end to end** — the Contriever
   download was still running when this session ended. Sparse retrieval and
   fusion are fully tested.
6. **Supervisor should hear about the zero-spend consequence** before WP4, with
   the asymmetry argument. It is a framing decision, not only a budget one.

**Next up**

1. Send `contract/schema/query_record-v1.0.0.json` to Member B and get the
   contract frozen. Everything downstream depends on it.
2. Add `GROQ_API_KEY` to `.env`, run `python scripts/discover_models.py`, then
   make one live `BULK` call and confirm the second identical call is served
   from cache at zero tokens.
3. Run `pytest -m slow` to confirm the dense-retrieval GPU path.
4. Train `FineTunedDetector` on mock data — not for accuracy, but to confirm it
   fits in 8 GB at the stated batch size and that the training loop runs.
5. **Start WP0.** It is gating and it is reading, not code. Begin with the three
   confirmed anchors in `docs/wp0_positioning_memo.md`.
6. Collect ~20 real documents and run `benchmark.mining.tier1_pilot` for the
   actual Tier-1 yield. Due end of week 4.
