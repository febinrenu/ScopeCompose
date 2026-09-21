# Conventions for this repository

Read `README.md` first — it is the plan of record. Read `PROGRESS.md` for what has actually been
done so far. The two research documents live one directory up and are the authority on *why*
anything here exists:

- `../conditional-conflict-rag-proposal-v6.md`
- `../implementation-plan-two-person-v2.md`

---

## Session protocol

**Before finishing any working session, append an entry to `PROGRESS.md`.** Newest entry goes at the
top, directly under the `## Sessions` heading. Use this shape:

```markdown
### YYYY-MM-DD — short title

**Done**
- bullet per completed item, with file paths

**Decisions**
- decision, and the reason behind it (the reason is the part that is expensive to reconstruct later)

**Blocked / open**
- anything waiting on Member B, a supervisor, data access, or an unanswered question

**Next up**
- the concrete next action, specific enough to start from cold
```

This is not optional bookkeeping. The project runs 22 weeks across two people; the reasoning behind
a decision is what gets lost, not the code.

Also update the **status column** in `README.md` §6 when a module changes state. A status that lies
is worse than no status.

---

## Ownership

This repo is Member A's working copy. Member A owns `retrieval/`, `detection/`, `scope/`, and leads
`contract/`. Member B owns `extraction/`, `composition/`, `generation/`, and leads `metrics/`.

**Do not implement Member B's modules.** They are a separately assessed research artifact. B's
directories hold typed stubs and an ownership note; leave them that way unless Member A explicitly
says otherwise. Filling them in would erase the split that both zeroth-review presentations depend
on.

---

## The contract is frozen

`contract/models.py` defines the schema both members build against. Changing it breaks the other
person's work silently.

Any schema change requires **all three** in the same commit:

1. bump `CONTRACT_VERSION` in `contract/models.py`
2. update `contract/mock.py` so generated records still validate
3. regenerate `contract/schema/v<version>.json` via `python -m contract.export_schema`

And tell Member B. A schema change that only one person knows about is the integration-drift failure
the frozen contract exists to prevent.

---

## Hardware rules

Target a **6 GB VRAM floor** so shared code runs unmodified on Member B's machine. This machine has
8 GB (RTX 4060 Laptop); the extra is opt-in through `config/hardware.yaml` profiles, never assumed.

- **Local GPU is for classification and scoring only** — cross-encoders, embeddings, the fine-tuned
  detector. Never a generative model in the reported pipeline.
- **All generation goes through `api_budget.client`.** Never call an LLM provider SDK directly from a
  module; that bypasses the cache and the cost log, which is how a budget problem gets discovered
  after a full-corpus run instead of before one.
- Read device, precision, batch size and sequence length from `config/hardware.yaml`. Do not hardcode
  `cuda`, `fp16`, or a batch size in a module.

---

## API rules

- Every LLM call goes through `api_budget.client.complete()`, tagged with a `step` name so the cost
  log can attribute spend.
- Model IDs are **never hardcoded**. Ask for a logical tier (`BULK`, `JUDGE`, `LONG_CONTEXT`,
  `SECOND_BACKBONE`); `config/models.yaml` maps tiers to concrete IDs, and
  `scripts/discover_models.py` writes that file from the provider's live model list.
- Tests use the `mock` backend and must never touch the network. `pytest` should run offline at zero
  cost, always.
- Before scaling anything to the full benchmark, run it on the ~50-instance WP1 pilot first.

---

## Code conventions

- Python 3.11, type hints throughout, Pydantic v2 for anything crossing a module boundary.
- Modules are importable as `python -m <module>` where a CLI makes sense.
- Determinism matters: seed everything. `contract/mock.py` with the same seed must produce a
  byte-identical file, because it is the fixture both members test against.
- `pytest` from the repo root. New module, new test file.

---

## Research-integrity rules

These come from §12 of the implementation plan and apply to code, commits, slides and the paper:

- **No target percentages presented as results.** Every number is a hypothesis until its experiment
  runs.
- **Never silently substitute a cheaper model** where the design flags one as load-bearing (the
  metric judge; the structured long-context baseline). If a substitution happens — including an
  automatic fallback from Groq to local Ollama after rate-limiting — the cost log records it and it
  gets reported, not buried.
- **Tier 1 and Tier 2 benchmark results are never blended** in headline numbers.
- **Report the falsifying outcome as prominently as a positive one.** The structured long-context
  baseline is designed to be able to disprove the central claim; that is the point of it.
