# Conventions for this repository

Read [`README.md`](README.md) first: it describes the study, the repository and how to reproduce every result.
[`study/ANALYSIS_PLAN.md`](study/ANALYSIS_PLAN.md) is the plan of record for the analysis, and
[`PROGRESS.md`](PROGRESS.md) is the dated history of the work.

---

## Research-integrity rules

These rules apply to the code, the commits and the paper.

- **No number is typed by hand.** Every number in the manuscript comes from a macro in `paper/numbers.tex`,
  which `python -m study.report` generates from the saved model outputs. `paper/check_paper.py` fails on a
  digit in running text that is not on its short allow-list of design constants.
- **Planned before run.** The question sets, pipeline settings, metrics and confirmatory tests were committed
  in `study/ANALYSIS_PLAN.md` before the first evaluation answer. Any later change is added there as a dated
  deviation with its reason, and the paper reports it. Analyses added afterwards are labelled exploratory.
- **Tune on the tuning set only.** Prompts, thresholds and operator settings are chosen on the tuning
  questions (`study/splits.json`). The evaluation questions are never used to choose anything.
- **Report the negative result as prominently as a positive one.** The completion remedy failed, and the
  paper says so.
- **Never silently substitute a model.** Every answer records the model and the provider that served it
  (`served_by`). A fallback is a deviation, not a detail.
- **A cell never mixes runs.** `study/run.py` refuses to append to a cell file whose answers came from a
  different model, prompt or chunking.
- **Partial results are not results.** Holm correction is applied only to a complete confirmatory family, and
  `study.report` renders a missing number as a visible marker rather than a stale value.

---

## API and model rules

- Every model call goes through `api_budget.client`, which provides the response cache, retries that honour
  rate-limit hints, provider fallback and the cost log (`logs/api_cost.jsonl`). Never call a provider SDK
  directly from a module.
- Model IDs are never hard-coded in a module. Ask for a logical tier (for example `STUDY_QWEN7B` or
  `STUDY_NEMOTRON`); `config/models.yaml` maps tiers to providers and concrete model IDs.
- API keys live only in `.env`, which is gitignored. `.env.example` lists the variables.
- Tests use the mock backend and never touch the network. `pytest` runs offline, at no cost.

---

## Hardware rules

- The reference machine has an 8 GB RTX 4060 Laptop GPU and 16 GB of RAM. Run one heavy job at a time:
  two local generators plus the GPU scorers do not fit in memory together. `study/schedule.py` unloads each
  local model before loading the next.
- Read device, precision and batch size from `config/hardware.yaml`; do not hard-code `cuda`, `fp16` or a
  batch size in a module.

---

## Code conventions

- Python 3.11, with type hints throughout.
- Modules run as `python -m <module>` where a command-line interface makes sense; each has a docstring that
  shows its usage.
- Determinism: seed everything. Saved rankings, operator decisions and model outputs are what every analysis
  reads, so two analyses can never disagree about the same run.
- Run `pytest` from the repository root, and add a test file for each new module.

---

## Commits

- Commit to `master` with a message that says what changed and why.
- Before committing a change to the paper, run `python -m study.report`, compile `paper/main.tex` and run
  `python paper/check_paper.py`.

---

## Earlier conventions

The earlier ScopeCompose project (see README, Section 12) had rules that no longer apply to the study:
- the two-member ownership split of `retrieval/`, `detection/` and `scope/` versus `extraction/`,
  `composition/` and `generation/`;
- the frozen JSON contract in `contract/` (bump `CONTRACT_VERSION`, update `contract/mock.py` and regenerate
  `contract/schema/` together);
- the Tier 1 / Tier 2 benchmark separation.

They still hold for anyone changing that legacy code.
