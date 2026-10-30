"""Run one cell of the system matrix: (system x backbone x population x setting).

A *cell* is the unit of an experiment, and three properties make a cell
trustworthy enough to publish:

* **Resumable.** Groq's free tier caps a model at ~200K tokens a day, so a
  cell is routinely interrupted. Answers are appended one line per item and a
  restart skips what is already there; a crash costs nothing already paid for.
* **One model per cell.** The cell records the concrete model its tier
  resolved to, and a cell file that already holds answers from a *different*
  model is refused. Tiers are pinned (``fallback: false``) so a rate limit
  stops the cell instead of silently completing it on another model.
* **Same inputs for every system.** Each system gets the identical
  ``(query, passages)``; B3 additionally gets gold conflict groups (an upper
  bound, see ``baselines/registry.py``).

Test-split cells additionally refuse to run before the pre-registration is
frozen (``experiments/staging.py``).

Settings:

``underspecified`` (primary)
    The query as written; scored by case-complete accuracy over personas.
``persona`` (secondary)
    The query plus one persona statement, one answer per persona; scored
    deterministically against the persona's gold outcome, with no reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import settings
from api_budget import BackendError, Tier
from api_budget.client import LLMClient
from baselines.registry import make_system, oracle_conflicts
from benchmark.strata import Stratum, stratum_of
from contract.gold import GoldInstance
from contract.models import Domain

BACKBONES = {"primary": Tier.BACKBONE_PRIMARY, "local": Tier.BACKBONE_LOCAL}
SYSTEM_IDS = ("B0", "B3", "B7c", "B8", "O3")
SETTINGS = ("underspecified", "persona")

#: Population name -> predicate over (instance, stratum).
POPULATIONS: dict[str, Callable[[GoldInstance, Stratum], bool]] = {
    "S1": lambda i, s: s is Stratum.S1,
    "S2": lambda i, s: s is Stratum.S2,
    "S3": lambda i, s: s is Stratum.S3,
    "S4": lambda i, s: s is Stratum.S4,
    "S5": lambda i, s: s is Stratum.S5,
    "controls": lambda i, s: s is Stratum.CONTROL,
}


class CellInterrupted(RuntimeError):
    """The cell stopped before finishing (rate limit, quota); it is resumable."""


class CellMismatch(RuntimeError):
    """The cell file holds answers from a different model or system than this run."""


@dataclass(frozen=True)
class Cell:
    system: str
    backbone: str            # "primary" | "local"
    population: str          # a key of POPULATIONS
    setting: str = "underspecified"
    split: str = "test"

    def __post_init__(self) -> None:
        if self.system not in SYSTEM_IDS:
            raise ValueError(f"unknown system {self.system!r}")
        if self.backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {self.backbone!r}")
        if self.population not in POPULATIONS:
            raise ValueError(f"unknown population {self.population!r}")
        if self.setting not in SETTINGS:
            raise ValueError(f"unknown setting {self.setting!r}")

    @property
    def name(self) -> str:
        return f"{self.system}__{self.backbone}__{self.population}__{self.setting}__{self.split}"

    @property
    def tier(self) -> Tier:
        return BACKBONES[self.backbone]

    @property
    def model(self) -> str:
        return settings.models().tier(self.tier.value).model or ""


def domain_of(inst: GoldInstance) -> str:
    return "immigration" if inst.domain is Domain.IMMIGRATION_ELIGIBILITY else "banking"


def select(instances: Iterable[GoldInstance], cell: Cell) -> list[GoldInstance]:
    """The instances of a cell: its split, its population, and (for CCA) those with personas."""
    keep = POPULATIONS[cell.population]
    out = [i for i in instances if i.split == cell.split and keep(i, stratum_of(i))]
    return sorted(out, key=lambda i: i.instance_id)


def items(inst: GoldInstance, cell: Cell) -> list[tuple[str, str | None, str]]:
    """(item key, persona id, query) for one instance in one setting."""
    if cell.setting == "underspecified":
        return [(inst.instance_id, None, inst.query)]
    return [(f"{inst.instance_id}|{p.persona_id}", p.persona_id, f"{inst.query} {p.statement}")
            for p in inst.personas]


def build_system(cell: Cell, domain: str, client: LLMClient | None):
    if cell.system == "O3":
        from pipelines.scopecompose import ScopeCompose

        return ScopeCompose(tier=cell.tier, domain=domain, client=client)
    return make_system(cell.system, tier=cell.tier, client=client)


def read_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["item"]] = rec
    return done


def check_cell_file(cell: Cell, path: Path) -> dict[str, dict]:
    """The records already in a cell file, refusing any from a different model or system.

    Called before a cell is run AND before a cell is declared complete: a finished file full of
    answers from the wrong model is not a finished cell."""
    done = read_done(path)
    for rec in done.values():
        if rec.get("model") != cell.model or rec.get("system") != cell.system:
            raise CellMismatch(
                f"{path.name} already holds answers from {rec.get('system')}/{rec.get('model')}, "
                f"this run is {cell.system}/{cell.model}: a cell never mixes models")
    return done


def run_cell(
    cell: Cell,
    instances: Iterable[GoldInstance],
    out_dir: Path,
    *,
    client: LLMClient | None = None,
    token_budget: int | None = None,
    max_errors: int = 3,
    on_item: Callable[[dict], None] | None = None,
) -> dict:
    """Run (or resume) a cell. Returns a summary; raises :class:`CellInterrupted` if it stops early."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cell.name}.jsonl"
    todo_instances = select(instances, cell)
    done = check_cell_file(cell, path)

    spent = errors = 0
    summary = {"cell": cell.name, "model": cell.model, "total": 0, "skipped": len(done), "ran": 0,
               "tokens": 0, "calls": 0}
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for inst in todo_instances:
            system = build_system(cell, domain_of(inst), client)
            conflicts = oracle_conflicts(inst) if cell.system == "B3" else ()
            for key, persona_id, query in items(inst, cell):
                summary["total"] += 1
                if key in done:
                    continue
                if token_budget is not None and spent >= token_budget:
                    raise CellInterrupted(f"{cell.name}: token budget {token_budget} reached "
                                          f"after {summary['ran']} new items; rerun to resume")
                try:
                    ans = system.answer(query, inst.passages, conflicts=conflicts)
                except BackendError as exc:
                    errors += 1
                    if errors >= max_errors:
                        raise CellInterrupted(f"{cell.name}: {errors} backend errors, last: {exc}") from exc
                    continue
                rec = {"item": key, "instance_id": inst.instance_id, "persona_id": persona_id,
                       "system": cell.system, "backbone": cell.backbone, "model": cell.model,
                       "setting": cell.setting, "split": cell.split, "text": ans.text, "calls": ans.calls,
                       "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
                       "trace": ans.trace}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                spent += ans.total_tokens
                summary["ran"] += 1
                summary["tokens"] += ans.total_tokens
                summary["calls"] += ans.calls
                if on_item:
                    on_item(rec)
    if errors:
        raise CellInterrupted(f"{cell.name}: finished with {errors} skipped item(s) after backend errors; rerun")
    return summary


def load_answers(path: Path) -> dict[str, str]:
    """``item -> answer text`` from a cell file."""
    return {k: r["text"] for k, r in read_done(path).items()}


def load_instances(path: Path) -> list[GoldInstance]:
    return [GoldInstance.model_validate(json.loads(l))
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--system", required=True, choices=SYSTEM_IDS)
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--population", required=True, choices=sorted(POPULATIONS))
    ap.add_argument("--setting", default="underspecified", choices=SETTINGS)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--token-budget", type=int, default=None)
    args = ap.parse_args(argv)

    cell = Cell(args.system, args.backbone, args.population, args.setting, args.split)
    if cell.split == "test":
        from experiments.staging import assert_frozen

        assert_frozen(args.data)
    try:
        s = run_cell(cell, load_instances(args.data), args.out, token_budget=args.token_budget)
    except CellInterrupted as exc:
        print(f"INTERRUPTED (resumable): {exc}", file=sys.stderr)
        return 3
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
