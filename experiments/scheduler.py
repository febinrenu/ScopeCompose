"""Run a list of cells to completion, unattended, through the free tier's daily caps.

A seven-day schedule on a 200K-tokens-a-day quota only works if the runs keep going while
nobody is watching, so this is the loop that does it:

* **Quota-aware.** Before each cell it reads the cost log, counts the tokens the cell's model
  has spent in the last 24 hours, and gives the cell only what is left (less a safety
  margin). It never asks the provider to refuse it.
* **Resumable by construction.** A cell stopped by quota or errors is simply retried later;
  ``run_cell`` skips everything already written, so waiting costs nothing and nothing is
  repeated.
* **Visible.** A heartbeat file is rewritten every cycle and a status file lists every
  job's progress, so "is it still alive?" is answered by a timestamp, not a guess.
* **Does not let the machine sleep** (Windows), and says so if it cannot.
* **Bounded.** ``max_rounds`` stops a run that is making no progress at all (a bad key, a
  retired model) instead of looping forever.

It never reorders work: jobs run in the order given, so the confirmatory cells listed first
finish first.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import settings
from api_budget.costlog import CostLog
from contract.gold import GoldInstance
from experiments.run_matrix import (
    Cell,
    CellInterrupted,
    CellMismatch,
    check_cell_file,
    items,
    read_done,
    run_cell,
    select,
)

SAFETY_MARGIN = 0.10          # keep this share of the daily cap unspent
DAILY_CAP = 200_000


@dataclass
class Job:
    cell: Cell
    expected: int = 0
    done: int = 0
    state: str = "pending"      # pending | running | waiting | done | failed
    note: str = ""

    @property
    def complete(self) -> bool:
        return self.expected > 0 and self.done >= self.expected


def expected_items(cell: Cell, instances: Iterable[GoldInstance]) -> int:
    return sum(len(items(i, cell)) for i in select(instances, cell))


def tokens_last_24h(model: str, log: CostLog, *, now: float | None = None) -> int:
    """Uncached tokens ``model`` has spent in the last 24 hours, from the cost log."""
    horizon = (time.time() if now is None else now) - 86_400
    total = 0
    for r in log.read():
        if r.model == model and not r.cached and not r.error and r.timestamp >= horizon:
            total += r.total_tokens
    return total


UNLIMITED = 10**12


def budget_for(model: str, log: CostLog, *, daily_cap: int = DAILY_CAP, margin: float = SAFETY_MARGIN,
               now: float | None = None, provider: str = "groq") -> int:
    """Tokens this model may still spend today, keeping a safety margin. Never negative.

    A local model (``provider="ollama"``) has no daily cap and is never throttled."""
    if provider == "ollama":
        return UNLIMITED
    return max(0, int(daily_cap * (1 - margin)) - tokens_last_24h(model, log, now=now))


@contextmanager
def prevent_sleep():
    """Ask Windows not to sleep while a run is going. A no-op elsewhere, and harmless if it fails."""
    ok = False
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))  # type: ignore[attr-defined]
    except Exception:
        ok = False
    try:
        yield ok
    finally:
        if ok:
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]
            except Exception:
                pass


def _write(path: Path | None, payload: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")


def run_schedule(
    cells: Sequence[Cell],
    instances: Sequence[GoldInstance],
    out_dir: Path,
    *,
    client=None,
    log: CostLog | None = None,
    daily_cap: int = DAILY_CAP,
    wait_s: float = 1800.0,
    max_rounds: int = 200,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    heartbeat: Path | None = None,
    status: Path | None = None,
) -> list[Job]:
    """Run every cell until all are complete or ``max_rounds`` passes with work left."""
    log = log if log is not None else CostLog()
    jobs = [Job(c, expected_items(c, instances)) for c in cells]
    for j in jobs:
        # Validate what is already on disk BEFORE trusting it: a cell file that looks complete but holds
        # answers from another model must fail, not be counted as done.
        try:
            j.done = len(check_cell_file(j.cell, out_dir / f"{j.cell.name}.jsonl"))
        except CellMismatch as exc:
            j.state, j.note = "failed", str(exc)[:160]
        else:
            if j.complete:
                j.state = "done"

    for rnd in range(1, max_rounds + 1):
        progressed = False
        for job in jobs:
            if job.complete or job.state == "failed":
                continue
            budget = budget_for(job.cell.model, log, daily_cap=daily_cap, now=clock(),
                                provider=settings.models().tier(job.cell.tier.value).provider)
            _write(heartbeat, {"time": clock(), "round": rnd, "job": job.cell.name, "budget": budget})
            if budget <= 0:
                job.state, job.note = "waiting", "daily quota spent"
                continue
            job.state = "running"
            before = job.done
            try:
                run_cell(job.cell, instances, out_dir, client=client, token_budget=budget)
            except CellInterrupted as exc:
                job.state, job.note = "waiting", str(exc)[:160]
            except CellMismatch as exc:
                job.state, job.note = "failed", str(exc)[:160]
            job.done = len(read_done(out_dir / f"{job.cell.name}.jsonl"))
            progressed = progressed or job.done > before
            if job.complete:
                job.state, job.note = "done", ""
            _write(status, {"time": clock(), "jobs": [{**asdict(j), "cell": j.cell.name, "complete": j.complete}
                                                       for j in jobs]})
        if all(j.complete or j.state == "failed" for j in jobs):
            break
        if not progressed:
            sleep(wait_s)
    _write(status, {"time": clock(), "jobs": [{**asdict(j), "cell": j.cell.name, "complete": j.complete}
                                               for j in jobs]})
    return jobs


def main(argv: list[str] | None = None) -> int:
    import argparse

    from experiments.run_matrix import BACKBONES, POPULATIONS, SYSTEM_IDS, load_instances

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--cell", action="append", required=True,
                    help="SYSTEM:BACKBONE:POPULATION[:SETTING], in the order to run")
    ap.add_argument("--wait", type=float, default=1800.0)
    ap.add_argument("--rounds", type=int, default=200)
    args = ap.parse_args(argv)

    cells = []
    for spec in args.cell:
        parts = spec.split(":")
        if len(parts) not in (3, 4) or parts[0] not in SYSTEM_IDS or parts[1] not in BACKBONES \
                or parts[2] not in POPULATIONS:
            ap.error(f"bad --cell {spec!r}")
        cells.append(Cell(parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else "underspecified", args.split))
    if args.split == "test":
        from experiments.staging import assert_frozen

        assert_frozen(args.data)
    with prevent_sleep() as awake:
        if not awake:
            print("note: could not stop the machine sleeping; keep it plugged in and awake")
        jobs = run_schedule(cells, load_instances(args.data), args.out, wait_s=args.wait, max_rounds=args.rounds,
                            heartbeat=args.out / "heartbeat.json", status=args.out / "status.json")
    for j in jobs:
        print(f"{j.state:<8} {j.done}/{j.expected}  {j.cell.name}  {j.note}")
    return 0 if all(j.complete for j in jobs) else 3


if __name__ == "__main__":
    raise SystemExit(main())
