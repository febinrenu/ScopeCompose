"""Run the whole local matrix, one model in memory at a time.

    python -m study.schedule local            # ConditionalQA + RAMDocs, both local models
    python -m study.schedule local --only cqa
    python -m study.schedule hosted           # the replication cells on Groq (quota-paced)

Cells are resumable, so the schedule can be stopped and restarted at any point.
Between local models the previous one is unloaded from Ollama: two 7-8B models
resident at once, plus the GPU scorers, is more than a 16 GB laptop holds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from api_budget import RateLimitError
from study import operators, qualifier  # noqa: F401  (register systems)
from study import ram
from study.run import CellInterrupted, MODELS, run_cell

LOCAL = ("qwen7b", "llama8b")
HOSTED = ("oss20b", "qwen27b", "oss120b")

#: Ordered so the confirmatory cells finish first (analysis plan, "Local matrix, trimmed").
CQA_CONFIRMATORY = ("gold", "gold_nocondel", "gold_nocondchk", "goldpad8", "top8", "rr8to3", "nli", "crs", "qc")
CQA_EXPLORATORY = ("closed", "top3", "adj", "page", "qc0", "select")
RAM_SYSTEMS = ("full", "oracle", "rr3", "nli", "crs", "select", "grade")

#: Hosted cells: (system, strata kept). See the analysis plan, "Hosted cells".
HOSTED_CELLS = (("gold", None), ("top8", None), ("qc", None), ("gold_nocondchk", "conditional"))


def unload_local() -> None:
    for tag in ("scopecompose-qwen7b", "scopecompose-reader"):
        subprocess.run(["ollama", "stop", tag], capture_output=True, check=False)


def prepare_qualifier() -> None:
    """Train the cross-fitted scorers, score every candidate, choose the threshold -- each step once."""
    plan = qualifier.fold_plan()
    unload_local()
    for fold in plan:
        if not (qualifier.MODELS / fold / "done.json").exists():
            print(f"training qualifier fold {fold}", flush=True)
            # A fresh process per fold: the GPU starts empty every time.
            subprocess.run([sys.executable, "-m", "study.qualifier", "train", "--fold", fold], check=True)
    if not qualifier.SCORES.exists():
        print("scoring qualifier candidates", flush=True)
        qualifier.main(["score"])
    if not qualifier.THRESHOLD_FILE.exists():
        qualifier.main(["threshold"])


def local(only: str | None) -> int:
    status = 0
    if only in (None, "cqa"):
        prepare_qualifier()
    for m in LOCAL:
        unload_local()
        if only in (None, "cqa"):
            for s in CQA_CONFIRMATORY + CQA_EXPLORATORY:
                try:
                    print(json.dumps(run_cell(s, m, "eval", progress=True)), flush=True)
                except CellInterrupted as exc:
                    print(f"INTERRUPTED (resumable): {exc}", file=sys.stderr, flush=True)
                    status = 3
        if only in (None, "ramdocs"):
            for s in RAM_SYSTEMS:
                try:
                    print(json.dumps(ram.run_cell(s, m)), flush=True)
                except RuntimeError as exc:
                    print(f"INTERRUPTED (resumable): {exc}", file=sys.stderr, flush=True)
                    status = 3
    unload_local()
    return status


HOSTED_WAIT_S = 1200          # Groq's daily cap is a rolling window: tokens come back through the day
HOSTED_WORKERS = 4            # requests in flight per model; NVIDIA's free endpoint queues each one for minutes


def hosted(models: tuple[str, ...]) -> int:
    """Run the hosted cells. Several models: one child process each, so every model works through its
    own quotas in parallel. One model: its cells in turn; when every provider of the model is spent,
    wait and try again. The ``qc`` cell waits until the qualifier is trained (it needs its scores)."""
    from study import cqa

    if len(models) > 1:
        procs = [subprocess.Popen([sys.executable, "-m", "study.schedule", "hosted", "--models", m]) for m in models]
        return max(p.wait() for p in procs)

    pending = [(m, s, k) for s, k in HOSTED_CELLS for m in models]
    while pending:
        nxt = []
        for m, s, k in pending:
            qs = [q for q in cqa.hosted_questions() if k is None or q.kind == k]
            try:
                print(json.dumps(run_cell(s, m, "hosted", questions=qs, max_errors=3, workers=HOSTED_WORKERS,
                                          progress=True)), flush=True)
            except (CellInterrupted, RateLimitError) as exc:
                print(f"{m}/{s}: paused ({str(exc)[:160]})", file=sys.stderr, flush=True)
                nxt.append((m, s, k))
            except FileNotFoundError as exc:           # the qualifier is not trained yet
                print(f"{m}/{s}: waiting ({str(exc)[:120]})", file=sys.stderr, flush=True)
                nxt.append((m, s, k))
        pending = nxt
        if pending:
            time.sleep(HOSTED_WAIT_S)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=("local", "hosted"))
    ap.add_argument("--only", choices=("cqa", "ramdocs"))
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS))
    args = ap.parse_args(argv)
    if args.which == "local":
        return local(args.only)
    return hosted(tuple(args.models or HOSTED))


if __name__ == "__main__":
    raise SystemExit(main())
