"""A small, fixed tuning-set sample for checking a model before spending quota on it.

    python -m study.probe --model qwen27b oss20b --system gold

30 conditional and 30 unconditional tuning questions (seed 0). Cells go to
``study/runs/tuning`` like any tuning cell, so they are scored the same way.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from study import cqa, reader
from study.run import MODELS, CellInterrupted, run_cell

N_PER_STRATUM = 30


def probe_questions(seed: int = 0) -> list[cqa.Question]:
    pool = [q for q in cqa.tuning_questions() if q.id not in reader.DEMO_QUESTION_IDS]
    rng = random.Random(seed)
    out: list[cqa.Question] = []
    for kind in ("conditional", "unconditional"):
        qs = sorted((q for q in pool if q.kind == kind), key=lambda q: q.id)
        out += rng.sample(qs, min(N_PER_STRATUM, len(qs)))
    return sorted(out, key=lambda q: q.id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--system", nargs="+", default=["gold"])
    args = ap.parse_args(argv)
    status = 0
    for m in args.model:
        for s in args.system:
            try:
                print(json.dumps(run_cell(s, m, "tuning", questions=probe_questions())), flush=True)
            except CellInterrupted as exc:
                print(f"INTERRUPTED (resumable): {exc}", file=sys.stderr, flush=True)
                status = 3
    return status


if __name__ == "__main__":
    raise SystemExit(main())
