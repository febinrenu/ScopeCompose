"""Run gate V1 (the validity battery) with the real reader and NLI model, on development instances.

    python -m experiments.validity_battery --data benchmark/data/gold_dev.jsonl --out experiments/runs/v1.json

The battery scores eight deliberately gameable outputs per instance (``metrics.validity``) and checks
the criteria fixed in that module. Nothing here is a system under test, so it spends no API tokens: the
reader is the local Llama-3.1-8B and the entailment model is the cached DeBERTa.

Two rules:

* **never the test split.** The battery is a property of the metrics, established on development data
  before the freeze; running it on test would put test gold in front of reader/threshold choices;
* **instances that cannot be scored are an error, not a skip.** A battery over a silently thinned set is
  not the battery the criteria were written for, so a file with an instance lacking personas, branches or
  reference answers is refused with the offending ids.

The result is written as JSON (every output's numbers, every criterion's verdict, the criteria hash and the
threshold, with whether it was tuned) so the claims ledger can cite it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from contract.gold import GoldInstance
from metrics.case_accuracy import NLIScorerLike, Reader
from metrics.validity import OUTPUTS, V1_CRITERIA_SHA, BatteryResult, run_battery

DEFAULT_THRESHOLD = 0.5


def scorable(instances: Sequence[GoldInstance]) -> tuple[list[GoldInstance], list[str]]:
    """(instances the battery can score, ids of those it cannot)."""
    ok, bad = [], []
    for i in instances:
        (ok if (i.personas and i.gold_branches and i.gold_scoped_answer and i.selection_answer) else bad).append(i)
    return ok, [i.instance_id for i in bad]


def select(instances: Sequence[GoldInstance], *, split: str) -> list[GoldInstance]:
    if split == "test":
        raise ValueError("the validity battery is run on development data, never on the test split")
    return [i for i in instances if i.split == split]


def to_json(res: BatteryResult) -> dict:
    return {
        "n_instances": res.n_instances,
        "criteria_sha256": V1_CRITERIA_SHA,
        "nli_threshold": res.nli_threshold,
        "nli_threshold_tuned": res.nli_threshold_tuned,
        "outputs": {o: {"legacy_pr": res.legacy_pr[o], "cca_strict": res.cca[o].cca_all.point,
                        "cca_lenient": res.cca[o].cca_all_lenient.point, "scoped_precision": res.scoped[o]}
                    for o in OUTPUTS},
        "criteria": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in res.criteria()],
        "v1_passed": res.v1_passed,
    }


def run(instances: Sequence[GoldInstance], reader: Reader, nli: NLIScorerLike, *, split: str = "dev",
        threshold: float = DEFAULT_THRESHOLD, tuned: bool = False) -> BatteryResult:
    chosen = select(instances, split=split)
    if not chosen:
        raise ValueError(f"no instances in split {split!r}")
    ok, bad = scorable(chosen)
    if bad:
        raise ValueError(f"{len(bad)} instance(s) cannot be scored (need personas, branches and both reference "
                         f"answers): {', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}")
    return run_battery(ok, reader, nli, threshold=threshold, threshold_tuned=tuned)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="gold instances, one JSON object per line")
    ap.add_argument("--out", type=Path, default=Path("experiments/runs/validity_v1.json"))
    ap.add_argument("--split", default="dev", choices=("train", "dev"))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--tuned", action="store_true", help="the threshold was calibrated from the entailment audit")
    args = ap.parse_args(argv)

    from detection.nli import CrossEncoderNLI
    from experiments.run_matrix import load_instances
    from metrics.case_accuracy import LLMReader

    res = run(load_instances(args.data), LLMReader(step="validity_battery"), CrossEncoderNLI(),
              split=args.split, threshold=args.threshold, tuned=args.tuned)
    print(res.render())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_json(res), indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0 if res.v1_passed else 1


if __name__ == "__main__":
    sys.exit(main())
