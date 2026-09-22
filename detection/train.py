"""Train and persist the stage-1 conflict head.

Without a trained head, ``Stage1Filter`` falls back to a hand-weighted rule.
That fallback exists so the pipeline runs before any labels do -- it is not a
model, and measuring it shows why: on mock data it scores at chance, which
means "resolved locally at zero API cost" was counting pairs it had resolved
*wrongly*. The escalation rate is only a cost saving if the pairs stage 1
keeps are the ones it gets right.

This trains the learned head over the NLI probabilities, the embedding
interaction block, and the hand-authored lexical features, reports held-out
accuracy against the rule-based baseline, and saves it where
``Stage1Filter`` picks it up automatically.

Usage::

    python -m detection.train --data train.jsonl
    python -m detection.train --data train.jsonl --kind mlp --dev-fraction 0.3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contract.models import QueryRecord
from detection.head import HeadConfig, PairHead
from detection.stage1 import DEFAULT_HEAD_PATH, Stage1Filter, feature_names


@dataclass
class TrainReport:
    n_train: int
    n_dev: int
    n_features: int
    trained_accuracy: float
    baseline_accuracy: float
    positive_rate: float
    top_features: list[tuple[str, float]]

    @property
    def improvement(self) -> float:
        return self.trained_accuracy - self.baseline_accuracy

    def render(self) -> str:
        lines = [
            "Stage-1 head training",
            "=" * 72,
            f"  train pairs          {self.n_train:>8,}",
            f"  held-out pairs       {self.n_dev:>8,}",
            f"  features             {self.n_features:>8,}",
            f"  positive rate        {self.positive_rate:>8.1%}",
            "",
            f"  rule-based baseline  {self.baseline_accuracy:>8.3f}",
            f"  trained head         {self.trained_accuracy:>8.3f}",
            f"  improvement          {self.improvement:>+8.3f}",
        ]
        if self.improvement <= 0:
            lines += [
                "",
                "  WARNING: the trained head does not beat the hand-weighted rule.",
                "  Do not ship it. Either the feature set is not separating these",
                "  classes or there are too few labelled pairs to learn from -- both",
                "  are findings worth recording rather than papering over.",
            ]
        if self.top_features:
            lines += ["", "  Most influential features (variant (i) is meant to be inspectable):"]
            for name, w in self.top_features:
                lines.append(f"    {name:<28} {w:>+8.3f}")
        return "\n".join(lines)


def _split(records: list[QueryRecord], dev_fraction: float) -> tuple[list, list]:
    """Split by RECORD, never by pair.

    Two pairs from the same record share passages, so splitting by pair leaks
    the dev set into training and inflates held-out accuracy.
    """
    cut = int(len(records) * (1 - dev_fraction))
    cut = max(1, min(cut, len(records) - 1)) if len(records) > 1 else len(records)
    return records[:cut], records[cut:]


def _matrix(records: list[QueryRecord], stage1: Stage1Filter):
    X, y = [], []
    for rec in records:
        labelled = {p.key: p.is_conflict for p in rec.conflict_pairs}
        if not labelled:
            continue
        for cand in stage1.score_pairs(rec.passages):
            if cand.key not in labelled:
                continue
            X.append(cand.feature_vector())
            y.append(str(labelled[cand.key]))
    return np.array(X, dtype=np.float64) if X else np.zeros((0, 1)), y


def _rule_accuracy(records: list[QueryRecord], stage1: Stage1Filter) -> float:
    """Accuracy of the untrained fallback, for comparison."""
    correct = total = 0
    saved, stage1.head = stage1.head, None
    try:
        for rec in records:
            labelled = {p.key: p.is_conflict for p in rec.conflict_pairs}
            for cand in stage1.score_pairs(rec.passages):
                if cand.key not in labelled:
                    continue
                total += 1
                correct += int(cand.is_conflict == labelled[cand.key])
    finally:
        stage1.head = saved
    return correct / total if total else 0.0


def train(
    records: list[QueryRecord],
    *,
    dev_fraction: float = 0.25,
    kind: str = "logistic",
    heuristic_nli: bool = False,
) -> tuple[PairHead, TrainReport]:
    # auto_load_head=False: we are FITTING a head, so loading a previous one
    # would mean the reported rule-based baseline is not the rule-based
    # baseline, and a stale checkpoint would warn on every training run.
    stage1 = Stage1Filter(heuristic_nli=heuristic_nli, auto_load_head=False)
    train_recs, dev_recs = _split(records, dev_fraction)

    X_tr, y_tr = _matrix(train_recs, stage1)
    if len(X_tr) == 0:
        raise ValueError("no gold-labelled pairs in the training split")
    if len(set(y_tr)) < 2:
        raise ValueError(
            f"training split has only the label {set(y_tr)}. A conflict detector "
            "needs both conflicting and non-conflicting pairs -- check that the "
            "corpus includes distractors."
        )

    head = PairHead(HeadConfig(kind=kind), feature_names=feature_names()).fit(X_tr, y_tr)

    X_dev, y_dev = _matrix(dev_recs, stage1)
    if len(X_dev):
        preds = head.predict(X_dev)
        trained_acc = sum(p == t for p, t in zip(preds, y_dev)) / len(y_dev)
        baseline_acc = _rule_accuracy(dev_recs, stage1)
    else:
        trained_acc = baseline_acc = 0.0

    report = TrainReport(
        n_train=len(X_tr), n_dev=len(X_dev), n_features=X_tr.shape[1],
        trained_accuracy=trained_acc, baseline_accuracy=baseline_acc,
        positive_rate=sum(v == "True" for v in y_tr) / len(y_tr),
        top_features=head.top_features("True", n=8),
    )
    return head, report


def main(argv: list[str] | None = None) -> int:
    from detection.variants.compare import load_records

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dev-fraction", type=float, default=0.25)
    ap.add_argument("--kind", choices=["logistic", "mlp"], default="logistic")
    ap.add_argument("--heuristic-nli", action="store_true")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_HEAD_PATH)
    ap.add_argument("--force", action="store_true",
                    help="save even if the head does not beat the rule-based baseline")
    args = ap.parse_args(argv)

    records = load_records(args.data)
    if not records:
        print(f"no records in {args.data}", file=sys.stderr)
        return 1

    head, report = train(records, dev_fraction=args.dev_fraction, kind=args.kind,
                         heuristic_nli=args.heuristic_nli)
    print(report.render())

    if report.improvement <= 0 and not args.force:
        print(f"\nNOT saved. Pass --force to save it anyway.", file=sys.stderr)
        return 1

    head.save(args.output)
    print(f"\nsaved -> {args.output}")
    print("Stage1Filter now loads this automatically. Re-run "
          "`python -m detection.threshold` to retune the escalation band against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
