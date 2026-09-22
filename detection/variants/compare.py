"""The three-architecture comparison.

Proposal section 5.2 commits to settling the detector architecture empirically
rather than assuming the modest, inspectable design is enough. This module is
that experiment.

The reported headline is **not** aggregate five-class accuracy. It is the
factual/conditional confusion cell, in both directions:

* ``conditional -> factual`` is the damaging one. A real exception labelled a
  plain contradiction routes to selection, the exception branch is discarded,
  and Suppression Rate rises. This is the project's own failure mode occurring
  inside the project's own pipeline.
* ``factual -> conditional`` is the opposite risk. A genuine contradiction
  labelled conditional routes to composition, which then presents two
  incompatible claims as if both were true under different conditions --
  inventing a scope that does not exist.

Aggregate accuracy hides both, because distractors dominate the label
distribution. So both cells are printed separately, and the variant with the
best cell -- not the best accuracy -- is the one reported as "the" detector.

Usage::

    python -m detection.variants.compare --data mock.jsonl --heuristic-nli
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from contract.gold import GoldInstance
from contract.models import ConflictType, QueryRecord, ScopeRelation
from detection.stage1 import Stage1Filter
from detection.variants.base import DetectorVariant, labels_from_record


@dataclass
class VariantResult:
    name: str
    description: str
    n_pairs: int = 0
    correct_type: int = 0
    correct_relation: int = 0
    n_with_relation: int = 0

    #: confusion[gold][predicted]
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    relation_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    error: str | None = None

    # -- headline numbers ------------------------------------------------------ #

    @property
    def type_accuracy(self) -> float:
        return self.correct_type / self.n_pairs if self.n_pairs else 0.0

    @property
    def relation_accuracy(self) -> float:
        return self.correct_relation / self.n_with_relation if self.n_with_relation else 0.0

    @property
    def conditional_as_factual(self) -> int:
        """The damaging cell: a real exception called a contradiction."""
        return self.confusion.get("conditional", {}).get("factual", 0)

    @property
    def factual_as_conditional(self) -> int:
        """The opposite cell: a contradiction called an exception."""
        return self.confusion.get("factual", {}).get("conditional", 0)

    @property
    def n_gold_conditional(self) -> int:
        return sum(self.confusion.get("conditional", {}).values())

    @property
    def n_gold_factual(self) -> int:
        return sum(self.confusion.get("factual", {}).values())

    @property
    def conditional_leak_rate(self) -> float:
        """Fraction of true exceptions lost to the factual label.

        The single number that decides which variant is reported, because it
        maps directly onto how often Member B's composition operator never
        gets invoked on an instance it exists for.
        """
        n = self.n_gold_conditional
        return self.conditional_as_factual / n if n else 0.0

    @property
    def factual_leak_rate(self) -> float:
        n = self.n_gold_factual
        return self.factual_as_conditional / n if n else 0.0

    @property
    def combined_leak_rate(self) -> float:
        """Both directions of the factual/conditional boundary, averaged.

        Selecting on ``conditional_leak_rate`` alone is gameable, and not
        theoretically: a detector that labels everything ``conditional``
        scores a perfect 0% on it while being useless. That is the same
        caveat-happy failure the Spurious-Condition Rate exists to catch, so
        the detector-selection criterion has to be symmetric too.
        """
        return (self.conditional_leak_rate + self.factual_leak_rate) / 2

    @property
    def degenerate(self) -> bool:
        """True if the variant collapses onto one predicted label.

        A detector that predicts a single class for almost everything can look
        excellent on whichever one-directional metric that class happens to
        favour. Flagged rather than silently ranked.
        """
        if not self.n_pairs:
            return False
        predicted: dict[str, int] = {}
        for row in self.confusion.values():
            for label, n in row.items():
                predicted[label] = predicted.get(label, 0) + n
        return bool(predicted) and max(predicted.values()) / self.n_pairs >= 0.8

    def per_type_f1(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for label in [c.value for c in ConflictType]:
            tp = self.confusion.get(label, {}).get(label, 0)
            fn = sum(self.confusion.get(label, {}).values()) - tp
            fp = sum(row.get(label, 0) for g, row in self.confusion.items() if g != label)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            out[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4), "support": tp + fn}
        return out

    #: The confusion cells the proposal names as hardest to annotate, and so
    #: hardest to predict. Reported separately from aggregate relation accuracy.
    def relation_cells(self) -> dict[str, int]:
        rc = self.relation_confusion
        return {
            "refinement_as_redundant": rc.get("refinement", {}).get("redundant", 0),
            "redundant_as_refinement": rc.get("redundant", {}).get("refinement", 0),
            "refinement_as_opposed": rc.get("refinement", {}).get("opposed", 0),
            "opposed_as_refinement": rc.get("opposed", {}).get("refinement", 0),
        }


def _bump(table: dict[str, dict[str, int]], gold: str, pred: str) -> None:
    table.setdefault(gold, {})
    table[gold][pred] = table[gold].get(pred, 0) + 1


def evaluate(
    variant: DetectorVariant,
    records: list[QueryRecord],
    stage1: Stage1Filter,
) -> VariantResult:
    """Score one variant on labelled records."""
    res = VariantResult(name=variant.name, description=variant.description)

    for rec in records:
        gold = labels_from_record(rec)
        if not gold:
            continue
        candidates = [c for c in stage1.score_pairs(rec.passages) if c.key in gold]
        if not candidates:
            continue

        try:
            predictions = variant.classify_many(candidates)
        except Exception as exc:  # a variant that cannot run is a result too
            res.error = f"{type(exc).__name__}: {exc}"
            return res

        for cand, (p_type, p_rel) in zip(candidates, predictions):
            g_type, g_rel = gold[cand.key]
            res.n_pairs += 1
            res.correct_type += int(p_type is g_type)
            _bump(res.confusion, g_type.value, p_type.value)

            if g_rel is not None:
                res.n_with_relation += 1
                res.correct_relation += int(p_rel is g_rel)
                _bump(res.relation_confusion, g_rel.value,
                      p_rel.value if p_rel else "none")

    return res


def render(results: list[VariantResult]) -> str:
    lines = [
        "Detector architecture comparison (A3)",
        "=" * 86,
        "",
        "The reported headline is the factual/conditional confusion cell, not aggregate",
        "accuracy: distractors dominate the label distribution, so accuracy can look",
        "healthy while every real exception is being routed to selection.",
        "",
        "Both directions are shown, and selection uses both: a detector that labels",
        "everything conditional scores a perfect cond->fact while being useless.",
        "",
        f"  {'variant':<24} {'type acc':>9} {'rel acc':>9} {'cond->fact':>11} "
        f"{'fact->cond':>11} {'combined':>9}",
        "  " + "-" * 92,
    ]

    for r in results:
        if r.error:
            lines.append(f"  {r.name:<24} {'ERROR':>9}   {r.error[:44]}")
            continue
        flag = "  DEGENERATE" if r.degenerate else ""
        lines.append(
            f"  {r.name:<24} {r.type_accuracy:>9.3f} {r.relation_accuracy:>9.3f} "
            f"{r.conditional_as_factual:>4}/{r.n_gold_conditional:<6} "
            f"{r.factual_as_conditional:>4}/{r.n_gold_factual:<6} "
            f"{r.combined_leak_rate:>9.3f}{flag}"
        )

    usable = [r for r in results if not r.error and r.n_gold_conditional]
    healthy = [r for r in usable if not r.degenerate]
    pool = healthy or usable

    if pool:
        best = min(pool, key=lambda r: (r.combined_leak_rate, -r.type_accuracy))
        lines += [
            "",
            f"  Best on the combined factual/conditional boundary: {best.name}",
            f"    cond->fact {best.conditional_leak_rate:.1%} "
            f"(true exceptions lost to selection)",
            f"    fact->cond {best.factual_leak_rate:.1%} "
            f"(real contradictions sent to composition)",
            f"    accuracy   {best.type_accuracy:.3f}",
            "  -> this is the variant to report as 'the' detector, per proposal section 5.2.",
        ]

    degenerate = [r for r in usable if r.degenerate]
    if degenerate:
        lines += [
            "",
            "  EXCLUDED as degenerate (one label covers >=80% of predictions):",
        ]
        for r in degenerate:
            lines.append(
                f"    {r.name} - scores well on one direction of the cell only because it "
                f"collapses onto a single class. This is the caveat-happy failure the "
                f"Spurious-Condition Rate exists to catch."
            )
        if not healthy:
            lines.append("    NOTE: every variant is degenerate, so the pick above is "
                         "the least bad, not a good one.")

    for r in results:
        if r.error:
            continue
        lines += ["", f"{r.name}", "-" * 86, f"  {r.description}", "",
                  "  per-type:",
                  f"    {'label':<16} {'prec':>7} {'recall':>7} {'f1':>7} {'n':>6}"]
        for label, m in r.per_type_f1().items():
            if m["support"] == 0:
                continue
            lines.append(
                f"    {label:<16} {m['precision']:>7.3f} {m['recall']:>7.3f} "
                f"{m['f1']:>7.3f} {m['support']:>6}"
            )

        cells = r.relation_cells()
        if any(cells.values()) or r.n_with_relation:
            lines += ["", "  scope-relation confusion cells "
                          "(the boundaries proposal section 8 expects to be hardest):"]
            for k, v in cells.items():
                lines.append(f"    {k:<28} {v:>6}")

    return "\n".join(lines)


def load_records(path: Path) -> list[QueryRecord]:
    """Read either gold instances or wire records; return wire records."""
    records: list[QueryRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "gold_conflict_type" in obj:
                records.append(GoldInstance.model_validate(obj).to_query_record())
            else:
                records.append(QueryRecord.model_validate(obj))
    return records


def build_variants(*, heuristic_nli: bool, stage1: Stage1Filter,
                   include_finetuned: bool) -> list[DetectorVariant]:
    from detection.variants.lexical_nli import LexicalNLIDetector
    from detection.variants.structured_entailment import StructuredEntailmentDetector

    variants: list[DetectorVariant] = [
        LexicalNLIDetector(stage1=stage1),
        StructuredEntailmentDetector(nli=stage1.nli),
    ]
    if include_finetuned:
        from detection.variants.finetuned import FineTunedDetector

        variants.append(FineTunedDetector())
    return variants


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="JSONL of gold instances or records")
    ap.add_argument("--train-split", type=float, default=0.6,
                    help="fraction used to train the trainable variants (default 0.6)")
    ap.add_argument("--heuristic-nli", action="store_true",
                    help="use the lexical NLI stand-in: no model download, no GPU. "
                         "For smoke-testing the harness only -- never for a reported number.")
    ap.add_argument("--include-finetuned", action="store_true",
                    help="also train and evaluate variant (ii). Needs torch and a GPU.")
    ap.add_argument("--json", type=Path, default=None, help="also write raw results here")
    args = ap.parse_args(argv)

    records = load_records(args.data)
    if not records:
        print(f"no records in {args.data}", file=sys.stderr)
        return 1

    cut = max(1, int(len(records) * args.train_split))
    train, test = records[:cut], records[cut:] or records[:1]
    print(f"{len(records)} records: {len(train)} train, {len(test)} test\n")

    if args.heuristic_nli:
        print("NOTE: running with the heuristic NLI stand-in. These numbers exercise the")
        print("      harness; they are not a measurement of any variant's real accuracy.\n")

    stage1 = Stage1Filter(heuristic_nli=args.heuristic_nli)
    variants = build_variants(heuristic_nli=args.heuristic_nli, stage1=stage1,
                              include_finetuned=args.include_finetuned)

    results = []
    for v in variants:
        if v.requires_training or hasattr(v, "fit"):
            try:
                v.fit(train)
            except Exception as exc:
                if v.requires_training:
                    results.append(VariantResult(name=v.name, description=v.description,
                                                 error=f"training failed: {exc}"))
                    continue
        results.append(evaluate(v, test, stage1))

    print(render(results))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nraw results -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
