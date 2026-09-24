"""Detection and classification scorers (Member A's contribution to the shared
metrics library).

Member B leads ``metrics/`` overall and owns the preservation metrics; these
are the scorers for A's own outputs, kept here rather than in ``detection/`` so
that the evaluation harness has one import path for everything.

The design principle across all of them: report the cell, not just the
aggregate. Distractors dominate the label distribution by construction -- they
have to, or Spurious-Condition Rate has no denominator -- which means
aggregate accuracy can look healthy while every conditional instance is being
misrouted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contract.models import ConflictPair, ConflictType, QueryRecord, ScopeRelation
from metrics.stats import Interval, bootstrap, mcnemar, wilson


# --------------------------------------------------------------------------- #
# Binary detection (A2)
# --------------------------------------------------------------------------- #


@dataclass
class BinaryScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    # -- uncertainty --------------------------------------------------------- #
    #
    # Precision and recall are proportions with a clear denominator, so Wilson
    # is exact enough and needs no resampling. F1 is a ratio of ratios and has
    # no closed form, so it is left to the bootstrap in score_detection_ci().

    def precision_ci(self, alpha: float = 0.05) -> Interval:
        return wilson(self.tp, self.tp + self.fp, alpha=alpha)

    def recall_ci(self, alpha: float = 0.05) -> Interval:
        return wilson(self.tp, self.tp + self.fn, alpha=alpha)

    def accuracy_ci(self, alpha: float = 0.05) -> Interval:
        return wilson(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn,
                      alpha=alpha)

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision_ci": self.precision_ci().as_dict(),
            "recall_ci": self.recall_ci().as_dict(),
        }

    def render(self, title: str = "Binary conflict detection") -> str:
        return (
            f"{title}\n" + "-" * 56 + "\n"
            f"  precision {self.precision_ci().render()}\n"
            f"  recall    {self.recall_ci().render()}\n"
            f"  f1        {self.f1:.4f}   (interval via score_detection_ci)\n"
            f"  tp {self.tp:<6} fp {self.fp:<6} fn {self.fn:<6} tn {self.tn:<6}"
        )


def score_detection(
    predicted: list[QueryRecord], gold: list[QueryRecord]
) -> BinaryScores:
    """Binary conflict-detection P/R/F1 over pairs.

    Pairs are matched by canonical key, so the score is invariant to the order
    passages arrived in and to the order pairs are listed in.
    """
    scores = BinaryScores()
    gold_by_id = {r.query_id: r for r in gold}

    for pred in predicted:
        g = gold_by_id.get(pred.query_id)
        if g is None:
            continue
        gold_pairs = {p.key: p.is_conflict for p in g.conflict_pairs}
        for pair in pred.conflict_pairs:
            if pair.key not in gold_pairs:
                continue
            truth = gold_pairs[pair.key]
            if pair.is_conflict and truth:
                scores.tp += 1
            elif pair.is_conflict and not truth:
                scores.fp += 1
            elif not pair.is_conflict and truth:
                scores.fn += 1
            else:
                scores.tn += 1
    return scores


# --------------------------------------------------------------------------- #
# Five-class classification (A3)
# --------------------------------------------------------------------------- #


@dataclass
class ConfusionMatrix:
    """Gold-by-predicted counts, with named cells."""

    labels: list[str]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, gold: str, predicted: str) -> None:
        self.counts.setdefault(gold, {})
        self.counts[gold][predicted] = self.counts[gold].get(predicted, 0) + 1

    def cell(self, gold: str, predicted: str) -> int:
        return self.counts.get(gold, {}).get(predicted, 0)

    def support(self, gold: str) -> int:
        return sum(self.counts.get(gold, {}).values())

    @property
    def total(self) -> int:
        return sum(self.support(g) for g in self.counts)

    @property
    def correct(self) -> int:
        return sum(self.cell(label, label) for label in self.labels)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def per_label(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for label in self.labels:
            tp = self.cell(label, label)
            fn = self.support(label) - tp
            fp = sum(self.cell(g, label) for g in self.counts if g != label)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            out[label] = {
                "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
                "support": self.support(label),
            }
        return out

    @property
    def macro_f1(self) -> float:
        rows = [m for m in self.per_label().values() if m["support"]]
        return round(sum(m["f1"] for m in rows) / len(rows), 4) if rows else 0.0

    @property
    def predicted_labels(self) -> list[str]:
        """Every label that was actually predicted, declared or not.

        A system can predict something outside ``labels`` -- most commonly
        ``"none"``, when a detector declines to assign a scope relation.
        Those predictions must appear in the matrix: hiding them makes the
        rows stop summing to their totals, which turns a diagnostic into a
        misleading one.
        """
        seen = {p for row in self.counts.values() for p in row}
        extra = sorted(seen - set(self.labels))
        return self.labels + extra

    def render(self, title: str = "Confusion matrix") -> str:
        cols = [
            l for l in self.predicted_labels
            if self.support(l) or any(self.cell(g, l) for g in self.counts)
        ]
        rows = [l for l in self.predicted_labels if self.support(l)]
        w = max((len(l) for l in cols + rows), default=8) + 2

        lines = [title, "-" * 56,
                 " " * w + "".join(f"{l[:10]:>12}" for l in cols) + f"{'total':>12}"]
        for g in rows:
            body = "".join(f"{self.cell(g, p):>12}" for p in cols)
            lines.append(f"{g:<{w}}{body}{self.support(g):>12}")

        lines.append(f"\n  accuracy {self.accuracy:.4f}   macro-F1 {self.macro_f1:.4f}")

        undeclared = [l for l in cols if l not in self.labels]
        if undeclared:
            n = sum(self.cell(g, l) for g in self.counts for l in undeclared)
            lines.append(
                f"  NOTE: {n} prediction(s) fell outside the declared label set "
                f"({', '.join(undeclared)}) -- most often a detector declining to "
                f"assign a relation at all."
            )
        return "\n".join(lines)


@dataclass
class ClassificationScores:
    matrix: ConfusionMatrix

    # -- the cells that actually decide things -------------------------------- #

    @property
    def conditional_as_factual(self) -> int:
        """A real exception labelled a plain contradiction.

        The damaging direction. It routes the pair to selection, the exception
        branch is discarded, and Suppression Rate rises -- the project's own
        failure mode, occurring inside the project's own pipeline.
        """
        return self.matrix.cell(ConflictType.CONDITIONAL.value, ConflictType.FACTUAL.value)

    @property
    def factual_as_conditional(self) -> int:
        """A genuine contradiction labelled an exception.

        The opposite risk: routes to composition, which then presents two
        incompatible claims as if both held under different conditions,
        inventing a scope that does not exist.
        """
        return self.matrix.cell(ConflictType.FACTUAL.value, ConflictType.CONDITIONAL.value)

    @property
    def conditional_leak_rate(self) -> float:
        n = self.matrix.support(ConflictType.CONDITIONAL.value)
        return round(self.conditional_as_factual / n, 4) if n else 0.0

    @property
    def factual_leak_rate(self) -> float:
        n = self.matrix.support(ConflictType.FACTUAL.value)
        return round(self.factual_as_conditional / n, 4) if n else 0.0

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.matrix.accuracy, 4),
            "macro_f1": self.matrix.macro_f1,
            "per_label": self.matrix.per_label(),
            "conditional_as_factual": self.conditional_as_factual,
            "factual_as_conditional": self.factual_as_conditional,
            "conditional_leak_rate": self.conditional_leak_rate,
            "factual_leak_rate": self.factual_leak_rate,
        }

    def render(self) -> str:
        return "\n".join([
            self.matrix.render("Five-class conflict classification (A3)"),
            "",
            "  The headline cell -- reported separately because aggregate accuracy",
            "  hides it (distractors dominate the label distribution):",
            f"    conditional -> factual   {self.conditional_as_factual:>5}  "
            f"({self.conditional_leak_rate:.1%} of true exceptions lost to selection)",
            f"    factual -> conditional   {self.factual_as_conditional:>5}  "
            f"({self.factual_leak_rate:.1%} of real contradictions sent to composition)",
        ])


def score_classification(
    predicted: list[QueryRecord], gold: list[QueryRecord]
) -> ClassificationScores:
    matrix = ConfusionMatrix(labels=[c.value for c in ConflictType])
    gold_by_id = {r.query_id: r for r in gold}

    for pred in predicted:
        g = gold_by_id.get(pred.query_id)
        if g is None:
            continue
        gold_pairs = {p.key: p for p in g.conflict_pairs}
        for pair in pred.conflict_pairs:
            truth = gold_pairs.get(pair.key)
            if truth is not None:
                matrix.add(truth.type.value, pair.type.value)
    return ClassificationScores(matrix=matrix)


# --------------------------------------------------------------------------- #
# Four-way scope relation (A4)
# --------------------------------------------------------------------------- #


@dataclass
class ScopeScores:
    matrix: ConfusionMatrix

    # The two boundaries the proposal expects to be hardest to annotate, and so
    # hardest to predict. Reported separately from aggregate accuracy, because
    # each has a distinct downstream consequence.

    @property
    def refinement_as_redundant(self) -> int:
        """A real exception merged away. The branch is lost silently."""
        return self.matrix.cell(ScopeRelation.REFINEMENT.value, ScopeRelation.REDUNDANT.value)

    @property
    def redundant_as_refinement(self) -> int:
        """A restatement composed as an exception. Fabricates a branch, which
        inflates Preservation Rate while corrupting the answer."""
        return self.matrix.cell(ScopeRelation.REDUNDANT.value, ScopeRelation.REFINEMENT.value)

    @property
    def refinement_as_opposed(self) -> int:
        """A valid exception treated as a contradiction. Falls back to
        selection unnecessarily -- exactly the behaviour this project exists
        to remove."""
        return self.matrix.cell(ScopeRelation.REFINEMENT.value, ScopeRelation.OPPOSED.value)

    @property
    def opposed_as_refinement(self) -> int:
        """A real contradiction composed as an exception. Presents two
        incompatible claims as both true."""
        return self.matrix.cell(ScopeRelation.OPPOSED.value, ScopeRelation.REFINEMENT.value)

    def named_cells(self) -> dict[str, int]:
        return {
            "refinement_as_redundant": self.refinement_as_redundant,
            "redundant_as_refinement": self.redundant_as_refinement,
            "refinement_as_opposed": self.refinement_as_opposed,
            "opposed_as_refinement": self.opposed_as_refinement,
        }

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.matrix.accuracy, 4),
            "macro_f1": self.matrix.macro_f1,
            "per_label": self.matrix.per_label(),
            "named_cells": self.named_cells(),
        }

    def render(self) -> str:
        consequences = {
            "refinement_as_redundant": "exception merged away, branch lost silently",
            "redundant_as_refinement": "restatement composed as an exception, branch fabricated",
            "refinement_as_opposed": "valid exception sent to selection unnecessarily",
            "opposed_as_refinement": "real contradiction composed as if both were true",
        }
        lines = [
            self.matrix.render("Four-way scope relation (A4)"),
            "",
            "  Confusion cells the proposal names as hardest, with their consequences:",
        ]
        for name, count in self.named_cells().items():
            lines.append(f"    {name:<28} {count:>5}   {consequences[name]}")
        return "\n".join(lines)


def score_scope_relations(
    predicted: list[QueryRecord], gold: list[QueryRecord]
) -> ScopeScores:
    """Score the four-way relation on pairs where gold has one.

    Pairs whose gold label carries no relation are skipped rather than counted
    as errors: a factual pair has no correct relation, so scoring one against
    it would measure nothing.
    """
    matrix = ConfusionMatrix(labels=[r.value for r in ScopeRelation])
    gold_by_id = {r.query_id: r for r in gold}

    for pred in predicted:
        g = gold_by_id.get(pred.query_id)
        if g is None:
            continue
        gold_pairs = {p.key: p for p in g.conflict_pairs}
        for pair in pred.conflict_pairs:
            truth = gold_pairs.get(pair.key)
            if truth is None or truth.scope_relation is None:
                continue
            matrix.add(
                truth.scope_relation.value,
                pair.scope_relation.value if pair.scope_relation else "none",
            )
    return ScopeScores(matrix=matrix)


# --------------------------------------------------------------------------- #
# Tier breakdown
# --------------------------------------------------------------------------- #


def split_by_construction(
    records: list[QueryRecord],
) -> dict[str, list[QueryRecord]]:
    """Partition records by Tier 1 / Tier 2.

    Every headline result is reported for the two tiers separately. Blending
    them would let constructed instances inflate the naturally-occurring claim
    that distinguishes this benchmark from prior single-document work, so the
    metrics library makes splitting the easy path and blending the deliberate one.
    """
    out: dict[str, list[QueryRecord]] = {}
    for r in records:
        out.setdefault(r.construction.value, []).append(r)
    return out


# --------------------------------------------------------------------------- #
# Interval-bearing variants of the scorers
# --------------------------------------------------------------------------- #
#
# These resample INSTANCES, not pairs. Ten pairs from one record share
# passages; resampling them independently pretends the corpus holds more
# information than it does and returns intervals that are too narrow. The unit
# of independence is the instance, so that is the unit of resampling.


def _paired(predicted: list[QueryRecord], gold: list[QueryRecord]
            ) -> list[tuple[QueryRecord, QueryRecord]]:
    gold_by_id = {r.query_id: r for r in gold}
    return [(p, gold_by_id[p.query_id]) for p in predicted if p.query_id in gold_by_id]


def score_detection_ci(
    predicted: list[QueryRecord],
    gold: list[QueryRecord],
    *,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Interval]:
    """Bootstrap intervals for detection F1, precision and recall."""
    units = _paired(predicted, gold)

    def _scores(sample) -> BinaryScores:
        return score_detection([p for p, _ in sample], [g for _, g in sample])

    return {
        "f1": bootstrap(units, lambda s: _scores(s).f1,
                        n_resamples=n_resamples, alpha=alpha, seed=seed),
        "precision": bootstrap(units, lambda s: _scores(s).precision,
                               n_resamples=n_resamples, alpha=alpha, seed=seed),
        "recall": bootstrap(units, lambda s: _scores(s).recall,
                            n_resamples=n_resamples, alpha=alpha, seed=seed),
    }


def score_classification_ci(
    predicted: list[QueryRecord],
    gold: list[QueryRecord],
    *,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Interval]:
    """Intervals for the numbers the detector is actually judged on.

    The two leak rates matter more than accuracy here, and they are computed
    over small denominators -- the conditional and factual subsets -- so their
    intervals are wide. Reporting the point estimate alone would suggest a
    precision the corpus size does not support.
    """
    units = _paired(predicted, gold)

    def _s(sample) -> ClassificationScores:
        return score_classification([p for p, _ in sample], [g for _, g in sample])

    return {
        "accuracy": bootstrap(units, lambda s: _s(s).matrix.accuracy,
                              n_resamples=n_resamples, alpha=alpha, seed=seed),
        "macro_f1": bootstrap(units, lambda s: _s(s).matrix.macro_f1,
                              n_resamples=n_resamples, alpha=alpha, seed=seed),
        "conditional_leak_rate": bootstrap(
            units, lambda s: _s(s).conditional_leak_rate,
            n_resamples=n_resamples, alpha=alpha, seed=seed),
        "factual_leak_rate": bootstrap(
            units, lambda s: _s(s).factual_leak_rate,
            n_resamples=n_resamples, alpha=alpha, seed=seed),
    }


def score_scope_ci(
    predicted: list[QueryRecord],
    gold: list[QueryRecord],
    *,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Interval]:
    units = _paired(predicted, gold)

    def _s(sample) -> ScopeScores:
        return score_scope_relations([p for p, _ in sample], [g for _, g in sample])

    return {
        "four_way_accuracy": bootstrap(units, lambda s: _s(s).matrix.accuracy,
                                       n_resamples=n_resamples, alpha=alpha, seed=seed),
        "macro_f1": bootstrap(units, lambda s: _s(s).matrix.macro_f1,
                              n_resamples=n_resamples, alpha=alpha, seed=seed),
    }


def compare_detectors_mcnemar(
    predicted_a: list[QueryRecord],
    predicted_b: list[QueryRecord],
    gold: list[QueryRecord],
):
    """McNemar's test between two detectors on the same pairs.

    The three A3 variants are scored on identical pairs, so the comparison is
    paired. An unpaired test discards exactly the information that makes a
    modest but consistent difference detectable, and on a few hundred pairs
    that is usually the difference between a result and a shrug.

    Correctness is per-pair agreement with the gold five-class label.
    """
    gold_by_id = {r.query_id: {p.key: p.type for p in r.conflict_pairs} for r in gold}
    by_a = {r.query_id: {p.key: p.type for p in r.conflict_pairs} for r in predicted_a}
    by_b = {r.query_id: {p.key: p.type for p in r.conflict_pairs} for r in predicted_b}

    correct_a: list[bool] = []
    correct_b: list[bool] = []
    for qid, gold_pairs in gold_by_id.items():
        a_pairs, b_pairs = by_a.get(qid, {}), by_b.get(qid, {})
        for key, truth in gold_pairs.items():
            if key not in a_pairs or key not in b_pairs:
                continue   # only items BOTH systems judged are comparable
            correct_a.append(a_pairs[key] is truth)
            correct_b.append(b_pairs[key] is truth)

    return mcnemar(correct_a, correct_b)
