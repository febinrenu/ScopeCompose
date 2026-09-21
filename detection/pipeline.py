"""A2: the two-stage conflict detector.

Stage 1 (local) resolves the confident majority of pairs at zero API cost;
stage 2 (API) adjudicates only the uncertain band. The output is the
``conflict_pairs`` array of a contract record.

The stage-1/stage-2 split is reported, not just used: the fraction of pairs
resolved locally is the number that makes the cost claim in the paper
checkable, so ``DetectionStats`` is part of the interface rather than a
debugging convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contract.models import ConflictPair, ConflictType, DecidedBy, QueryRecord, ScopeRelation
from detection.stage1 import PairCandidate, Stage1Filter
from detection.stage2 import Stage2Judge


@dataclass
class DetectionStats:
    """Where each decision came from. Reported with every run."""

    pairs_total: int = 0
    resolved_stage1: int = 0
    escalated_stage2: int = 0
    stage2_cached: int = 0
    stage2_fell_back: int = 0
    conflicts_found: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    @property
    def escalation_rate(self) -> float:
        return self.escalated_stage2 / self.pairs_total if self.pairs_total else 0.0

    def merge(self, other: DetectionStats) -> None:
        self.pairs_total += other.pairs_total
        self.resolved_stage1 += other.resolved_stage1
        self.escalated_stage2 += other.escalated_stage2
        self.stage2_cached += other.stage2_cached
        self.stage2_fell_back += other.stage2_fell_back
        self.conflicts_found += other.conflicts_found
        for k, v in other.by_type.items():
            self.by_type[k] = self.by_type.get(k, 0) + v

    def render(self) -> str:
        lines = [
            "Detection",
            "-" * 56,
            f"  pairs examined       {self.pairs_total:>8,}",
            f"  resolved locally     {self.resolved_stage1:>8,}  "
            f"({1 - self.escalation_rate:.0%} at zero API cost)",
            f"  escalated to stage 2 {self.escalated_stage2:>8,}  "
            f"({self.stage2_cached:,} served from cache)",
            f"  conflicts found      {self.conflicts_found:>8,}",
        ]
        if self.by_type:
            lines.append("  by type:")
            for k, v in sorted(self.by_type.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<16} {v:>6,}")
        if self.stage2_fell_back:
            lines.append(
                f"  !! {self.stage2_fell_back:,} stage-2 call(s) fell back to the local model"
            )
        return "\n".join(lines)


class TwoStageDetector:
    """A2 end to end."""

    def __init__(
        self,
        *,
        stage1: Stage1Filter | None = None,
        stage2: Stage2Judge | None = None,
        use_stage2: bool = True,
        classifier=None,
    ):
        """
        Parameters
        ----------
        use_stage2
            Turn off to run fully offline. The detector still works; it just
            keeps stage 1's verdict on the uncertain band. Used by the tests
            and by the cost/accuracy ablation.
        classifier
            Optional A3 detector. When supplied, it assigns the five-class
            label and scope relation for pairs stage 1 resolved on its own.
            Without it, locally-resolved conflicts are labelled ``factual``,
            which is the honest default: stage 1 decides *whether* two
            passages disagree, not *how*.
        """
        self.stage1 = stage1 or Stage1Filter()
        self.stage2 = stage2 or Stage2Judge()
        self.use_stage2 = use_stage2
        self.classifier = classifier

    def detect(self, record: QueryRecord) -> tuple[QueryRecord, DetectionStats]:
        """Run detection and return a record with ``conflict_pairs`` filled in."""
        stats = DetectionStats()
        candidates = self.stage1.score_pairs(record.passages)
        stats.pairs_total = len(candidates)

        escalate = [c for c in candidates if c.needs_escalation] if self.use_stage2 else []
        stats.escalated_stage2 = len(escalate)
        stats.resolved_stage1 = len(candidates) - len(escalate)

        judgements = self.stage2.judge_many(record.query, escalate) if escalate else {}
        stats.stage2_cached = sum(1 for j in judgements.values() if j.cached)
        stats.stage2_fell_back = sum(1 for j in judgements.values() if j.fell_back)

        pairs: list[ConflictPair] = []
        for cand in candidates:
            pair = self._to_pair(cand, judgements.get(cand.key))
            pairs.append(pair)
            if pair.is_conflict:
                stats.conflicts_found += 1
            stats.by_type[pair.type.value] = stats.by_type.get(pair.type.value, 0) + 1

        return record.model_copy(update={"conflict_pairs": pairs}), stats

    def detect_all(self, records: list[QueryRecord]) -> tuple[list[QueryRecord], DetectionStats]:
        out, total = [], DetectionStats()
        for rec in records:
            r, s = self.detect(rec)
            out.append(r)
            total.merge(s)
        return out, total

    # -- pair construction ---------------------------------------------------- #

    def _to_pair(self, cand: PairCandidate, judgement) -> ConflictPair:
        if judgement is not None:
            return ConflictPair(
                doc_i=cand.doc_i, doc_j=cand.doc_j,
                is_conflict=judgement.is_conflict,
                type=judgement.type,
                scope_relation=judgement.scope_relation,
                confidence=judgement.confidence,
                decided_by=DecidedBy.STAGE2_API,
            )

        if not cand.is_conflict:
            return ConflictPair(
                doc_i=cand.doc_i, doc_j=cand.doc_j, is_conflict=False,
                type=ConflictType.NO_CONFLICT, scope_relation=None,
                confidence=cand.confidence, decided_by=DecidedBy.STAGE1_LOCAL,
            )

        ctype, relation = self._classify(cand)
        return ConflictPair(
            doc_i=cand.doc_i, doc_j=cand.doc_j, is_conflict=True,
            type=ctype, scope_relation=relation,
            confidence=cand.confidence, decided_by=DecidedBy.STAGE1_LOCAL,
        )

    def _classify(self, cand: PairCandidate) -> tuple[ConflictType, ScopeRelation | None]:
        if self.classifier is not None:
            return self.classifier.classify(cand)

        # No A3 classifier attached. Stage 1 established only THAT these
        # disagree, so label the weakest defensible thing rather than guessing
        # at 'conditional' -- an unearned conditional label would route to
        # composition and could fabricate a branch.
        return ConflictType.FACTUAL, None
