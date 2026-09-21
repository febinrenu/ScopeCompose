"""The interface all three A3 detector variants implement.

Proposal section 5.2 commits to comparing three architectures head-to-head
rather than assuming the modest, inspectable one is sufficient. A shared
protocol is what makes that comparison mechanically fair: the same data, the
same harness, the same metric, and the only thing varying is the model.

The headline number is NOT five-class accuracy. It is the
**factual/conditional confusion cell** -- how often a real exception gets
labelled a plain contradiction, and vice versa. That single cell decides
whether Member B's composition operator is ever invoked on the instances it
exists for, so it is what determines which variant gets reported as "the"
detector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from contract.models import ConflictType, QueryRecord, ScopeRelation
from detection.stage1 import PairCandidate


class DetectorVariant(ABC):
    """A five-class conflict classifier."""

    #: Short identifier used in result tables and filenames.
    name: str = "unnamed"

    #: One line for the comparison table, so a reader of the results knows what
    #: they are comparing without going back to the paper.
    description: str = ""

    @abstractmethod
    def classify(self, candidate: PairCandidate) -> tuple[ConflictType, ScopeRelation | None]:
        """Label one pair.

        Returns the five-class label and, for conditional (or borderline
        factual) pairs, the four-way scope relation. Returning ``None`` for the
        relation on a conditional pair is a contract violation -- the routing
        logic has nothing to act on -- so a variant that cannot decide must
        return a non-conditional label instead.
        """

    def classify_many(
        self, candidates: list[PairCandidate]
    ) -> list[tuple[ConflictType, ScopeRelation | None]]:
        """Batch entry point. Override where batching is a real speedup."""
        return [self.classify(c) for c in candidates]

    # -- optional training ---------------------------------------------------- #

    def fit(self, records: list[QueryRecord]) -> DetectorVariant:
        """Train on labelled records. Variants that need no training no-op."""
        return self

    def save(self, path: str | Path) -> None:  # pragma: no cover - variant specific
        raise NotImplementedError(f"{self.name} does not support saving")

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def is_trained(self) -> bool:
        return not self.requires_training


def labels_from_record(record: QueryRecord) -> dict[tuple[str, str], tuple[ConflictType, ScopeRelation | None]]:
    """Index a record's gold pair labels by canonical pair key."""
    return {p.key: (p.type, p.scope_relation) for p in record.conflict_pairs}
