"""Shared scoring library.

Member B leads this package and owns the preservation metrics (PR/SR/HCR/SCR)
and the metric-judge validation. Member A contributes the detection,
classification and scope-relation scorers.

Both members import from here rather than reimplementing a metric on their own
side -- two implementations of Suppression Rate that disagree by a rounding
convention is a paper-level problem, not a code-level one.
"""

from metrics.classification import (
    BinaryScores,
    ClassificationScores,
    ConfusionMatrix,
    ScopeScores,
    score_classification,
    score_detection,
    score_scope_relations,
    split_by_construction,
)
from metrics.preservation import (
    BranchJudgement,
    PreservationScores,
    judge_branches,
    score_preservation,
    validate_judge,
)

__all__ = [
    "BinaryScores",
    "BranchJudgement",
    "ClassificationScores",
    "ConfusionMatrix",
    "PreservationScores",
    "judge_branches",
    "score_preservation",
    "validate_judge",
    "ScopeScores",
    "score_classification",
    "score_detection",
    "score_scope_relations",
    "split_by_construction",
]
