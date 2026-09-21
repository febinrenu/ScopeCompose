"""A3: three detector architectures, compared head-to-head.

Proposal section 5.2 commits to settling this empirically. The variant with
the lowest conditional-leak rate on the factual/conditional confusion cell --
not the best aggregate accuracy -- is the one reported as "the" detector.
"""

from detection.variants.base import DetectorVariant, labels_from_record
from detection.variants.lexical_nli import LexicalNLIDetector
from detection.variants.structured_entailment import (
    EntailmentEvidence,
    StructuredEntailmentDetector,
)

__all__ = [
    "DetectorVariant",
    "EntailmentEvidence",
    "LexicalNLIDetector",
    "StructuredEntailmentDetector",
    "labels_from_record",
]
