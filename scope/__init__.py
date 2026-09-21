"""A4: four-way scope-relation analysis.

Decides whether two claims stand in a refinement, disjoint, redundant or
opposed relation -- by deciding applicability and outcome agreement
SEPARATELY and combining them. Deciding only applicability cannot tell a
redundant restatement from a real exception, which is the correction the
four-way relation exists to make.

Branch roles derive from the applicability relation alone, never from
retrieval order. ``order_invariance.py`` verifies that property.


The order-invariance harness lives in ``scope.order_invariance`` and is
deliberately NOT re-exported here: importing it from the package would make
``python -m scope.order_invariance`` emit a double-import warning.
"""

from scope.attributes import SetRelation, compare, describe, intersects, narrower
from scope.descriptor import ClaimDescriptor, DescriptorExtractor, rule_based_descriptor
from scope.pipeline import ScopePipeline, ScopeStats
from scope.relation import Evidence, ScopeAnalyser, ScopeDecision

__all__ = [
    "ClaimDescriptor",
    "DescriptorExtractor",
    "Evidence",
    "ScopeAnalyser",
    "ScopeDecision",
    "ScopePipeline",
    "ScopeStats",
    "SetRelation",
    "compare",
    "describe",
    "intersects",
    "narrower",
    "rule_based_descriptor",
]
