"""Baseline systems.

Member A owns the retrieval/detection-side baselines (standard RAG, rerank,
NLI-filter). Member B owns the resolution-side ones (credibility selection,
chain-of-thought, full-context, taxonomy-aware prompting) and, jointly with A,
the structured long-context baseline -- the paper's decisive experiment.

See README.md section 5 before interpreting the structured long-context result:
on the zero-spend configuration it can falsify the central claim but cannot
confirm it.
"""

from baselines.retrieval_side import (
    ALL_BASELINES,
    BaselineOutput,
    NLIFilter,
    RerankTop1,
    StandardRAG,
)

__all__ = [
    "ALL_BASELINES",
    "BaselineOutput",
    "NLIFilter",
    "RerankTop1",
    "StandardRAG",
]
