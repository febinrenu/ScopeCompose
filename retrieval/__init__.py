"""A1: hybrid retrieval (BM25 + dense, reciprocal-rank fusion, K=5).

Held deliberately identical to prior conflict-aware systems so that
differences in results isolate the analysis and resolution stages rather than
reflecting a better retriever.
"""

from retrieval.corpus import Corpus, dump_corpus_stats
from retrieval.fusion import FusedResult, reciprocal_rank_fusion
from retrieval.pipeline import HybridRetriever, record_from_passages
from retrieval.sparse import BM25Retriever, ScoredPassage, tokenize

__all__ = [
    "BM25Retriever",
    "Corpus",
    "FusedResult",
    "HybridRetriever",
    "ScoredPassage",
    "dump_corpus_stats",
    "record_from_passages",
    "reciprocal_rank_fusion",
    "tokenize",
]
