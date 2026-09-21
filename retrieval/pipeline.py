"""A1: hybrid retrieval.

BM25 + dense, fused by reciprocal rank, K = 5.

This stage is intentionally conventional. Holding retrieval identical to prior
conflict-aware systems is what lets a difference in the paper's results be
attributed to the analysis and resolution stages rather than to a better
retriever -- so "we did nothing clever here" is the design, not a shortcut.

The dense half can be switched off (``use_dense=False``) to run the whole
pipeline sparse-only, which is how the contract, detection and scope modules
get exercised on a machine with no GPU and no model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import settings
from contract.models import Construction, Domain, Passage, QueryRecord
from retrieval.corpus import Corpus
from retrieval.fusion import FusedResult, reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever


@dataclass
class HybridRetriever:
    """Sparse + dense retrieval fused by RRF."""

    corpus: Corpus
    profile: str | None = None
    use_dense: bool = True
    embedding_cache: str | Path | None = None

    _sparse: BM25Retriever | None = field(default=None, repr=False)
    _dense: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.hw = settings.hardware(self.profile)
        self.top_k = self.hw.top_k
        self.rrf_k = self.hw.rrf_k

    @property
    def sparse(self) -> BM25Retriever:
        if self._sparse is None:
            self._sparse = BM25Retriever(self.corpus)
        return self._sparse

    @property
    def dense(self):
        if self._dense is None:
            from retrieval.dense import DenseRetriever

            d = DenseRetriever(self.corpus, profile=self.profile)
            d.index(cache_path=self.embedding_cache)
            self._dense = d
        return self._dense

    # -- retrieval ----------------------------------------------------------- #

    def search(self, query: str, k: int | None = None) -> list[FusedResult]:
        k = k or self.top_k
        # Over-retrieve from each ranker before fusing. Fusing two top-5 lists
        # would throw away a passage ranked 6th by one ranker and 1st by the
        # other, which is exactly the case fusion exists to rescue.
        pool = max(k * 4, 20)

        rankings = [self.sparse.search(query, k=pool)]
        if self.use_dense:
            rankings.append(self.dense.search(query, k=pool))

        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_k=k)

    def retrieve_passages(self, query: str, k: int | None = None) -> list[Passage]:
        """Search and return the passages themselves, in fused rank order."""
        return [self.corpus.get(r.passage_id) for r in self.search(query, k=k)]

    def to_record(
        self,
        query: str,
        *,
        query_id: str,
        domain: Domain = Domain.FINANCIAL_TERMS,
        construction: Construction = Construction.NATURAL,
        k: int | None = None,
    ) -> QueryRecord:
        """Retrieve and emit a contract record with no conflict pairs yet.

        This is the A1 output that A2 consumes: the ``passages`` array is
        populated, ``conflict_pairs`` is left for detection to fill in.
        """
        passages = self.retrieve_passages(query, k=k)

        # Renumber to p0..pN. Downstream code and the contract's canonical
        # pair ordering both assume record-local ids; corpus ids are
        # namespaced by source record and would leak that namespacing into
        # every conflict pair.
        renumbered = [p.model_copy(update={"id": f"p{i}"}) for i, p in enumerate(passages)]

        return QueryRecord(
            query_id=query_id,
            query=query,
            domain=domain,
            construction=construction,
            passages=renumbered,
            conflict_pairs=[],
        )


def record_from_passages(
    query: str,
    passages: list[Passage],
    *,
    query_id: str,
    domain: Domain = Domain.FINANCIAL_TERMS,
    construction: Construction = Construction.NATURAL,
) -> QueryRecord:
    """Build a record from an already-chosen passage set.

    The oracle-retrieval path. Every experiment that isolates detection or
    scope analysis from retrieval quality goes through here, including the
    order-permutation ablation, which needs to control the passage set exactly.
    """
    return QueryRecord(
        query_id=query_id,
        query=query,
        domain=domain,
        construction=construction,
        passages=[p.model_copy(update={"id": f"p{i}"}) for i, p in enumerate(passages)],
        conflict_pairs=[],
    )
