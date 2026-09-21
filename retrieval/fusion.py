"""Reciprocal-rank fusion.

Combines the sparse and dense rankings into one. RRF is used rather than score
interpolation because BM25 scores and cosine similarities live on
incomparable scales, and any weighting between them would be a tuned
hyperparameter that varies by corpus -- an extra degree of freedom in a stage
this project deliberately holds fixed.

    RRF(d) = sum over rankers r of  1 / (k + rank_r(d))

with ``k`` damping the influence of top ranks (60 is the standard default,
from Cormack et al. 2009, and is set in ``config/hardware.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.sparse import ScoredPassage


@dataclass(frozen=True)
class FusedResult:
    passage_id: str
    score: float
    rank: int
    sparse_rank: int | None
    dense_rank: int | None

    @property
    def found_by_both(self) -> bool:
        return self.sparse_rank is not None and self.dense_rank is not None


def reciprocal_rank_fusion(
    rankings: list[list[ScoredPassage]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[FusedResult]:
    """Fuse several rankings into one.

    Ties break on passage id. That is not cosmetic: without a deterministic
    tiebreak, two passages with identical RRF scores come back in dict
    insertion order, which depends on which ranker happened to be listed
    first -- and the whole point of A4's order-invariance property is that no
    downstream decision may depend on input ordering.
    """
    scores: dict[str, float] = {}
    positions: dict[str, list[int | None]] = {}

    n = len(rankings)
    for r_idx, ranking in enumerate(rankings):
        for item in ranking:
            scores[item.passage_id] = scores.get(item.passage_id, 0.0) + 1.0 / (k + item.rank)
            if item.passage_id not in positions:
                positions[item.passage_id] = [None] * n
            positions[item.passage_id][r_idx] = item.rank

    order = sorted(scores, key=lambda pid: (-scores[pid], pid))

    out: list[FusedResult] = []
    for rank, pid in enumerate(order[:top_k], start=1):
        pos = positions[pid]
        out.append(
            FusedResult(
                passage_id=pid,
                score=scores[pid],
                rank=rank,
                sparse_rank=pos[0] if n > 0 else None,
                dense_rank=pos[1] if n > 1 else None,
            )
        )
    return out
