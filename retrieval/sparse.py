"""BM25 sparse retrieval.

Half of the hybrid retriever. Deliberately ordinary: retrieval is held
identical to prior conflict-aware systems on purpose, so that a difference in
results isolates the analysis and resolution stages rather than reflecting a
better retriever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval.corpus import Corpus

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation.

    No stemming and no stopword list. Both would be defensible, but rule-heavy
    prose in this domain leans on function words that a stopword list removes
    ("only if", "does not apply to"), and those are exactly the cues the
    conditional class depends on.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ScoredPassage:
    passage_id: str
    score: float
    rank: int


class BM25Retriever:
    """BM25 over a corpus, via ``rank_bm25``."""

    def __init__(self, corpus: Corpus):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install rank-bm25") from exc

        self.corpus = corpus
        self._ids = corpus.ids
        tokenized = [tokenize(t) for t in corpus.texts]
        # rank_bm25 divides by the average document length, which is zero for
        # an empty corpus.
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int = 5) -> list[ScoredPassage]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        # Sort by score desc, then by id asc. The id tiebreak matters: without
        # it, equal-scoring passages come back in corpus order, which makes
        # the whole pipeline's output depend on how the corpus was loaded.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._ids[i]))
        return [
            ScoredPassage(passage_id=self._ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order[:k], start=1)
        ]
