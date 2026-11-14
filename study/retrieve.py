"""Open retrieval over the chunked ConditionalQA collection.

The retriever is the repository's existing one (``retrieval.pipeline``): BM25
and Contriever, fused by reciprocal rank. It is deliberately ordinary -- the
study is about what an ordinary pipeline loses.

Rankings are computed once per (chunking, ranker) and saved, so every system
and every analysis reads the *same* ranked lists.

    python -m study.retrieve                       # default chunking, all three rankers
    python -m study.retrieve --words 100 --rankers hybrid
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import settings
from contract.models import Passage
from retrieval.corpus import Corpus
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.pipeline import HybridRetriever
from study import cqa

ARTIFACTS = settings.REPO_ROOT / "study" / "artifacts"
RANKERS = ("bm25", "dense", "hybrid")
DEPTH = 50                     # how far each saved ranking goes
_POOL = 200                    # per-ranker candidates fused for the hybrid ranking


def tag(words: int, strategy: str) -> str:
    return f"{strategy}{words}"


@lru_cache(maxsize=None)
def retriever(words: int = cqa.DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> HybridRetriever:
    corpus = Corpus([Passage(id=c.id, text=c.text) for c in cqa.chunks(words, strategy)])
    return HybridRetriever(corpus, embedding_cache=ARTIFACTS / "embeddings" / f"cqa_{tag(words, strategy)}.npz")


def rank(query: str, *, ranker: str, words: int = cqa.DEFAULT_CHUNK_WORDS, strategy: str = "fixed",
         depth: int = DEPTH) -> list[str]:
    r = retriever(words, strategy)
    if ranker == "bm25":
        return [s.passage_id for s in r.sparse.search(query, k=depth)]
    if ranker == "dense":
        return [s.passage_id for s in r.dense.search(query, k=depth)]
    if ranker == "hybrid":
        lists = [r.sparse.search(query, k=_POOL), r.dense.search(query, k=_POOL)]
        return [f.passage_id for f in reciprocal_rank_fusion(lists, k=r.rrf_k, top_k=depth)]
    raise ValueError(f"unknown ranker {ranker!r}; one of {RANKERS}")


def rankings_path(ranker: str, words: int, strategy: str) -> Path:
    return ARTIFACTS / "rankings" / f"{ranker}_{tag(words, strategy)}.json"


def build_rankings(questions: Sequence[cqa.Question], *, ranker: str, words: int, strategy: str,
                   refresh: bool = False) -> dict[str, list[str]]:
    """``question id -> ranked chunk ids``, extended (never recomputed) for new questions."""
    path = rankings_path(ranker, words, strategy)
    have: dict[str, list[str]] = {}
    if path.exists() and not refresh:
        have = json.loads(path.read_text(encoding="utf-8"))
    todo = [q for q in questions if q.id not in have]
    for q in todo:
        have[q.id] = rank(q.retrieval_query, ranker=ranker, words=words, strategy=strategy)
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(have), encoding="utf-8")
    return have


@lru_cache(maxsize=None)
def load_rankings(ranker: str = "hybrid", words: int = cqa.DEFAULT_CHUNK_WORDS,
                  strategy: str = "fixed") -> dict[str, list[str]]:
    path = rankings_path(ranker, words, strategy)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m study.retrieve`")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def chunk_index(words: int = cqa.DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> dict[str, cqa.Chunk]:
    return {c.id: c for c in cqa.chunks(words, strategy)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", type=int, default=cqa.DEFAULT_CHUNK_WORDS)
    ap.add_argument("--strategy", default="fixed", choices=cqa.STRATEGIES)
    ap.add_argument("--rankers", nargs="+", default=list(RANKERS), choices=RANKERS)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    qs = cqa.eval_questions() + cqa.tuning_questions()
    for ranker in args.rankers:
        got = build_rankings(qs, ranker=ranker, words=args.words, strategy=args.strategy, refresh=args.refresh)
        print(f"{ranker}_{tag(args.words, args.strategy)}: {len(got)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
