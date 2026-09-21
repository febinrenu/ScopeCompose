"""Tests for A1: corpus handling, BM25, and reciprocal-rank fusion.

Dense retrieval is exercised only behind a ``slow`` marker, because it needs a
model download. The fusion and corpus logic -- where the order-determinism
guarantees live -- is tested exhaustively and offline.
"""

from __future__ import annotations

import pytest

from contract.models import Passage, SourceType
from retrieval.corpus import Corpus, dump_corpus_stats
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse import ScoredPassage, tokenize

FEE_RULE = "International transactions incur a 3% fee."
FEE_EXCEPTION = "The fee is waived for premium-tier cardholders."
UNRELATED = "Statements are issued on the first business day of each month."
VISA = "An F-1 holder may not accept employment in the United States."


def _sp(pid: str, rank: int, score: float = 1.0) -> ScoredPassage:
    return ScoredPassage(passage_id=pid, score=score, rank=rank)


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Fee, 3% -- WAIVED!") == ["fee", "3", "waived"]


def test_tokenize_keeps_function_words():
    """No stopword list, on purpose: rule-governed prose leans on exactly the
    function words a stopword list removes ('only if', 'does not apply to'),
    and those are the cues the conditional class depends on."""
    toks = tokenize("The fee does not apply to students")
    assert "not" in toks and "to" in toks


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def test_corpus_from_texts():
    c = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION])
    assert len(c) == 2
    assert c.get("p0").text == FEE_RULE


def test_corpus_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate passage ids"):
        Corpus([
            Passage(id="p0", text="a", source_type=SourceType.FAQ),
            Passage(id="p0", text="b", source_type=SourceType.FAQ),
        ])


def test_corpus_add_rejects_a_collision():
    c = Corpus.from_texts([FEE_RULE])
    with pytest.raises(ValueError, match="already in corpus"):
        c.add(Passage(id="p0", text="other", source_type=SourceType.FAQ))


def test_corpus_get_missing_is_a_keyerror():
    with pytest.raises(KeyError, match="no passage"):
        Corpus.from_texts([FEE_RULE]).get("nope")


def test_corpus_round_trips_through_jsonl(tmp_path):
    original = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION, VISA])
    path = tmp_path / "c.jsonl"
    original.to_jsonl(path)
    assert Corpus.from_jsonl(path).texts == original.texts


def test_corpus_from_records_namespaces_ids():
    """'p0' means different things in different records, so a flat corpus
    would collide them."""
    from contract.mock import generate

    corpus = Corpus.from_records([i.to_query_record() for i in generate(5, seed=0)])
    assert all("::" in pid for pid in corpus.ids)
    assert len(set(corpus.ids)) == len(corpus.ids)


def test_corpus_stats():
    stats = dump_corpus_stats(Corpus.from_texts([FEE_RULE, FEE_EXCEPTION]))
    assert stats["passages"] == 2
    assert stats["mean_words"] > 0


# --------------------------------------------------------------------------- #
# Reciprocal-rank fusion
# --------------------------------------------------------------------------- #


def test_fusion_rewards_agreement():
    """A passage both rankers like should beat one only a single ranker
    placed first -- that is the entire purpose of fusing."""
    sparse = [_sp("a", 1), _sp("b", 2), _sp("c", 3)]
    dense = [_sp("b", 1), _sp("a", 2), _sp("d", 3)]
    fused = reciprocal_rank_fusion([sparse, dense], k=60, top_k=4)
    assert {fused[0].passage_id, fused[1].passage_id} == {"a", "b"}


def test_fusion_records_both_ranks():
    fused = reciprocal_rank_fusion(
        [[_sp("a", 1)], [_sp("a", 3)]], k=60, top_k=1
    )
    assert fused[0].sparse_rank == 1
    assert fused[0].dense_rank == 3
    assert fused[0].found_by_both


def test_fusion_handles_a_passage_only_one_ranker_found():
    fused = reciprocal_rank_fusion([[_sp("a", 1)], [_sp("b", 1)]], k=60, top_k=2)
    by_id = {f.passage_id: f for f in fused}
    assert by_id["a"].dense_rank is None
    assert by_id["b"].sparse_rank is None
    assert not by_id["a"].found_by_both


def test_fusion_respects_top_k():
    ranking = [_sp(f"p{i}", i + 1) for i in range(20)]
    assert len(reciprocal_rank_fusion([ranking], k=60, top_k=5)) == 5


def test_fusion_ties_break_deterministically():
    """Without a deterministic tiebreak, equal-scoring passages come back in
    dict insertion order -- which depends on which ranker was listed first.
    Every downstream decision would then inherit that dependence."""
    a = reciprocal_rank_fusion([[_sp("z", 1), _sp("y", 1), _sp("x", 1)]], k=60, top_k=3)
    b = reciprocal_rank_fusion([[_sp("x", 1), _sp("y", 1), _sp("z", 1)]], k=60, top_k=3)
    assert [f.passage_id for f in a] == [f.passage_id for f in b] == ["x", "y", "z"]


def test_fusion_is_invariant_to_ranker_order():
    sparse = [_sp("a", 1), _sp("b", 2)]
    dense = [_sp("b", 1), _sp("c", 2)]
    forward = {f.passage_id: round(f.score, 10)
               for f in reciprocal_rank_fusion([sparse, dense], k=60, top_k=5)}
    backward = {f.passage_id: round(f.score, 10)
                for f in reciprocal_rank_fusion([dense, sparse], k=60, top_k=5)}
    assert forward == backward


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([], k=60, top_k=5) == []
    assert reciprocal_rank_fusion([[]], k=60, top_k=5) == []


def test_rrf_scores_follow_the_formula():
    fused = reciprocal_rank_fusion([[_sp("a", 1)], [_sp("a", 2)]], k=60, top_k=1)
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-12


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


def test_bm25_finds_the_relevant_passage():
    pytest.importorskip("rank_bm25")
    from retrieval.sparse import BM25Retriever

    corpus = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION, UNRELATED, VISA])
    results = BM25Retriever(corpus).search("international transaction fee", k=2)
    assert results
    assert results[0].rank == 1
    assert results[0].passage_id in {"p0", "p1"}


def test_bm25_ranks_are_contiguous_from_one():
    pytest.importorskip("rank_bm25")
    from retrieval.sparse import BM25Retriever

    corpus = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION, UNRELATED, VISA])
    results = BM25Retriever(corpus).search("fee", k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_bm25_on_an_empty_corpus_returns_nothing():
    pytest.importorskip("rank_bm25")
    from retrieval.sparse import BM25Retriever

    assert BM25Retriever(Corpus([])).search("anything", k=5) == []


def test_bm25_is_deterministic_across_corpus_orderings():
    """Equal-scoring passages must not come back in load order, or the whole
    pipeline's output depends on how the corpus file was written."""
    pytest.importorskip("rank_bm25")
    from retrieval.sparse import BM25Retriever

    texts = ["alpha beta", "alpha beta", "gamma delta"]
    a = BM25Retriever(Corpus.from_texts(texts)).search("alpha", k=3)
    b = BM25Retriever(Corpus.from_texts(texts)).search("alpha", k=3)
    assert [r.passage_id for r in a] == [r.passage_id for r in b]


# --------------------------------------------------------------------------- #
# Hybrid pipeline
# --------------------------------------------------------------------------- #


def test_hybrid_retriever_sparse_only():
    pytest.importorskip("rank_bm25")
    from retrieval.pipeline import HybridRetriever

    corpus = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION, UNRELATED, VISA])
    results = HybridRetriever(corpus, use_dense=False).search("fee", k=3)
    assert len(results) == 3
    assert all(r.dense_rank is None for r in results)


def test_to_record_renumbers_passages():
    """Corpus ids are namespaced by source record; leaking that namespacing
    into conflict pairs would make every pair key unwieldy and record-specific."""
    pytest.importorskip("rank_bm25")
    from retrieval.pipeline import HybridRetriever

    corpus = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION, UNRELATED], prefix="corpus_")
    rec = HybridRetriever(corpus, use_dense=False).to_record(
        "fee", query_id="q0", k=2
    )
    assert [p.id for p in rec.passages] == ["p0", "p1"]
    assert rec.conflict_pairs == []


def test_record_from_passages_is_the_oracle_path():
    """Every experiment that isolates detection or scope from retrieval
    quality needs to control the passage set exactly."""
    from retrieval.pipeline import record_from_passages

    passages = [
        Passage(id="x", text=FEE_RULE, source_type=SourceType.OFFICIAL_POLICY),
        Passage(id="y", text=FEE_EXCEPTION, source_type=SourceType.PRODUCT_TERMS),
    ]
    rec = record_from_passages("fee?", passages, query_id="q1")
    assert [p.id for p in rec.passages] == ["p0", "p1"]
    assert rec.passages[0].text == FEE_RULE


@pytest.mark.slow
def test_dense_retrieval_end_to_end():
    """Needs a model download. Deselect with -m 'not slow'."""
    pytest.importorskip("sentence_transformers")
    from retrieval.pipeline import HybridRetriever

    corpus = Corpus.from_texts([FEE_RULE, FEE_EXCEPTION, UNRELATED, VISA])
    results = HybridRetriever(corpus, use_dense=True).search("do I pay a card fee?", k=3)
    assert len(results) == 3
    assert any(r.found_by_both for r in results)
