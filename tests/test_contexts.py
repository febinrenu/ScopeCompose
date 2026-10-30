"""K-passage contexts: length-matched, shuffled, screened-not-trimmed, gold always included."""

from __future__ import annotations

import json
import statistics

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog
from benchmark.contexts import (
    DEFAULT_K,
    build_context,
    candidate_chunks,
    drop_adjudicated,
    eval_screener,
    rank_by_bm25,
    target_words,
)
from contract.gold import GoldInstance
from contract.mock import generate
from contract.models import Passage


def sent(n, tag):
    w = " ".join(f"{tag}{i}" for i in range(n))
    return w[0].upper() + w[1:] + "."


def doc(doc_id, text):
    return {"doc_id": doc_id, "provider": "p", "kind": "guide", "url": f"https://www.gov.uk/{doc_id}", "text": text}


def inst() -> GoldInstance:
    base = next(i for i in generate(60, seed=0) if i.personas)
    return base.model_copy(update={"query": "Do I pay a fee on overseas transfers?"})


def pool(n=8):
    body = lambda k: " ".join(                                    # noqa: E731
        f"Overseas transfers fee note {k}{j} applies to account holders here." for j in range(12))
    return [doc(f"d{k}", body(k)) for k in range(n)]


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def test_the_context_has_k_passages_with_every_gold_passage_included():
    i = inst()
    r = build_context(i, pool(), k=DEFAULT_K)
    assert r.k == DEFAULT_K == 6 and len(r.negatives) == 4
    assert {p.id for p in i.passages} <= {p.id for p in r.instance.passages}, "100% gold inclusion"
    assert {p.text for p in i.passages} <= {p.text for p in r.instance.passages}
    assert all(n.startswith("n") for n in r.negatives)


def test_k_below_the_gold_count_is_refused():
    with pytest.raises(ValueError, match="smaller than"):
        build_context(inst(), pool(), k=1)


def test_distractors_are_length_matched_to_the_gold_so_length_does_not_give_the_gold_away():
    i = inst()
    gold_len = statistics.mean(len(p.text.split()) for p in i.passages)
    r = build_context(i, pool(), k=6)
    neg_len = [len(p.text.split()) for p in r.instance.passages if p.id in r.negatives]
    assert all(n <= max(2.2 * gold_len, 40) for n in neg_len), (gold_len, neg_len)
    assert target_words(i.passages) >= 25


def test_the_order_is_shuffled_per_instance_and_reproducible():
    i = inst()
    a = [p.id for p in build_context(i, pool(), seed=0).instance.passages]
    assert a == [p.id for p in build_context(i, pool(), seed=0).instance.passages]
    orders = {tuple(p.id for p in build_context(i, pool(), seed=s).instance.passages) for s in range(12)}
    assert len(orders) > 1, "the seed changes the order"
    firsts = [build_context(i, pool(), seed=s).instance.passages[0].id for s in range(30)]
    assert any(f.startswith("n") for f in firsts) and any(not f.startswith("n") for f in firsts), \
        "gold is not always first, and not always buried"


def test_gold_documents_and_gold_duplicates_never_appear_as_negatives():
    i = inst()
    gold_doc = i.passages[0].document_id
    docs = pool() + [doc(gold_doc, i.passages[0].text + " " + i.passages[1].text)]
    chunks = candidate_chunks(docs, {gold_doc}, [p.text for p in i.passages], words=30)
    assert all(c.doc_id != gold_doc for c in chunks)
    dup = doc("dup", i.passages[0].text)
    assert candidate_chunks([dup], set(), [p.text for p in i.passages], words=30) == []


def test_negatives_come_from_the_pool_best_first_against_the_query_with_a_per_document_cap():
    docs = [doc("good", "Overseas transfers fee applies to every overseas transfer fee payment. " * 6),
            doc("meh", "Opening hours are nine to five on weekdays at every branch we run. " * 6)]
    ranked = rank_by_bm25("overseas transfers fee", candidate_chunks(docs, set(), [], words=20))
    assert ranked[0].doc_id == "good"


def test_no_single_document_supplies_more_than_two_distractors_when_the_pool_allows_it():
    from collections import Counter

    r = build_context(inst(), pool(8), k=8)                       # 6 negatives from 8 documents
    counts = Counter(p.document_id for p in r.instance.passages if p.id in r.negatives)
    assert len(r.negatives) == 6 and max(counts.values()) <= 2 and len(counts) >= 3


def test_a_thin_pool_relaxes_the_cap_rather_than_returning_a_short_context():
    r = build_context(inst(), [doc("only", " ".join(sent(30, f"w{j}x") for j in range(8)))], k=6)
    assert r.k == 6


def test_an_empty_pool_gives_a_short_context_not_an_error_and_gold_is_still_there():
    i = inst()
    r = build_context(i, [], k=6)
    assert r.k == len(i.passages) and r.negatives == []


def test_negatives_carry_provenance_for_audit():
    r = build_context(inst(), pool(), k=6)
    n = next(p for p in r.instance.passages if p.id in r.negatives)
    assert n.document_id.startswith("d") and n.source_url.startswith("https://www.gov.uk/")


# --------------------------------------------------------------------------- #
# Screening: flagged, kept, and only dropped by an explicit act
# --------------------------------------------------------------------------- #


def test_flagged_negatives_are_kept_in_the_context_and_listed():
    r = build_context(inst(), pool(), k=6, screener=lambda q, t: "note 00" in t or "note 10" in t)
    assert r.flagged and set(r.flagged) <= set(r.negatives)
    assert all(f in {p.id for p in r.instance.passages} for f in r.flagged), "never silently removed"


def test_no_screener_flags_nothing():
    assert build_context(inst(), pool(), k=6).flagged == []


def test_dropping_requires_the_negative_to_be_flagged_and_named():
    r = build_context(inst(), pool(), k=6, screener=lambda q, t: True)
    one = r.flagged[0]
    out = drop_adjudicated(r, [one])
    assert one not in {p.id for p in out.instance.passages} and one not in out.negatives and out.k == r.k - 1
    with pytest.raises(ValueError, match="not flagged"):
        drop_adjudicated(build_context(inst(), pool(), k=6, screener=lambda q, t: False), ["n0"])


def test_gold_passages_can_never_be_dropped():
    i = inst()
    r = build_context(i, pool(), k=6, screener=lambda q, t: True)
    with pytest.raises(ValueError, match="not flagged"):
        drop_adjudicated(r, [i.passages[0].id])


def test_the_eval_screener_uses_the_eval_tier_and_flags_unreadable_verdicts(tmp_path):
    backend = MockBackend(scripted={"Passage:": json.dumps({"bears_on_answer": False})})
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"), backend_override=backend)
    assert eval_screener(c)("q", "some text") is False
    assert backend.calls[0]["model"] == "qwen/qwen3.8-27b"
    bad = MockBackend(scripted={"Passage:": "not json"})
    c2 = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l2.jsonl"), backend_override=bad)
    assert eval_screener(c2)("q", "text") is True, "an unreadable verdict goes to a person"
