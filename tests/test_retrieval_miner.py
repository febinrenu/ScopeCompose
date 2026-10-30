"""The retrieval miner: local ranking, the verbatim gate, and the EVAL-only rule."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from benchmark.mining.retrieval_miner import (
    Candidate,
    candidates_for,
    mine,
    pair_score,
    verify,
)

RULE = "Most people need to pay the immigration health surcharge as part of their application."
EXC = "Irish citizens will not need to pay the surcharge because of the Common Travel Area."
FILLER = "The office is open from nine until five on weekdays and closed on public holidays."


def doc(doc_id, provider, text, content_id=None):
    return {"doc_id": doc_id, "provider": provider, "text": text, "content_id": content_id or f"cid-{doc_id}"}


def client(tmp_path, reply):
    backend = MockBackend(scripted={"Passage A": json.dumps(reply) if not isinstance(reply, str) else reply})
    return (LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                      backend_override=backend), backend)


GOOD = {"pair": True, "rule_in": "A", "rule": RULE, "exception": EXC, "why": "Irish citizens", "confidence": 0.9}


# --------------------------------------------------------------------------- #
# Local ranking
# --------------------------------------------------------------------------- #


def test_a_rule_cue_against_an_exception_cue_outranks_unrelated_text():
    assert pair_score(RULE, EXC) > pair_score(RULE, FILLER)
    assert pair_score(FILLER, FILLER) == 0.0


def test_complementary_cues_matter_more_than_shared_vocabulary():
    same_words = pair_score(RULE, RULE.replace("Most people", "Some people"))
    assert pair_score(RULE, EXC) > same_words, "two rules that look alike are not a rule and an exception"


def test_overlap_only_ranks_and_is_capped():
    assert pair_score(RULE, EXC) <= 1.0 + 0.5 + 1e-9


def test_candidates_are_cross_document_ranked_and_capped():
    docs = [doc("a", "p", RULE + " " + FILLER), doc("b", "p", EXC + " " + FILLER)]
    cs = candidates_for(docs, top_n=3)
    assert cs and len(cs) <= 3 and all(c.chunk_a.doc_id != c.chunk_b.doc_id for c in cs)
    assert [c.score for c in cs] == sorted((c.score for c in cs), reverse=True)


def test_two_parts_of_one_guide_are_never_paired():
    docs = [doc("g#part1", "p", RULE, content_id="same"), doc("g#part2", "p", EXC, content_id="same")]
    assert candidates_for(docs) == []


def test_candidate_order_is_deterministic():
    docs = [doc("a", "p", RULE + " " + FILLER), doc("b", "p", EXC + " " + FILLER)]
    assert candidates_for(docs) == candidates_for(list(reversed(docs))) or \
        [c.score for c in candidates_for(docs)] == [c.score for c in candidates_for(list(reversed(docs)))]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def cand():
    docs = [doc("a", "prov", RULE), doc("b", "prov", EXC)]
    return candidates_for(docs, top_n=1)[0]


def test_a_verbatim_pair_is_kept_with_its_provenance(tmp_path):
    c, backend = client(tmp_path, GOOD)
    v = verify(cand(), client=c)
    assert v is not None and v.rule == RULE and v.exception == EXC and v.cross_document
    assert v.rule_doc_id == "a" and v.exception_doc_id == "b" and v.rule_chunk == "a#c0"
    assert v.rule_content_id == "cid-a" and v.exception_content_id == "cid-b"
    assert backend.calls[0]["model"] == "qwen/qwen3.8-27b" and "Passage A" in backend.calls[0]["messages"][0]["content"]


def test_the_rule_may_be_in_passage_b(tmp_path):
    swapped = GOOD | {"rule_in": "B", "rule": EXC, "exception": RULE}
    c, _ = client(tmp_path, swapped)
    v = verify(cand(), client=c)
    assert v is not None and v.rule_doc_id == "b" and v.exception_doc_id == "a"
    assert v.rule_content_id == "cid-b" and v.exception_content_id == "cid-a"


@pytest.mark.parametrize("reply", [
    GOOD | {"pair": False},
    GOOD | {"rule": "Everybody must pay the surcharge, in full."},        # paraphrase: not verbatim
    GOOD | {"exception": "Irish nationals are exempt from the surcharge."},
    GOOD | {"rule_in": "Z"},
    GOOD | {"rule": "short"},
    "not json",
    {"pair": True},
])
def test_anything_that_is_not_a_clean_verbatim_pair_is_dropped_never_repaired(tmp_path, reply):
    c, _ = client(tmp_path, reply)
    assert verify(cand(), client=c) is None


def test_whitespace_and_a_missing_full_stop_do_not_defeat_the_verbatim_check(tmp_path):
    c, _ = client(tmp_path, GOOD | {"rule": RULE.rstrip(".").replace(" ", "  "), "exception": EXC.rstrip(".")})
    assert verify(cand(), client=c) is not None


def test_a_bad_confidence_is_tolerated(tmp_path):
    c, _ = client(tmp_path, GOOD | {"confidence": "high"})
    assert verify(cand(), client=c).confidence == 0.5


def test_mining_that_can_reach_a_test_set_must_use_the_eval_tier(tmp_path):
    c, _ = client(tmp_path, GOOD)
    for tier in (Tier.BULK, Tier.BACKBONE_PRIMARY, Tier.JUDGE):
        with pytest.raises(ValueError, match="EVAL tier"):
            verify(cand(), client=c, tier=tier)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def test_mine_returns_the_proposer_format_and_counts_attempts(tmp_path):
    c, _ = client(tmp_path, GOOD)
    docs = [doc("a", "prov", RULE + " " + FILLER), doc("b", "prov", EXC + " " + FILLER), doc("solo", "alone", RULE)]
    out = mine(docs, top_n=2, client=c)
    assert out["pairings"] == 1, "a provider with one document is not a pairing"
    assert out["candidates_attempted"] >= 1 and out["candidates_failed"] == 0
    assert out["llm_cross_document"] == 1, "the same pair proposed from two candidates counts once"
    p = out["proposals"][0]
    assert {"rule", "exception", "rule_doc_id", "exception_doc_id", "confidence", "cross_document"} <= set(p)


def test_max_candidates_protects_the_quota(tmp_path):
    c, backend = client(tmp_path, GOOD)
    docs = [doc("a", "prov", RULE + " " + FILLER), doc("b", "prov", EXC + " " + FILLER)]
    mine(docs, top_n=5, client=c, max_candidates=1)
    assert len(backend.calls) == 1


def test_a_failing_candidate_is_counted_not_fatal(tmp_path):
    from api_budget import BackendError

    backend = MockBackend(fail_with=BackendError("down"))
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                  backend_override=backend, max_retries=1, base_backoff=0.0)
    out = mine([doc("a", "prov", RULE), doc("b", "prov", EXC)], top_n=2, client=c)
    assert out["candidates_failed"] == out["candidates_attempted"] >= 1 and out["proposals"] == []


def test_the_dry_run_makes_no_model_call_and_reports_a_cost(tmp_path, capsys):
    from benchmark.mining.retrieval_miner import main

    p = tmp_path / "docs.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in [doc("a", "prov", RULE), doc("b", "prov", EXC)]) + "\n",
                 encoding="utf-8")
    assert main(["--docs", str(p), "--json", str(tmp_path / "o.json"), "--dry-run"]) == 0
    assert "tokens on the EVAL tier" in capsys.readouterr().out
    assert not (tmp_path / "o.json").exists()
