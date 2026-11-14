"""Regression tests for the defects found in the November 2026 audit.

Each test fails on the code as it stood at commit 424b7c5.
"""

from __future__ import annotations

import random

import pytest

import settings
from api_budget.backends import MockBackend, RateLimitError
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from baselines.registry import oracle_conflicts
from contract.gold import GoldInstance
from contract.models import ConflictPair, ConflictType, Passage
from metrics.classification import score_detection
from metrics.values import canonicalise_outcome, figures, values_match

MSG = [{"role": "user", "content": "x"}]


@pytest.fixture(autouse=True)
def _fresh_config():
    settings.reload()
    yield
    settings.reload()


# --------------------------------------------------------------------------- #
# Currency amounts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gold, predicted", [
    ("a 2.50 fee applies", "A £2.50 fee applies."),
    ("funds of 1270 held for 28 consecutive days", "You need £1,270 held for 28 days"),
    ("a fee of 35 dollars applies", "a fee of $35 dollars applies"),
    ("the charge is 12 euros", "the charge is € 12 euros"),
])
def test_a_currency_sign_does_not_split_the_amount(gold, predicted):
    assert values_match(gold, predicted)


def test_a_currency_amount_is_one_figure():
    assert figures("You need £1,270 held for 28 days") == {"amount": {"1270"}, "day": {"28"}}
    assert figures("A £2.50 fee applies.") == {"amount": {"2.5"}}


def test_a_different_currency_amount_still_conflicts():
    assert not values_match("a 2.50 fee applies", "A £3.50 fee applies.")


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", [
    "you are not required to pay",
    "You are not eligible for the scheme",
    "applicants must provide a passport",
    "the fee is waived",
    "you may not work",
    "Students are allowed to work 20 hours",
])
def test_canonicalising_twice_changes_nothing(text):
    once = canonicalise_outcome(text)
    assert canonicalise_outcome(once) == once


def test_a_class_never_rewrites_inside_another_classes_output():
    out = canonicalise_outcome("you are not required to pay")
    assert "is not is" not in out and "are is" not in out
    assert "is not required" in out


# --------------------------------------------------------------------------- #
# The labelled pair in a shuffled context
# --------------------------------------------------------------------------- #


def _instance() -> GoldInstance:
    import json

    line = next(l for l in (settings.REPO_ROOT / "benchmark" / "data" / "corpus.jsonl")
                .read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["gold_conflict_type"] == "conditional")
    return GoldInstance.model_validate(json.loads(line))


def _expanded(inst: GoldInstance, seed: int) -> GoldInstance:
    negs = [Passage(id=f"n{i}", text=f"An unrelated sentence number {i}.") for i in range(4)]
    allp = list(inst.passages) + negs
    random.Random(seed).shuffle(allp)
    return inst.model_copy(update={"passages": allp})


def test_the_oracle_group_is_the_gold_pair_wherever_it_sits():
    inst = _instance()
    want = tuple(sorted(p.id for p in inst.passages))
    for seed in range(20):
        groups = oracle_conflicts(_expanded(inst, seed))
        assert len(groups) == 1 and groups[0].passage_ids == want


def test_the_projected_gold_pair_never_names_a_distractor():
    inst = _instance()
    want = {p.id for p in inst.passages}
    for seed in range(20):
        rec = _expanded(inst, seed).to_query_record()
        assert len(rec.conflict_pairs) == 1
        assert {rec.conflict_pairs[0].doc_i, rec.conflict_pairs[0].doc_j} == want


# --------------------------------------------------------------------------- #
# Detection recall
# --------------------------------------------------------------------------- #


def test_a_gold_conflict_the_detector_never_emits_is_a_miss():
    gold = _instance().to_query_record()
    assert gold.conflict_pairs[0].is_conflict
    silent = gold.model_copy(update={"conflict_pairs": []})
    s = score_detection([silent], [gold])
    assert (s.tp, s.fn) == (0, 1)
    assert s.recall == 0.0


def test_an_emitted_pair_is_still_counted_once():
    gold = _instance().to_query_record()
    s = score_detection([gold], [gold])
    assert (s.tp, s.fp, s.fn, s.tn) == (1, 0, 0, 0)


# --------------------------------------------------------------------------- #
# Fallback
# --------------------------------------------------------------------------- #


def test_the_configured_chain_is_read_in_order(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "local-fallback")
    settings.reload()
    chain = settings.models().fallback.chain
    assert chain[0][0] == "groq" and chain[-1] == ("ollama", "local-fallback")


def test_a_fallback_answer_is_not_cached_as_the_primary_models(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "local-fallback")
    settings.reload()
    cache = RequestCache(tmp_path / "c.sqlite")
    client = LLMClient(cache=cache, cost_log=CostLog(tmp_path / "l.jsonl"), max_retries=3, base_backoff=0.0)
    groq_down = MockBackend(fail_with=RateLimitError("429"))
    backends = {"groq": groq_down, "ollama": MockBackend()}
    client._backend = lambda name: backends[name]  # type: ignore[method-assign]

    first = client.complete(messages=MSG, tier=Tier.JUDGE, step="t")
    assert first.fell_back is True

    # The provider recovers. The primary model must be asked, not answered from
    # the fallback's cached text with fell_back=False.
    groq_up = MockBackend()
    backends["groq"] = groq_up
    second = client.complete(messages=MSG, tier=Tier.JUDGE, step="t")
    assert second.fell_back is False and second.cached is False
    assert len(groq_up.calls) == 1
