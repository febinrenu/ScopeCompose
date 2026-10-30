"""The ConfRAG-style coverage comparator -- and the demonstration it exists for (H-K)."""

from __future__ import annotations

import pytest

from contract.gold import GoldInstance
from contract.mock import generate
from metrics.coverage_confrag import answer_coverage, branch_keywords, mean_coverage
from metrics.validity import OUTPUTS, battery_outputs


def _instances() -> list[GoldInstance]:
    out = [i for i in generate(120, seed=0) if i.personas and i.gold_scoped_answer]
    assert len(out) >= 5
    return out


def test_keywords_are_distinct_between_branches():
    """A keyword may not cover two clusters (one-to-one)."""
    for inst in _instances():
        kws = branch_keywords(inst)
        sets = list(kws.values())
        assert not (sets[0] & sets[1]), kws


def test_every_branch_has_at_least_one_keyword():
    for inst in _instances():
        assert all(kws for kws in branch_keywords(inst).values())


def test_a_branch_is_covered_by_a_substring_of_the_answer():
    inst = _instances()[0]
    assert answer_coverage(inst, inst.gold_scoped_answer).score == 1.0
    default_only = inst.gold_scoped_answer.split(" However, ")[0]
    cov = answer_coverage(inst, default_only)
    assert cov.score == 0.5 and cov.covered == (inst.default_branch.branch_id,)


def test_matching_is_case_insensitive_and_ignores_citation_digits():
    inst = _instances()[0]
    assert answer_coverage(inst, "THE FEE IS WAIVED (per p1)").covered != ()
    # "p0" in a citation must not satisfy a numeric keyword
    assert answer_coverage(inst, "Nothing relevant here (per p0) [p1]").score == 0.0


def test_a_null_answer_covers_nothing():
    inst = _instances()[0]
    assert answer_coverage(inst, "I'm sorry, I don't have enough information.").score == 0.0


def test_h_k_coverage_cannot_tell_an_unattached_listing_from_the_gold_answer():
    """The point of the comparator. Every output that merely MENTIONS both
    outcomes is fully covered, including the two that tell no one which applies."""
    insts = _instances()
    cov = {o: mean_coverage(insts, {i.instance_id: battery_outputs(i)[o] for i in insts})
           for o in OUTPUTS}
    assert cov["gold"] == 1.0
    assert cov["unattached_hedge"] >= cov["gold"]
    assert cov["modality_hedge"] >= cov["gold"]
    assert cov["null"] == 0.0
    assert cov["selection"] < cov["gold"], "a pure selection answer is the one thing it does catch"


def test_mean_coverage_is_zero_for_no_instances():
    assert mean_coverage([], {}) == 0.0


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_answers_cover_nothing(answer):
    assert answer_coverage(_instances()[0], answer).score == 0.0
