"""Scoped precision: aligned / misattached / unsupported / extraneous, and what each answer shape scores."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog
from contract.gold import Applicability, Branch, GoldInstance
from contract.mock import generate
from contract.models import ConflictType, Construction, Domain, Passage, ScopeRelation
from metrics.scoped import (
    ATTACH_OVERLAP,
    V3_ACCURACY_BAR,
    Label,
    aggregate,
    conditioned,
    eval_judge,
    fabricated_exception,
    label_claim,
    mean_precision,
    score,
    split_claims,
    validate_judge,
)
from metrics.validity import battery_outputs

DEFAULT = Branch(branch_id="b0", condition=None, outcome="a 3% fee applies",
                 applicability=Applicability(descriptor="everyone", is_default=True), supporting_passage="p0")
EXC = Branch(branch_id="b1", condition="the holder has a premium-tier card", outcome="no fee applies",
             applicability=Applicability(descriptor="premium-tier cardholders"), supporting_passage="p1")
BRANCHES = [DEFAULT, EXC]


def inst(branches=BRANCHES) -> GoldInstance:
    passages = [Passage(id="p0", text="Transfers incur a 3% fee."),
                Passage(id="p1", text="The fee is waived for premium-tier cardholders.")]
    if len(branches) == 1:                        # a control: one rule, no exception
        return GoldInstance(instance_id="sc", query="Do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
                            construction=Construction.NATURAL, passages=passages[:1],
                            gold_conflict_type=ConflictType.NO_CONFLICT, gold_branches=branches)
    return GoldInstance(instance_id="sc", query="Do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
                        construction=Construction.NATURAL, passages=passages,
                        gold_conflict_type=ConflictType.CONDITIONAL, gold_scope_relation=ScopeRelation.REFINEMENT,
                        gold_branches=branches)


def lab(claim) -> Label:
    return label_claim(claim, BRANCHES).label


# --------------------------------------------------------------------------- #
# Splitting and conditioning
# --------------------------------------------------------------------------- #


def test_claims_are_sentences_with_connectives_and_citations_removed():
    assert split_claims("A 3% fee applies (per p0). However, if premium, no fee applies (per p1).") == [
        "A 3% fee applies.", "if premium, no fee applies."]
    assert split_claims("") == [] and split_claims("   ") == []


@pytest.mark.parametrize("claim,expected", [
    ("no fee applies for premium-tier cardholders", True), ("if you hold a premium card no fee applies", True),
    ("unless you are a student, a fee applies", True), ("a 3% fee applies", False),
    ("the fee is waived for signature-tier cardholders", True), ("for all cardholders a 3% fee applies", True),
    ("the fee is 3%", False), ("students pay nothing", False)])
def test_conditioned_detects_a_restricting_phrase(claim, expected):
    assert conditioned(claim) is expected


# --------------------------------------------------------------------------- #
# The four labels
# --------------------------------------------------------------------------- #


def test_a_default_outcome_stated_plainly_is_aligned():
    c = label_claim("A 3% fee applies.", BRANCHES)
    assert c.label is Label.ALIGNED and c.branch_id == "b0" and c.by == "deterministic"


def test_an_exception_outcome_attached_to_its_group_is_aligned():
    c = label_claim("if you hold a premium-tier card, no fee applies.", BRANCHES)
    assert c.label is Label.ALIGNED and c.branch_id == "b1"


def test_the_exceptions_outcome_given_to_everyone_is_misattached():
    """Right words, wrong people: the error a coverage metric cannot see."""
    assert lab("No fee applies.") is Label.MISATTACHED
    assert lab("The fee is waived.") is Label.MISATTACHED


def test_the_defaults_outcome_given_to_the_exception_group_is_misattached():
    assert lab("For premium-tier cardholders, a 3% fee applies.") is Label.MISATTACHED


def test_an_exception_attached_to_the_wrong_group_is_misattached():
    assert lab("For student account holders, no fee applies.") is Label.MISATTACHED


def test_a_contradicting_figure_or_polarity_is_unsupported():
    assert lab("A 5% fee applies.") is Label.UNSUPPORTED
    assert lab("A fee of 7% applies to every transfer.") is Label.UNSUPPORTED


def test_filler_and_unrelated_sentences_are_extraneous():
    for c in ("Please contact us for more information.", "Our branches open at nine.", "This summary is general."):
        assert lab(c) is Label.EXTRANEOUS


# --------------------------------------------------------------------------- #
# Scoring answers
# --------------------------------------------------------------------------- #


def test_the_scoped_gold_answer_scores_perfectly_and_the_selection_answer_is_half_right():
    i = inst()
    gold = score(i, "A 3% fee applies (per p0). However, if the holder has a premium-tier card, no fee applies (per p1).")
    assert gold.precision == 1.0 and gold.covered_branches() == {"b0", "b1"} and gold.attachment_error_rate == 0.0
    sel = score(i, "A 3% fee applies.")
    assert sel.precision == 1.0 and sel.covered_branches() == {"b0"}, "precise but incomplete: precision is half the picture"


def test_padding_lowers_precision_without_adding_a_wrong_claim():
    i = inst()
    base = "A 3% fee applies. However, if the holder has a premium-tier card, no fee applies."
    padded = base + " Please keep a copy of your agreement. Contact us if you have questions. Branches open at nine."
    assert score(i, base).precision == 1.0
    p = score(i, padded)
    assert p.precision == pytest.approx(2 / 5) and p.count(Label.EXTRANEOUS) == 3 and p.n_words > score(i, base).n_words


def test_an_unattached_listing_is_misattached_not_aligned():
    s = score(inst(), "The outcome is either that a 3% fee applies or that no fee applies.")
    assert s.count(Label.ALIGNED) <= 1 and s.n_claims == 1


def test_no_claims_means_precision_zero_not_one():
    s = score(inst(), "")
    assert s.n_claims == 0 and s.precision == 0.0 and s.attachment_error_rate is None


def test_the_attachment_error_rate_counts_misattached_among_stated_outcomes():
    s = score(inst(), "A 3% fee applies. No fee applies. Contact us.")
    assert s.count(Label.ALIGNED) == 1 and s.count(Label.MISATTACHED) == 1 and s.attachment_error_rate == 0.5


def test_instances_without_branches_are_refused():
    with pytest.raises(ValueError, match="needs gold branches"):
        score(inst([]), "x")


# --------------------------------------------------------------------------- #
# The validity-battery claim: gold beats pasting every passage
# --------------------------------------------------------------------------- #


def test_gold_beats_a_concatenation_that_includes_distractors():
    base = next(i for i in generate(60, seed=0) if i.personas)
    noisy = base.model_copy(update={"passages": [*base.passages,
                                                 Passage(id="x1", text="Our branches open at nine on weekdays."),
                                                 Passage(id="x2", text="Please keep a copy of your agreement.")]})
    out = battery_outputs(noisy)
    gold = score(noisy, out["gold"]).precision
    assert gold == 1.0
    assert score(noisy, out["concat_k"]).precision < gold
    assert score(noisy, out["padded_gold"]).precision < gold


# --------------------------------------------------------------------------- #
# Fabrication on controls
# --------------------------------------------------------------------------- #


def test_inventing_an_exception_on_a_control_is_detected():
    control = inst([DEFAULT])
    assert fabricated_exception(control, "If you hold a premium card, no fee applies.") is True
    assert fabricated_exception(control, "A 3% fee applies.") is False
    with pytest.raises(ValueError, match="controls only"):
        fabricated_exception(inst(), "x")


# --------------------------------------------------------------------------- #
# Aggregation, the judge, V3
# --------------------------------------------------------------------------- #


def test_aggregate_is_micro_averaged_over_claims():
    a = score(inst(), "A 3% fee applies. Contact us. Branches open at nine. Keep a copy.")        # 1 of 4
    b = score(inst(), "A 3% fee applies.")                                                        # 1 of 1
    agg = aggregate([a, b])
    assert agg.precision == pytest.approx(2 / 5), "a long answer cannot hide its extraneous claims"
    assert agg.mean_claims == 2.5 and "scoped precision 0.400" in agg.render()
    with pytest.raises(ValueError):
        aggregate([])


def test_mean_precision_over_a_set_of_answers():
    i = inst()
    assert mean_precision([i], {"sc": "A 3% fee applies."}) == 1.0


def test_a_judge_can_rescue_an_unreadable_claim_but_never_overrule_a_matched_one():
    always_aligned = lambda claim, branches: Label.ALIGNED  # noqa: E731
    rescued = label_claim("Customers who hold the premium product pay nothing at all.", BRANCHES, judge=always_aligned)
    assert rescued.label in (Label.ALIGNED, Label.MISATTACHED) and rescued.by in ("judge", "deterministic")
    never = lambda claim, branches: Label.EXTRANEOUS  # noqa: E731
    assert label_claim("No fee applies.", BRANCHES, judge=never).label is Label.MISATTACHED, \
        "a judge cannot turn a misattached claim into filler"
    assert label_claim("A 3% fee applies.", BRANCHES, judge=never).label is Label.ALIGNED
    assert label_claim("Branches open at nine.", BRANCHES, judge=lambda c, b: None).label is Label.EXTRANEOUS


def test_the_eval_judge_uses_the_eval_tier_and_treats_garbage_as_no_verdict(tmp_path):
    good = MockBackend(scripted={"Claim:": json.dumps({"label": "aligned"})})
    c = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"), backend_override=good)
    assert eval_judge(c)("some claim", BRANCHES) is Label.ALIGNED
    assert good.calls[0]["model"] == "qwen/qwen3.8-27b"
    bad = MockBackend(scripted={"Claim:": json.dumps({"label": "great"})})
    c2 = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l2.jsonl"), backend_override=bad)
    assert eval_judge(c2)("x", BRANCHES) is None


def test_v3_accuracy_against_human_labels():
    pred = {"a": Label.ALIGNED, "b": Label.EXTRANEOUS, "c": Label.ALIGNED, "d": Label.UNSUPPORTED}
    human = {"a": Label.ALIGNED, "b": Label.EXTRANEOUS, "c": Label.MISATTACHED, "e": Label.ALIGNED}
    acc, n = validate_judge(pred, human)
    assert (acc, n) == (pytest.approx(2 / 3), 3) and V3_ACCURACY_BAR == 0.80
    assert validate_judge({}, human) == (None, 0)
    assert 0 < ATTACH_OVERLAP < 1
