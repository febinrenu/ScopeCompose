"""QNFA pair decisions (D5) on hand-built frames: no model is involved, so every outcome is
decided by the signature algebra and the scope algebra alone."""

from __future__ import annotations

import random

import pytest

from contract.frames import Modality, NormativeFrame
from contract.gold import Applicability, AttributeKind, ScopeAttribute
from detection.qnfa import QNFA, Decision, commensurable
from detection.signature import Kind, make_signature

DEFAULT = Applicability(descriptor="everyone", is_default=True)


def cat(name, *vals):
    return ScopeAttribute(name=name, kind=AttributeKind.CATEGORICAL, values=list(vals))


def rng(name, lo=None, hi=None):
    return ScopeAttribute(name=name, kind=AttributeKind.NUMERIC_RANGE, min_value=lo, max_value=hi)


def fr(pid, outcome, *, app=None, sig=None, span=None, modality=Modality.STATEMENT, domain="banking"):
    return NormativeFrame(
        passage_id=pid, signature=make_signature(domain, sig or {"event": "cash_withdrawal",
                                                                  "decision_kind": "flat_fee"}),
        applicability=app or DEFAULT, outcome=outcome, modality=modality, span=span or f"span of {pid}")


PREMIUM = Applicability(descriptor="premium cardholders", attributes=[cat("tier", "premium")])
PLATINUM = Applicability(descriptor="platinum cardholders", attributes=[cat("tier", "platinum")])
q = QNFA()


def only(result):
    assert len(result.pairs) == 1, result
    return result.pairs[0]


# --------------------------------------------------------------------------- #
# Relations
# --------------------------------------------------------------------------- #


def test_a_rule_and_its_exception_is_a_conditional_conflict_and_names_the_general_passage():
    p = only(q.detect_frames([fr("p0", "a 3% fee applies"), fr("p1", "no fee applies", app=PREMIUM)]))
    assert p.decision is Decision.CONDITIONAL and p.is_conditional
    assert p.general == "p0" and p.unification is Kind.EQUIVALENT and p.score == pytest.approx(0.95 * 1.0 * 0.95)


def test_the_general_passage_is_found_regardless_of_argument_order():
    a, b = fr("p0", "a 3% fee applies"), fr("p1", "no fee applies", app=PREMIUM)
    assert q.decide_pair(a, b)[0].general == "p0" and q.decide_pair(b, a)[0].general == "p0"


def test_agreeing_outcomes_over_nested_scopes_are_redundant_not_conflict():
    p = only(q.detect_frames([fr("p0", "a 3% fee applies"), fr("p1", "a 3 per cent fee applies", app=PREMIUM)]))
    assert p.decision is Decision.REDUNDANT and not p.is_conditional


def test_equal_scopes_that_disagree_are_a_same_scope_conflict():
    p = only(q.detect_frames([fr("p0", "a 3% fee applies", app=PREMIUM), fr("p1", "a 5% fee applies", app=PREMIUM)]))
    assert p.decision is Decision.SAME_SCOPE


def test_crossing_scopes_are_crossing_never_conditional():
    a = fr("p0", "employment is permitted", sig={"event": "work", "decision_kind": "permission"}, domain="immigration",
           app=Applicability(descriptor="first year", attributes=[cat("year", "first")]))
    b = fr("p1", "employment is prohibited", sig={"event": "work", "decision_kind": "permission"}, domain="immigration",
           app=Applicability(descriptor="part-time", attributes=[cat("mode", "part_time")]))
    p = only(q.detect_frames([a, b]))
    assert p.decision is Decision.CROSSING and not p.is_conditional


def test_disjoint_scopes_are_not_a_conflict():
    young = Applicability(descriptor="under 18", attributes=[rng("age", 0, 17)])
    old = Applicability(descriptor="65 and over", attributes=[rng("age", 65, None)])
    p = only(q.detect_frames([fr("p0", "a 1% fee applies", app=young), fr("p1", "a 5% fee applies", app=old)]))
    assert p.decision is Decision.DISJOINT


def test_scopes_that_cannot_be_compared_are_unknown_with_a_zero_score():
    free = ScopeAttribute(name="circumstances", kind=AttributeKind.FREE_TEXT, text="in exceptional hardship")
    a = fr("p0", "a 3% fee applies", app=PREMIUM)
    b = fr("p1", "no fee applies", app=Applicability(descriptor="hardship", attributes=[free]))
    p = only(q.detect_frames([a, b]))
    assert p.decision is Decision.UNKNOWN and p.score == 0.0 and not p.is_conditional


# --------------------------------------------------------------------------- #
# The point of the method
# --------------------------------------------------------------------------- #


def test_a_rule_and_carve_out_that_share_no_words_still_align_through_signatures():
    """The pilot detector gated on lexical topicality and recovered 2 of 67 of these. No similarity
    of any kind is consulted here: the signatures align the pair."""
    sig = {"event": "cash_withdrawal", "location": "abroad", "decision_kind": "flat_fee"}
    rule = fr("p0", "a flat 2.50 charge applies", span="Overseas teller machine use costs 2.50.", sig=sig)
    carve = fr("p1", "no charge applies", span="Reward holders enjoy fee-free access worldwide.", sig=sig,
               app=Applicability(descriptor="reward holders", attributes=[cat("account", "reward")]))
    shared = set(rule.span.lower().replace(".", "").split()) & set(carve.span.lower().replace(".", "").split())
    assert shared == set(), f"the premise of this test is that the wording is disjoint, got {shared}"
    p = only(q.detect_frames([rule, carve]))
    assert p.decision is Decision.CONDITIONAL and p.general == "p0" and p.unification is Kind.EQUIVALENT


def test_an_atm_only_rule_against_an_all_cash_carve_out_is_crossing_not_a_refinement():
    """Each is broader than the other somewhere: ATM-only-for-everyone vs all-cash-for-reward-holders.
    A reward holder at an ATM is covered by both, so the overlap is contested -- not 'rule plus exception'."""
    rule = fr("p0", "a flat 2.50 charge applies",
              sig={"event": "atm_withdrawal", "location": "abroad", "decision_kind": "flat_fee"})
    carve = fr("p1", "no charge applies", sig={"event": "cash_withdrawal", "location": "abroad",
                                               "decision_kind": "flat_fee"},
               app=Applicability(descriptor="reward holders", attributes=[cat("account", "reward")]))
    p = only(q.detect_frames([rule, carve]))
    assert p.unification is Kind.SUBSUMES
    assert p.general is None, "only a refinement names a general-rule passage"
    assert p.decision is Decision.CROSSING and not p.is_conditional


def test_signature_subsumption_turns_a_narrower_variable_into_a_refinement():
    broad = fr("p0", "a 3% fee applies", sig={"event": "cash_withdrawal", "location": "abroad",
                                              "decision_kind": "percentage_fee"})
    narrow = fr("p1", "no fee applies", sig={"event": "atm_withdrawal", "location": "abroad",
                                             "decision_kind": "percentage_fee"})
    p = only(q.detect_frames([broad, narrow]))
    assert p.decision is Decision.CONDITIONAL and p.unification is Kind.SUBSUMES and p.general == "p0"


# --------------------------------------------------------------------------- #
# What it skips
# --------------------------------------------------------------------------- #


def test_different_decisions_are_skipped_and_counted():
    flat = fr("p0", "a 2.50 fee applies", sig={"event": "transfer", "decision_kind": "flat_fee"})
    pct = fr("p1", "a 3% fee applies", sig={"event": "transfer", "decision_kind": "percentage_fee"}, app=PREMIUM)
    r = q.detect_frames([flat, pct])
    assert r.pairs == [] and r.skipped_distinct == 1


def test_incommensurable_outcomes_are_skipped_and_counted():
    a = fr("p0", "a 2.50 fee applies", sig={"event": "currency_conversion"})
    b = fr("p1", "the conversion rate is 1.2", sig={"event": "currency_conversion"}, app=PREMIUM)
    r = q.detect_frames([a, b])
    assert r.pairs == [] and r.skipped_incommensurable == 1


def test_an_uncertain_pair_abstains_without_a_tiebreak_and_is_decided_with_one():
    free = make_signature("banking", {"event": "lounge entry", "decision_kind": "access"})
    a = NormativeFrame(passage_id="p0", signature=free, applicability=DEFAULT, outcome="lounge access is included",
                       span="s0")
    b = fr("p1", "lounge access is not included", sig={"event": "cash_withdrawal", "decision_kind": "access"},
           app=PREMIUM)
    r = q.detect_frames([a, b])
    assert r.pairs == [] and r.abstained_uncertain == 1
    decided = QNFA(tiebreak=lambda x, y: Kind.EQUIVALENT).detect_frames([a, b])
    p = only(decided)
    assert p.decision is Decision.CONDITIONAL and p.score < 0.95 * 0.95 * 1.0, "a tie-break lowers the score"


def test_a_tiebreak_that_abstains_or_says_distinct_decides_nothing():
    free = make_signature("banking", {"event": "lounge entry", "decision_kind": "access"})
    a = NormativeFrame(passage_id="p0", signature=free, applicability=DEFAULT, outcome="lounge access is included",
                       span="s0")
    b = fr("p1", "lounge access is not included", sig={"event": "cash_withdrawal", "decision_kind": "access"},
           app=PREMIUM)
    for answer in (None, Kind.UNCERTAIN, Kind.DISTINCT):
        assert QNFA(tiebreak=lambda x, y, a_=answer: a_).detect_frames([a, b]).pairs == []


def test_hedged_frames_never_form_a_conflict():
    a = fr("p0", "a 3% fee applies")
    b = fr("p1", "no fee applies", app=PREMIUM, modality=Modality.HEDGED)
    p = only(q.detect_frames([a, b]))
    assert p.decision is Decision.UNKNOWN and not p.is_conditional


def test_pairs_within_one_passage_are_skipped_unless_asked_for():
    a, b = fr("p0", "a 3% fee applies"), fr("p0", "no fee applies", app=PREMIUM, span="another span")
    assert q.detect_frames([a, b]).pairs == []
    assert len(q.detect_frames([a, b], cross_source_only=False).pairs) == 1


# --------------------------------------------------------------------------- #
# Invariances
# --------------------------------------------------------------------------- #


def test_the_result_does_not_depend_on_frame_order():
    frames = [fr("p0", "a 3% fee applies"), fr("p1", "no fee applies", app=PREMIUM),
              fr("p2", "a 1% fee applies", app=PLATINUM)]
    base = [(p.key, p.decision, p.score) for p in q.detect_frames(frames).pairs]
    r = random.Random(0)
    for _ in range(8):
        shuffled = frames[:]
        r.shuffle(shuffled)
        assert [(p.key, p.decision, p.score) for p in q.detect_frames(shuffled).pairs] == base


def test_scores_rank_attribute_decided_pairs_above_fallback_ones():
    clean = only(q.detect_frames([fr("p0", "a 3% fee applies"), fr("p1", "no fee applies", app=PREMIUM)]))
    lexical = only(q.detect_frames([fr("p0", "the fee is high"), fr("p1", "the fee is low", app=PREMIUM)]))
    assert lexical.decision is Decision.CONDITIONAL and lexical.score < clean.score


def test_the_strongest_decision_per_passage_pair_prefers_a_conditional_conflict():
    frames = [fr("p0", "a 3% fee applies"), fr("p0", "a 5% fee applies", app=PREMIUM, span="other"),
              fr("p1", "no fee applies", app=PREMIUM)]
    res = q.detect_frames(frames)
    best = q.best_conditional_per_passage_pair(res)
    assert best[("p0", "p1")].decision is Decision.CONDITIONAL
    assert len(best) == 1


# --------------------------------------------------------------------------- #
# Commensurability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("a,b,expected", [
    ("a 3% fee applies", "no fee applies", True),
    ("employment is permitted", "employment is prohibited", True),
    ("funds of 1270 must be held", "funds of 1000 must be held", True),
    ("a 2.50 fee applies", "the conversion rate is 1.2", False),
    ("a 3% fee applies", "lounge access is included", False),
    ("", "no fee applies", False),
])
def test_commensurability(a, b, expected):
    assert commensurable(a, b) is expected


def test_a_charge_and_a_fee_are_the_same_quantity():
    assert commensurable("a flat 2.50 charge applies", "no charge applies")
    assert commensurable("a 3% surcharge applies", "no fee applies")
    assert not commensurable("a 2.50 charge applies", "the conversion rate is 1.2")
