"""Tests for branch alignment and the decisive experiment.

This is the code that decides whether the paper's central claim survives, so
the tests concentrate on the two ways it can silently produce a wrong answer:
matching that is too strict (understating a baseline) and a comparison that is
not like-for-like (overstating the pipeline).

Both of those actually happened during development. The first version reported
the pipeline beating the baseline by 66.7 points at p=3e-05; correcting the two
defects moved the same comparison to -8.3 points at p=0.73. The tests below are
the ones that would have caught it.
"""

from __future__ import annotations

import pytest

from contract.gold import Applicability, Branch, GoldInstance
from contract.models import ConflictType, Construction, Domain, Passage, ScopeRelation, SourceType
from metrics.branch_match import (
    BranchJudgement,
    align,
    containment,
    is_grounded,
    numbers_conflict,
    outcomes_match,
    similarity,
)


def _branch(bid, outcome, *, default=False, condition=None, descriptor="all",
            source="p0"):
    return Branch(
        branch_id=bid,
        condition=None if default else (condition or "some condition"),
        outcome=outcome,
        applicability=Applicability(descriptor=descriptor, is_default=default),
        supporting_passage=source,
    )


# --------------------------------------------------------------------------- #
# Outcome matching
# --------------------------------------------------------------------------- #


def test_verbose_prediction_of_the_right_answer_is_not_a_distortion():
    """The defect that inflated the pipeline's advantage.

    Gold outcomes are terse annotations; model outputs are verbose prose.
    Symmetric Jaccard punishes the prediction for its extra words, and scored
    two identical outcomes at 0.19 -- marking a correct answer as distorted and
    understating the baseline across the board.
    """
    gold = "no fee applies"
    pred = ("For these customers no fee applies to any international transaction "
            "made using the card or the mobile application")
    assert similarity(gold, pred) < 0.45, "Jaccard alone fails on a verbose prediction"
    assert containment(gold, pred) == 1.0, "every gold word is present"
    assert outcomes_match(gold, pred)


def test_containment_does_not_rescue_every_real_case():
    """Honest limit of the lexical fix.

    The case that first exposed the problem was gold "a 2.50 per-transaction
    charge applies" against "You will be charged a 2.50 fee per cash withdrawal
    plus any fee charged by the operator" -- same outcome, different vocabulary
    for it. Containment reaches only 0.5 there, so lexical matching still fails
    and the NLI signal is what is actually required. Recording that here rather
    than implying containment solved the whole problem.
    """
    gold = "a 2.50 per-transaction charge applies"
    pred = ("You will be charged a 2.50 fee per cash withdrawal plus any fee "
            "charged by the operator")
    assert containment(gold, pred) < 0.6
    assert not outcomes_match(gold, pred), "lexical alone cannot see this"


def test_containment_is_asymmetric():
    """What matters is whether the prediction SAYS what gold says, not whether
    it says only that."""
    terse, verbose = "no fee applies", "no fee applies to this account at all"
    assert containment(terse, verbose) > containment(verbose, terse)


def test_different_figures_are_never_the_same_outcome():
    """'a 3% fee applies' and 'a 5% fee applies' share almost every word, so
    numeric disagreement has to override both similarity signals."""
    assert numbers_conflict("a 3% fee applies", "a 5% fee applies")
    assert not outcomes_match("a 3% fee applies", "a 5% fee applies")


def test_numeric_guard_beats_high_containment():
    gold, pred = "a 3% fee applies", "a 5% fee applies to every transaction"
    assert containment(gold, pred) >= 0.6, "containment alone would match these"
    assert not outcomes_match(gold, pred)


def test_unrelated_outcomes_do_not_match():
    assert not outcomes_match("no fee applies", "employment is prohibited")


def test_nli_rescues_a_paraphrase_that_shares_no_words():
    """Lexical matching cannot see that 'no fee applies' and 'the charge is
    waived' are the same outcome. That is what the optional NLI signal is for."""
    from detection.nli import HeuristicNLI

    class AlwaysEntails:
        def score_batch(self, pairs):
            from detection.nli import NLIScores
            return [NLIScores(entailment=0.9, neutral=0.05, contradiction=0.05)
                    for _ in pairs]

    gold, pred = "no fee applies", "the charge is waived entirely"
    assert not outcomes_match(gold, pred), "lexical alone should miss this"
    assert outcomes_match(gold, pred, nli=AlwaysEntails())
    assert isinstance(HeuristicNLI(), object)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def test_a_matching_branch_is_preserved():
    gold = [_branch("b0", "a 3% fee applies", default=True)]
    pred = [_branch("p0", "a 3% fee applies", default=True)]
    r = align(gold, pred)
    assert r.preserved == 1 and r.suppressed == 0


def test_a_missing_branch_is_suppressed_not_distorted():
    """Suppression and distortion are different failures and must not be
    conflated: one loses the branch, the other corrupts it."""
    gold = [_branch("b0", "a 3% fee applies", default=True),
            _branch("b1", "no fee applies")]
    r = align(gold, [_branch("p0", "a 3% fee applies", default=True)])
    assert r.preserved == 1
    assert r.suppressed == 1
    assert r.distorted == 0


def test_a_wrong_outcome_is_distorted_not_suppressed():
    gold = [_branch("b0", "a 3% fee applies", default=True)]
    r = align(gold, [_branch("p0", "a 9% fee applies", default=True)])
    assert r.distorted == 1
    assert r.suppressed == 0


def test_a_default_cannot_be_matched_by_an_exception():
    """Letting an exception claim the default branch cascades: every
    subsequent alignment in the instance shifts."""
    gold = [_branch("b0", "a 3% fee applies", default=True)]
    r = align(gold, [_branch("p0", "a 3% fee applies", default=False)])
    assert r.alignments[0].judgement is BranchJudgement.SUPPRESSED


def test_each_prediction_is_used_at_most_once():
    gold = [_branch("b0", "a 3% fee applies", default=True),
            _branch("b1", "a 3% fee applies")]
    r = align(gold, [_branch("p0", "a 3% fee applies", default=True)])
    assert r.preserved == 1
    assert r.suppressed == 1


def test_extra_predictions_are_reported_for_hallucination_checking():
    gold = [_branch("b0", "a 3% fee applies", default=True)]
    pred = [_branch("p0", "a 3% fee applies", default=True),
            _branch("p1", "something entirely invented", source=None)]
    r = align(gold, pred)
    assert len(r.unmatched_predictions) == 1


def test_grounding_distinguishes_invention_from_an_annotation_gap():
    """A branch citing a real passage but absent from gold may be an
    annotation gap; only an ungrounded one is a fabrication."""
    known = {"p0", "p1"}
    assert is_grounded(_branch("x", "o", source="p1"), known)
    assert not is_grounded(_branch("x", "o", source=None), known)
    assert not is_grounded(_branch("x", "o", source="p9"), known)


def test_aligning_against_nothing_suppresses_everything():
    gold = [_branch("b0", "a 3% fee applies", default=True),
            _branch("b1", "no fee applies")]
    r = align(gold, [])
    assert r.suppressed == 2 and r.preserved == 0


# --------------------------------------------------------------------------- #
# The comparison itself
# --------------------------------------------------------------------------- #


def _instance(iid: str, ctype=ConflictType.CONDITIONAL,
              relation=ScopeRelation.REFINEMENT) -> GoldInstance:
    passages = [
        Passage(id="p0", text="International transactions incur a 3% fee.",
                source_type=SourceType.OFFICIAL_POLICY, date="2024-01",
                document_id=f"{iid}_a"),
        Passage(id="p1", text="The fee is waived for premium-tier cardholders.",
                source_type=SourceType.PRODUCT_TERMS, date="2024-02",
                document_id=f"{iid}_b"),
    ]
    branches = [
        _branch("b0", "a 3% fee applies", default=True, source="p0"),
        _branch("b1", "no fee applies", condition="premium tier",
                descriptor="premium holders", source="p1"),
    ]
    return GoldInstance(
        instance_id=iid, query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=passages,
        gold_conflict_type=ctype, gold_scope_relation=relation,
        gold_branches=branches,
    )


def test_oracle_routing_is_labelled_as_an_upper_bound():
    """Routing on gold labels is not the pipeline's score.

    Comparing it against a baseline that had to do its own detection flatters
    the pipeline by exactly the detector's error rate. That comparison produced
    a 66.7-point advantage at p=3e-05 before it was corrected, so the two modes
    must stay distinguishable in the output.
    """
    from experiments.run_decisive import score_pipeline_routing

    score = score_pipeline_routing([_instance("i1")], oracle=True)
    assert "oracle" in score.name
    assert score.preservation_rate == 1.0, "gold refinement labels route to compose"


def test_real_detection_is_the_default_and_is_labelled():
    from experiments.run_decisive import score_pipeline_routing

    score = score_pipeline_routing([_instance("i1")], oracle=False)
    assert "real" in score.name


def test_oracle_is_at_least_as_good_as_real_detection():
    """Perfect detection cannot do worse than the actual detector. If this ever
    fails, the routing or the scoring is wired wrong."""
    from experiments.run_decisive import score_pipeline_routing

    insts = [_instance(f"i{i}") for i in range(4)]
    oracle = score_pipeline_routing(insts, oracle=True)
    real = score_pipeline_routing(insts, oracle=False)
    assert oracle.preservation_rate >= real.preservation_rate


def test_a_failed_baseline_call_preserves_nothing():
    """Counting a failure as missing data would quietly flatter whichever
    system failed more often."""
    from baselines.structured_long_context import BaselineOutput
    from experiments.run_decisive import score_baseline

    inst = _instance("i1")
    failed = BaselineOutput(inst.instance_id, "structured", error="rate limited")
    score = score_baseline([inst], [failed], name="x")
    assert score.preservation_rate == 0.0
    assert score.suppressed == 2


def test_comparison_reports_a_null_result_as_a_null_result():
    """Per proposal 6.5 the falsifying outcome is reported as prominently as a
    positive one, so the renderer must say so rather than bury it."""
    from experiments.run_decisive import SystemScore, compare, render

    a = SystemScore(name="pipeline", gold_branches=10, preserved=5, suppressed=5)
    b = SystemScore(name="baseline", gold_branches=10, preserved=5, suppressed=5)
    for i in range(10):
        a.branch_preserved[("i", f"b{i}")] = i < 5
        b.branch_preserved[("i", f"b{i}")] = i < 5

    out = render(a, [b], compare(a, b))
    assert "INTERVAL CONTAINS ZERO" in out
    assert "falsifying outcome" in out


def test_comparison_warns_when_it_has_no_power():
    from experiments.run_decisive import SystemScore, compare, render

    a = SystemScore(name="pipeline", gold_branches=4, preserved=3, suppressed=1)
    b = SystemScore(name="baseline", gold_branches=4, preserved=2, suppressed=2)
    a.branch_preserved = {("i", "b0"): True, ("i", "b1"): True,
                          ("i", "b2"): True, ("i", "b3"): False}
    b.branch_preserved = {("i", "b0"): True, ("i", "b1"): True,
                          ("i", "b2"): False, ("i", "b3"): False}

    out = render(a, [b], compare(a, b))
    assert "no power" in out
    assert "absence of evidence" in out


def test_a_baseline_win_is_called_decisive():
    """A weaker model beating the pipeline cannot be explained away by model
    tier, so that direction is the one that actually settles the claim."""
    from experiments.run_decisive import SystemScore, compare, render

    a = SystemScore(name="pipeline", gold_branches=40)
    b = SystemScore(name="baseline", gold_branches=40)
    for i in range(40):
        a.branch_preserved[("i", f"b{i}")] = i < 10
        b.branch_preserved[("i", f"b{i}")] = i < 35
    a.preserved, a.suppressed = 10, 30
    b.preserved, b.suppressed = 35, 5

    out = render(a, [b], compare(a, b))
    assert "BASELINE PRESERVES MORE" in out
    assert "decisive falsification" in out
