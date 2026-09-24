"""Tests for B1 (probing), B2 (composition), B3 (generation) and the metrics.

The tests concentrate on the ways each stage can silently undo the stage before
it. Composition can fabricate a branch by recording a restatement as an
exception; generation can drop one by adjudicating; the scorer can credit a
branch that was never preserved. All three produce a plausible headline number
and none of them raises.
"""

from __future__ import annotations

import pytest

from composition import (
    ComposedAnswer,
    CompositionOperator,
    Resolution,
    classify_exception_pair,
)
from contract.gold import Applicability, Branch, GoldInstance, ScopeAttribute
from contract.gold import AttributeKind
from contract.models import (
    ConflictPair,
    ConflictType,
    Construction,
    Domain,
    Passage,
    QueryRecord,
    ScopeRelation,
    SourceType,
)
from generation import ScopedAnswerGenerator, check_faithfulness, render_template
from metrics.branch_match import BranchJudgement, polarity_conflict
from metrics.preservation import judge_branches, score_preservation, validate_judge


def _cat(name: str, *values: str) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.CATEGORICAL, values=list(values))


def _branch(bid, outcome, *, default=False, condition=None, descriptor="all cases",
            attrs=None, source="p0"):
    return Branch(
        branch_id=bid,
        condition=None if default else (condition or "some condition"),
        outcome=outcome,
        applicability=Applicability(descriptor=descriptor, is_default=default,
                                    attributes=list(attrs or [])),
        supporting_passage=source,
    )


def _record(pairs=None, *, qid="q1") -> QueryRecord:
    return QueryRecord(
        query_id=qid, query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL,
        passages=[
            Passage(id="p0", text="International transactions incur a 3% fee.",
                    source_type=SourceType.OFFICIAL_POLICY, date="2024-01",
                    document_id="d0"),
            Passage(id="p1", text="The fee is waived for premium cardholders.",
                    source_type=SourceType.PRODUCT_TERMS, date="2024-02",
                    document_id="d1"),
        ],
        conflict_pairs=pairs if pairs is not None else [],
    )


def _pair(relation=ScopeRelation.REFINEMENT, ctype=ConflictType.CONDITIONAL):
    return ConflictPair(doc_i="p0", doc_j="p1", is_conflict=True, type=ctype,
                        scope_relation=relation, confidence=0.9)


# --------------------------------------------------------------------------- #
# The negation bug in outcome matching
# --------------------------------------------------------------------------- #


def test_an_outcome_and_its_negation_are_not_the_same_outcome():
    """The bug that made the composition operator miss nested structure.

    "no" and "not" were stopwords, so "no fee applies" and "a fee applies"
    scored a Jaccard of 1.00 -- a perfect match between exact opposites. A
    system emitting the reverse of a gold branch was scored as preserving it,
    which inflates PR for every system at once and is invisible in the output.
    """
    from metrics.branch_match import outcomes_match, similarity

    assert not outcomes_match("no fee applies", "a fee applies")
    assert not outcomes_match("the certificate is required",
                              "no certificate is required")
    assert similarity("no fee applies", "a fee applies") < 1.0, \
        "negation must be a content word, not a stopword"


def test_two_negations_agreeing_are_not_a_conflict():
    """Parity, not presence. "never lapses" and "does not lapse" are both
    negated and agree; flagging on the mere presence of a negator would call
    them a contradiction."""
    assert not polarity_conflict("leave does not lapse", "leave never lapses")


def test_a_quantifier_bound_is_not_a_negation():
    """"Employment is permitted for no more than 20 hours" is a permission with
    a limit, not a prohibition. The bare "no" made it read as the opposite
    polarity to "employment is permitted", so the branch the answer actually
    stated was scored as dropped."""
    assert not polarity_conflict(
        "employment is permitted",
        "for holders working no more than 20 hours, employment is permitted")
    # A real negation alongside a quantifier is still a negation.
    assert polarity_conflict("no fee applies", "a fee of no more than 3% applies")


def test_a_decimal_is_not_a_clause_boundary():
    """A plain ``[;.]`` split tore "a 2.75% fee applies" into "a 2" and
    "75% fee applies", so the branch matched neither fragment and every answer
    quoting a decimal was reported as having dropped its own branches."""
    branches = [_branch("b0", "a 2.75% fee applies", default=True)]
    report = check_faithfulness("By default, a 2.75% fee applies (per p0).", branches)
    assert report.missing_branches == []
    assert report.ok


def test_polarity_check_does_not_fire_on_unrelated_outcomes():
    """Without a content-overlap guard every negated outcome would conflict
    with every unrelated positive one -- and crucially, a hard veto there would
    stop NLI ever rescuing a genuine paraphrase like "no fee applies" against
    "the charge is waived"."""
    assert not polarity_conflict("no fee applies", "the charge is waived")
    assert not polarity_conflict("no fee applies", "employment is prohibited")


# --------------------------------------------------------------------------- #
# B2: composition
# --------------------------------------------------------------------------- #


def test_a_refinement_composes_and_keeps_both_branches():
    op = CompositionOperator()
    branches = [_branch("b0", "a 3% fee applies", default=True),
                _branch("b1", "no fee applies", descriptor="premium holders",
                        source="p1")]
    a = op.compose(_record([_pair()]), branches)
    assert a.resolution is Resolution.COMPOSED
    assert a.n_branches == 2


def test_an_opposed_pair_falls_back_to_selection_and_says_so():
    """Selection is prior work's operator. Using it where composition is not
    honest is correct; hiding that it happened is not."""
    op = CompositionOperator()
    a = op.compose(_record([_pair(ScopeRelation.OPPOSED)]),
                   [_branch("b0", "x", default=True), _branch("b1", "y", source="p1")])
    assert a.resolution is Resolution.SELECTED
    assert a.selected_passage is not None


def test_a_restatement_is_merged_not_recorded_as_a_branch():
    """The precise failure that separating 'redundant' from 'refinement'
    exists to prevent. A restatement kept as a second branch is a fabricated
    branch, and it inflates Preservation Rate with something gold never had.
    """
    op = CompositionOperator()
    branches = [_branch("b0", "a 2.99% charge applies", default=True),
                _branch("b1", "a 2.99% charge applies", descriptor="premium holders",
                        source="p1")]
    a = op.compose(_record([_pair(ScopeRelation.REDUNDANT)]), branches)
    assert a.resolution is Resolution.MERGED
    assert a.n_branches == 1
    assert a.merged_pairs == 1


def test_prior_work_classes_are_not_composed():
    op = CompositionOperator()
    a = op.compose(_record([_pair(None, ConflictType.FACTUAL)]),
                   [_branch("b0", "x", default=True)])
    assert a.resolution is Resolution.PRIOR_WORK


def test_no_conflict_passes_through():
    op = CompositionOperator()
    a = op.compose(_record([]), [_branch("b0", "x", default=True)])
    assert a.resolution is Resolution.PASS_THROUGH


# -- second-order structure -------------------------------------------------- #


def test_nested_exceptions_are_flagged_not_composed():
    """An exception carved back by a further exception needs
    exception-to-exception precedence, which is second-order and out of scope.
    Flagged honestly rather than resolved by guessing."""
    inner = _branch("b1", "no fee applies", descriptor="premium holders",
                    attrs=[_cat("tier", "premium")], source="p1")
    deeper = _branch("b2", "a 3% fee applies",
                     descriptor="premium holders over 5000",
                     attrs=[_cat("tier", "premium"), _cat("amount_band", "over_5000")],
                     source="p1")
    flag, why = classify_exception_pair(inner, deeper)
    assert flag == "nested"
    assert "precedence" in why


def test_crossed_exceptions_are_flagged_separately_from_nested():
    """Two independent exceptions that can co-occur and disagree. A different
    pattern from nested, and reported as its own rate -- collapsing them into
    one 'complex' flag discards the distinction."""
    a = _branch("b1", "employment permitted", descriptor="hardship holders",
                attrs=[_cat("authorisation", "hardship")], source="p1")
    b = _branch("b2", "employment prohibited", descriptor="part-time holders",
                attrs=[_cat("enrolment", "part_time")], source="p1")
    flag, _ = classify_exception_pair(a, b)
    assert flag == "crossed"


def test_nested_scopes_that_agree_are_not_flagged():
    """Nesting alone is not second-order structure. Without disagreement there
    is no precedence question to answer."""
    a = _branch("b1", "no fee applies", descriptor="premium holders",
                attrs=[_cat("tier", "premium")], source="p1")
    b = _branch("b2", "no fee applies", descriptor="premium holders over 5000",
                attrs=[_cat("tier", "premium"), _cat("amount_band", "over_5000")],
                source="p1")
    assert classify_exception_pair(a, b)[0] is None


def test_disjoint_exceptions_are_not_flagged():
    a = _branch("b1", "x", descriptor="students",
                attrs=[_cat("kind", "student")], source="p1")
    b = _branch("b2", "y", descriptor="pensioners",
                attrs=[_cat("kind", "pensioner")], source="p1")
    assert classify_exception_pair(a, b)[0] is None


def test_an_undecidable_scope_pair_is_not_flagged():
    """Flagging on ignorance would inflate the nested and crossed rates, and
    those rates are reported as findings about the data."""
    a = _branch("b1", "x", descriptor="some unparsed scope", source="p1")
    b = _branch("b2", "y", descriptor="another unparsed scope", source="p1")
    assert classify_exception_pair(a, b)[0] is None


def test_a_flagged_instance_reduces_to_the_selected_branch():
    """Routing to selection while still reporting every branch would credit
    the instance with preserving branches the answer never states."""
    op = CompositionOperator()
    branches = [
        _branch("b0", "employment prohibited", default=True),
        _branch("b1", "employment permitted", descriptor="hardship holders",
                attrs=[_cat("authorisation", "hardship")], source="p1"),
        _branch("b2", "employment prohibited", descriptor="part-time holders",
                attrs=[_cat("enrolment", "part_time")], source="p1"),
    ]
    a = op.compose(_record([_pair()]), branches)
    assert a.resolution is Resolution.SELECTED
    assert a.flags.crossed
    assert a.n_branches < 3


# --------------------------------------------------------------------------- #
# B3: generation
# --------------------------------------------------------------------------- #


def test_the_template_renderer_states_every_branch_with_attribution():
    a = ComposedAnswer("q1", Resolution.COMPOSED, [
        _branch("b0", "a 3% fee applies", default=True),
        _branch("b1", "no fee applies", descriptor="premium cardholders", source="p1"),
    ])
    text = render_template(a)
    assert "3% fee" in text and "no fee" in text
    assert "p0" in text and "p1" in text


def test_a_dropped_branch_is_caught():
    """Suppression at the generation step, after composition preserved the
    branch correctly. Invisible to answer-correctness scoring, which is the
    blind spot this project is about, reappearing at the last step."""
    branches = [_branch("b0", "a 3% fee applies", default=True),
                _branch("b1", "no fee applies", descriptor="premium", source="p1")]
    report = check_faithfulness("A 3% fee applies (per p0).", branches)
    assert not report.ok
    assert "b1" in report.missing_branches


def test_an_invented_figure_is_caught():
    branches = [_branch("b0", "a 3% fee applies", default=True)]
    report = check_faithfulness("A 3% fee applies, plus a 12 monthly charge (per p0).",
                                branches)
    assert "12" in report.unsupported_numbers


def test_a_passage_id_is_not_mistaken_for_an_invented_figure():
    """The lookbehind in the number regex. Without it "(per p0)" reported a
    figure no branch contained, and every correctly-attributed answer failed --
    including the template renderer's, which cannot invent anything."""
    branches = [_branch("b0", "no fee applies", default=True)]
    report = check_faithfulness("No fee applies (per p0).", branches)
    assert report.unsupported_numbers == []
    assert report.ok


def test_adjudication_language_is_caught():
    """The generator ranking sources instead of scoping them is selection
    under another name, undoing the pipeline at the last step."""
    branches = [_branch("b0", "a 3% fee applies", default=True)]
    report = check_faithfulness(
        "A 3% fee applies (per p0), which is more authoritative.", branches)
    assert report.adjudicated
    assert not report.ok


def test_the_template_path_costs_nothing_and_is_faithful():
    gen = ScopedAnswerGenerator(use_llm=False)
    a = ComposedAnswer("q1", Resolution.COMPOSED, [
        _branch("b0", "a 3% fee applies", default=True),
        _branch("b1", "no fee applies", descriptor="premium cardholders", source="p1"),
    ])
    out = gen.generate(a, "do I pay a fee?")
    assert out.method == "template"
    assert out.api_calls == 0
    assert out.is_faithful


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _gold(iid="q1", *, ctype=ConflictType.CONDITIONAL,
          relation=ScopeRelation.REFINEMENT, branches=None,
          distractor=False) -> GoldInstance:
    return GoldInstance(
        instance_id=iid, query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL,
        passages=_record().passages,
        gold_conflict_type=ctype,
        gold_scope_relation=relation if ctype is ConflictType.CONDITIONAL else None,
        is_distractor=distractor,
        gold_branches=branches if branches is not None else [
            _branch("b0", "a 3% fee applies", default=True),
            _branch("b1", "no fee applies", descriptor="premium holders", source="p1"),
        ],
    )


def test_the_two_branch_judgement_enums_are_one_enum():
    """They were two identical copies, so ``align()`` returned one class and
    the scorer compared against the other. Every identity check was False and a
    perfectly preserved run scored PR = 0.0 with every branch distorted."""
    from metrics.branch_match import BranchJudgement as A
    from metrics.preservation import BranchJudgement as B
    assert A is B


def test_preservation_separates_composition_from_selection():
    """The central instrument. If SR does not rise when branches are dropped,
    the project has no way to show the thing it exists to show."""
    golds = [_gold(f"q{i}") for i in range(6)]
    composed = [ComposedAnswer(g.instance_id, Resolution.COMPOSED, g.gold_branches)
                for g in golds]
    selected = [ComposedAnswer(g.instance_id, Resolution.SELECTED, g.gold_branches[:1])
                for g in golds]

    pr_c = score_preservation(composed, golds)
    pr_s = score_preservation(selected, golds)

    assert pr_c.preservation_rate == 1.0
    assert pr_s.suppression_rate > pr_c.suppression_rate
    assert pr_s.suppression_rate == pytest.approx(0.5)


def test_results_are_broken_out_by_tier():
    """Blending Tier 1 and Tier 2 would let constructed instances carry a claim
    about naturally occurring ones."""
    natural = _gold("q1")
    split = _gold("q2")
    split = split.model_copy(update={"construction": Construction.SPLIT})
    outs = [ComposedAnswer(g.instance_id, Resolution.COMPOSED, g.gold_branches)
            for g in (natural, split)]

    scores = score_preservation(outs, [natural, split])
    assert set(scores.by_construction) == {"natural", "split"}


def test_hallucinated_conditions_need_a_real_passage():
    ungrounded = [_branch("b0", "a 3% fee applies", default=True),
                  _branch("b1", "no fee applies", descriptor="premium",
                          source="p99")]
    gold = _gold()
    scores = score_preservation(
        [ComposedAnswer(gold.instance_id, Resolution.COMPOSED, ungrounded)], [gold])
    assert scores.hallucinated_conditions == 1


def test_spurious_conditions_are_counted_only_where_gold_has_none():
    """SCR guards the opposite failure to suppression: a caveat-happy system
    inventing structure to game PR. Its denominator is the distractors, which
    is why they are mandatory in the benchmark."""
    plain = _gold("q1", ctype=ConflictType.FACTUAL, relation=None,
                  branches=[_branch("b0", "a 3% fee applies", default=True)])
    invented = [_branch("b0", "a 3% fee applies", default=True),
                _branch("b1", "no fee applies", descriptor="premium", source="p1")]

    scores = score_preservation(
        [ComposedAnswer(plain.instance_id, Resolution.COMPOSED, invented)], [plain])
    assert scores.no_conflict_instances == 1
    assert scores.spurious_conditions == 1
    assert scores.spurious_condition_rate == 1.0


def test_scr_has_no_denominator_without_distractors():
    """Stated as a test because a rate of 0.0 from an empty denominator reads
    identically to a genuinely clean run."""
    gold = _gold()
    scores = score_preservation(
        [ComposedAnswer(gold.instance_id, Resolution.COMPOSED, gold.gold_branches)],
        [gold])
    assert scores.no_conflict_instances == 0
    assert scores.spurious_condition_rate == 0.0


def test_the_llm_judge_path_refuses_rather_than_silently_substituting():
    """Proposal 6.1 requires the judge validated against human labels before
    its numbers are reported. An unvalidated judge that quietly works would get
    used."""
    with pytest.raises(NotImplementedError, match="validated against human labels"):
        judge_branches("text", [], client=object())


def test_judge_validation_reports_agreement_and_ranking_correlation():
    """A judge can agree on most branches and still mis-rank two systems if
    its errors concentrate in one of them -- and the ranking is what a
    conclusion rests on."""
    P, S = BranchJudgement.PRESERVED, BranchJudgement.SUPPRESSED
    llm = {"q1": [P, P], "q2": [P, S], "q3": [S, S]}
    human = {"q1": [P, P], "q2": [P, S], "q3": [S, S]}

    out = validate_judge(llm, human)
    assert out["branch_agreement"] == 1.0
    assert out["meets_precedent_bar"] == 1.0
    assert out["pearson_pr"] == pytest.approx(1.0)


def test_judge_validation_rejects_misaligned_judgement_lists():
    with pytest.raises(ValueError, match="same gold branches"):
        validate_judge({"q1": [BranchJudgement.PRESERVED]},
                       {"q1": [BranchJudgement.PRESERVED,
                               BranchJudgement.SUPPRESSED]})


# --------------------------------------------------------------------------- #
# B1: probing
# --------------------------------------------------------------------------- #


def test_the_grounding_gate_rejects_an_unsupported_condition():
    """Without the gate a condition proposer is a hallucination engine."""
    from extraction import ContrastiveScopeProbe
    from extraction.probing import CandidateCondition

    class NeverEntails:
        def score_batch(self, pairs):
            from detection.nli import NLIScores
            return [NLIScores(entailment=0.01, neutral=0.9, contradiction=0.09)
                    for _ in pairs]

    probe = ContrastiveScopeProbe(use_llm=False, nli=NeverEntails())
    cand = CandidateCondition(condition="the holder is a student",
                              outcome="no fee applies", applicability="students",
                              evidence_span="", source_passage="p0")
    scored = probe.gate([cand], {"p0": "International transactions incur a 3% fee."})
    assert not scored[0].admitted
    assert "does not support" in scored[0].rejection_reason


def test_the_gate_admits_a_supported_condition():
    from extraction import ContrastiveScopeProbe
    from extraction.probing import CandidateCondition

    class AlwaysEntails:
        def score_batch(self, pairs):
            from detection.nli import NLIScores
            return [NLIScores(entailment=0.95, neutral=0.03, contradiction=0.02)
                    for _ in pairs]

    probe = ContrastiveScopeProbe(use_llm=False, nli=AlwaysEntails())
    cand = CandidateCondition(condition="the holder is premium",
                              outcome="no fee applies", applicability="premium",
                              evidence_span="", source_passage="p1")
    scored = probe.gate([cand], {"p1": "The fee is waived for premium cardholders."})
    assert scored[0].admitted


def test_a_gate_that_never_fires_is_reported_as_suspicious():
    """A gate rejecting nothing is not evidence that nothing was hallucinated;
    it is evidence the threshold does not bite."""
    from extraction.probing import ProbeStats

    stats = ProbeStats(instances=10, candidates=20, admitted=20, rejected=0,
                       explicit_admitted=10, implicit_admitted=10)
    assert "rejected NOTHING" in stats.render()


def test_explicit_and_implicit_admissions_are_reported_separately():
    """Averaging them hides the number the method is judged on: the unmarked
    cases are the hard half."""
    from extraction.probing import ProbeStats

    stats = ProbeStats(instances=5, candidates=10, admitted=6, rejected=4,
                       explicit_admitted=6, implicit_admitted=0)
    out = stats.render()
    assert "explicitly marked" in out
    assert "unmarked" in out
    assert "could not" in out, "an all-explicit run must say it proved nothing"


def test_duplicate_candidates_from_different_questions_collapse():
    """Several probe questions usually recover the same condition. Counting
    them separately would inflate both the candidate count and apparent
    recall."""
    from extraction.probing import CandidateCondition, ContrastiveScopeProbe

    dupes = [CandidateCondition(condition="the holder is premium", outcome="no fee",
                                applicability="premium", evidence_span="",
                                source_passage="p1", question=f"q{i}")
             for i in range(3)]
    assert len(ContrastiveScopeProbe._dedupe(dupes)) == 1
