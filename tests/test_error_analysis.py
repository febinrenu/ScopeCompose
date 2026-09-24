"""Tests for the qualitative error analysis.

Proposal §5.2 asks for error analysis on the confused instances, and the point
of it is to tell a systematic failure from a scatter of one-offs. So the tests
are about the grouping, not about the counting: an analysis that lists twenty
errors without noticing that eighteen share a cause has done nothing the
confusion matrix did not already do.
"""

from __future__ import annotations

import pytest

from contract.models import ConflictType, ScopeRelation
from experiments.error_analysis import ErrorCase, ErrorReport, analyse


def _case(gold="conditional", pred="factual", *, cues=0.0, clash=0.0,
          entail=0.1, neutral=0.2, contra=0.7, asym=0.0, qid="q0",
          gold_rel=None, pred_rel=None):
    return ErrorCase(
        query_id=qid, query="do I pay a fee?", pair=("p0", "p1"),
        text_i="International transactions incur a 3% fee.",
        text_j="The fee is waived for premium-tier cardholders.",
        gold_type=gold, pred_type=pred,
        gold_relation=gold_rel, pred_relation=pred_rel,
        nli_entailment=entail, nli_neutral=neutral, nli_contradiction=contra,
        exception_cues=cues, restriction_asymmetry=asym, numeric_clash=clash,
    )


# --------------------------------------------------------------------------- #
# Error kinds
# --------------------------------------------------------------------------- #


def test_the_two_directions_of_confusion_are_not_merged():
    """They are different bugs with different consequences.

    conditional -> factual routes to selection and discards the exception --
    the failure mode this project exists to fix, occurring inside this
    project's own pipeline. factual -> conditional routes to composition and
    presents two incompatible claims as if each held under some condition,
    inventing a scope that does not exist. A single "confused" count hides
    which one is happening.
    """
    r = ErrorReport(n_pairs=4, n_errors=2, cases=[
        _case("conditional", "factual"),
        _case("factual", "conditional"),
    ])
    kinds = r.by_kind()
    assert kinds["conditional -> factual"] == 1
    assert kinds["factual -> conditional"] == 1


def test_a_relation_error_is_reported_by_cell_not_as_one_bucket():
    """refinement-as-redundant loses a branch; redundant-as-refinement
    fabricates one. Opposite failures, so they are separate cells."""
    r = ErrorReport(cases=[
        _case("conditional", "conditional", gold_rel="refinement", pred_rel="redundant"),
        _case("conditional", "conditional", gold_rel="redundant", pred_rel="refinement"),
    ])
    kinds = r.by_kind()
    assert kinds["relation refinement -> redundant"] == 1
    assert kinds["relation redundant -> refinement"] == 1


# --------------------------------------------------------------------------- #
# Signatures -- the part that turns a list into a finding
# --------------------------------------------------------------------------- #


def test_a_shared_cause_produces_a_shared_signature():
    """Six conditional instances missed for the same reason -- an exception
    stated without any cue word -- must land in one bucket, or the analysis
    reports six anecdotes instead of one fixable weakness."""
    r = ErrorReport(cases=[_case(cues=0.0, contra=0.9, qid=f"q{i}") for i in range(6)])
    sigs = r.by_signature("conditional -> factual")
    assert len(sigs) == 1
    assert sigs.most_common(1)[0][1] == 6


def test_signatures_separate_cases_that_failed_differently():
    cued = _case(cues=2.0)
    uncued = _case(cues=0.0)
    assert cued.signature() != uncued.signature()
    assert "no-cue" in uncued.signature()


def test_signature_reports_the_dominant_nli_class():
    assert "nli-contradiction" in _case(entail=0.1, neutral=0.2, contra=0.7).signature()
    assert "nli-neutral" in _case(entail=0.1, neutral=0.7, contra=0.2).signature()
    assert "nli-entailment" in _case(entail=0.7, neutral=0.2, contra=0.1).signature()


def test_a_dominant_pattern_is_called_systematic():
    """The whole reason for grouping. Five of six errors sharing a cause is a
    weakness with a name; the renderer has to say so rather than leave the
    reader to spot it in a table."""
    cases = [_case(cues=0.0, qid=f"q{i}") for i in range(5)]
    cases.append(_case(cues=3.0, clash=1.0, qid="q9"))
    out = ErrorReport(n_pairs=20, n_errors=6, cases=cases).render()
    assert "systematic" in out


def test_a_scatter_of_one_offs_is_not_called_systematic():
    """The counterpart. Four errors with four different causes is not a
    finding, and labelling it one would be the analysis inventing a result."""
    cases = [
        _case(cues=0.0, clash=0.0, entail=0.1, neutral=0.1, contra=0.8, qid="q0"),
        _case(cues=2.0, clash=1.0, entail=0.1, neutral=0.8, contra=0.1, qid="q1"),
        _case(cues=2.0, clash=0.0, entail=0.8, neutral=0.1, contra=0.1, asym=2.0, qid="q2"),
        _case(cues=0.0, clash=1.0, entail=0.1, neutral=0.8, contra=0.1, asym=2.0, qid="q3"),
    ]
    out = ErrorReport(n_pairs=20, n_errors=4, cases=cases).render()
    assert "systematic" not in out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_a_clean_run_is_not_reported_as_a_result():
    """Zero errors on eight mock instances is a fixture property. Printing it
    as an accuracy claim is exactly the kind of number §12 of the
    implementation plan forbids."""
    out = ErrorReport(n_pairs=8, n_errors=0).render()
    assert "fixture result" in out


def test_empty_report_does_not_divide_by_zero():
    assert ErrorReport().error_rate == 0.0


def test_markdown_sheet_asks_the_question_that_produces_a_fix():
    md = ErrorReport(n_pairs=2, n_errors=1, cases=[_case()]).to_markdown()
    assert "gold label is correct" in md
    assert "enough signal" in md
    assert "3% fee" in md, "the sheet must carry the actual passage text"


def test_worked_examples_are_capped_per_kind():
    cases = [_case(qid=f"q{i}") for i in range(20)]
    out = ErrorReport(n_pairs=40, n_errors=20, cases=cases).render(sample=3)
    assert out.count("pattern:") == 3


# --------------------------------------------------------------------------- #
# Against the real pipeline
# --------------------------------------------------------------------------- #


def test_correct_predictions_are_counted_but_not_reported_as_errors():
    """An error rate needs its denominator. Counting only the errors would
    make one mistake in a thousand look the same as one in two."""
    from contract.mock import generate

    gold = [g.to_query_record() for g in generate(6, seed=0)]
    report = analyse(gold, gold)  # predictions identical to gold

    assert report.n_pairs > 0, "pairs must be scored even when nothing is wrong"
    assert report.n_errors == 0
    assert report.error_rate == 0.0


def test_a_flipped_label_is_caught_with_its_features_attached():
    """The features are the payload. An error case without them says a mistake
    happened; with them it says what the detector was looking at when it made
    the mistake, which is the only version that points at a fix.
    """
    from contract.mock import generate

    gold = [g.to_query_record() for g in generate(8, seed=0)]
    conditional = [r for r in gold
                   if any(p.type is ConflictType.CONDITIONAL for p in r.conflict_pairs)]
    assert conditional, "mock data must contain a conditional pair to flip"

    target = conditional[0]
    flipped = target.model_copy(update={"conflict_pairs": [
        p.model_copy(update={"type": ConflictType.FACTUAL, "scope_relation": None})
        if p.type is ConflictType.CONDITIONAL else p
        for p in target.conflict_pairs
    ]})

    report = analyse([flipped], [target])
    assert report.n_errors >= 1
    case = next(c for c in report.cases if c.kind == "conditional -> factual")
    assert case.text_i, "the passage text must be attached"
    assert case.signature()


def test_records_absent_from_gold_are_skipped_not_scored_as_errors():
    """A prediction with no gold counterpart cannot be judged either way.

    Scoring it as correct inflates accuracy; scoring it as an error invents a
    mistake. Both are wrong, so it is skipped and left out of the denominator.

    The ids have to be renamed explicitly rather than taken from a second seed:
    mock ids are derived from the template name, so two seeds produce the SAME
    ids and this test would pass without ever exercising the skip.
    """
    from contract.mock import generate

    gold = [g.to_query_record() for g in generate(4, seed=0)]
    unknown = [r.model_copy(update={"query_id": f"not_in_gold_{i}"})
               for i, r in enumerate(gold)]
    assert not {r.query_id for r in unknown} & {r.query_id for r in gold}

    report = analyse(unknown, gold)
    assert report.n_pairs == 0, "an unjudgeable prediction is not a denominator entry"
    assert report.n_errors == 0
