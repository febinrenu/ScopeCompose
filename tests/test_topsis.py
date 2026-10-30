"""The B3 selection core: Entropy-TOPSIS against hand-computed values, the
sufficiency rule, the specificity ablation, and order independence."""

from __future__ import annotations

import itertools
import math

import pytest

from baselines.conflictrag_topsis import (
    DEFAULT_DELTA,
    credibility,
    entropy_weights,
    recency_scores,
    select_by_credibility,
    select_by_recency,
    specificity_score,
    topsis_closeness,
)
from contract.models import Passage, SourceType


def P(pid, text="Transfers incur a fee.", st=SourceType.UNKNOWN, date=None):
    return Passage(id=pid, text=text, source_type=st, date=date)


# --------------------------------------------------------------------------- #
# Entropy weights and closeness, against hand computation
# --------------------------------------------------------------------------- #


def test_symmetric_matrix_gets_equal_weights_and_equal_closeness():
    """[[1,3],[3,1]]: both columns have the same shares, so w = (0.5, 0.5), and
    the two alternatives are mirror images, so C = (0.5, 0.5)."""
    m = [[1, 3], [3, 1]]
    assert entropy_weights(m) == pytest.approx([0.5, 0.5])
    assert topsis_closeness(m, [0.5, 0.5]) == pytest.approx([0.5, 0.5])


def test_a_constant_criterion_has_zero_weight():
    """[[4,1],[2,1]]: column 2 is constant (entropy 1, weight 0); column 1 alone
    decides, so alternative 1 is the ideal (C=1) and alternative 2 the worst (C=0)."""
    m = [[4, 1], [2, 1]]
    assert entropy_weights(m) == pytest.approx([1.0, 0.0])
    assert topsis_closeness(m, [1.0, 0.0]) == pytest.approx([1.0, 0.0])


def test_entropy_matches_an_independent_formula():
    m = [[3.0, 5.0, 1.0], [1.0, 5.0, 4.0], [2.0, 5.0, 7.0]]
    w = entropy_weights(m)
    div = []
    for j in range(3):
        col = [m[i][j] for i in range(3)]
        t = sum(col)
        e = -sum((x / t) * math.log(x / t) for x in col) / math.log(3)
        div.append(1 - e)
    assert w == pytest.approx([d / sum(div) for d in div])
    assert sum(w) == pytest.approx(1.0) and w[1] == pytest.approx(0.0)


def test_weights_sum_to_one_and_closeness_is_in_unit_interval():
    import random

    rng = random.Random(0)
    for _ in range(50):
        m = [[rng.random() for _ in range(3)] for _ in range(rng.randint(2, 5))]
        w = entropy_weights(m)
        c = topsis_closeness(m, w)
        assert sum(w) == pytest.approx(1.0)
        assert all(-1e-9 <= x <= 1 + 1e-9 for x in c)


def test_degenerate_inputs_do_not_crash():
    assert entropy_weights([[1.0, 2.0]]) == [0.5, 0.5], "one alternative: nothing to choose between"
    assert entropy_weights([[0.0, 0.0], [0.0, 0.0]]) == [0.5, 0.5], "all-zero criteria cannot discriminate"
    assert topsis_closeness([[2, 2], [2, 2]], [0.5, 0.5]) == [0.5, 0.5]
    with pytest.raises(ValueError):
        entropy_weights([])
    with pytest.raises(ValueError):
        entropy_weights([[1, 2], [3]])
    with pytest.raises(ValueError):
        entropy_weights([[1, -2], [3, 4]])


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #


def test_recency_is_min_max_and_undated_scores_zero():
    ps = [P("a", date="2024-01-01"), P("b", date="2025-01-01"), P("c")]
    s = recency_scores(ps)
    assert s[0] == 0.0 and s[1] == 1.0 and s[2] == 0.0
    assert recency_scores([P("a", date="2024-05"), P("b", date="2024-05")]) == [1.0, 1.0]
    assert recency_scores([P("a"), P("b")]) == [0.0, 0.0]


def test_specificity_rewards_figures_and_conditions_and_is_bounded():
    plain = specificity_score("Transfers incur a fee.")
    narrow = specificity_score("For holders who are over 18, only if the balance exceeds 500, a 2.5% fee applies.")
    assert 0 <= plain < narrow < 1


# --------------------------------------------------------------------------- #
# Selection and the sufficiency rule
# --------------------------------------------------------------------------- #


def test_a_clearly_more_credible_source_is_selected():
    ps = [P("terms", st=SourceType.OFFICIAL_POLICY, date="2025-01-01"),
          P("blog", st=SourceType.THIRD_PARTY, date="2023-01-01")]
    sel = select_by_credibility(ps)
    assert sel.kind == "selected" and sel.chosen == ("terms",)
    assert sel.result.margin > DEFAULT_DELTA


def test_insufficient_margin_presents_both_sources_instead_of_choosing():
    a = P("a", st=SourceType.PRODUCT_TERMS, date="2025-01-01")
    b = P("b", st=SourceType.PRODUCT_TERMS, date="2025-01-01")
    sel = select_by_credibility([a, b])
    assert sel.kind == "both" and sel.chosen == ("a", "b")
    assert "insufficient" in sel.notes[0]


def test_delta_is_the_dial_between_selecting_and_presenting_both():
    ps = [P("a", st=SourceType.OFFICIAL_POLICY, date="2025-01-01"),
          P("b", st=SourceType.PRODUCT_TERMS, date="2024-06-01")]
    margin = credibility(ps).margin
    assert select_by_credibility(ps, delta=margin - 0.01).kind == "selected"
    assert select_by_credibility(ps, delta=margin + 0.01).kind == "both"


def test_a_single_passage_is_trivially_selected():
    sel = select_by_credibility([P("only")])
    assert sel.kind == "selected" and sel.chosen == ("only",)


def test_selection_is_independent_of_retrieval_order():
    ps = [P("a", st=SourceType.OFFICIAL_POLICY, date="2025-03-01"),
          P("b", st=SourceType.PRODUCT_TERMS, date="2024-01-01"),
          P("c", "For holders over 18 only, a 2.5% fee applies.", SourceType.FAQ, "2024-09-01")]
    base = select_by_credibility(ps)
    for perm in itertools.permutations(ps):
        assert select_by_credibility(list(perm)).chosen == base.chosen
        r = credibility(list(perm))
        assert {s.passage_id: round(s.closeness, 9) for s in r.scored} == {
            s.passage_id: round(s.closeness, 9) for s in credibility(ps).scored}


def test_ablating_specificity_removes_the_criterion_and_can_flip_the_winner():
    """The specificity criterion partly re-creates the exception-over-rule
    preference by a side door; the ablation exists to measure that."""
    rule = P("rule", "Transfers incur a fee.", SourceType.OFFICIAL_POLICY, "2024-01-01")
    exc = P("exc", "For premium holders over 18 only, where the balance exceeds 500, the fee is waived.",
            SourceType.PRODUCT_TERMS, "2025-01-01")
    with_spec = credibility([rule, exc], include_specificity=True)
    without = credibility([rule, exc], include_specificity=False)
    assert "specificity" in with_spec.weights and "specificity" not in without.weights
    assert all("specificity" not in s.criteria for s in without.scored)
    assert with_spec.ranked[0].passage_id != without.ranked[0].passage_id or with_spec.margin != without.margin


def test_no_passages_is_an_error():
    with pytest.raises(ValueError):
        credibility([])


# --------------------------------------------------------------------------- #
# Recency (the temporal branch)
# --------------------------------------------------------------------------- #


def test_recency_picks_the_newest_and_undated_never_wins():
    ps = [P("old", date="2023-01-01"), P("new", date="2025-06-01"), P("undated")]
    assert select_by_recency(ps).chosen == ("new",)


def test_recency_tie_keeps_all_and_no_dates_selects_nothing():
    tie = select_by_recency([P("a", date="2025-01-01"), P("b", date="2025-01-01"), P("c", date="2020-01-01")])
    assert tie.chosen == ("a", "b") and "tie" in tie.notes[0]
    none = select_by_recency([P("a"), P("b")])
    assert none.kind == "both" and none.chosen == ("a", "b")
