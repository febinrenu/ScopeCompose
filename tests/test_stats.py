"""Tests for the statistics layer.

These guard the numbers that go in the paper. A bug here does not crash
anything -- it produces an interval that is too narrow or a p-value that is too
small, which is the failure mode that survives review and then does not
replicate.

Known values are checked against hand-computable cases rather than against the
implementation's own output.
"""

from __future__ import annotations

import pytest

from metrics.stats import (
    Interval,
    bootstrap,
    bootstrap_difference,
    holm_adjust,
    mcnemar,
    render_intervals,
    wilson,
)


# --------------------------------------------------------------------------- #
# Wilson
# --------------------------------------------------------------------------- #


def test_wilson_matches_a_hand_computation():
    """45/50 at 95%, worked through so the reference is checkable.

        z      = 1.959964,  p = 0.9,  n = 50
        denom  = 1 + z^2/n                      = 1.076786
        centre = (p + z^2/2n) / denom           = 0.871478
        half   = (z/denom) * sqrt(p(1-p)/n + z^2/4n^2)
               = 1.820213 * sqrt(0.0018 + 0.000384)
               = 0.085074
        -> [0.78640, 0.95652]

    Tolerance is 1e-4, not tighter: the reference above rounds its
    intermediates, so asserting more precision than the hand computation
    carries would be testing the arithmetic of this docstring rather than the
    implementation.
    """
    iv = wilson(45, 50)
    assert iv.point == 0.9
    assert iv.low == pytest.approx(0.78640, abs=1e-4)
    assert iv.high == pytest.approx(0.95652, abs=1e-4)


def test_wilson_does_not_claim_certainty_at_a_boundary():
    """The whole reason for preferring Wilson here.

    72 successes out of 72 is not proof the rate is 1.0. The normal
    approximation returns [1.0, 1.0] and claims certainty from 72
    observations; Wilson gives a real lower bound.
    """
    iv = wilson(72, 72)
    assert iv.point == 1.0
    assert iv.high == 1.0
    assert iv.low < 1.0
    assert iv.low == pytest.approx(0.949, abs=0.01)


def test_wilson_at_zero_successes_bounds_the_rate_from_above():
    """A 0% observed leak rate on 19 items is consistent with a real rate
    around 17%. Reporting the point estimate alone hides that entirely."""
    iv = wilson(0, 19)
    assert iv.point == 0.0
    assert iv.low == 0.0
    assert 0.10 < iv.high < 0.25


def test_wilson_interval_narrows_as_n_grows():
    assert wilson(50, 100).width > wilson(500, 1000).width > wilson(5000, 10000).width


def test_wilson_stays_inside_zero_one():
    for k, n in [(0, 5), (5, 5), (1, 3), (99, 100)]:
        iv = wilson(k, n)
        assert 0.0 <= iv.low <= iv.high <= 1.0


def test_wilson_of_nothing_is_maximally_uncertain():
    iv = wilson(0, 0)
    assert (iv.low, iv.high) == (0.0, 1.0)


def test_tighter_alpha_widens_the_interval():
    assert wilson(45, 50, alpha=0.01).width > wilson(45, 50, alpha=0.05).width


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _mean(xs):
    return sum(xs) / len(xs)


def test_bootstrap_brackets_the_point_estimate():
    data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    iv = bootstrap(data, _mean, n_resamples=500)
    assert iv.low <= iv.point <= iv.high
    assert iv.point == pytest.approx(3.0)


def test_bootstrap_is_reproducible_under_a_seed():
    data = list(range(50))
    a = bootstrap(data, _mean, n_resamples=300, seed=7)
    b = bootstrap(data, _mean, n_resamples=300, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_bootstrap_seeds_differ():
    data = list(range(50))
    a = bootstrap(data, _mean, n_resamples=300, seed=1)
    b = bootstrap(data, _mean, n_resamples=300, seed=2)
    assert (a.low, a.high) != (b.low, b.high)


def test_bootstrap_narrows_as_the_sample_grows():
    small = bootstrap([1.0, 5.0] * 10, _mean, n_resamples=400, seed=0)
    large = bootstrap([1.0, 5.0] * 500, _mean, n_resamples=400, seed=0)
    assert large.width < small.width


def test_bootstrap_of_nothing_is_safe():
    assert bootstrap([], _mean, n_resamples=10).n == 0


def test_bootstrap_survives_a_statistic_that_divides_by_zero():
    """Some resamples contain no positives, so a rate over them is undefined.
    Those draws are skipped rather than crashing the run."""
    def rate(xs):
        pos = [x for x in xs if x > 0]
        return sum(pos) / len(pos)

    assert bootstrap([0, 0, 0, 1], rate, n_resamples=200, seed=0).n == 4


def test_bootstrap_difference_is_paired_not_two_separate_intervals():
    """Two systems differing by a constant on every item have a difference
    interval that excludes zero, even though their individual intervals
    overlap heavily. Comparing separate intervals is a weaker, different
    test -- and a common mistake."""
    units = [(x, x + 0.5) for x in [1.0, 2.0, 3.0, 4.0, 5.0] * 12]
    a = bootstrap(units, lambda s: _mean([u[1] for u in s]), n_resamples=400, seed=0)
    b = bootstrap(units, lambda s: _mean([u[0] for u in s]), n_resamples=400, seed=0)
    assert a.low < b.high, "individual intervals should overlap in this fixture"

    diff = bootstrap_difference(
        units,
        lambda s: _mean([u[1] for u in s]),
        lambda s: _mean([u[0] for u in s]),
        n_resamples=400, seed=0,
    )
    assert diff.point == pytest.approx(0.5)
    assert diff.excludes(0.0), "a constant advantage must be detectable when paired"


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #


def test_mcnemar_counts_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    r = mcnemar(a, b)
    assert r.b == 2          # a right, b wrong
    assert r.c == 1          # b right, a wrong
    assert r.both_right == 1
    assert r.both_wrong == 1
    assert r.n == 5


def test_identical_systems_are_not_distinguishable():
    outcomes = [True, False, True, True, False] * 10
    r = mcnemar(outcomes, outcomes)
    assert r.n_discordant == 0
    assert r.p_value == 1.0
    assert not r.significant()


def test_a_consistently_better_system_is_detected():
    a = [True] * 30 + [False] * 5
    b = [False] * 30 + [False] * 5
    r = mcnemar(a, b)
    assert r.b == 30 and r.c == 0
    assert r.p_value < 0.001
    assert r.significant()


def test_small_discordant_counts_use_the_exact_test():
    """On a few hundred instances the number of items where two systems
    disagree is often single digits, and the chi-square approximation is
    unreliable at that size."""
    assert mcnemar([True, True, False], [False, True, False]).method == "exact binomial"


def test_large_discordant_counts_use_chi_square():
    a = [True] * 40 + [False] * 40
    b = [False] * 40 + [True] * 40
    assert mcnemar(a, b).method.startswith("chi-square")


def test_exact_binomial_matches_a_hand_computation():
    """b=1, c=4: two-sided p = 2 * P(X <= 1) for n=5, p=0.5
    = 2 * (1 + 5) / 32 = 0.375."""
    a = [True] + [False] * 4
    b = [False] + [True] * 4
    assert mcnemar(a, b).p_value == pytest.approx(0.375, abs=1e-9)


def test_mcnemar_rejects_mismatched_lengths():
    """The test is only meaningful on the SAME items."""
    with pytest.raises(ValueError, match="SAME items"):
        mcnemar([True, False], [True])


def test_underpowered_comparison_is_flagged_in_the_report():
    out = mcnemar([True, False, True], [False, False, True]).render()
    assert "almost no power" in out
    assert "absence of evidence" in out


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #


def test_holm_is_stricter_than_raw_alpha():
    """Running a dozen comparisons and reporting the one that cleared 0.05 is
    how a null result becomes a finding."""
    adj = holm_adjust({"a": 0.04, "b": 0.20, "c": 0.60})
    assert adj["a"][0] > 0.04
    assert not adj["a"][1], "0.04 across three comparisons should not survive Holm"


def test_holm_keeps_a_strong_result():
    assert holm_adjust({"a": 0.0001, "b": 0.5, "c": 0.9})["a"][1]


def test_holm_stops_at_the_first_failure():
    """Once one hypothesis fails, every larger p must also fail -- that is what
    makes the procedure control the family-wise error rate."""
    adj = holm_adjust({"a": 0.001, "b": 0.30, "c": 0.002})
    assert "b" not in [n for n, (_, r) in adj.items() if r]


def test_holm_adjusted_values_are_non_decreasing():
    adj = holm_adjust({"a": 0.001, "b": 0.002, "c": 0.30})
    ordered = [adj[n][0] for n in ("a", "b", "c")]
    assert ordered == sorted(ordered)


def test_holm_is_less_conservative_than_bonferroni():
    raw = {"a": 0.001, "b": 0.002, "c": 0.003}
    assert holm_adjust(raw)["c"][0] < 3 * raw["c"]


def test_holm_of_nothing_is_empty():
    assert holm_adjust({}) == {}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_render_warns_when_an_interval_is_uselessly_wide():
    rows = {"tight": wilson(500, 1000), "hopeless": wilson(1, 3)}
    assert "supports very little" in render_intervals(rows, title="t")


def test_render_is_quiet_when_every_interval_is_tight():
    assert "supports very little" not in render_intervals(
        {"a": wilson(500, 1000)}, title="t"
    )


def test_interval_excludes_detects_a_real_difference():
    assert Interval(0.5, 0.2, 0.8).excludes(0.0)
    assert not Interval(0.1, -0.2, 0.4).excludes(0.0)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_seeding_makes_random_draws_repeatable():
    import random

    import reproducibility

    reproducibility.seed_everything(123)
    first = [random.random() for _ in range(5)]
    reproducibility.seed_everything(123)
    assert [random.random() for _ in range(5)] == first


def test_manifest_captures_what_a_number_depends_on():
    import reproducibility

    m = reproducibility.run_manifest(seed=42)
    assert m.seed == 42
    assert m.contract_version
    assert m.hardware_profile
    assert len(m.fingerprint()) == 12


def test_manifest_fingerprint_changes_with_the_seed():
    import reproducibility

    assert (reproducibility.run_manifest(seed=1).fingerprint()
            != reproducibility.run_manifest(seed=2).fingerprint())


def test_manifest_flags_an_untuned_threshold_band():
    """A reader must be able to see from the provenance block that the
    escalation band was never fitted to data."""
    import reproducibility

    assert "PLACEHOLDER" in reproducibility.run_manifest().render()
