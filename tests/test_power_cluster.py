"""Exact McNemar power, the MDD table, and cluster-aware inference."""

from __future__ import annotations

import random
from statistics import mean

import pytest

from metrics.cluster import (
    bca,
    bca_difference,
    cluster_bootstrap,
    one_sided_lower,
    sign_flip_test,
)
from metrics.power_sim import (
    HOLM_FIRST_STEP_ALPHA,
    SCENARIOS,
    mcnemar_power,
    mdd,
    mdd_table,
    observed_discordance,
    render_table,
    simulate_power,
)
from metrics.stats import mcnemar as stats_mcnemar

# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #

#: Proposal v8 section 6.7 (normal approximation): n -> (Holm d.15, Holm d.35, .05 d.25).
PROPOSAL = {60: (16, 24, 18), 80: (14, 21, 15), 140: (10, 16, 12), 220: (8, 13, 9)}


@pytest.mark.parametrize("n", sorted(PROPOSAL))
def test_exact_mdd_reproduces_the_proposal_table_within_one_point(n):
    got = mdd_table([n])[n]
    for (label, _, _), want in zip(SCENARIOS, PROPOSAL[n]):
        assert abs(got[label] - want) <= 1.0, f"n={n} {label}: exact {got[label]} vs proposal {want}"


def test_holm_first_step_is_alpha_over_three():
    assert HOLM_FIRST_STEP_ALPHA == pytest.approx(0.0166667, rel=1e-4)


def test_power_is_alpha_at_zero_effect_or_below_and_one_at_a_huge_effect():
    assert mcnemar_power(80, 0.05, 0.25, 0.0) <= 0.05 + 1e-9, "exact test never exceeds its size"
    assert mcnemar_power(200, 0.05, 0.4, 0.4) > 0.999


def test_power_increases_with_n_effect_and_alpha():
    assert mcnemar_power(60, 0.05, 0.25, 0.15) < mcnemar_power(140, 0.05, 0.25, 0.15)
    assert mcnemar_power(80, 0.05, 0.25, 0.10) < mcnemar_power(80, 0.05, 0.25, 0.20)
    assert mcnemar_power(80, 0.0167, 0.25, 0.15) < mcnemar_power(80, 0.05, 0.25, 0.15)


def test_power_is_symmetric_in_the_sign_of_the_effect():
    assert mcnemar_power(70, 0.05, 0.3, 0.2) == pytest.approx(mcnemar_power(70, 0.05, 0.3, -0.2))


def test_exact_power_agrees_with_a_seeded_simulation():
    exact = mcnemar_power(60, 0.05, 0.3, 0.2)
    sim = simulate_power(60, 0.05, 0.3, 0.2, reps=5000, seed=1)
    assert abs(exact - sim) < 0.03


def test_force_exact_agrees_with_the_power_calculation_and_the_default_does_not_above_25():
    """The pre-registered test is exact at ANY size. The default switches to a
    chi-square approximation at 25 discordant items, which gives a different p."""
    from metrics.power_sim import _two_sided_binom_p

    a = [True] * 20 + [False] * 5 + [True] * 10
    b = [False] * 20 + [True] * 5 + [True] * 10
    exact = stats_mcnemar(a, b, force_exact=True)
    assert exact.method == "exact binomial"
    assert exact.p_value == pytest.approx(_two_sided_binom_p(20, 25))
    default = stats_mcnemar(a, b)
    assert default.method.startswith("chi-square") and default.p_value != pytest.approx(exact.p_value)


def test_force_exact_changes_nothing_below_the_switch_point():
    a, b = [True] * 8 + [False] * 2, [False] * 8 + [True] * 2
    assert stats_mcnemar(a, b).p_value == stats_mcnemar(a, b, force_exact=True).p_value


def test_input_validation():
    with pytest.raises(ValueError):
        mcnemar_power(50, 0.05, 0.0, 0.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        mcnemar_power(50, 0.05, 0.1, 0.3)
    with pytest.raises(ValueError):
        observed_discordance([True], [True, False])
    with pytest.raises(ValueError):
        observed_discordance([], [])


def test_observed_discordance():
    assert observed_discordance([True, True, False, False], [True, False, False, True]) == 0.5


def test_mdd_is_none_when_no_effect_inside_d_is_detectable():
    assert mdd(8, 0.0167, 0.1) is None


def test_render_table_lists_every_n():
    text = render_table()
    for n in PROPOSAL:
        assert str(n) in text


# --------------------------------------------------------------------------- #
# Cluster bootstrap and BCa
# --------------------------------------------------------------------------- #


def test_clustering_widens_the_interval_when_rows_within_an_instance_agree():
    """Ten instances, six personas each, all personas of an instance identical.
    Treating the 60 rows as independent claims 60 observations; there are 10."""
    outcomes = [1] * 5 + [0] * 5
    clusters = [[float(o)] * 6 for o in outcomes]
    flat = [r for c in clusters for r in c]
    naive = cluster_bootstrap([[r] for r in flat], mean, seed=0, n_resamples=2000)
    clustered = cluster_bootstrap(clusters, mean, seed=0, n_resamples=2000)
    assert clustered.point == naive.point == 0.5
    assert clustered.width > naive.width * 1.8


def test_cluster_bootstrap_is_seeded_and_handles_empty():
    clusters = [[1.0, 0.0], [1.0], [0.0, 0.0, 1.0]]
    assert cluster_bootstrap(clusters, mean, seed=3) == cluster_bootstrap(clusters, mean, seed=3)
    assert cluster_bootstrap([], mean).n == 0


def test_bca_brackets_the_point_estimate_and_is_seeded():
    rng = random.Random(0)
    units = [rng.random() for _ in range(40)]
    iv = bca(units, mean, seed=1, n_resamples=2000)
    assert iv.low <= iv.point <= iv.high
    assert iv == bca(units, mean, seed=1, n_resamples=2000)
    assert iv.method == "bca"


def test_bca_coverage_for_a_mean_is_close_to_nominal():
    rng = random.Random(7)
    hits = 0
    reps = 150
    for _ in range(reps):
        units = [rng.gauss(0.0, 1.0) for _ in range(30)]
        iv = bca(units, mean, n_resamples=300, seed=rng.randrange(10**6))
        hits += iv.low <= 0.0 <= iv.high
    assert 0.88 <= hits / reps <= 0.995, hits / reps


def test_bca_corrects_for_skew_where_the_percentile_interval_is_biased():
    """A skewed statistic (a ratio on a small sample): BCa and percentile differ."""
    rng = random.Random(3)
    units = [(rng.random() < 0.15, rng.random() < 0.3) for _ in range(40)]

    def ratio(u):
        num = sum(1 for a, b in u if a)
        den = sum(1 for a, b in u if b)
        return num / den

    from metrics.stats import bootstrap

    b, p = bca(units, ratio, n_resamples=2000, seed=0), bootstrap(units, ratio, n_resamples=2000, seed=0)
    assert (b.low, b.high) != (p.low, p.high)


def test_bca_with_too_few_units_falls_back_and_says_so():
    iv = bca([1.0, 0.0], mean, n_resamples=200)
    assert "percentile-fallback" in iv.method


def test_bca_difference_preserves_pairing():
    units = [(1.0, 0.0)] * 30 + [(1.0, 1.0)] * 30
    iv = bca_difference(units, lambda u: mean(a for a, _ in u), lambda u: mean(b for _, b in u),
                        seed=0, n_resamples=1500)
    assert iv.point == pytest.approx(0.5) and iv.low > 0.3 and iv.excludes(0.0)


def test_one_sided_lower_bound_is_below_the_point_and_tighter_than_the_two_sided_lower():
    rng = random.Random(5)
    units = [rng.gauss(0.2, 0.3) for _ in range(50)]
    lb = one_sided_lower(units, mean, seed=0, n_resamples=3000)
    two = bca(units, mean, alpha=0.05, seed=0, n_resamples=3000)
    assert lb < mean(units)
    assert lb >= two.low - 0.02, "a one-sided 95% bound is not looser than a two-sided 95% lower end"
    one = bca(units, mean, alpha=0.05, seed=0, n_resamples=3000, one_sided=True)
    assert one.high == float("inf") and one.method.startswith("bca-one-sided")


# --------------------------------------------------------------------------- #
# Sign-flip test
# --------------------------------------------------------------------------- #


def test_sign_flip_exact_small_case():
    # four positive differences: only the all-plus and all-minus flips are as extreme
    assert sign_flip_test([1, 1, 1, 1]) == pytest.approx(2 / 16)


def test_sign_flip_is_one_for_no_difference_and_symmetric():
    assert sign_flip_test([0, 0, 0]) == 1.0
    assert sign_flip_test([1, -1, 1, -1]) == pytest.approx(sign_flip_test([-1, 1, -1, 1]))


def test_sign_flip_detects_a_consistent_effect_and_not_noise():
    assert sign_flip_test([0.2] * 30, n_monte_carlo=20000) < 0.001
    rng = random.Random(0)
    noise = [rng.choice([-1, 1]) * rng.random() for _ in range(40)]
    assert sign_flip_test(noise, n_monte_carlo=20000) > 0.05


def test_sign_flip_monte_carlo_is_seeded_and_never_zero():
    d = [0.5] * 40
    assert sign_flip_test(d, n_monte_carlo=5000, seed=1) == sign_flip_test(d, n_monte_carlo=5000, seed=1)
    assert sign_flip_test(d, n_monte_carlo=5000) > 0


def test_sign_flip_rejects_empty_input():
    with pytest.raises(ValueError):
        sign_flip_test([])
