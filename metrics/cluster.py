"""Cluster-aware inference: instances are the independent unit, nothing smaller is.

Branches, personas and pairs from one instance share passages and a query, so
they are not independent observations. Treating them as independent makes every
interval too narrow and every p-value too small -- the failure that makes a
small benchmark look decisive. So resampling is always **by instance**.

What the confirmatory analysis needs, beyond ``metrics.stats``:

* :func:`cluster_bootstrap` -- resample *instances*, then recompute a statistic
  over the persona- or branch-level rows they contain.
* :func:`bca` / :func:`bca_difference` -- bias-corrected and accelerated
  intervals, the interval proposal v8 section 10 specifies for the H2 F1
  difference (F1 is a ratio and its sampling distribution is skewed; the
  percentile interval is biased exactly where the sample is small).
* :func:`one_sided_lower` -- the one-sided 95% lower confidence bound that H2's
  15-point minimum effect of interest is checked against.
* :func:`sign_flip_test` -- an instance-level permutation test for paired
  differences: the distribution-free sensitivity check.
"""

from __future__ import annotations

import random
from itertools import product
from statistics import NormalDist
from typing import Callable, Sequence, TypeVar

from metrics.stats import DEFAULT_ALPHA, Interval

T = TypeVar("T")
_N = NormalDist()
DEFAULT_RESAMPLES = 4000


def _quantile(sorted_draws: list[float], q: float) -> float:
    i = min(len(sorted_draws) - 1, max(0, int(q * len(sorted_draws))))
    return sorted_draws[i]


def cluster_bootstrap(
    clusters: Sequence[Sequence[T]],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Percentile interval, resampling whole clusters and flattening their rows.

    ``clusters[i]`` is every row (persona, branch, pair) belonging to instance
    ``i``. The statistic sees the flattened rows of the resampled clusters.
    """
    if not clusters:
        return Interval(0.0, 0.0, 0.0, alpha, "cluster-bootstrap", 0)
    flat = [row for c in clusters for row in c]
    point = statistic(flat)
    rng = random.Random(seed)
    n = len(clusters)
    draws = []
    for _ in range(n_resamples):
        rows = [row for _ in range(n) for row in clusters[rng.randrange(n)]]
        try:
            draws.append(statistic(rows))
        except (ZeroDivisionError, ValueError):
            continue
    if not draws:
        return Interval(point, point, point, alpha, "cluster-bootstrap", n)
    draws.sort()
    return Interval(point, _quantile(draws, alpha / 2), _quantile(draws, 1 - alpha / 2),
                    alpha, "cluster-bootstrap", n)


def _bca_levels(draws: list[float], point: float, jack: list[float], alpha_lo: float, alpha_hi: float
                ) -> tuple[float, float]:
    """The BCa-adjusted percentile levels for the two tail probabilities."""
    below = sum(d < point for d in draws) / len(draws)
    below = min(max(below, 1.0 / (len(draws) + 1)), 1 - 1.0 / (len(draws) + 1))
    z0 = _N.inv_cdf(below)

    mean_j = sum(jack) / len(jack)
    num = sum((mean_j - j) ** 3 for j in jack)
    den = 6.0 * (sum((mean_j - j) ** 2 for j in jack) ** 1.5)
    a = num / den if den > 0 else 0.0

    def adj(p: float) -> float:
        z = _N.inv_cdf(p)
        denom = 1 - a * (z0 + z)
        if denom <= 0:
            return 0.0 if p < 0.5 else 1.0
        return min(max(_N.cdf(z0 + (z0 + z) / denom), 0.0), 1.0)

    return adj(alpha_lo), adj(alpha_hi)


def bca(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    one_sided: bool = False,
) -> Interval:
    """BCa interval over independent ``units`` (instances).

    ``one_sided=True`` returns a one-sided **lower** bound at level
    ``1 - alpha``: ``low`` is the bound and ``high`` is +inf. Fewer than three
    units has no usable jackknife; the percentile interval is returned and the
    method says so.
    """
    n = len(units)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, alpha, "bca", 0)
    point = statistic(units)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        try:
            draws.append(statistic(sample))
        except (ZeroDivisionError, ValueError):
            continue
    if not draws:
        return Interval(point, point, point, alpha, "bca", n)
    draws.sort()

    lo_p, hi_p = (alpha, 1 - 1e-9) if one_sided else (alpha / 2, 1 - alpha / 2)
    method = "bca-one-sided-lower" if one_sided else "bca"
    if n >= 3:
        try:
            jack = [statistic([u for k, u in enumerate(units) if k != i]) for i in range(n)]
            lo_p, hi_p = _bca_levels(draws, point, jack, lo_p, hi_p)
        except (ZeroDivisionError, ValueError):
            method += "(percentile-fallback)"
    else:
        method += "(percentile-fallback)"
    high = float("inf") if one_sided else _quantile(draws, hi_p)
    return Interval(point, _quantile(draws, lo_p), high, alpha, method, n)


def bca_difference(
    units: Sequence[T],
    stat_a: Callable[[Sequence[T]], float],
    stat_b: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    one_sided: bool = False,
) -> Interval:
    """BCa interval for ``stat_a - stat_b`` on the *same* resamples (pairing preserved)."""
    return bca(units, lambda u: stat_a(u) - stat_b(u), n_resamples=n_resamples,
               alpha=alpha, seed=seed, one_sided=one_sided)


def one_sided_lower(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> float:
    """One-sided ``1 - alpha`` lower confidence bound (BCa). H2 compares this to 0.15."""
    return bca(units, statistic, n_resamples=n_resamples, alpha=alpha, seed=seed, one_sided=True).low


def sign_flip_test(
    diffs: Sequence[float],
    *,
    n_monte_carlo: int = 100_000,
    exact_max: int = 16,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation (sign-flip) p-value for ``mean(diffs) == 0``.

    Under the null each instance's difference is as likely to be positive as
    negative, so flip signs at random and ask how often the mean is at least as
    extreme. Exact by enumeration up to ``exact_max`` instances, seeded Monte
    Carlo (with the +1 correction, so p is never 0) above that.
    """
    d = [float(x) for x in diffs]
    if not d:
        raise ValueError("no differences")
    obs = abs(sum(d))
    if all(x == 0 for x in d):
        return 1.0
    eps = 1e-12
    if len(d) <= exact_max:
        extreme = total = 0
        for signs in product((1, -1), repeat=len(d)):
            total += 1
            if abs(sum(s * x for s, x in zip(signs, d))) >= obs - eps:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    extreme = sum(
        abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - eps for _ in range(n_monte_carlo))
    return (extreme + 1) / (n_monte_carlo + 1)
