"""Uncertainty and significance for every reported number.

A single point estimate with no interval does not survive review at a serious
venue, and this project's numbers are computed on a corpus of a few hundred
instances where sampling error is large. Two systems differing by three points
of F1 on 300 items may well be indistinguishable.

Three tools, each matched to a specific shape of question:

**Wilson intervals** for a simple proportion — accuracy, a leak rate, a
preservation rate. Preferred over the normal approximation because the rates
here sit near 0 or 1 (an order-invariance pass rate of 1.0000, a leak rate of
0.02), and the normal interval misbehaves badly at the extremes: it produces
bounds outside [0, 1] and collapses to zero width at exactly 0 or 1, claiming
certainty from a small sample.

**Bootstrap intervals** for anything that is not a simple proportion — F1,
macro-F1, a confusion cell, a ratio of ratios. Resampled at the INSTANCE level,
never the pair level, because pairs drawn from the same instance share passages
and are not independent; resampling pairs would understate the interval.

**McNemar's test** for comparing two systems on the same items. The three
detector variants are scored on identical pairs, so the comparison is paired,
and an unpaired test throws away exactly the information that makes a modest
difference detectable.

Nothing here corrects for multiple comparisons on its own. :func:`holm_adjust`
is provided and the ablation suite should use it: running a dozen comparisons
and reporting the one that cleared 0.05 is how a null result becomes a finding.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Interval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    alpha: float = DEFAULT_ALPHA
    method: str = ""
    n: int = 0

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """Whether ``value`` falls outside the interval.

        The usual question is ``excludes(0.0)`` for a difference: an interval
        that contains zero is a difference the data cannot distinguish from
        none.
        """
        return not (self.low <= value <= self.high)

    def render(self, pct: bool = True, places: int = 3) -> str:
        if pct:
            return (f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]")
        return (f"{self.point:.{places}f} "
                f"[{self.low:.{places}f}, {self.high:.{places}f}]")

    def as_dict(self) -> dict:
        return {
            "point": round(self.point, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "confidence": self.confidence,
            "method": self.method,
            "n": self.n,
        }


# --------------------------------------------------------------------------- #
# Proportions
# --------------------------------------------------------------------------- #

#: Two-sided z for a few common alphas. Avoids a scipy dependency for what is
#: a three-entry lookup in practice.
_Z = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}


def _z_for(alpha: float) -> float:
    if alpha in _Z:
        return _Z[alpha]
    # Acklam's inverse-normal approximation; accurate to ~1e-9 over (0,1).
    p = 1.0 - alpha / 2.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def wilson(successes: int, n: int, *, alpha: float = DEFAULT_ALPHA) -> Interval:
    """Wilson score interval for a proportion.

    Correct at the boundaries, which is where this project lives: an
    order-invariance pass rate of 72/72 gets a real upper-bounded interval
    rather than the normal approximation's [1.0, 1.0], which would claim
    certainty from 72 observations.
    """
    if n <= 0:
        return Interval(0.0, 0.0, 1.0, alpha, "wilson", 0)

    z = _z_for(alpha)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half),
                    alpha, "wilson", n)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for any statistic.

    ``units`` must be the INDEPENDENT sampling units. For this project that is
    instances, not pairs: ten pairs drawn from one instance share passages, and
    resampling them independently pretends there is more information in the
    corpus than there is, producing intervals that are too narrow.

    Seeded, so a reported interval is reproducible.
    """
    if not units:
        return Interval(0.0, 0.0, 0.0, alpha, "bootstrap", 0)

    point = statistic(units)
    rng = random.Random(seed)
    n = len(units)
    draws = []
    for _ in range(n_resamples):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        try:
            draws.append(statistic(sample))
        except (ZeroDivisionError, ValueError):
            continue

    if not draws:
        return Interval(point, point, point, alpha, "bootstrap", n)

    draws.sort()
    lo = draws[max(0, int((alpha / 2) * len(draws)))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return Interval(point, lo, hi, alpha, "bootstrap", n)


def bootstrap_difference(
    units: Sequence[T],
    stat_a: Callable[[Sequence[T]], float],
    stat_b: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Interval for ``stat_a - stat_b`` on the SAME resamples.

    Resampling both systems together preserves the pairing, which is what makes
    a small consistent advantage detectable. Computing two separate intervals
    and checking whether they overlap is a different, much weaker test — and a
    common mistake.

    An interval excluding zero is a difference the data supports.
    """
    if not units:
        return Interval(0.0, 0.0, 0.0, alpha, "bootstrap-diff", 0)

    point = stat_a(units) - stat_b(units)
    rng = random.Random(seed)
    n = len(units)
    draws = []
    for _ in range(n_resamples):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        try:
            draws.append(stat_a(sample) - stat_b(sample))
        except (ZeroDivisionError, ValueError):
            continue

    if not draws:
        return Interval(point, point, point, alpha, "bootstrap-diff", n)

    draws.sort()
    lo = draws[max(0, int((alpha / 2) * len(draws)))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return Interval(point, lo, hi, alpha, "bootstrap-diff", n)


# --------------------------------------------------------------------------- #
# Paired significance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of McNemar's test on two systems over the same items."""

    b: int
    """Items system A got right and system B got wrong."""
    c: int
    """Items system B got right and system A got wrong."""
    both_right: int
    both_wrong: int
    p_value: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.b + self.c

    @property
    def n(self) -> int:
        return self.b + self.c + self.both_right + self.both_wrong

    def significant(self, alpha: float = DEFAULT_ALPHA) -> bool:
        return self.p_value < alpha

    def render(self, name_a: str = "A", name_b: str = "B",
               alpha: float = DEFAULT_ALPHA) -> str:
        verdict = "DIFFERENT" if self.significant(alpha) else "not distinguishable"
        lines = [
            f"  {name_a} right / {name_b} wrong   {self.b:>5}",
            f"  {name_b} right / {name_a} wrong   {self.c:>5}",
            f"  both right                        {self.both_right:>5}",
            f"  both wrong                        {self.both_wrong:>5}",
            f"  p = {self.p_value:.4g} ({self.method})  ->  {verdict} at alpha={alpha}",
        ]
        if self.n_discordant < 10:
            lines.append(
                f"  NOTE: only {self.n_discordant} discordant items. The test has almost "
                f"no power here; absence of significance means absence of evidence."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"b": self.b, "c": self.c, "both_right": self.both_right,
                "both_wrong": self.both_wrong, "p_value": self.p_value,
                "method": self.method, "n": self.n}


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p under H0: each discordant item is a coin flip."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """McNemar's test for two systems evaluated on the same items.

    Uses the EXACT binomial test when there are fewer than 25 discordant
    items, and the chi-square approximation with continuity correction above
    that. The exact test matters here: on a few hundred instances the number of
    items where the two systems disagree is often single digits, and the
    chi-square approximation is unreliable at that size.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"McNemar compares two systems on the SAME items, but got "
            f"{len(correct_a)} and {len(correct_b)} outcomes"
        )

    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if bb and not a)
    both_right = sum(1 for a, bb in zip(correct_a, correct_b) if a and bb)
    both_wrong = sum(1 for a, bb in zip(correct_a, correct_b) if not a and not bb)

    if b + c < 25:
        return McNemarResult(b, c, both_right, both_wrong,
                             _binom_two_sided(b, c), "exact binomial")

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Survival function of chi-square with 1 df, via the normal tail.
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return McNemarResult(b, c, both_right, both_wrong, min(1.0, p),
                         "chi-square, continuity-corrected")


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #


def holm_adjust(p_values: dict[str, float], *, alpha: float = DEFAULT_ALPHA
                ) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni correction. Returns {name: (adjusted_p, reject)}.

    The ablation suite runs a dozen comparisons. At alpha=0.05, one in twenty
    null comparisons clears the bar by chance, so reporting the one that did is
    how a null result becomes a finding. Holm controls the family-wise error
    rate and is uniformly more powerful than plain Bonferroni, with no extra
    assumptions.
    """
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    still_rejecting = True

    for i, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - i) * p))
        running = adjusted
        if adjusted >= alpha:
            still_rejecting = False
        out[name] = (adjusted, still_rejecting and adjusted < alpha)
    return out


# --------------------------------------------------------------------------- #
# Reporting helper
# --------------------------------------------------------------------------- #


def render_intervals(rows: dict[str, Interval], *, title: str = "",
                     pct: bool = True) -> str:
    """Table of point estimates with intervals, widest last.

    Sorted by width so the least certain number is the last thing read, which
    is usually the one that should temper the conclusion.
    """
    if not rows:
        return f"{title}\n  (nothing to report)" if title else "  (nothing to report)"

    width = max(len(k) for k in rows) + 2
    lines = []
    if title:
        lines += [title, "-" * max(58, len(title))]
    for name, iv in sorted(rows.items(), key=lambda kv: kv[1].width):
        lines.append(f"  {name:<{width}} {iv.render(pct=pct):>28}   n={iv.n}")

    widest = max(rows.values(), key=lambda iv: iv.width)
    if widest.width > 0.25:
        lines.append("")
        lines.append(f"  The widest interval spans {widest.width:.0%}. At this sample "
                     f"size that number supports very little.")
    return "\n".join(lines)
