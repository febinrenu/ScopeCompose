"""Exact power of the paired McNemar test, and the minimum detectable difference.

Every confirmatory comparison in this paper is paired and binary per instance
(did system A / system B get the instance right?), tested with the exact
McNemar test. How large a difference can n instances detect? That is a property
of the *design*, so it is computed before any test answer exists and written
into the pre-registration -- not reverse-engineered from a result.

**Exact, not simulated.** Conditional on the number of discordant instances
``m ~ Binomial(n, d)`` (``d`` is the discordance rate), the count where A wins
is ``Binomial(m, p_b / d)`` and the test rejects when the two-sided binomial
p-value is at most alpha. Summing over ``m`` gives power with no sampling
noise, so the same inputs return the same number forever. A seeded Monte Carlo
version (:func:`simulate_power`) exists only to cross-check this one.

``d`` matters as much as ``n``. Two systems that mostly agree (small ``d``)
need a larger true difference to separate than two that disagree often. The
proposal reports two levels (0.15, 0.35); :func:`observed_discordance` supplies
the real one from dev data before the freeze.

**Relation to the proposal's table.** Proposal v8 section 6.7 quotes figures
from a normal approximation. The exact values differ from them by at most one
point, in either direction (n = 60, d = 0.15: 15 against the proposal's 16;
n = 60, d = 0.35: 25 against 24). The exact values are the ones to report, and
the confirmatory test must be the exact one too (``mcnemar(..., force_exact=True)``).
"""

from __future__ import annotations

import argparse
import random
from math import comb
from typing import Sequence

#: Holm over three confirmatory hypotheses: the strictest step is alpha / 3.
HOLM_FIRST_STEP_ALPHA = 0.05 / 3


def _two_sided_binom_p(b: int, m: int) -> float:
    """Exact two-sided p-value of ``b`` successes in ``m`` trials under p = 0.5."""
    if m == 0:
        return 1.0
    k = min(b, m - b)
    tail = sum(comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)


def mcnemar_power(n: int, alpha: float, d: float, delta: float) -> float:
    """Power of the exact McNemar test.

    ``n`` instances; ``d`` = P(discordant); ``delta`` = true accuracy difference
    (A minus B), so P(A right, B wrong) = (d + delta) / 2 and the reverse is
    (d - delta) / 2. Requires ``0 < d <= 1`` and ``|delta| <= d``.
    """
    if not 0 < d <= 1:
        raise ValueError(f"discordance d must be in (0, 1], got {d}")
    if abs(delta) > d + 1e-12:
        raise ValueError(f"|delta|={abs(delta)} cannot exceed the discordance rate d={d}")
    q = (d + delta) / (2 * d)               # P(A wins | discordant)
    total = 0.0
    for m in range(n + 1):
        pm = comb(n, m) * d**m * (1 - d) ** (n - m)
        if pm < 1e-15:
            continue
        reject = 0.0
        for b in range(m + 1):
            if _two_sided_binom_p(b, m) <= alpha:
                reject += comb(m, b) * q**b * (1 - q) ** (m - b)
        total += pm * reject
    return total


def mdd(n: int, alpha: float, d: float, *, power: float = 0.8, step: float = 0.005) -> float | None:
    """Smallest true difference (in points, 0-100) detected with the given power.

    Scans upward in ``step`` and returns the first difference that reaches the
    target; discreteness makes power slightly non-monotone, so the *first*
    crossing is reported, which is the optimistic reading and is labelled so.
    ``None`` if even ``delta = d`` is not enough.
    """
    delta = 0.0
    while delta + step <= d + 1e-12:
        delta += step
        if mcnemar_power(n, alpha, d, delta) >= power:
            return round(delta * 100, 1)
    return None


def simulate_power(n: int, alpha: float, d: float, delta: float, *, reps: int = 4000, seed: int = 0) -> float:
    """Monte Carlo power, for cross-checking :func:`mcnemar_power` only."""
    rng = random.Random(seed)
    pb, pc = (d + delta) / 2, (d - delta) / 2
    hits = 0
    for _ in range(reps):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            if u < pb:
                b += 1
            elif u < pb + pc:
                c += 1
        if _two_sided_binom_p(b, b + c) <= alpha:
            hits += 1
    return hits / reps


def observed_discordance(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Share of instances on which the two systems disagree. Feeds ``d``."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired outcomes must have equal length")
    if not correct_a:
        raise ValueError("no instances")
    return sum(a != b for a, b in zip(correct_a, correct_b)) / len(correct_a)


#: The three columns of the proposal's table (section 6.7).
SCENARIOS = (("Holm alpha, d=0.15", HOLM_FIRST_STEP_ALPHA, 0.15),
             ("Holm alpha, d=0.35", HOLM_FIRST_STEP_ALPHA, 0.35),
             ("alpha=0.05, d=0.25", 0.05, 0.25))


def mdd_table(ns: Sequence[int] = (60, 80, 140, 220)) -> dict[int, dict[str, float | None]]:
    return {n: {label: mdd(n, a, d) for label, a, d in SCENARIOS} for n in ns}


def render_table(ns: Sequence[int] = (60, 80, 140, 220)) -> str:
    rows = mdd_table(ns)
    head = f"  {'n':>5}" + "".join(f"{label:>22}" for label, _, _ in SCENARIOS)
    lines = ["Minimum detectable difference, points (exact McNemar, power 0.80)", "=" * len(head), head,
             "  " + "-" * (len(head) - 2)]
    for n, r in rows.items():
        lines.append(f"  {n:>5}" + "".join(f"{(r[label] if r[label] is not None else float('nan')):>22.1f}"
                                          for label, _, _ in SCENARIOS))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[60, 80, 140, 220])
    args = ap.parse_args(argv)
    print(render_table(args.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
