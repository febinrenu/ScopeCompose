"""How large does the corpus need to be?

Every comparison this project has run so far has been underpowered, and said
so: the decisive experiment reported −12.5% [−37.5%, +12.5%] on 24 branches
with 9 discordant, and the κ pilots repeatedly cleared a threshold on
intervals running to 0.000. "Not distinguishable at this size" is an honest
report and it is not a result.

This module answers the question that fixes it: **how many instances before
the experiment can see the effect it is looking for?** Deciding that in
advance is the difference between a corpus sized by argument and one sized by
when the annotators got tired — and it is the first thing a reviewer asks when
a headline interval spans zero.

Two regimes, because the project's two headline comparisons have different
shapes:

**Paired, per branch.** The decisive experiment scores both systems on the
same gold branches, so the test is McNemar and only *discordant* branches
carry information. Branches where both systems agree contribute nothing, which
is why 24 branches gave 9 usable ones.

**Unpaired proportion.** Preservation Rate against a fixed target, or two
independent corpora.

Both use normal approximations. They are planning tools, deliberately rough,
and a plan accurate to ±10 instances is worth far more than no plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Conventional defaults. Stated as constants so a later run cannot quietly
#: relax them to make a target look reachable.
ALPHA = 0.05
POWER = 0.80


def _z(p: float) -> float:
    """Inverse normal CDF, Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability out of range: {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass(frozen=True)
class PowerResult:
    """What a given size can and cannot detect."""

    branches_needed: int
    instances_needed: int
    discordant_needed: int
    effect: float
    discordant_rate: float
    branches_per_instance: float

    def render(self) -> str:
        return "\n".join([
            f"  to detect a {self.effect:.0%} difference in preservation",
            f"    discordant branches needed   {self.discordant_needed:>6}",
            f"    gold branches needed         {self.branches_needed:>6}"
            f"   (at {self.discordant_rate:.0%} discordant)",
            f"    INSTANCES needed             {self.instances_needed:>6}"
            f"   (at {self.branches_per_instance:.1f} branches each)",
        ])


def paired_size(
    effect: float,
    *,
    discordant_rate: float = 0.35,
    branches_per_instance: float = 2.0,
    alpha: float = ALPHA,
    power: float = POWER,
) -> PowerResult:
    """Corpus size for a McNemar comparison of two systems.

    ``effect`` is the difference in preservation rate worth detecting -- 0.10
    means ten percentage points.

    ``discordant_rate`` is the share of gold branches on which the two systems
    differ, and it is the parameter that actually drives the answer. Only
    discordant branches inform McNemar: the decisive experiment's 24 branches
    yielded 9, a rate of 0.375, and the 0.35 default comes from that
    observation rather than from convention.

    Under McNemar the discordant branches split b/c, and detecting a
    difference means detecting a departure from 50/50 among them. A difference
    of ``effect`` over all branches implies a proportion of
    ``0.5 + effect/(2 * discordant_rate)`` among the discordant ones.

    **That relationship runs the opposite way to the obvious intuition, and
    the intuition is worth naming because it is the one to resist.** "Only
    discordant branches inform the test, so fewer of them must be worse" is
    wrong. The effect being sized is a difference over *all* branches, so a
    low discordant rate means it is concentrated in a small subset -- easier
    to see, not harder. At a 10-point effect, a 20% discordant rate implies a
    75/25 split among those branches and needs 145 of them; a 50% rate implies
    only 60/40 and needs 388.

    For corpus planning: two systems that mostly agree but differ sharply
    where they differ are **cheaper** to distinguish than two that disagree
    everywhere by a little.

    What a low rate does cost is robustness of the estimate. A greater share
    of the corpus has to be discordant for the figure to hold, so observing a
    rate well below the assumed one invalidates the plan rather than merely
    loosening it.
    """
    if not 0 < effect < 1:
        raise ValueError("effect must be a proportion between 0 and 1")
    if not 0 < discordant_rate <= 1:
        raise ValueError("discordant_rate must be in (0, 1]")

    p = 0.5 + effect / (2 * discordant_rate)
    if p >= 1.0:
        # The effect is larger than the discordant branches can express: every
        # disagreement would have to favour one system and that still would
        # not reach it. Not a sample-size problem.
        return PowerResult(0, 0, 0, effect, discordant_rate, branches_per_instance)

    za, zb = _z(1 - alpha / 2), _z(power)
    n_disc = math.ceil(((za * 0.5 + zb * math.sqrt(p * (1 - p))) ** 2)
                       / ((p - 0.5) ** 2))
    n_branch = math.ceil(n_disc / discordant_rate)
    n_inst = math.ceil(n_branch / max(branches_per_instance, 1e-9))
    return PowerResult(n_branch, n_inst, n_disc, effect, discordant_rate,
                       branches_per_instance)


def proportion_size(
    baseline: float,
    effect: float,
    *,
    alpha: float = ALPHA,
    power: float = POWER,
) -> int:
    """Branches needed to tell ``baseline`` from ``baseline + effect``.

    Two independent groups, so no pairing to exploit -- always larger than the
    McNemar answer, and the reason the decisive experiment is designed paired.
    """
    p1, p2 = baseline, baseline + effect
    if not 0 < p1 < 1 or not 0 < p2 < 1:
        raise ValueError("baseline and baseline+effect must be in (0, 1)")
    pbar = (p1 + p2) / 2
    za, zb = _z(1 - alpha / 2), _z(power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (effect ** 2))


def detectable_effect(
    n_branches: int,
    *,
    discordant_rate: float = 0.35,
    alpha: float = ALPHA,
    power: float = POWER,
) -> float:
    """The smallest difference a corpus of this size can see.

    The inverse question, and the more useful one once a corpus exists: not
    "how many do I need" but "what can I already conclude, and what can I not".
    """
    # Bounded above by the discordant rate, not by 1. An effect at or beyond
    # it cannot be expressed under this model -- every disagreement would have
    # to favour one system and still fall short -- and `paired_size` returns 0
    # there. Searching past that point made the bisection read "needs no
    # branches" as "needs more", and it walked to 0.999: at 83 branches the
    # function claimed the smallest detectable difference was 100%, when the
    # table directly above it showed 20% needing only 63.
    lo, hi = 1e-4, discordant_rate * 0.999
    if paired_size(hi, discordant_rate=discordant_rate, alpha=alpha,
                   power=power).branches_needed > n_branches:
        return hi  # not even the largest expressible effect is detectable

    for _ in range(60):
        mid = (lo + hi) / 2
        need = paired_size(mid, discordant_rate=discordant_rate,
                           alpha=alpha, power=power).branches_needed
        if need == 0 or need > n_branches:
            lo = mid
        else:
            hi = mid
    return hi


def render_plan(
    *,
    current_branches: int,
    current_instances: int,
    discordant_rate: float = 0.35,
) -> str:
    """A sizing table for the corpus plan and the paper's methods section."""
    bpi = (current_branches / current_instances) if current_instances else 2.0
    lines = [
        "Statistical power",
        "=" * 70,
        f"  corpus now: {current_instances} instances, {current_branches} gold "
        f"branches ({bpi:.1f} each)",
        f"  assumed discordant rate: {discordant_rate:.0%}"
        "   (observed 9/24 = 37.5% in the decisive run)",
        "",
    ]
    for effect in (0.20, 0.15, 0.10, 0.05):
        lines.append(paired_size(effect, discordant_rate=discordant_rate,
                                 branches_per_instance=bpi).render())
        lines.append("")

    if current_branches:
        mde = detectable_effect(current_branches, discordant_rate=discordant_rate)
        lines += [
            "-" * 70,
            f"  At {current_branches} branches the smallest detectable difference is "
            f"about {mde:.0%}.",
            "",
            "  Anything smaller than that will return an interval spanning zero",
            "  however the experiment is run, and reporting such a null as",
            "  evidence of equivalence would be the error the decisive",
            "  experiment's own output already warns about.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="corpus JSONL; sizes the plan against what exists")
    ap.add_argument("--discordant-rate", type=float, default=0.35)
    args = ap.parse_args(argv)

    branches = instances = 0
    if args.data and args.data.exists():
        for line in args.data.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("gold_branches"):
                instances += 1
                branches += len(obj["gold_branches"])

    print(render_plan(current_branches=branches, current_instances=instances,
                      discordant_rate=args.discordant_rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
