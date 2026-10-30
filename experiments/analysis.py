"""The confirmatory analysis: per-instance CCA, paired comparisons, Holm, and honest power.

Everything the pre-registration promises is computed here and nowhere else:

* **Per-instance CCA-all** from a cell's answers, scored by the pinned reader and
  the local NLI model (``metrics.case_accuracy``).
* **Paired comparisons** between two systems on the same instances: exact McNemar
  (``force_exact=True``, always -- the approximation is not the pre-registered
  test), a BCa interval for the accuracy difference resampling *instances*, and
  an instance-level sign-flip test as the distribution-free sensitivity check.
* **Holm** across the confirmatory family.
* **Power as achieved**, not as planned: the observed discordance and the minimum
  detectable difference at that discordance and this n. A comparison that cannot
  detect a plausible effect is labelled as underpowered next to its p-value, so
  "not significant" is never read as "no difference".
* **The narrowing rule**: if S1 is below the floor every confirmatory result is
  labelled *exploratory*, by the data, not by the prose.

Direction is part of each hypothesis. H1 predicts B3 is *worse* than B0, so a
significant difference in the other direction is reported as such and does not
support H1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Mapping, Sequence

from contract.gold import GoldInstance
from metrics.case_accuracy import (
    DEFAULT_NLI_THRESHOLD,
    InstanceCCA,
    NLIScorerLike,
    Reader,
    score_instance,
)
from metrics.cluster import bca_difference, sign_flip_test
from metrics.power_sim import HOLM_FIRST_STEP_ALPHA, mdd, observed_discordance
from metrics.stats import holm_adjust, mcnemar
from metrics.values import values_match

ALPHA = 0.05


def cca_by_instance(
    instances: Sequence[GoldInstance],
    answers: Mapping[str, str],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
) -> dict[str, InstanceCCA]:
    """CCA for every instance that has personas. A missing answer is an error, not a zero."""
    out: dict[str, InstanceCCA] = {}
    for inst in sorted(instances, key=lambda i: i.instance_id):
        if not inst.personas:
            continue
        if inst.instance_id not in answers:
            raise KeyError(f"no answer for {inst.instance_id}; a missing answer is not a wrong one")
        out[inst.instance_id] = score_instance(inst, answers[inst.instance_id], reader, nli, threshold=threshold)
    return out


def persona_setting_accuracy(instances: Sequence[GoldInstance], answers: Mapping[str, str]) -> dict[str, bool]:
    """Persona setting: one answer per persona, scored deterministically against its gold outcome.

    No reader and no judge: the answer to "does <persona> pay a fee?" either states the persona's gold
    outcome or it does not. Items are keyed ``instance_id|persona_id``.
    """
    out: dict[str, bool] = {}
    for inst in instances:
        for p in inst.personas:
            key = f"{inst.instance_id}|{p.persona_id}"
            if key not in answers:
                raise KeyError(f"no answer for {key}; a missing answer is not a wrong one")
            out[key] = values_match(inst.branch_for_persona(p.persona_id).outcome, answers[key])
    return out


@dataclass(frozen=True)
class Paired:
    name_a: str
    name_b: str
    n: int
    acc_a: float
    acc_b: float
    diff: float
    b: int                 # a right, b wrong
    c: int                 # b right, a wrong
    p_exact: float
    ci_low: float
    ci_high: float
    sign_flip_p: float
    discordance: float
    mdd_points: float | None
    """Smallest true difference (points) this n and discordance detect with 80% power, at the Holm alpha."""

    @property
    def underpowered_for(self) -> str:
        if self.mdd_points is None:
            return "no effect inside the observed discordance is detectable at this n"
        return f"cannot detect a true difference smaller than about {self.mdd_points:.1f} points"

    def render(self) -> str:
        return (f"{self.name_a} {self.acc_a:.3f} vs {self.name_b} {self.acc_b:.3f}  "
                f"diff {self.diff:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}]  "
                f"exact McNemar p={self.p_exact:.4f}  sign-flip p={self.sign_flip_p:.4f}  "
                f"(n={self.n}, discordant {self.b + self.c}; {self.underpowered_for})")


def paired(a: Mapping[str, bool], b: Mapping[str, bool], name_a: str = "A", name_b: str = "B",
           *, seed: int = 0, n_resamples: int = 4000) -> Paired:
    """Paired comparison on the instances both systems were scored on."""
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("the two systems share no scored instances")
    if set(a) != set(b):
        raise ValueError(f"the systems were scored on different instances ({len(set(a) ^ set(b))} differ): "
                         "a paired comparison needs identical items")
    xa, xb = [bool(a[i]) for i in ids], [bool(b[i]) for i in ids]
    mc = mcnemar(xa, xb, force_exact=True)
    units = list(range(len(ids)))
    iv = bca_difference(units, lambda u: mean(xa[i] for i in u), lambda u: mean(xb[i] for i in u),
                        n_resamples=n_resamples, seed=seed)
    diffs = [float(p) - float(q) for p, q in zip(xa, xb)]
    d = observed_discordance(xa, xb)
    return Paired(name_a, name_b, len(ids), mean(xa), mean(xb), mean(xa) - mean(xb), mc.b, mc.c, mc.p_value,
                  iv.low, iv.high, sign_flip_test(diffs, seed=seed), d,
                  mdd(len(ids), HOLM_FIRST_STEP_ALPHA, d) if d > 0 else None)


@dataclass
class Confirmatory:
    results: dict[str, Paired] = field(default_factory=dict)
    adjusted: dict[str, tuple[float, bool]] = field(default_factory=dict)
    directions_supported: dict[str, bool] = field(default_factory=dict)
    exploratory: bool = False
    exploratory_reason: str = ""

    def supported(self, h: str) -> bool:
        """Significant after Holm AND in the predicted direction AND the family is confirmatory."""
        return (not self.exploratory and self.adjusted[h][1] and self.directions_supported[h])

    def render(self) -> str:
        lines = ["Confirmatory family (S1, exact McNemar on CCA-all strict, Holm)", "=" * 72]
        if self.exploratory:
            lines += [f"  *** EXPLORATORY: {self.exploratory_reason} ***", ""]
        for h, r in self.results.items():
            p_adj, rej = self.adjusted[h]
            direction = "predicted direction" if self.directions_supported[h] else "OPPOSITE to prediction"
            lines += [f"  {h}: {r.render()}",
                      f"      Holm-adjusted p={p_adj:.4f}  reject={rej}  ({direction})"
                      f"  -> {'SUPPORTED' if self.supported(h) else 'not supported'}"]
        return "\n".join(lines)


def confirmatory(
    cca: Mapping[str, Mapping[str, bool]],
    *,
    comparator: str,
    s1_test: int,
    s1_floor: int = 60,
    seed: int = 0,
    h2_p: float | None = None,
) -> Confirmatory:
    """H1 (B3 worse than B0) and H3 (O3 better than the pre-registered comparator), Holm-adjusted.

    ``cca`` maps system id -> {instance id -> CCA-all strict}, all on the same S1 instances.
    ``h2_p`` is H2's p-value when the detection analysis has produced one, so the family is adjusted
    as pre-registered; without it the family is H1 and H3 and says so.
    """
    for need in ("B0", "B3", "O3", comparator):
        if need not in cca:
            raise KeyError(f"missing system {need!r} for the confirmatory family")
    out = Confirmatory()
    out.results["H1"] = paired(cca["B3"], cca["B0"], "B3", "B0", seed=seed)
    out.results["H3"] = paired(cca["O3"], cca[comparator], "O3", comparator, seed=seed)
    out.directions_supported = {"H1": out.results["H1"].diff < 0, "H3": out.results["H3"].diff > 0}
    ps = {h: r.p_exact for h, r in out.results.items()}
    if h2_p is not None:
        ps["H2"] = h2_p
        out.directions_supported["H2"] = True      # H2's direction is part of its own test
    out.adjusted = holm_adjust(ps, alpha=ALPHA)
    if s1_test < s1_floor:
        out.exploratory = True
        out.exploratory_reason = (f"S1 has {s1_test} test instances, below the pre-registered floor of "
                                  f"{s1_floor}: the narrowing rule makes every cross-document claim exploratory")
    return out


def bootstrap_p_at_most(units: Sequence, statistic, threshold: float, *, n_resamples: int = 4000,
                        seed: int = 0) -> float:
    """One-sided bootstrap p for ``H0: statistic <= threshold``: the share of resamples at or below it.

    Used for H2 (F1 difference against the 15-point minimum effect of interest). Resamples instances.
    """
    if not units:
        raise ValueError("no units")
    rng = random.Random(seed)
    n = len(units)
    hits = 0
    for _ in range(n_resamples):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        try:
            hits += statistic(sample) <= threshold
        except (ZeroDivisionError, ValueError):
            hits += 1          # an undefined statistic is not evidence for the hypothesis
    return (hits + 1) / (n_resamples + 1)
