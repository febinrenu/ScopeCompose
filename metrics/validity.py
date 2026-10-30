"""The validity battery: can this metric be gamed? (proposal v8 sections 3.5, 7.5)

A metric earns the right to carry a paper's headline only if the cheapest ways
of cheating it score badly. This module generates those cheats -- eight outputs
per instance -- and checks, against criteria **fixed before the first run**,
that each metric treats them the way a valid metric must.

The eight outputs
-----------------

======================  ====================================================
``gold``                the human-written scoped answer
``selection``           one unconditioned outcome (what a selection system says)
``concat_gold``         the passages that ground the gold branches, pasted
``concat_k``            every retrieved passage, pasted
``padded_gold``         gold plus neutral filler: length with no content
``unattached_hedge``    every outcome listed, hedged, attached to nobody
``modality_hedge``      each outcome attached to a group, but only as "may"
``null``                a refusal that says nothing
======================  ====================================================

``concat_*`` is the one that broke the old metric: containment-style Preservation
Rate scored verbatim concatenation (0.855) above the human gold answer (0.699),
because it has no precision side. ``unattached_hedge`` and ``modality_hedge`` are
what a coverage metric credits and a case-complete one must not: the answer
*mentions* every outcome without telling anyone which applies to them.

The V1 criteria
---------------

Each metric has its own, because each is gamed differently:

* **legacy PR** -- concatenation >= gold. This is a *replication* of the pilot,
  stated as a criterion so that "PR is gameable" is a measured result, not an
  assertion.
* **CCA-all (strict)** -- gold >= concatenation; unattached hedge <= 0.05;
  modality hedge <= 0.10; null <= 0.05.
* **scoped precision** -- gold > concatenation. Not implemented in this module
  and reported as such: a criterion that cannot be evaluated is never silently
  passed.

The numbers below are constants so the pre-registration can hash them. Editing
a threshold after seeing a result is the failure this file exists to prevent;
the only legitimate reason to change one is a bug in how it is *computed*,
and that goes in ``docs/v7_decisions.md`` with the reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from contract.gold import GoldInstance
from metrics.branch_match import BranchJudgement
from metrics.case_accuracy import (
    CCAScores,
    InstanceCCA,
    NLIScorerLike,
    Reader,
    aggregate,
    score_instance,
)
from metrics.preservation import judge_branches

OUTPUTS = (
    "gold", "selection", "concat_gold", "concat_k", "padded_gold",
    "unattached_hedge", "modality_hedge", "null",
)

#: Neutral filler: adds length and states nothing about any outcome. Free of
#: modal and hedge words on purpose, so the padded output isolates *length*.
FILLER = (
    "This summary is provided for general information only. The full terms are "
    "available from the provider on request. Please keep a copy of your agreement "
    "for your records. Contact the provider directly if you have a question about "
    "your own account."
)

NULL_ANSWER = "I'm sorry, I don't have enough information to answer that question."


def battery_outputs(inst: GoldInstance) -> dict[str, str]:
    """The eight outputs for one instance. Deterministic: same instance, same text."""
    if not inst.gold_scoped_answer or not inst.selection_answer:
        raise ValueError(
            f"{inst.instance_id}: the battery needs both gold_scoped_answer and selection_answer"
        )
    by_id = {p.id: p for p in inst.passages}
    grounding = [by_id[b.supporting_passage] for b in inst.gold_branches
                 if b.supporting_passage in by_id]
    seen: set[str] = set()
    grounding = [p for p in grounding if not (p.id in seen or seen.add(p.id))]

    outcomes = [b.outcome for b in inst.gold_branches]
    default = inst.default_branch
    exceptions = inst.exception_branches

    unattached = (
        "The outcome depends on your circumstances. In some cases "
        + ", and in other cases ".join(outcomes) + "."
    )

    if default is not None and exceptions:
        modality = f"{default.outcome[0].upper()}{default.outcome[1:]}. " + " ".join(
            f"{(b.applicability.descriptor or 'Some people')[0].upper()}"
            f"{(b.applicability.descriptor or 'Some people')[1:]} may be eligible for the following: "
            f"{b.outcome}."
            for b in exceptions
        )
    else:
        modality = "People in some situations may be eligible for the following: " + "; ".join(outcomes) + "."

    return {
        "gold": inst.gold_scoped_answer,
        "selection": inst.selection_answer,
        "concat_gold": " ".join(p.text for p in grounding),
        "concat_k": " ".join(p.text for p in inst.passages),
        "padded_gold": f"{inst.gold_scoped_answer} {FILLER}",
        "unattached_hedge": unattached,
        "modality_hedge": modality,
        "null": NULL_ANSWER,
    }


# --------------------------------------------------------------------------- #
# The criteria -- fixed before the first run
# --------------------------------------------------------------------------- #

CCA_UNATTACHED_HEDGE_MAX = 0.05
CCA_MODALITY_HEDGE_MAX = 0.10
CCA_NULL_MAX = 0.05

V1_CRITERIA: dict[str, str] = {
    "legacy_pr/concat_gold>=gold": "legacy containment PR of concat_gold >= PR of gold (replicates the pilot)",
    "legacy_pr/concat_k>=gold": "legacy containment PR of concat_k >= PR of gold (replicates the pilot)",
    "cca/gold>=concat_gold": "CCA-all strict of gold >= concat_gold",
    "cca/gold>=concat_k": "CCA-all strict of gold >= concat_k",
    f"cca/unattached_hedge<={CCA_UNATTACHED_HEDGE_MAX}": "CCA-all strict of unattached_hedge",
    f"cca/modality_hedge<={CCA_MODALITY_HEDGE_MAX}": "CCA-all strict of modality_hedge",
    f"cca/null<={CCA_NULL_MAX}": "CCA-all strict of null",
    "scoped_precision/gold>concat": "NOT EVALUATED: scoped precision is not implemented in this module",
}

#: Hashed into the pre-registration. Changing a criterion changes this.
V1_CRITERIA_SHA = hashlib.sha256(
    json.dumps({"criteria": V1_CRITERIA, "outputs": OUTPUTS}, sort_keys=True).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    passed: bool | None
    """``None`` means *not evaluated*, which is never the same as passed."""
    detail: str = ""


@dataclass
class BatteryResult:
    n_instances: int
    cca: dict[str, CCAScores] = field(default_factory=dict)
    legacy_pr: dict[str, float] = field(default_factory=dict)
    per_instance: dict[str, list[InstanceCCA]] = field(default_factory=dict)
    nli_threshold: float = 0.5
    nli_threshold_tuned: bool = False

    def criteria(self) -> list[Criterion]:
        out: list[Criterion] = []
        pr, cca = self.legacy_pr, {k: v.cca_all.point for k, v in self.cca.items()}

        for name, desc in V1_CRITERIA.items():
            kind, rest = name.split("/", 1)
            if kind == "scoped_precision":
                out.append(Criterion(name, desc, None, "scoped precision not implemented"))
            elif kind == "legacy_pr":
                other = rest.split(">=")[0]
                ok = pr[other] >= pr["gold"]
                out.append(Criterion(name, desc, ok, f"{other}={pr[other]:.3f} vs gold={pr['gold']:.3f}"))
            elif rest.startswith("gold>="):
                other = rest.split(">=")[1]
                ok = cca["gold"] >= cca[other]
                out.append(Criterion(name, desc, ok, f"gold={cca['gold']:.3f} vs {other}={cca[other]:.3f}"))
            else:
                key, bound = rest.split("<=")
                ok = cca[key] <= float(bound)
                out.append(Criterion(name, desc, ok, f"{key}={cca[key]:.3f} (max {bound})"))
        return out

    @property
    def v1_passed(self) -> bool:
        """All *evaluated* criteria hold. Unevaluated ones are reported, not counted."""
        return all(c.passed for c in self.criteria() if c.passed is not None)

    def render(self) -> str:
        lines = [f"Validity battery over {self.n_instances} instances"
                 + ("" if self.nli_threshold_tuned
                    else f"  (NLI threshold {self.nli_threshold} is UNTUNED)"),
                 "=" * 72,
                 f"  {'output':<18}{'legacy PR':>11}{'CCA strict':>13}{'CCA lenient':>13}",
                 "  " + "-" * 53]
        for o in OUTPUTS:
            c = self.cca.get(o)
            lines.append(f"  {o:<18}{self.legacy_pr.get(o, float('nan')):>11.3f}"
                         f"{(c.cca_all.point if c else float('nan')):>13.3f}"
                         f"{(c.cca_all_lenient.point if c else float('nan')):>13.3f}")
        lines += ["", "V1 criteria (fixed before the first run)", "-" * 72]
        for c in self.criteria():
            mark = "n/a " if c.passed is None else ("PASS" if c.passed else "FAIL")
            lines.append(f"  [{mark}] {c.name:<40} {c.detail}")
        lines.append("")
        lines.append(f"  V1: {'PASSED' if self.v1_passed else 'FAILED'} "
                     f"(unevaluated criteria are not counted as passes)")
        lines.append(f"  criteria sha256: {V1_CRITERIA_SHA[:16]}")
        return "\n".join(lines)


def legacy_pr(instances: list[GoldInstance], answers: dict[str, str]) -> float:
    """Pooled containment PR: preserved branches / gold branches, textual path."""
    preserved = total = 0
    for inst in instances:
        js = judge_branches(answers[inst.instance_id], inst.gold_branches)
        preserved += sum(j is BranchJudgement.PRESERVED for j in js)
        total += len(js)
    return preserved / total if total else 0.0


def run_battery(
    instances: list[GoldInstance],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = 0.5,
    threshold_tuned: bool = False,
) -> BatteryResult:
    """Generate the eight outputs for every instance and score them all.

    Every instance needs personas, branches and both reference answers; one that
    lacks them is an error, not a skip -- a battery over a silently thinned set
    would not be the battery the criteria were written for.
    """
    if not instances:
        raise ValueError("the validity battery needs at least one instance")
    outputs = {i.instance_id: battery_outputs(i) for i in instances}
    result = BatteryResult(n_instances=len(instances), nli_threshold=threshold,
                           nli_threshold_tuned=threshold_tuned)
    for name in OUTPUTS:
        answers = {iid: outs[name] for iid, outs in outputs.items()}
        per = [score_instance(i, answers[i.instance_id], reader, nli, threshold=threshold)
               for i in instances]
        result.per_instance[name] = per
        result.cca[name] = aggregate(per, threshold=threshold, threshold_tuned=threshold_tuned)
        result.legacy_pr[name] = legacy_pr(instances, answers)
    return result
