"""Preservation metrics -- PR / SR / HCR / SCR.

The signatures were frozen while this was a stub, so that both halves of the
pipeline could be wired together before either was implemented. They have not
changed since.

The four metrics, from the proposal:

Preservation Rate (PR)
    preserved branches / gold branches. The headline metric.

Suppression Rate (SR)
    suppressed branches / gold branches. The critical safety metric --
    expected to spike for selection-based methods on conditional instances.
    This is the number the whole project is built to expose: it measures a
    valid, separately-scoped branch of the truth being silently dropped, which
    answer-correctness cannot see.

Hallucinated-Condition Rate (HCR)
    conditions asserted with no passage support / instances. Guards against
    fabricating content. Its denominator includes candidates rejected by the
    grounding gate.

Spurious-Condition Rate (SCR)
    conditional structure imposed on instances that have none / no-conflict
    instances. Guards the opposite failure: a caveat-happy model that invents
    structure to game PR. This is why distractor instances are mandatory in
    the benchmark -- without them SCR has no denominator.

A judged branch is exactly one of Preserved, Suppressed, or Distorted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from api_budget.client import Tier
from contract.gold import Branch, GoldInstance
from contract.models import ConflictType
from metrics.branch_match import (
    BranchJudgement,
    align,
    containment,
    is_grounded,
    similarity,
)

# Imported, never redefined. An identical copy of this enum used to live
# here, so ``align()`` returned one class and the scorer compared against
# the other -- every ``j is BranchJudgement.PRESERVED`` was False and a
# perfectly preserved run scored PR = 0.0 with every branch counted as
# distorted. Two enums with the same members are not the same enum.


@dataclass
class PreservationScores:
    """The four headline metrics plus their raw counts."""

    gold_branches: int = 0
    preserved: int = 0
    suppressed: int = 0
    distorted: int = 0

    instances: int = 0
    hallucinated_conditions: int = 0

    no_conflict_instances: int = 0
    spurious_conditions: int = 0

    #: Per-tier breakdown. Tier 1 and Tier 2 are never blended in headline
    #: reporting, so the container carries the split rather than relying on the
    #: caller to remember.
    by_construction: dict[str, PreservationScores] = field(default_factory=dict)

    @property
    def preservation_rate(self) -> float:
        return self.preserved / self.gold_branches if self.gold_branches else 0.0

    @property
    def suppression_rate(self) -> float:
        return self.suppressed / self.gold_branches if self.gold_branches else 0.0

    @property
    def distortion_rate(self) -> float:
        return self.distorted / self.gold_branches if self.gold_branches else 0.0

    @property
    def hallucinated_condition_rate(self) -> float:
        return self.hallucinated_conditions / self.instances if self.instances else 0.0

    @property
    def spurious_condition_rate(self) -> float:
        n = self.no_conflict_instances
        return self.spurious_conditions / n if n else 0.0

    def as_dict(self) -> dict:
        return {
            "PR": round(self.preservation_rate, 4),
            "SR": round(self.suppression_rate, 4),
            "distortion_rate": round(self.distortion_rate, 4),
            "HCR": round(self.hallucinated_condition_rate, 4),
            "SCR": round(self.spurious_condition_rate, 4),
            "gold_branches": self.gold_branches,
            "instances": self.instances,
            "by_construction": {k: v.as_dict() for k, v in self.by_construction.items()},
        }

    def render(self) -> str:
        lines = [
            "Preservation metrics",
            "-" * 56,
            f"  PR  (preservation)  {self.preservation_rate:>8.4f}   <- headline",
            f"  SR  (suppression)   {self.suppression_rate:>8.4f}   <- safety metric",
            f"  HCR (hallucinated)  {self.hallucinated_condition_rate:>8.4f}",
            f"  SCR (spurious)      {self.spurious_condition_rate:>8.4f}",
            f"  gold branches       {self.gold_branches:>8,}",
        ]
        if self.by_construction:
            lines += ["", "  by tier (never blend these into one headline number):"]
            for tier, s in sorted(self.by_construction.items()):
                lines.append(
                    f"    {tier:<10} PR {s.preservation_rate:.4f}  "
                    f"SR {s.suppression_rate:.4f}  (n={s.gold_branches})"
                )
        return "\n".join(lines)



# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def judge_branches(
    system_output: str,
    gold_branches: list[Branch],
    *,
    predicted_branches: list[Branch] | None = None,
    nli=None,
    client=None,
    tier: Tier = Tier.JUDGE,
) -> list[BranchJudgement]:
    """Judge each gold branch as preserved, suppressed, or distorted.

    Two paths, and which one ran must travel with any number derived from it:

    **Structural** (``predicted_branches`` supplied). The system emitted a
    branch structure, so this is alignment rather than interpretation --
    deterministic, free, and reproducible forever. Used for the pipeline and
    the structured baseline.

    **Textual** (only ``system_output``). The system emitted prose, so each gold
    branch has to be looked for in it. Containment rather than symmetric
    overlap, because the answer is longer than any single branch by design.
    This is the weaker instrument and it is a LOWER BOUND on preservation: a
    branch paraphrased beyond lexical recognition reads as suppressed.

    The LLM judge that proposal 6.1 specifies is a third path and is not
    implemented here. It must be validated against human labels before anything
    it produces is reported -- see :func:`validate_judge` -- and on the
    zero-spend configuration it runs on open weights, which makes that
    validation load-bearing rather than a formality. Passing ``client`` raises
    rather than silently falling back, so a judged number cannot be produced by
    accident.
    """
    if client is not None:
        raise NotImplementedError(
            "the LLM judge path is not implemented. Use the structural path by "
            "passing predicted_branches, or the textual path. A judge must be "
            "validated against human labels before its numbers are reported "
            "(proposal 6.1), so it is deliberately not reachable by default."
        )

    if predicted_branches is not None:
        result = align(gold_branches, predicted_branches, nli=nli)
        return [a.judgement for a in result.alignments]

    out: list[BranchJudgement] = []
    for b in gold_branches:
        if containment(b.outcome, system_output) >= 0.6:
            out.append(BranchJudgement.PRESERVED)
        elif similarity(b.outcome, system_output) >= 0.25:
            # Present but altered. Distinguishing a distorted branch from a
            # suppressed one matters: one corrupts the answer, the other
            # deletes it, and only the second is what Suppression Rate counts.
            out.append(BranchJudgement.DISTORTED)
        else:
            out.append(BranchJudgement.SUPPRESSED)
    return out


def score_preservation(
    system_outputs: list,
    gold_instances: list[GoldInstance],
    *,
    nli=None,
) -> PreservationScores:
    """Compute PR / SR / HCR / SCR over a run.

    ``system_outputs`` may be :class:`~composition.operator.ComposedAnswer`
    objects, :class:`~generation.scoped_answer.GeneratedAnswer` objects, or
    plain strings. Branch structure is used when present, because it is the
    stronger instrument; prose falls back to textual judging.

    Results are broken out by ``construction``. A blended PR across Tier 1 and
    Tier 2 would let constructed instances carry a claim about naturally
    occurring ones, which is the reporting rule the benchmark design rests on.
    """
    overall = PreservationScores()
    per_tier: dict[str, PreservationScores] = {}
    by_id = {g.instance_id: g for g in gold_instances}

    for out in system_outputs:
        gid = getattr(out, "query_id", None)
        gold = by_id.get(gid) if gid else None
        if gold is None:
            continue

        bucket = per_tier.setdefault(gold.construction.value, PreservationScores())
        text, predicted = _unpack(out)
        judgements = judge_branches(
            text, gold.gold_branches, predicted_branches=predicted, nli=nli)

        known = {p.id for p in gold.passages}
        hallucinated = _count_hallucinated(predicted, known)
        unconditional = _has_no_conditional_structure(gold)
        spurious = _count_spurious(gold, predicted, text)

        for scores in (overall, bucket):
            scores.instances += 1
            scores.gold_branches += len(gold.gold_branches)
            for j in judgements:
                if j is BranchJudgement.PRESERVED:
                    scores.preserved += 1
                elif j is BranchJudgement.SUPPRESSED:
                    scores.suppressed += 1
                else:
                    scores.distorted += 1
            if hallucinated:
                scores.hallucinated_conditions += 1
            # SCR's denominator is the instances with no conditional structure
            # to find. Distractors are mandatory in the benchmark for exactly
            # this reason: without them SCR is 0/0, and a caveat-happy system
            # looks indistinguishable from a careful one.
            if unconditional:
                scores.no_conflict_instances += 1
                if spurious:
                    scores.spurious_conditions += 1

    overall.by_construction = per_tier
    return overall


def _unpack(out) -> tuple[str, list[Branch] | None]:
    """Normalise a system output into ``(text, branches or None)``."""
    if isinstance(out, str):
        return out, None
    text = getattr(out, "text", "") or ""
    branches = getattr(out, "branches", None)
    if branches is None:
        # A GeneratedAnswer carries prose; the structure sits on the
        # ComposedAnswer it came from, when the caller attached it.
        inner = getattr(out, "composed", None)
        branches = getattr(inner, "branches", None) if inner is not None else None
    return text, branches


def _count_hallucinated(predicted: list[Branch] | None, known: set[str]) -> int:
    """Branches citing a passage that does not exist, or citing nothing.

    An ungrounded branch is a fabrication. A branch grounded in a real passage
    but absent from gold is a different thing -- possibly an annotation gap --
    and is deliberately not counted, because counting it would penalise a
    system for being right about something the annotator missed.
    """
    if not predicted:
        return 0
    return sum(1 for b in predicted if not is_grounded(b, known))


def _has_no_conditional_structure(gold: GoldInstance) -> bool:
    return (gold.gold_conflict_type is not ConflictType.CONDITIONAL
            or gold.is_distractor)


def _count_spurious(gold: GoldInstance, predicted: list[Branch] | None,
                    text: str) -> int:
    """Conditional structure imposed where gold has none.

    The opposite failure to suppression, and the reason it needs its own
    metric: a system hedging everything with invented conditions scores well on
    PR while being useless, so PR alone can be gamed by caveating.
    """
    if not _has_no_conditional_structure(gold):
        return 0
    if predicted is not None:
        return sum(1 for b in predicted if not b.is_default)
    hedges = ("unless", "except", "only if", "does not apply to", "provided that")
    low = (text or "").lower()
    return sum(1 for h in hedges if h in low)


def validate_judge(
    llm_judgements: dict[str, list[BranchJudgement]],
    human_judgements: dict[str, list[BranchJudgement]],
) -> dict[str, float]:
    """Compare a judge against human labels.

    Proposal 6.1 requires this before any judged number is reported, and
    Cattan et al. report 0.89 as the precedent bar. Reports per-branch
    agreement **and** the correlation between headline rates computed each way,
    because they answer different questions. A judge can agree on 85% of
    branches and still mis-rank two systems if its errors concentrate in one of
    them -- and it is the ranking a paper's conclusion rests on.
    """
    shared = sorted(set(llm_judgements) & set(human_judgements))
    if not shared:
        raise ValueError("no instances judged by both; nothing to validate")

    agree = total = 0
    llm_pr: list[float] = []
    human_pr: list[float] = []
    for iid in shared:
        a, b = llm_judgements[iid], human_judgements[iid]
        if len(a) != len(b):
            raise ValueError(
                f"{iid}: {len(a)} judgements against {len(b)}. The two must "
                "cover the same gold branches in the same order, or the "
                "comparison aligns different things."
            )
        agree += sum(1 for x, y in zip(a, b) if x is y)
        total += len(a)
        if a:
            llm_pr.append(sum(j is BranchJudgement.PRESERVED for j in a) / len(a))
            human_pr.append(sum(j is BranchJudgement.PRESERVED for j in b) / len(b))

    out = {
        "branch_agreement": agree / total if total else 0.0,
        "n_branches": float(total),
        "n_instances": float(len(shared)),
        "pearson_pr": _pearson(llm_pr, human_pr),
        "spearman_pr": _spearman(llm_pr, human_pr),
    }
    out["meets_precedent_bar"] = float(out["branch_agreement"] >= 0.89)
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def _rank(values: list[float]) -> list[float]:
    """Average ranks, ties shared. Needed for Spearman on PR values, which tie
    constantly -- most instances have two branches, so PR is one of 0, 0.5, 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_rank(xs), _rank(ys))
