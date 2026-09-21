"""Preservation metrics -- PR / SR / HCR / SCR.

**Owned by Member B.** This file is a typed stub written by Member A so the
shared evaluation harness imports cleanly and both halves can be wired
together before B's implementation lands. The signatures and the definitions
are fixed here by agreement; the bodies are B's.

Member A: do not implement these. They are the core of B's separately assessed
research artifact.

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

from dataclasses import dataclass, field
from enum import Enum


class BranchJudgement(str, Enum):
    """How one gold branch fared in a system's output."""

    PRESERVED = "preserved"
    SUPPRESSED = "suppressed"
    DISTORTED = "distorted"


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
# Member B implements the functions below.
# --------------------------------------------------------------------------- #


class NotImplementedByMemberB(NotImplementedError):
    """Raised by a stub Member B has not filled in yet.

    A distinct type so the harness can report 'B's half is not wired up yet'
    rather than crashing with a bare NotImplementedError that looks like a bug.
    """


def judge_branches(system_output: str, gold_branches: list, **kwargs) -> list[BranchJudgement]:
    """Judge each gold branch as preserved, suppressed, or distorted.

    Member B: this is the LLM-as-judge entry point. The judge must be
    validated against human labels on the full conditional subset before any
    number it produces is trusted -- Cattan et al. report 0.89 accuracy as the
    precedent bar. On the zero-spend configuration this runs on open weights,
    so the validation result is load-bearing rather than a formality: see
    README section 5.
    """
    raise NotImplementedByMemberB("metrics.preservation.judge_branches is Member B's")


def score_preservation(system_outputs, gold_instances, **kwargs) -> PreservationScores:
    """Compute PR / SR / HCR / SCR over a run.

    Member B: results must be broken out by ``construction`` into
    ``by_construction``. Reporting one blended PR across Tier 1 and Tier 2
    would let constructed instances carry the naturally-occurring claim.
    """
    raise NotImplementedByMemberB("metrics.preservation.score_preservation is Member B's")


def validate_judge(llm_judgements, human_judgements) -> dict[str, float]:
    """Compare the LLM judge against human labels.

    Member B: report per-branch agreement accuracy AND the Pearson and
    Spearman correlation between headline PR/SR/HCR/SCR computed under LLM
    judging versus human judging, on the same systems and instances.
    """
    raise NotImplementedByMemberB("metrics.preservation.validate_judge is Member B's")
