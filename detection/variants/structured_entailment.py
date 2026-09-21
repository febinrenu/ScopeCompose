"""A3 variant (iii): structured entailment.

Instead of predicting one end-to-end label, this variant asks the cross-encoder
two separate questions and combines the answers by the formal rule in A4:

1. **Condition entailment** -- does one passage's applicability entail the
   other's? That decides whether the scopes nest, are disjoint, or merely
   overlap.
2. **Outcome entailment** -- do the two passages agree about what happens?
   That decides whether there is anything to resolve at all.

The pairing of those two answers *is* the four-way relation. Keeping them
separate is the point: it is the same correction the four-way relation made
over the earlier two-way version, which conflated "do the scopes overlap" with
"is this a conflict" and so could not tell a redundant restatement from a real
exception.

Structurally this variant is the closest of the three to how the scope
analyser itself reasons, which makes its comparison against the other two
informative beyond a leaderboard: if it wins on the factual/conditional cell,
the structure is doing the work, not the parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract.models import ConflictType, ScopeRelation
from detection.nli import NLIScorer, get_nli
from detection.stage1 import PairCandidate
from detection.variants.base import DetectorVariant

# Hypothesis templates. Phrased so that a standard MNLI-trained cross-encoder
# is being asked something close to what it was trained on -- an NLI model
# asked an out-of-distribution question returns confident noise.
_SCOPE_HYPOTHESIS = "This statement applies to the same cases as: {other}"
_OUTCOME_HYPOTHESIS = "The outcome described here is the same as: {other}"


@dataclass(frozen=True)
class EntailmentEvidence:
    """The four scores this variant reasons over."""

    scope_i_entails_j: float
    scope_j_entails_i: float
    outcome_agreement: float
    outcome_contradiction: float

    @property
    def scopes_nest(self) -> bool:
        """One applicability set sits inside the other."""
        return max(self.scope_i_entails_j, self.scope_j_entails_i) >= 0.5

    @property
    def scopes_overlap(self) -> bool:
        return max(self.scope_i_entails_j, self.scope_j_entails_i) >= 0.25

    @property
    def outcomes_agree(self) -> bool:
        return self.outcome_agreement > self.outcome_contradiction


class StructuredEntailmentDetector(DetectorVariant):
    name = "structured_entailment"
    description = (
        "cross-encoder called twice -- condition-entailment and outcome-entailment "
        "scored separately, then combined by the formal four-way rule"
    )

    def __init__(
        self,
        *,
        nli: NLIScorer | None = None,
        profile: str | None = None,
        heuristic_nli: bool = False,
        nest_threshold: float = 0.5,
        overlap_threshold: float = 0.25,
    ):
        self.nli = nli if nli is not None else get_nli(profile, heuristic=heuristic_nli)
        self.nest_threshold = nest_threshold
        self.overlap_threshold = overlap_threshold

    # -- evidence -------------------------------------------------------------- #

    def gather(self, text_i: str, text_j: str) -> EntailmentEvidence:
        """Score the two questions in one batch.

        Scope entailment is asked in BOTH directions deliberately: subset is
        asymmetric, and that asymmetry is exactly what distinguishes a
        refinement (one inside the other) from an opposed pair (overlapping,
        neither inside the other).
        """
        scores = self.nli.score_batch([
            (text_i, _SCOPE_HYPOTHESIS.format(other=text_j)),
            (text_j, _SCOPE_HYPOTHESIS.format(other=text_i)),
            (text_i, _OUTCOME_HYPOTHESIS.format(other=text_j)),
        ])
        scope_ij, scope_ji, outcome = scores
        return EntailmentEvidence(
            scope_i_entails_j=scope_ij.entailment,
            scope_j_entails_i=scope_ji.entailment,
            outcome_agreement=outcome.entailment,
            outcome_contradiction=outcome.contradiction,
        )

    # -- the formal rule -------------------------------------------------------- #

    def relation_from_evidence(self, ev: EntailmentEvidence) -> ScopeRelation:
        """Combine the two determinations into the four-way relation.

        This is the rule from the formal definition, written out as code:

        ============ ==================== ====================
        scopes       outcomes agree       outcomes disagree
        ============ ==================== ====================
        disjoint     disjoint             disjoint
        nested       redundant            refinement
        overlapping  redundant            opposed
        ============ ==================== ====================

        Both columns matter. Deciding scope alone cannot separate redundant
        from refinement, and deciding outcome alone cannot separate refinement
        from opposed.
        """
        max_scope = max(ev.scope_i_entails_j, ev.scope_j_entails_i)

        if max_scope < self.overlap_threshold:
            # No case falls under both, so outcome comparison is vacuous.
            return ScopeRelation.DISJOINT

        if ev.outcomes_agree:
            # Overlapping or nested scope, same outcome throughout: a
            # restatement, not a conflict. Merge rather than compose.
            return ScopeRelation.REDUNDANT

        if max_scope >= self.nest_threshold:
            # One scope inside the other, different outcome there: a genuine
            # exception.
            return ScopeRelation.REFINEMENT

        # Overlapping, neither contains the other, outcomes disagree.
        return ScopeRelation.OPPOSED

    # -- classification -------------------------------------------------------- #

    def classify(self, candidate: PairCandidate) -> tuple[ConflictType, ScopeRelation | None]:
        f = candidate.features

        # Temporal and opinion are surface-cued and are not what this variant's
        # structure is for; handle them before the entailment reasoning so they
        # do not pollute the factual/conditional decision.
        if f.temporal_cues_max >= 1 and (f.year_clash or f.numeric_clash):
            return ConflictType.TEMPORAL, None
        if f.opinion_cues_max >= 1:
            return ConflictType.OPINION, None

        ev = self.gather(candidate.text_i, candidate.text_j)
        relation = self.relation_from_evidence(ev)

        if relation is ScopeRelation.OPPOSED:
            # Overlapping scopes with neither containing the other and
            # disagreeing outcomes is a plain contradiction: the two passages
            # cannot both be true. That is the factual class, and calling it
            # conditional would route a real contradiction to composition.
            return ConflictType.FACTUAL, ScopeRelation.OPPOSED

        if relation is ScopeRelation.REDUNDANT and not candidate.is_conflict:
            return ConflictType.NO_CONFLICT, None

        return ConflictType.CONDITIONAL, relation

    def explain(self, candidate: PairCandidate) -> dict[str, float | str]:
        """The four scores plus the resulting relation.

        Used in the qualitative error analysis on confused instances, where the
        useful question is which of the two determinations went wrong, not just
        that the label was wrong.
        """
        ev = self.gather(candidate.text_i, candidate.text_j)
        return {
            "scope_i_entails_j": round(ev.scope_i_entails_j, 4),
            "scope_j_entails_i": round(ev.scope_j_entails_i, 4),
            "outcome_agreement": round(ev.outcome_agreement, 4),
            "outcome_contradiction": round(ev.outcome_contradiction, 4),
            "scopes_nest": ev.scopes_nest,
            "outcomes_agree": ev.outcomes_agree,
            "relation": self.relation_from_evidence(ev).value,
        }
