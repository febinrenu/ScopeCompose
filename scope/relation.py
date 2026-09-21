"""A4: the four-way scope-relation analyser.

The decision Member B's entire resolution stage routes on, and the place where
this project's central correction lives.

**Two determinations, kept separate.** An earlier version of this design asked
one question -- "do these scopes overlap?" -- and treated overlap as conflict.
That cannot distinguish a passage restating the general rule for a specific
group (redundant) from a passage carving out a genuine exception (refinement),
because both have nested scope. The fix is to decide applicability and outcome
agreement *independently* and combine them:

============ ======================== ========================
scopes       outcomes AGREE           outcomes DISAGREE
============ ======================== ========================
disjoint     disjoint                 disjoint
nested       **redundant**            **refinement**
overlapping  **redundant**            **opposed**
============ ======================== ========================

Neither determination alone yields the right answer for any row.

**Order invariance is structural, not tested-in.** Branch roles come from the
applicability relation -- whichever set is the superset is the default, full
stop. Nothing here reads the order passages arrived in. That is what
``order_invariance.py`` verifies, but the property is established here, by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contract.gold import Applicability
from contract.models import ScopeRelation
from detection.nli import NLIScorer, get_nli
from scope.attributes import SetRelation, compare
from scope.descriptor import ClaimDescriptor

_OUTCOME_SAME = "The outcome is: {other}"


class Evidence(str, Enum):
    """Where the scope determination came from.

    Reported per decision. Attribute-based determinations are exact;
    entailment-based ones are a model's judgement. Mixing them silently would
    make the four-way accuracy number impossible to interpret.
    """

    ATTRIBUTES = "attributes"
    ENTAILMENT = "entailment"
    DEFAULT_BRANCH = "default_branch"


@dataclass(frozen=True)
class ScopeDecision:
    """The full result of an A4 analysis."""

    relation: ScopeRelation
    default: ClaimDescriptor
    """The wider branch. Derived from applicability, never from input order."""
    exception: ClaimDescriptor
    """The narrower branch."""

    set_relation: SetRelation
    """CANONICAL: expressed as the exception relative to the default, never
    relative to argument order.

    ``compare(a, b)`` is inherently argument-relative -- it returns SUBSET for
    one ordering and SUPERSET for the mirror -- so storing its raw output here
    would make this field flip under passage reordering even though the
    decision did not. A nested pair is therefore always recorded as SUBSET
    (the exception sits inside the default), which is both order-invariant and
    the more useful statement.
    """

    outcomes_agree: bool
    evidence: Evidence
    confidence: float
    rationale: str

    @property
    def roles_swapped(self) -> bool:
        """True when the exception arrived before the default in the input.

        Purely informational -- it has no effect on the decision. It is logged
        so the order-invariance report can show that role assignment did in
        fact flip with input order while the decision did not.
        """
        return self.exception.passage_id < self.default.passage_id


class ScopeAnalyser:
    """Decides the four-way relation between two claims."""

    def __init__(
        self,
        *,
        nli: NLIScorer | None = None,
        profile: str | None = None,
        heuristic_nli: bool = False,
        entailment_nest_threshold: float = 0.5,
        entailment_overlap_threshold: float = 0.25,
        outcome_agreement_margin: float = 0.0,
    ):
        self._nli = nli
        self._profile = profile
        self._heuristic = heuristic_nli
        self.nest_threshold = entailment_nest_threshold
        self.overlap_threshold = entailment_overlap_threshold
        self.outcome_margin = outcome_agreement_margin

    @property
    def nli(self) -> NLIScorer:
        if self._nli is None:
            self._nli = get_nli(self._profile, heuristic=self._heuristic)
        return self._nli

    # -- determination 1: applicability ---------------------------------------- #

    def scope_relation(
        self, a: Applicability, b: Applicability, *, text_a: str = "", text_b: str = ""
    ) -> tuple[SetRelation, Evidence, float]:
        """Decide how two applicability sets relate.

        Typed attributes are tried first because they give an exact answer.
        Entailment is the fallback, not the default: an LLM-scored subset
        judgement is a probability, and treating it as certain is how a
        redundant restatement gets composed as an exception.
        """
        if a.is_default or b.is_default:
            return compare(a, b), Evidence.DEFAULT_BRANCH, 1.0

        rel = compare(a, b)
        if rel is not SetRelation.UNKNOWN:
            return rel, Evidence.ATTRIBUTES, 0.95

        # Fall back to entailment over the descriptors, asked in both
        # directions -- subset is asymmetric, and that asymmetry is what
        # separates refinement from opposed.
        prem_a = text_a or a.descriptor
        prem_b = text_b or b.descriptor
        s_ab, s_ba = self.nli.score_batch([
            (prem_a, f"This applies to: {b.descriptor}"),
            (prem_b, f"This applies to: {a.descriptor}"),
        ])
        ent_ab, ent_ba = s_ab.entailment, s_ba.entailment
        strongest = max(ent_ab, ent_ba)

        if strongest < self.overlap_threshold:
            return SetRelation.DISJOINT, Evidence.ENTAILMENT, 1.0 - strongest

        if strongest >= self.nest_threshold:
            # Whichever direction entails is the narrower set. Ties are broken
            # toward OVERLAPPING rather than picking arbitrarily: guessing a
            # direction here would assign branch roles on a coin flip, which is
            # exactly the order-dependence the design forbids.
            if abs(ent_ab - ent_ba) < 1e-6:
                return SetRelation.OVERLAPPING, Evidence.ENTAILMENT, strongest
            return (
                (SetRelation.SUBSET if ent_ab > ent_ba else SetRelation.SUPERSET),
                Evidence.ENTAILMENT,
                strongest,
            )

        return SetRelation.OVERLAPPING, Evidence.ENTAILMENT, strongest

    # -- determination 2: outcome agreement ------------------------------------ #

    def outcomes_agree(self, outcome_a: str, outcome_b: str) -> tuple[bool, float]:
        """Whether two outcomes say the same thing.

        Asked symmetrically and averaged. An asymmetric answer here would make
        the relation depend on which passage was listed first, defeating order
        invariance at the second determination even though the first is safe.
        """
        if not outcome_a.strip() or not outcome_b.strip():
            return False, 0.0
        if outcome_a.strip().lower() == outcome_b.strip().lower():
            return True, 1.0

        fwd, rev = self.nli.score_batch([
            (outcome_a, _OUTCOME_SAME.format(other=outcome_b)),
            (outcome_b, _OUTCOME_SAME.format(other=outcome_a)),
        ])
        agree = (fwd.entailment + rev.entailment) / 2.0
        disagree = (fwd.contradiction + rev.contradiction) / 2.0
        return agree > disagree + self.outcome_margin, max(agree, disagree)

    # -- the combination ------------------------------------------------------- #

    def analyse(
        self,
        claim_a: ClaimDescriptor,
        claim_b: ClaimDescriptor,
        *,
        text_a: str = "",
        text_b: str = "",
    ) -> ScopeDecision:
        """Run both determinations and combine them into the four-way relation."""
        set_rel, evidence, scope_conf = self.scope_relation(
            claim_a.applicability, claim_b.applicability, text_a=text_a, text_b=text_b
        )

        # Assign roles from the SET RELATION, never from argument order.
        #
        # `canonical` restates the relation in terms of the ROLES rather than
        # the argument positions. Without it, `set_relation` would read SUBSET
        # for one ordering and SUPERSET for the mirror, and the
        # order-invariance harness would flag a violation on every nested pair
        # even though nothing about the decision changed.
        canonical = set_rel
        if set_rel is SetRelation.SUBSET:
            default, exception = claim_b, claim_a
            canonical = SetRelation.SUBSET      # exception sits inside default
        elif set_rel is SetRelation.SUPERSET:
            default, exception = claim_a, claim_b
            canonical = SetRelation.SUBSET      # same structure, stated the same way
        else:
            # Neither contains the other. There is no default/exception
            # structure, so pick a stable, order-independent representative:
            # sort by passage id. The relation itself does not depend on this.
            default, exception = (
                (claim_a, claim_b) if claim_a.passage_id <= claim_b.passage_id
                else (claim_b, claim_a)
            )

        agree, outcome_conf = self.outcomes_agree(claim_a.outcome, claim_b.outcome)
        relation, why = self._combine(set_rel, agree)

        return ScopeDecision(
            relation=relation,
            default=default,
            exception=exception,
            set_relation=canonical,
            outcomes_agree=agree,
            evidence=evidence,
            confidence=round(min(scope_conf, outcome_conf) if evidence is Evidence.ENTAILMENT
                             else scope_conf * 0.5 + outcome_conf * 0.5, 4),
            rationale=why,
        )

    @staticmethod
    def _combine(set_rel: SetRelation, outcomes_agree: bool) -> tuple[ScopeRelation, str]:
        """The formal rule, written out.

        Every branch names why, because these four words are what the paper's
        scope-relation accuracy is measured against and a wrong one has a
        specific downstream consequence worth being able to trace.
        """
        if set_rel is SetRelation.DISJOINT:
            return (
                ScopeRelation.DISJOINT,
                "the applicability sets do not intersect, so no case falls under both "
                "and outcome comparison is vacuous",
            )

        if set_rel is SetRelation.UNKNOWN:
            # Cannot establish the scope relation. Calling it a refinement
            # would license composition on evidence we do not have, and
            # composition invents a branch. Opposed routes to selection, which
            # is what prior work does and is the safe default.
            return (
                ScopeRelation.OPPOSED,
                "the scope relation could not be established from attributes or entailment; "
                "falling back to selection rather than composing on unestablished scope",
            )

        if outcomes_agree:
            return (
                ScopeRelation.REDUNDANT,
                "the scopes overlap or nest but the outcomes agree throughout the overlap -- "
                "a restatement, not an exception; merge rather than compose",
            )

        if set_rel in (SetRelation.SUBSET, SetRelation.SUPERSET):
            return (
                ScopeRelation.REFINEMENT,
                "one applicability set is strictly inside the other and the outcomes differ "
                "there -- a genuine exception that narrows the default without contesting it",
            )

        if set_rel is SetRelation.EQUAL:
            return (
                ScopeRelation.OPPOSED,
                "the two branches cover exactly the same cases and disagree about the "
                "outcome -- a direct contradiction",
            )

        return (
            ScopeRelation.OPPOSED,
            "the scopes intersect, neither contains the other, and the outcomes disagree "
            "on the overlap -- a real contradiction; selection is the honest resolution",
        )
