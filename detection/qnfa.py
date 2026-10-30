"""QNFA pair decisions: detector D5 (proposal v8 section 5.4).

Given frames extracted from separate passages, decide for every cross-source pair
whether they form a **conditional conflict** -- a refinement: same decision, one
scope strictly inside the other, outcomes that differ on the inner scope.

======================================================================  =============
step                                                                     skips the pair if
======================================================================  =============
1. unify the signatures                                                  ``DISTINCT`` (different decisions)
2. (``UNCERTAIN``: a tie-break may decide; with none, abstain)           uncertain and no tie-break
3. outcomes **commensurable**: the same kind of quantity                 not comparable ("fee" vs "conversion rate")
4. fold the signature difference into applicability, then compare        --
5. relation from the typed scope algebra                                 --
======================================================================  =============

Only ``REFINEMENT`` is a conditional conflict. Equal scopes that disagree are a
*same-scope* conflict (the one case where selection can be right), scopes that
cross are *crossing*, scopes that agree are *redundant*, and scopes that do not
meet are *disjoint*. They are labelled, never merged into "conflict".

**What it deliberately does not use.** No lexical-overlap gate, no embedding
similarity, and no similarity threshold of any kind decides alignment: the whole
claim of the method is that a signature can align a rule and its carve-out when
their wording shares almost nothing. The pilot detector gated on lexical
topicality and recovered 2 of 67 such pairs; this decision procedure is where
that gate used to be.

The score is a product of three factors in [0, 1] and is a *ranking*, not a
probability. Calibrating it (isotonic regression on dev, ECE reported) is a
separate, later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from contract.frames import Modality, NormativeFrame
from detection.signature import (
    Kind,
    Schema,
    coreference_components,
    fold_into_applicability,
    load_schema,
    unify,
)
from metrics.values import canonicalise_outcome, values_match
from scope.attributes import SetRelation, compare

#: Words that carry no information about *what quantity* an outcome is about.
_QUANTITY_STOP = {"applies", "apply", "applied", "is", "are", "a", "an", "the", "no", "not", "of", "to", "for",
                  "be", "been", "any", "all", "per", "in", "on", "and", "or", "will", "may", "must", "can",
                  "is_permitted", "permitted", "prohibited", "required", "eligible", "ineligible"}

S_ATTRIBUTE_DECIDED = 0.95
S_FALLBACK = 0.60
S_LEXICAL_MISMATCH = 0.80


class Decision(str, Enum):
    CONDITIONAL = "conditional"      # refinement: the paper's subject
    SAME_SCOPE = "same_scope"
    CROSSING = "crossing"
    REDUNDANT = "redundant"
    DISJOINT = "disjoint"
    UNKNOWN = "unknown"              # scopes not comparable from typed attributes


@dataclass(frozen=True)
class PairDecision:
    passage_i: str
    passage_j: str
    decision: Decision
    score: float
    unification: Kind
    general: str | None = None
    """For a refinement: which passage states the general rule (``passage_i`` / ``passage_j``)."""
    reason: str = ""

    @property
    def is_conditional(self) -> bool:
        return self.decision is Decision.CONDITIONAL

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.passage_i, self.passage_j)))  # type: ignore[return-value]


@dataclass
class DetectionResult:
    pairs: list[PairDecision] = field(default_factory=list)
    skipped_distinct: int = 0
    skipped_incommensurable: int = 0
    abstained_uncertain: int = 0

    @property
    def conditional(self) -> list[PairDecision]:
        return [p for p in self.pairs if p.is_conditional]


#: Different words for the same quantity. "A flat 2.50 charge applies" against "no charge applies" is a
#: fee compared with a fee; only the negative phrasing is canonicalised to "fee" upstream, so without
#: this the positive side would say "charge" and the pair would wrongly read as incomparable.
_QUANTITY_SYNONYMS = {"charge": "fee", "charges": "fee", "fees": "fee", "cost": "fee", "costs": "fee",
                      "surcharge": "fee", "penalty": "fee", "penalties": "fee",
                      "wage": "pay", "salary": "pay", "income": "pay", "earnings": "pay"}


def _content_tokens(outcome: str) -> set[str]:
    import re

    words = re.findall(r"[a-z0-9%.]+", canonicalise_outcome(outcome))
    return {_QUANTITY_SYNONYMS.get(w, w) for w in words if w not in _QUANTITY_STOP}


def commensurable(a: str, b: str) -> bool:
    """Do two outcomes speak about the same quantity?

    "A 2.50 fee applies" and "the conversion rate is 1.2" both contain a number but are not
    comparable, and calling them a conflict is a false positive the proposal names. Two outcomes are
    commensurable when, after dropping words that say nothing about *what* is decided, they still
    share a content word (fee, employment, funds, access ...).
    """
    return bool(_content_tokens(a) & _content_tokens(b))


TieBreak = Callable[[NormativeFrame, NormativeFrame], Kind | None]


class QNFA:
    """Decide conditional conflicts between frames. Pure: no model call happens here."""

    def __init__(self, *, schema_domain: str | None = None, schema: Schema | None = None,
                 tiebreak: TieBreak | None = None):
        self._schema = schema
        self._domain = schema_domain
        self.tiebreak = tiebreak

    def _schema_for(self, f: NormativeFrame) -> Schema:
        return self._schema or load_schema(self._domain or f.signature.domain)

    # ------------------------------------------------------------------ #

    def decide_pair(self, a: NormativeFrame, b: NormativeFrame) -> tuple[PairDecision | None, str]:
        """One pair. ``(None, why)`` when the pair is skipped rather than decided."""
        sch = self._schema_for(a)
        u = unify(a.signature, b.signature, schema=sch)
        kind = u.kind
        tiebroken = False
        if kind is Kind.DISTINCT:
            return None, "distinct"
        if kind is Kind.UNCERTAIN:
            resolved = self.tiebreak(a, b) if self.tiebreak else None
            if resolved is None or resolved in (Kind.UNCERTAIN, Kind.DISTINCT):
                return None, "uncertain"
            # The tie-break settles *co-reference*. The slot comparison is still uncertain, so
            # nothing is folded into applicability and the alignment score is the fallback one.
            kind, tiebroken = resolved, True
        if not commensurable(a.outcome, b.outcome):
            return None, "incommensurable"

        app_a, app_b = fold_into_applicability(u, a.signature, b.signature, a.applicability, b.applicability,
                                               schema=sch)
        rel = compare(app_a, app_b)
        match = values_match(a.outcome, b.outcome)
        s_align = S_FALLBACK if tiebroken else S_ATTRIBUTE_DECIDED
        s_mis = 1.0 if not match else 0.0
        if not match and not (_figures_differ(a.outcome, b.outcome) or _polarity_differs(a.outcome, b.outcome)):
            s_mis = S_LEXICAL_MISMATCH
        s_rel = S_ATTRIBUTE_DECIDED if rel is not SetRelation.UNKNOWN else S_FALLBACK

        ids = (a.passage_id, b.passage_id)
        # Hedged frames never *determine* an outcome, so they cannot form a conflict either.
        if a.modality is Modality.HEDGED or b.modality is Modality.HEDGED:
            return PairDecision(*ids, Decision.UNKNOWN, 0.0, kind, None, "a hedged frame determines nothing"), ""

        def out(d: Decision, general: str | None = None, why: str = "") -> tuple[PairDecision, str]:
            return PairDecision(*ids, d, round(s_align * s_mis * s_rel, 6) if d is not Decision.UNKNOWN else 0.0,
                                kind, general, why), ""

        if rel is SetRelation.DISJOINT:
            return out(Decision.DISJOINT, why="scopes do not meet")
        if rel is SetRelation.UNKNOWN:
            return out(Decision.UNKNOWN, why="scopes are not comparable from typed attributes")
        if match:
            return out(Decision.REDUNDANT, why="outcomes agree where the scopes meet")
        if rel is SetRelation.EQUAL:
            return out(Decision.SAME_SCOPE, why="equal scopes, differing outcomes")
        if rel is SetRelation.OVERLAPPING:
            return out(Decision.CROSSING, why="scopes overlap, neither contains the other")
        # SUBSET / SUPERSET: the broader frame states the general rule
        general = b.passage_id if rel is SetRelation.SUBSET else a.passage_id
        return out(Decision.CONDITIONAL, general=general, why="strict refinement with differing outcomes")

    def detect_frames(self, frames: Sequence[NormativeFrame], *, cross_source_only: bool = True) -> DetectionResult:
        """Every pair of frames, from different passages by default (a rule and its carve-out in
        one passage is within-provision scope, which is a different paper's subject)."""
        res = DetectionResult()
        ordered = sorted(frames, key=lambda f: (f.passage_id, f.span))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if cross_source_only and a.passage_id == b.passage_id:
                    continue
                pd, why = self.decide_pair(a, b)
                if pd is None:
                    if why == "distinct":
                        res.skipped_distinct += 1
                    elif why == "incommensurable":
                        res.skipped_incommensurable += 1
                    else:
                        res.abstained_uncertain += 1
                else:
                    res.pairs.append(pd)
        return res

    def best_conditional_per_passage_pair(self, result: DetectionResult) -> dict[tuple[str, str], PairDecision]:
        """Several frame pairs can come from the same two passages; keep the strongest decision."""
        best: dict[tuple[str, str], PairDecision] = {}
        rank = {Decision.CONDITIONAL: 5, Decision.CROSSING: 4, Decision.SAME_SCOPE: 4, Decision.REDUNDANT: 3,
                Decision.DISJOINT: 2, Decision.UNKNOWN: 1}
        for p in result.pairs:
            cur = best.get(p.key)
            if cur is None or (rank[p.decision], p.score) > (rank[cur.decision], cur.score):
                best[p.key] = p
        return best


def _figures_differ(a: str, b: str) -> bool:
    from metrics.values import figures_conflict

    return figures_conflict(a, b)


def _polarity_differs(a: str, b: str) -> bool:
    from metrics.branch_match import polarity_conflict

    return polarity_conflict(canonicalise_outcome(a), canonicalise_outcome(b))
