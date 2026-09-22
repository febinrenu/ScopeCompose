"""Hand-authored features for detector variant (i).

The inspectable half of the three-architecture comparison. Every feature here
is a signal a human annotator could point at and explain, which is the point:
variant (i) is meant to be the design a reviewer can audit, and its
performance on the factual/conditional confusion cell is the baseline the
other two variants have to beat.

The three feature families are the ones named in the proposal:

1. **Hedge and contrast markers** -- "except", "unless", "does not apply to",
   "only if", "provided that". The surface form of an exception.
2. **Entity-type / category restriction** -- one passage names a narrower
   population or product than the other.
3. **Specificity asymmetry** -- one passage is measurably more qualified than
   the other.

Feature (3) is what separates a *conditional* conflict from a *factual* one,
and it is deliberately asymmetric: a factual contradiction is two claims at
the same specificity that disagree, whereas a conditional conflict is a
general claim and a narrower one.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Cues are split by strength, which is not cosmetic.
#
# STRONG cues mark an EXCEPTION: the passage is signalling that it overrides or
# carves out of something else. WEAK cues mark a CONDITION generally -- "where
# the balance is settled", "transfers subject to verification" -- and appear
# constantly in ordinary rule-governed prose that carves out nothing.
#
# Lumping them together was a real defect: "where the", "if the" and "subject
# to" fired on plain conditional sentences, so `exception_cues_max` was partly
# measuring "this text describes a rule" rather than "this text states an
# exception". Since that feature is one of the strongest signals for the
# conditional class, the noise landed directly on the factual/conditional cell
# the detector is judged on.

STRONG_EXCEPTION_CUES: tuple[str, ...] = (
    "except", "unless", "excluding", "other than", "save for", "apart from",
    "does not apply", "do not apply", "shall not apply", "is not applicable",
    "notwithstanding", "provided that", "provided, that",
    "only if", "only when", "only for", "solely for",
    "waived for", "exempt from", "exemption", "carve-out", "carve out",
    "unless otherwise", "with the exception of", "save that",
)

WEAK_CONDITION_CUES: tuple[str, ...] = (
    "subject to", "in the case of", "where the", "if the", "limited to",
    "provided", "for the purposes of", "in respect of",
)

#: Union, kept because callers outside this module treat it as "any cue".
EXCEPTION_CUES: tuple[str, ...] = STRONG_EXCEPTION_CUES + WEAK_CONDITION_CUES

RESTRICTION_CUES: tuple[str, ...] = (
    "tier", "premium", "platinum", "signature", "gold", "silver", "bronze",
    "student", "senior", "resident", "non-resident", "eligible", "qualifying",
    "category", "class", "type", "holders", "members", "customers who",
    "accounts with", "cards with",
)

TEMPORAL_CUES: tuple[str, ...] = (
    "effective", "as of", "from 20", "until", "prior to", "before 20",
    "after 20", "with effect from", "no longer", "previously", "currently",
    "has been updated", "revised",
)

OPINION_CUES: tuple[str, ...] = (
    "we believe", "in our view", "arguably", "many consider", "most advisers",
    "some argue", "regard", "widely seen", "generally considered",
    "practitioners", "critics", "reportedly", "may be seen as",
)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%?")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MODAL_RE = re.compile(r"\b(may|must|shall|can|cannot|will not|won't|should)\b", re.I)
_WORD_RE = re.compile(r"[a-z']+")


def _count_cues(text: str, cues: tuple[str, ...]) -> int:
    low = text.lower()
    return sum(1 for c in cues if c in low)


def _qualifier_count(text: str) -> int:
    """How heavily qualified a statement is.

    The specificity proxy. A bare rule ("international transactions incur a 3%
    fee") has almost none; a scoped one ("waived for premium-tier cardholders
    on transactions under 5,000") has several.
    """
    low = text.lower()
    return (
        _count_cues(low, STRONG_EXCEPTION_CUES)
        + _count_cues(low, WEAK_CONDITION_CUES)
        + _count_cues(low, RESTRICTION_CUES)
        + len(_MODAL_RE.findall(low))
        + low.count(",")
    )


@dataclass(frozen=True)
class PairFeatures:
    """Hand-authored features for one passage pair.

    Ordering is fixed by :meth:`names` so the vector a model was trained on
    matches the vector it is scored on. Adding a feature in the middle would
    silently invalidate a saved model, so new features go at the end.
    """

    # exception structure
    exception_cues_i: float
    exception_cues_j: float
    exception_cues_max: float
    exception_cues_asymmetry: float

    # category / population restriction
    restriction_cues_i: float
    restriction_cues_j: float
    restriction_asymmetry: float

    # specificity
    qualifiers_i: float
    qualifiers_j: float
    specificity_asymmetry: float
    length_ratio: float

    # numeric / temporal disagreement -- the factual and temporal signals
    numeric_overlap: float
    numeric_clash: float
    year_clash: float
    temporal_cues_max: float

    # opinion
    opinion_cues_max: float

    # lexical similarity -- are these even about the same thing?
    token_jaccard: float

    @staticmethod
    def names() -> list[str]:
        return [
            "exception_cues_i", "exception_cues_j", "exception_cues_max",
            "exception_cues_asymmetry",
            "restriction_cues_i", "restriction_cues_j", "restriction_asymmetry",
            "qualifiers_i", "qualifiers_j", "specificity_asymmetry", "length_ratio",
            "numeric_overlap", "numeric_clash", "year_clash", "temporal_cues_max",
            "opinion_cues_max", "token_jaccard",
        ]

    def as_vector(self) -> list[float]:
        d = asdict(self)
        return [float(d[n]) for n in self.names()]


def extract_pair_features(text_i: str, text_j: str) -> PairFeatures:
    """Compute the hand-authored feature vector for a passage pair.

    Symmetric features (``*_max``, ``*_clash``, ``token_jaccard``) are
    order-independent by construction. The asymmetric ones use ``abs()``, so
    they are order-independent too -- which matters, because a feature that
    changed under passage reordering would break the order-invariance property
    A4 is required to hold.
    """
    # Strong cues only: a weak cue means "this sentence states a condition",
    # which is true of almost every passage in this domain and therefore
    # separates nothing.
    exc_i = _count_cues(text_i, STRONG_EXCEPTION_CUES)
    exc_j = _count_cues(text_j, STRONG_EXCEPTION_CUES)
    res_i = _count_cues(text_i, RESTRICTION_CUES)
    res_j = _count_cues(text_j, RESTRICTION_CUES)
    qual_i = _qualifier_count(text_i)
    qual_j = _qualifier_count(text_j)

    nums_i = {n.strip() for n in _NUM_RE.findall(text_i)}
    nums_j = {n.strip() for n in _NUM_RE.findall(text_j)}
    num_union = nums_i | nums_j
    num_overlap = len(nums_i & nums_j) / len(num_union) if num_union else 0.0
    num_clash = 1.0 if (nums_i and nums_j and not (nums_i & nums_j)) else 0.0

    years_i = set(_YEAR_RE.findall(text_i))
    years_j = set(_YEAR_RE.findall(text_j))
    year_clash = 1.0 if (years_i and years_j and years_i != years_j) else 0.0

    tok_i = set(_WORD_RE.findall(text_i.lower()))
    tok_j = set(_WORD_RE.findall(text_j.lower()))
    union = tok_i | tok_j
    jaccard = len(tok_i & tok_j) / len(union) if union else 0.0

    len_i, len_j = max(1, len(text_i.split())), max(1, len(text_j.split()))

    return PairFeatures(
        exception_cues_i=exc_i,
        exception_cues_j=exc_j,
        exception_cues_max=max(exc_i, exc_j),
        exception_cues_asymmetry=abs(exc_i - exc_j),
        restriction_cues_i=res_i,
        restriction_cues_j=res_j,
        restriction_asymmetry=abs(res_i - res_j),
        qualifiers_i=qual_i,
        qualifiers_j=qual_j,
        specificity_asymmetry=abs(qual_i - qual_j),
        length_ratio=min(len_i, len_j) / max(len_i, len_j),
        numeric_overlap=num_overlap,
        numeric_clash=num_clash,
        year_clash=year_clash,
        temporal_cues_max=max(_count_cues(text_i, TEMPORAL_CUES),
                              _count_cues(text_j, TEMPORAL_CUES)),
        opinion_cues_max=max(_count_cues(text_i, OPINION_CUES),
                             _count_cues(text_j, OPINION_CUES)),
        token_jaccard=jaccard,
    )


def more_specific(text_i: str, text_j: str) -> int:
    """Which passage is more heavily qualified.

    Returns -1 if ``i``, +1 if ``j``, 0 if tied.

    This is a heuristic prior on which branch is the exception, NOT the
    decision. Branch roles are assigned in ``scope/relation.py`` from the
    applicability relation, because a heuristic that got this backwards on a
    hard case would silently invert a composed answer.
    """
    qi, qj = _qualifier_count(text_i), _qualifier_count(text_j)
    if qi == qj:
        return 0
    return -1 if qi > qj else 1
