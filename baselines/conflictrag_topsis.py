"""The selection core of a ConflictRAG-style resolver (baseline B3).

ConflictRAG (Wang et al., 2026, arXiv:2605.17301) types an inter-document
conflict and resolves it by type:

====================  ============================================
factual               **Entropy-TOPSIS credibility selection**
temporal              recency
opinion               multi-perspective synthesis
====================  ============================================

A rule-and-exception pair is typed factual or temporal, so it is resolved by
*selection*: whichever passage wins, the answer is wrong for one group of users.
This module is that selection step, so the paper can run the design it argues
against instead of a proxy for it.

**It is called ConflictRAG-*style*, throughout.** We reimplement the mechanism
from the paper's description; the authors' code has been requested and is not
used. Everything below that the paper does not pin down is our choice and is
named as such: the criteria, their scales, and the sufficiency margin.

Entropy-TOPSIS
--------------

``m`` alternatives (passages) scored on ``n`` benefit criteria::

    1. r_ij = x_ij / sum_i x_ij                       (column shares)
    2. e_j  = -(1 / ln m) * sum_i r_ij ln r_ij        (entropy of criterion j)
    3. w_j  = (1 - e_j) / sum_k (1 - e_k)             (more diverse => more weight)
    4. v_ij = w_j * x_ij / sqrt(sum_i x_ij^2)         (weighted vector-normalised)
    5. C_i  = S-_i / (S+_i + S-_i)                    (closeness to the ideal)

A criterion on which every passage scores the same has entropy 1 and weight 0:
it cannot discriminate, so it is not allowed to.

**The sufficiency rule** (plan v4 D-12). Selection is the right strategy for a
*same-scope* conflict only when the evidence is sufficient. If the winning
closeness does not beat the runner-up by more than ``delta``, the resolver
**presents both sources** instead of choosing -- which is the behaviour the
proposal's softened selection claim requires ("subject to evidence sufficiency
and source reliability").

**Specificity is an ablatable criterion.** A credibility score that rewards the
more specific passage partly *re-creates* the exception-over-rule preference by
a side door. ``include_specificity=False`` removes it, so the paper can report
how much of B3's behaviour that single criterion explains.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Sequence

from contract.models import Passage, SourceType

#: Source authority on a 0-1 scale. Our ordering, not the paper's: official
#: policy over product terms over FAQ over third-party. Fixed here, hashed into
#: the pre-registration, never tuned on test.
AUTHORITY: dict[SourceType, float] = {
    SourceType.OFFICIAL_POLICY: 1.0,
    SourceType.GOVERNMENT_GUIDANCE: 0.9,
    SourceType.AMENDMENT: 0.8,
    SourceType.PRODUCT_TERMS: 0.7,
    SourceType.FAQ: 0.5,
    SourceType.CATEGORY_PAGE: 0.4,
    SourceType.THIRD_PARTY: 0.3,
    SourceType.SYNTHETIC: 0.2,
    SourceType.UNKNOWN: 0.1,
}

#: Default sufficiency margin. A PLACEHOLDER until tuned on dev; results carry it.
DEFAULT_DELTA = 0.10

CRITERIA = ("authority", "recency", "specificity")

_FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%?")
_CONDITION_RE = re.compile(
    r"\b(?:if|unless|only|where|provided|except|excluding|for (?:those|people|holders|customers|"
    r"applicants)|who|whose|over|under|at least|no more than|more than|less than)\b",
    re.IGNORECASE,
)
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Entropy-TOPSIS
# --------------------------------------------------------------------------- #


def entropy_weights(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Entropy weights for the columns of an ``m x n`` benefit matrix.

    All values must be non-negative. A single alternative, or a matrix on which
    no criterion discriminates, gets equal weights (there is nothing to choose
    between them, and dividing by a zero total would be a crash, not an answer).
    """
    m = len(matrix)
    if m == 0:
        raise ValueError("no alternatives")
    n = len(matrix[0])
    if any(len(r) != n for r in matrix):
        raise ValueError("ragged matrix")
    if any(x < 0 for r in matrix for x in r):
        raise ValueError("entropy weights need non-negative scores")
    if m == 1:
        return [1.0 / n] * n

    diversity = []
    for j in range(n):
        total = sum(matrix[i][j] for i in range(m))
        if total <= _EPS:
            diversity.append(0.0)
            continue
        e = 0.0
        for i in range(m):
            r = matrix[i][j] / total
            if r > _EPS:
                e -= r * math.log(r)
        e /= math.log(m)
        diversity.append(max(0.0, 1.0 - e))
    s = sum(diversity)
    return [1.0 / n] * n if s <= _EPS else [d / s for d in diversity]


def topsis_closeness(matrix: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """TOPSIS closeness ``C_i`` in [0, 1] for benefit criteria. 1 = ideal, 0 = anti-ideal."""
    m, n = len(matrix), len(matrix[0])
    norms = [math.sqrt(sum(matrix[i][j] ** 2 for i in range(m))) or 1.0 for j in range(n)]
    v = [[weights[j] * matrix[i][j] / norms[j] for j in range(n)] for i in range(m)]
    best = [max(v[i][j] for i in range(m)) for j in range(n)]
    worst = [min(v[i][j] for i in range(m)) for j in range(n)]
    out = []
    for i in range(m):
        s_plus = math.sqrt(sum((v[i][j] - best[j]) ** 2 for j in range(n)))
        s_minus = math.sqrt(sum((v[i][j] - worst[j]) ** 2 for j in range(n)))
        out.append(0.5 if s_plus + s_minus <= _EPS else s_minus / (s_plus + s_minus))
    return out


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #


def _parse_date(s: str | None) -> _date | None:
    if not s:
        return None
    parts = [int(p) for p in s.split("-")]
    return _date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)


def recency_scores(passages: Sequence[Passage]) -> list[float]:
    """Min-max recency in [0, 1]; undated passages score 0. Equal dates all score 1."""
    dates = [_parse_date(p.date) for p in passages]
    known = [d for d in dates if d is not None]
    if not known:
        return [0.0] * len(passages)
    lo, hi = min(known), max(known)
    span = (hi - lo).days
    return [0.0 if d is None else (1.0 if span == 0 else (d - lo).days / span) for d in dates]


def specificity_score(text: str) -> float:
    """How narrowly a passage speaks: figures plus condition cues, squashed to [0, 1).

    A crude, documented proxy. It is the criterion the ablation removes, so its
    exact form matters less than its being separable.
    """
    raw = len(_FIGURE_RE.findall(text)) + 1.5 * len(_CONDITION_RE.findall(text))
    return raw / (raw + 4.0)


@dataclass(frozen=True)
class Scored:
    passage_id: str
    closeness: float
    criteria: dict[str, float]


@dataclass(frozen=True)
class CredibilityResult:
    scored: tuple[Scored, ...]
    weights: dict[str, float]
    margin: float
    """Winning closeness minus runner-up's (0 with one passage)."""
    include_specificity: bool

    @property
    def ranked(self) -> list[Scored]:
        return sorted(self.scored, key=lambda s: (-s.closeness, s.passage_id))


def credibility(passages: Sequence[Passage], *, include_specificity: bool = True) -> CredibilityResult:
    """Entropy-TOPSIS closeness of each passage. Order-independent (ties break by id)."""
    if not passages:
        raise ValueError("no passages")
    names = [c for c in CRITERIA if include_specificity or c != "specificity"]
    rec = recency_scores(passages)
    cols = {
        "authority": [AUTHORITY[p.source_type] for p in passages],
        "recency": rec,
        "specificity": [specificity_score(p.text) for p in passages],
    }
    matrix = [[cols[c][i] for c in names] for i in range(len(passages))]
    w = entropy_weights(matrix)
    close = topsis_closeness(matrix, w)
    scored = tuple(Scored(p.id, close[i], {c: cols[c][i] for c in names}) for i, p in enumerate(passages))
    top = sorted(close, reverse=True)
    margin = top[0] - top[1] if len(top) > 1 else 0.0
    return CredibilityResult(scored, dict(zip(names, w)), margin, include_specificity)


# --------------------------------------------------------------------------- #
# Selection, with the sufficiency rule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Selection:
    kind: str
    """``"selected"``, ``"both"`` (insufficient evidence: present every source) or ``"recency"``."""
    chosen: tuple[str, ...]
    result: CredibilityResult | None = None
    delta: float = DEFAULT_DELTA
    notes: list[str] = field(default_factory=list)


def select_by_credibility(
    passages: Sequence[Passage],
    *,
    delta: float = DEFAULT_DELTA,
    include_specificity: bool = True,
) -> Selection:
    """Factual-conflict resolution: pick the most credible passage, if clearly best.

    With one passage there is nothing to select. When the margin does not exceed
    ``delta`` every passage is returned: choosing between sources we cannot
    tell apart would be an arbitrary deletion presented as a decision.
    """
    if len(passages) == 1:
        return Selection("selected", (passages[0].id,), None, delta, ["single passage"])
    res = credibility(passages, include_specificity=include_specificity)
    if res.margin > delta:
        return Selection("selected", (res.ranked[0].passage_id,), res, delta)
    return Selection("both", tuple(sorted(p.id for p in passages)), res, delta,
                     [f"margin {res.margin:.3f} <= delta {delta}: evidence insufficient to select"])


def select_by_recency(passages: Sequence[Passage]) -> Selection:
    """Temporal-conflict resolution: the most recently dated passage.

    Undated passages never win over a dated one. If the newest date is shared,
    every passage carrying it is kept (a tie is not a decision).
    """
    dates = {p.id: _parse_date(p.date) for p in passages}
    known = {pid: d for pid, d in dates.items() if d is not None}
    if not known:
        return Selection("both", tuple(sorted(p.id for p in passages)), None, 0.0,
                         ["no dates available: cannot select by recency"])
    newest = max(known.values())
    winners = tuple(sorted(pid for pid, d in known.items() if d == newest))
    return Selection("recency", winners, None, 0.0,
                     [] if len(winners) == 1 else ["tie on the newest date: all kept"])
