"""A ConfRAG-style *answer coverage* comparator (proposal v8 section 2.1, H-K).

ConfRAG (Yuan et al., ACL 2026) scores conflict handling by **answer coverage**:
a gold viewpoint cluster counts as covered if any of its judge keywords appears
as a *substring* of the predicted answer. Its own Figure-1 example gets full
credit when one answer contains "Yes" and another contains "No".

That is a perfectly reasonable measure of whether a response *mentions* each
viewpoint. It cannot tell whether the response says **who** each applies to, so
an answer that lists every outcome and attaches none of them to anybody is fully
covered -- while the person the exception protects learns nothing. This module
exists to *show* that, not to assert it: it is run on the same validity battery
as case-complete accuracy, where the unattached hedge must score at least as
high as the gold answer (H-K).

**This is a reimplementation, and its limits are stated.** ConfRAG's judge
keywords are written by hand for each question. Ours are derived automatically
from each branch's outcome, so they are a stand-in. The derivation is
deliberately *generous* -- every common phrasing of an outcome counts -- because
a stingy stand-in would make the comparator look worse than the original and
flatter CCA. A generous comparator that still cannot separate the unattached
hedge from the gold answer is the stronger result.

Keywords shared between two branches are dropped (one keyword may not cover two
clusters), which is ConfRAG's one-to-one rule applied at the keyword level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contract.gold import Branch, GoldInstance
from metrics.values import _normalise_figures, figures, strip_citations  # noqa: PLC2701

#: Phrasings of the outcome classes ``metrics.values`` collapses, expanded back
#: out as keywords. Generous on purpose: see the module docstring.
_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "no fee applies": ("no fee", "no charge", "waived", "free of charge", "fee-free", "fee free",
                       "not charged", "exempt", "pays nothing", "nothing to pay", "without a fee"),
    "is ineligible": ("not eligible", "ineligible", "do not qualify", "does not qualify",
                      "not entitled", "cannot qualify"),
    "is eligible": ("eligible", "entitled", "qualify", "qualifies"),
    "is not required": ("not required", "need not", "no need to", "optional", "not mandatory"),
    "is required": ("required", "must", "mandatory", "need to", "have to"),
    "is prohibited": ("not permitted", "not allowed", "prohibited", "forbidden", "may not",
                      "cannot", "must not", "barred"),
    "is permitted": ("permitted", "allowed", "may ", "can "),
}


def _canon(text: str) -> str:
    from metrics.values import canonicalise_outcome

    return canonicalise_outcome(text)


def _raw_keywords(branch: Branch) -> set[str]:
    canon = _canon(branch.outcome)
    kws: set[str] = set()
    # figures with their unit: "3%", "28 days"
    for unit, vals in figures(branch.outcome).items():
        for v in vals:
            kws.add(f"{v}%" if unit == "%" else v)
    for cls, phrasings in _CLASS_KEYWORDS.items():
        if cls in canon:
            kws.update(phrasings)
    if not kws:
        kws.add(canon)  # nothing recognisable: the whole canonical outcome is the keyword
    return {k.lower() for k in kws if k.strip()}


def branch_keywords(inst: GoldInstance) -> dict[str, frozenset[str]]:
    """Judge keywords per gold branch, with keywords shared across branches removed."""
    raw = {b.branch_id: _raw_keywords(b) for b in inst.gold_branches}
    out: dict[str, frozenset[str]] = {}
    for bid, kws in raw.items():
        others = set().union(*(v for k, v in raw.items() if k != bid)) if len(raw) > 1 else set()
        unique = kws - others
        out[bid] = frozenset(unique or kws)   # a branch never ends up keyword-less
    return out


@dataclass(frozen=True)
class Coverage:
    instance_id: str
    covered: tuple[str, ...]
    total: int

    @property
    def score(self) -> float:
        return len(self.covered) / self.total if self.total else 0.0


def answer_coverage(inst: GoldInstance, answer: str) -> Coverage:
    """Branches whose keywords appear as substrings of the answer (case-insensitive)."""
    text = _normalise_figures(strip_citations(answer or "")).lower()
    # "(per p1)"-style citations carry digits that must not match a figure keyword
    text = re.sub(r"\bp\d+\b", " ", text)
    kws = branch_keywords(inst)
    covered = tuple(bid for bid, ks in kws.items() if any(k in text for k in ks))
    return Coverage(inst.instance_id, covered, len(kws))


def mean_coverage(instances: list[GoldInstance], answers: dict[str, str]) -> float:
    """Macro-average answer coverage over instances."""
    if not instances:
        return 0.0
    return sum(answer_coverage(i, answers[i.instance_id]).score for i in instances) / len(instances)
