"""Extract an applicability descriptor and outcome from a passage.

A4 needs two things per claim before it can decide anything: *who or what does
this apply to* (the applicability set) and *what happens* (the outcome). This
module gets them, preferring typed attributes wherever the text supports them,
because a typed attribute makes the subset question arithmetic instead of
a prose judgement.

Extraction is a ``BULK``-tier call, cached. A rule-based extractor is provided
for offline runs; it is coarse, and it says so -- anything it cannot type
becomes free text, which routes the decision to entailment rather than
producing a confidently wrong typed attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from api_budget import Tier, get_client
from api_budget.client import LLMClient
from contract.gold import Applicability, AttributeKind, ScopeAttribute

STEP = "a4_descriptor_extraction"

SYSTEM_PROMPT = """\
You extract the SCOPE and the OUTCOME of a rule stated in a passage.

Return ONLY a JSON object:
{
  "descriptor": "<short phrase naming who or what the rule applies to>",
  "is_default": <true if the rule applies generally with no restriction, else false>,
  "outcome": "<short phrase naming what happens>",
  "attributes": [
    {"name": "<dimension>", "kind": "<categorical|ordered_tier|numeric_range|boolean|free_text>",
     "values": ["..."], "min_value": null, "max_value": null, "boolean_value": null,
     "text": null}
  ]
}

Rules for "attributes" -- this is the part that matters:

- Break the scope into the DIMENSIONS it restricts. A dimension is a property
  of a case: card_tier, visa_category, age, transaction_amount, account_kind.
- Use "ordered_tier" when the values sit on a scale (bronze < silver < gold).
  List ALL values at or above the stated level, because "premium cardholders"
  usually means premium and everything above it.
- Use "categorical" for unordered membership; list the allowed values.
- Use "numeric_range" with min_value / max_value for amounts, ages, durations.
  Use null for an open end.
- Use "boolean" for a yes/no property, with boolean_value.
- Use "free_text" ONLY when the restriction genuinely cannot be typed. Prefer a
  typed attribute wherever one is defensible, but do NOT invent a dimension the
  passage does not state -- a wrong typed attribute is worse than free text,
  because it will be reasoned over arithmetically as if it were certain.
- If the rule applies with no restriction at all, set "is_default" to true and
  leave "attributes" empty.

"outcome" must be the consequence, not a restatement of the scope.
Good: "a 3% fee applies", "employment is permitted", "no fee".
Bad: "premium cardholders", "for students"."""

USER_TEMPLATE = """\
Question under consideration: {query}

Passage:
{text}

JSON:"""


@dataclass(frozen=True)
class ClaimDescriptor:
    """One passage's scope and outcome."""

    applicability: Applicability
    outcome: str
    passage_id: str
    source: str = "llm"
    """'llm' or 'rules' -- carried through so a run that silently fell back to
    the coarse extractor is visible in the results."""


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #


def _parse_attribute(raw: dict) -> ScopeAttribute | None:
    try:
        kind = AttributeKind(str(raw.get("kind", "free_text")).strip().lower())
    except ValueError:
        kind = AttributeKind.FREE_TEXT

    name = str(raw.get("name") or "").strip()
    if not name:
        return None

    values = raw.get("values")
    values = [str(v) for v in values] if isinstance(values, list) and values else None

    def _num(key):
        v = raw.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    lo, hi = _num("min_value"), _num("max_value")
    bv = raw.get("boolean_value")
    bv = bool(bv) if isinstance(bv, bool) else None
    text = raw.get("text")
    text = str(text) if text else None

    # Repair a payload that does not match its declared kind. The model
    # routinely says "categorical" and then fills min_value; rather than drop
    # the attribute, re-kind it to what it actually carries, and degrade to
    # free text only when nothing usable is present.
    if kind in (AttributeKind.CATEGORICAL, AttributeKind.ORDERED_TIER) and not values:
        if lo is not None or hi is not None:
            kind = AttributeKind.NUMERIC_RANGE
        elif bv is not None:
            kind = AttributeKind.BOOLEAN
        else:
            kind, text = AttributeKind.FREE_TEXT, text or name
    elif kind is AttributeKind.NUMERIC_RANGE and lo is None and hi is None:
        if values:
            kind = AttributeKind.CATEGORICAL
        else:
            kind, text = AttributeKind.FREE_TEXT, text or name
    elif kind is AttributeKind.BOOLEAN and bv is None:
        kind, text = AttributeKind.FREE_TEXT, text or name
    elif kind is AttributeKind.FREE_TEXT and not text:
        text = name

    # An inverted range is a model error, not a constraint. Swapping is the
    # charitable repair; dropping it would lose the dimension entirely.
    if kind is AttributeKind.NUMERIC_RANGE and lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo

    try:
        return ScopeAttribute(name=name, kind=kind, values=values,
                              min_value=lo, max_value=hi, boolean_value=bv, text=text)
    except Exception:
        return None


class DescriptorExtractor:
    """LLM-based scope and outcome extraction."""

    def __init__(self, client: LLMClient | None = None, *, tier: Tier = Tier.BULK):
        self._client = client
        self.tier = tier

    @property
    def client(self) -> LLMClient:
        return self._client if self._client is not None else get_client()

    def extract(self, query: str, passage_id: str, text: str) -> ClaimDescriptor:
        result = self.client.complete(
            messages=[{"role": "user", "content": USER_TEMPLATE.format(query=query, text=text)}],
            system=SYSTEM_PROMPT,
            tier=self.tier,
            step=STEP,
            response_format={"type": "json_object"},
        )
        try:
            raw = result.json()
        except ValueError:
            from detection.stage2 import _salvage_json

            raw = _salvage_json(result.text)

        if not isinstance(raw, dict) or not raw.get("descriptor"):
            return rule_based_descriptor(passage_id, text)

        is_default = bool(raw.get("is_default", False))
        attrs = [a for a in (_parse_attribute(x) for x in (raw.get("attributes") or [])
                             if isinstance(x, dict)) if a is not None]
        if is_default:
            attrs = []   # the default branch is U; any constraint contradicts that

        return ClaimDescriptor(
            applicability=Applicability(
                descriptor=str(raw["descriptor"])[:200],
                attributes=attrs,
                is_default=is_default,
            ),
            outcome=str(raw.get("outcome") or "")[:200] or "unspecified",
            passage_id=passage_id,
            source="llm",
        )


# --------------------------------------------------------------------------- #
# Rule-based fallback
# --------------------------------------------------------------------------- #

_TIER_SCALE = ["basic", "standard", "classic", "silver", "gold", "premium",
               "platinum", "signature"]
_VISA_RE = re.compile(r"\b([A-Z]-?\d{1,2})\b")
_AMOUNT_RE = re.compile(r"(?:above|over|exceeding|more than|greater than)\s+([\d,]+)", re.I)
_UNDER_RE = re.compile(r"(?:below|under|less than|up to|at most)\s+([\d,]+)", re.I)
_SCOPE_LEAD_RE = re.compile(
    r"\b(?:for|to|by|applies to|available to|waived for|exempt for)\s+"
    r"([a-z][a-z \-]{3,50}?)(?=[,.;]|\s+(?:may|must|shall|are|is|will|who|that)\b)",
    re.I,
)
_OUTCOME_RE = re.compile(
    r"\b(no fee applies|no fee|fee is waived|waived|fee-free|"
    r"a? ?\d+(?:\.\d+)?%\s*fee|may not accept employment|may accept employment|"
    r"employment is (?:permitted|prohibited)|permitted|prohibited|eligible|ineligible)\b",
    re.I,
)

_GENERAL_MARKERS = ("all ", "any ", "every ", "customers", "cardholders incur",
                    "transactions incur")


def rule_based_descriptor(passage_id: str, text: str) -> ClaimDescriptor:
    """Coarse offline extraction.

    Deliberately conservative: it types only the patterns it can recognise with
    confidence (tiers, visa categories, amount thresholds) and leaves everything
    else untyped, so an undecidable case reaches entailment instead of being
    settled by a bad guess.
    """
    low = text.lower()
    attrs: list[ScopeAttribute] = []

    for tier in reversed(_TIER_SCALE):
        if re.search(rf"\b{tier}[- ]?(?:tier|level|card|programme|program)?\b", low):
            idx = _TIER_SCALE.index(tier)
            attrs.append(ScopeAttribute(name="card_tier", kind=AttributeKind.ORDERED_TIER,
                                        values=_TIER_SCALE[idx:]))
            break

    visas = _VISA_RE.findall(text)
    if visas:
        attrs.append(ScopeAttribute(name="visa_category", kind=AttributeKind.CATEGORICAL,
                                    values=sorted({v.replace("-", "-") for v in visas})))

    if (m := _AMOUNT_RE.search(text)):
        attrs.append(ScopeAttribute(name="amount", kind=AttributeKind.NUMERIC_RANGE,
                                    min_value=float(m.group(1).replace(",", ""))))
    elif (m := _UNDER_RE.search(text)):
        attrs.append(ScopeAttribute(name="amount", kind=AttributeKind.NUMERIC_RANGE,
                                    max_value=float(m.group(1).replace(",", ""))))

    from detection.features import EXCEPTION_CUES

    has_exception_cue = any(c in low for c in EXCEPTION_CUES)
    looks_general = any(m in low for m in _GENERAL_MARKERS)
    is_default = not attrs and not has_exception_cue and looks_general

    if (m := _SCOPE_LEAD_RE.search(text)):
        descriptor = m.group(1).strip()
    elif attrs:
        descriptor = " and ".join(
            f"{a.name}={a.values or [a.min_value, a.max_value]}" for a in attrs
        )
    else:
        descriptor = "all cases" if is_default else text[:60].strip()

    outcome_m = _OUTCOME_RE.search(text)
    outcome = outcome_m.group(0).strip() if outcome_m else text[:80].strip()

    return ClaimDescriptor(
        applicability=Applicability(descriptor=descriptor or "unspecified",
                                    attributes=attrs, is_default=is_default),
        outcome=outcome,
        passage_id=passage_id,
        source="rules",
    )
