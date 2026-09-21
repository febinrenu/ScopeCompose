"""A2 stage 2: selective LLM escalation.

Only pairs stage 1 was unsure about reach this code. That selectivity is the
whole design: on a free-tier rate limit, sending all ten pairs per query to an
API is what makes a full-corpus run impossible, and most pairs are not close
calls.

Every call goes through ``api_budget`` so it is cached and attributed. The
second run of any experiment costs nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from api_budget import BackendError, Tier, get_client
from api_budget.client import LLMClient
from contract.models import ConflictType, ScopeRelation
from detection.stage1 import PairCandidate

STEP = "a2_stage2_detection"

SYSTEM_PROMPT = """\
You classify the relationship between two passages retrieved for the same question.

Return ONLY a JSON object with these keys:
  "is_conflict":    true or false
  "type":           one of "no_conflict", "factual", "temporal", "opinion", "conditional"
  "scope_relation": one of "refinement", "disjoint", "redundant", "opposed", or null
  "confidence":     a number from 0.0 to 1.0
  "rationale":      one short sentence

Definitions, which differ from ordinary usage -- read them carefully:

factual      Both passages make the same kind of claim about the same situation,
             and they disagree. One of them is simply wrong.

temporal     They disagree because they describe different points in time.
             One has been superseded.

opinion      They express differing judgements rather than differing facts.

conditional  Both passages are TRUE, but they apply to DIFFERENT SCOPES. One
             states a general rule, the other a narrower case with a different
             outcome. This is NOT a contradiction. Example: "international
             transactions incur a 3% fee" and "fees are waived for premium-tier
             cardholders" are both true and both must be preserved.

no_conflict  They are about different subjects, or they agree.

When "type" is "conditional", set "scope_relation" by deciding TWO things
separately -- whether the scopes overlap, AND whether the outcomes agree where
they do:

  refinement  One scope is strictly inside the other AND the outcomes differ
              there. A genuine exception.
  disjoint    The scopes do not overlap at all.
  redundant   The scopes overlap or nest BUT the outcomes AGREE throughout the
              overlap. A restatement, not an exception. Do not treat a passage
              that merely repeats the general rule for a specific group as an
              exception.
  opposed     The scopes overlap, NEITHER contains the other, and the outcomes
              disagree. A real contradiction.

For any other "type", set "scope_relation" to null.

The hardest and most important distinction is factual vs. conditional. Ask: can
both passages be true at once, for different cases? If yes, it is conditional,
not factual."""

USER_TEMPLATE = """\
Question: {query}

Passage A ({id_i}):
{text_i}

Passage B ({id_j}):
{text_j}

JSON:"""


@dataclass(frozen=True)
class Stage2Judgement:
    is_conflict: bool
    type: ConflictType
    scope_relation: ScopeRelation | None
    confidence: float
    rationale: str
    cached: bool = False
    fell_back: bool = False


def _coerce(raw: dict) -> Stage2Judgement:
    """Parse the model's JSON into a judgement, repairing what is repairable.

    Models routinely emit a scope_relation on a factual pair, or omit one on a
    conditional pair. Both would be rejected by the contract, so they are
    normalised here rather than crashing a full-corpus run -- and the
    normalisation is conservative: an unusable conditional label degrades to
    factual rather than being invented.
    """
    try:
        ctype = ConflictType(str(raw.get("type", "no_conflict")).strip().lower())
    except ValueError:
        ctype = ConflictType.NO_CONFLICT

    rel_raw = raw.get("scope_relation")
    rel: ScopeRelation | None = None
    if rel_raw not in (None, "", "null", "none"):
        try:
            rel = ScopeRelation(str(rel_raw).strip().lower())
        except ValueError:
            rel = None

    is_conflict = bool(raw.get("is_conflict", ctype is not ConflictType.NO_CONFLICT))

    # Contract repairs.
    if ctype is ConflictType.NO_CONFLICT:
        is_conflict, rel = False, None
    elif not is_conflict:
        ctype, rel = ConflictType.NO_CONFLICT, None
    if ctype is ConflictType.CONDITIONAL and rel is None:
        # A conditional label with no relation is unroutable. Downgrading to
        # factual is the conservative repair: it sends the pair to the
        # selection path, which is what prior work would have done anyway.
        ctype = ConflictType.FACTUAL
    if rel is not None and ctype not in (ConflictType.CONDITIONAL, ConflictType.FACTUAL):
        rel = None

    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5

    return Stage2Judgement(
        is_conflict=is_conflict,
        type=ctype,
        scope_relation=rel,
        confidence=min(1.0, max(0.0, conf)),
        rationale=str(raw.get("rationale", ""))[:300],
    )


class Stage2Judge:
    """LLM adjudication for pairs stage 1 could not resolve."""

    def __init__(self, client: LLMClient | None = None, *, tier: Tier = Tier.BULK):
        self._client = client
        self.tier = tier

    @property
    def client(self) -> LLMClient:
        return self._client if self._client is not None else get_client()

    def judge(self, query: str, candidate: PairCandidate) -> Stage2Judgement:
        result = self.client.complete(
            messages=[{
                "role": "user",
                "content": USER_TEMPLATE.format(
                    query=query,
                    id_i=candidate.doc_i, text_i=candidate.text_i,
                    id_j=candidate.doc_j, text_j=candidate.text_j,
                ),
            }],
            system=SYSTEM_PROMPT,
            tier=self.tier,
            step=STEP,
            response_format={"type": "json_object"},
        )
        try:
            raw = result.json()
        except ValueError:
            raw = _salvage_json(result.text)

        judgement = _coerce(raw if isinstance(raw, dict) else {})
        return Stage2Judgement(
            **{**judgement.__dict__, "cached": result.cached, "fell_back": result.fell_back}
        )

    def judge_many(self, query: str, candidates: list[PairCandidate]) -> dict[tuple[str, str], Stage2Judgement]:
        out: dict[tuple[str, str], Stage2Judgement] = {}
        for cand in candidates:
            try:
                out[cand.key] = self.judge(query, cand)
            except BackendError:
                # One unreachable pair must not abort a corpus run. Stage 1's
                # verdict stands, and the miss shows up in the cost log.
                continue
        return out


def _salvage_json(text: str) -> dict:
    """Pull a JSON object out of a response wrapped in prose or fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return {}
