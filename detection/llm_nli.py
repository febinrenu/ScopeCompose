"""An entailment scorer backed by the API, satisfying the NLI protocol.

Drop-in for :class:`detection.nli.NLIScorer`, so anything taking a local
cross-encoder takes this instead: the grounding gate, outcome matching in the
metric suite, the scope analyser.

**It exists to be ablated against, not to replace the local scorer.** The WP1
spec names local-NLI-versus-LLM-judge as a decision to make on evidence rather
than assume. The trade is stark and worth stating before any number is
measured:

* The local DeBERTa cross-encoder costs nothing per call and runs inside the
  6 GB VRAM floor. The grounding gate scores every candidate, so "nothing per
  call" is the difference between a gate that runs on the full corpus and one
  that does not.
* This costs an API call per pair. On a capped free tier that is wall-clock as
  well as money, and the gate is the highest-volume consumer in the pipeline.

So the local scorer stays the default for full-corpus runs unless the ablation
shows a gap large enough to justify the spend. Defaulting the other way would
be choosing on intuition.

**Batching is real, not cosmetic.** One call per pair would make the gate
unusable; pairs go up in groups with indexed results. A reply that loses or
reorders an index falls back to neutral for the affected pair rather than
silently misaligning scores against the wrong hypotheses -- which would be
undetectable downstream and would corrupt every number the gate feeds.
"""

from __future__ import annotations

import json
from typing import Sequence

from api_budget.client import LLMClient, Tier, get_client
from detection.nli import NLIScores

#: Pairs per request. Large enough that the gate is affordable, small enough
#: that one malformed reply does not cost a whole run.
BATCH_SIZE = 10

_SYSTEM = """You judge textual entailment, one pair at a time.

For each numbered pair, decide whether the PREMISE supports the HYPOTHESIS:

- entailment: the premise states or clearly implies the hypothesis
- neutral: the premise neither supports nor rules it out
- contradiction: the premise rules the hypothesis out

Judge ONLY from the premise. Outside knowledge is not evidence, and a
hypothesis that is true in the world but absent from the premise is neutral.

Return JSON with one entry per pair, keeping the given index:
{"results": [{"i": 0, "label": "entailment", "confidence": 0.9}]}"""


class LLMEntailment:
    """NLI scores from a hosted model. Satisfies the NLIScorer protocol."""

    def __init__(self, *, client: LLMClient | None = None,
                 tier: Tier = Tier.JUDGE, step: str = "llm_nli",
                 batch_size: int = BATCH_SIZE):
        self.client = client or get_client()
        self.tier = tier
        self.step = step
        self.batch_size = batch_size
        self.calls = 0

    def score(self, premise: str, hypothesis: str) -> NLIScores:
        return self.score_batch([(premise, hypothesis)])[0]

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        out: list[NLIScores] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = list(pairs[start:start + self.batch_size])
            out.extend(self._score_chunk(chunk))
        return out

    def _score_chunk(self, chunk: list[tuple[str, str]]) -> list[NLIScores]:
        body = "\n\n".join(
            f"[{i}]\nPREMISE: {p}\nHYPOTHESIS: {h}"
            for i, (p, h) in enumerate(chunk))

        try:
            result = self.client.complete(
                messages=[{"role": "user", "content": body}],
                system=_SYSTEM,
                tier=self.tier,
                step=self.step,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            self.calls += 1
            raw = result.json() or {}
        except Exception:
            # A failed call is not evidence of anything, so every pair in the
            # chunk goes to neutral. Scoring them as entailment would admit
            # ungrounded conditions on an API error, which is the one direction
            # a gate must never fail in.
            return [_neutral() for _ in chunk]

        by_index: dict[int, NLIScores] = {}
        for item in raw.get("results") or []:
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            if not 0 <= i < len(chunk):
                continue
            by_index[i] = _from_label(
                str(item.get("label", "")).strip().lower(),
                item.get("confidence"))

        # Missing indices become neutral rather than shifting later results
        # into earlier slots. A silent misalignment here would score each
        # hypothesis against the wrong premise and be invisible downstream.
        return [by_index.get(i, _neutral()) for i in range(len(chunk))]


def _neutral() -> NLIScores:
    return NLIScores(entailment=0.0, neutral=1.0, contradiction=0.0)


def _from_label(label: str, confidence) -> NLIScores:
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        c = 0.8
    c = min(1.0, max(0.34, c))
    rest = (1.0 - c) / 2

    if label.startswith("entail"):
        return NLIScores(entailment=c, neutral=rest, contradiction=rest)
    if label.startswith("contra"):
        return NLIScores(entailment=rest, neutral=rest, contradiction=c)
    return NLIScores(entailment=rest, neutral=c, contradiction=rest)
