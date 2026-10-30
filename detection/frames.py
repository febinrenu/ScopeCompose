"""Extract normative frames from retrieved passages (QNFA step 1, proposal v8 section 5.4).

One **batched** call per query returns, for every passage, the frames relevant
to that query: a verbatim supporting span, a decision-variable signature, an
applicability (typed where the text supports it), an outcome and a modality.

Three rules keep the extractor honest, and each produces a reported number:

1. **Spans are verbatim or the frame is dropped.** A frame whose span the
   passage does not contain is a fabrication with a citation attached. Offending
   frames get *one* re-ask (the model is told which spans failed and must copy
   them exactly); any that still fail are dropped and counted
   (``dropped_span``). A dropped frame is never repaired by us.
2. **Out-of-vocabulary values are kept, not guessed.** A signature value outside
   the schema stays free text (``Signature.untyped_slots``). The out-of-vocabulary
   rate is reported, because it measures what the hand-designed schema misses.
3. **Hedging is decided deterministically, and it only goes one way.** The model
   names a modality, but if :func:`metrics.values.detect_modality` finds a hedge
   in the span the frame is ``HEDGED`` whatever the model said. A false hedge
   costs recall; a false determination would inflate case-complete accuracy.

The extractor never sees gold labels and is called with the *cell's own*
backbone tier: frames come from the same model that will resolve them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

from api_budget import Tier, get_client
from api_budget.client import LLMClient
from contract.frames import Modality, NormativeFrame
from contract.gold import Applicability
from contract.models import Passage
from detection.signature import Schema, load_schema, make_signature
from metrics.values import detect_modality
from scope.descriptor import _parse_attribute

STEP = "qnfa_frame_extraction"
STEP_REASK = "qnfa_frame_reask"


def _vocabulary_block(schema: Schema) -> str:
    lines = []
    for slot in schema.slots:
        top = schema.root(slot)
        vals = [v for v in schema.parents[slot] if v != top]
        lines.append(f"  {slot}: {', '.join(vals)}")
    return "\n".join(lines)


def build_system_prompt(schema: Schema) -> str:
    return f"""\
You read passages and extract the NORMATIVE FRAMES that bear on a question. A frame
is one passage's statement about ONE decision.

Return ONLY a JSON object:
{{"frames": [
  {{"passage_id": "<id of the passage>",
    "span": "<one sentence or clause COPIED WORD FOR WORD from that passage>",
    "decision": {{"<slot>": "<value>", ...}},
    "applicability": {{
      "descriptor": "<short phrase: who or what this applies to>",
      "is_default": <true if it applies to everyone, no restriction>,
      "attributes": [{{"name": "<dimension>", "kind": "categorical|ordered_tier|numeric_range|boolean|free_text",
                      "values": [...], "min_value": null, "max_value": null,
                      "boolean_value": null, "text": null}}]}},
    "outcome": "<short phrase: what happens>",
    "modality": "obligation|permission|prohibition|statement|hedged",
    "confidence": 0.0-1.0}}
]}}

Rules:
- "decision" names WHAT IS BEING DECIDED, not who it applies to. Use these slots and,
  where one fits, one of these values; if none fits, write a short free-text value:
{_vocabulary_block(schema)}
- Leave a slot out if the passage does not say. Do not guess it.
- "applicability" is WHO OR WHAT the rule covers; put restrictions there, not in "decision".
- "span" must appear in the passage exactly as written. Do not paraphrase or merge sentences.
- One frame per distinct rule. Skip passages that say nothing about the question.
- Use modality "hedged" when the span says "may", "might", "could", "in some cases" or "subject to"
  about whether the outcome happens."""


def _user_prompt(query: str, passages: Sequence[Passage]) -> str:
    body = "\n\n".join(f"[{p.id}] {p.text}" for p in passages)
    return f"Question: {query}\n\nPassages:\n{body}"


def _reask_prompt(bad: list[dict], passages: Sequence[Passage]) -> str:
    body = "\n\n".join(f"[{p.id}] {p.text}" for p in passages)
    listing = "\n".join(f'- passage {b.get("passage_id")}: "{b.get("span")}"' for b in bad)
    return (
        "These spans were not verbatim in their passages:\n"
        f"{listing}\n\n"
        "Return the same JSON object with only the corrected frames, each span copied word for "
        "word from its passage. If a frame has no exact supporting span, leave it out.\n\n"
        f"Passages:\n{body}"
    )


@dataclass
class ExtractionStats:
    n_passages: int = 0
    n_raw: int = 0
    n_kept: int = 0
    dropped_span: int = 0
    dropped_other: int = 0
    reasked: int = 0
    recovered_by_reask: int = 0
    typed_slots: int = 0
    untyped_slots: int = 0
    forced_hedged: int = 0
    calls: int = 0

    @property
    def frame_yield(self) -> float:
        """Kept frames per passage."""
        return self.n_kept / self.n_passages if self.n_passages else 0.0

    @property
    def dropped_span_rate(self) -> float:
        return self.dropped_span / self.n_raw if self.n_raw else 0.0

    @property
    def oov_rate(self) -> float:
        n = self.typed_slots + self.untyped_slots
        return self.untyped_slots / n if n else 0.0

    def merge(self, other: "ExtractionStats") -> "ExtractionStats":
        out = ExtractionStats()
        for f in self.__dataclass_fields__:
            setattr(out, f, getattr(self, f) + getattr(other, f))
        return out


@dataclass
class Extraction:
    frames: list[NormativeFrame]
    stats: ExtractionStats
    raw: list[str] = field(default_factory=list)


def _parse_json(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        from detection.stage2 import _salvage_json

        return _salvage_json(text)


class FrameExtractor:
    """Batched, schema-validated frame extraction with one verbatim re-ask."""

    def __init__(self, domain: str, *, tier: Tier, client: LLMClient | None = None,
                 schema: Schema | None = None, max_reasks: int = 1):
        self.domain = domain
        self.tier = tier
        self._client = client
        self.schema = schema or load_schema(domain)
        self.max_reasks = max_reasks
        self._system = build_system_prompt(self.schema)

    @property
    def client(self) -> LLMClient:
        return self._client if self._client is not None else get_client()

    def _call(self, content: str, step: str, stats: ExtractionStats, raw: list[str]) -> list[dict]:
        res = self.client.complete(
            messages=[{"role": "user", "content": content}], system=self._system,
            tier=self.tier, step=step, response_format={"type": "json_object"})
        stats.calls += 1
        raw.append(res.text)
        data = _parse_json(res.text)
        frames = data.get("frames") if isinstance(data, dict) else None
        return [f for f in (frames or []) if isinstance(f, dict)]

    def _build(self, raw: dict, by_id: dict[str, Passage], stats: ExtractionStats) -> NormativeFrame | str:
        """A validated frame, or the reason it was rejected: ``"span"`` or ``"other"``."""
        pid, span, outcome = (_text(raw.get(k)) for k in ("passage_id", "span", "outcome"))
        if pid not in by_id or not span or not outcome:
            return "other"
        app = raw.get("applicability") if isinstance(raw.get("applicability"), dict) else {}
        is_default = bool(app.get("is_default", False))
        attrs = [] if is_default else [a for a in (
            _parse_attribute(x) for x in (app.get("attributes") or []) if isinstance(x, dict)) if a]
        descriptor = (_text(app.get("descriptor")) or ("everyone" if is_default else outcome))[:200]
        try:
            applicability = Applicability(descriptor=descriptor, attributes=attrs, is_default=is_default)
            decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
            sig = make_signature(self.domain, {str(k): str(v) for k, v in decision.items()
                                               if v is not None}, schema=self.schema)
            modality = _modality(raw.get("modality"), span)
            frame = NormativeFrame(
                passage_id=pid, signature=sig, applicability=applicability, outcome=outcome[:200],
                modality=modality, span=span,
                confidence=_clamp(raw.get("confidence", 1.0)))
        except Exception:
            return "other"
        if not frame.span_is_verbatim_in(by_id[pid].text):
            return "span"
        # Counters are updated only for frames that are kept.
        stats.typed_slots += len(frame.signature.slots)
        stats.untyped_slots += len(frame.signature.untyped_slots)
        if modality is Modality.HEDGED and str(raw.get("modality", "")).lower() != "hedged":
            stats.forced_hedged += 1
        return frame

    def extract(self, query: str, passages: Sequence[Passage]) -> Extraction:
        by_id = {p.id: p for p in passages}
        stats, raw_texts = ExtractionStats(n_passages=len(passages)), []
        raws = self._call(_user_prompt(query, passages), STEP, stats, raw_texts)
        stats.n_raw = len(raws)

        kept: list[NormativeFrame] = []
        bad: list[dict] = []
        for r in raws:
            out = self._build(r, by_id, stats)
            if isinstance(out, NormativeFrame):
                kept.append(out)
            elif out == "span":
                bad.append(r)
            else:
                stats.dropped_other += 1

        initially_bad = len(bad)
        for _ in range(self.max_reasks):
            if not bad:
                break
            stats.reasked += len(bad)
            retry = self._call(_reask_prompt(bad, passages), STEP_REASK, stats, raw_texts)
            still: list[dict] = []
            for r in retry:
                out = self._build(r, by_id, stats)
                if isinstance(out, NormativeFrame):
                    kept.append(out)
                    stats.recovered_by_reask += 1
                elif out == "span":
                    still.append(r)
            bad = still
        # A frame that was bad and was not recovered is dropped, whether the re-ask
        # returned it still wrong or did not return it at all.
        stats.dropped_span = max(0, initially_bad - stats.recovered_by_reask)

        seen: set[tuple[str, str]] = set()
        unique = []
        for f in kept:
            key = (f.passage_id, " ".join(f.span.split()))
            if key not in seen:
                seen.add(key)
                unique.append(f)
        stats.n_kept = len(unique)
        unique.sort(key=lambda f: (f.passage_id, f.span))   # canonical order: never the model's
        return Extraction(unique, stats, raw_texts)


def _text(x) -> str:
    """A JSON string field as text. ``null`` is empty, not the word "None"."""
    return "" if x is None else str(x).strip()


def _clamp(x) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return 1.0


def _modality(model_said, span: str) -> Modality:
    """The model's modality, overridden toward HEDGED by the deterministic detector."""
    detected = detect_modality(span)
    if detected is Modality.HEDGED:
        return Modality.HEDGED
    try:
        said = Modality(str(model_said).strip().lower())
    except ValueError:
        return detected
    return said
