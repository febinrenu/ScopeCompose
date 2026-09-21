"""Local NLI cross-encoder.

The workhorse of the local GPU budget. Used in four places:

* A2 stage 1, as the fast filter over all C(5,2) = 10 pairs per query
* A3 variant (i), as one feature among the hand-authored lexical cues
* A3 variant (iii), called twice -- condition-entailment and
  outcome-entailment scored separately
* Member B's grounding gate (B1), scoring whether a passage entails a
  proposed condition

One model loaded once, reused everywhere, which is what keeps the whole local
stack inside the 6 GB floor.

A ``HeuristicNLI`` stand-in is provided for tests and CPU-only machines. It is
a lexical approximation, not a substitute -- it exists so the surrounding
pipeline can be exercised without a model download, and it is never used for a
reported number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import settings


@dataclass(frozen=True)
class NLIScores:
    """Probabilities over the three NLI labels, for one (premise, hypothesis)."""

    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        return max(
            (("entailment", self.entailment),
             ("neutral", self.neutral),
             ("contradiction", self.contradiction)),
            key=lambda kv: kv[1],
        )[0]

    @property
    def conflict_signal(self) -> float:
        """How much this pair looks like a disagreement.

        Contradiction is the direct signal. Neutral gets a small weight
        because a genuine conditional conflict often scores NEUTRAL rather
        than CONTRADICTION -- the general rule and its exception are not
        logically contradictory, which is the entire premise of this project.
        An NLI model that only fires on contradiction would systematically
        miss the class the pipeline exists to catch.
        """
        return self.contradiction + 0.25 * self.neutral

    def as_features(self) -> list[float]:
        return [self.entailment, self.neutral, self.contradiction]


class NLIScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> NLIScores: ...
    def score_batch(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[NLIScores]: ...


# --------------------------------------------------------------------------- #
# Transformer cross-encoder
# --------------------------------------------------------------------------- #


class CrossEncoderNLI:
    """DeBERTa-v3-class NLI cross-encoder, fp16, from the hardware profile."""

    def __init__(self, profile: str | None = None, model_name: str | None = None):
        self.hw = settings.hardware(profile)
        self.spec = self.hw.cross_encoder
        self.model_name = model_name or self.spec.model
        self._model = None
        self._tokenizer = None
        self._label_order: list[str] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self.hw.resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.hw.use_fp16() else torch.float32,
        )
        model.eval().to(device)
        self._model = model
        self._device = device

        # Label order varies between checkpoints. Reading it from the config
        # rather than assuming one avoids the quiet failure where entailment
        # and contradiction are swapped and every score is inverted.
        id2label = getattr(model.config, "id2label", None) or {}
        self._label_order = [str(id2label.get(i, i)).lower() for i in range(model.config.num_labels)]

    def _index_of(self, *candidates: str) -> int | None:
        assert self._label_order is not None
        for i, name in enumerate(self._label_order):
            if any(c in name for c in candidates):
                return i
        return None

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        self._load()
        import torch

        out: list[NLIScores] = []
        bs = self.spec.batch_size
        for start in range(0, len(pairs), bs):
            chunk = list(pairs[start:start + bs])
            enc = self._tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                padding=True, truncation=True,
                max_length=self.spec.max_seq_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            i_ent = self._index_of("entail")
            i_con = self._index_of("contradict")
            i_neu = self._index_of("neutral")

            for row in probs:
                ent = float(row[i_ent]) if i_ent is not None else 0.0
                con = float(row[i_con]) if i_con is not None else 0.0
                neu = float(row[i_neu]) if i_neu is not None else max(0.0, 1.0 - ent - con)
                out.append(NLIScores(entailment=ent, neutral=neu, contradiction=con))
        return out

    def score(self, premise: str, hypothesis: str) -> NLIScores:
        return self.score_batch([(premise, hypothesis)])[0]


# --------------------------------------------------------------------------- #
# Heuristic stand-in
# --------------------------------------------------------------------------- #

_NEGATIONS = {"not", "no", "never", "cannot", "won't", "doesn't", "does", "may", "prohibited",
              "excluded", "waived", "exempt", "denied"}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")


class HeuristicNLI:
    """Lexical approximation of an NLI model.

    Exists so the detection, scope and order-invariance code paths can be
    tested on a machine with no GPU and no model download. It is a fixture,
    not a model: it never produces a reported number, and anything that
    depends on real entailment quality must be run with
    :class:`CrossEncoderNLI`.
    """

    def score(self, premise: str, hypothesis: str) -> NLIScores:
        p_tok = set(re.findall(r"[a-z']+", premise.lower()))
        h_tok = set(re.findall(r"[a-z']+", hypothesis.lower()))
        if not p_tok or not h_tok:
            return NLIScores(0.0, 1.0, 0.0)

        overlap = len(p_tok & h_tok) / max(1, len(h_tok))

        p_nums = set(_NUM_RE.findall(premise))
        h_nums = set(_NUM_RE.findall(hypothesis))
        numeric_clash = bool(p_nums and h_nums and not (p_nums & h_nums))

        p_neg = bool(p_tok & _NEGATIONS)
        h_neg = bool(h_tok & _NEGATIONS)
        polarity_clash = p_neg != h_neg

        if numeric_clash and overlap > 0.3:
            return NLIScores(0.05, 0.15, 0.80)
        if polarity_clash and overlap > 0.3:
            return NLIScores(0.10, 0.25, 0.65)
        if overlap > 0.7:
            return NLIScores(0.75, 0.20, 0.05)
        if overlap > 0.35:
            return NLIScores(0.25, 0.65, 0.10)
        return NLIScores(0.05, 0.90, 0.05)

    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        return [self.score(p, h) for p, h in pairs]


@lru_cache(maxsize=4)
def get_nli(profile: str | None = None, *, heuristic: bool = False) -> NLIScorer:
    """Shared NLI scorer.

    Cached so that A2, A3 and the grounding gate share one loaded model rather
    than each holding their own copy of the weights.
    """
    return HeuristicNLI() if heuristic else CrossEncoderNLI(profile)
