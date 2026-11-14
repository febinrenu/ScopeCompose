"""Scoring with the benchmark's own script.

The primary metric is ConditionalQA's official one, computed by the authors'
``evaluate.py`` (vendored unchanged in ``third_party/conditionalqa``): answer
EM / F1, and the same multiplied by the F1 of the predicted condition set
against the gold condition set, matched by exact element string.

Three properties of that script shape everything upstream of it:

* conditions are compared as **exact element strings**, so a system must cite
  the elements it relied on, and a chunk must keep elements whole;
* a predicted condition where the gold has none scores **zero** for that answer
  -- over-conditioning is punished as hard as omitting a condition;
* each predicted answer beyond the gold count multiplies the score by
  ``exp(1 - n_pred / n_gold)`` -- hedging by listing answers is punished.

So the metric cannot be gamed by verbosity, which is the property the retired
Preservation Rate lacked.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import settings

MAX_ANSWERS = 4      # the scorer tries every permutation; an output contract, not a change to the scorer


@lru_cache(maxsize=None)
def _official():
    path = settings.REPO_ROOT / "third_party" / "conditionalqa" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("conditionalqa_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Score:
    em: float
    em_cond: float
    f1: float
    f1_cond: float

    def as_dict(self) -> dict[str, float]:
        return {"em": self.em, "em_cond": self.em_cond, "f1": self.f1, "f1_cond": self.f1_cond}


Prediction = Sequence[tuple[str, Sequence[str]]]


def score(prediction: Prediction, reference: Sequence[Sequence]) -> Score:
    """One question, by the official ``compute_metrics``.

    ``prediction`` and ``reference`` are lists of ``(answer, [condition element strings])``.
    """
    pred = [(str(a), list(dict.fromkeys(c))) for a, c in list(prediction)[:MAX_ANSWERS]]
    ref = [(str(a), list(c)) for a, c in reference]
    em, em_c, f1, f1_c = _official().compute_metrics(pred, ref)     # the script pads `pred` in place
    return Score(float(em), float(em_c), float(f1), float(f1_c))


def condition_counts(prediction: Prediction, reference: Sequence[Sequence]) -> tuple[int, int, int]:
    """(correct, predicted, gold) over the union of condition elements -- for element-level precision and recall."""
    pred = {c for _, conds in prediction for c in conds}
    gold = {c for _, conds in reference for c in conds}
    return len(pred & gold), len(pred), len(gold)
