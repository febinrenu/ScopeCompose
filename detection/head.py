"""Learned classification head over pair features.

Small enough to train in seconds on CPU, which is deliberate: the expensive
part of A2/A3 is the encoder forward pass, and keeping the head cheap means
threshold sweeps and cross-validation are free.

Two feature sources are combined:

* the NLI cross-encoder's three label probabilities
* the embedding interaction block ``[e_i; e_j; |e_i - e_j|; e_i * e_j]``
* (optionally) the hand-authored lexical features from ``features.py``

The interaction block is the standard sentence-pair encoding: the difference
term captures disagreement and the product term captures shared topic, which
is exactly the distinction between "these contradict" and "these are
unrelated".
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


def feature_signature() -> str:
    """Fingerprint of everything that determines what a feature VALUE means.

    A saved head is only valid for the feature extractor it was fitted on. The
    vector's *shape* is not enough to check that: splitting the cue lists
    changed what ``exception_cues_max`` counts without changing how many
    numbers there are, so a stale head loaded fine and scored nonsense. This
    hashes the feature names together with the cue vocabularies, so any change
    to either invalidates the checkpoint loudly.
    """
    from detection.features import (
        OPINION_CUES,
        RESTRICTION_CUES,
        STRONG_EXCEPTION_CUES,
        TEMPORAL_CUES,
        WEAK_CONDITION_CUES,
        PairFeatures,
    )

    payload = json.dumps({
        "features": PairFeatures.names(),
        "strong": list(STRONG_EXCEPTION_CUES),
        "weak": list(WEAK_CONDITION_CUES),
        "restriction": list(RESTRICTION_CUES),
        "temporal": list(TEMPORAL_CUES),
        "opinion": list(OPINION_CUES),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class StaleHeadError(RuntimeError):
    """A saved head was fitted on a different feature extractor."""


def interaction_features(e_i: np.ndarray, e_j: np.ndarray) -> np.ndarray:
    """Build ``[e_i; e_j; |e_i - e_j|; e_i * e_j]`` for one or many pairs.

    Made ORDER-SYMMETRIC by sorting the two embeddings into a canonical
    position before concatenation. Without that, swapping the passage order
    changes the feature vector and the detector can return a different answer
    for the same pair -- which would break the order-invariance property A4 is
    required to satisfy, and would do it invisibly.
    """
    e_i = np.atleast_2d(e_i)
    e_j = np.atleast_2d(e_j)

    # Canonical ordering: whichever vector has the smaller first-element sign
    # pattern goes first. Using a scalar summary keeps this cheap and stable.
    key_i = e_i.sum(axis=1, keepdims=True)
    key_j = e_j.sum(axis=1, keepdims=True)
    swap = (key_i > key_j).astype(e_i.dtype)
    a = e_i * (1 - swap) + e_j * swap
    b = e_j * (1 - swap) + e_i * swap

    return np.concatenate([a, b, np.abs(a - b), a * b], axis=1)


@dataclass
class HeadConfig:
    kind: Literal["logistic", "mlp"] = "logistic"
    hidden: tuple[int, ...] = (64,)
    max_iter: int = 2000
    C: float = 1.0
    random_state: int = 0
    class_weight: str | None = "balanced"
    """Balanced by default: conditional instances are a minority class in any
    realistic corpus, and an unweighted head trained on them quietly learns to
    predict the majority."""


class PairHead:
    """A classifier over pair features. Wraps scikit-learn."""

    def __init__(self, config: HeadConfig | None = None, *, feature_names: list[str] | None = None):
        self.config = config or HeadConfig()
        self.feature_names = feature_names
        self._model = None
        self._scaler = None
        self.classes_: list[str] = []

    def _build(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        if self.config.kind == "logistic":
            return LogisticRegression(
                max_iter=self.config.max_iter,
                C=self.config.C,
                random_state=self.config.random_state,
                class_weight=self.config.class_weight,
            )
        return MLPClassifier(
            hidden_layer_sizes=self.config.hidden,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray, y: Sequence) -> PairHead:
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        y = list(y)
        if len(set(y)) < 2:
            raise ValueError(
                f"need at least two classes to train, got {sorted(set(y))}. "
                "A fixture with only one label cannot exercise a classifier."
            )

        self._scaler = StandardScaler().fit(X)
        self._model = self._build()
        self._model.fit(self._scaler.transform(X), y)
        self.classes_ = [str(c) for c in self._model.classes_]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("head is not trained; call fit() first")
        return self._model.predict_proba(self._scaler.transform(np.asarray(X, dtype=np.float64)))

    def predict(self, X: np.ndarray) -> list[str]:
        if self._model is None:
            raise RuntimeError("head is not trained; call fit() first")
        return [str(c) for c in self._model.predict(self._scaler.transform(np.asarray(X, dtype=np.float64)))]

    def predict_with_confidence(self, X: np.ndarray) -> list[tuple[str, float]]:
        """Label plus the probability assigned to it.

        The confidence is what A2's stage-2 escalation thresholds on, so it
        has to be the probability of the PREDICTED class, not of a fixed one.
        """
        probs = self.predict_proba(X)
        return [(self.classes_[int(row.argmax())], float(row.max())) for row in probs]

    # -- persistence --------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(
                {"model": self._model, "scaler": self._scaler, "classes": self.classes_,
                 "config": self.config, "feature_names": self.feature_names,
                 "feature_signature": feature_signature()},
                fh,
            )
        p.with_suffix(".json").write_text(
            json.dumps({"classes": self.classes_, "kind": self.config.kind,
                        "feature_names": self.feature_names}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> PairHead:
        """Load a saved head.

        ``strict=True`` refuses a head whose feature signature no longer
        matches the current extractor. That refusal is the point: a stale head
        does not crash, it quietly scores against features that have changed
        meaning, and every downstream number is then wrong in a way nothing
        surfaces.
        """
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)

        saved_sig = blob.get("feature_signature")
        current_sig = feature_signature()
        if strict and saved_sig != current_sig:
            raise StaleHeadError(
                f"{path} was fitted on feature signature {saved_sig or '(none recorded)'}, "
                f"but the current extractor is {current_sig}. The feature set or a cue "
                f"vocabulary has changed since it was trained. Retrain it: "
                f"python -m detection.train --data <labelled.jsonl>"
            )

        head = cls(blob["config"], feature_names=blob.get("feature_names"))
        head._model = blob["model"]
        head._scaler = blob["scaler"]
        head.classes_ = blob["classes"]
        return head

    # -- inspection ---------------------------------------------------------- #

    def coefficients(self) -> dict[str, dict[str, float]] | None:
        """Per-class feature weights, for the logistic head.

        Variant (i)'s selling point is that it is inspectable, so this is part
        of the deliverable rather than a debugging aid -- it is what lets the
        paper say *which* cues drive the conditional decision.
        """
        if self._model is None or self.config.kind != "logistic":
            return None
        names = self.feature_names or [f"f{i}" for i in range(self._model.coef_.shape[1])]
        coefs = self._model.coef_
        if coefs.shape[0] == 1:  # binary
            return {self.classes_[1]: dict(zip(names, map(float, coefs[0])))}
        return {
            cls: dict(zip(names, map(float, row)))
            for cls, row in zip(self.classes_, coefs)
        }

    def top_features(self, cls: str, n: int = 8) -> list[tuple[str, float]]:
        coefs = self.coefficients()
        if not coefs or cls not in coefs:
            return []
        return sorted(coefs[cls].items(), key=lambda kv: -abs(kv[1]))[:n]
