"""A2 stage 1: the local fast filter.

With K = 5 there are C(5,2) = 10 pairs per query. Sending all of them to an
LLM is both slow and wasteful, and on a free-tier rate limit it is the thing
that makes a full-corpus run infeasible. Stage 1 resolves the confident
majority locally at zero marginal cost, and only the uncertain remainder is
escalated.

The signal is the NLI cross-encoder's three label probabilities combined with
the hand-authored lexical features, fed to a small learned head. Where no
trained head is available, a transparent rule-based fallback is used so the
stage still works end to end -- flagged as such, never reported as the
trained system.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from contract.models import Passage, QueryRecord
from detection.features import PairFeatures, extract_pair_features
from detection.head import HeadConfig, PairHead
from detection.nli import NLIScorer, NLIScores, get_nli

DEFAULT_HEAD_PATH = Path("models/stage1_head.pkl")
"""Where `python -m detection.train` saves the head, and where Stage1Filter
looks for one. Gitignored: a trained head is a build artifact, and committing
one would let a stale model silently outlive the features it was fit on."""


@dataclass(frozen=True)
class PairCandidate:
    """One passage pair with its stage-1 evidence."""

    doc_i: str
    doc_j: str
    text_i: str
    text_j: str
    nli: NLIScores
    features: PairFeatures
    is_conflict: bool
    confidence: float
    needs_escalation: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_i, self.doc_j) if self.doc_i < self.doc_j else (self.doc_j, self.doc_i)

    def feature_vector(self) -> list[float]:
        return self.nli.as_features() + self.features.as_vector()


def feature_names() -> list[str]:
    return ["nli_entailment", "nli_neutral", "nli_contradiction"] + PairFeatures.names()


def enumerate_pairs(passages: list[Passage]) -> list[tuple[Passage, Passage]]:
    """All unordered pairs, in canonical (sorted-id) order.

    Sorting here rather than at the call site is what makes the pair set
    identical under any permutation of the input passages.
    """
    ordered = sorted(passages, key=lambda p: p.id)
    return list(combinations(ordered, 2))


class Stage1Filter:
    """Local pairwise conflict filter."""

    def __init__(
        self,
        *,
        nli: NLIScorer | None = None,
        head: PairHead | None = None,
        profile: str | None = None,
        heuristic_nli: bool = False,
        low_threshold: float = 0.35,
        high_threshold: float = 0.75,
        auto_load_head: bool = True,
    ):
        """
        Parameters
        ----------
        low_threshold, high_threshold
            The escalation band. Pairs scoring below ``low`` are confidently
            non-conflicting and pairs above ``high`` are confidently
            conflicting; only the middle is sent to stage 2. These defaults are
            placeholders -- tune them with ``python -m detection.threshold``,
            because the escalation rate is the API cost of detection and on a
            capped tier it is the wall-clock cost too.
        auto_load_head
            Load a trained head from :data:`DEFAULT_HEAD_PATH` when one exists.
            Without a trained head the filter falls back to a hand-weighted
            rule, which is a scaffold rather than a model: measured on mock
            data it scores at chance, so its "resolved locally" pairs are
            resolved *wrongly*. Train one with ``python -m detection.train``.
        """
        self.nli = nli if nli is not None else get_nli(profile, heuristic=heuristic_nli)
        self.head = head
        self.low = low_threshold
        self.high = high_threshold

        if self.head is None and auto_load_head and DEFAULT_HEAD_PATH.exists():
            try:
                self.head = PairHead.load(DEFAULT_HEAD_PATH)
            except Exception:
                # A head saved against a different feature set must not stop
                # the pipeline; the rule-based fallback still runs.
                self.head = None

    # -- scoring -------------------------------------------------------------- #

    def score_pairs(self, passages: list[Passage]) -> list[PairCandidate]:
        pairs = enumerate_pairs(passages)
        if not pairs:
            return []

        nli_scores = self.nli.score_batch([(a.text, b.text) for a, b in pairs])
        feats = [extract_pair_features(a.text, b.text) for a, b in pairs]

        if self.head is not None:
            X = np.array(
                [n.as_features() + f.as_vector() for n, f in zip(nli_scores, feats)],
                dtype=np.float64,
            )
            probs = self.head.predict_proba(X)
            conflict_idx = self._conflict_column()
            conflict_p = [float(row[conflict_idx]) for row in probs]
        else:
            conflict_p = [self._rule_score(n, f) for n, f in zip(nli_scores, feats)]

        out = []
        for (a, b), n, f, p in zip(pairs, nli_scores, feats, conflict_p):
            out.append(
                PairCandidate(
                    doc_i=a.id, doc_j=b.id, text_i=a.text, text_j=b.text,
                    nli=n, features=f,
                    is_conflict=p >= 0.5,
                    confidence=p if p >= 0.5 else 1.0 - p,
                    needs_escalation=self.low <= p <= self.high,
                )
            )
        return out

    def _conflict_column(self) -> int:
        assert self.head is not None
        for i, c in enumerate(self.head.classes_):
            if c in ("True", "conflict", "1", "true"):
                return i
        return len(self.head.classes_) - 1

    @staticmethod
    def _rule_score(nli: NLIScores, feats: PairFeatures) -> float:
        """Transparent fallback when no head has been trained.

        Weighted so that exception cues contribute even when NLI says
        NEUTRAL. That case is not an edge case here -- a general rule and its
        valid exception are not logically contradictory, so an NLI model
        correctly calls them neutral, and a detector keyed only on
        contradiction would miss the entire conditional class.
        """
        score = 0.6 * nli.conflict_signal
        score += 0.20 * min(1.0, feats.exception_cues_max / 2.0)
        score += 0.10 * min(1.0, feats.restriction_asymmetry / 2.0)
        score += 0.15 * feats.numeric_clash
        score += 0.10 * feats.year_clash
        # Passages about different things are not in conflict, however many
        # cue words they happen to contain.
        score *= 0.35 + 0.65 * min(1.0, feats.token_jaccard * 3.0)
        return float(min(1.0, max(0.0, score)))

    # -- training ------------------------------------------------------------- #

    def train_head(
        self,
        records: list[QueryRecord],
        *,
        config: HeadConfig | None = None,
    ) -> PairHead:
        """Fit the binary conflict head on labelled records.

        Labels come from each record's ``conflict_pairs``. A pair absent from
        that list is treated as a negative, which is correct for gold records
        (where only the annotated relationship is listed) and is why
        distractor instances are mandatory in the benchmark.
        """
        X, y = [], []
        for rec in records:
            labelled = {p.key: p.is_conflict for p in rec.conflict_pairs}
            for cand in self.score_pairs(rec.passages):
                X.append(cand.feature_vector())
                y.append(str(labelled.get(cand.key, False)))

        if not X:
            raise ValueError("no training pairs: every record had fewer than two passages")

        head = PairHead(config or HeadConfig(), feature_names=feature_names())
        head.fit(np.array(X, dtype=np.float64), y)
        self.head = head
        return head

    def load_head(self, path: str | Path) -> None:
        self.head = PairHead.load(path)

    @property
    def is_trained(self) -> bool:
        return self.head is not None
