"""A3 variant (i): lexical cues + NLI features.

The inspectable design. Cheapest to build, and the only one of the three whose
decisions can be read off a coefficient table -- which is the point of
including it. If it matches the fine-tuned transformer on the
factual/conditional cell, that is a result worth reporting: it would mean the
distinction is carried by surface cues a human can name, not by learned
representation.

An untrained rule-based mode is available so the variant works before any
labels exist. It is a fallback, flagged as such, and never the reported system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contract.models import ConflictType, QueryRecord, ScopeRelation
from detection.features import PairFeatures
from detection.head import HeadConfig, PairHead
from detection.stage1 import PairCandidate, Stage1Filter, feature_names
from detection.variants.base import DetectorVariant, labels_from_record


class LexicalNLIDetector(DetectorVariant):
    name = "lexical_nli"
    description = "NLI probabilities + hand-authored exception/restriction/specificity cues"

    def __init__(
        self,
        *,
        stage1: Stage1Filter | None = None,
        head: PairHead | None = None,
        relation_head: PairHead | None = None,
        config: HeadConfig | None = None,
    ):
        self.stage1 = stage1
        self.head = head
        self.relation_head = relation_head
        self.config = config or HeadConfig()

    @property
    def requires_training(self) -> bool:
        return False  # falls back to rules

    @property
    def is_trained(self) -> bool:
        return self.head is not None

    # -- classification -------------------------------------------------------- #

    def classify(self, candidate: PairCandidate) -> tuple[ConflictType, ScopeRelation | None]:
        if self.head is None:
            return self._rule_classify(candidate)

        X = np.array([candidate.feature_vector()], dtype=np.float64)
        ctype = ConflictType(self.head.predict(X)[0])

        if ctype is not ConflictType.CONDITIONAL:
            return ctype, None

        if self.relation_head is not None:
            return ctype, ScopeRelation(self.relation_head.predict(X)[0])

        # A conditional label with no relation head is unroutable. Fall back to
        # the rule, which always yields one, rather than emitting a pair the
        # contract will reject.
        return ctype, self._rule_relation(candidate.features)

    def _rule_classify(self, cand: PairCandidate) -> tuple[ConflictType, ScopeRelation | None]:
        """Transparent rules, used before training.

        Order matters. Temporal and opinion are checked first because their
        cues are distinctive; the factual/conditional split is checked last
        and is the hard one.
        """
        f = cand.features

        if f.temporal_cues_max >= 1 and (f.year_clash or f.numeric_clash):
            return ConflictType.TEMPORAL, None
        if f.opinion_cues_max >= 1:
            return ConflictType.OPINION, None

        # The load-bearing test. An exception cue OR a clear asymmetry in how
        # narrowly the two passages are scoped says "both true, different
        # scopes" rather than "one is wrong".
        scoped = f.exception_cues_max >= 1 or f.restriction_asymmetry >= 1
        specific = f.specificity_asymmetry >= 1

        if scoped and (specific or f.exception_cues_max >= 1):
            return ConflictType.CONDITIONAL, self._rule_relation(f)
        if f.numeric_clash or f.token_jaccard > 0.45:
            return ConflictType.FACTUAL, None
        return ConflictType.NO_CONFLICT, None

    @staticmethod
    def _rule_relation(f: PairFeatures) -> ScopeRelation:
        """Guess the four-way relation from surface cues alone.

        Deliberately crude. The real determination is A4's, which decides
        applicability and outcome agreement separately; this exists only so an
        untrained variant still emits a routable label.
        """
        if f.token_jaccard < 0.2:
            return ScopeRelation.DISJOINT
        # Overlapping scope with no outcome disagreement showing up as a
        # numeric or polarity clash is more likely a restatement than an
        # exception.
        if not f.numeric_clash and f.exception_cues_max == 0:
            return ScopeRelation.REDUNDANT
        if f.restriction_asymmetry >= 1 or f.specificity_asymmetry >= 2:
            return ScopeRelation.REFINEMENT
        return ScopeRelation.OPPOSED

    # -- training -------------------------------------------------------------- #

    def fit(self, records: list[QueryRecord]) -> LexicalNLIDetector:
        if self.stage1 is None:
            raise ValueError("LexicalNLIDetector needs a Stage1Filter to extract features")

        X, y_type, X_rel, y_rel = [], [], [], []
        for rec in records:
            gold = labels_from_record(rec)
            for cand in self.stage1.score_pairs(rec.passages):
                if cand.key not in gold:
                    continue
                ctype, relation = gold[cand.key]
                vec = cand.feature_vector()
                X.append(vec)
                y_type.append(ctype.value)
                if relation is not None:
                    X_rel.append(vec)
                    y_rel.append(relation.value)

        if not X:
            raise ValueError("no labelled pairs found in the training records")

        names = feature_names()
        self.head = PairHead(self.config, feature_names=names).fit(np.array(X), y_type)

        # The relation head needs at least two relation classes present. With a
        # small or skewed fixture it may not train, and falling back to the
        # rule is better than refusing to run.
        if len(set(y_rel)) >= 2:
            self.relation_head = PairHead(self.config, feature_names=names).fit(
                np.array(X_rel), y_rel
            )
        return self

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if self.head is not None:
            self.head.save(p.with_name(p.stem + "_type.pkl"))
        if self.relation_head is not None:
            self.relation_head.save(p.with_name(p.stem + "_relation.pkl"))

    def explain(self, n: int = 8) -> dict[str, list[tuple[str, float]]]:
        """Per-class top features. The variant's whole selling point.

        This is what lets the paper say which cues drive the conditional
        decision, rather than only that something does.
        """
        if self.head is None:
            return {}
        return {cls: self.head.top_features(cls, n) for cls in self.head.classes_}
