"""Evaluate alignment methods on VACS: does structure beat similarity? (H-V, gate G4)

Five methods answer one question per item -- *do these two passages govern the same
decision?* -- and are scored on aligned-versus-distinct accuracy, overall and **per quadrant**
(``benchmark.vacs``):

====================  =============================================================
``passage_cosine``    cosine of whole-passage embeddings (Contriever)
``variable_cosine``   cosine of the extracted decision-variable strings
``nli``               the NLI model's strongest relation (entailment or contradiction)
``llm_direct``        the model is simply asked
``qnfa``              signature unification of the extracted frames
====================  =============================================================

**The baselines are given every advantage the data allows.** Each similarity method's
threshold is *tuned on VACS itself* (the accuracy-maximising cut), which no deployed system
could do; ``qnfa`` has no threshold to tune. Any advantage that survives this cannot be put
down to a handicapped comparator. The cost is that the baselines' numbers are optimistic, and
they are labelled so.

**Abstention is wrong.** A method that cannot decide (QNFA with no usable frame, or only
``UNCERTAIN`` pairs) is scored *incorrect* and its abstention rate is reported beside the
accuracy. Counting abstentions as correct, or dropping them, would flatter exactly the method
that gives up most.

**H-V** predicts QNFA's gain over the similarity methods is largest in the two off-diagonal
quadrants. **G4** requires QNFA to beat ``variable_cosine`` on VACS overall (and the plan
adds the dev detection floor, which is measured elsewhere).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from benchmark.vacs import Quadrant, VacsItem, median_threshold
from detection.signature import Kind, unify

#: Per item: True (aligned), False (distinct), or None (the method abstained).
Method = Callable[[VacsItem], "bool | None"]
Embed = Callable[[Sequence[str]], np.ndarray]

OFF_DIAGONAL = (Quadrant.HI_DIFF, Quadrant.LO_SAME)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


def cosine_scores(embed: Embed, pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Cosine similarity for each pair, in one batched encode."""
    if not pairs:
        return []
    flat = [t for p in pairs for t in p]
    v = np.asarray(embed(flat), dtype=np.float64)
    a, b = v[0::2], v[1::2]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return [float(x) for x in np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)]


def best_threshold(scores: Sequence[float], aligned: Sequence[bool]) -> tuple[float, float]:
    """(threshold, accuracy) maximising accuracy: the *oracle* cut, which favours the baseline.

    Ties go to the lower threshold. With no items, ``(0.5, 0.0)``."""
    if not scores:
        return 0.5, 0.0
    best_acc, best_t = -1.0, 0.5
    for t in sorted(set(scores)):
        acc = sum((s >= t) == y for s, y in zip(scores, aligned)) / len(scores)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def threshold_method(name: str, scorer: Callable[[Sequence[VacsItem]], list[float]],
                     items: Sequence[VacsItem]) -> tuple[Method, float]:
    """A similarity method with its threshold tuned on ``items`` (labelled). Returns (method, threshold)."""
    labelled = [i for i in items if i.label is not None]
    scores = scorer(labelled)
    thr, _ = best_threshold(scores, [bool(i.aligned) for i in labelled])
    cache = dict(zip((i.item_id for i in labelled), scores))
    return (lambda item: cache[item.item_id] >= thr), thr


def qnfa_method(frames_for: Callable[[str, VacsItem], list]) -> Method:
    """Aligned iff any frame pair unifies as equivalent, subsumes or overlap; abstain if none can be decided.

    ``frames_for(which, item)`` returns the extracted frames of passage ``"a"`` or ``"b"``.
    """
    def predict(item: VacsItem) -> bool | None:
        fa, fb = frames_for("a", item), frames_for("b", item)
        if not fa or not fb:
            return None
        kinds = [unify(x.signature, y.signature).kind for x in fa for y in fb]
        if any(k in (Kind.EQUIVALENT, Kind.SUBSUMES, Kind.OVERLAP) for k in kinds):
            return True
        if all(k is Kind.UNCERTAIN for k in kinds):
            return None
        return False

    return predict


def decision_string(frame) -> str:
    """The frame's decision variable as text: typed and untyped slot values, in a fixed order."""
    s = frame.signature
    parts = [f"{k} {v}".replace("_", " ") for k, v in sorted(s.slots.items())]
    parts += [f"{k} {v}" for k, v in sorted(s.untyped_slots.items())]
    return " ; ".join(parts)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass
class MethodResult:
    name: str
    correct: int = 0
    n: int = 0
    abstained: int = 0
    by_quadrant: dict[Quadrant, list[int]] = field(default_factory=dict)   # quadrant -> [correct, n]
    threshold: float | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.n if self.n else 0.0

    def quadrant_accuracy(self, q: Quadrant) -> float | None:
        c, n = self.by_quadrant.get(q, [0, 0])
        return c / n if n else None

    @property
    def off_diagonal_accuracy(self) -> float | None:
        c = sum(self.by_quadrant.get(q, [0, 0])[0] for q in OFF_DIAGONAL)
        n = sum(self.by_quadrant.get(q, [0, 0])[1] for q in OFF_DIAGONAL)
        return c / n if n else None


def score(name: str, method: Method, items: Sequence[VacsItem], *, overlap_threshold: float,
          threshold: float | None = None) -> MethodResult:
    r = MethodResult(name, threshold=threshold)
    for it in items:
        if it.label is None:
            continue                                  # an unlabelled item has no right answer
        pred = method(it)
        ok = pred is not None and pred == it.aligned
        r.n += 1
        r.correct += ok
        r.abstained += pred is None                  # abstention is scored wrong AND reported
        q = it.quadrant(overlap_threshold)
        cell = r.by_quadrant.setdefault(q, [0, 0])
        cell[0] += ok
        cell[1] += 1
    return r


@dataclass
class VacsReport:
    results: dict[str, MethodResult]
    overlap_threshold: float
    n_labelled: int

    def gain(self, over: str, quadrant: Quadrant | None = None, *, method: str = "qnfa") -> float | None:
        """QNFA's accuracy minus another method's, overall or in one quadrant."""
        a, b = self.results[method], self.results[over]
        if quadrant is None:
            return a.accuracy - b.accuracy
        qa, qb = a.quadrant_accuracy(quadrant), b.quadrant_accuracy(quadrant)
        return None if qa is None or qb is None else qa - qb

    def g4_passes(self, baseline: str = "variable_cosine") -> bool:
        """QNFA beats the variable-string baseline overall. (The dev-F1 half of G4 is measured elsewhere.)"""
        return self.gain(baseline) is not None and self.gain(baseline) > 0

    def h_v_holds(self, baselines: Sequence[str] = ("passage_cosine", "variable_cosine")) -> bool | None:
        """Is QNFA's gain over each similarity baseline larger off the diagonal than on it?"""
        verdicts = []
        for b in baselines:
            off = [self.gain(b, q) for q in OFF_DIAGONAL]
            on = [self.gain(b, q) for q in (Quadrant.HI_SAME, Quadrant.LO_DIFF)]
            if any(x is None for x in off + on):
                return None
            verdicts.append(sum(off) / len(off) > sum(on) / len(on))
        return all(verdicts)

    def render(self) -> str:
        qs = list(Quadrant)
        head = f"  {'method':<17}{'acc':>7}{'abst.':>7}" + "".join(f"{q.value:>25}" for q in qs)
        lines = [f"VACS ({self.n_labelled} labelled items; overlap split at {self.overlap_threshold:.3f})",
                 "=" * len(head), head, "  " + "-" * (len(head) - 2)]
        for r in self.results.values():
            cells = "".join(f"{('-' if r.quadrant_accuracy(q) is None else f'{r.quadrant_accuracy(q):.2f}'):>25}"
                            for q in qs)
            thr = "" if r.threshold is None else f"   (threshold {r.threshold:.3f}, tuned ON VACS)"
            lines.append(f"  {r.name:<17}{r.accuracy:>7.3f}{r.abstain_rate:>7.2f}{cells}{thr}")
        if "qnfa" in self.results:
            lines += ["", f"  G4 (QNFA beats variable_cosine overall): {'PASS' if self.g4_passes() else 'FAIL'}"]
            hv = self.h_v_holds()
            lines.append(f"  H-V (gain is larger off the diagonal): "
                         f"{'holds' if hv else 'does not hold' if hv is False else 'not evaluable'}")
        return "\n".join(lines)


def evaluate(methods: Mapping[str, Method], items: Sequence[VacsItem], *,
             thresholds: Mapping[str, float] | None = None, overlap_threshold: float | None = None) -> VacsReport:
    labelled = [i for i in items if i.label is not None]
    if not labelled:
        raise ValueError("VACS has no labelled items")
    thr = median_threshold(labelled) if overlap_threshold is None else overlap_threshold
    return VacsReport({n: score(n, m, labelled, overlap_threshold=thr, threshold=(thresholds or {}).get(n))
                       for n, m in methods.items()}, thr, len(labelled))
