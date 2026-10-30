"""E2: detecting conditional conflicts, and the H2 comparison (proposal v8 sections 7.4, 10).

Every detector is reduced to the same thing -- a score in [0, 1] per instance,
"how likely is it that these passages contain a rule and a valid exception to
it?" -- and judged the same way:

* the **operating point is fixed on dev** (the threshold that maximises F1 there)
  and then applied unchanged to the test set, so no detector gets to tune itself on
  the data it is scored on;
* **precision, recall and F1** at that point, plus average precision (a
  threshold-free ranking summary);
* sampling uncertainty by resampling **instances**.

Detectors
---------

======================  ============================================================
``D0``                  the pilot's stage-1 rule: NLI conflict signal, exception cues,
                        numeric clash -- **and** its lexical-topicality gate
``D0-nogate``           identical, from the same NLI output, with only the gate removed
``D1``                  NLI contradiction/neutral signal alone
``D3``                  zero-shot LLM: "is there a general rule and a valid exception?"
``D5``                  QNFA: frames, signature unification, typed scope algebra
======================  ============================================================

``D0`` against ``D0-nogate`` is the measurement behind the paper's claim about *why* the pilot
detector recovered 2 of 67: the two differ in nothing but the multiplier
``0.35 + 0.65 * min(1, jaccard * 3)``, which scales a pair's score down in proportion to how little
vocabulary its two passages share -- exactly the property of a rule and a carve-out written by
different teams.

H2: D5's F1 exceeds the best non-D5 detector (chosen on **dev**), with a one-sided 95% BCa lower
bound against the pre-registered 15-point minimum effect of interest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Mapping, Sequence

from benchmark.relabel_v7 import V7Label, relabel
from contract.gold import GoldInstance
from detection.stage1 import Stage1Filter, enumerate_pairs
from experiments.analysis import bootstrap_p_at_most
from metrics.cluster import bca_difference, one_sided_lower

Detector = Callable[[GoldInstance], float]

MIN_EFFECT = 0.15


def is_positive(inst: GoldInstance) -> bool:
    """Gold: does this instance contain a conditional conflict (a refinement)?"""
    return relabel(inst).label is V7Label.CONDITIONAL_CONFLICT


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def prf(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> dict[str, float]:
    tp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y)
    fp = sum(1 for s, y in zip(scores, labels) if s >= threshold and not y)
    fn = sum(1 for s, y in zip(scores, labels) if s < threshold and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "threshold": threshold}


def f1_at(threshold: float):
    """A statistic over ``(score, label)`` pairs, for resampling."""
    def stat(units: Sequence[tuple[float, bool]]) -> float:
        return prf([s for s, _ in units], [y for _, y in units], threshold)["f1"]

    return stat


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the precision-recall curve by the step-wise (average precision) rule."""
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    total = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            total += tp / rank
    return total / n_pos


def choose_threshold(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """The threshold maximising F1 on dev. Ties go to the *higher* threshold (the more conservative
    operating point). With no positive instance there is nothing to tune and 0.5 is returned."""
    if not any(labels):
        return 0.5
    best_f1, best_t = -1.0, 0.5
    for t in sorted(set(scores)):
        f = prf(scores, labels, t)["f1"]
        if f >= best_f1:
            best_f1, best_t = f, t
    return best_t


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #


def legacy_detectors(stage1: Stage1Filter) -> tuple[Detector, Detector]:
    """(D0, D0-nogate), computed from the SAME NLI output so the gate is the only difference."""
    from detection.features import extract_pair_features

    cache: dict[str, tuple[float, float]] = {}

    def both(inst: GoldInstance) -> tuple[float, float]:
        if inst.instance_id not in cache:
            pairs = enumerate_pairs(list(inst.passages))
            if not pairs:
                cache[inst.instance_id] = (0.0, 0.0)
            else:
                nli = stage1.nli.score_batch([(a.text, b.text) for a, b in pairs])
                gated, ungated = [], []
                for (a, b), n in zip(pairs, nli):
                    f = extract_pair_features(a.text, b.text)
                    g = stage1._rule_score(n, f)
                    gate = 0.35 + 0.65 * min(1.0, f.token_jaccard * 3.0)
                    gated.append(g)
                    ungated.append(min(1.0, g / gate) if gate > 0 else g)   # undo only the final multiplier
            cache[inst.instance_id] = (max(gated), max(ungated)) if pairs else (0.0, 0.0)
        return cache[inst.instance_id]

    return (lambda i: both(i)[0]), (lambda i: both(i)[1])


def nli_detector(nli) -> Detector:
    """D1: the strongest NLI conflict signal over the passage pairs. No cues, no gate."""
    def score(inst: GoldInstance) -> float:
        pairs = enumerate_pairs(list(inst.passages))
        if not pairs:
            return 0.0
        return max(min(1.0, s.conflict_signal) for s in nli.score_batch([(a.text, b.text) for a, b in pairs]))

    return score


LLM_SYSTEM = (
    "You decide whether a set of passages contains a GENERAL RULE and a VALID EXCEPTION to it: a narrower "
    "group of people for whom the outcome differs. Two passages that merely disagree about the same "
    "group, or that talk about different things, do not count. Return ONLY JSON: "
    '{"conditional": true or false, "confidence": a number from 0 to 1}.'
)


def llm_pair_detector(client, tier) -> Detector:
    """D3: zero-shot. The model gets the passages and one question, no structure."""
    import json

    def score(inst: GoldInstance) -> float:
        body = "\n\n".join(f"[{p.id}] {p.text}" for p in inst.passages)
        res = client.complete(messages=[{"role": "user", "content": f"Question: {inst.query}\n\nPassages:\n{body}"}],
                              system=LLM_SYSTEM, tier=tier, step="det_d3_llm_pair",
                              response_format={"type": "json_object"})
        try:
            d = json.loads(res.text)
            conf = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
            return conf if bool(d.get("conditional")) else 1.0 - conf
        except (ValueError, TypeError, AttributeError):
            return 0.0

    return score


def qnfa_detector(extractor_for, qnfa=None) -> Detector:
    """D5: frames -> signature unification -> typed scope algebra. Score = strongest conditional pair."""
    from detection.qnfa import QNFA

    q = qnfa or QNFA()

    def score(inst: GoldInstance) -> float:
        ext = extractor_for(inst).extract(inst.query, inst.passages)
        res = q.detect_frames(ext.frames)
        return max((p.score for p in res.conditional), default=0.0)

    return score


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class DetectorResult:
    name: str
    threshold: float
    dev: dict[str, float]
    test: dict[str, float]
    average_precision: float
    test_scores: list[float] = field(default_factory=list)


def evaluate(detectors: Mapping[str, Detector], dev: Sequence[GoldInstance], test: Sequence[GoldInstance]
             ) -> dict[str, DetectorResult]:
    dev_y = [is_positive(i) for i in dev]
    test_y = [is_positive(i) for i in test]
    out: dict[str, DetectorResult] = {}
    for name, det in detectors.items():
        ds, ts = [det(i) for i in dev], [det(i) for i in test]
        thr = choose_threshold(ds, dev_y)
        out[name] = DetectorResult(name, thr, prf(ds, dev_y, thr), prf(ts, test_y, thr),
                                   average_precision(ts, test_y), ts)
    return out


@dataclass
class H2Result:
    d5: str
    best_baseline: str
    diff: float
    ci_low: float
    ci_high: float
    lower_bound_95: float
    p_at_most_min_effect: float
    min_effect: float
    n_test: int
    n_positive: int

    @property
    def clears_minimum_effect(self) -> bool:
        return self.lower_bound_95 >= self.min_effect

    def render(self) -> str:
        return (f"H2: {self.d5} F1 minus {self.best_baseline} F1 = {self.diff:+.3f} "
                f"(95% BCa [{self.ci_low:+.3f}, {self.ci_high:+.3f}]; one-sided 95% lower bound "
                f"{self.lower_bound_95:+.3f} vs the {self.min_effect:.2f} minimum effect: "
                f"{'CLEARED' if self.clears_minimum_effect else 'not cleared'}; "
                f"n={self.n_test}, {self.n_positive} positive)")


def h2(results: Mapping[str, DetectorResult], test: Sequence[GoldInstance], *, d5: str = "D5",
       baselines: Sequence[str] | None = None, min_effect: float = MIN_EFFECT, seed: int = 0,
       n_resamples: int = 4000) -> H2Result:
    """D5 against the best non-D5 detector, **chosen on dev F1** -- never on test."""
    names = [b for b in (baselines or [n for n in results if n != d5])]
    if d5 not in results or not names:
        raise KeyError("H2 needs D5 and at least one baseline")
    best = max(names, key=lambda n: (results[n].dev["f1"], n))
    y = [is_positive(i) for i in test]
    a, b = results[d5], results[best]
    units = [(a.test_scores[k], b.test_scores[k], y[k]) for k in range(len(test))]
    stat_a = lambda u: f1_at(a.threshold)([(s, l) for s, _, l in u])  # noqa: E731
    stat_b = lambda u: f1_at(b.threshold)([(s, l) for _, s, l in u])  # noqa: E731
    diff = lambda u: stat_a(u) - stat_b(u)  # noqa: E731
    iv = bca_difference(units, stat_a, stat_b, n_resamples=n_resamples, seed=seed)
    lb = one_sided_lower(units, diff, n_resamples=n_resamples, seed=seed)
    p = bootstrap_p_at_most(units, diff, min_effect, n_resamples=n_resamples, seed=seed)
    return H2Result(d5, best, stat_a(units) - stat_b(units), iv.low, iv.high, lb, p, min_effect,
                    len(test), sum(y))


def render(results: Mapping[str, DetectorResult]) -> str:
    lines = ["Detection of conditional conflicts (operating point fixed on dev)", "=" * 78,
             f"  {'detector':<12}{'thr':>7}{'dev F1':>9}{'P':>8}{'R':>8}{'F1':>8}{'AP':>8}", "  " + "-" * 60]
    for r in results.values():
        lines.append(f"  {r.name:<12}{r.threshold:>7.3f}{r.dev['f1']:>9.3f}{r.test['precision']:>8.3f}"
                     f"{r.test['recall']:>8.3f}{r.test['f1']:>8.3f}{r.average_precision:>8.3f}")
    return "\n".join(lines)
