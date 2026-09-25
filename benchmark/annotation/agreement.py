"""Cohen's kappa, computed per label axis and refused when it would be a lie.

The manual (§1) requires kappa reported **separately** for the five-class
conflict type and the four-way scope relation, never blended. The scope
relation is expected to be the harder of the two, and a blended figure hides
exactly that.

This module also enforces the precondition that makes kappa mean anything.
Kappa measures whether two *independent* judgements converge. Compute it from
two passes by the same annotator, or from a model's labels against that same
model's labels, and the number still renders -- it just no longer measures
agreement between observers. It measures self-consistency, which is not what
the word kappa claims in a paper.

So :func:`cohen_kappa` refuses rather than returning a number it cannot stand
behind. The guard is in code, not in a comment, for the same reason the
``tuned: false`` flag on the escalation band is in code: a convention that
lives only in someone's memory is one tired evening away from being violated,
and a fabricated kappa is unrecoverable once it is in a submission.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from metrics.stats import DEFAULT_ALPHA, Interval, bootstrap


class AgreementError(RuntimeError):
    """Raised when a kappa would not measure what kappa claims to measure."""


#: Annotator ids that denote a model rather than a person. Two of these, or one
#: of these against itself, cannot produce an inter-annotator statistic.
#:
#: ``"model-proposed"`` is the exact string ``benchmark.wp1_probe_set`` writes,
#: and it was missing here while only the underscored spelling was listed -- so
#: the guard would have let the project's own model-proposed labels through.
MODEL_ANNOTATORS = frozenset({
    "model", "model_proposed", "model-proposed", "modelproposed",
    "llm", "auto", "silver", "proposed", "gpt", "ai",
})

#: Conventional floor for a label a paper reports: Landis & Koch "substantial".
ACCEPTABLE_KAPPA = 0.61

#: Landis & Koch bands. Included because a bare kappa invites the reader to
#: supply their own threshold, and the bands are what the field actually uses.
_BANDS = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.01, "slight"),
    (-1.0, "poor (worse than chance)"),
]


def _band(k: float) -> str:
    for floor, name in _BANDS:
        if k >= floor:
            return name
    return "poor"


@dataclass(frozen=True)
class KappaResult:
    """Kappa on one label axis."""

    axis: str
    kappa: float
    observed_agreement: float
    expected_agreement: float
    n: int
    interval: Interval | None = None
    confusion: dict[tuple[str, str], int] | None = None
    degenerate: bool = False
    """Both annotators used a single category, so kappa is 0/0 and undefined.
    Not a result either way: the batch had nothing to disagree about."""

    @property
    def band(self) -> str:
        return "undefined (degenerate batch)" if self.degenerate else _band(self.kappa)

    @property
    def is_acceptable(self) -> bool:
        """Whether this axis clears the bar for bulk annotation to proceed.

        0.61 (substantial) is the conventional floor for a label a paper
        reports. Below it, the disagreements are telling you the manual is
        ambiguous, and annotating 300 more instances against an ambiguous
        manual just produces 300 more disagreements.

        A degenerate batch is never acceptable, however unanimous: it has not
        tested the distinction, so it cannot clear a bar about the distinction.
        """
        return not self.degenerate and self.kappa >= ACCEPTABLE_KAPPA

    def disagreements(self) -> list[tuple[tuple[str, str], int]]:
        """Off-diagonal cells, largest first. This is the actionable output.

        The kappa says whether there is a problem; the cells say where it is,
        and a single dominant cell usually means one paragraph of the manual
        needs rewriting rather than that the task is inherently hard.
        """
        if not self.confusion:
            return []
        off = {k: v for k, v in self.confusion.items() if k[0] != k[1]}
        return sorted(off.items(), key=lambda kv: -kv[1])

    def render(self) -> str:
        head = "undefined" if self.degenerate else f"{self.kappa:>7.3f}"
        lines = [
            f"{self.axis}",
            "-" * 74,
            f"  kappa                {head:>7}   ({self.band})",
            f"  observed agreement   {self.observed_agreement:>7.3f}",
            f"  expected by chance   {self.expected_agreement:>7.3f}",
            f"  instances            {self.n:>7,}",
        ]
        if self.interval and not self.degenerate:
            lines.append(f"  95% CI               [{self.interval.low:.3f}, "
                         f"{self.interval.high:.3f}]")

        if self.degenerate:
            lines += [
                "",
                "  Both annotators used ONE category for every instance, so kappa is",
                "  0/0 and undefined. Unanimity here is not evidence of agreement --",
                "  the batch contained nothing to disagree about. Re-draw the pilot",
                "  with a mix of types, including the distractors §5 requires.",
            ]
            return "\n".join(lines)

        # A point estimate above the bar with an interval reaching well below it
        # is not the same evidence as a point estimate above the bar on a large
        # batch, and the conventional "substantial" label hides the difference.
        # The pilot of 2026-09-25 returned 0.775 on 23 instances with an interval
        # from 0.395 -- "clear to proceed" on a range that includes "fair".
        if (self.is_acceptable and self.interval
                and self.interval.low < ACCEPTABLE_KAPPA):
            lines += [
                "",
                f"  CAUTION: the point estimate clears {ACCEPTABLE_KAPPA:.2f} but the interval",
                f"  reaches down to {self.interval.low:.3f}. On {self.n} instances this axis has not",
                "  been shown to be reliable, only shown not to be unreliable. Widen the",
                "  pilot on this axis, or treat the result as provisional.",
            ]

        if not self.is_acceptable:
            lines += [
                "",
                f"  BELOW {ACCEPTABLE_KAPPA:.2f}. Do not start bulk annotation on this axis yet.",
                "  The disagreements below are telling you the manual is ambiguous;",
                "  fix the manual and re-run the pilot rather than labelling 300",
                "  more instances against the same ambiguity.",
            ]
        dis = self.disagreements()
        if dis:
            lines += ["", "  disagreement cells (largest first)"]
            for (a, b), n in dis[:6]:
                lines.append(f"    {a:<22} vs {b:<22} {n:>4}")
            top = dis[0][1]
            total_off = sum(n for _, n in dis)
            if total_off and top / total_off >= 0.5 and total_off >= 4:
                lines += [
                    "",
                    f"    One cell carries {top / total_off:.0%} of the disagreement.",
                    "    That is usually one ambiguous paragraph in the manual, not an",
                    "    inherently hard task. Fix that paragraph first.",
                ]
        return "\n".join(lines)


def _check_independence(annotator_a: str, annotator_b: str) -> None:
    a, b = annotator_a.strip().lower(), annotator_b.strip().lower()

    if a == b:
        raise AgreementError(
            f"both label sets are attributed to {annotator_a!r}. Kappa measures "
            "agreement between two INDEPENDENT observers; computed over one "
            "observer's two passes it measures self-consistency, which is a "
            "different quantity that happens to share a formula. Recruit a "
            "second annotator."
        )

    offenders = {annotator_a for x in [a] if x in MODEL_ANNOTATORS}
    offenders |= {annotator_b for x in [b] if x in MODEL_ANNOTATORS}
    if offenders:
        raise AgreementError(
            f"{sorted(offenders)} denotes model-generated labels. An "
            "inter-annotator agreement statistic computed against model output "
            "is not an inter-annotator agreement statistic. Model labels are "
            "PROPOSALS to be adjudicated by a human, and the adjudicated result "
            "is what enters a kappa."
        )


def cohen_kappa(
    labels_a: dict[str, str],
    labels_b: dict[str, str],
    *,
    axis: str,
    annotator_a: str,
    annotator_b: str,
    with_interval: bool = True,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> KappaResult:
    """Cohen's kappa over the instances both annotators labelled.

    Parameters
    ----------
    labels_a, labels_b
        instance_id -> label. Only ids present in both are scored; an instance
        one annotator skipped is not a disagreement, it is missing data, and
        counting it as either would be wrong.
    annotator_a, annotator_b
        Real identities. Checked, not decorative -- see :func:`_check_independence`.

    Raises
    ------
    AgreementError
        If the two label sets could not constitute independent observation, or
        if there is no overlap to score.
    """
    _check_independence(annotator_a, annotator_b)

    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        raise AgreementError(
            "the two annotators have no instances in common. Kappa needs an "
            "overlapping subset -- assign both annotators the same pilot batch."
        )

    pairs = [(labels_a[i], labels_b[i]) for i in shared]
    kappa, po, pe = _kappa_from_pairs(pairs)
    confusion = Counter(pairs)

    used = {a for a, _ in pairs} | {b for _, b in pairs}
    degenerate = len(used) <= 1

    interval = None
    if with_interval and not degenerate and len(pairs) >= 2:
        # Resampling the instance pairs, which are the independent units here:
        # one instance contributes exactly one judgement per annotator.
        interval = bootstrap(
            pairs,
            lambda sample: _kappa_from_pairs(list(sample))[0],
            alpha=alpha,
            seed=seed,
        )

    return KappaResult(
        axis=axis, kappa=kappa, observed_agreement=po, expected_agreement=pe,
        n=len(pairs), interval=interval, confusion=dict(confusion),
        degenerate=degenerate,
    )


def _kappa_from_pairs(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Return ``(kappa, observed, expected)`` for paired labels."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0

    observed = sum(1 for a, b in pairs if a == b) / n

    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    categories = set(count_a) | set(count_b)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if expected >= 1.0:
        # Both annotators used exactly one category for everything, so kappa is
        # 0/0. This is the kappa paradox: agreement is unanimous and chance
        # agreement is also unanimous, so the statistic cannot distinguish
        # careful agreement from a batch with nothing to disagree about.
        # Reporting 1.0 would read as flawless annotation; reporting 0.0 would
        # read as failure. Neither is true, so the caller is told it is
        # degenerate and the number is withheld.
        return 0.0, observed, expected

    return (observed - expected) / (1.0 - expected), observed, expected
