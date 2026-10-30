"""VACS: the Variable-Alignment Challenge Set (proposal v8 section 7.4, plan v4 D-15).

QNFA's central claim is that **structure aligns passages that similarity cannot**. VACS
tests that claim directly and in isolation, with no conflict, no resolution and no composition
in the way: given two passages, do they govern the *same decision*?

Items are passage pairs, each labelled by people as

* ``equivalent`` -- the same decision, stated at the same level;
* ``a_broader`` / ``b_broader`` -- the same decision, one stated more generally
  (``subsumes``); and
* ``distinct`` -- different decisions.

The first three are *aligned*; the last is not. Methods are compared on aligned-versus-
distinct accuracy, because cosine similarity has no notion of direction.

**The quadrants are the point.** Two axes cross:

======================  ==========================  ==========================
                        same decision               different decision
======================  ==========================  ==========================
**high lexical overlap**  easy for everyone            *a trap for similarity*
**low lexical overlap**   *a trap for similarity*      easy for everyone
======================  ==========================  ==========================

A set sampled at random would be almost all easy cells, and every method would look fine. So
the sampler *oversamples the two off-diagonal quadrants*, where a rule and its carve-out
are written in different vocabulary (same decision, low overlap) and where two unrelated
fees share the word "fee" (different decision, high overlap). H-V predicts QNFA's gain is
largest exactly there.

**Built only from train and dev document groups.** Test documents never feed VACS, the
same leakage rule as everywhere else, and ``assert_no_test_documents`` enforces it.

Whether two passages share a decision is not known before a person labels them, so the
sampler uses an *evaluation-model proposal* only to *steer* sampling toward the quadrants. The
quadrant a pair is **reported** in comes from the human label and the measured overlap, never
from the proposal.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

ALIGNED = ("equivalent", "a_broader", "b_broader")
LABELS = (*ALIGNED, "distinct")

_STOP = frozenset("""a an the and or of to in on for with at by from is are was were be been it its this that these those
as not no you your we our they their will can may must should would could has have had do does did if then than there
which who whom whose what when where how also any all each other more most such only into over under about between per""".split())


class Quadrant(str, Enum):
    HI_SAME = "high_overlap/same"
    HI_DIFF = "high_overlap/different"
    LO_SAME = "low_overlap/same"
    LO_DIFF = "low_overlap/different"

    @property
    def off_diagonal(self) -> bool:
        return self in (Quadrant.HI_DIFF, Quadrant.LO_SAME)


def content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in _STOP}


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words: the *measured* axis of the quadrant."""
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


@dataclass(frozen=True)
class VacsItem:
    item_id: str
    text_a: str
    text_b: str
    doc_a: str
    doc_b: str
    overlap: float
    proposed_same: bool | None = None
    """The evaluation model's *proposal*, used only to steer sampling. Never a label."""
    label: str | None = None
    """The human label. ``None`` until labelled."""
    split: str = "train"

    def quadrant(self, overlap_threshold: float) -> Quadrant | None:
        """Where this item is *reported*: measured overlap x the human label. ``None`` if unlabelled."""
        if self.label is None:
            return None
        same = self.label in ALIGNED
        hi = self.overlap >= overlap_threshold
        return {(True, True): Quadrant.HI_SAME, (True, False): Quadrant.HI_DIFF,
                (False, True): Quadrant.LO_SAME, (False, False): Quadrant.LO_DIFF}[(hi, same)]

    @property
    def aligned(self) -> bool | None:
        return None if self.label is None else self.label in ALIGNED


def median_threshold(items: Iterable[VacsItem]) -> float:
    """The overlap that splits the pool into low and high halves. Computed from the data, then frozen."""
    xs = sorted(i.overlap for i in items)
    if not xs:
        raise ValueError("no items")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def assert_no_test_documents(items: Iterable[VacsItem]) -> None:
    bad = [i.item_id for i in items if i.split == "test"]
    if bad:
        raise ValueError(f"VACS may only be built from train/dev document groups; test items: {bad[:5]}")


def sample(
    pool: Sequence[VacsItem],
    *,
    n: int = 120,
    seed: int = 0,
    off_diagonal_share: float = 0.5,
    threshold: float | None = None,
) -> list[VacsItem]:
    """Draw ``n`` items, steering toward the off-diagonal quadrants.

    Uses the *proposed* decision-sharing to steer (it is all that exists before labelling) and the
    *measured* overlap for the other axis. ``off_diagonal_share`` of the draw targets the two trap
    quadrants, split evenly between them; the rest splits between the two easy ones. A quadrant
    that cannot supply its share is filled from the others, and the shortfall is not hidden: the
    caller can count what it got.
    """
    assert_no_test_documents(pool)
    if not 0.0 <= off_diagonal_share <= 1.0:
        raise ValueError("off_diagonal_share must be in [0, 1]")
    thr = median_threshold(pool) if threshold is None else threshold
    rng = random.Random(seed)

    def cell(i: VacsItem) -> Quadrant | None:
        if i.proposed_same is None:
            return None
        hi = i.overlap >= thr
        return {(True, True): Quadrant.HI_SAME, (True, False): Quadrant.HI_DIFF,
                (False, True): Quadrant.LO_SAME, (False, False): Quadrant.LO_DIFF}[(hi, i.proposed_same)]

    buckets: dict[Quadrant, list[VacsItem]] = {q: [] for q in Quadrant}
    for i in sorted(pool, key=lambda x: x.item_id):
        c = cell(i)
        if c is not None:
            buckets[c].append(i)
    for b in buckets.values():
        rng.shuffle(b)

    off = int(round(n * off_diagonal_share))
    target = {Quadrant.HI_DIFF: off // 2 + off % 2, Quadrant.LO_SAME: off // 2,
              Quadrant.HI_SAME: (n - off) // 2 + (n - off) % 2, Quadrant.LO_DIFF: (n - off) // 2}
    chosen: list[VacsItem] = []
    for q, k in target.items():
        chosen += buckets[q][:k]
        buckets[q] = buckets[q][k:]
    leftover = [i for q in (Quadrant.HI_DIFF, Quadrant.LO_SAME, Quadrant.HI_SAME, Quadrant.LO_DIFF) for i in buckets[q]]
    chosen += leftover[: max(0, n - len(chosen))]
    return sorted(chosen, key=lambda i: i.item_id)


def quadrant_counts(items: Iterable[VacsItem], threshold: float) -> dict[Quadrant, int]:
    out = {q: 0 for q in Quadrant}
    for i in items:
        q = i.quadrant(threshold)
        if q is not None:
            out[q] += 1
    return out


def load(path: Path) -> list[VacsItem]:
    return [VacsItem(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(items: Iterable[VacsItem], path: Path) -> None:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for i in items:
            fh.write(json.dumps(asdict(i), ensure_ascii=False) + "\n")


def with_labels(items: Iterable[VacsItem], labels: dict[str, str]) -> list[VacsItem]:
    """Attach human labels by item id; an unknown label is refused, a missing one leaves the item unlabelled."""
    bad = {k: v for k, v in labels.items() if v not in LABELS}
    if bad:
        raise ValueError(f"unknown VACS label(s): {bad}; choose from {LABELS}")
    from dataclasses import replace

    return [replace(i, label=labels.get(i.item_id, i.label)) for i in items]
