"""Storage for annotation passes, kept blind by construction.

Each annotator writes to their own JSONL file and reads only their own. That
separation is not filing tidiness -- it is the mechanism that makes the kappa
in :mod:`benchmark.annotation.agreement` mean anything. An annotator who can
see the other's label before committing their own is anchored by it, and the
resulting agreement measures the anchoring rather than the labelling.

The tool therefore cannot show one annotator another's labels during a pass.
:meth:`AnnotationStore.load` takes a single annotator id and returns only that
annotator's records; comparing two passes happens in a separate step, after
both are sealed.

A pass is **sealed** once it has been used to compute agreement. Editing a
sealed pass would let an annotator revise toward the other's answers after
seeing the disagreements, which is exactly the contamination the blind pass
exists to prevent. Reconciliation after discussion is a real and expected part
of the protocol -- it just gets recorded as an adjudication, with both original
labels preserved, rather than by overwriting history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from contract.models import ConflictType, ScopeRelation

DEFAULT_DIR = Path("benchmark/data/annotations")

#: Written into the pass file when an annotator finishes a batch. The presence
#: of this file is what marks a pass sealed.
SEAL_SUFFIX = ".sealed"


class AnnotationError(RuntimeError):
    """Raised when an operation would corrupt the blind-pass guarantee."""


@dataclass
class LabelRecord:
    """One annotator's judgement on one instance.

    Deliberately narrower than :class:`~contract.gold.GoldInstance`: this is
    what a single pass produces, before reconciliation. The full gold instance
    is assembled from two of these plus an adjudication.
    """

    instance_id: str
    annotator: str
    conflict_type: str
    scope_relation: str | None = None
    is_distractor: bool = False
    #: Set when the annotator invoked the manual's escalation clause (§4) rather
    #: than guessing. Tracked because an escalated case and a confidently wrong
    #: case look identical in a confusion matrix and are completely different
    #: problems.
    escalated: bool = False
    notes: str | None = None
    seconds_spent: float | None = None
    timestamp: str = ""

    #: Where the starting point came from. ``"blank"`` means the annotator
    #: judged from the passages alone. ``"model"`` means they adjudicated a
    #: model-proposed label, which is faster but carries automation bias and
    #: must be disclosed -- see :mod:`benchmark.annotation.adjudicate`.
    basis: str = "blank"

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Validate against the frozen enums rather than accepting free text: a
        # typo'd label silently becomes its own category and deflates kappa.
        ConflictType(self.conflict_type)
        if self.scope_relation is not None:
            ScopeRelation(self.scope_relation)
        if (self.conflict_type == ConflictType.CONDITIONAL.value
                and not self.scope_relation and not self.escalated):
            raise AnnotationError(
                f"{self.instance_id}: a conditional label needs a scope relation"
            )
        # An escalated conditional is allowed to carry no relation, and that is
        # the point of the exemption. The manual (§4) says an annotator who
        # cannot decide should escalate rather than guess; if the record format
        # demanded a relation anyway, the tool would force the guess the manual
        # forbids, and the corpus would gain a coin-flip wearing a real label.
        # The reviewer fills it in later.
        if self.conflict_type == ConflictType.NO_CONFLICT.value and self.scope_relation:
            raise AnnotationError(
                f"{self.instance_id}: a no_conflict label must not carry a scope relation"
            )

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id, "annotator": self.annotator,
            "conflict_type": self.conflict_type, "scope_relation": self.scope_relation,
            "is_distractor": self.is_distractor, "escalated": self.escalated,
            "notes": self.notes, "seconds_spent": self.seconds_spent,
            "timestamp": self.timestamp, "basis": self.basis,
        }

    @classmethod
    def from_json(cls, obj: dict) -> LabelRecord:
        return cls(**{k: obj.get(k) for k in (
            "instance_id", "annotator", "conflict_type", "scope_relation",
            "is_distractor", "escalated", "notes", "seconds_spent",
            "timestamp", "basis") if k in obj})


@dataclass
class AnnotationStore:
    """Per-annotator JSONL files under one directory."""

    root: Path = field(default_factory=lambda: DEFAULT_DIR)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------- #

    def path_for(self, annotator: str) -> Path:
        safe = "".join(c for c in annotator.lower() if c.isalnum() or c in "-_")
        if not safe:
            raise AnnotationError(f"unusable annotator id {annotator!r}")
        return self.root / f"pass_{safe}.jsonl"

    def is_sealed(self, annotator: str) -> bool:
        return self.path_for(annotator).with_suffix(".jsonl" + SEAL_SUFFIX).exists()

    def seal(self, annotator: str) -> None:
        p = self.path_for(annotator).with_suffix(".jsonl" + SEAL_SUFFIX)
        p.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     encoding="utf-8")

    # -- reading and writing --------------------------------------------------- #

    def append(self, record: LabelRecord) -> None:
        if self.is_sealed(record.annotator):
            raise AnnotationError(
                f"{record.annotator}'s pass is sealed. It has already been compared "
                "against another annotator's, so editing it now would let a label be "
                "revised after seeing the disagreements -- which is the contamination "
                "the blind pass prevents. Record the change as an adjudication "
                "instead: both original labels are preserved and the reconciliation "
                "is visible."
            )
        path = self.path_for(record.annotator)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    def load(self, annotator: str) -> list[LabelRecord]:
        """Read ONE annotator's pass. There is deliberately no 'load everything'.

        A function that returned every annotator's labels at once would be the
        obvious thing to call from the annotation UI, and calling it there is
        precisely what breaks blindness.
        """
        path = self.path_for(annotator)
        if not path.exists():
            return []
        out: list[LabelRecord] = []
        seen: dict[str, LabelRecord] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = LabelRecord.from_json(json.loads(line))
            # Append-only file, so a re-labelled instance appears twice. The
            # last write wins, and the earlier one stays on disk as history.
            seen[rec.instance_id] = rec
        out = list(seen.values())
        return out

    def annotators(self) -> list[str]:
        """Who has a pass on disk. Ids only -- never their labels."""
        return sorted(p.stem.removeprefix("pass_") for p in self.root.glob("pass_*.jsonl"))

    def labels(self, annotator: str, axis: str) -> dict[str, str]:
        """``instance_id -> label`` on one axis, for the agreement computation.

        Instances the annotator left unlabelled on this axis are omitted rather
        than defaulted. A scope relation is only defined for a conditional
        instance, so defaulting the others to a placeholder would manufacture
        agreement on instances where the question was never asked.
        """
        out: dict[str, str] = {}
        for rec in self.load(annotator):
            if axis == "conflict_type":
                out[rec.instance_id] = rec.conflict_type
            elif axis == "scope_relation":
                if rec.scope_relation:
                    out[rec.instance_id] = rec.scope_relation
            else:
                raise AnnotationError(f"unknown axis {axis!r}")
        return out

    def progress(self, annotator: str, total: int | None = None) -> str:
        recs = self.load(annotator)
        done = len(recs)
        esc = sum(1 for r in recs if r.escalated)
        timed = [r.seconds_spent for r in recs if r.seconds_spent]
        lines = [f"  {annotator:<16} {done:>5} labelled"]
        if total:
            lines[0] += f" of {total} ({done / total:.0%})"
        if esc:
            lines.append(f"  {'':<16} {esc:>5} escalated to the third reviewer")
        if timed:
            median = sorted(timed)[len(timed) // 2]
            lines.append(f"  {'':<16} {median:>5.0f}s median per instance")
            if total and done < total:
                remaining = (total - done) * median / 3600
                lines.append(f"  {'':<16} {remaining:>5.1f}h remaining at that pace")
        if self.is_sealed(annotator):
            lines.append(f"  {'':<16}       SEALED")
        return "\n".join(lines)
