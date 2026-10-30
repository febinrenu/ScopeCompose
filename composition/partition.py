"""Partition composition over typed constraints (proposal v8 Propositions 2 and 3).

The pilot operator routes on a four-way relation label between two passages.
That is the wrong unit once there are three or more frames, and it makes
correctness depend on a *label* some detector got right. This module composes
from the frames' **applicability sets** instead, so that nothing a detector
decided can delete anything.

The construction
----------------

A frame's applicability is a conjunction of per-attribute constraints over a
case space U. For frames ``F = {f_1 .. f_n}`` every choice of *inside* or
*outside* for each frame (a **sign vector**) names a candidate region::

    R(s) = AND_{i inside} A_i   AND   AND_{i outside} NOT A_i

The regions partition U by construction. Each is then:

* dropped if **empty** (the satisfiability test below -- exact, not a heuristic);
* **uncovered** if no frame contains it (no source speaks to those people);
* **assigned** if the frames containing it have a unique ``subset``-minimal
  element (specificity), or all minimal elements agree on the outcome;
* **contested** otherwise: both sources are reported, neither is silently chosen.

Satisfiability
--------------

``NOT A_i`` is a *disjunction* over the attributes ``A_i`` constrains, so a
region is non-empty iff some choice of one *witness attribute* per outside frame
leaves every attribute's allowed values non-empty. That is decidable per
attribute because the domains are:

* **boolean** -- exactly ``{true, false}``;
* **categorical / tiered** -- *open*: a frame naming ``{premium, platinum}``
  never exhausts the domain, so "not premium" is satisfiable. (Treating the
  listed values as the whole domain would invent an empty region and silently
  delete the default's own scope.)
* **numeric** -- closed intervals over the reals.

Frames whose own constraints are contradictory are **discarded and logged**.
Frames with a free-text constraint cannot be reasoned over; each is reported
verbatim as **indeterminate** and no outcome is inferred for the overlap.

Proposition 3 (no deletion)
---------------------------

Every typed, non-discarded frame is *accounted for* in the output: it either
governs at least one region, or sits in :attr:`Partition.overridden` together
with the frames that override it. Nothing is dropped without a record, and no
relation label is consulted, so a wrong label cannot cause it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Literal

from contract.gold import Applicability, AttributeKind, Branch, ScopeAttribute
from metrics.values import values_match
from scope.attributes import SetRelation, compare

#: Exhaustive partition is exponential in the number of frames. Real queries
#: carry 2-4; beyond this the caller must reduce, and is told so rather than
#: handed a truncated answer.
MAX_FRAMES = 8


class TooManyFrames(ValueError):
    pass


@dataclass(frozen=True)
class Piece:
    """One frame, reduced to what composition needs."""

    piece_id: str
    applicability: Applicability
    outcome: str
    passage_id: str | None = None


def pieces_from_branches(branches: Iterable[Branch]) -> list[Piece]:
    return [Piece(b.branch_id, b.applicability, b.outcome, b.supporting_passage) for b in branches]


# --------------------------------------------------------------------------- #
# Typed constraints and exact satisfiability
# --------------------------------------------------------------------------- #

_BOOL_DOMAIN = frozenset({"true", "false"})
_Interval = tuple[float, float]


def _norm_attr(a: ScopeAttribute) -> tuple[str, str, object]:
    """(name, family, payload) with family in {'set', 'bool', 'num'}."""
    name = a.name.strip().lower()
    if a.kind in (AttributeKind.CATEGORICAL, AttributeKind.ORDERED_TIER):
        return name, "set", frozenset(v.strip().lower() for v in (a.values or []))
    if a.kind is AttributeKind.BOOLEAN:
        return name, "bool", frozenset({str(a.boolean_value).lower()})
    if a.kind is AttributeKind.NUMERIC_RANGE:
        lo = a.min_value if a.min_value is not None else -math.inf
        hi = a.max_value if a.max_value is not None else math.inf
        return name, "num", (lo, hi)
    raise ValueError("free text has no typed form")


def _is_typed(app: Applicability) -> bool:
    if app.is_default:
        return True
    return bool(app.attributes) and all(a.kind is not AttributeKind.FREE_TEXT for a in app.attributes)


def _typed_constraints(app: Applicability) -> dict[str, tuple[str, object]]:
    """attribute -> (family, payload). A repeated attribute is intersected."""
    out: dict[str, tuple[str, object]] = {}
    for a in app.attributes:
        name, fam, payload = _norm_attr(a)
        if name in out:
            ofam, opay = out[name]
            if ofam != fam:
                raise ValueError(f"attribute {name!r} is used with two kinds")
            payload = _intersect(fam, opay, payload)
        out[name] = (fam, payload)
    return out


def _intersect(fam: str, a: object, b: object) -> object:
    if fam == "num":
        (alo, ahi), (blo, bhi) = a, b  # type: ignore[misc]
        return (max(alo, blo), min(ahi, bhi))
    return a & b  # type: ignore[operator]


def _num_nonempty(positive: _Interval, excluded: list[_Interval]) -> bool:
    """Is positive minus the union of closed intervals non-empty (over the reals)?"""
    plo, phi = positive
    if plo > phi:
        return False
    merged: list[list[float]] = []
    for lo, hi in sorted(e for e in excluded if e[0] <= e[1]):
        if merged and lo <= merged[-1][1]:       # closed intervals: touching ones merge
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    covered = any(m_lo <= plo and phi <= m_hi for m_lo, m_hi in merged)
    return not covered


def _attr_feasible(fam: str, positive: object | None, excluded: list[object]) -> bool:
    if fam == "num":
        return _num_nonempty(positive if positive is not None else (-math.inf, math.inf),
                             excluded)  # type: ignore[arg-type]
    pos = positive if positive is not None else (_BOOL_DOMAIN if fam == "bool" else None)
    if pos is None:
        return True            # open-world categorical: some unmentioned value always exists
    remaining = set(pos)       # type: ignore[arg-type]
    for e in excluded:
        remaining -= e         # type: ignore[operator]
    return bool(remaining)


def satisfiable(inside: list[Applicability], outside: list[Applicability]) -> bool:
    """Is there a case in every ``inside`` set and in no ``outside`` set? Exact."""
    positives: dict[str, tuple[str, object]] = {}
    for app in inside:
        if app.is_default:
            continue
        for name, (fam, payload) in _typed_constraints(app).items():
            if name in positives:
                pfam, ppay = positives[name]
                if pfam != fam:
                    return False
                payload = _intersect(fam, ppay, payload)
            positives[name] = (fam, payload)

    # Every attribute must be non-empty on its own even with nothing excluded.
    for fam, payload in positives.values():
        if not _attr_feasible(fam, payload, []):
            return False

    outs = [_typed_constraints(a) for a in outside]
    if any(not o for o in outs):
        return False           # NOT(U) is empty: an outside frame that is the default

    for witness in product(*[list(o) for o in outs]):
        exclusions: dict[str, list[object]] = {}
        fam_of: dict[str, str] = {n: f for n, (f, _) in positives.items()}
        for cons, w in zip(outs, witness):
            fam, payload = cons[w]
            if fam_of.setdefault(w, fam) != fam:
                break          # an attribute used with two kinds: this choice is meaningless
            exclusions.setdefault(w, []).append(payload)
        else:
            if all(_attr_feasible(fam_of[a], positives.get(a, (fam_of[a], None))[1], ex)
                   for a, ex in exclusions.items()):
                return True
    return False if outs else True


def _contains(inner: Applicability, outer: Applicability) -> bool:
    """inner is a subset of (or equal to) outer."""
    return compare(inner, outer) in (SetRelation.SUBSET, SetRelation.EQUAL)


# --------------------------------------------------------------------------- #
# The partition
# --------------------------------------------------------------------------- #

Status = Literal["assigned", "contested", "uncovered"]


@dataclass(frozen=True)
class Region:
    inside: frozenset[str]
    outside: frozenset[str]
    status: Status
    outcome: str | None
    governing: tuple[str, ...]
    """The minimal frames that decided the region (the contenders, if contested)."""
    description: str = ""


@dataclass
class Partition:
    regions: list[Region] = field(default_factory=list)
    pieces: dict[str, Piece] = field(default_factory=dict)
    discarded: list[tuple[Piece, str]] = field(default_factory=list)
    indeterminate: list[Piece] = field(default_factory=list)
    """Frames with a free-text constraint: presented verbatim, outcome not inferred."""
    overridden: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Typed frames that govern no region, with the frames that override them."""

    # -- Proposition 3 --------------------------------------------------------- #

    @property
    def accounted(self) -> set[str]:
        """Frames the output accounts for: governing a region, or recorded as overridden."""
        gov = {pid for r in self.regions for pid in r.governing}
        return gov | set(self.overridden)

    @property
    def typed_ids(self) -> set[str]:
        return set(self.pieces) - {p.piece_id for p, _ in self.discarded} - {
            p.piece_id for p in self.indeterminate}

    @property
    def deleted(self) -> set[str]:
        """Typed, non-discarded frames the output does not account for. Must be empty."""
        return self.typed_ids - self.accounted

    @property
    def incomplete(self) -> bool:
        return bool(self.indeterminate)

    # -- queries ---------------------------------------------------------------- #

    def region_for(self, case: dict[str, object]) -> Region | None:
        """The region a concrete case falls in. ``case`` must fix every attribute used."""
        inside = frozenset(pid for pid in self.typed_ids if _holds(self.pieces[pid].applicability, case))
        for r in self.regions:
            if r.inside == inside:
                return r
        return None

    def render(self) -> str:
        lines = [f"{len(self.regions)} region(s)"]
        for r in self.regions:
            who = r.description or "everyone"
            if r.status == "assigned":
                lines.append(f"  {who}: {r.outcome}  [{', '.join(r.governing)}]")
            elif r.status == "contested":
                lines.append(f"  {who}: CONTESTED between {', '.join(r.governing)}")
            else:
                lines.append(f"  {who}: UNCOVERED (no source speaks to this)")
        for pid, by in self.overridden.items():
            lines.append(f"  (overridden: {pid} by {', '.join(by)})")
        for p in self.indeterminate:
            lines.append(f"  INDETERMINATE (untyped condition, shown verbatim): {p.applicability.descriptor}")
        for p, why in self.discarded:
            lines.append(f"  DISCARDED {p.piece_id}: {why}")
        return "\n".join(lines)


def _holds(app: Applicability, case: dict[str, object]) -> bool:
    if app.is_default:
        return True
    for a in app.attributes:
        name, fam, payload = _norm_attr(a)
        if name not in case:
            raise KeyError(f"case does not fix attribute {name!r}")
        v = case[name]
        if fam == "num":
            lo, hi = payload  # type: ignore[misc]
            if not (lo <= float(v) <= hi):  # type: ignore[arg-type]
                return False
        else:
            if str(v).strip().lower() not in payload:  # type: ignore[operator]
                return False
    return True


def _describe(inside: list[Piece], outside: list[Piece]) -> str:
    parts = [p.applicability.descriptor for p in inside if not p.applicability.is_default]
    text = ", ".join(parts) if parts else ""
    if outside:
        excl = " or ".join(p.applicability.descriptor for p in outside)
        text = f"{text}, but not {excl}" if text else f"everyone except {excl}"
    return text


def partition_compose(pieces: list[Piece]) -> Partition:
    """Compose frames into a partition of the case space. See the module docstring."""
    ids = [p.piece_id for p in pieces]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate piece ids: {ids}")
    part = Partition(pieces={p.piece_id: p for p in pieces})

    typed: list[Piece] = []
    for p in pieces:
        if not _is_typed(p.applicability):
            part.indeterminate.append(p)
            continue
        try:
            ok = satisfiable([p.applicability], [])
        except ValueError as exc:
            part.discarded.append((p, str(exc)))
            continue
        if ok:
            typed.append(p)
        else:
            part.discarded.append((p, "its own constraints are unsatisfiable (empty applicability)"))

    # A name used with two kinds across frames cannot be reasoned over jointly.
    kinds: dict[str, set[str]] = {}
    for p in typed:
        for name, (fam, _) in _typed_constraints(p.applicability).items():
            kinds.setdefault(name, set()).add(fam)
    clash = {n for n, k in kinds.items() if len(k) > 1}
    if clash:
        keep = []
        for p in typed:
            if clash & set(_typed_constraints(p.applicability)):
                part.indeterminate.append(p)
            else:
                keep.append(p)
        typed = keep

    if len(typed) > MAX_FRAMES:
        raise TooManyFrames(f"{len(typed)} typed frames exceeds MAX_FRAMES={MAX_FRAMES}")

    # Order-invariance: iterate in a canonical order, never the caller's.
    typed.sort(key=lambda p: p.piece_id)
    n = len(typed)

    def below(i: Piece, j: Piece) -> bool:       # strictly smaller
        return _contains(i.applicability, j.applicability) and not _contains(j.applicability, i.applicability)

    for signs in product((True, False), repeat=n):
        inside = [p for p, s in zip(typed, signs) if s]
        outside = [p for p, s in zip(typed, signs) if not s]
        if not satisfiable([p.applicability for p in inside], [p.applicability for p in outside]):
            continue
        in_ids = frozenset(p.piece_id for p in inside)
        out_ids = frozenset(p.piece_id for p in outside)
        desc = _describe(inside, outside)

        if not inside:
            part.regions.append(Region(in_ids, out_ids, "uncovered", None, (), desc))
            continue
        minimal = [p for p in inside if not any(below(q, p) for q in inside if q is not p)]
        outcomes_agree = all(values_match(minimal[0].outcome, m.outcome) for m in minimal[1:])
        gov = tuple(sorted(m.piece_id for m in minimal))
        if len(minimal) == 1 or outcomes_agree:
            part.regions.append(Region(in_ids, out_ids, "assigned", minimal[0].outcome, gov, desc))
        else:
            part.regions.append(Region(in_ids, out_ids, "contested", None, gov, desc))

    # Proposition 3: record every typed frame that governs no region, and who overrides it.
    governing = {pid for r in part.regions for pid in r.governing}
    for p in typed:
        if p.piece_id in governing:
            continue
        by = {pid for r in part.regions if p.piece_id in r.inside for pid in r.governing}
        part.overridden[p.piece_id] = tuple(sorted(by))
    return part
