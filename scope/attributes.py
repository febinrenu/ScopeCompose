"""Set algebra over typed applicability attributes.

The "light structured reasoning" half of A4. Where an applicability set can be
decomposed into typed attributes -- a card tier, an age range, a visa category
-- the subset/disjoint question is arithmetic and can be answered exactly,
instead of being handed to a language model as a prose puzzle.

That matters more than it sounds. "Is {premium, platinum} a subset of {silver,
gold, premium, platinum}?" is a question an LLM answers correctly most of the
time and wrongly some of the time, with no signal about which. Here it is a
set operation with a definite answer, and the cases that genuinely cannot be
decided arithmetically return ``UNKNOWN`` and fall through to entailment --
explicitly, rather than by silently guessing.

**The model.** An applicability set is a CONJUNCTION of per-attribute
constraints over a case space U. Each constraint restricts one dimension;
dimensions not mentioned are unconstrained. So more constraints means a
*smaller* set, and the default (unconditioned) branch has no constraints at
all, making it U itself.
"""

from __future__ import annotations

from enum import Enum

from contract.gold import Applicability, AttributeKind, ScopeAttribute


class SetRelation(str, Enum):
    """How two applicability sets relate.

    ``SUBSET`` means the *first* is contained in the second.
    """

    SUBSET = "subset"
    SUPERSET = "superset"
    EQUAL = "equal"
    DISJOINT = "disjoint"
    OVERLAPPING = "overlapping"
    """Intersecting, with neither containing the other."""

    UNKNOWN = "unknown"
    """Not decidable from the typed attributes -- free text, or the two sets
    constrain dimensions that cannot be compared. The caller falls back to
    entailment rather than guessing."""


class _Constraint:
    """One attribute's allowed values, in a comparable form."""

    __slots__ = ("kind", "values", "lo", "hi")

    def __init__(self, attr: ScopeAttribute):
        self.kind = attr.kind
        self.values: frozenset[str] | None = None
        self.lo: float | None = None
        self.hi: float | None = None

        if attr.kind in (AttributeKind.CATEGORICAL, AttributeKind.ORDERED_TIER):
            self.values = frozenset(v.strip().lower() for v in (attr.values or []))
        elif attr.kind is AttributeKind.BOOLEAN:
            self.values = frozenset({str(attr.boolean_value).lower()})
        elif attr.kind is AttributeKind.NUMERIC_RANGE:
            self.lo = attr.min_value if attr.min_value is not None else float("-inf")
            self.hi = attr.max_value if attr.max_value is not None else float("inf")

    @property
    def decidable(self) -> bool:
        return self.kind is not AttributeKind.FREE_TEXT

    # -- pairwise tests ------------------------------------------------------- #

    def disjoint_from(self, other: _Constraint) -> bool:
        if not (self.decidable and other.decidable) or self.kind is not other.kind:
            return False
        if self.values is not None and other.values is not None:
            return not (self.values & other.values)
        if self.lo is not None and other.lo is not None:
            return self.hi < other.lo or other.hi < self.lo
        return False

    def subset_of(self, other: _Constraint) -> bool:
        if not (self.decidable and other.decidable) or self.kind is not other.kind:
            return False
        if self.values is not None and other.values is not None:
            return self.values <= other.values
        if self.lo is not None and other.lo is not None:
            return other.lo <= self.lo and self.hi <= other.hi
        return False

    def equals(self, other: _Constraint) -> bool:
        return self.subset_of(other) and other.subset_of(self)


def _constraints(app: Applicability) -> dict[str, _Constraint]:
    return {a.name.strip().lower(): _Constraint(a) for a in app.attributes}


def compare(a: Applicability, b: Applicability) -> SetRelation:
    """Decide how applicability set ``a`` relates to ``b``.

    The default branch short-circuits everything: it is U by definition, so it
    contains every other set. Getting this wrong would invert branch roles --
    the general rule would be treated as the exception -- so it is handled
    first and explicitly.
    """
    if a.is_default and b.is_default:
        return SetRelation.EQUAL
    if a.is_default:
        return SetRelation.SUPERSET
    if b.is_default:
        return SetRelation.SUBSET

    ca, cb = _constraints(a), _constraints(b)

    # No typed attributes to reason over on either side.
    if not ca or not cb:
        return SetRelation.UNKNOWN

    # Disjointness first: one incompatible dimension makes the whole
    # conjunction empty, regardless of how the other dimensions relate.
    for name, con_a in ca.items():
        con_b = cb.get(name)
        if con_b is not None and con_a.disjoint_from(con_b):
            return SetRelation.DISJOINT

    shared = set(ca) & set(cb)
    if not shared:
        # The two sets constrain entirely different dimensions. They almost
        # certainly intersect -- a case can satisfy both -- but neither
        # contains the other, and saying so confidently would be overreach.
        return SetRelation.UNKNOWN

    if any(not c.decidable for c in list(ca.values()) + list(cb.values())):
        return SetRelation.UNKNOWN

    # a subset of b requires a to be at least as restrictive on EVERY
    # dimension b constrains. A dimension b constrains and a does not means a
    # admits cases b excludes, so a is not contained in b.
    a_in_b = all(
        name in ca and ca[name].subset_of(con_b)
        for name, con_b in cb.items()
    )
    b_in_a = all(
        name in cb and cb[name].subset_of(con_a)
        for name, con_a in ca.items()
    )

    if a_in_b and b_in_a:
        return SetRelation.EQUAL
    if a_in_b:
        return SetRelation.SUBSET
    if b_in_a:
        return SetRelation.SUPERSET
    return SetRelation.OVERLAPPING


def intersects(a: Applicability, b: Applicability) -> bool | None:
    """Whether any case falls under both. ``None`` when undecidable."""
    rel = compare(a, b)
    if rel is SetRelation.DISJOINT:
        return False
    if rel is SetRelation.UNKNOWN:
        return None
    return True


def narrower(a: Applicability, b: Applicability) -> Applicability | None:
    """The strictly smaller of two sets, or ``None`` if neither contains the other.

    This is how branch roles are assigned: the wider set is the default, the
    narrower one the exception. Derived from the sets themselves, never from
    the order the passages were retrieved in.
    """
    rel = compare(a, b)
    if rel is SetRelation.SUBSET:
        return a
    if rel is SetRelation.SUPERSET:
        return b
    return None


def describe(a: Applicability, b: Applicability) -> str:
    """Human-readable explanation. Used in error analysis and the annotation UI."""
    rel = compare(a, b)
    reasons = {
        SetRelation.EQUAL: "the two scopes are the same",
        SetRelation.SUBSET: f"{a.descriptor!r} is strictly inside {b.descriptor!r}",
        SetRelation.SUPERSET: f"{b.descriptor!r} is strictly inside {a.descriptor!r}",
        SetRelation.DISJOINT: "no case satisfies both scopes",
        SetRelation.OVERLAPPING: "the scopes intersect but neither contains the other",
        SetRelation.UNKNOWN: (
            "not decidable from typed attributes "
            "(free text, or no shared dimension) -- falling back to entailment"
        ),
    }
    return f"{rel.value}: {reasons[rel]}"
