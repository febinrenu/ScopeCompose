"""Propositions 2 and 3, as unit cases and as property tests against an independent oracle.

The oracle is the point. ``partition_compose`` decides region emptiness with an
intensional satisfiability test; this file checks it *extensionally*, by
enumerating a concrete universe of cases rich enough to tell every region apart
and asking, for each case, what specificity says. Two different ways of
computing the same answer agreeing on hundreds of random frame sets is what
turns "the operator is correct" from an assertion into a measurement.
"""

from __future__ import annotations

import itertools
import random

import pytest

from composition.partition import (
    MAX_FRAMES,
    Piece,
    TooManyFrames,
    _holds,
    partition_compose,
    pieces_from_branches,
    satisfiable,
)
from contract.gold import Applicability, AttributeKind, ScopeAttribute
from contract.mock import generate
from metrics.values import values_match


def cat(name, *vals):
    return ScopeAttribute(name=name, kind=AttributeKind.CATEGORICAL, values=list(vals))


def num(name, lo=None, hi=None):
    return ScopeAttribute(name=name, kind=AttributeKind.NUMERIC_RANGE, min_value=lo, max_value=hi)


def boolean(name, v):
    return ScopeAttribute(name=name, kind=AttributeKind.BOOLEAN, boolean_value=v)


def app(desc, *attrs):
    return Applicability(descriptor=desc, attributes=list(attrs))


DEFAULT = Applicability(descriptor="everyone", is_default=True)


def piece(pid, a, outcome):
    return Piece(pid, a, outcome)


def by_inside(part):
    return {r.inside: r for r in part.regions}


# --------------------------------------------------------------------------- #
# Worked cases
# --------------------------------------------------------------------------- #


def test_rule_and_exception_is_two_regions():
    part = partition_compose([
        piece("rule", DEFAULT, "a 3% fee applies"),
        piece("exc", app("premium", cat("tier", "premium")), "no fee applies"),
    ])
    r = by_inside(part)
    assert r[frozenset({"rule", "exc"})].outcome == "no fee applies"
    assert r[frozenset({"rule"})].outcome == "a 3% fee applies"
    assert len(part.regions) == 2 and not part.deleted


def test_nested_exceptions_resolve_by_specificity_not_order():
    frames = [
        piece("all", DEFAULT, "a 3% fee applies"),
        piece("prem", app("premium", cat("tier", "premium", "platinum")), "no fee applies"),
        piece("plat", app("platinum", cat("tier", "platinum")), "a 1% fee applies"),
    ]
    for perm in itertools.permutations(frames):
        r = by_inside(partition_compose(list(perm)))
        assert r[frozenset({"all", "prem", "plat"})].outcome == "a 1% fee applies"
        assert r[frozenset({"all", "prem"})].outcome == "no fee applies"
        assert r[frozenset({"all"})].outcome == "a 3% fee applies"


def test_crossing_scopes_are_contested_on_the_overlap_and_assigned_elsewhere():
    """First-year and part-time: the overlap is contested, the rest is as stated."""
    part = partition_compose([
        piece("first", app("first-year", cat("year", "first")), "employment is permitted"),
        piece("part", app("part-time", cat("mode", "part_time")), "employment is prohibited"),
    ])
    r = by_inside(part)
    assert r[frozenset({"first", "part"})].status == "contested"
    assert r[frozenset({"first", "part"})].governing == ("first", "part")
    assert r[frozenset({"first"})].outcome == "employment is permitted"
    assert r[frozenset({"part"})].outcome == "employment is prohibited"
    assert r[frozenset()].status == "uncovered"


def test_equal_scope_disagreement_is_contested_and_agreement_is_assigned():
    a = app("premium", cat("tier", "premium"))
    dis = partition_compose([piece("x", a, "a 3% fee applies"), piece("y", a, "a 5% fee applies")])
    assert by_inside(dis)[frozenset({"x", "y"})].status == "contested"
    agree = partition_compose([piece("x", a, "a 3% fee applies"), piece("y", a, "a 3 per cent fee applies")])
    assert by_inside(agree)[frozenset({"x", "y"})].status == "assigned"


def test_a_region_nobody_speaks_to_is_uncovered_not_filled():
    part = partition_compose([piece("exc", app("premium", cat("tier", "premium")), "no fee applies")])
    r = by_inside(part)
    assert r[frozenset()].status == "uncovered" and r[frozenset()].outcome is None


def test_categorical_domains_are_open_world():
    """'Not premium' must be satisfiable: a frame naming some values never exhausts the domain."""
    part = partition_compose([piece("all", DEFAULT, "a 3% fee applies"),
                              piece("p", app("premium", cat("tier", "premium")), "no fee applies")])
    assert frozenset({"all"}) in by_inside(part)


def test_boolean_domains_are_closed():
    t = piece("t", app("students", boolean("student", True)), "no fee applies")
    f = piece("f", app("non-students", boolean("student", False)), "a 3% fee applies")
    part = partition_compose([t, f])
    assert frozenset() not in by_inside(part), "true and false exhaust a boolean: nobody is left over"


def test_numeric_adjacent_closed_intervals_leave_no_gap_but_a_real_gap_is_uncovered():
    touching = partition_compose([piece("a", app("young", num("age", 0, 18)), "x"),
                                  piece("b", app("older", num("age", 18, 65)), "y")])
    assert by_inside(touching)[frozenset({"a", "b"})].status == "contested", "age 18 is in both"
    gap = partition_compose([piece("a", app("young", num("age", 0, 17)), "x"),
                             piece("b", app("older", num("age", 30, 65)), "y")])
    assert by_inside(gap)[frozenset()].status == "uncovered"


def test_a_frame_that_is_wholly_overridden_is_recorded_not_deleted():
    """{premium, platinum} is exactly covered by {premium} and {platinum}."""
    part = partition_compose([
        piece("both", app("premium or platinum", cat("tier", "premium", "platinum")), "no fee applies"),
        piece("prem", app("premium", cat("tier", "premium")), "a 1% fee applies"),
        piece("plat", app("platinum", cat("tier", "platinum")), "a 2% fee applies"),
    ])
    assert "both" in part.overridden
    assert set(part.overridden["both"]) == {"prem", "plat"}
    assert not part.deleted


def test_untyped_constraints_are_indeterminate_and_never_given_an_outcome():
    free = ScopeAttribute(name="circumstances", kind=AttributeKind.FREE_TEXT, text="in exceptional hardship")
    part = partition_compose([piece("rule", DEFAULT, "a 3% fee applies"),
                              piece("odd", app("hardship cases", free), "no fee applies")])
    assert [p.piece_id for p in part.indeterminate] == ["odd"]
    assert part.incomplete
    assert all("odd" not in r.governing for r in part.regions), "no outcome is inferred for it"
    assert "INDETERMINATE" in part.render() and "hardship" in part.render()


def test_a_frame_with_contradictory_constraints_is_discarded_and_logged():
    bad = app("impossible", cat("tier", "gold"), cat("tier", "silver"))
    part = partition_compose([piece("rule", DEFAULT, "a 3% fee applies"), piece("bad", bad, "no fee applies")])
    assert [p.piece_id for p, _ in part.discarded] == ["bad"]
    assert "unsatisfiable" in part.discarded[0][1]
    assert not part.deleted, "a discarded frame is logged, so it is not a deletion"


def test_one_attribute_used_with_two_kinds_is_indeterminate_not_a_crash():
    part = partition_compose([piece("a", app("x", cat("size", "big")), "p"),
                              piece("b", app("y", num("size", 1, 5)), "q")])
    assert {p.piece_id for p in part.indeterminate} == {"a", "b"}


def test_too_many_frames_is_an_explicit_error():
    many = [piece(f"p{i}", app(f"g{i}", cat(f"a{i}", "v")), "x") for i in range(MAX_FRAMES + 1)]
    with pytest.raises(TooManyFrames):
        partition_compose(many)


def test_duplicate_piece_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        partition_compose([piece("a", DEFAULT, "x"), piece("a", DEFAULT, "y")])


def test_region_for_a_concrete_case():
    part = partition_compose([piece("rule", DEFAULT, "a 3% fee applies"),
                              piece("exc", app("premium", cat("tier", "premium")), "no fee applies")])
    assert part.region_for({"tier": "premium"}).outcome == "no fee applies"
    assert part.region_for({"tier": "standard"}).outcome == "a 3% fee applies"
    with pytest.raises(KeyError):
        part.region_for({})


def test_works_on_real_gold_branches_from_the_mock():
    inst = next(i for i in generate(60, seed=0) if i.personas)
    part = partition_compose(pieces_from_branches(inst.gold_branches))
    outcomes = {r.outcome for r in part.regions if r.status == "assigned"}
    assert {b.outcome for b in inst.gold_branches} <= outcomes
    assert not part.deleted


def test_satisfiable_basics():
    p, q = app("p", cat("t", "a")), app("q", cat("t", "b"))
    assert satisfiable([p], [])
    assert not satisfiable([p, q], [])
    assert satisfiable([p], [q])
    assert not satisfiable([p], [p])
    assert not satisfiable([], [DEFAULT])
    assert satisfiable([], [])


# --------------------------------------------------------------------------- #
# The property tests: an independent, extensional oracle
# --------------------------------------------------------------------------- #

TIERS = ["a", "b", "c", "d"]
OUTCOMES = ["no fee applies", "a 3% fee applies", "a 5% fee applies"]
AGE_POINTS = [-1, 0, 9, 18, 24, 30, 47, 65, 100]
UNIVERSE = [
    {"tier": t, "region": r, "age": g, "student": s}
    for t, r, g, s in itertools.product(TIERS + ["zz"], ["x", "y", "zz"], AGE_POINTS, [True, False])
]


def random_pieces(rng: random.Random, n: int) -> list[Piece]:
    out = []
    for i in range(n):
        attrs = []
        if rng.random() < 0.6:
            attrs.append(cat("tier", *rng.sample(TIERS, rng.randint(1, 3))))
        if rng.random() < 0.3:
            attrs.append(cat("region", *rng.sample(["x", "y"], rng.randint(1, 2))))
        if rng.random() < 0.4:
            lo, hi = sorted(rng.sample([0, 18, 30, 65], 2))
            attrs.append(num("age", lo, hi))
        if rng.random() < 0.2:
            attrs.append(boolean("student", rng.random() < 0.5))
        a = Applicability(descriptor=f"g{i}", attributes=attrs) if attrs else (
            DEFAULT if rng.random() < 0.7 else app(f"g{i}", cat("tier", *TIERS[:3])))
        out.append(Piece(f"p{i}", a, rng.choice(OUTCOMES)))
    return out


def oracle(pieces: list[Piece], case) -> tuple[str, str | None]:
    """Specificity, computed by extension over the universe -- no shared code with the SUT."""
    ext = {p.piece_id: frozenset(i for i, c in enumerate(UNIVERSE) if _holds(p.applicability, c))
           for p in pieces}
    cid = UNIVERSE.index(case)
    containing = [p for p in pieces if cid in ext[p.piece_id]]
    if not containing:
        return "uncovered", None
    minimal = [p for p in containing
               if not any(ext[q.piece_id] < ext[p.piece_id] for q in containing if q is not p)]
    if len(minimal) == 1 or all(values_match(minimal[0].outcome, m.outcome) for m in minimal[1:]):
        return "assigned", minimal[0].outcome
    return "contested", None


SEEDS = range(120)


def _usable(seed):
    rng = random.Random(seed)
    pieces = random_pieces(rng, rng.randint(1, 5))
    part = partition_compose(pieces)
    return pieces, part


@pytest.mark.parametrize("seed", SEEDS)
def test_every_case_lands_in_exactly_one_region_and_the_oracle_agrees(seed):
    pieces, part = _usable(seed)
    typed = [p for p in pieces if p.piece_id in part.typed_ids]
    for case in UNIVERSE:
        region = part.region_for(case)
        assert region is not None, f"seed {seed}: case {case} fell in no region (a region was wrongly dropped)"
        status, outcome = oracle(typed, case)
        assert region.status == status, (seed, case, region, status)
        if status == "assigned":
            assert values_match(region.outcome, outcome), (seed, case, region.outcome, outcome)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_reported_region_is_empty(seed):
    """Soundness of the satisfiability test: every region has a witness case."""
    pieces, part = _usable(seed)
    witnessed = {frozenset(pid for pid in part.typed_ids if _holds(part.pieces[pid].applicability, c))
                 for c in UNIVERSE}
    for r in part.regions:
        assert r.inside in witnessed, f"seed {seed}: reported region {sorted(r.inside)} has no witness case"


@pytest.mark.parametrize("seed", SEEDS)
def test_order_invariance(seed):
    pieces, part = _usable(seed)
    canon = lambda p: sorted((tuple(sorted(r.inside)), r.status, r.outcome, r.governing)  # noqa: E731
                             for r in p.regions)
    rng = random.Random(seed + 1000)
    shuffled = pieces[:]
    rng.shuffle(shuffled)
    assert canon(partition_compose(shuffled)) == canon(part)


@pytest.mark.parametrize("seed", SEEDS)
def test_proposition_3_no_typed_frame_is_ever_deleted(seed):
    _, part = _usable(seed)
    assert part.deleted == set(), f"seed {seed}: unaccounted frames {part.deleted}"
    assert part.accounted == part.typed_ids


@pytest.mark.parametrize("seed", range(60))
def test_proposition_3_holds_for_arbitrary_outcome_labels(seed):
    """No relation label is consulted, so scrambling outcomes cannot delete a frame."""
    rng = random.Random(seed)
    pieces, _ = _usable(seed)
    scrambled = [Piece(p.piece_id, p.applicability, rng.choice(OUTCOMES + ["x", "y"]), p.passage_id)
                 for p in pieces]
    assert partition_compose(scrambled).deleted == set()
