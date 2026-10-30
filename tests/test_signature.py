"""Signature unification: the proposal's worked example, algebraic properties over
random signatures, and the fold into the existing scope algebra."""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import pytest

from contract.frames import Signature
from contract.gold import Applicability, AttributeKind, ScopeAttribute
from detection.signature import (
    SIG_ATTR_PREFIX,
    Kind,
    Schema,
    SchemaError,
    SlotRel,
    fold_into_applicability,
    load_schema,
    make_signature,
    schema_hashes,
    unify,
)
from scope.attributes import SetRelation, compare

BANK = load_schema("banking")
IMM = load_schema("immigration")


def sig(**slots) -> Signature:
    return make_signature("banking", slots)


# --------------------------------------------------------------------------- #
# The proposal's worked example (v8 section 5.3)
# --------------------------------------------------------------------------- #

ROW1 = dict(event="cash_withdrawal", location="abroad", decision_kind="flat_fee")           # international cash withdrawal fee
ROW2 = dict(event="any_txn", location="abroad", decision_kind="percentage_fee")             # foreign currency transaction charge
ROW3 = dict(event="cash_withdrawal", location="abroad", channel="atm", decision_kind="percentage_fee")  # overseas ATM conversion fee


def test_rows_1_and_2_are_distinct_because_flat_and_percentage_fees_are_different_variables():
    u = unify(sig(**ROW1), sig(**ROW2))
    assert u.kind is Kind.DISTINCT and u.incompatible_slots == ["decision_kind"]


def test_row_2_subsumes_row_3():
    u = unify(sig(**ROW2), sig(**ROW3))
    assert u.kind is Kind.SUBSUMES and u.general == "a"
    assert u.slots["event"] is SlotRel.A_BROADER and u.slots["channel"] is SlotRel.A_BROADER
    assert u.slots["decision_kind"] is SlotRel.EQUAL and u.slots["location"] is SlotRel.EQUAL


def test_rows_1_and_3_are_distinct():
    assert unify(sig(**ROW1), sig(**ROW3)).kind is Kind.DISTINCT


def test_the_example_reproduces_in_both_argument_orders():
    assert unify(sig(**ROW3), sig(**ROW2)).general == "b"
    assert unify(sig(**ROW3), sig(**ROW1)).kind is Kind.DISTINCT


# --------------------------------------------------------------------------- #
# Unification cases
# --------------------------------------------------------------------------- #


def test_identical_signatures_are_equivalent():
    s = sig(**ROW3)
    assert unify(s, s).kind is Kind.EQUIVALENT


def test_two_empty_signatures_are_equivalent():
    assert unify(sig(), sig()).kind is Kind.EQUIVALENT


def test_unspecified_is_the_top_element():
    broad, narrow = sig(decision_kind="flat_fee"), sig(decision_kind="flat_fee", location="abroad")
    u = unify(broad, narrow)
    assert u.kind is Kind.SUBSUMES and u.general == "a"


def test_sibling_values_are_incompatible():
    assert unify(sig(event="transfer"), sig(event="cash_withdrawal")).kind is Kind.DISTINCT


def test_crossing_generality_is_overlap():
    """Cash withdrawals anywhere vs any transaction abroad: each broader on a different slot."""
    a = sig(event="cash_withdrawal")
    b = sig(location="abroad")
    assert unify(a, b).kind is Kind.OVERLAP


def test_free_text_against_a_typed_value_is_uncertain_never_distinct():
    a = Signature(domain="banking", slots={}, untyped_slots={"event": "lounge entry"}, schema_version="t")
    b = sig(event="cash_withdrawal")
    assert unify(a, b).kind is Kind.UNCERTAIN


def test_two_different_free_text_values_are_uncertain_two_equal_ones_are_equal():
    def free(v):
        return Signature(domain="banking", slots={}, untyped_slots={"event": v}, schema_version="t")

    assert unify(free("lounge entry"), free("airport lounge")).kind is Kind.UNCERTAIN
    assert unify(free("Lounge  Entry"), free("lounge entry")).kind is Kind.EQUIVALENT


def test_free_text_against_nothing_is_decidable():
    a = Signature(domain="banking", slots={}, untyped_slots={"event": "lounge entry"}, schema_version="t")
    u = unify(a, sig())
    assert u.kind is Kind.SUBSUMES and u.general == "b", "the side with no value is the broader statement"


def test_an_incompatible_typed_slot_wins_over_free_text_elsewhere():
    a = Signature(domain="banking", slots={"event": "transfer"}, untyped_slots={"channel": "carrier pigeon"},
                  schema_version="t")
    b = Signature(domain="banking", slots={"event": "cash_withdrawal", "channel": "atm"}, schema_version="t")
    assert unify(a, b).kind is Kind.DISTINCT


def test_cross_domain_unification_is_refused():
    with pytest.raises(ValueError, match="different domains"):
        unify(sig(event="transfer"), make_signature("immigration", {"event": "work"}))


# --------------------------------------------------------------------------- #
# Properties over random signatures
# --------------------------------------------------------------------------- #


def random_sig(rng: random.Random, sch: Schema) -> Signature:
    slots = {}
    for slot in sch.slots:
        r = rng.random()
        if r < 0.55:
            slots[slot] = rng.choice(sorted(sch.parents[slot]))
    return make_signature(sch.domain, slots, schema=sch)


@pytest.mark.parametrize("sch", [BANK, IMM], ids=["banking", "immigration"])
def test_properties_hold_over_random_signatures(sch):
    rng = random.Random(0)
    sigs = [random_sig(rng, sch) for _ in range(60)]
    flip = {"a": "b", "b": "a", None: None}
    for a in sigs:
        assert unify(a, a, schema=sch).kind is Kind.EQUIVALENT                      # reflexive
    for a in sigs:
        for b in sigs:
            ab, ba = unify(a, b, schema=sch), unify(b, a, schema=sch)
            assert ab.kind is ba.kind, (a, b)                                       # kind is symmetric
            assert ba.general == flip[ab.general], (a, b)                           # subsumption is antisymmetric
            assert ba == ab.flipped()


@pytest.mark.parametrize("sch", [BANK, IMM], ids=["banking", "immigration"])
def test_subsumption_is_transitive(sch):
    rng = random.Random(1)
    sigs = [random_sig(rng, sch) for _ in range(45)]

    def gen(x, y):  # x is strictly broader than y
        u = unify(x, y, schema=sch)
        return u.kind is Kind.SUBSUMES and u.general == "a"

    for a in sigs:
        for b in sigs:
            if not gen(a, b):
                continue
            for c in sigs:
                if gen(b, c):
                    assert gen(a, c)


@pytest.mark.parametrize("sch", [BANK, IMM], ids=["banking", "immigration"])
def test_distinct_means_some_slot_is_incompatible_and_nothing_else_is(sch):
    rng = random.Random(2)
    for _ in range(300):
        a, b = random_sig(rng, sch), random_sig(rng, sch)
        u = unify(a, b, schema=sch)
        assert (u.kind is Kind.DISTINCT) == bool(u.incompatible_slots)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_aliases_normalise_to_canonical_values():
    s = make_signature("banking", {"event": "ATM", "decision_kind": "fixed fee", "location": "Overseas"})
    assert s.slots == {"event": "atm_withdrawal", "decision_kind": "flat_fee", "location": "abroad"}


def test_out_of_vocabulary_values_are_kept_untyped_not_guessed():
    s = make_signature("banking", {"event": "lounge entry", "decision_kind": "flat_fee", "colour": "blue"})
    assert s.slots == {"decision_kind": "flat_fee"}
    assert s.untyped_slots == {"event": "lounge entry", "colour": "blue"}


def test_the_top_element_means_unspecified():
    s = make_signature("banking", {"event": "any_txn", "location": "any_location"})
    assert s.slots == {} and s.untyped_slots == {}


def test_blank_values_are_dropped():
    assert make_signature("banking", {"event": "  ", "location": ""}).slots == {}


def test_immigration_schema_normalises_routes():
    s = make_signature("immigration", {"route": "Graduate visa", "event": "switching", "location": "from inside UK"})
    assert s.slots == {"route": "graduate", "event": "switch", "location": "in_country"}


# --------------------------------------------------------------------------- #
# The taxonomy and schema validation
# --------------------------------------------------------------------------- #


def test_the_shipped_schemas_load_and_are_hashed():
    for d in ("banking", "immigration"):
        sch = load_schema(d)
        assert sch.version.startswith(d) and len(sch.sha) == 64
        for slot in sch.slots:
            assert sch.root(slot).startswith("any_")
    assert set(schema_hashes()) == {"banking", "immigration"}


def test_generality_chain_and_descendants():
    assert BANK.leq("event", "atm_withdrawal", "any_txn")
    assert BANK.leq("event", "atm_withdrawal", "cash_withdrawal")
    assert not BANK.leq("event", "cash_withdrawal", "atm_withdrawal")
    assert BANK.descendants_or_self("event", "cash_withdrawal") == [
        "atm_withdrawal", "branch_withdrawal", "cash_withdrawal"]


def _bad_schema(tmp_path: Path, body: str) -> Schema:
    (tmp_path / "x.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    load_schema.cache_clear()
    try:
        return load_schema("x", tmp_path)
    finally:
        load_schema.cache_clear()


def test_schema_validation_catches_each_defect(tmp_path):
    base = "domain: x\nschema_version: x-1\nslots:\n  s:\n"
    with pytest.raises(SchemaError, match="exactly one top"):
        _bad_schema(tmp_path, base + "    a: {}\n    b: {}\n")
    with pytest.raises(SchemaError, match="unknown parent"):
        _bad_schema(tmp_path, base + "    a: {}\n    b: {parent: nope}\n")
    with pytest.raises(SchemaError, match="names both"):
        _bad_schema(tmp_path, base + "    a: {}\n    b: {parent: a, aliases: [c]}\n    d: {parent: a, aliases: [c]}\n")
    with pytest.raises(SchemaError, match="cycle"):
        _bad_schema(tmp_path, base + "    r: {}\n    a: {parent: b}\n    b: {parent: a}\n")
    with pytest.raises(SchemaError, match="declares domain"):
        _bad_schema(tmp_path, "domain: y\nschema_version: v\nslots:\n  s:\n    a: {}\n")


# --------------------------------------------------------------------------- #
# Folding into applicability, checked against the existing scope algebra
# --------------------------------------------------------------------------- #

PREMIUM = Applicability(descriptor="premium cardholders",
                        attributes=[ScopeAttribute(name="card_tier", kind=AttributeKind.CATEGORICAL,
                                                   values=["premium"])])
EVERYONE = Applicability(descriptor="all cardholders", is_default=True)


def test_folding_turns_a_narrower_variable_into_a_typed_refinement():
    """The case lexical gating misses: an ATM-only statement is a refinement of a
    statement about all cash withdrawals, once the variable difference is typed."""
    broad = sig(event="cash_withdrawal", location="abroad", decision_kind="percentage_fee")
    narrow = sig(event="atm_withdrawal", location="abroad", decision_kind="percentage_fee")
    u = unify(broad, narrow)
    a, b = fold_into_applicability(u, broad, narrow, EVERYONE, PREMIUM)
    assert a.is_default, "the broader side gains no constraint"
    assert any(c.name == f"{SIG_ATTR_PREFIX}event" and c.values == ["atm_withdrawal"] for c in b.attributes)
    assert compare(b, a) is SetRelation.SUBSET


def test_folding_without_a_signature_difference_changes_nothing():
    s = sig(**ROW3)
    u = unify(s, s)
    a, b = fold_into_applicability(u, s, s, EVERYONE, PREMIUM)
    assert (a, b) == (EVERYONE, PREMIUM)


def test_distinct_and_uncertain_pairs_are_never_folded():
    s1, s2 = sig(event="transfer"), sig(event="cash_withdrawal")
    assert fold_into_applicability(unify(s1, s2), s1, s2, EVERYONE, PREMIUM) == (EVERYONE, PREMIUM)


def test_overlap_folds_both_sides_so_the_scope_algebra_sees_a_crossing():
    a_sig, b_sig = sig(event="cash_withdrawal"), sig(location="abroad")
    u = unify(a_sig, b_sig)
    assert u.kind is Kind.OVERLAP
    a, b = fold_into_applicability(u, a_sig, b_sig, EVERYONE, EVERYONE)
    assert compare(a, b) is SetRelation.OVERLAPPING


def test_folded_constraint_uses_every_descendant_so_narrower_values_stay_inside():
    broad = sig(event="any_txn", decision_kind="flat_fee")
    mid = sig(event="cash_withdrawal", decision_kind="flat_fee")
    leaf = sig(event="atm_withdrawal", decision_kind="flat_fee")
    _, mid_app = fold_into_applicability(unify(broad, mid), broad, mid, EVERYONE, EVERYONE)
    _, leaf_app = fold_into_applicability(unify(broad, leaf), broad, leaf, EVERYONE, EVERYONE)
    assert compare(leaf_app, mid_app) is SetRelation.SUBSET, "ATM is inside cash withdrawal"


# --------------------------------------------------------------------------- #
# Groups of co-referential frames
# --------------------------------------------------------------------------- #

from detection.signature import coreference_components, fold_component  # noqa: E402


def test_components_join_subsumption_and_keep_distinct_decisions_apart():
    """Row 2 subsumes row 3 (one component); row 1 is a different decision from both (its own)."""
    groups, _ = coreference_components([sig(**ROW2), sig(**ROW3)])
    assert groups == [[0, 1]]
    groups, _ = coreference_components([sig(**ROW1), sig(**ROW2), sig(**ROW3)])
    assert groups == [[0], [1, 2]]


def test_overlap_joins_frames_into_one_component():
    groups, _ = coreference_components([sig(event="cash_withdrawal"), sig(location="abroad")])
    assert groups == [[0, 1]]


def test_distinct_signatures_stay_in_separate_components():
    groups, uncertain = coreference_components([sig(event="transfer"), sig(event="cash_withdrawal")])
    assert groups == [[0], [1]] and uncertain == []


def test_uncertain_pairs_are_returned_not_joined():
    free = Signature(domain="banking", slots={}, untyped_slots={"event": "lounge entry"}, schema_version="t")
    groups, uncertain = coreference_components([free, sig(event="cash_withdrawal")])
    assert groups == [[0], [1]] and uncertain == [(0, 1)]


def test_components_are_independent_of_input_order():
    sigs = [sig(**ROW2), sig(**ROW3), sig(**ROW1), sig(event="transfer")]
    base, _ = coreference_components(sigs)
    rng = random.Random(0)
    for _ in range(10):
        perm = list(range(len(sigs)))
        rng.shuffle(perm)
        groups, _ = coreference_components([sigs[i] for i in perm])
        mapped = sorted(sorted(perm[k] for k in g) for g in groups)
        assert mapped == sorted(sorted(g) for g in base)


def test_fold_component_constrains_only_the_narrower_frames():
    broad, mid, leaf = sig(event="any_txn"), sig(event="cash_withdrawal"), sig(event="atm_withdrawal")
    apps = fold_component([broad, mid, leaf], [EVERYONE, EVERYONE, EVERYONE])
    assert apps[0].is_default and not apps[1].is_default and not apps[2].is_default
    assert compare(apps[2], apps[1]) is SetRelation.SUBSET and compare(apps[1], apps[0]) is SetRelation.SUBSET


def test_fold_component_matches_the_pairwise_fold_on_two_frames():
    broad = sig(event="cash_withdrawal", location="abroad", decision_kind="percentage_fee")
    narrow = sig(event="atm_withdrawal", location="abroad", decision_kind="percentage_fee")
    pair = fold_into_applicability(unify(broad, narrow), broad, narrow, EVERYONE, PREMIUM)
    group = fold_component([broad, narrow], [EVERYONE, PREMIUM])
    assert group[0] == pair[0] and [a.model_dump() for a in group[1:]] == [pair[1].model_dump()]


def test_sibling_narrowings_come_out_disjoint():
    atm, branch, cash = sig(event="atm_withdrawal"), sig(event="branch_withdrawal"), sig(event="cash_withdrawal")
    a, b, c = fold_component([atm, branch, cash], [EVERYONE] * 3)
    assert compare(a, b) is SetRelation.DISJOINT
    assert c.is_default, "cash_withdrawal is broader than both, so it gains no constraint"


def test_fold_component_unspecified_slot_makes_the_specified_frames_narrower():
    a, b = fold_component([sig(decision_kind="flat_fee"), sig(decision_kind="flat_fee", location="abroad")],
                          [EVERYONE, EVERYONE])
    assert a.is_default and any(c.name == "sig:location" for c in b.attributes)


def test_fold_component_requires_one_applicability_per_signature():
    with pytest.raises(ValueError):
        fold_component([sig(), sig()], [EVERYONE])
    assert fold_component([], []) == []
