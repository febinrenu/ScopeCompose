"""Tests for A4: the four-way scope-relation analyser.

This is the most consequential module Member A owns, because Member B's entire
resolution stage routes on its output. Two things are tested hardest:

1. **The two determinations stay separate.** Applicability and outcome
   agreement must be decided independently. Collapsing them is the bug the
   four-way relation replaced a two-way version to fix, and it is invisible
   from aggregate accuracy.
2. **Order invariance.** Role assignment must come from the applicability
   relation, never from input order. A failure here would make every
   downstream metric a function of retrieval order.
"""

from __future__ import annotations

import pytest

from contract.gold import Applicability, AttributeKind, ScopeAttribute
from contract.models import ScopeRelation
from scope.attributes import SetRelation, compare, intersects, narrower
from scope.descriptor import ClaimDescriptor, rule_based_descriptor
from scope.relation import Evidence, ScopeAnalyser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def tier(*values: str) -> ScopeAttribute:
    return ScopeAttribute(name="card_tier", kind=AttributeKind.ORDERED_TIER, values=list(values))


def cat(name: str, *values: str) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.CATEGORICAL, values=list(values))


def rng(name: str, lo: float | None = None, hi: float | None = None) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.NUMERIC_RANGE, min_value=lo, max_value=hi)


def app(descriptor: str, *attrs: ScopeAttribute, default: bool = False) -> Applicability:
    return Applicability(descriptor=descriptor, attributes=list(attrs), is_default=default)


def claim(pid: str, applicability: Applicability, outcome: str) -> ClaimDescriptor:
    return ClaimDescriptor(applicability=applicability, outcome=outcome, passage_id=pid)


# --------------------------------------------------------------------------- #
# Attribute algebra
# --------------------------------------------------------------------------- #


def test_default_branch_contains_everything():
    """The default branch is U by definition.

    Getting this wrong inverts branch roles -- the general rule would be
    treated as the exception -- so it is checked first and explicitly.
    """
    default = app("all cardholders", default=True)
    specific = app("premium holders", tier("premium", "platinum"))
    assert compare(default, specific) is SetRelation.SUPERSET
    assert compare(specific, default) is SetRelation.SUBSET


def test_two_defaults_are_equal():
    assert compare(app("all", default=True), app("everyone", default=True)) is SetRelation.EQUAL


def test_ordered_tier_subset():
    narrow = app("premium+", tier("premium", "platinum"))
    wide = app("silver+", tier("silver", "gold", "premium", "platinum"))
    assert compare(narrow, wide) is SetRelation.SUBSET
    assert compare(wide, narrow) is SetRelation.SUPERSET


def test_disjoint_categories():
    a = app("F-1 holders", cat("visa_category", "F-1"))
    b = app("H-1B holders", cat("visa_category", "H-1B"))
    assert compare(a, b) is SetRelation.DISJOINT
    assert intersects(a, b) is False


def test_overlapping_non_nested():
    a = app("students with gold cards", cat("status", "student"), tier("gold", "premium"))
    b = app("seniors with gold cards", cat("status", "senior"), tier("gold", "premium"))
    # Disjoint on `status`, so the whole conjunction is empty.
    assert compare(a, b) is SetRelation.DISJOINT


def test_genuinely_overlapping_sets():
    a = app("silver and gold", tier("silver", "gold"))
    b = app("gold and premium", tier("gold", "premium"))
    assert compare(a, b) is SetRelation.OVERLAPPING
    assert intersects(a, b) is True
    assert narrower(a, b) is None


def test_equal_sets():
    a = app("gold", tier("gold"))
    b = app("gold tier", tier("gold"))
    assert compare(a, b) is SetRelation.EQUAL


def test_numeric_range_subset():
    inner = app("500 to 1000", rng("amount", 500, 1000))
    outer = app("100 to 5000", rng("amount", 100, 5000))
    assert compare(inner, outer) is SetRelation.SUBSET


def test_numeric_ranges_disjoint():
    low = app("under 100", rng("amount", hi=100))
    high = app("over 500", rng("amount", lo=500))
    assert compare(low, high) is SetRelation.DISJOINT


def test_open_ended_range_contains_bounded():
    bounded = app("5000 to 9000", rng("amount", 5000, 9000))
    open_ended = app("above 1000", rng("amount", lo=1000))
    assert compare(bounded, open_ended) is SetRelation.SUBSET


def test_extra_constraint_makes_a_set_narrower():
    """More constraints means a SMALLER set. An applicability is a conjunction,
    so an unconstrained dimension admits everything on it."""
    two = app("premium under 5000", tier("premium"), rng("amount", hi=5000))
    one = app("premium", tier("premium"))
    assert compare(two, one) is SetRelation.SUBSET
    assert compare(one, two) is SetRelation.SUPERSET


def test_free_text_is_undecidable():
    """Free text cannot be reasoned over arithmetically.

    Returning UNKNOWN and falling through to entailment is correct; guessing
    would produce a confidently wrong subset judgement.
    """
    a = app("complicated", ScopeAttribute(name="x", kind=AttributeKind.FREE_TEXT, text="hmm"))
    b = app("also complicated", ScopeAttribute(name="x", kind=AttributeKind.FREE_TEXT, text="hm"))
    assert compare(a, b) is SetRelation.UNKNOWN
    assert intersects(a, b) is None


def test_different_dimensions_overlap_exactly():
    """Constraining different dimensions is DECIDABLE, not unknown.

    {card_tier = premium} and {status = student} necessarily intersect (a
    premium-holding student), and neither contains the other (each leaves the
    other's dimension unconstrained). That is OVERLAPPING, and deciding it
    here rather than deferring to entailment keeps the answer exact -- and
    avoids the UNKNOWN -> `opposed` fallback needlessly costing composability
    on pairs that merely cross-cut.
    """
    a = app("premium", tier("premium"))
    b = app("students", cat("status", "student"))
    assert compare(a, b) is SetRelation.OVERLAPPING
    assert intersects(a, b) is True
    assert narrower(a, b) is None


def test_free_text_still_blocks_a_decision_even_beside_typed_dimensions():
    """One untyped dimension can constrain the set arbitrarily, so no
    conclusion may be drawn from the typed ones."""
    a = app("premium but complicated", tier("premium"),
            ScopeAttribute(name="misc", kind=AttributeKind.FREE_TEXT, text="unclear"))
    b = app("students", cat("status", "student"))
    assert compare(a, b) is SetRelation.UNKNOWN


def test_no_attributes_is_undecidable():
    assert compare(app("something"), app("something else")) is SetRelation.UNKNOWN


def test_narrower_picks_the_smaller_set():
    small = app("premium", tier("premium"))
    big = app("silver+", tier("silver", "gold", "premium"))
    assert narrower(small, big) is small
    assert narrower(big, small) is small


def test_comparison_is_symmetric_under_argument_swap():
    """compare(a,b) and compare(b,a) must be exact mirrors.

    Any asymmetry here leaks into branch-role assignment and breaks order
    invariance at the first determination.
    """
    pairs = [
        (app("premium", tier("premium")), app("silver+", tier("silver", "gold", "premium"))),
        (app("F-1", cat("v", "F-1")), app("H-1B", cat("v", "H-1B"))),
        (app("a", rng("n", 0, 10)), app("b", rng("n", 5, 15))),
        (app("all", default=True), app("some", tier("gold"))),
    ]
    mirror = {
        SetRelation.SUBSET: SetRelation.SUPERSET,
        SetRelation.SUPERSET: SetRelation.SUBSET,
        SetRelation.EQUAL: SetRelation.EQUAL,
        SetRelation.DISJOINT: SetRelation.DISJOINT,
        SetRelation.OVERLAPPING: SetRelation.OVERLAPPING,
        SetRelation.UNKNOWN: SetRelation.UNKNOWN,
    }
    for a, b in pairs:
        assert compare(b, a) is mirror[compare(a, b)]


# --------------------------------------------------------------------------- #
# The four-way combination rule
# --------------------------------------------------------------------------- #


@pytest.fixture
def analyser() -> ScopeAnalyser:
    return ScopeAnalyser(heuristic_nli=True)


@pytest.mark.parametrize(
    "set_rel,agree,expected",
    [
        # Disjoint ignores outcomes -- no case falls under both.
        (SetRelation.DISJOINT, True, ScopeRelation.DISJOINT),
        (SetRelation.DISJOINT, False, ScopeRelation.DISJOINT),
        # Nested scope: outcome agreement is what separates the two answers.
        (SetRelation.SUBSET, True, ScopeRelation.REDUNDANT),
        (SetRelation.SUBSET, False, ScopeRelation.REFINEMENT),
        (SetRelation.SUPERSET, True, ScopeRelation.REDUNDANT),
        (SetRelation.SUPERSET, False, ScopeRelation.REFINEMENT),
        # Overlapping, neither contained.
        (SetRelation.OVERLAPPING, True, ScopeRelation.REDUNDANT),
        (SetRelation.OVERLAPPING, False, ScopeRelation.OPPOSED),
        # Same cases, different outcomes: a flat contradiction.
        (SetRelation.EQUAL, True, ScopeRelation.REDUNDANT),
        (SetRelation.EQUAL, False, ScopeRelation.OPPOSED),
    ],
)
def test_combination_rule(analyser, set_rel, agree, expected):
    """The formal rule, exhaustively.

    Every cell of the scope x outcome table. Neither determination alone
    yields the right answer for any row, which is the whole argument for the
    four-way relation.
    """
    relation, why = analyser._combine(set_rel, agree)
    assert relation is expected
    assert why, "every decision must carry a rationale"


def test_undecidable_scope_falls_back_to_selection(analyser):
    """Composing on unestablished scope would fabricate a branch.

    Selection is what prior work does and is the safe default, so UNKNOWN
    routes there rather than optimistically composing.
    """
    relation, why = analyser._combine(SetRelation.UNKNOWN, False)
    assert relation is ScopeRelation.OPPOSED
    assert "could not be established" in why


def test_nested_scope_with_same_outcome_is_redundant_not_refinement(analyser):
    """The confusion that matters most.

    A passage restating the general rule for a specific group has nested
    scope, exactly like a real exception. Only the outcome tells them apart.
    Calling this a refinement would make Member B compose a branch that does
    not exist -- inflating Preservation Rate while corrupting the answer.
    """
    default = claim("p0", app("all cardholders", default=True), "a 3% fee applies")
    restatement = claim("p1", app("premium holders", tier("premium")), "a 3% fee applies")

    decision = analyser.analyse(default, restatement)
    # SUBSET here is canonical: the narrower branch sits inside the default.
    assert decision.set_relation is SetRelation.SUBSET   # scopes DO nest
    assert decision.outcomes_agree                        # but outcomes match
    assert decision.relation is ScopeRelation.REDUNDANT


def test_nested_scope_with_different_outcome_is_refinement(analyser):
    """The motivating case: same scope structure as above, different outcome."""
    default = claim("p0", app("all cardholders", default=True), "a 3% fee applies")
    exception = claim("p1", app("premium holders", tier("premium")), "no fee applies")

    decision = analyser.analyse(default, exception)
    assert decision.set_relation is SetRelation.SUBSET
    assert not decision.outcomes_agree
    assert decision.relation is ScopeRelation.REFINEMENT


def test_identical_outcome_strings_agree_without_a_model(analyser):
    agree, conf = analyser.outcomes_agree("no fee applies", "No Fee Applies")
    assert agree
    assert conf == 1.0


def test_empty_outcome_does_not_agree(analyser):
    assert analyser.outcomes_agree("", "a fee applies")[0] is False


# --------------------------------------------------------------------------- #
# Branch-role assignment
# --------------------------------------------------------------------------- #


def test_default_role_goes_to_the_wider_scope(analyser):
    default = claim("p0", app("all cardholders", default=True), "a 3% fee applies")
    exception = claim("p1", app("premium holders", tier("premium")), "no fee applies")

    decision = analyser.analyse(default, exception)
    assert decision.default.passage_id == "p0"
    assert decision.exception.passage_id == "p1"


def test_roles_are_identical_when_arguments_are_swapped(analyser):
    """The core order-invariance property, at the unit level.

    'Whichever branch has the superset applicability is the default, full
    stop.' If this ever depends on argument order, every downstream number
    becomes a function of retrieval order.
    """
    default = claim("p0", app("all cardholders", default=True), "a 3% fee applies")
    exception = claim("p1", app("premium holders", tier("premium")), "no fee applies")

    forward = analyser.analyse(default, exception)
    backward = analyser.analyse(exception, default)

    assert forward.relation is backward.relation
    assert forward.default.passage_id == backward.default.passage_id == "p0"
    assert forward.exception.passage_id == backward.exception.passage_id == "p1"
    assert forward.outcomes_agree == backward.outcomes_agree

    # set_relation is stored CANONICALLY -- as the exception relative to the
    # default -- precisely so it does not mirror under argument swap. Storing
    # compare()'s raw output would flip SUBSET/SUPERSET here and make the
    # order-invariance harness report a violation on every nested pair.
    assert forward.set_relation is backward.set_relation is SetRelation.SUBSET


def test_set_relation_is_recorded_canonically(analyser):
    """A nested pair is always recorded as SUBSET, whichever way it came in."""
    default = claim("p0", app("all cardholders", default=True), "a 3% fee applies")
    exception = claim("p1", app("premium holders", tier("premium")), "no fee applies")
    for a, b in [(default, exception), (exception, default)]:
        assert analyser.analyse(a, b).set_relation is SetRelation.SUBSET


def test_roles_stable_even_when_the_exception_is_listed_first(analyser):
    """Retrieval can return the exception before the general rule. The
    composed answer must be the same either way."""
    exception = claim("p0", app("premium holders", tier("premium")), "no fee applies")
    default = claim("p1", app("all cardholders", default=True), "a 3% fee applies")

    decision = analyser.analyse(exception, default)
    assert decision.default.passage_id == "p1"      # the wider scope, despite arriving second
    assert decision.exception.passage_id == "p0"
    assert decision.relation is ScopeRelation.REFINEMENT


@pytest.mark.parametrize(
    "a_app,a_out,b_app,b_out",
    [
        (app("all", default=True), "fee", app("premium", tier("premium")), "no fee"),
        (app("F-1", cat("v", "F-1")), "prohibited", app("H-1B", cat("v", "H-1B")), "permitted"),
        (app("silver+", tier("silver", "gold")), "x", app("gold+", tier("gold", "premium")), "y"),
        (app("all", default=True), "fee", app("premium", tier("premium")), "fee"),
    ],
)
def test_relation_is_invariant_under_argument_order(analyser, a_app, a_out, b_app, b_out):
    fwd = analyser.analyse(claim("p0", a_app, a_out), claim("p1", b_app, b_out))
    rev = analyser.analyse(claim("p1", b_app, b_out), claim("p0", a_app, a_out))
    assert fwd.relation is rev.relation
    assert fwd.default.passage_id == rev.default.passage_id
    assert fwd.confidence == rev.confidence


def test_tied_entailment_does_not_pick_a_direction():
    """A coin-flip subset direction would assign branch roles arbitrarily,
    which is precisely the order dependence the design forbids. A tie must
    return OVERLAPPING instead."""

    class TiedNLI:
        def score(self, premise, hypothesis):
            from detection.nli import NLIScores
            return NLIScores(entailment=0.8, neutral=0.1, contradiction=0.1)

        def score_batch(self, pairs):
            return [self.score(p, h) for p, h in pairs]

    analyser = ScopeAnalyser(nli=TiedNLI())
    rel, evidence, _ = analyser.scope_relation(app("a"), app("b"))
    assert evidence is Evidence.ENTAILMENT
    assert rel is SetRelation.OVERLAPPING


# --------------------------------------------------------------------------- #
# Evidence provenance
# --------------------------------------------------------------------------- #


def test_typed_attributes_are_preferred_over_entailment(analyser):
    """Attribute decisions are exact; entailment decisions are a model's
    judgement. Mixing them silently would make four-way accuracy
    uninterpretable, so the source is recorded."""
    _, evidence, conf = analyser.scope_relation(
        app("premium", tier("premium")), app("silver+", tier("silver", "gold", "premium"))
    )
    assert evidence is Evidence.ATTRIBUTES
    assert conf > 0.9


def test_default_branch_is_its_own_evidence_class(analyser):
    _, evidence, conf = analyser.scope_relation(app("all", default=True), app("some", tier("gold")))
    assert evidence is Evidence.DEFAULT_BRANCH
    assert conf == 1.0


def test_untyped_scopes_fall_through_to_entailment(analyser):
    _, evidence, _ = analyser.scope_relation(app("something"), app("something else"))
    assert evidence is Evidence.ENTAILMENT


# --------------------------------------------------------------------------- #
# Rule-based descriptor extraction
# --------------------------------------------------------------------------- #


def test_rule_extractor_recognises_a_tier():
    d = rule_based_descriptor("p1", "The fee is waived for premium-tier cardholders.")
    names = {a.name for a in d.applicability.attributes}
    assert "card_tier" in names
    assert not d.applicability.is_default
    assert d.source == "rules"


def test_rule_extractor_recognises_a_general_rule():
    d = rule_based_descriptor("p0", "All international transactions incur a 3% fee.")
    assert d.applicability.is_default


def test_an_exception_cue_prevents_a_default_label():
    """A passage carving something out is never the default branch, however
    general its surrounding language looks."""
    d = rule_based_descriptor("p1", "All transactions incur a fee, except for gift cards.")
    assert not d.applicability.is_default


def test_rule_extractor_recognises_a_visa_category():
    d = rule_based_descriptor("p0", "An F-1 holder may not accept employment.")
    assert any(a.name == "visa_category" for a in d.applicability.attributes)


def test_rule_extractor_recognises_an_amount_threshold():
    d = rule_based_descriptor("p2", "The waiver does not apply to transactions above 5,000.")
    amounts = [a for a in d.applicability.attributes if a.name == "amount"]
    assert amounts and amounts[0].min_value == 5000.0


def test_rule_extractor_is_deterministic():
    text = "The fee is waived for premium-tier cardholders."
    a = rule_based_descriptor("p1", text)
    b = rule_based_descriptor("p1", text)
    assert a.applicability.model_dump() == b.applicability.model_dump()
    assert a.outcome == b.outcome
