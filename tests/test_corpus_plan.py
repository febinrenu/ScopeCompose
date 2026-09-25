"""Tests for the corpus plan and the three-way reporting split.

The decision these encode is "store same-guide as Tier 2, report it
separately". The first half is free; the second half only happens if something
can do the separating, and these are the tests that it does.
"""

from __future__ import annotations

import pytest

from benchmark.corpus_plan import (
    TARGET_TOTAL,
    TARGETS,
    TIER2_STRATA,
    PrevalenceClaimError,
    measure,
    prevalence_estimate,
)
from contract.gold import GoldInstance
from contract.models import (
    ConflictType,
    Construction,
    Domain,
    Passage,
    QueryRecord,
    ScopeRelation,
    Separation,
    SourceType,
)
from metrics.classification import split_for_reporting


def _passages():
    return [
        Passage(id="p0", text="International transactions incur a 3% fee.",
                source_type=SourceType.OFFICIAL_POLICY, date="2024-01",
                document_id="d0"),
        Passage(id="p1", text="The fee is waived for premium cardholders.",
                source_type=SourceType.PRODUCT_TERMS, date="2024-02",
                document_id="d1"),
    ]


def _record(iid="q1", *, construction=Construction.SPLIT, separation=None):
    return QueryRecord(
        query_id=iid, query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=construction, separation=separation, passages=_passages(),
    )


def _gold(iid="q1", *, construction=Construction.SPLIT, separation=None,
          relation=ScopeRelation.REFINEMENT):
    return GoldInstance(
        instance_id=iid, query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=construction, separation=separation, passages=_passages(),
        gold_conflict_type=ConflictType.CONDITIONAL, gold_scope_relation=relation,
    )


# --------------------------------------------------------------------------- #
# The three-way split
# --------------------------------------------------------------------------- #


def test_same_guide_is_lifted_out_of_tier_2():
    """The whole point of the field.

    Same-guide instances are stored as ``construction=split`` so routing and
    validation are untouched. Without ``separation`` they would be folded into
    Tier 2 by every report, and "report it separately" would quietly not happen.
    """
    rows = split_for_reporting([
        _record("a", construction=Construction.NATURAL,
                separation=Separation.CROSS_DOCUMENT),
        _record("b", separation=Separation.SYNTHETIC_SPLIT),
        _record("c", separation=Separation.SAME_GUIDE),
    ])
    assert {k: len(v) for k, v in rows.items()} == {
        "natural cross-document": 1,
        "synthetic split": 1,
        "same-guide retrieval split": 1,
    }


def test_same_guide_is_still_tier_2_to_the_two_way_split():
    """`construction` stays a two-way tag, which is what keeps the change
    additive: nothing that routes or validates on it sees a new value."""
    from metrics.classification import split_by_construction

    rows = split_by_construction([_record("c", separation=Separation.SAME_GUIDE)])
    assert set(rows) == {"split"}


def test_an_unlabelled_instance_falls_back_conservatively():
    """Every instance predating the field has ``separation=None``. Reporting an
    unlabelled split as Tier 1 would inflate the claim that matters most, so
    the fallback goes the other way."""
    rows = split_for_reporting([_record("x", separation=None)])
    assert list(rows) == ["synthetic split"]


def test_an_unlabelled_natural_instance_still_reports_as_tier_1():
    rows = split_for_reporting([
        _record("x", construction=Construction.NATURAL, separation=None)])
    assert list(rows) == ["natural cross-document"]


def test_empty_rows_are_omitted_rather_than_shown_as_zero():
    rows = split_for_reporting([_record("x", separation=Separation.SAME_GUIDE)])
    assert list(rows) == ["same-guide retrieval split"]


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


def test_the_targets_sum_to_the_total():
    assert sum(TARGETS.values()) == TARGET_TOTAL == 300


def test_tier_1_is_evidence_not_the_main_corpus():
    """The pilot found Tier 1 cannot carry the corpus at ~0.08 pairs per
    pairing. The plan reflects that: Tier 1 is a fifth of the corpus, not
    half."""
    assert TARGETS[Separation.CROSS_DOCUMENT] < TARGETS[Separation.SYNTHETIC_SPLIT]
    assert TARGETS[Separation.CROSS_DOCUMENT] / TARGET_TOTAL == pytest.approx(0.2)


def test_progress_reports_shortfall_per_row():
    p = measure([_gold(f"q{i}", construction=Construction.NATURAL,
                       separation=Separation.CROSS_DOCUMENT) for i in range(10)])
    assert p.by_row["natural cross-document"] == 10
    assert p.shortfall()["natural cross-document"] == 50
    assert not p.complete


def test_a_row_too_small_to_report_is_named():
    out = measure([_gold("q1", construction=Construction.NATURAL,
                         separation=Separation.CROSS_DOCUMENT)]).render()
    assert "cannot carry the" in out


def test_unlabelled_separation_is_surfaced_not_silently_absorbed():
    """An unlabelled Tier-1 instance is invisible in the row that matters most,
    so the count is reported rather than left to be discovered later."""
    out = measure([_gold(f"q{i}", separation=None) for i in range(5)]).render()
    assert "carry no `separation`" in out


def test_tier_2_strata_are_checked_not_just_the_count():
    """180 instances that happen to be 170 refinements would leave the relation
    boundaries -- what the paper reports -- untested."""
    all_refinement = [
        _gold(f"q{i}", separation=Separation.SYNTHETIC_SPLIT,
              relation=ScopeRelation.REFINEMENT) for i in range(180)
    ]
    p = measure(all_refinement)
    assert p.by_row["synthetic split"] == 180
    assert p.shortfall()["synthetic split"] == 0, "the count target is met"
    assert not p.complete, "but the plan is not, because the strata are not"
    assert "relation:opposed" in p.empty_strata()


def test_every_stratum_floor_is_positive():
    assert all(v > 0 for v in TIER2_STRATA.values())


# --------------------------------------------------------------------------- #
# The claim the pilot cannot support
# --------------------------------------------------------------------------- #


def test_the_pilot_yield_cannot_be_turned_into_a_prevalence_rate():
    """38 curated pairings across two domains supports a go/no-go decision and
    not a population claim.

    "Only 8% of policy document pairs contain cross-document exceptions" is the
    easy sentence to write by accident, and it is not one this design can make.
    A function that exists only to refuse is cheaper than remembering.
    """
    with pytest.raises(PrevalenceClaimError, match="not a prevalence estimate"):
        prevalence_estimate()
