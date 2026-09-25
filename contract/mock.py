"""Deterministic mock benchmark generator.

The point of this module is scheduling, not realism. Member B cannot build
against Member A's detector until it exists, and Member A cannot evaluate
against Member B's gold labels until those exist. A seeded generator that
emits schema-valid instances covering every conflict type, every scope
relation, and both multi-exception patterns breaks that deadlock in week one.

Determinism is a hard requirement, not a nicety: the same seed must produce a
byte-identical file, because this is the shared fixture both members test
against. Anything non-deterministic here turns a real regression into an
argument about whose machine is wrong.

These instances are NOT benchmark data. They are structurally valid and
semantically plausible, but they are templated, and no claim in the paper may
rest on them. Real data comes from ``benchmark/``.

Usage::

    python -m contract.mock --n 50 --seed 0 -o mock.jsonl
    python -m contract.mock --n 20 --seed 0 --format record   # wire records only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Iterator

from contract.gold import (
    Annotation,
    Applicability,
    AttributeKind,
    Branch,
    Explicitness,
    GoldInstance,
    ScopeAttribute,
)
from contract.models import (
    ConflictType,
    Construction,
    Domain,
    MultiExceptionFlags,
    Passage,
    ScopeRelation,
    Separation,
    SourceType,
)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Template:
    """One instance shape. ``weight`` controls how often it is drawn."""

    name: str
    domain: Domain
    conflict_type: ConflictType
    scope_relation: ScopeRelation | None
    weight: int
    is_distractor: bool = False
    nested: bool = False
    crossed: bool = False
    explicitness: Explicitness = Explicitness.EXPLICIT


TEMPLATES: list[Template] = [
    # -- the core case: a real exception, both branches true ---------------- #
    Template("fin_refinement_explicit", Domain.FINANCIAL_TERMS,
             ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, weight=6),
    Template("fin_refinement_implicit", Domain.FINANCIAL_TERMS,
             ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, weight=4,
             explicitness=Explicitness.IMPLICIT),
    Template("imm_refinement_explicit", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, weight=5),
    Template("imm_refinement_implicit", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, weight=4,
             explicitness=Explicitness.IMPLICIT),

    # -- the other three relations ------------------------------------------ #
    Template("fin_disjoint", Domain.FINANCIAL_TERMS,
             ConflictType.CONDITIONAL, ScopeRelation.DISJOINT, weight=3),
    Template("fin_redundant", Domain.FINANCIAL_TERMS,
             ConflictType.CONDITIONAL, ScopeRelation.REDUNDANT, weight=3),
    Template("imm_opposed", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.CONDITIONAL, ScopeRelation.OPPOSED, weight=3),

    # -- second-order structure (flagged, not composed) --------------------- #
    Template("fin_nested", Domain.FINANCIAL_TERMS,
             ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, weight=2, nested=True),
    Template("imm_crossed", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, weight=2, crossed=True),

    # -- mandatory distractors ---------------------------------------------- #
    # Without these the Spurious-Condition Rate has no denominator, and the
    # factual/conditional confusion cell -- the detector's headline number --
    # cannot be measured at all.
    Template("fin_factual", Domain.FINANCIAL_TERMS,
             ConflictType.FACTUAL, None, weight=5, is_distractor=True),
    Template("imm_factual", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.FACTUAL, None, weight=4, is_distractor=True),
    Template("fin_temporal", Domain.FINANCIAL_TERMS,
             ConflictType.TEMPORAL, None, weight=3, is_distractor=True),
    Template("imm_opinion", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.OPINION, None, weight=2, is_distractor=True),
    Template("fin_no_conflict", Domain.FINANCIAL_TERMS,
             ConflictType.NO_CONFLICT, None, weight=4, is_distractor=True),
    Template("imm_no_conflict", Domain.IMMIGRATION_ELIGIBILITY,
             ConflictType.NO_CONFLICT, None, weight=3, is_distractor=True),
]


# Slot fillers. Small, closed vocabularies keep the output deterministic and
# the templates readable.
_CARD_TIERS = ["standard", "classic", "premium", "platinum", "signature"]
_FEE_PCTS = ["1.5%", "2%", "2.75%", "3%", "3.5%"]
_TXN_KINDS = ["international transactions", "foreign currency purchases",
              "cross-border transfers", "overseas ATM withdrawals"]
_VISA_CATS = ["F-1", "J-1", "M-1", "H-4", "B-2"]
_WORK_CONDS = [
    "enrolled full-time and authorised for curricular practical training",
    "granted economic hardship authorisation by the service centre",
    "employed on-campus for no more than 20 hours per week",
    "holding a valid employment authorisation document",
]
_ACCOUNT_KINDS = ["savings accounts", "current accounts", "student accounts", "joint accounts"]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _passage(pid: str, text: str, source_type: SourceType, date: str, doc_id: str) -> Passage:
    return Passage(
        id=pid,
        text=text,
        source_type=source_type,
        date=date,
        document_id=doc_id,
        source_url=f"https://example.invalid/{doc_id}",
    )


def _tier_attr(name: str, values: list[str]) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.ORDERED_TIER, values=values)


def _cat_attr(name: str, values: list[str]) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.CATEGORICAL, values=values)


def _build_financial_refinement(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """The motivating case: a general fee, waived for a narrower population."""
    pct = rng.choice(_FEE_PCTS)
    txn = rng.choice(_TXN_KINDS)
    tier_i = rng.randrange(2, len(_CARD_TIERS))
    exc_tiers = _CARD_TIERS[tier_i:]
    exc_label = _CARD_TIERS[tier_i]

    if tpl.explicitness is Explicitness.EXPLICIT:
        exception_text = (
            f"The {txn} fee is waived for {exc_label}-tier cardholders."
        )
    else:
        # Implicit: the passage never says "exception", it silently narrows
        # applicability by describing only the narrower product. This is the
        # case Contrastive Scope Probing exists to recover.
        exception_text = (
            f"{exc_label.capitalize()} cardholders enjoy fee-free {txn} as a standard benefit "
            f"of the {exc_label} programme."
        )

    passages = [
        _passage("p0", f"{txn.capitalize()} incur a {pct} fee.".capitalize(),
                 SourceType.OFFICIAL_POLICY, "2024-03", f"doc_terms_{idx}"),
        _passage("p1", exception_text, SourceType.PRODUCT_TERMS, "2025-01", f"doc_prod_{idx}"),
    ]

    branches = [
        Branch(
            branch_id="b0",
            condition=None,
            outcome=f"a {pct} fee applies",
            applicability=Applicability(descriptor="all cardholders", is_default=True),
            supporting_passage="p0",
        ),
        Branch(
            branch_id="b1",
            condition=f"the cardholder holds a {exc_label}-tier card",
            outcome="no fee applies",
            applicability=Applicability(
                descriptor=f"{exc_label}-tier cardholders",
                attributes=[_tier_attr("card_tier", exc_tiers)],
            ),
            explicitness=tpl.explicitness,
            supporting_passage="p1",
        ),
    ]

    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=f"Do I pay a fee on {txn}?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_scoped_answer=(
            f"{txn.capitalize()} normally incur a {pct} fee (per p0). "
            f"However, the fee is waived for {exc_label}-tier cardholders (per p1)."
        ),
        selection_answer=f"{txn.capitalize()} incur a {pct} fee.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build_immigration_refinement(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    cat = rng.choice(_VISA_CATS)
    cond = rng.choice(_WORK_CONDS)

    if tpl.explicitness is Explicitness.EXPLICIT:
        exception_text = (
            f"A {cat} holder may accept employment if {cond}."
        )
    else:
        exception_text = (
            f"Students {cond} may be issued an employment authorisation record "
            f"under the {cat} programme."
        )

    passages = [
        _passage("p0", f"A {cat} holder may not accept employment in the United States.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-09", f"doc_policy_{idx}"),
        _passage("p1", exception_text, SourceType.FAQ, "2025-02", f"doc_faq_{idx}"),
    ]

    branches = [
        Branch(
            branch_id="b0",
            condition=None,
            outcome="employment is prohibited",
            applicability=Applicability(descriptor=f"all {cat} holders", is_default=True),
            supporting_passage="p0",
        ),
        Branch(
            branch_id="b1",
            condition=cond,
            outcome="employment is permitted",
            applicability=Applicability(
                descriptor=f"{cat} holders who are {cond}",
                attributes=[
                    _cat_attr("visa_category", [cat]),
                    ScopeAttribute(name="authorisation", kind=AttributeKind.BOOLEAN, boolean_value=True),
                ],
            ),
            explicitness=tpl.explicitness,
            supporting_passage="p1",
        ),
    ]

    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=f"Can a {cat} holder work?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_scoped_answer=(
            f"A {cat} holder generally may not accept employment (per p0), "
            f"but employment is permitted if {cond} (per p1)."
        ),
        selection_answer=f"A {cat} holder may not accept employment.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build_disjoint(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """Two rules about non-overlapping populations. No case falls under both."""
    a, b = rng.sample(_ACCOUNT_KINDS, 2)
    fee_a, fee_b = rng.sample(_FEE_PCTS, 2)

    passages = [
        _passage("p0", f"A monthly maintenance charge of {fee_a} applies to {a}.",
                 SourceType.OFFICIAL_POLICY, "2024-06", f"doc_a_{idx}"),
        _passage("p1", f"A monthly maintenance charge of {fee_b} applies to {b}.",
                 SourceType.PRODUCT_TERMS, "2024-08", f"doc_b_{idx}"),
    ]
    # Neither branch is the default. A disjoint pair has two separately scoped
    # rules and no unconditioned general one; marking either as the default
    # would make its applicability the whole case space, which would make the
    # other a subset of it -- i.e. refinement, the relation this template
    # exists to be distinguished from.
    branches = [
        Branch(branch_id="b0", condition=f"the account is one of the {a}",
               outcome=f"{fee_a} charge",
               applicability=Applicability(descriptor=f"holders of {a}",
                                           attributes=[_cat_attr("account_kind", [a])]),
               supporting_passage="p0"),
        Branch(branch_id="b1", condition=f"the account is one of the {b}",
               outcome=f"{fee_b} charge",
               applicability=Applicability(descriptor=f"holders of {b}",
                                           attributes=[_cat_attr("account_kind", [b])]),
               supporting_passage="p1"),
    ]
    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query="What is my monthly maintenance charge?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_scoped_answer=(
            f"{a.capitalize()} carry a {fee_a} monthly charge (per p0); "
            f"{b} carry {fee_b} (per p1). These apply to different account types."
        ),
        selection_answer=f"A monthly maintenance charge of {fee_a} applies.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build_redundant(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """Overlapping scope, SAME outcome. Not a conflict -- must be merged, not composed.

    This is the template that catches the most damaging confusion in the whole
    pipeline: treating a restatement as an exception makes the system invent a
    branch that does not exist, which inflates Preservation Rate while
    corrupting the answer.
    """
    pct = rng.choice(_FEE_PCTS)
    txn = rng.choice(_TXN_KINDS)
    tier = rng.choice(_CARD_TIERS[2:])

    passages = [
        _passage("p0", f"{txn.capitalize()} incur a {pct} fee.".capitalize(),
                 SourceType.OFFICIAL_POLICY, "2024-03", f"doc_terms_{idx}"),
        _passage("p1", f"{tier.capitalize()} cardholders are charged {pct} on {txn}, "
                       f"in line with the standard schedule.",
                 SourceType.FAQ, "2024-11", f"doc_faq_{idx}"),
    ]
    branches = [
        Branch(branch_id="b0", condition=None, outcome=f"a {pct} fee applies",
               applicability=Applicability(descriptor="all cardholders", is_default=True),
               supporting_passage="p0"),
    ]
    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=f"Do I pay a fee on {txn}?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_scoped_answer=f"{txn.capitalize()} incur a {pct} fee, including for {tier} cardholders.",
        selection_answer=f"{txn.capitalize()} incur a {pct} fee.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build_opposed(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """Overlapping scope, NEITHER NESTED, outcomes disagree. A real contradiction.

    The non-nesting is the whole content of this template, and it is easy to get
    wrong. An earlier version scoped the two branches as "first-year holders"
    and "first-year holders at designated institutions" -- which is a nested
    pair, and therefore a refinement, not an opposition. It trained the detector
    to call refinements opposed.

    Here the two scopes cross: first-year and part-time. A first-year full-time
    holder is in the first only, a part-time holder in a later year is in the
    second only, and a first-year part-time holder is in both with the outcomes
    disagreeing. That is what opposed means.
    """
    cat = rng.choice(_VISA_CATS)
    passages = [
        _passage("p0", f"{cat} holders in their first academic year may accept on-campus employment.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-01", f"doc_g_{idx}"),
        _passage("p1", f"{cat} holders enrolled on a part-time basis may not accept "
                       f"on-campus employment.",
                 SourceType.THIRD_PARTY, "2024-05", f"doc_t_{idx}"),
    ]
    branches = [
        Branch(branch_id="b0", condition="the holder is in their first academic year",
               outcome="on-campus employment permitted",
               applicability=Applicability(
                   descriptor=f"first-year {cat} holders",
                   attributes=[_cat_attr("enrolment_year", ["first"])]),
               supporting_passage="p0"),
        Branch(branch_id="b1",
               condition="the holder is enrolled on a part-time basis",
               outcome="on-campus employment prohibited",
               applicability=Applicability(
                   descriptor=f"part-time {cat} holders",
                   attributes=[_cat_attr("enrolment_mode", ["part_time"])]),
               supporting_passage="p1"),
    ]
    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=f"Can a first-year {cat} holder work on campus?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_scoped_answer=(
            "The sources genuinely disagree for the same population; the more authoritative "
            "source (p0, government guidance) is reported."
        ),
        selection_answer=f"{cat} holders in their first academic year may accept on-campus employment.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock",
                              disagreed=True, resolved_by_discussion=True),
    )


def _build_nested(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """An exception to an exception. Second-order -- flagged, never composed."""
    pct = rng.choice(_FEE_PCTS)
    txn = rng.choice(_TXN_KINDS)

    passages = [
        _passage("p0", f"{txn.capitalize()} incur a {pct} fee.".capitalize(),
                 SourceType.OFFICIAL_POLICY, "2024-03", f"doc_terms_{idx}"),
        _passage("p1", f"The fee is waived for premium-tier cardholders.",
                 SourceType.PRODUCT_TERMS, "2024-07", f"doc_prod_{idx}"),
        _passage("p2", f"The premium-tier waiver does not apply to transactions above 5,000.",
                 SourceType.AMENDMENT, "2025-03", f"doc_amend_{idx}"),
    ]
    branches = [
        Branch(branch_id="b0", condition=None, outcome=f"a {pct} fee applies",
               applicability=Applicability(descriptor="all cardholders", is_default=True),
               supporting_passage="p0"),
        Branch(branch_id="b1", condition="the cardholder holds a premium-tier card",
               outcome="no fee applies",
               applicability=Applicability(descriptor="premium-tier cardholders",
                                           attributes=[_tier_attr("card_tier", ["premium", "platinum", "signature"])]),
               supporting_passage="p1"),
        Branch(branch_id="b2",
               condition="the cardholder holds a premium-tier card and the amount exceeds 5,000",
               outcome=f"a {pct} fee applies",
               applicability=Applicability(
                   descriptor="premium-tier cardholders, transactions above 5,000",
                   attributes=[
                       _tier_attr("card_tier", ["premium", "platinum", "signature"]),
                       ScopeAttribute(name="amount", kind=AttributeKind.NUMERIC_RANGE, min_value=5000.0),
                   ]),
               supporting_passage="p2"),
    ]
    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=f"Do I pay a fee on a large {txn.rstrip('s')}?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_multi_exception_flags=MultiExceptionFlags(nested=True),
        gold_scoped_answer=(
            "Second-order structure: the premium waiver is itself carved back above 5,000. "
            "Outside this project's first-order scope; flagged and routed to selection."
        ),
        selection_answer=f"{txn.capitalize()} incur a {pct} fee.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build_crossed(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """Two independent exceptions that can co-occur and disagree."""
    cat = rng.choice(_VISA_CATS)
    passages = [
        _passage("p0", f"A {cat} holder may not accept employment.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-09", f"doc_policy_{idx}"),
        _passage("p1", "Holders with economic hardship authorisation may accept employment.",
                 SourceType.FAQ, "2024-04", f"doc_faq_{idx}"),
        _passage("p2", "Holders enrolled less than full-time may not accept employment "
                       "under any authorisation.",
                 SourceType.AMENDMENT, "2025-01", f"doc_amend_{idx}"),
    ]
    branches = [
        Branch(branch_id="b0", condition=None, outcome="employment prohibited",
               applicability=Applicability(descriptor=f"all {cat} holders", is_default=True),
               supporting_passage="p0"),
        Branch(branch_id="b1", condition="the holder has economic hardship authorisation",
               outcome="employment permitted",
               applicability=Applicability(
                   descriptor="holders with hardship authorisation",
                   attributes=[ScopeAttribute(name="hardship_auth", kind=AttributeKind.BOOLEAN,
                                              boolean_value=True)]),
               supporting_passage="p1"),
        Branch(branch_id="b2", condition="the holder is enrolled less than full-time",
               outcome="employment prohibited",
               applicability=Applicability(
                   descriptor="holders enrolled less than full-time",
                   attributes=[ScopeAttribute(name="full_time", kind=AttributeKind.BOOLEAN,
                                              boolean_value=False)]),
               supporting_passage="p2"),
    ]
    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=f"Can a part-time {cat} holder with hardship authorisation work?",
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=tpl.scope_relation,
        gold_branches=branches,
        gold_multi_exception_flags=MultiExceptionFlags(crossed=True),
        gold_scoped_answer=(
            "Two exceptions co-occur and disagree for this population. Outside this "
            "project's first-order scope; flagged and routed to selection."
        ),
        selection_answer=f"A {cat} holder may not accept employment.",
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build_distractor(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    """Factual / temporal / opinion / no-conflict negatives."""
    if tpl.conflict_type is ConflictType.FACTUAL:
        if tpl.domain is Domain.FINANCIAL_TERMS:
            a, b = rng.sample(_FEE_PCTS, 2)
            txn = rng.choice(_TXN_KINDS)
            passages = [
                _passage("p0", f"{txn.capitalize()} incur a {a} fee.".capitalize(),
                         SourceType.OFFICIAL_POLICY, "2024-03", f"doc_a_{idx}"),
                _passage("p1", f"{txn.capitalize()} incur a {b} fee.".capitalize(),
                         SourceType.THIRD_PARTY, "2024-04", f"doc_b_{idx}"),
            ]
            query = f"What is the fee on {txn}?"
            sel = f"{txn.capitalize()} incur a {a} fee."
        else:
            cat = rng.choice(_VISA_CATS)
            passages = [
                _passage("p0", f"The {cat} category permits a maximum stay of five years.",
                         SourceType.GOVERNMENT_GUIDANCE, "2024-02", f"doc_a_{idx}"),
                _passage("p1", f"The {cat} category permits a maximum stay of three years.",
                         SourceType.THIRD_PARTY, "2024-06", f"doc_b_{idx}"),
            ]
            query = f"How long can I stay on a {cat}?"
            sel = f"The {cat} category permits a maximum stay of five years."

    elif tpl.conflict_type is ConflictType.TEMPORAL:
        old, new = rng.sample(_FEE_PCTS, 2)
        txn = rng.choice(_TXN_KINDS)
        passages = [
            _passage("p0", f"As of 2022, {txn} incur a {old} fee.",
                     SourceType.OFFICIAL_POLICY, "2022-01", f"doc_old_{idx}"),
            _passage("p1", f"Effective January 2025, {txn} incur a {new} fee.",
                     SourceType.AMENDMENT, "2025-01", f"doc_new_{idx}"),
        ]
        query = f"What is the current fee on {txn}?"
        sel = f"Effective January 2025, {txn} incur a {new} fee."

    elif tpl.conflict_type is ConflictType.OPINION:
        cat = rng.choice(_VISA_CATS)
        passages = [
            _passage("p0", f"Most advisers consider the {cat} route the simplest option for students.",
                     SourceType.THIRD_PARTY, "2024-03", f"doc_a_{idx}"),
            _passage("p1", f"Many practitioners regard the {cat} route as unnecessarily restrictive.",
                     SourceType.THIRD_PARTY, "2024-09", f"doc_b_{idx}"),
        ]
        query = f"Is the {cat} route a good option?"
        sel = f"Most advisers consider the {cat} route the simplest option for students."

    else:  # NO_CONFLICT
        if tpl.domain is Domain.FINANCIAL_TERMS:
            pct = rng.choice(_FEE_PCTS)
            txn = rng.choice(_TXN_KINDS)
            passages = [
                _passage("p0", f"{txn.capitalize()} incur a {pct} fee.".capitalize(),
                         SourceType.OFFICIAL_POLICY, "2024-03", f"doc_a_{idx}"),
                _passage("p1", f"Statements are issued on the first business day of each month.",
                         SourceType.PRODUCT_TERMS, "2024-03", f"doc_b_{idx}"),
            ]
            query = f"What is the fee on {txn}?"
            sel = f"{txn.capitalize()} incur a {pct} fee."
        else:
            cat = rng.choice(_VISA_CATS)
            passages = [
                _passage("p0", f"A {cat} application requires a completed form and the filing fee.",
                         SourceType.GOVERNMENT_GUIDANCE, "2024-02", f"doc_a_{idx}"),
                _passage("p1", "Biometrics appointments are scheduled by the local field office.",
                         SourceType.FAQ, "2024-07", f"doc_b_{idx}"),
            ]
            query = f"What do I need for a {cat} application?"
            sel = f"A {cat} application requires a completed form and the filing fee."

    return GoldInstance(
        instance_id=f"mock_{tpl.name}_{idx:04d}",
        query=query,
        domain=tpl.domain,
        construction=Construction.NATURAL,
        passages=passages,
        gold_conflict_type=tpl.conflict_type,
        gold_scope_relation=None,
        is_distractor=True,
        gold_branches=[],
        gold_scoped_answer=sel,
        selection_answer=sel,
        annotation=Annotation(annotator_a="mock", annotator_b="mock"),
    )


def _build(rng: random.Random, idx: int, tpl: Template) -> GoldInstance:
    if tpl.nested:
        return _build_nested(rng, idx, tpl)
    if tpl.crossed:
        return _build_crossed(rng, idx, tpl)
    if tpl.is_distractor:
        return _build_distractor(rng, idx, tpl)
    if tpl.scope_relation is ScopeRelation.DISJOINT:
        return _build_disjoint(rng, idx, tpl)
    if tpl.scope_relation is ScopeRelation.REDUNDANT:
        return _build_redundant(rng, idx, tpl)
    if tpl.scope_relation is ScopeRelation.OPPOSED:
        return _build_opposed(rng, idx, tpl)
    if tpl.domain is Domain.FINANCIAL_TERMS:
        return _build_financial_refinement(rng, idx, tpl)
    return _build_immigration_refinement(rng, idx, tpl)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate(
    n: int = 50,
    *,
    seed: int = 0,
    tier2_fraction: float = 0.0,
    guarantee_coverage: bool = True,
) -> list[GoldInstance]:
    """Generate ``n`` deterministic mock gold instances.

    Parameters
    ----------
    n
        How many instances to produce.
    seed
        Random seed. The same seed must always produce identical output.
    tier2_fraction
        Fraction retagged ``construction='split'`` (Tier 2). Useful for
        exercising the Tier-1/Tier-2 breakdown reporting path before real
        Tier-2 data exists. Retagging also clears ``document_id`` collision
        checks by giving both passages the same document, which is what a real
        Tier-2 instance looks like.
    guarantee_coverage
        Emit one instance of every template before sampling by weight. Keeps
        small ``n`` from silently omitting a scope relation -- a fixture that
        happens to contain no ``redundant`` case cannot catch the
        redundant/refinement confusion it exists to catch.
    """
    rng = random.Random(seed)
    out: list[GoldInstance] = []

    ordered: list[Template] = []
    if guarantee_coverage:
        ordered.extend(TEMPLATES)
    population = [t for t in TEMPLATES for _ in range(t.weight)]
    while len(ordered) < n:
        ordered.append(rng.choice(population))
    ordered = ordered[:n]

    n_tier2 = int(round(n * tier2_fraction))
    tier2_idx = set(rng.sample(range(n), n_tier2)) if n_tier2 else set()

    for i, tpl in enumerate(ordered):
        inst = _build(rng, i, tpl)
        if i in tier2_idx:
            shared = f"doc_single_{i}"
            inst = inst.model_copy(
                update={
                    "construction": Construction.SPLIT,
                    "separation": Separation.SYNTHETIC_SPLIT,
                    "passages": [p.model_copy(update={"document_id": shared}) for p in inst.passages],
                }
            )
        else:
            inst = inst.model_copy(update={"separation": Separation.CROSS_DOCUMENT})
        out.append(inst)

    return out


def iter_records(n: int = 50, *, seed: int = 0, **kw) -> Iterator[dict]:
    """Yield wire-format :class:`~contract.models.QueryRecord` dicts."""
    for inst in generate(n, seed=seed, **kw):
        yield inst.to_query_record().model_dump(mode="json")


def coverage_report(instances: list[GoldInstance]) -> dict[str, dict[str, int]]:
    """Count instances per conflict type, scope relation, tier and flag.

    Printed by the CLI so a fixture's blind spots are visible at generation
    time rather than discovered as a mysteriously passing test.
    """
    rep: dict[str, dict[str, int]] = {
        "conflict_type": {}, "scope_relation": {}, "construction": {},
        "explicitness": {}, "flags": {},
    }
    for inst in instances:
        rep["conflict_type"][inst.gold_conflict_type.value] = (
            rep["conflict_type"].get(inst.gold_conflict_type.value, 0) + 1
        )
        key = inst.gold_scope_relation.value if inst.gold_scope_relation else "none"
        rep["scope_relation"][key] = rep["scope_relation"].get(key, 0) + 1
        rep["construction"][inst.construction.value] = (
            rep["construction"].get(inst.construction.value, 0) + 1
        )
        for b in inst.exception_branches:
            rep["explicitness"][b.explicitness.value] = (
                rep["explicitness"].get(b.explicitness.value, 0) + 1
            )
        if inst.gold_multi_exception_flags.nested:
            rep["flags"]["nested"] = rep["flags"].get("nested", 0) + 1
        if inst.gold_multi_exception_flags.crossed:
            rep["flags"]["crossed"] = rep["flags"].get("crossed", 0) + 1
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="number of instances (default 50)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    ap.add_argument("--tier2-fraction", type=float, default=0.0,
                    help="fraction retagged as Tier 2 'split' (default 0.0)")
    ap.add_argument("--format", choices=["gold", "record"], default="gold",
                    help="'gold' = full annotations; 'record' = wire contract only")
    ap.add_argument("-o", "--output", default="-", help="output path, or - for stdout")
    ap.add_argument("--coverage", action="store_true", help="print a coverage report to stderr")
    args = ap.parse_args(argv)

    instances = generate(args.n, seed=args.seed, tier2_fraction=args.tier2_fraction)

    lines = []
    for inst in instances:
        obj = inst.model_dump(mode="json") if args.format == "gold" \
            else inst.to_query_record().model_dump(mode="json")
        lines.append(json.dumps(obj, sort_keys=True, ensure_ascii=False))
    payload = "\n".join(lines) + "\n"

    if args.output == "-":
        sys.stdout.write(payload)
    else:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        print(f"wrote {len(instances)} instances to {args.output}", file=sys.stderr)

    if args.coverage:
        rep = coverage_report(instances)
        print("\ncoverage:", file=sys.stderr)
        for section, counts in rep.items():
            body = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
            print(f"  {section:16s} {body}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
