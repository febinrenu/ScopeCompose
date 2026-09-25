"""WP1 probe set — ~50 hand-built cases for Member B's feasibility probe.

**STATUS: model-proposed, human-verified 2026-09-25. Usable as gold.**

Every instance carries ``annotator_a="model-proposed"`` and an ``annotator_b``
recording the human sign-off. Both are kept: dropping the first would erase the
fact that a model drafted these, and that provenance is why the set can never
enter a kappa -- the two passes share a starting point, so their agreement
measures the proposal rather than the task.

The review changed three cases and dropped one; ``docs/wp1_probe_verification.md``
records every decision. Reviewing is much faster than authoring, which was the
point of drafting them this way.

**What the probe needs.** WP1 asks whether Contrastive Scope Probing recovers
*implicit* conditions better than direct extraction, without inflating the
Hallucinated-Condition Rate. So the set is built around the hard case:

* **28 implicit-condition instances.** The exception passage never signals that
  it is an exception. No "except", no "unless", no "does not apply to". It
  names a product, a scheme, a population or a category and states an outcome,
  and the narrowing is left for the reader to infer. Each one carries a
  ``why_implicit`` note saying exactly what the reader has to infer, which is
  also the rubric for verifying it.
* **8 explicit-condition instances.** The control arm. Explicit and implicit
  recovery are reported separately and never averaged, so the probe needs both.
* **14 distractors.** Factual, temporal, opinion, no-conflict and redundant
  pairs. Without these the Spurious-Condition Rate has no denominator and the
  probe cannot tell a method that finds real conditions from one that invents
  them everywhere.

**Why not reuse ``contract/mock.py``.** That generator is templated on purpose:
it exists so code can be tested deterministically. Its text is too regular to
probe extraction with — a method can succeed on it by pattern-matching the
template. These cases are written as varied prose, with the scope carried by
different devices each time (product name, scheme membership, tier, threshold,
jurisdiction, time window, status), so a method that only handles one device
is exposed.

**Provenance.** The prose is written for this benchmark, not copied from any
provider. It is realistic in shape but fictional in content, which keeps the
probe set free of licensing questions. Tier 1 / Tier 2 does not apply here:
these are hand-built probe cases, and the ``construction`` tag is set to
``split`` to keep them clearly separate from mined corpus instances.

Usage::

    python -m benchmark.wp1_probe_set -o benchmark/data/wp1_probe.jsonl
    python -m benchmark.wp1_probe_set --review          # human-readable review sheet
    python -m benchmark.wp1_probe_set --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

FIN = Domain.FINANCIAL_TERMS
IMM = Domain.IMMIGRATION_ELIGIBILITY

#: Recorded as ``annotator_b`` on every case: a human reviewed the whole set on
#: 2026-09-25 and signed off case by case (see docs/wp1_probe_verification.md).
#:
#: This is model-proposed plus human-adjudicated, which is NOT two independent
#: annotations. It is valid gold and it is not valid input to a kappa -- the two
#: passes share a starting point, so their agreement measures the proposal, not
#: the task. benchmark.annotation.agreement refuses it for that reason.
VERIFIED_BY = "human-verified-2026-09-25"


# --------------------------------------------------------------------------- #
# Attribute helpers
# --------------------------------------------------------------------------- #


def cat(name: str, *values: str) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.CATEGORICAL, values=list(values))


def tier(name: str, *values: str) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.ORDERED_TIER, values=list(values))


def num(name: str, lo: float | None = None, hi: float | None = None) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.NUMERIC_RANGE,
                          min_value=lo, max_value=hi)


def flag(name: str, value: bool = True) -> ScopeAttribute:
    return ScopeAttribute(name=name, kind=AttributeKind.BOOLEAN, boolean_value=value)


# --------------------------------------------------------------------------- #
# Case specification
# --------------------------------------------------------------------------- #


@dataclass
class Src:
    """One passage's text and provenance."""

    text: str
    source: SourceType
    date: str
    doc: str


@dataclass
class Case:
    """One probe instance, in a form a human can review quickly."""

    id: str
    domain: Domain
    query: str
    rule: Src
    other: Src

    conflict_type: ConflictType
    relation: ScopeRelation | None = None

    # conditional cases only
    default_outcome: str = ""
    condition: str = ""
    exception_outcome: str = ""
    applicability: str = ""
    attrs: list[ScopeAttribute] = field(default_factory=list)
    explicitness: Explicitness = Explicitness.IMPLICIT

    # OPPOSED / DISJOINT only: the rule branch is itself scoped.
    #
    # For refinement and redundant, p0 states an unconditioned general rule and
    # becomes the default branch. For opposed and disjoint it does not: "Reward
    # account holders get lounge access" is already restricted to Reward
    # holders, and "instant access savings earn 4.50%" to instant-access
    # accounts. Encoding either as a default whose applicability is the whole
    # case space makes the other branch a subset of it -- that is refinement,
    # contradicting the relation the case exists to test.
    rule_condition: str = ""
    rule_applicability: str = ""
    rule_attrs: list[ScopeAttribute] = field(default_factory=list)

    # A third branch, nested inside the exception. Set only where the exception
    # passage carves itself back again ("five years, or four for Swiss
    # nationals"). Encoding that as two branches forces one of the three real
    # outcomes to be dropped or folded into another, and a system that
    # correctly produced all three would then be penalised for the extra one.
    sub_condition: str = ""
    sub_outcome: str = ""
    sub_applicability: str = ""
    sub_attrs: list[ScopeAttribute] = field(default_factory=list)

    scoped_answer: str = ""
    selection_answer: str = ""

    why: str = ""
    """What the reader must infer (implicit cases), or why this is a distractor.
    This doubles as the verification rubric for Member A."""

    is_distractor: bool = False
    merged_outcome: str = ""
    """For REDUNDANT cases, which resolve to a single merged branch."""

    def build(self) -> GoldInstance:
        passages = [
            Passage(id="p0", text=self.rule.text, source_type=self.rule.source,
                    date=self.rule.date, document_id=self.rule.doc,
                    source_url=f"https://example.invalid/{self.rule.doc}"),
            Passage(id="p1", text=self.other.text, source_type=self.other.source,
                    date=self.other.date, document_id=self.other.doc,
                    source_url=f"https://example.invalid/{self.other.doc}"),
        ]

        branches: list[Branch] = []
        if self.conflict_type is ConflictType.CONDITIONAL:
            if self.relation is ScopeRelation.REDUNDANT:
                branches = [Branch(
                    branch_id="b0", condition=None,
                    outcome=self.merged_outcome or self.default_outcome,
                    applicability=Applicability(descriptor="all cases", is_default=True),
                    supporting_passage="p0",
                )]
            elif self.rule_condition:
                # Two conditioned branches, no default. See rule_condition.
                branches = [
                    Branch(branch_id="b0", condition=self.rule_condition,
                           outcome=self.default_outcome,
                           applicability=Applicability(
                               descriptor=self.rule_applicability,
                               attributes=list(self.rule_attrs)),
                           supporting_passage="p0"),
                    Branch(branch_id="b1", condition=self.condition,
                           outcome=self.exception_outcome,
                           applicability=Applicability(descriptor=self.applicability,
                                                       attributes=list(self.attrs)),
                           explicitness=self.explicitness,
                           supporting_passage="p1"),
                ]
            else:
                branches = [
                    Branch(branch_id="b0", condition=None, outcome=self.default_outcome,
                           applicability=Applicability(descriptor="all cases", is_default=True),
                           supporting_passage="p0"),
                    Branch(branch_id="b1", condition=self.condition,
                           outcome=self.exception_outcome,
                           applicability=Applicability(descriptor=self.applicability,
                                                       attributes=list(self.attrs)),
                           explicitness=self.explicitness,
                           supporting_passage="p1"),
                ]

            if self.sub_condition:
                branches.append(Branch(
                    branch_id="b2", condition=self.sub_condition,
                    outcome=self.sub_outcome,
                    applicability=Applicability(descriptor=self.sub_applicability,
                                                attributes=list(self.sub_attrs)),
                    explicitness=self.explicitness,
                    supporting_passage="p1"))

        return GoldInstance(
            instance_id=self.id,
            query=self.query,
            domain=self.domain,
            # 'split' on purpose: these are hand-built probe cases, not mined
            # instances, and must never be counted toward the Tier-1 total.
            construction=Construction.SPLIT,
            # Hand-authored, so the separation is manufactured by definition.
            separation=Separation.SYNTHETIC_SPLIT,
            split="dev",
            passages=passages,
            gold_conflict_type=self.conflict_type,
            gold_scope_relation=self.relation,
            is_distractor=self.is_distractor,
            gold_branches=branches,
            gold_scoped_answer=self.scoped_answer or None,
            selection_answer=self.selection_answer or None,
            # A sub-branch nested inside the exception is second-order
            # structure, which this project flags rather than composes. Setting
            # the flag here is what makes the gold agree with what B2 will do
            # with the instance.
            gold_multi_exception_flags=MultiExceptionFlags(
                nested=bool(self.sub_condition)),
            annotation=Annotation(
                annotator_a="model-proposed",
                annotator_b=VERIFIED_BY,
                notes=self.why or None,
            ),
        )


# --------------------------------------------------------------------------- #
# A. Implicit-condition instances  (the core of the probe)
# --------------------------------------------------------------------------- #
#
# In every one of these the exception passage contains NO exception cue. The
# narrowing is carried by a product name, a scheme, a tier, a threshold, a
# jurisdiction, a time window or a status. The devices are varied on purpose:
# a method that only handles one of them should visibly fail on the others.

IMPLICIT: list[Case] = [
    Case(
        id="wp1_fin_imp_001", domain=FIN,
        query="Will I be charged for withdrawing cash abroad?",
        rule=Src("Cash withdrawals at non-network ATMs are charged at 2.50 per transaction, "
                 "plus any fee levied by the ATM operator.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "bank_tariff"),
        other=Src("Your Global Traveller account includes unlimited fee-free cash "
                  "withdrawals worldwide.",
                  SourceType.PRODUCT_TERMS, "2024-09", "bank_globaltraveller"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 2.50 per-transaction charge applies",
        condition="the account is a Global Traveller account",
        exception_outcome="no withdrawal charge applies",
        applicability="Global Traveller account holders",
        attrs=[cat("account_product", "global_traveller")],
        scoped_answer="Non-network ATM withdrawals normally cost 2.50 per transaction "
                      "(per p0), but Global Traveller accounts include unlimited "
                      "fee-free withdrawals worldwide (per p1).",
        selection_answer="Cash withdrawals at non-network ATMs are charged at 2.50 "
                         "per transaction.",
        why="Device: PRODUCT NAME. p1 states a benefit of a named product and never "
            "signals that it overrides the tariff. The reader must infer that "
            "'includes unlimited fee-free' contradicts '2.50 per transaction' for "
            "that product only.",
    ),
    Case(
        id="wp1_fin_imp_002", domain=FIN,
        query="What is the conversion charge on a euro purchase?",
        rule=Src("Transactions in a currency other than sterling are subject to a "
                 "2.99% currency conversion charge.",
                 SourceType.OFFICIAL_POLICY, "2023-11", "bank_fx_terms"),
        other=Src("Euro transactions on cards issued in the Eurozone are settled at the "
                  "interbank reference rate published by the European Central Bank.",
                  SourceType.CATEGORY_PAGE, "2025-01", "bank_eurozone_cards"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 2.99% conversion charge applies",
        condition="the transaction is in euro on a Eurozone-issued card",
        exception_outcome="settled at the interbank reference rate, no conversion charge",
        applicability="euro transactions on Eurozone-issued cards",
        attrs=[cat("currency", "EUR"), cat("card_issue_region", "eurozone")],
        scoped_answer="Non-sterling transactions normally carry a 2.99% conversion "
                      "charge (per p0); euro transactions on Eurozone-issued cards are "
                      "settled at the ECB interbank rate instead (per p1).",
        selection_answer="A 2.99% currency conversion charge applies.",
        why="Device: JURISDICTION + CURRENCY. Two attributes must both hold. p1 never "
            "says 'no charge' -- the reader must infer that settling at the interbank "
            "rate means the 2.99% markup is not applied.",
    ),
    Case(
        id="wp1_fin_imp_003", domain=FIN,
        query="Is there a monthly fee on my current account?",
        rule=Src("A monthly account fee of 12.00 is payable on all current accounts.",
                 SourceType.OFFICIAL_POLICY, "2024-04", "bank_charges"),
        other=Src("Accounts opened under the Graduate Scheme carry no monthly charge "
                  "for the first three years after registration.",
                  SourceType.FAQ, "2024-10", "bank_graduate_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 12.00 monthly fee applies",
        condition="the account was opened under the Graduate Scheme and is within three "
                  "years of registration",
        exception_outcome="no monthly fee applies",
        applicability="Graduate Scheme accounts within three years of registration",
        attrs=[cat("scheme", "graduate"), num("years_since_registration", hi=3)],
        scoped_answer="Current accounts normally carry a 12.00 monthly fee (per p0). "
                      "Graduate Scheme accounts pay nothing for their first three years "
                      "(per p1).",
        selection_answer="A monthly account fee of 12.00 is payable on all current accounts.",
        why="Device: SCHEME MEMBERSHIP + TIME WINDOW. The exception is bounded in time, "
            "so the applicability set is a conjunction. A method that recovers the "
            "scheme but drops the three-year bound has extracted a WIDER condition than "
            "the text supports -- worth catching separately from a total miss.",
    ),
    Case(
        id="wp1_fin_imp_004", domain=FIN,
        query="Do I pay a fee to send money overseas?",
        rule=Src("International payments are subject to a 15.00 handling fee per transfer.",
                 SourceType.OFFICIAL_POLICY, "2024-01", "bank_payments"),
        other=Src("Transfers between accounts you hold with us, including those in "
                  "different currencies, are processed free of charge.",
                  SourceType.PRODUCT_TERMS, "2024-06", "bank_internal_transfers"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 15.00 handling fee applies",
        condition="the transfer is between accounts held by the same customer",
        exception_outcome="no handling fee applies",
        applicability="transfers between own accounts",
        attrs=[flag("same_customer_accounts", True)],
        scoped_answer="International payments normally carry a 15.00 handling fee "
                      "(per p0), but transfers between your own accounts are free, even "
                      "across currencies (per p1).",
        selection_answer="International payments are subject to a 15.00 handling fee.",
        why="Device: RELATIONSHIP BETWEEN PARTIES. 'Accounts you hold with us' narrows "
            "by who owns them, not by product or amount. The phrase 'including those in "
            "different currencies' is the only hint that this overlaps the international "
            "rule at all.",
    ),
    Case(
        id="wp1_fin_imp_005", domain=FIN,
        query="Will I be charged interest on a purchase?",
        rule=Src("Interest accrues on purchase balances at the standard rate from the "
                 "transaction date.",
                 SourceType.OFFICIAL_POLICY, "2023-08", "card_terms"),
        other=Src("Where the statement balance is settled in full by the payment due "
                  "date, purchases benefit from up to 56 days of interest-free credit.",
                  SourceType.FAQ, "2024-12", "card_interest_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="interest accrues from the transaction date",
        condition="the statement balance is paid in full by the due date",
        exception_outcome="no interest accrues, up to 56 days",
        applicability="accounts settled in full by the due date",
        attrs=[flag("paid_in_full_by_due_date", True)],
        scoped_answer="Interest normally accrues on purchases from the transaction date "
                      "(per p0); if you settle the statement balance in full by the due "
                      "date, purchases are interest-free for up to 56 days (per p1).",
        selection_answer="Interest accrues on purchase balances from the transaction date.",
        why="Device: BEHAVIOURAL CONDITION. The narrowing depends on what the customer "
            "does, not on a product attribute. 'Where the statement balance is settled' "
            "reads as descriptive, not as a carve-out.",
    ),
    Case(
        id="wp1_fin_imp_006", domain=FIN,
        query="Is there a charge for a paper statement?",
        rule=Src("Paper statements are provided at a cost of 3.00 per statement.",
                 SourceType.OFFICIAL_POLICY, "2024-03", "bank_statements"),
        other=Src("Customers registered as requiring accessible formats receive printed "
                  "correspondence in large print at no cost.",
                  SourceType.CATEGORY_PAGE, "2024-11", "bank_accessibility"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 3.00 per-statement charge applies",
        condition="the customer is registered as requiring accessible formats",
        exception_outcome="no charge applies",
        applicability="customers registered for accessible formats",
        attrs=[flag("accessible_format_registered", True)],
        scoped_answer="Paper statements normally cost 3.00 each (per p0), but customers "
                      "registered as requiring accessible formats receive printed "
                      "correspondence free of charge (per p1).",
        selection_answer="Paper statements are provided at a cost of 3.00 per statement.",
        why="Device: CUSTOMER STATUS. Note the vocabulary barely overlaps -- 'paper "
            "statements' vs 'printed correspondence in large print'. Lexical-overlap "
            "methods should struggle here, which is exactly what makes it informative.",
    ),
    Case(
        id="wp1_fin_imp_007", domain=FIN,
        query="What is the overdraft interest rate?",
        rule=Src("Arranged overdrafts are charged at 39.9% EAR variable.",
                 SourceType.OFFICIAL_POLICY, "2024-05", "bank_overdraft"),
        other=Src("The first 500 of any arranged overdraft on a Reward account is "
                  "provided at 0% EAR.",
                  SourceType.PRODUCT_TERMS, "2024-07", "bank_reward_account"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="39.9% EAR variable applies",
        condition="the account is a Reward account and the balance is within the first 500",
        exception_outcome="0% EAR applies",
        applicability="the first 500 of an arranged overdraft on a Reward account",
        attrs=[cat("account_product", "reward"), num("overdraft_amount", hi=500)],
        scoped_answer="Arranged overdrafts are normally 39.9% EAR (per p0); on a Reward "
                      "account the first 500 is at 0% EAR (per p1).",
        selection_answer="Arranged overdrafts are charged at 39.9% EAR variable.",
        why="Device: PRODUCT + AMOUNT THRESHOLD. The exception is partial -- it applies "
            "to a slice of the balance, not to the whole account. A method that "
            "extracts 'Reward accounts pay 0%' has over-generalised.",
    ),
    Case(
        id="wp1_fin_imp_008", domain=FIN,
        query="Is a fee charged for early repayment?",
        rule=Src("An early repayment charge equal to two months' interest is applied "
                 "where a fixed-rate loan is settled before the end of its term.",
                 SourceType.OFFICIAL_POLICY, "2023-09", "loan_terms"),
        other=Src("Loans taken out on or after 1 April 2024 may be repaid at any time "
                  "with no further amount payable.",
                  SourceType.AMENDMENT, "2024-04", "loan_amendment_2024"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="an early repayment charge of two months' interest applies",
        condition="the loan was taken out on or after 1 April 2024",
        exception_outcome="no early repayment charge applies",
        applicability="loans originated on or after 1 April 2024",
        attrs=[cat("origination_period", "2024-04-onwards")],
        scoped_answer="Settling a fixed-rate loan early normally incurs a charge of two "
                      "months' interest (per p0); loans taken out on or after 1 April "
                      "2024 can be repaid at any time with nothing further to pay (per p1).",
        selection_answer="An early repayment charge equal to two months' interest applies.",
        why="Device: ORIGINATION DATE. Deliberately adjacent to a TEMPORAL conflict and "
            "must NOT be labelled one: both passages are currently in force, and which "
            "applies depends on when the loan started, not on which text is newer. This "
            "is a key annotation-boundary case for Member A to check.",
    ),
    Case(
        id="wp1_fin_imp_009", domain=FIN,
        query="Do I need to maintain a minimum balance?",
        rule=Src("A minimum balance of 1,000 must be maintained to avoid a quarterly "
                 "service charge.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "bank_balance_terms"),
        other=Src("Our Young Saver commitment covers customers aged 18 to 24, whose "
                  "accounts operate without a minimum balance.",
                  SourceType.CATEGORY_PAGE, "2024-08", "bank_youngsaver"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 1,000 minimum balance is required",
        condition="the account holder is aged between 18 and 24",
        exception_outcome="no minimum balance requirement",
        applicability="account holders aged 18-24",
        attrs=[num("age", 18, 24)],
        scoped_answer="A 1,000 minimum balance is normally required to avoid the "
                      "quarterly service charge (per p0); customers aged 18 to 24 are "
                      "exempt under the Young Saver commitment (per p1).",
        selection_answer="A minimum balance of 1,000 must be maintained.",
        why="Device: AGE RANGE. Straightforward numeric applicability, included so the "
            "probe has an easy implicit case to calibrate against. Originally worded "
            "with 'exempt from', which is a strong exception cue and would have made "
            "this an explicit case mislabelled as implicit -- caught by the cue check "
            "below and reworded.",
    ),
    Case(
        id="wp1_fin_imp_010", domain=FIN,
        query="Are card payments abroad covered by purchase protection?",
        rule=Src("Purchase protection covers eligible items bought with the card against "
                 "damage or theft for 90 days.",
                 SourceType.OFFICIAL_POLICY, "2024-01", "card_protection"),
        other=Src("Items bought through peer-to-peer marketplaces and private sellers "
                  "fall outside the scope of the protection programme.",
                  SourceType.FAQ, "2024-09", "card_protection_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="purchases are covered for 90 days",
        condition="the item was bought through a peer-to-peer marketplace or private seller",
        exception_outcome="no cover applies",
        applicability="purchases from peer-to-peer marketplaces and private sellers",
        attrs=[cat("seller_type", "peer_to_peer", "private")],
        scoped_answer="Purchase protection covers eligible items for 90 days (per p0), "
                      "except that items bought from peer-to-peer marketplaces or "
                      "private sellers are not covered (per p1).",
        selection_answer="Purchase protection covers eligible items for 90 days.",
        why="Device: CHANNEL/SELLER TYPE. Note this exception NARROWS cover rather than "
            "granting a benefit -- most of the set grants benefits, so a method biased "
            "toward 'exceptions are good news' should be caught here.",
    ),
    Case(
        id="wp1_imm_imp_011", domain=IMM,
        query="Can I work while studying?",
        rule=Src("Holders of a student visa are not permitted to undertake employment "
                 "during the validity of their leave.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-06", "gov_student_conditions"),
        other=Src("Students sponsored by a higher education provider with track record "
                  "status may work up to 20 hours per week during term time.",
                  SourceType.CATEGORY_PAGE, "2024-10", "gov_sponsor_categories"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="employment is not permitted",
        condition="the student is sponsored by a provider with track record status",
        exception_outcome="employment permitted up to 20 hours per week in term time",
        applicability="students sponsored by track-record providers",
        attrs=[flag("sponsor_track_record", True), num("weekly_hours", hi=20)],
        scoped_answer="Student visa holders generally may not work (per p0), but those "
                      "sponsored by a provider with track record status may work up to "
                      "20 hours per week during term time (per p1).",
        selection_answer="Holders of a student visa are not permitted to undertake employment.",
        why="Device: SPONSOR ATTRIBUTE. The narrowing is a property of the institution, "
            "not the applicant -- a class of condition the probe should be tested on, "
            "since it requires resolving who 'sponsored by' refers to.",
    ),
    Case(
        id="wp1_imm_imp_012", domain=IMM,
        query="Does time spent outside the country break continuous residence?",
        rule=Src("Applicants must demonstrate five years of continuous residence, with "
                 "no single absence exceeding 180 days.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-03", "gov_settlement"),
        other=Src("Periods spent overseas in Crown service, or accompanying a partner in "
                  "Crown service, are treated as time spent in the United Kingdom.",
                  SourceType.CATEGORY_PAGE, "2024-05", "gov_crown_service"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="absences over 180 days break continuous residence",
        condition="the absence was in Crown service, or accompanying a partner in Crown service",
        exception_outcome="the period counts as residence and does not break continuity",
        applicability="absences in Crown service",
        attrs=[flag("crown_service", True)],
        scoped_answer="Continuous residence normally requires no single absence over 180 "
                      "days (per p0); time overseas in Crown service, or accompanying a "
                      "partner in Crown service, counts as time in the UK (per p1).",
        selection_answer="Applicants must demonstrate five years of continuous residence "
                         "with no single absence exceeding 180 days.",
        why="Device: RECHARACTERISATION. p1 does not create an exemption -- it redefines "
            "what counts as residence, which reaches the same outcome by a different "
            "route. A method looking for 'X does not apply' will miss it entirely.",
    ),
    Case(
        id="wp1_imm_imp_013", domain=IMM,
        query="Do I have to pay the immigration health surcharge?",
        rule=Src("The immigration health surcharge is payable by all applicants for "
                 "leave to remain of more than six months.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-02", "gov_ihs"),
        other=Src("Applicants under the Health and Care Worker route, and their "
                  "dependants, are not required to make this payment.",
                  SourceType.CATEGORY_PAGE, "2024-07", "gov_health_care_worker"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="the health surcharge is payable",
        condition="the applicant is on the Health and Care Worker route, or is a dependant",
        exception_outcome="the surcharge is not payable",
        applicability="Health and Care Worker route applicants and dependants",
        attrs=[cat("route", "health_and_care_worker")],
        scoped_answer="The immigration health surcharge is normally payable on "
                      "applications for more than six months (per p0); Health and Care "
                      "Worker route applicants and their dependants are not required to "
                      "pay it (per p1).",
        selection_answer="The immigration health surcharge is payable by all applicants "
                         "for leave of more than six months.",
        why="Device: ROUTE MEMBERSHIP, with the exception extending to dependants. "
            "Check whether the probe recovers the dependant extension or only the "
            "primary applicant -- a partial applicability is a distinct error from a "
            "missed one.",
    ),
    Case(
        id="wp1_imm_imp_014", domain=IMM,
        query="Is an English language test required?",
        rule=Src("Applicants must provide evidence of English language ability at level "
                 "B1 or above.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-10", "gov_english"),
        other=Src("Nationals of majority English-speaking countries listed in Appendix "
                  "ENG meet this requirement on the basis of nationality alone.",
                  SourceType.CATEGORY_PAGE, "2024-06", "gov_appendix_eng"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="evidence of B1 English is required",
        condition="the applicant is a national of a country listed in Appendix ENG",
        exception_outcome="the requirement is met by nationality, no test needed",
        applicability="nationals of Appendix ENG countries",
        attrs=[flag("appendix_eng_national", True)],
        scoped_answer="Applicants must normally evidence English at B1 or above (per p0); "
                      "nationals of majority English-speaking countries in Appendix ENG "
                      "meet the requirement by nationality alone (per p1).",
        selection_answer="Applicants must provide evidence of English language ability at "
                         "level B1 or above.",
        why="Device: EXTERNAL LIST REFERENCE. The applicability set is defined by "
            "pointing at a document the passage does not contain. The condition is "
            "recoverable, but the *membership* is not -- a good test of whether the "
            "probe over-claims by inventing a country list.",
    ),
    Case(
        id="wp1_imm_imp_015", domain=IMM,
        query="Can my partner apply at the same time?",
        rule=Src("Dependants may apply only once the main applicant has been granted leave.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-12", "gov_dependants"),
        other=Src("Where an application is made under the Global Talent route, family "
                  "members may be included in the same application.",
                  SourceType.CATEGORY_PAGE, "2024-09", "gov_global_talent"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="dependants must wait until the main applicant is granted leave",
        condition="the application is under the Global Talent route",
        exception_outcome="family members may be included in the same application",
        applicability="Global Talent route applications",
        attrs=[cat("route", "global_talent")],
        scoped_answer="Dependants normally apply only after the main applicant is granted "
                      "leave (per p0); under the Global Talent route family members can "
                      "be included in the same application (per p1).",
        selection_answer="Dependants may apply only once the main applicant has been "
                         "granted leave.",
        why="Device: ROUTE MEMBERSHIP changing a procedural rule rather than an "
            "eligibility outcome. Included so the probe is not only tested on "
            "fee/permission outcomes.",
    ),
    Case(
        id="wp1_imm_imp_016", domain=IMM,
        query="How long can I stay outside the country?",
        rule=Src("Leave will lapse where the holder remains outside the country for a "
                 "continuous period of two years.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-05", "gov_lapsing"),
        other=Src("For holders of settled status, the relevant period is five years, or "
                  "four years for Swiss nationals and their family members.",
                  SourceType.CATEGORY_PAGE, "2024-11", "gov_settled_status"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="leave lapses after two years outside the country",
        condition="the holder has settled status",
        exception_outcome="the period is five years",
        applicability="holders of settled status",
        attrs=[flag("settled_status", True)],
        sub_condition="the holder is a Swiss national or their family member with "
                      "settled status",
        sub_outcome="the period is four years",
        sub_applicability="Swiss nationals and their family members with settled status",
        sub_attrs=[flag("settled_status", True), cat("nationality", "swiss")],
        scoped_answer="Leave normally lapses after two continuous years abroad (per p0); "
                      "for settled status holders the period is five years, and four "
                      "years for Swiss nationals and their family members (per p1).",
        selection_answer="Leave will lapse after a continuous period of two years outside "
                         "the country.",
        why="Device: NESTED SUB-CASE. Three distinct outcomes, not two -- two years "
            "generally, five with settled status, four for Swiss nationals holding "
            "settled status. The Swiss branch is nested inside the settled-status "
            "branch and disagrees with it, which is second-order structure: gold "
            "carries nested=True and the instance is expected to be FLAGGED rather "
            "than composed. Encoding it as two branches would have forced one of the "
            "three real outcomes to be dropped, and penalised a system that "
            "recovered all three.",
    ),
    Case(
        id="wp1_fin_imp_017", domain=FIN,
        query="Is there a limit on contactless payments?",
        rule=Src("Contactless payments are limited to 100 per transaction.",
                 SourceType.OFFICIAL_POLICY, "2024-03", "card_contactless"),
        other=Src("Payments authenticated through the mobile banking app are processed "
                  "without a per-transaction ceiling.",
                  SourceType.PRODUCT_TERMS, "2024-08", "card_mobile_app"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 100 per-transaction limit applies",
        condition="the payment is authenticated through the mobile banking app",
        exception_outcome="no per-transaction ceiling applies",
        applicability="app-authenticated payments",
        attrs=[cat("authentication_method", "mobile_app")],
        scoped_answer="Contactless payments are normally capped at 100 per transaction "
                      "(per p0); payments authenticated through the mobile banking app "
                      "have no per-transaction ceiling (per p1).",
        selection_answer="Contactless payments are limited to 100 per transaction.",
        why="Device: AUTHENTICATION METHOD. Neither passage uses the other's vocabulary "
            "('contactless' vs 'authenticated through the app'), so the reader must know "
            "that app-authenticated payments are a kind of contactless payment.",
    ),
    Case(
        id="wp1_fin_imp_018", domain=FIN,
        query="What happens if a direct debit fails?",
        rule=Src("A returned payment fee of 10.00 is charged where a direct debit cannot "
                 "be met.",
                 SourceType.OFFICIAL_POLICY, "2024-01", "bank_returned_payments"),
        other=Src("Where funds are credited before the end of the same working day, the "
                  "payment is re-presented at no cost to the customer.",
                  SourceType.FAQ, "2024-10", "bank_payments_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="a 10.00 returned payment fee applies",
        condition="funds are credited before the end of the same working day",
        exception_outcome="the payment is re-presented at no cost",
        applicability="payments funded before end of the same working day",
        attrs=[flag("funded_same_working_day", True)],
        scoped_answer="A failed direct debit normally incurs a 10.00 returned payment fee "
                      "(per p0); if funds arrive before the end of the same working day "
                      "the payment is re-presented free of charge (per p1).",
        selection_answer="A returned payment fee of 10.00 is charged where a direct debit "
                         "cannot be met.",
        why="Device: TIMING WINDOW measured in working days. Tests whether the probe "
            "preserves the precise boundary or degrades it to 'if you pay quickly'.",
    ),
    Case(
        id="wp1_fin_imp_019", domain=FIN,
        query="Can I get a refund on an annual fee?",
        rule=Src("The annual card fee is non-refundable once charged.",
                 SourceType.OFFICIAL_POLICY, "2023-07", "card_annual_fee"),
        other=Src("Where an account is closed within 14 days of the fee being applied, "
                  "the amount is returned in full.",
                  SourceType.AMENDMENT, "2024-02", "card_cooling_off"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="the annual fee is non-refundable",
        condition="the account is closed within 14 days of the fee being applied",
        exception_outcome="the fee is refunded in full",
        applicability="accounts closed within 14 days of the charge",
        attrs=[num("days_since_charge", hi=14)],
        scoped_answer="The annual card fee is normally non-refundable (per p0), but it is "
                      "returned in full if the account is closed within 14 days of the "
                      "fee being applied (per p1).",
        selection_answer="The annual card fee is non-refundable once charged.",
        why="Device: DEADLINE. Clean numeric boundary. A direct contradiction on the "
            "surface ('non-refundable' vs 'returned in full') that is NOT a factual "
            "conflict -- a good test of the factual/conditional boundary for Member A's "
            "detector as well as for the probe.",
    ),
    Case(
        id="wp1_fin_imp_020", domain=FIN,
        query="Is my deposit protected?",
        rule=Src("Eligible deposits are protected up to 85,000 per depositor.",
                 SourceType.OFFICIAL_POLICY, "2024-04", "bank_deposit_protection"),
        other=Src("Balances arising from a property sale are covered up to 1 million for "
                  "six months from the date of credit.",
                  SourceType.CATEGORY_PAGE, "2024-09", "bank_temporary_balances"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="protection is capped at 85,000 per depositor",
        condition="the balance arises from a property sale and is within six months of credit",
        exception_outcome="protection is up to 1 million",
        applicability="property-sale balances within six months of credit",
        attrs=[cat("balance_origin", "property_sale"), num("months_since_credit", hi=6)],
        scoped_answer="Deposits are normally protected up to 85,000 per depositor (per "
                      "p0); balances from a property sale are covered up to 1 million for "
                      "six months from credit (per p1).",
        selection_answer="Eligible deposits are protected up to 85,000 per depositor.",
        why="Device: ORIGIN OF FUNDS + TIME WINDOW. The exception RAISES a limit rather "
            "than removing a charge, and the two numbers (85,000 vs 1 million) could "
            "easily be read as a factual contradiction. Another boundary case.",
    ),
    Case(
        id="wp1_imm_imp_021", domain=IMM,
        query="Do I need to show maintenance funds?",
        rule=Src("Applicants must show funds of 1,270 held for 28 consecutive days.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-01", "gov_maintenance"),
        other=Src("Where the sponsor certifies maintenance on the certificate of "
                  "sponsorship, no evidence of funds is required from the applicant.",
                  SourceType.CATEGORY_PAGE, "2024-08", "gov_sponsor_certification"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="funds of 1,270 must be evidenced for 28 days",
        condition="the sponsor has certified maintenance on the certificate of sponsorship",
        exception_outcome="no evidence of funds is required",
        applicability="applicants whose sponsor certifies maintenance",
        attrs=[flag("sponsor_certifies_maintenance", True)],
        scoped_answer="Applicants normally show 1,270 held for 28 consecutive days (per "
                      "p0); where the sponsor certifies maintenance on the certificate of "
                      "sponsorship, no evidence of funds is needed (per p1).",
        selection_answer="Applicants must show funds of 1,270 held for 28 consecutive days.",
        why="Device: THIRD-PARTY ACTION. The condition depends on something the sponsor "
            "does, not on any attribute of the applicant.",
    ),
    Case(
        id="wp1_imm_imp_022", domain=IMM,
        query="Can I switch to another visa from inside the country?",
        rule=Src("An application to switch into this route must be made from outside the "
                 "United Kingdom.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-11", "gov_switching"),
        other=Src("Those currently holding leave as a student or a graduate may make the "
                  "application without leaving.",
                  SourceType.FAQ, "2024-07", "gov_switching_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="the application must be made from outside the country",
        condition="the applicant currently holds student or graduate leave",
        exception_outcome="the application may be made from inside the country",
        applicability="current student and graduate leave holders",
        attrs=[cat("current_leave", "student", "graduate")],
        scoped_answer="Applications to switch into this route are normally made from "
                      "outside the UK (per p0); current student and graduate leave "
                      "holders can apply without leaving (per p1).",
        selection_answer="An application to switch into this route must be made from "
                         "outside the United Kingdom.",
        why="Device: CURRENT STATUS as a category set with two members.",
    ),
    Case(
        id="wp1_imm_imp_023", domain=IMM,
        query="Is there an age limit?",
        rule=Src("Applicants must be aged 18 or over on the date of application.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-04", "gov_age"),
        other=Src("Applications made as part of a family group are accepted for children "
                  "aged 16 and 17 where both parents are included.",
                  SourceType.CATEGORY_PAGE, "2024-10", "gov_family_group"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="applicants must be 18 or over",
        condition="the application is part of a family group including both parents, and "
                  "the applicant is 16 or 17",
        exception_outcome="the application is accepted",
        applicability="16- and 17-year-olds in a family group with both parents",
        attrs=[num("age", 16, 17), flag("family_group_both_parents", True)],
        scoped_answer="Applicants must normally be 18 or over (per p0); children aged 16 "
                      "and 17 are accepted as part of a family group where both parents "
                      "are included (per p1).",
        selection_answer="Applicants must be aged 18 or over on the date of application.",
        why="Device: AGE RANGE + CONJUNCTIVE CONDITION. Two things must both hold. Tests "
            "whether the probe recovers the conjunction or only the age.",
    ),
    Case(
        id="wp1_fin_imp_024", domain=FIN,
        query="What notice do I need to give to close a savings account?",
        rule=Src("Ninety days' notice is required to withdraw funds from a notice account.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "savings_terms"),
        other=Src("Funds may be released immediately on production of a grant of probate.",
                  SourceType.CATEGORY_PAGE, "2024-06", "savings_bereavement"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="ninety days' notice is required",
        condition="a grant of probate is produced",
        exception_outcome="funds are released immediately",
        applicability="accounts where probate has been granted",
        attrs=[flag("probate_granted", True)],
        scoped_answer="Notice accounts normally require ninety days' notice (per p0); "
                      "funds are released immediately on production of a grant of probate "
                      "(per p1).",
        selection_answer="Ninety days' notice is required to withdraw funds from a notice "
                         "account.",
        why="Device: DOCUMENTARY TRIGGER. Very low lexical overlap between the two "
            "passages -- 'notice' and 'probate' share nothing. Hard case.",
    ),
    Case(
        id="wp1_fin_imp_025", domain=FIN,
        query="Are there charges for using my card in the UK?",
        rule=Src("No charge is applied to sterling transactions made within the United Kingdom.",
                 SourceType.OFFICIAL_POLICY, "2024-03", "card_domestic"),
        other=Src("Quasi-cash transactions, including the purchase of foreign currency "
                  "and gambling stakes, attract a 3% handling charge and interest from "
                  "the transaction date.",
                  SourceType.PRODUCT_TERMS, "2024-05", "card_quasi_cash"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="no charge applies",
        condition="the transaction is quasi-cash (foreign currency purchase, gambling stakes)",
        exception_outcome="a 3% handling charge and interest from the transaction date apply",
        applicability="quasi-cash transactions",
        attrs=[cat("transaction_class", "quasi_cash")],
        scoped_answer="Sterling transactions in the UK are normally free of charge (per "
                      "p0), but quasi-cash transactions such as buying foreign currency "
                      "or gambling stakes carry a 3% handling charge and interest from "
                      "the transaction date (per p1).",
        selection_answer="No charge is applied to sterling transactions made within the "
                         "United Kingdom.",
        why="Device: TRANSACTION CLASS. The DEFAULT here is the favourable outcome and "
            "the exception is unfavourable -- the reverse polarity of most of the set. "
            "Include to check the probe is not keyed on 'exceptions grant benefits'.",
    ),
    Case(
        id="wp1_imm_imp_026", domain=IMM,
        query="Do I need a tuberculosis test certificate?",
        rule=Src("Applicants resident in a listed country must provide a tuberculosis "
                 "test certificate from an approved clinic.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-09", "gov_tb_testing"),
        other=Src("Those who have been living in the United Kingdom for the six months "
                  "immediately before applying are not asked for one.",
                  SourceType.FAQ, "2024-04", "gov_tb_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.DISJOINT,
        default_outcome="a tuberculosis test certificate is required",
        rule_condition="the applicant is resident in a listed country",
        rule_applicability="applicants resident in a listed country",
        rule_attrs=[cat("residence", "listed_country")],
        condition="the applicant has lived in the UK for the six months immediately before applying",
        exception_outcome="no certificate is required",
        applicability="applicants resident in the UK for the preceding six months",
        attrs=[cat("residence", "uk_six_months")],
        scoped_answer="A tuberculosis test certificate is required from applicants "
                      "resident in a listed country (per p0), but not from those who have "
                      "lived in the UK for the six months immediately before applying "
                      "(per p1). These describe different populations.",
        selection_answer="Applicants resident in a listed country must provide a "
                         "tuberculosis test certificate.",
        why="Device: RECENT RESIDENCE HISTORY. Relabelled from refinement to DISJOINT on "
            "review. The text never establishes that someone who has lived in the UK for "
            "the preceding six months is a subset of those resident in a listed country "
            "-- in practice the two populations barely meet -- so the nesting refinement "
            "requires is not supported by what the passages actually say. Both branches "
            "are separately scoped and neither is the default; the hard case here is "
            "resisting the pull to treat p0 as a general rule just because it is stated "
            "first.",
    ),
    Case(
        id="wp1_fin_imp_027", domain=FIN,
        query="How quickly will a payment arrive?",
        rule=Src("Payments are received by the beneficiary bank within two hours of "
                 "authorisation.",
                 SourceType.OFFICIAL_POLICY, "2024-05", "bank_payment_times"),
        other=Src("Transfers of 25,000 or more are subject to additional verification and "
                  "are released on the next working day.",
                  SourceType.FAQ, "2024-11", "bank_large_payments"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="payments arrive within two hours",
        condition="the transfer is 25,000 or more",
        exception_outcome="released on the next working day",
        applicability="transfers of 25,000 or more",
        attrs=[num("amount", 25000)],
        scoped_answer="Payments normally reach the beneficiary bank within two hours (per "
                      "p0); transfers of 25,000 or more undergo additional verification "
                      "and are released the next working day (per p1).",
        selection_answer="Payments are received by the beneficiary bank within two hours "
                         "of authorisation.",
        why="Device: AMOUNT THRESHOLD, open-ended above. Clean numeric case.",
    ),
    Case(
        id="wp1_imm_imp_028", domain=IMM,
        query="Can I bring my child with me?",
        rule=Src("A dependent child must be under 18 on the date of application.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-08", "gov_dependent_child"),
        other=Src("A child who was last granted leave as a dependant, and has not formed "
                  "an independent family unit, may continue to be included.",
                  SourceType.CATEGORY_PAGE, "2024-09", "gov_continuing_dependants"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        default_outcome="the child must be under 18",
        condition="the child was last granted leave as a dependant and has not formed an "
                  "independent family unit",
        exception_outcome="the child may continue to be included regardless of age",
        applicability="children continuing existing dependant leave",
        attrs=[flag("previously_dependant", True),
               flag("independent_family_unit", False)],
        scoped_answer="A dependent child must normally be under 18 at the date of "
                      "application (per p0); a child previously granted leave as a "
                      "dependant who has not formed an independent family unit may "
                      "continue to be included (per p1).",
        selection_answer="A dependent child must be under 18 on the date of application.",
        why="Device: CONTINUITY OF PRIOR STATUS, with a NEGATIVE sub-condition ('has not "
            "formed'). Negated conditions are a known weak spot for extraction; this is "
            "the case that probes it.",
    ),
]


# --------------------------------------------------------------------------- #
# B. Explicit-condition instances  (the control arm)
# --------------------------------------------------------------------------- #
#
# These DO carry an overt cue ("unless", "except", "only if", "does not apply
# to"). Explicit and implicit recovery are reported separately and never
# averaged, so the probe needs a control arm to show what the method achieves
# when the condition is handed to it.

EXPLICIT: list[Case] = [
    Case(
        id="wp1_fin_exp_029", domain=FIN,
        query="Is there a fee for a replacement card?",
        rule=Src("A replacement card is issued at a charge of 10.00.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "card_replacement"),
        other=Src("This charge does not apply where the card has been reported as lost "
                  "through fraud.",
                  SourceType.FAQ, "2024-08", "card_fraud_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="a 10.00 replacement charge applies",
        condition="the card was reported lost through fraud",
        exception_outcome="no charge applies",
        applicability="cards reported lost through fraud",
        attrs=[flag("fraud_reported", True)],
        scoped_answer="A replacement card costs 10.00 (per p0), except where the card was "
                      "reported lost through fraud (per p1).",
        selection_answer="A replacement card is issued at a charge of 10.00.",
        why="CONTROL. Explicit cue: 'does not apply where'.",
    ),
    Case(
        id="wp1_fin_exp_030", domain=FIN,
        query="Do I earn interest on my current account?",
        rule=Src("Credit interest is paid at 1.5% AER on balances up to 5,000.",
                 SourceType.OFFICIAL_POLICY, "2024-03", "bank_credit_interest"),
        other=Src("Interest is payable only if at least two direct debits are active in "
                  "the calendar month.",
                  SourceType.PRODUCT_TERMS, "2024-07", "bank_interest_conditions"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="interest is paid at 1.5% AER up to 5,000",
        condition="fewer than two direct debits are active in the month",
        exception_outcome="no interest is paid",
        applicability="months with fewer than two active direct debits",
        attrs=[num("active_direct_debits", hi=1)],
        scoped_answer="Credit interest is 1.5% AER on balances up to 5,000 (per p0), but "
                      "only in months where at least two direct debits are active (per p1).",
        selection_answer="Credit interest is paid at 1.5% AER on balances up to 5,000.",
        why="CONTROL. Explicit cue: 'only if'. Note the condition is stated positively "
            "but the EXCEPTION is its negation -- check the probe gets the polarity right.",
    ),
    Case(
        id="wp1_fin_exp_031", domain=FIN,
        query="Can I use my card for a cash advance?",
        rule=Src("Cash advances are available up to the cash limit shown on your statement.",
                 SourceType.OFFICIAL_POLICY, "2024-01", "card_cash_advance"),
        other=Src("Cash advances are not permitted unless the account has been open for "
                  "at least 60 days.",
                  SourceType.PRODUCT_TERMS, "2024-05", "card_new_accounts"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="cash advances are available up to the cash limit",
        condition="the account has been open for fewer than 60 days",
        exception_outcome="cash advances are not permitted",
        applicability="accounts open fewer than 60 days",
        attrs=[num("account_age_days", hi=60)],
        scoped_answer="Cash advances are available up to your cash limit (per p0), but not "
                      "until the account has been open for at least 60 days (per p1).",
        selection_answer="Cash advances are available up to the cash limit shown on your "
                         "statement.",
        why="CONTROL. Explicit cue: 'not permitted unless'.",
    ),
    Case(
        id="wp1_imm_exp_032", domain=IMM,
        query="Can I apply for settlement after five years?",
        rule=Src("Applicants may apply for settlement after five years of continuous "
                 "lawful residence.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-07", "gov_settlement_route"),
        other=Src("This route is not available unless the applicant has held leave in an "
                  "eligible category for the whole of that period.",
                  SourceType.CATEGORY_PAGE, "2024-03", "gov_eligible_categories"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="settlement may be applied for after five years",
        condition="leave was not held in an eligible category throughout the period",
        exception_outcome="the route is not available",
        applicability="applicants without eligible-category leave throughout",
        attrs=[flag("eligible_category_throughout", False)],
        scoped_answer="Settlement can be applied for after five years of continuous lawful "
                      "residence (per p0), but only where leave was held in an eligible "
                      "category throughout (per p1).",
        selection_answer="Applicants may apply for settlement after five years of "
                         "continuous lawful residence.",
        why="CONTROL. Explicit cue: 'not available unless'.",
    ),
    Case(
        id="wp1_imm_exp_033", domain=IMM,
        query="Do I need a sponsor?",
        rule=Src("A valid certificate of sponsorship is required for all applications "
                 "under this route.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-02", "gov_sponsorship"),
        other=Src("Except for applications for settlement, sponsorship must remain valid "
                  "at the date of decision.",
                  SourceType.CATEGORY_PAGE, "2024-08", "gov_sponsorship_validity"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="sponsorship must remain valid at the date of decision",
        condition="the application is for settlement",
        exception_outcome="sponsorship need not remain valid at the date of decision",
        applicability="settlement applications",
        attrs=[cat("application_type", "settlement")],
        scoped_answer="A valid certificate of sponsorship is required, and must remain "
                      "valid at the date of decision (per p0, p1) -- except for settlement "
                      "applications (per p1).",
        selection_answer="A valid certificate of sponsorship is required for all "
                         "applications under this route.",
        why="CONTROL. Explicit cue: 'Except for'. The cue is sentence-initial, which some "
            "cue-matching implementations handle differently from mid-sentence cues.",
    ),
    Case(
        id="wp1_fin_exp_034", domain=FIN,
        query="Will I be charged for going over my limit?",
        rule=Src("An unarranged overdraft fee of 8.00 per day is applied.",
                 SourceType.OFFICIAL_POLICY, "2024-04", "bank_unarranged"),
        other=Src("No fee is charged provided that the account is returned to credit "
                  "before the close of the following business day.",
                  SourceType.FAQ, "2024-09", "bank_grace_period"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="an 8.00 per-day fee applies",
        condition="the account returns to credit before close of the following business day",
        exception_outcome="no fee is charged",
        applicability="accounts restored to credit within one business day",
        attrs=[num("business_days_overdrawn", hi=1)],
        scoped_answer="An unarranged overdraft costs 8.00 per day (per p0), unless the "
                      "account is back in credit before the close of the following "
                      "business day (per p1).",
        selection_answer="An unarranged overdraft fee of 8.00 per day is applied.",
        why="CONTROL. Explicit cue: 'provided that'.",
    ),
    Case(
        id="wp1_imm_exp_035", domain=IMM,
        query="Is a criminal record check required?",
        rule=Src("A criminal record certificate is required from each country where the "
                 "applicant has lived for 12 months or more.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-10", "gov_criminal_record"),
        other=Src("This requirement does not apply to applicants under the age of 18.",
                  SourceType.CATEGORY_PAGE, "2024-05", "gov_minors"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="a criminal record certificate is required",
        condition="the applicant is under 18",
        exception_outcome="no certificate is required",
        applicability="applicants under 18",
        attrs=[num("age", hi=17)],
        scoped_answer="A criminal record certificate is required from each country of "
                      "12 months' residence or more (per p0), except for applicants under "
                      "18 (per p1).",
        selection_answer="A criminal record certificate is required from each country "
                         "where the applicant has lived for 12 months or more.",
        why="CONTROL. Explicit cue: 'does not apply to'.",
    ),
    Case(
        id="wp1_fin_exp_036", domain=FIN,
        query="Is travel insurance included with my account?",
        rule=Src("Worldwide travel insurance is included as an account benefit.",
                 SourceType.PRODUCT_TERMS, "2024-03", "account_benefits"),
        other=Src("Cover is not provided unless the account holder is under 70 at the "
                  "start of the trip.",
                  SourceType.OFFICIAL_POLICY, "2024-06", "insurance_terms"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REFINEMENT,
        explicitness=Explicitness.EXPLICIT,
        default_outcome="worldwide travel insurance is included",
        condition="the account holder is 70 or over at the start of the trip",
        exception_outcome="no cover is provided",
        applicability="account holders aged 70 or over",
        attrs=[num("age", 70)],
        scoped_answer="Worldwide travel insurance is included as an account benefit (per "
                      "p0), but cover does not apply if the account holder is 70 or over "
                      "at the start of the trip (per p1).",
        selection_answer="Worldwide travel insurance is included as an account benefit.",
        why="CONTROL. Explicit cue: 'not provided unless'. Age boundary stated as an "
            "upper bound on eligibility, so the EXCEPTION set is the complement.",
    ),
]


# --------------------------------------------------------------------------- #
# C. Distractors  (without these, SCR has no denominator)
# --------------------------------------------------------------------------- #

DISTRACTORS: list[Case] = [
    # -- genuine factual contradictions ------------------------------------- #
    Case(
        id="wp1_fin_fact_037", domain=FIN,
        query="What is the fee for an international payment?",
        rule=Src("International payments are subject to a 15.00 handling fee.",
                 SourceType.OFFICIAL_POLICY, "2024-01", "bank_payments"),
        other=Src("International payments are subject to a 25.00 handling fee.",
                  SourceType.THIRD_PARTY, "2024-02", "comparison_site"),
        conflict_type=ConflictType.FACTUAL, is_distractor=True,
        selection_answer="International payments are subject to a 15.00 handling fee.",
        why="DISTRACTOR (factual). Same population, same time, two different numbers. "
            "One is simply wrong. Must NOT be composed -- if the probe proposes a "
            "condition here, that is a spurious condition.",
    ),
    Case(
        id="wp1_imm_fact_038", domain=IMM,
        query="How long is leave granted for?",
        rule=Src("Leave under this route is granted for a maximum of five years.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-01", "gov_leave_duration"),
        other=Src("Leave under this route is granted for a maximum of three years.",
                  SourceType.THIRD_PARTY, "2024-03", "advice_blog"),
        conflict_type=ConflictType.FACTUAL, is_distractor=True,
        selection_answer="Leave under this route is granted for a maximum of five years.",
        why="DISTRACTOR (factual). Direct numeric contradiction, no scope difference.",
    ),
    Case(
        id="wp1_fin_fact_039", domain=FIN,
        query="What is the minimum opening deposit?",
        rule=Src("A minimum opening deposit of 100 is required.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "savings_opening"),
        other=Src("No minimum opening deposit is required for this account.",
                  SourceType.THIRD_PARTY, "2024-04", "comparison_site"),
        conflict_type=ConflictType.FACTUAL, is_distractor=True,
        selection_answer="A minimum opening deposit of 100 is required.",
        why="DISTRACTOR (factual). Tempting to read as conditional because one says 'this "
            "account' -- but no narrower scope is actually identified. Good adversarial "
            "case for the factual/conditional boundary.",
    ),
    Case(
        id="wp1_imm_fact_040", domain=IMM,
        query="What is the application fee?",
        rule=Src("The application fee is 1,500.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-03", "gov_fees"),
        other=Src("The application fee is 1,846.",
                  SourceType.THIRD_PARTY, "2024-05", "advice_site"),
        conflict_type=ConflictType.FACTUAL, is_distractor=True,
        selection_answer="The application fee is 1,500.",
        why="DISTRACTOR (factual).",
    ),

    # -- temporal ------------------------------------------------------------ #
    Case(
        id="wp1_fin_temp_041", domain=FIN,
        query="What is the current overdraft rate?",
        rule=Src("With effect from 1 January 2022, arranged overdrafts are charged at "
                 "35.0% EAR.",
                 SourceType.OFFICIAL_POLICY, "2022-01", "bank_overdraft_2022"),
        other=Src("With effect from 1 March 2025, arranged overdrafts are charged at "
                  "39.9% EAR.",
                  SourceType.AMENDMENT, "2025-03", "bank_overdraft_2025"),
        conflict_type=ConflictType.TEMPORAL, is_distractor=True,
        selection_answer="With effect from 1 March 2025, arranged overdrafts are charged "
                         "at 39.9% EAR.",
        why="DISTRACTOR (temporal). One supersedes the other. Contrast with imp_008, "
            "where BOTH remain in force and which applies depends on loan origination "
            "date -- that one is conditional, this one is not.",
    ),
    Case(
        id="wp1_imm_temp_042", domain=IMM,
        query="What is the salary threshold?",
        rule=Src("As of April 2023, the general salary threshold is 26,200.",
                 SourceType.GOVERNMENT_GUIDANCE, "2023-04", "gov_salary_2023"),
        other=Src("From 4 April 2024, the general salary threshold is 38,700.",
                  SourceType.AMENDMENT, "2024-04", "gov_salary_2024"),
        conflict_type=ConflictType.TEMPORAL, is_distractor=True,
        selection_answer="From 4 April 2024, the general salary threshold is 38,700.",
        why="DISTRACTOR (temporal).",
    ),
    Case(
        id="wp1_fin_temp_043", domain=FIN,
        query="Is there a fee for contactless payments?",
        rule=Src("Prior to 2023, contactless payments above 45 required chip and PIN "
                 "verification.",
                 SourceType.OFFICIAL_POLICY, "2022-06", "card_contactless_old"),
        other=Src("The contactless limit is currently 100 per transaction.",
                  SourceType.OFFICIAL_POLICY, "2024-03", "card_contactless"),
        conflict_type=ConflictType.TEMPORAL, is_distractor=True,
        selection_answer="The contactless limit is currently 100 per transaction.",
        why="DISTRACTOR (temporal).",
    ),

    # -- opinion ------------------------------------------------------------- #
    Case(
        id="wp1_imm_op_044", domain=IMM,
        query="Is this route a good option for graduates?",
        rule=Src("Most advisers regard the graduate route as the most straightforward "
                 "option for recent students.",
                 SourceType.THIRD_PARTY, "2024-02", "advice_blog_a"),
        other=Src("Many practitioners consider the graduate route unnecessarily "
                  "restrictive given its two-year cap.",
                  SourceType.THIRD_PARTY, "2024-07", "advice_blog_b"),
        conflict_type=ConflictType.OPINION, is_distractor=True,
        selection_answer="Most advisers regard the graduate route as the most "
                         "straightforward option for recent students.",
        why="DISTRACTOR (opinion). Differing judgements, not differing facts. Neither is "
            "checkable.",
    ),
    Case(
        id="wp1_fin_op_045", domain=FIN,
        query="Is this account good value?",
        rule=Src("Reviewers generally rate this account well for its fee-free overseas "
                 "spending.",
                 SourceType.THIRD_PARTY, "2024-03", "review_site_a"),
        other=Src("Some commentators argue the monthly fee outweighs the travel benefits "
                  "for infrequent travellers.",
                  SourceType.THIRD_PARTY, "2024-08", "review_site_b"),
        conflict_type=ConflictType.OPINION, is_distractor=True,
        selection_answer="Reviewers generally rate this account well for its fee-free "
                         "overseas spending.",
        why="DISTRACTOR (opinion). Note 'for infrequent travellers' LOOKS like a scope "
            "restriction -- but it qualifies an opinion, not a rule. Adversarial case for "
            "a probe that pattern-matches on population phrases.",
    ),

    # -- no conflict --------------------------------------------------------- #
    Case(
        id="wp1_fin_none_046", domain=FIN,
        query="What is the fee on foreign transactions?",
        rule=Src("Foreign currency transactions carry a 2.99% conversion charge.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "bank_fx"),
        other=Src("Statements are issued on the first working day of each month.",
                  SourceType.PRODUCT_TERMS, "2024-02", "bank_statements"),
        conflict_type=ConflictType.NO_CONFLICT, is_distractor=True,
        selection_answer="Foreign currency transactions carry a 2.99% conversion charge.",
        why="DISTRACTOR (no conflict). Different subjects entirely.",
    ),
    Case(
        id="wp1_imm_none_047", domain=IMM,
        query="What documents do I need?",
        rule=Src("A valid passport and a completed application form are required.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-01", "gov_documents"),
        other=Src("Biometric appointments are booked through the commercial partner's "
                  "online portal.",
                  SourceType.FAQ, "2024-06", "gov_biometrics"),
        conflict_type=ConflictType.NO_CONFLICT, is_distractor=True,
        selection_answer="A valid passport and a completed application form are required.",
        why="DISTRACTOR (no conflict). Sequential steps in one process, not competing "
            "claims.",
    ),
    Case(
        id="wp1_fin_none_048", domain=FIN,
        query="How do I report a lost card?",
        rule=Src("Lost or stolen cards should be reported immediately through the mobile "
                 "app or by telephone.",
                 SourceType.OFFICIAL_POLICY, "2024-04", "card_lost"),
        other=Src("Replacement cards are dispatched within five working days of a report.",
                  SourceType.FAQ, "2024-04", "card_replacement_faq"),
        conflict_type=ConflictType.NO_CONFLICT, is_distractor=True,
        selection_answer="Lost or stolen cards should be reported immediately through the "
                         "mobile app or by telephone.",
        why="DISTRACTOR (no conflict). Related topic, complementary information, no "
            "disagreement. This is the 'complementary information' category from DRAGged "
            "into Conflicts -- symmetric and co-equal, NOT an asymmetric default/exception.",
    ),

    # -- redundant ----------------------------------------------------------- #
    Case(
        id="wp1_fin_red_049", domain=FIN,
        query="Do premium cardholders pay the foreign transaction fee?",
        rule=Src("Foreign currency transactions carry a 2.99% conversion charge.",
                 SourceType.OFFICIAL_POLICY, "2024-02", "bank_fx"),
        other=Src("Premium cardholders are charged 2.99% on foreign currency "
                  "transactions, in line with the standard tariff.",
                  SourceType.FAQ, "2024-10", "bank_premium_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REDUNDANT,
        is_distractor=True,
        merged_outcome="a 2.99% conversion charge applies",
        default_outcome="a 2.99% conversion charge applies",
        scoped_answer="Foreign currency transactions carry a 2.99% conversion charge, "
                      "including for premium cardholders.",
        selection_answer="Foreign currency transactions carry a 2.99% conversion charge.",
        why="DISTRACTOR (redundant). The scopes NEST -- premium cardholders are a subset "
            "-- but the outcomes AGREE. This is a restatement, not an exception. A probe "
            "that proposes a condition here has fabricated a branch, which inflates "
            "Preservation Rate while corrupting the answer. This is the single most "
            "important distractor in the set.",
    ),
    Case(
        id="wp1_imm_red_050", domain=IMM,
        query="Do students need to show maintenance funds?",
        rule=Src("Applicants must show funds of 1,270 held for 28 consecutive days.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-01", "gov_maintenance"),
        other=Src("Student applicants are required to evidence 1,270 held for 28 days, "
                  "as for other routes.",
                  SourceType.FAQ, "2024-09", "gov_student_faq"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.REDUNDANT,
        is_distractor=True,
        merged_outcome="funds of 1,270 must be held for 28 consecutive days",
        default_outcome="funds of 1,270 must be held for 28 consecutive days",
        scoped_answer="Applicants, including students, must show funds of 1,270 held for "
                      "28 consecutive days.",
        selection_answer="Applicants must show funds of 1,270 held for 28 consecutive days.",
        why="DISTRACTOR (redundant). Nested scope, identical outcome. The phrase 'as for "
            "other routes' is the giveaway that it restates rather than narrows.",
    ),
]


# --------------------------------------------------------------------------- #
# D. The other two scope relations
# --------------------------------------------------------------------------- #
#
# Added after `contract.validate` pointed out the set contained no `disjoint`
# or `opposed` cases. A fixture missing a relation cannot catch confusions
# involving it, and both of these route differently from `refinement`:
# disjoint composes trivially, opposed falls back to selection. Member B's
# composition operator needs to be exercised on all three.

EDGE_RELATIONS: list[Case] = [
    Case(
        id="wp1_fin_dis_051", domain=FIN,
        query="What interest rate will I earn?",
        rule=Src("Instant access savings accounts earn 4.50% AER variable.",
                 SourceType.OFFICIAL_POLICY, "2024-03", "savings_instant"),
        other=Src("Fixed-term bonds held to maturity earn 5.20% AER.",
                  SourceType.PRODUCT_TERMS, "2024-03", "savings_bonds"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.DISJOINT,
        default_outcome="4.50% AER variable",
        rule_condition="the product is an instant access savings account",
        rule_applicability="instant access savings accounts",
        rule_attrs=[cat("product_type", "instant_access")],
        condition="the product is a fixed-term bond held to maturity",
        exception_outcome="5.20% AER",
        applicability="fixed-term bonds",
        attrs=[cat("product_type", "fixed_term_bond")],
        scoped_answer="Instant access savings earn 4.50% AER variable (per p0); "
                      "fixed-term bonds held to maturity earn 5.20% AER (per p1). These "
                      "are different products.",
        selection_answer="Instant access savings accounts earn 4.50% AER variable.",
        why="DISJOINT. No account is both instant-access and a fixed-term bond, so no "
            "case falls under both and outcome comparison is vacuous. Must NOT be read "
            "as a contradiction just because the two rates differ -- this is the "
            "disjoint/opposed boundary.",
    ),
    Case(
        id="wp1_imm_dis_052", domain=IMM,
        query="How long can I stay?",
        rule=Src("A standard visitor visa permits a stay of up to six months.",
                 SourceType.GOVERNMENT_GUIDANCE, "2024-01", "gov_visitor"),
        other=Src("A student visa is granted for the duration of the course plus a "
                  "wrap-up period.",
                  SourceType.GOVERNMENT_GUIDANCE, "2024-01", "gov_student_duration"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.DISJOINT,
        default_outcome="a stay of up to six months",
        rule_condition="the applicant holds a standard visitor visa",
        rule_applicability="standard visitor visa holders",
        rule_attrs=[cat("visa_type", "visitor")],
        condition="the applicant holds a student visa",
        exception_outcome="the duration of the course plus a wrap-up period",
        applicability="student visa holders",
        attrs=[cat("visa_type", "student")],
        scoped_answer="A standard visitor visa allows up to six months (per p0); a "
                      "student visa is granted for the course duration plus a wrap-up "
                      "period (per p1). These are different routes.",
        selection_answer="A standard visitor visa permits a stay of up to six months.",
        why="DISJOINT. Different visa categories; nobody holds both at once.",
    ),
    Case(
        id="wp1_fin_opp_053", domain=FIN,
        query="Can I use the airport lounge?",
        rule=Src("Reward account holders are entitled to complimentary airport lounge "
                 "access.",
                 SourceType.PRODUCT_TERMS, "2023-05", "account_lounge"),
        other=Src("Accounts opened from January 2024 onwards do not include lounge "
                  "access as a benefit.",
                  SourceType.THIRD_PARTY, "2024-06", "review_site"),
        conflict_type=ConflictType.CONDITIONAL, relation=ScopeRelation.OPPOSED,
        default_outcome="lounge access is included",
        rule_condition="the account is a Reward account",
        rule_applicability="Reward account holders",
        rule_attrs=[cat("account_type", "reward")],
        condition="the account was opened from January 2024 onwards",
        exception_outcome="lounge access is not included",
        applicability="accounts opened from January 2024",
        attrs=[cat("opened_period", "2024-01-onwards")],
        scoped_answer="The sources genuinely disagree for Reward accounts opened from "
                      "January 2024. The more authoritative source (p0, product terms) "
                      "is reported.",
        selection_answer="Reward account holders are entitled to complimentary airport "
                         "lounge access.",
        why="OPPOSED. The sets OVERLAP (a Reward account opened in 2024 is in both) but "
            "NEITHER CONTAINS the other -- a 2023 Reward account is only in the first, a "
            "2024 non-Reward account only in the second -- and the outcomes disagree on "
            "the overlap. Routes to selection, not composition. The hardest relation to "
            "tell from refinement.",
    ),
    # wp1_imm_opp_054 was DROPPED on review (2026-09-25).
    #
    # It read as opposed but the two passages never contradicted each other:
    # p0 concerned satisfaction of "the skill requirement" and p1 satisfaction
    # of "the requirements of this route", which are different propositions. A
    # 46-year-old graduate meets the first and fails the second, both true at
    # once -- so it was closer to a no-conflict distractor than an opposed pair.
    #
    # Dropped rather than reworded: changing p0 to say "the requirements of this
    # route" would have made it a valid case, but a different one from the case
    # that was originally reviewed, and quietly editing evidence to fit a label
    # is the wrong habit to build into a benchmark.
    #
    # Consequence: OPPOSED now has a single instance (wp1_fin_opp_053). Thin
    # coverage of the relation that is hardest to tell from refinement, and a
    # known limitation of this probe set until a replacement is authored.
]


ALL_CASES: list[Case] = IMPLICIT + EXPLICIT + DISTRACTORS + EDGE_RELATIONS


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def check_implicitness() -> list[str]:
    """Verify the implicit/explicit split is real, not merely declared.

    The whole value of this set is that the implicit cases are genuinely
    implicit. If an exception passage contains a strong exception cue, the case
    is explicit and mislabelled, and Member B's implicit-recovery number would
    be measuring the wrong thing -- flattered by cases the method never had to
    infer anything about.

    Checked against STRONG cues only. Weak condition markers ("where the",
    "subject to") appear throughout ordinary rule prose and do not make a
    passage an explicit exception.
    """
    from detection.features import STRONG_EXCEPTION_CUES

    problems: list[str] = []
    for c in IMPLICIT:
        hits = [q for q in STRONG_EXCEPTION_CUES if q in c.other.text.lower()]
        if hits:
            problems.append(
                f"{c.id}: labelled IMPLICIT but its exception passage contains "
                f"{hits} -- reword it or move it to EXPLICIT"
            )
    for c in EXPLICIT:
        if not any(q in c.other.text.lower() for q in STRONG_EXCEPTION_CUES):
            problems.append(
                f"{c.id}: labelled EXPLICIT but its exception passage has no strong "
                f"cue -- it belongs in IMPLICIT"
            )
    return problems


def build_all() -> list[GoldInstance]:
    problems = check_implicitness()
    if problems:
        raise ValueError(
            "the implicit/explicit split is broken:\n  " + "\n  ".join(problems)
        )
    return [c.build() for c in ALL_CASES]


def stats(cases: list[Case]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        "conflict_type": {}, "scope_relation": {}, "explicitness": {}, "domain": {},
    }
    for c in cases:
        out["conflict_type"][c.conflict_type.value] = \
            out["conflict_type"].get(c.conflict_type.value, 0) + 1
        key = c.relation.value if c.relation else "none"
        out["scope_relation"][key] = out["scope_relation"].get(key, 0) + 1
        out["domain"][c.domain.value] = out["domain"].get(c.domain.value, 0) + 1
        if c.conflict_type is ConflictType.CONDITIONAL and not c.is_distractor:
            out["explicitness"][c.explicitness.value] = \
                out["explicitness"].get(c.explicitness.value, 0) + 1
    return out


#: Cases whose label is a genuine judgement call rather than a clear read.
#: Surfaced first because verification attention is finite and these are where
#: a wrong label does the most damage -- each one sits on a boundary the
#: proposal itself names as hard.
HIGH_RISK: dict[str, str] = {
    "wp1_fin_imp_008": "CONDITIONAL vs TEMPORAL. Both texts remain in force; which "
                       "applies depends on loan origination date, not on which is newer. "
                       "If you disagree, it belongs with the temporal distractors.",
    "wp1_fin_imp_019": "CONDITIONAL vs FACTUAL. 'Non-refundable' against 'returned in "
                       "full' reads as a flat contradiction on the surface.",
    "wp1_fin_imp_020": "CONDITIONAL vs FACTUAL. Two different protection limits (85,000 "
                       "vs 1 million) could be read as a numeric contradiction.",
    "wp1_fin_red_049": "REDUNDANT vs REFINEMENT. Nested scope, agreeing outcome. The "
                       "single most consequential case in the set: proposing a condition "
                       "here fabricates a branch.",
    "wp1_imm_red_050": "REDUNDANT vs REFINEMENT. Same shape as 049.",
    "wp1_fin_opp_053": "OPPOSED vs REFINEMENT. Check neither scope contains the other.",
    "wp1_fin_imp_009": "IMPLICIT vs EXPLICIT. Reworded once already to remove 'exempt "
                       "from'. Confirm the current wording carries no cue for you.",
    "wp1_fin_imp_005": "IMPLICIT vs EXPLICIT. 'Where the statement balance is settled' "
                       "is a conditional marker, though not an exception marker.",
    "wp1_imm_imp_021": "IMPLICIT vs EXPLICIT. Same 'Where the...' construction as 005; "
                       "decide once and apply to both.",
    "wp1_fin_imp_025": "POLARITY. The default is the favourable outcome and the exception "
                       "is unfavourable -- the reverse of most of the set.",
    "wp1_imm_imp_026": "NESTED DEFAULT. The default is itself already conditioned "
                       "('resident in a listed country'). Check the listed-country clause "
                       "is not being read as the exception.",
    "wp1_imm_imp_016": "SUB-STRUCTURE. The exception branch contains its own carve-out "
                       "(four years for Swiss nationals). First-order for the pair, but "
                       "confirm the sub-case is not silently dropped.",
    "wp1_fin_imp_007": "PARTIAL SCOPE. The exception applies to a slice of the balance "
                       "(first 500), not the whole account.",
}


def review_sheet(cases: list[Case]) -> str:
    """Human-readable sheet for Member A to verify against.

    High-risk cases first, then the rest by group. Verification attention is
    finite: the boundary cases are where a wrong label propagates furthest,
    and a reviewer who runs out of patience should run out of it on the easy
    ones.
    """
    lines = [
        "WP1 PROBE SET - VERIFICATION SHEET",
        "=" * 100,
        "",
        "Human-verified 2026-09-25. Re-run this sheet after any edit.",
        "",
        "For each case check, in this order:",
        "  1. Are BOTH passages true at once, for different cases?  -> conditional",
        "     If not, it is factual (or temporal / opinion / no-conflict).",
        "  2. Do the scopes nest, and do the outcomes DIFFER on the overlap?",
        "     nested + differ = refinement.  nested + agree = REDUNDANT.",
        "  3. Is the condition genuinely implicit -- no 'except', 'unless',",
        "     'only if', 'does not apply to' in the exception passage?",
        "  4. Do the typed attributes match what the text actually says, with",
        "     nothing invented?",
        "",
    ]
    risky = [c for c in cases if c.id in HIGH_RISK]
    if risky:
        lines += ["", "!! VERIFY THESE FIRST -- genuine judgement calls", "=" * 100, ""]
        for c in risky:
            label = c.conflict_type.value + (f"/{c.relation.value}" if c.relation else "")
            lines += [
                f"[{c.id}]  proposed: {label}",
                f"  Q:    {c.query}",
                f"  p0:   {c.rule.text}",
                f"  p1:   {c.other.text}",
                f"  RISK: {HIGH_RISK[c.id]}",
                "  DECISION: [ ] accept as proposed   [ ] relabel to ______________",
                "",
            ]
        lines += ["", f"({len(risky)} judgement calls above; "
                      f"{len(cases) - len(risky)} clearer cases below)", ""]

    for group, title in ((IMPLICIT, "A. IMPLICIT CONDITIONS (the probe's core)"),
                         (EXPLICIT, "B. EXPLICIT CONDITIONS (control arm)"),
                         (DISTRACTORS, "C. DISTRACTORS (SCR denominator)"),
                         (EDGE_RELATIONS, "D. DISJOINT AND OPPOSED RELATIONS")):
        lines += ["", title, "=" * 100]
        for c in group:
            label = c.conflict_type.value
            if c.relation:
                label += f"/{c.relation.value}"
            lines += [
                "",
                f"[{c.id}]  {label}   ({c.explicitness.value if c.conflict_type is ConflictType.CONDITIONAL else '-'})",
                f"  Q:    {c.query}",
                f"  p0:   {c.rule.text}",
                f"  p1:   {c.other.text}",
            ]
            if c.condition:
                lines += [
                    f"  cond: {c.condition}",
                    f"  then: {c.exception_outcome}",
                    f"  attr: {', '.join(f'{a.name}:{a.kind.value}' for a in c.attrs) or '(none)'}",
                ]
            lines += [f"  WHY:  {c.why}", "  VERIFY: [ ] type  [ ] relation  [ ] implicit  [ ] attributes"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write validated JSONL here")
    ap.add_argument("--review", action="store_true",
                    help="print the human verification sheet")
    ap.add_argument("--stats", action="store_true", help="print composition only")
    args = ap.parse_args(argv)

    instances = build_all()  # raises if any case violates the schema

    if args.review:
        print(review_sheet(ALL_CASES))
        return 0

    s = stats(ALL_CASES)
    print(f"WP1 probe set: {len(instances)} instances (human-verified 2026-09-25)\n")
    for section, counts in s.items():
        body = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {section:<16} {body}")
    print(f"\n  implicit conditional cases: {len(IMPLICIT)}  <- what the probe is for")
    print(f"  explicit control cases:     {len(EXPLICIT)}")
    print(f"  distractors:                {len(DISTRACTORS)}")

    if args.stats:
        return 0

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as fh:
            for inst in instances:
                fh.write(json.dumps(inst.model_dump(mode="json"), sort_keys=True,
                                    ensure_ascii=False) + "\n")
        print(f"\nwrote {len(instances)} instances -> {args.output}")
        print("\nNEXT: `python -m benchmark.wp1_probe_set --review` and check every case.")
        print("NOTE: OPPOSED has one instance only -- thin coverage of the relation")
        print("that is hardest to tell from refinement. A known limitation.")
    else:
        print("\n(no -o given; nothing written)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
