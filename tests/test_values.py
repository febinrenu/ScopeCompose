"""Outcome matching and modality detection (plan v4 acceptance M-1: at least 60
value cases and 30 modality cases).

Expectations are written from what the right answer IS, not from what the code
currently prints. A failure here is either a real bug or a deliberate limit that
must be documented, and the "documented limits" block pins the latter so a
future change to them is noticed instead of silent.
"""

from __future__ import annotations

import pytest

from contract.frames import Modality
from metrics.values import (
    canonicalise_outcome,
    detect_modality,
    determines_outcome,
    is_hedged,
    strip_citations,
    values_match,
)

# --------------------------------------------------------------------------- #
# Value matching -- same meaning
# --------------------------------------------------------------------------- #

SAME = [
    ("no fee applies", "no fee applies"),
    ("no fee applies", "The fee is waived."),
    ("no fee applies", "You pay no fee."),
    ("no fee applies", "The charge is waived for premium cardholders (per p1)"),
    ("no fee applies", "fee-free"),
    ("no fee applies", "free of charge"),
    ("no fee applies", "you will not be charged"),
    ("no fee applies", "exempt from the fee"),
    ("no fee applies", "You do not pay a fee."),
    ("the fee is waived", "no charge applies"),
    ("a 3% fee applies", "a fee of 3% applies"),
    ("a 3% fee applies", "a three percent fee applies"),
    ("a 3% fee applies", "a 3 per cent fee applies (per p0)"),
    ("a 3% fee applies", "A 3% fee applies [p0]"),
    ("funds of 1270 must be held for 28 days", "funds of 1,270 must be held for twenty-eight days"),
    ("funds of 1270 must be held for 28 days", "you need 1270 held for 28 days"),
    ("the applicant may apply", "you are allowed to apply"),
    ("the applicant may apply", "applicants can apply"),
    ("the applicant is not eligible", "you do not qualify"),
    ("the applicant is eligible", "you are entitled to it"),
    ("lounge access is included", "lounge access is included in your account"),
    ("employment is prohibited", "employment is not permitted"),
    ("employment is prohibited", "you may not take employment"),
    ("evidence is required", "you must provide evidence"),
    ("evidence is not required", "you need not provide evidence"),
    ("a 2.50 charge applies", "a charge of 2.50 applies"),
    ("a 2.5 charge applies", "a 2.50 charge applies"),
]


@pytest.mark.parametrize("gold,pred", SAME)
def test_same_meaning_matches(gold, pred):
    assert values_match(gold, pred), (canonicalise_outcome(gold), canonicalise_outcome(pred))


@pytest.mark.parametrize("n,word", [(2, "two"), (3, "three"), (4, "four"), (5, "five"), (6, "six"),
                                    (7, "seven"), (10, "ten"), (12, "twelve"), (14, "fourteen"),
                                    (20, "twenty"), (21, "twenty-one"), (28, "twenty-eight"),
                                    (30, "thirty"), (45, "forty-five")])
def test_spelled_out_numbers_match_digits(n, word):
    assert values_match(f"the period is {n} years", f"the period is {word} years")


# --------------------------------------------------------------------------- #
# Value matching -- different meaning. These are the ones that inflate CCA if
# they slip through, so they carry the most weight.
# --------------------------------------------------------------------------- #

DIFFERENT = [
    ("a 3% fee applies", "a 5% fee applies"),
    ("a 3% fee applies", "no fee applies"),
    ("no fee applies", "a fee applies"),
    ("no fee applies", "the fee applies"),
    ("no fee applies", "a 3% fee applies (per p0)"),
    ("the period is five years", "the period is four years"),
    ("funds of 1270 for 28 days", "funds of 1270 for 14 days"),
    ("funds of 1270 for 28 days", "funds of 1000 for 28 days"),
    ("employment is permitted", "employment is prohibited"),
    ("the applicant may apply", "the applicant may not apply"),
    ("the applicant may not apply", "you may apply"),
    ("lounge access is included", "lounge access is not included"),
    ("the applicant is eligible", "the applicant is not eligible"),
    ("the applicant is eligible", "you do not qualify"),
    ("evidence is required", "you need not provide evidence"),
    ("evidence is not required", "you must provide evidence"),
    ("must provide evidence", "evidence is optional"),
    ("a 3% fee applies", "lounge access is included"),
    ("the fee is waived", "a 3% fee applies"),
    ("a 2.50 charge applies", "a 5.00 charge applies"),
    ("", "no fee applies"),
    ("no fee applies", ""),
    ("   ", "no fee applies"),
]


@pytest.mark.parametrize("gold,pred", DIFFERENT)
def test_different_meaning_does_not_match(gold, pred):
    assert not values_match(gold, pred), (canonicalise_outcome(gold), canonicalise_outcome(pred))


@pytest.mark.parametrize("a", range(2, 12))
def test_adjacent_numbers_never_match(a):
    """The 'five years' vs 'four years' bug class, generated: neighbouring
    figures share every word but one, and that word is the whole meaning."""
    assert not values_match(f"the period is {a} years", f"the period is {a + 1} years")
    assert values_match(f"the period is {a} years", f"the period is {a} years")


def test_values_match_is_deterministic_and_symmetric_on_the_veto_cases():
    for g, p in DIFFERENT[:20]:
        assert values_match(g, p) == values_match(g, p)
        assert values_match(p, g) is False


# --------------------------------------------------------------------------- #
# Documented lexical limits. These are real and stated: deterministic matching
# is a lower bound on agreement. Pinned so any change is noticed.
# --------------------------------------------------------------------------- #


def test_acronyms_and_wide_paraphrase_are_a_stated_limit():
    assert not values_match("the applicant may apply for indefinite leave to remain",
                            "you may apply for ILR")
    assert not values_match("on-campus employment permitted", "you are allowed to work at the university")


def test_canonicalisation_strips_citations_and_subjects():
    assert strip_citations("a fee applies (per p0)") == "a fee applies"
    assert strip_citations("a fee applies [p1]") == "a fee applies"
    assert strip_citations("a fee applies (p0, p2)") == "a fee applies"
    assert "applicant" not in canonicalise_outcome("the applicant may apply")
    assert "you" not in canonicalise_outcome("you may apply").split()


def test_numbers_inside_words_are_not_rewritten():
    # "one" inside "money" must not become "1"
    assert "money" in canonicalise_outcome("the money is returned")
    assert canonicalise_outcome("a one off fee") == "a 1 off fee"


# --------------------------------------------------------------------------- #
# Modality
# --------------------------------------------------------------------------- #

HEDGED = [
    "Premium customers may receive fee waivers",          # the reviewer's example
    "A waiver may be granted",
    "You might be charged a fee",
    "The fee could apply",
    "In some cases the fee is waived",
    "In certain circumstances no fee is charged",
    "Fees are waived at our discretion",
    "Subject to approval, the fee is waived",
    "It depends on your account type",
    "Customers may qualify for a refund",
    "A charge may be applied",
    "You may be able to apply",
    "The fee is sometimes waived",
    "This may be free",
    "Applicants may not be eligible",
    "Premium customers pay no fee, although in some cases a fee may still apply",
    "Customers will probably pay a fee",
    "You may incur a fee",
    "A fee may apply",
    "Late fees may apply",
    "The fee can be waived for premium holders",
    "It is not guaranteed that the fee is waived",
    "The fee is not always charged",
    "",
    "   ",
]

NOT_HEDGED = [
    ("Applicants may apply for leave", Modality.PERMISSION),
    ("Students may work up to 20 hours a week", Modality.PERMISSION),
    ("Applicants may not work on campus", Modality.PROHIBITION),
    ("Premium-tier cardholders pay no fee", Modality.STATEMENT),
    ("Transfers normally incur a 3% fee", Modality.STATEMENT),
    ("The fee is waived for premium cardholders (per p1)", Modality.STATEMENT),
    ("You must provide evidence of funds", Modality.OBLIGATION),
    ("You cannot apply from outside the UK", Modality.PROHIBITION),
    ("Applicants are required to show funds", Modality.OBLIGATION),
    ("Holders can apply after five years", Modality.PERMISSION),
    ("A 3% fee applies", Modality.STATEMENT),
    ("Account holders are entitled to lounge access", Modality.PERMISSION),
    ("Fees are usually 3%", Modality.STATEMENT),
    ("The fee is generally waived for premium holders, as stated in the product terms", Modality.STATEMENT),
    ("You will be charged 3%", Modality.STATEMENT),
    ("Students who hold a fee-paying place may apply", Modality.PERMISSION),
    ("Overseas visitors shall not take employment", Modality.OBLIGATION),
]


@pytest.mark.parametrize("span", HEDGED)
def test_hedged_spans_never_determine_an_outcome(span):
    assert detect_modality(span) is Modality.HEDGED, span
    assert is_hedged(span)
    assert not determines_outcome(span)


@pytest.mark.parametrize("span,expected", NOT_HEDGED)
def test_committing_spans_determine_an_outcome(span, expected):
    assert detect_modality(span) is expected, span
    assert determines_outcome(span)


def test_there_are_at_least_30_modality_cases():
    assert len(HEDGED) + len(NOT_HEDGED) >= 30


def test_the_reviewers_gap_filling_example_is_not_determined():
    """'Premium customers may receive fee waivers' + persona 'I have a premium
    account' must not count as a determination (proposal v8 section 7.2)."""
    assert not determines_outcome("Premium customers may receive fee waivers")


def test_one_hedge_anywhere_in_a_span_wins():
    assert is_hedged("Premium cardholders pay nothing. In some cases a fee may still apply.")


def test_hedged_modality_matches_the_frame_contract():
    assert not Modality.HEDGED.determines_outcome
    assert Modality.PERMISSION.determines_outcome


# --------------------------------------------------------------------------- #
# Payment due (found on real gov.uk outcomes: the health surcharge)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gold,predicted", [
    ("you do not need to pay the surcharge", "the surcharge is not payable"),
    ("the surcharge is payable", "you must pay the surcharge"),
    ("you need to pay the surcharge", "the surcharge is payable"),
    ("the applicant will not pay the surcharge", "no payment is due for the surcharge")])
def test_payment_due_paraphrases_match(gold, predicted):
    assert values_match(gold, predicted)


@pytest.mark.parametrize("gold,predicted", [
    ("you must pay the surcharge", "you do not need to pay the surcharge"),
    ("the surcharge is not payable", "the surcharge is payable"),
    ("you will pay 5 pounds", "you will pay 7 pounds"),
    ("you must pay the surcharge", "the surcharge is waived for you")])
def test_payment_due_opposites_and_different_figures_do_not_match(gold, predicted):
    assert not values_match(gold, predicted)
