"""The claims ledger: numbers come from results or not at all."""

from __future__ import annotations

import math

import pytest

from reporting.claims import (
    Claim,
    ClaimError,
    Registry,
    ledger,
    placeholders,
    raw_numbers,
    render,
    resolve,
    unmeasured,
    value_of,
)

RESULTS = {"h1": {"p_exact": 0.00001, "diff": -0.183, "n": 47},
           "detectors": [{"f1": 0.712}, {"f1": 0.05}],
           "s1": {"n_test": 47, "narrowing": True}}


def reg():
    return Registry([
        Claim("h1-p", "h1.p_exact", "H1 exact McNemar p", "p"),
        Claim("h1-diff", "h1.diff", "H1 accuracy difference, B3 minus B0", "pts"),
        Claim("s1-n", "s1.n_test", "S1 test instances", "int"),
        Claim("d5-f1", "detectors.0.f1", "D5 F1", "f3"),
        Claim("not-yet", "h3.p_exact", "H3 p (not run yet)", "p"),
    ])


def test_resolve_follows_dicts_and_list_indices():
    assert resolve(RESULTS, "h1.n") == 47 and resolve(RESULTS, "detectors.1.f1") == 0.05
    for bad in ("nope", "h1.nope", "detectors.9.f1", "detectors.x.f1", "h1.n.deeper"):
        with pytest.raises(KeyError):
            resolve(RESULTS, bad)


def test_values_are_formatted_from_the_results_not_typed():
    r = reg()
    assert value_of(r["h1-p"], RESULTS) == "<0.0001"
    assert value_of(r["h1-diff"], RESULTS) == "-18.3"
    assert value_of(r["s1-n"], RESULTS) == "47" and value_of(r["d5-f1"], RESULTS) == "0.712"
    assert value_of(r["d5-f1"], RESULTS, "pct") == "71.2%", "a placeholder may override the format"


def test_render_substitutes_every_placeholder_and_changes_when_the_results_change():
    text = "D5 reached F1 {{d5-f1}} on {{s1-n}} instances (p {{h1-p}}); {{h1-diff|f2}} points."
    out = render(text, reg(), RESULTS)
    assert out == "D5 reached F1 0.712 on 47 instances (p <0.0001); -0.18 points."
    changed = render(text, reg(), RESULTS | {"s1": {"n_test": 52}})
    assert "52 instances" in changed, "a re-run updates the text; a stale number cannot survive"


def test_a_placeholder_for_an_unregistered_claim_is_an_error():
    with pytest.raises(ClaimError, match="names no registered claim"):
        render("The effect was {{made-up}}.", reg(), RESULTS)


def test_a_claim_whose_result_does_not_exist_yet_is_an_error_not_a_blank_or_a_guess():
    with pytest.raises(ClaimError, match="not yet measured"):
        render("H3 gave p = {{not-yet}}.", reg(), RESULTS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "high", None, True])
def test_a_value_that_is_not_a_reportable_number_is_refused(bad):
    r = Registry([Claim("x", "v", "x", "f3")])
    with pytest.raises(ClaimError):
        value_of(r["x"], {"v": bad})


def test_raw_format_is_allowed_for_text_values():
    r = Registry([Claim("lbl", "s1.status", "status", "raw")])
    assert value_of(r["lbl"], {"s1": {"status": "floor met"}}) == "floor met"


def test_registering_a_duplicate_or_an_invalid_claim_is_refused():
    r = reg()
    with pytest.raises(ValueError, match="already registered"):
        r.add(Claim("s1-n", "s1.n_test", "dup", "int"))
    with pytest.raises(ValueError):
        Claim("bad id!", "x", "x")
    with pytest.raises(ValueError, match="unknown format"):
        Claim("ok", "x", "x", "percent")


def test_placeholders_lists_each_id_once_in_order():
    assert placeholders("{{a}} then {{b|pct}} then {{a}}") == ["a", "b"]


def test_numbers_typed_into_prose_are_found_but_placeholders_years_and_allowed_ones_are_not():
    text = "We found {{s1-n}} instances, 45 in total, in 2026 (p = 0.03, 12.5%), see section 3."
    assert raw_numbers(text, allow=(r"3",)) == ["45", "0.03", "12.5%"]
    assert raw_numbers("Only {{d5-f1}} and {{h1-p}} here.") == []


def test_the_ledger_lists_every_claim_with_its_source_and_marks_the_unmeasured():
    t = ledger(reg(), RESULTS)
    assert "| `s1-n` | S1 test instances | `s1.n_test` | 47 |" in t
    assert "not yet measured" in t and t.count("\n") == len(reg()) + 1


def test_unmeasured_names_what_stands_between_the_draft_and_the_paper():
    assert unmeasured(reg(), RESULTS) == ["not-yet"]
    assert unmeasured(reg(), RESULTS | {"h3": {"p_exact": 0.2}}) == []
