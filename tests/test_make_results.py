"""End to end: answers -> CCA -> confirmatory family -> cascade -> results.json -> ledger -> text.

This is the integration test of the whole analysis chain on mock data, and it holds the assembler and
the claims registry in step: every registered claim must resolve in what the assembler produces."""

from __future__ import annotations

import json
import math

import pytest

from baselines.registry import verbalise_branches
from contract.mock import generate
from detection.nli import NLIScores
from experiments.make_results import SYSTEMS_PRIMARY, assemble, paired_dict, write
from experiments.oracle_cascade import gold_structure_answer
from metrics.case_accuracy import ReaderOutput
from reporting.claims import ClaimError, render, unmeasured
from reporting.registry import default_registry


class NLI:
    def score_batch(self, pairs):
        return [NLIScores(0.95, 0.05, 0.0) for _ in pairs]


def reader(*, query, answer, persona):
    parts = [s.strip() for s in answer.replace(" However, ", ". However, ").split(". ") if s.strip()]
    pick = parts[1] if ("-tier" in persona and len(parts) > 1) else parts[0]
    return ReaderOutput(pick, pick)


def population(n=14):
    ii = [i for i in generate(300, seed=0) if i.personas and i.gold_branches][:n]
    assert len(ii) == n
    return ii


def answers_for(ii):
    """B0 suppresses every exception. B3 states the structure for 4 instances (it 'presented both'); B7c
    suppresses it for the last 4. B8 and O3 state it everywhere. So every comparison has discordant pairs."""
    sel = {i.instance_id: i.selection_answer for i in ii}
    gold = {i.instance_id: gold_structure_answer(i) for i in ii}
    ids = [i.instance_id for i in ii]
    b3 = {k: (gold[k] if k in ids[:4] else sel[k]) for k in ids}
    b7 = {k: (sel[k] if k in ids[-4:] else gold[k]) for k in ids}
    return {"B0": sel, "B3": b3, "B7c": b7, "B8": gold, "O3": gold}


@pytest.fixture(scope="module")
def results():
    ii = population()
    return assemble(ii, answers_for(ii), reader, NLI(), comparator="B7c")


def test_results_are_plain_json_with_no_nan(results):
    text = json.dumps(results)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == results


def test_the_narrowing_rule_is_applied_from_the_data(results):
    assert results["s1"]["n_test"] == 14 and results["s1"]["narrowing"] is True
    assert "exploratory" in results["s1"]["status"]
    assert results["confirmatory"]["exploratory"] is True
    assert not any(results["confirmatory"]["supported"].values()), "significant or not, nothing is confirmatory"


def test_per_system_scores_reflect_what_each_system_states(results):
    cca = results["cca"]
    assert cca["B0"]["cca_all"] == 0.0, "one unconditioned outcome (Proposition 1)"
    assert cca["B3"]["cca_all"] == pytest.approx(4 / 14) and cca["B7c"]["cca_all"] == pytest.approx(10 / 14)
    assert cca["O3"]["cca_all"] == 1.0
    assert cca["B0"]["over_generalisation"] == 0.0 and cca["B0"]["cca_default"] == 1.0
    assert cca["B0"]["cca_exception"] == 0.0, "the people the exception protects are the ones who lose"


def test_h1_and_h3_are_computed_with_the_exact_test_and_a_direction(results):
    h1, h3 = results["h1"], results["h3"]
    assert h1["name_a"] == "B3" and h1["name_b"] == "B0" and h1["diff"] == pytest.approx(4 / 14)
    assert h3["name_a"] == "O3" and h3["name_b"] == "B7c" and h3["diff"] == pytest.approx(4 / 14)
    assert h1["b"] == 4 and h1["c"] == 0 and h1["p_exact"] == pytest.approx(0.125), "exact: 2 * 0.5**4"
    assert results["confirmatory"]["direction_as_predicted"]["H1"] is False, "B3 beat B0 here: not the predicted direction"
    assert results["confirmatory"]["direction_as_predicted"]["H3"] is True
    assert results["confirmatory"]["comparator"] == "B7c"
    assert results["h1"]["underpowered_for"]


def test_the_cascade_is_included_and_consistent(results):
    c = results["cascade"]
    assert c["accuracy"] == {"L0": 1.0, "L1": 1.0, "L2": 1.0} and c["ceiling_gap"] == 0.0


def test_every_registered_claim_resolves_so_the_assembler_and_ledger_cannot_drift_apart(results):
    assert unmeasured(default_registry(), results) == []


def test_a_manuscript_sentence_is_rendered_from_the_results_not_typed(results):
    text = ("On {{s1.n}} instances ({{s1.status}}), O3 reached {{cca.O3}} CCA-all against {{cca.B0}} for full "
            "context (exact p = {{h3.p}}).")
    out = render(text, default_registry(), results)
    assert out == ("On 14 instances (exploratory (below floor)), O3 reached 100.0% CCA-all against 0.0% for "
                   "full context (exact p = 0.1250).")


def test_a_claim_about_an_experiment_that_has_not_run_cannot_be_rendered():
    ii = population()
    partial = assemble(ii, {k: v for k, v in answers_for(ii).items() if k in ("B0", "B3", "O3", "B7c")},
                       reader, NLI(), comparator="B7c")
    assert "B8" not in partial["cca"]
    with pytest.raises(ClaimError, match="not yet measured"):
        render("B8 reached {{cca.B8}}.", default_registry(), partial)


def test_missing_systems_for_the_family_are_an_error():
    ii = population()
    with pytest.raises(KeyError, match="missing system"):
        assemble(ii, {"B0": answers_for(ii)["B0"]}, reader, NLI(), comparator="B7c")


def test_write_round_trips(tmp_path, results):
    p = tmp_path / "r" / "results.json"
    write(results, p)
    assert json.loads(p.read_text(encoding="utf-8")) == results


def test_nan_and_inf_become_none_and_are_then_unreportable():
    from experiments.make_results import _plain

    assert _plain({"a": float("nan"), "b": [float("inf"), 1.0]}) == {"a": None, "b": [None, 1.0]}
    from reporting.claims import Claim, Registry, value_of

    with pytest.raises(ClaimError):
        value_of(Registry([Claim("x", "a", "x")])["x"], {"a": None})


def test_the_default_registry_covers_every_system_and_both_hypotheses():
    reg = default_registry()
    ids = {c.id for c in reg}
    for s in SYSTEMS_PRIMARY:
        assert f"cca.{s}" in ids and f"cca.{s}.lenient" in ids
    assert {"h1.p", "h3.p", "h1.holm", "h3.mdd", "cascade.ceiling", "s1.n"} <= ids
    assert len(reg) > 60


def test_a_comparison_that_cannot_detect_anything_says_so_in_a_form_the_ledger_can_print(results):
    """n = 14 with 4 discordant pairs: the smallest possible exact p is 0.125, above the Holm alpha."""
    assert results["h1"]["mdd_points"] is None
    text = render("This design can detect {{h1.mdd}}.", default_registry(), results)
    assert text == "This design can detect no effect within the observed discordance is detectable at this sample size."


def test_a_comparison_with_power_reports_the_detectable_difference_in_points():
    from experiments.analysis import paired

    a = {f"i{k}": k < 60 for k in range(80)}
    b = {f"i{k}": k < 20 for k in range(80)}
    d = paired_dict(paired(a, b))
    assert d["mdd_points"] is not None and d["mdd_text"].endswith("points") and "no effect" not in d["mdd_text"]
