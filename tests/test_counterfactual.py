"""Counterfactual pairs: reader sensitivity (V2) and system consistency."""

from __future__ import annotations

import pytest

from contract.gold import GoldInstance
from contract.mock import generate
from detection.nli import NLIScores
from metrics.case_accuracy import NOT_DETERMINED, ReaderOutput
from metrics.counterfactual import (
    IRRELEVANT_CHANGE_MAX,
    RELEVANT_CHANGE_MIN,
    Sensitivity,
    counterfactual_consistency,
    reader_sensitivity,
)


def _instances() -> list[GoldInstance]:
    return [i for i in generate(120, seed=0) if i.persona_pairs and i.gold_scoped_answer]


class ConstNLI:
    def score_batch(self, pairs):
        return [NLIScores(0.95, 0.05, 0.0) for _ in pairs]


def _sentences(inst):
    first, second = inst.gold_scoped_answer.split(" However, ")
    return first, "However, " + second


def make_reader(inst, *, flips_on_city=False, ignores_tier=False, silent_on=()):
    """A scripted reader keyed on persona text, with switchable defects."""
    first, second = _sentences(inst)
    d, e = inst.branch_for_persona("u_default").outcome, inst.branch_for_persona("u_exception").outcome

    def reader(*, query, answer, persona):
        if any(s in persona for s in silent_on):
            return ReaderOutput(NOT_DETERMINED, "")
        if "York" in persona:
            return ReaderOutput(e if flips_on_city else d, second if flips_on_city else first)
        if "-tier" in persona and not ignores_tier:
            return ReaderOutput(e, second)
        return ReaderOutput(d, first)

    return reader



def test_the_mock_provides_pairs_of_both_kinds():
    flips = {p.flip for i in _instances() for p in i.persona_pairs}
    assert flips == {"relevant", "irrelevant"}


def test_a_sound_reader_passes_v2():
    insts = _instances()
    # one reader per instance: the script is keyed on that instance's outcomes
    s = Sensitivity(0, 0, 0, 0)
    for inst in insts:
        one = reader_sensitivity([inst], make_reader(inst), ConstNLI())
        s.n_relevant += one.n_relevant; s.n_irrelevant += one.n_irrelevant
        s.relevant_changed += one.relevant_changed; s.irrelevant_changed += one.irrelevant_changed
    assert s.relevant_changed == s.n_relevant > 0
    assert s.irrelevant_changed == 0 and s.n_irrelevant > 0
    assert s.passes_v2 is True
    assert "V2 sensitivity: PASS" in s.render()


def test_a_reader_thrown_by_an_irrelevant_attribute_fails_v2():
    inst = _instances()[0]
    s = reader_sensitivity([inst], make_reader(inst, flips_on_city=True), ConstNLI())
    assert s.irrelevant_changed == s.n_irrelevant == 1
    assert s.passes_v2 is False


def test_a_reader_that_ignores_the_relevant_attribute_fails_v2():
    inst = _instances()[0]
    s = reader_sensitivity([inst], make_reader(inst, ignores_tier=True), ConstNLI())
    assert s.relevant_changed == 0 and s.passes_v2 is False


def test_a_reader_whose_answer_evaporates_is_unstable_not_stable():
    """Determined -> undetermined is a change: it must not count as 'held'."""
    inst = _instances()[0]
    s = reader_sensitivity([inst], make_reader(inst, silent_on=("York",)), ConstNLI())
    assert s.irrelevant_changed == 1 and s.passes_v2 is False


def test_v2_is_not_evaluated_without_both_kinds_of_pair():
    assert Sensitivity(0, 3, 0, 0).passes_v2 is None
    assert Sensitivity(3, 0, 3, 0).passes_v2 is None
    assert "NOT EVALUATED" in Sensitivity(0, 0, 0, 0).render()


def test_v2_boundaries_are_inclusive():
    n = 20
    ok = Sensitivity(n_relevant=n, n_irrelevant=n, relevant_changed=int(RELEVANT_CHANGE_MIN * n),
                     irrelevant_changed=int(IRRELEVANT_CHANGE_MAX * n))
    assert ok.passes_v2 is True
    assert Sensitivity(n, n, int(RELEVANT_CHANGE_MIN * n) - 1, 0).passes_v2 is False
    assert Sensitivity(n, n, n, int(IRRELEVANT_CHANGE_MAX * n) + 1).passes_v2 is False


def test_instances_without_pairs_are_ignored_by_sensitivity():
    bare = _instances()[0].model_copy(update={"persona_pairs": []})
    assert reader_sensitivity([bare], lambda **k: ReaderOutput(NOT_DETERMINED, ""), ConstNLI()).n_relevant == 0


# --------------------------------------------------------------------------- #
# System-level consistency
# --------------------------------------------------------------------------- #


def test_a_scoped_answer_is_consistent_on_every_pair():
    inst = _instances()[0]
    rate, vs = counterfactual_consistency([inst], {inst.instance_id: inst.gold_scoped_answer},
                                          make_reader(inst), ConstNLI())
    assert rate.point == 1.0 and all(v.consistent and v.determined for v in vs)


def test_an_answer_that_leaves_people_undetermined_is_not_consistent():
    """Holding steady by saying nothing is not consistency."""
    inst = _instances()[0]
    reader = lambda **k: ReaderOutput(NOT_DETERMINED, "")  # noqa: E731
    rate, vs = counterfactual_consistency([inst], {inst.instance_id: "x"}, reader, ConstNLI())
    assert rate.point == 0.0
    assert not any(v.determined for v in vs)


def test_a_selection_answer_fails_the_relevant_pair_but_can_pass_the_irrelevant_one():
    inst = _instances()[0]
    first, _ = _sentences(inst)
    d = inst.branch_for_persona("u_default").outcome
    selection_reader = lambda **k: ReaderOutput(d, first)  # noqa: E731  (everyone gets the rule)
    rate, vs = counterfactual_consistency([inst], {inst.instance_id: first}, selection_reader, ConstNLI())
    by_flip = {v.flip: v.consistent for v in vs}
    assert by_flip == {"relevant": False, "irrelevant": True}
    assert rate.point == 0.5


def test_missing_answer_is_an_error_not_a_zero():
    inst = _instances()[0]
    with pytest.raises(KeyError, match="not a wrong one"):
        counterfactual_consistency([inst], {}, make_reader(inst), ConstNLI())


def test_no_pairs_gives_no_score():
    rate, vs = counterfactual_consistency([], {}, lambda **k: None, ConstNLI())
    assert rate is None and vs == []
