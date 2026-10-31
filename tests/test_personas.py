"""Persona generation: structural validation of minimal pairs, one repair, and
attachment that the contract re-checks."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog
from benchmark.personas import (
    PersonaProposal,
    generate_personas,
    user_prompt,
    validate,
    with_personas,
)
from contract.gold import GoldInstance
from contract.mock import generate


def _inst() -> GoldInstance:
    base = next(i for i in generate(60, seed=0) if i.personas)
    return base.model_copy(update={"personas": [], "persona_pairs": []})


def _good(inst: GoldInstance) -> dict:
    exc = inst.exception_branches[0].branch_id
    return {
        "default": "I hold an ordinary current account and I send money abroad each month.",
        "exceptions": {exc: "I hold a premium current account and I send money abroad each month."},
        "irrelevant": {"attribute": "town of residence",
                       "statement": "I hold an ordinary current account and I send money abroad each month "
                                    "from my flat in Leeds."},
    }


def _prop(inst, **over) -> PersonaProposal:
    g = _good(inst)
    p = PersonaProposal(inst.instance_id, g["default"], dict(g["exceptions"]),
                        g["irrelevant"]["attribute"], g["irrelevant"]["statement"])
    for k, v in over.items():
        setattr(p, k, v)
    return p


def client_with(tmp_path, replies):
    """Scripted backend that returns each reply in turn."""
    it = iter(replies)

    class Seq(MockBackend):
        def complete(self, **kw):
            self.calls.append(kw)
            text = next(it)
            return self._wrap(text if isinstance(text, str) else json.dumps(text), kw["model"], "x")

    b = Seq()
    return LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                     backend_override=b), b


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def test_a_well_formed_proposal_has_no_problems():
    inst = _inst()
    assert validate(inst, _prop(inst)) == []


@pytest.mark.parametrize("default,expect", [
    ("", "empty"),
    ("I pay.", "words"),
    ("The customer holds an ordinary current account and sends money abroad.", "first person"),
    ("I hold a card_tier standard account and send money abroad each month.", "leaks"),
    ("I hold an account with sig:event set and send money abroad each month.", "leaks"),
])
def test_default_statement_defects_are_reported(default, expect):
    inst = _inst()
    probs = validate(inst, _prop(inst, default=default))
    assert any(expect in p for p in probs), probs


def test_an_exception_identical_to_the_default_is_not_a_condition_made_true():
    inst = _inst()
    p = _prop(inst)
    p.exceptions = {inst.exception_branches[0].branch_id: p.default}
    assert any("identical to the default" in x for x in validate(inst, p))


def test_an_exception_that_rewrites_the_whole_sentence_is_not_a_minimal_edit():
    inst = _inst()
    p = _prop(inst)
    p.exceptions = {inst.exception_branches[0].branch_id: "I was given a platinum card by a bank last year for travelling."}
    assert any("minimal edit" in x for x in validate(inst, p))


def test_the_irrelevant_variant_must_keep_the_default_and_add_a_fact_no_condition_mentions():
    inst = _inst()
    p = _prop(inst, irrelevant="I live in Leeds and enjoy hiking at weekends with friends.")
    assert any("keeps only" in x for x in validate(inst, p))
    cond = inst.exception_branches[0].condition
    p2 = _prop(inst)
    p2.irrelevant = p2.default + f" {cond}."
    assert any("overlaps a branch condition" in x for x in validate(inst, p2))


def test_exceptions_must_cover_exactly_the_exception_branches():
    inst = _inst()
    assert any("must cover exactly" in x for x in validate(inst, _prop(inst, exceptions={})))
    assert any("must cover exactly" in x for x in validate(inst, _prop(inst, exceptions={
        inst.exception_branches[0].branch_id: "I hold a premium account and send money abroad monthly.",
        "bogus": "I hold a premium account and send money abroad monthly."})))


def test_the_unrelated_attribute_must_be_named():
    inst = _inst()
    assert any("not named" in x for x in validate(inst, _prop(inst, irrelevant_attribute="")))


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def test_the_prompt_lists_the_exception_conditions_and_withholds_every_outcome():
    inst = _inst()
    p = user_prompt(inst)
    assert inst.query in p and inst.exception_branches[0].condition in p
    assert all(b.outcome not in p for b in inst.gold_branches), "outcomes leak into the descriptions if shown"
    assert "id " + inst.exception_branches[0].branch_id in p


def test_a_good_first_reply_is_accepted_without_a_reask(tmp_path):
    inst = _inst()
    c, b = client_with(tmp_path, [_good(inst)])
    prop = generate_personas(inst, client=c)
    assert prop.ok and len(b.calls) == 1


def test_a_bad_reply_is_repaired_once(tmp_path):
    inst = _inst()
    bad = _good(inst) | {"default": "I pay."}
    c, b = client_with(tmp_path, [bad, _good(inst)])
    prop = generate_personas(inst, client=c)
    assert prop.ok and len(b.calls) == 2
    assert "problems" in b.calls[1]["messages"][-1]["content"]


def test_a_reply_that_stays_bad_is_returned_with_its_problems_not_hidden(tmp_path):
    inst = _inst()
    bad = _good(inst) | {"default": "I pay."}
    c, b = client_with(tmp_path, [bad, bad])
    prop = generate_personas(inst, client=c)
    assert not prop.ok and prop.problems and len(b.calls) == 2


def test_a_worse_repair_does_not_replace_a_better_first_attempt(tmp_path):
    inst = _inst()
    first = _good(inst) | {"default": "I pay."}                 # one problem
    worse = {"default": "", "exceptions": {}, "irrelevant": {}}  # many problems
    c, _ = client_with(tmp_path, [first, worse])
    prop = generate_personas(inst, client=c)
    assert prop.default == "I pay."


def test_garbage_is_a_problem_not_a_crash(tmp_path):
    inst = _inst()
    c, _ = client_with(tmp_path, ["not json", "still not json"])
    assert not generate_personas(inst, client=c).ok


def test_personas_need_a_default_and_an_exception():
    inst = _inst()
    no_exc = inst.model_copy(update={"gold_branches": [inst.default_branch], "gold_scope_relation": None,
                                     "gold_conflict_type": inst.gold_conflict_type})
    with pytest.raises(ValueError, match="at least one exception"):
        generate_personas(no_exc, client=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# attach
# --------------------------------------------------------------------------- #


def test_attaching_builds_personas_and_both_kinds_of_pair_the_contract_accepts():
    inst = _inst()
    out = with_personas(inst, _prop(inst))
    assert {p.persona_id for p in out.personas} == {"u_default", f"u_{inst.exception_branches[0].branch_id}",
                                                    "u_default_irrelevant"}
    assert {p.flip for p in out.persona_pairs} == {"relevant", "irrelevant"}
    assert out.branch_for_persona("u_default_irrelevant").is_default
    assert not out.branch_for_persona(f"u_{inst.exception_branches[0].branch_id}").is_default


def test_a_proposal_with_problems_is_refused():
    inst = _inst()
    p = _prop(inst)
    p.problems = ["something"]
    with pytest.raises(ValueError, match="refusing"):
        with_personas(inst, p)


def test_a_relevant_pair_that_would_not_change_the_gold_is_refused_by_the_contract():
    """Default and exception with the same outcome: not a relevant flip."""
    inst = _inst()
    d = inst.default_branch
    same = inst.model_copy(update={"gold_branches": [
        d, inst.exception_branches[0].model_copy(update={"outcome": d.outcome})]})
    with pytest.raises(ValidationError, match="agree in outcome"):
        with_personas(same, _prop(same))


def test_a_statement_that_states_an_outcome_is_a_problem():
    """The first live run produced 'I do not pay any fee when ...' and 'I pay a two-fifty fee ...': the answer."""
    inst = _inst()
    default_outcome = inst.default_branch.outcome
    prop = _prop(inst, default=f"I hold an ordinary current account and {default_outcome} on my monthly payments.")
    problems = validate(inst, prop)
    assert any("states an outcome" in p and p.startswith("default") for p in problems), problems
    assert not any("states an outcome" in p for p in validate(inst, _prop(inst)))
