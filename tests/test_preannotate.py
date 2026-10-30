"""Branch pre-annotation: EVAL-only, marked as a proposal, answers derived, edits re-derive them."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from benchmark.preannotate import (
    MODEL_PROPOSED,
    apply_edits,
    branch_provenance,
    derive_answers,
    propose,
    set_branch_provenance,
)
from contract.gold import GoldInstance
from contract.mock import generate
from contract.models import ConflictType, ScopeRelation


def skeleton() -> GoldInstance:
    """An adjudicated refinement with passages and a query but no branches yet."""
    base = next(i for i in generate(60, seed=0) if i.gold_scope_relation is ScopeRelation.REFINEMENT)
    return base.model_copy(update={"gold_branches": [], "gold_scoped_answer": None, "selection_answer": None,
                                   "personas": [], "persona_pairs": []})


def good_reply(inst: GoldInstance) -> dict:
    p0, p1 = inst.passages[0].id, inst.passages[1].id
    return {"branches": [
        {"is_default": True, "condition": None, "applies_to": "everyone else", "outcome": "a 3% fee applies",
         "source": p0},
        {"is_default": False, "condition": "the holder is a premium customer", "applies_to": "premium customers",
         "outcome": "no fee applies", "source": p1}]}


def client_with(tmp_path, reply):
    backend = MockBackend(scripted={"Question:": reply if isinstance(reply, str) else json.dumps(reply)})
    return (LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                      backend_override=backend), backend)


# --------------------------------------------------------------------------- #
# propose
# --------------------------------------------------------------------------- #


def test_a_valid_proposal_yields_a_valid_instance_marked_model_proposed(tmp_path):
    inst = skeleton()
    c, backend = client_with(tmp_path, good_reply(inst))
    out = propose(inst, client=c)
    assert isinstance(out, GoldInstance) and len(out.gold_branches) == 2
    assert out.default_branch.outcome == "a 3% fee applies"
    assert branch_provenance(out) == MODEL_PROPOSED
    assert out.annotation.annotator_a == inst.annotation.annotator_a, "type/relation provenance is untouched"
    assert backend.calls[0]["model"] == "qwen/qwen3.8-27b"


def test_the_answers_are_derived_from_the_branches_not_written_by_the_model(tmp_path):
    inst = skeleton()
    c, _ = client_with(tmp_path, good_reply(inst) | {"scoped_answer": "THE MODEL SAYS THIS", "selection_answer": "x"})
    out = propose(inst, client=c)
    assert "THE MODEL SAYS THIS" not in out.gold_scoped_answer
    assert out.gold_scoped_answer.startswith("A 3% fee applies (per")
    assert "However, if the holder is a premium customer, no fee applies" in out.gold_scoped_answer
    assert out.selection_answer == "A 3% fee applies."


@pytest.mark.parametrize("tier", [Tier.BULK, Tier.BACKBONE_PRIMARY, Tier.JUDGE, Tier.READER])
def test_only_the_eval_tier_may_pre_annotate(tmp_path, tier):
    inst = skeleton()
    c, _ = client_with(tmp_path, good_reply(inst))
    with pytest.raises(ValueError, match="EVAL tier"):
        propose(inst, client=c, tier=tier)


def test_the_proposal_does_not_use_an_evaluated_systems_prompt(tmp_path):
    import baselines.registry as reg
    from benchmark.preannotate import SYSTEM

    assert SYSTEM != reg.PARSE_SYSTEM and SYSTEM != reg.ANSWER_SYSTEM
    inst = skeleton()
    c, backend = client_with(tmp_path, good_reply(inst))
    propose(inst, client=c)
    assert backend.calls[0]["system"] == SYSTEM


@pytest.mark.parametrize("reply", [
    {"branches": []},
    {"branches": [{"is_default": True, "outcome": "only a default", "source": "p0"}]},
    {"branches": [{"is_default": False, "condition": "a", "outcome": "x", "source": "p0"},
                  {"is_default": False, "condition": "b", "outcome": "y", "source": "p1"}]},
    "not json",
])
def test_a_proposal_without_one_default_and_an_exception_is_refused(tmp_path, reply):
    c, _ = client_with(tmp_path, reply)
    with pytest.raises(ValueError, match="needs one default"):
        propose(skeleton(), client=c)


def test_only_an_adjudicated_refinement_is_eligible(tmp_path):
    base = skeleton()
    c, _ = client_with(tmp_path, good_reply(base))
    for update in ({"gold_scope_relation": ScopeRelation.DISJOINT}, {"gold_conflict_type": ConflictType.NO_CONFLICT,
                                                                       "gold_scope_relation": None}):
        with pytest.raises(ValueError, match="only an adjudicated refinement"):
            propose(base.model_copy(update=update), client=c)
    with pytest.raises(ValueError, match="at least two passages"):
        propose(base.model_copy(update={"passages": base.passages[:1]}), client=c)


def test_an_unknown_citation_is_dropped_by_the_parser_not_remapped(tmp_path):
    inst = skeleton()
    reply = good_reply(inst)
    reply["branches"][1]["source"] = "p99"
    c, _ = client_with(tmp_path, reply)
    with pytest.raises(Exception):
        propose(inst, client=c)           # an ungrounded exception cannot be gold (contract validator)


# --------------------------------------------------------------------------- #
# edits
# --------------------------------------------------------------------------- #


def proposed(tmp_path) -> GoldInstance:
    inst = skeleton()
    c, _ = client_with(tmp_path, good_reply(inst))
    return propose(inst, client=c)


def test_an_edit_re_derives_the_answers_and_replaces_the_model_marker_with_the_reviewer(tmp_path):
    inst = proposed(tmp_path)
    exc = inst.exception_branches[0].branch_id
    out = apply_edits(inst, {exc: {"outcome": "a 1% fee applies", "condition": "the holder is a student"}},
                      annotator="johann")
    assert branch_provenance(out) == "reviewed:johann"
    assert out.annotation.annotator_a == inst.annotation.annotator_a, "the type/relation annotator is not overwritten"
    assert "a 1% fee applies" in out.gold_scoped_answer and "student" in out.gold_scoped_answer
    assert "no fee applies" not in out.gold_scoped_answer, "the answer cannot disagree with its branches"
    assert out.exception_branches[0].applicability.descriptor == "the holder is a student"


def test_editing_the_default_outcome_updates_the_selection_answer(tmp_path):
    inst = proposed(tmp_path)
    out = apply_edits(inst, {inst.default_branch.branch_id: {"outcome": "a 2% fee applies"}}, annotator="johann")
    assert out.selection_answer == "A 2% fee applies."


def test_a_reviewer_must_be_a_person(tmp_path):
    inst = proposed(tmp_path)
    for bad in ("", "  ", MODEL_PROPOSED, "model", "model_proposed"):
        with pytest.raises(ValueError, match="human annotator"):
            apply_edits(inst, {}, annotator=bad)


def test_edits_are_validated(tmp_path):
    inst = proposed(tmp_path)
    exc = inst.exception_branches[0].branch_id
    with pytest.raises(KeyError):
        apply_edits(inst, {"nope": {"outcome": "x"}}, annotator="johann")
    with pytest.raises(ValueError, match="empty outcome"):
        apply_edits(inst, {exc: {"outcome": "  "}}, annotator="johann")
    with pytest.raises(ValueError, match="cannot lose its condition"):
        apply_edits(inst, {exc: {"condition": ""}}, annotator="johann")


def test_no_edits_still_stamps_the_reviewer(tmp_path):
    inst = proposed(tmp_path)
    out = apply_edits(inst, {}, annotator="febin")
    assert branch_provenance(out) == "reviewed:febin" and out.gold_scoped_answer == inst.gold_scoped_answer


def test_provenance_notes_keep_other_notes_and_replace_only_the_branch_token():
    assert set_branch_provenance(None, "model-proposed") == "branches=model-proposed"
    once = set_branch_provenance("pilot 9 | check wording", "model-proposed")
    assert once == "pilot 9 | check wording | branches=model-proposed"
    assert set_branch_provenance(once, "reviewed:johann") == "pilot 9 | check wording | branches=reviewed:johann"


def test_derive_answers_needs_a_default():
    with pytest.raises(StopIteration):
        derive_answers([])
