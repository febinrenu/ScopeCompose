"""The oracle cascade: gold substituted stage by stage on the same instances."""

from __future__ import annotations

import pytest

from contract.gold import Applicability, AttributeKind, Branch, ScopeAttribute
from contract.mock import generate
from detection.nli import NLIScores
from experiments.oracle_cascade import (
    Cascade,
    cascade,
    gold_frames_answer,
    gold_structure_answer,
)
from metrics.case_accuracy import NOT_DETERMINED, ReaderOutput


class NLI:
    def score_batch(self, pairs):
        return [NLIScores(0.95, 0.05, 0.0) for _ in pairs]


def insts(n=12):
    out = [i for i in generate(200, seed=0) if i.personas and i.gold_branches][:n]
    assert len(out) == n
    return out


def sentence_reader(*, query, answer, persona):
    """Quotes the sentence addressing the person; says nothing if none does."""
    parts = [s.strip() for s in answer.replace(" However, ", ". However, ").split(". ") if s.strip()]
    pick = parts[1] if ("-tier" in persona and len(parts) > 1) else parts[0]
    return ReaderOutput(pick, pick)


# --------------------------------------------------------------------------- #
# The oracle answers
# --------------------------------------------------------------------------- #


def test_gold_structure_is_the_reference_answer_by_construction():
    inst = insts(1)[0]
    assert gold_structure_answer(inst).startswith(inst.default_branch.outcome[0].upper() + inst.default_branch.outcome[1:])
    for b in inst.exception_branches:
        assert b.outcome.lower() in gold_structure_answer(inst).lower()


def test_gold_frames_keep_every_gold_outcome_and_attach_each_to_its_group():
    inst = insts(1)[0]
    text = gold_frames_answer(inst)
    for b in inst.gold_branches:
        assert b.outcome.lower() in text.lower(), text
    assert "However, if" in text or "If " in text, "the exception is attached to its condition, not just listed"


def test_composition_changes_the_gold_only_where_the_operator_does_something_to_it():
    """With typed, nested scopes the operator splits overlapping regions; with untyped descriptor-only
    scopes it states them verbatim. Either way no outcome disappears."""
    inst = insts(1)[0]
    typed = Applicability(descriptor="premium holders",
                          attributes=[ScopeAttribute(name="tier", kind=AttributeKind.CATEGORICAL, values=["premium"])])
    exc = inst.exception_branches[0].model_copy(update={"applicability": typed})
    modified = inst.model_copy(update={"gold_branches": [inst.default_branch, exc]})
    out = gold_frames_answer(modified)
    assert inst.default_branch.outcome.lower() in out.lower() and exc.outcome.lower() in out.lower()


# --------------------------------------------------------------------------- #
# The cascade
# --------------------------------------------------------------------------- #


def test_a_system_that_suppresses_every_exception_loses_everything_at_the_extraction_stage():
    ii = insts(12)
    selection = {i.instance_id: i.selection_answer for i in ii}        # only the general rule
    c = cascade(ii, selection, sentence_reader, NLI())
    assert isinstance(c, Cascade)
    assert c.accuracy("L0") == 0.0, "one unconditioned outcome is wrong for the exception person (Proposition 1)"
    assert c.accuracy("L2") == 1.0 and c.ceiling_gap == 0.0
    assert c.accuracy("L1") == pytest.approx(1.0)
    assert c.deltas["L1-L0"].diff == pytest.approx(1.0) and c.deltas["L1-L0"].p_exact < 0.01
    assert c.deltas["L2-L1"].diff == 0.0, "composition costs nothing when the scopes are simple"


def test_a_perfect_system_has_no_deltas():
    ii = insts(12)
    perfect = {i.instance_id: gold_structure_answer(i) for i in ii}
    c = cascade(ii, perfect, sentence_reader, NLI())
    assert [c.accuracy(l) for l in ("L0", "L1", "L2")] == [1.0, 1.0, 1.0]
    assert all(d.diff == 0 for d in c.deltas.values())


def test_the_ceiling_is_a_property_of_the_reader_not_the_system():
    """A reader that cannot parse even the gold structure puts the ceiling below 1."""
    ii = insts(12)
    blind = lambda *, query, answer, persona: ReaderOutput(NOT_DETERMINED, "")  # noqa: E731
    c = cascade(ii, {i.instance_id: gold_structure_answer(i) for i in ii}, blind, NLI())
    assert c.accuracy("L2") == 0.0 and c.ceiling_gap == 1.0
    assert "not of the system" in c.render()


def test_levels_are_scored_on_the_same_instances_and_missing_answers_are_errors():
    ii = insts(6)
    with pytest.raises(KeyError):
        cascade(ii, {ii[0].instance_id: "x"}, sentence_reader, NLI())
    c = cascade(ii, {i.instance_id: i.selection_answer for i in ii}, sentence_reader, NLI())
    assert set(c.levels["L0"]) == set(c.levels["L1"]) == set(c.levels["L2"])


def test_instances_without_branches_or_personas_are_not_eligible():
    ii = insts(4)
    bare = [i.model_copy(update={"personas": [], "persona_pairs": []}) for i in ii]
    with pytest.raises(ValueError, match="gold branches and personas"):
        cascade(bare, {}, sentence_reader, NLI())


def test_render_lists_every_level_delta_and_the_ceiling():
    ii = insts(12)
    text = cascade(ii, {i.instance_id: i.selection_answer for i in ii}, sentence_reader, NLI()).render()
    for token in ("L0", "L1", "L2", "extraction + alignment", "composition + verbalisation", "ceiling"):
        assert token in text
