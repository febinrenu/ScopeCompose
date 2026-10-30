"""Case-complete accuracy: each strict condition in isolation, and the property
the metric exists for -- a reader that fills a gap passes lenient and fails strict."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog
from contract.gold import GoldInstance
from contract.mock import generate
from contract.models import ScopeRelation
from detection.nli import NLIScores
from metrics.case_accuracy import (
    NOT_DETERMINED,
    READER_PROMPT_SHA,
    Failed,
    LLMReader,
    ReaderOutput,
    aggregate,
    evidence_is_verbatim,
    is_clarification_only,
    parse_reader_json,
    score_corpus,
    score_instance,
    score_persona,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _instance() -> GoldInstance:
    return next(i for i in generate(80, seed=0)
                if i.gold_scope_relation is ScopeRelation.REFINEMENT and i.personas)


@pytest.fixture()
def inst() -> GoldInstance:
    return _instance()


def _sentences(inst: GoldInstance) -> tuple[str, str]:
    """(default sentence, exception sentence) of the gold scoped answer."""
    a = inst.gold_scoped_answer
    assert a
    first, second = a.split(" However, ")
    return first, "However, " + second


class ConstNLI:
    def __init__(self, entailment: float):
        self.e = entailment
        self.calls: list[tuple[str, str]] = []

    def score_batch(self, pairs):
        self.calls.extend(pairs)
        return [NLIScores(self.e, 1 - self.e, 0.0) for _ in pairs]


def reader_for(inst: GoldInstance, answer: str, **override: ReaderOutput):
    """A scripted reader: the ideal one, with per-persona overrides."""
    first, second = _sentences(inst)
    by_statement = {
        inst.personas[0].statement: ReaderOutput(inst.branch_for_persona("u_default").outcome, first),
        inst.personas[1].statement: ReaderOutput(inst.branch_for_persona("u_exception").outcome, second),
        inst.personas[2].statement: ReaderOutput(inst.branch_for_persona("u_default_city").outcome, first),
    }
    for pid, out in override.items():
        by_statement[next(p.statement for p in inst.personas if p.persona_id == pid)] = out

    def reader(*, query: str, answer: str, persona: str) -> ReaderOutput:
        return by_statement[persona]

    return reader


# --------------------------------------------------------------------------- #
# The ideal case
# --------------------------------------------------------------------------- #


def test_gold_answer_with_an_ideal_reader_is_correct_for_every_persona(inst):
    answer = inst.gold_scoped_answer
    res = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(0.95))
    assert res.cca_all and res.cca_all_lenient
    assert all(v.failed is None for v in res.verdicts)


# --------------------------------------------------------------------------- #
# Each strict condition, alone
# --------------------------------------------------------------------------- #


def test_condition_0_reader_says_not_determined(inst):
    answer = inst.gold_scoped_answer
    r = reader_for(inst, answer, u_exception=ReaderOutput(NOT_DETERMINED, ""))
    res = score_instance(inst, answer, r, ConstNLI(0.95))
    v = res.verdicts[1]
    assert v.failed is Failed.NOT_DETERMINED and not v.strict_correct and not v.lenient_correct
    assert not res.cca_all


def test_condition_1_evidence_must_be_verbatim_in_the_answer(inst):
    answer = inst.gold_scoped_answer
    right = inst.branch_for_persona("u_exception").outcome
    r = reader_for(inst, answer, u_exception=ReaderOutput(right, "Premium holders never pay anything at all."))
    v = score_instance(inst, answer, r, ConstNLI(0.95)).verdicts[1]
    assert v.failed is Failed.NOT_VERBATIM
    assert not v.strict_correct and v.lenient_correct, "right outcome, unsupported quote: lenient passes, strict does not"


def test_condition_2_the_quoted_span_must_entail_the_outcome(inst):
    answer = inst.gold_scoped_answer
    v = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(0.10)).verdicts[1]
    assert v.failed is Failed.NOT_ENTAILED and v.lenient_correct and not v.strict_correct
    assert v.entailment == pytest.approx(0.10)


def test_condition_2_uses_evidence_plus_persona_as_premise_and_the_outcome_as_hypothesis(inst):
    answer = inst.gold_scoped_answer
    nli = ConstNLI(0.95)
    score_persona(inst, inst.personas[1], answer, reader_for(inst, answer), nli)
    premise, hypothesis = nli.calls[0]
    assert inst.personas[1].statement in premise
    assert _sentences(inst)[1] in premise
    assert inst.branch_for_persona("u_exception").outcome in hypothesis


def test_condition_3_hedged_evidence_never_determines(inst):
    """The reviewer's example: 'Premium customers may receive fee waivers'."""
    hedged = "Premium customers may receive fee waivers"
    answer = f"{_sentences(inst)[0]} {hedged}."
    right = inst.branch_for_persona("u_exception").outcome
    r = reader_for(inst, answer, u_exception=ReaderOutput(right, hedged))
    v = score_instance(inst, answer, r, ConstNLI(0.99)).verdicts[1]
    assert v.failed is Failed.HEDGED
    assert not v.strict_correct and v.lenient_correct


def test_condition_4_wrong_outcome_is_wrong_everywhere(inst):
    answer = inst.gold_scoped_answer
    r = reader_for(inst, answer, u_exception=ReaderOutput("a 99% fee applies", _sentences(inst)[1]))
    v = score_instance(inst, answer, r, ConstNLI(0.95)).verdicts[1]
    assert v.failed is Failed.WRONG and not v.strict_correct and not v.lenient_correct


def test_unparseable_reader_output_is_not_a_guess(inst):
    answer = inst.gold_scoped_answer
    bad = ReaderOutput(NOT_DETERMINED, "", raw="I think the fee is waived", parse_ok=False)
    v = score_instance(inst, answer, reader_for(inst, answer, u_exception=bad), ConstNLI(0.95)).verdicts[1]
    assert v.failed is Failed.UNPARSEABLE and not v.lenient_correct


# --------------------------------------------------------------------------- #
# The property the metric exists for
# --------------------------------------------------------------------------- #


def test_a_selection_answer_scores_zero_even_with_a_gap_filling_reader(inst):
    """Only the general rule is stated. A reader that 'knows' premium holders are
    exempt says so and quotes the general-rule sentence -- the quote is verbatim,
    but it does not entail the exception outcome, so strict refuses it."""
    general_only = _sentences(inst)[0]
    exc_outcome = inst.branch_for_persona("u_exception").outcome
    filler = reader_for(inst, general_only, u_exception=ReaderOutput(exc_outcome, general_only))

    class EntailsOnlyWhatItSays:
        def score_batch(self, pairs):
            # entailed only if the hypothesis outcome is the general rule's outcome
            default = inst.branch_for_persona("u_default").outcome
            return [NLIScores(0.95 if default in h else 0.02, 0.0, 0.0) for _, h in pairs]

    res = score_instance(inst, general_only, filler, EntailsOnlyWhatItSays())
    exc = res.verdicts[1]
    assert exc.lenient_correct, "lenient credits the guess"
    assert not exc.strict_correct and exc.failed is Failed.NOT_ENTAILED
    assert not res.cca_all, "CCA-all is 0 for any answer that states one unconditioned outcome"


def test_strict_is_never_more_generous_than_lenient(inst):
    answer = inst.gold_scoped_answer
    for e in (0.0, 0.3, 0.6, 0.95):
        res = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(e))
        for v in res.verdicts:
            assert not (v.strict_correct and not v.lenient_correct)


def test_over_generalisation_flags_a_default_person_told_the_exception(inst):
    answer = inst.gold_scoped_answer
    exc_outcome = inst.branch_for_persona("u_exception").outcome
    r = reader_for(inst, answer, u_default=ReaderOutput(exc_outcome, _sentences(inst)[1]))
    v = score_instance(inst, answer, r, ConstNLI(0.95)).verdicts[0]
    assert v.over_generalised and not v.strict_correct


def test_a_correct_default_is_not_over_generalised(inst):
    answer = inst.gold_scoped_answer
    res = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(0.95))
    assert not any(v.over_generalised for v in res.verdicts)


# --------------------------------------------------------------------------- #
# Evidence normalisation, parsing, clarification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("evidence,expected", [
    ("The fee is waived for premium cardholders.", True),
    ("the fee is waived for premium cardholders", True),
    ("  The fee   is waived\nfor premium cardholders  ", True),
    ('"The fee is waived for premium cardholders."', True),
    ("The fee is waived for premium cardholders...", True),
    ("The fee is waived for platinum cardholders.", False),
    ("The fee is waived for everyone.", False),
    ("", False),
    ("   ", False),
])
def test_evidence_is_verbatim(evidence, expected):
    answer = "Transfers normally incur a 3% fee. The fee is waived for premium cardholders. See terms."
    assert evidence_is_verbatim(evidence, answer) is expected


def test_parse_reader_json_variants():
    ok = parse_reader_json('{"outcome": "no fee applies", "evidence": "The fee is waived.", "condition_used": "premium"}')
    assert ok.parse_ok and ok.outcome == "no fee applies" and ok.condition_used == "premium"
    fenced = parse_reader_json('Sure!\n```json\n{"outcome": "x", "evidence": "y", "condition_used": null}\n```')
    assert fenced.parse_ok and fenced.condition_used is None
    for junk in ("", "no json here", '{"outcome": 3}', "{broken"):
        r = parse_reader_json(junk)
        assert not r.parse_ok and r.outcome == NOT_DETERMINED


@pytest.mark.parametrize("answer,expected", [
    ("Could you tell me which account type you hold?", True),
    ("Which card do you have? And are you a student?", True),
    ("A 3% fee applies. Which card do you have?", False),
    ("Premium cardholders pay 3%? No, they pay nothing.", False),
    ("The fee is 3%.", False),
    ("", False),
])
def test_clarification_only(answer, expected):
    assert is_clarification_only(answer) is expected


def test_reader_prompt_is_hashed_for_the_preregistration():
    assert len(READER_PROMPT_SHA) == 64


# --------------------------------------------------------------------------- #
# The production reader, through the cached client, on the mock backend
# --------------------------------------------------------------------------- #


def test_llm_reader_parses_a_scripted_reply(tmp_path):
    backend = MockBackend(scripted={"Person:": json.dumps(
        {"outcome": "no fee applies", "evidence": "The fee is waived.", "condition_used": None})})
    client = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                       backend_override=backend)
    out = LLMReader(client=client)(query="q", answer="The fee is waived.", persona="I hold a premium card.")
    assert out.parse_ok and out.outcome == "no fee applies"
    call = backend.calls[0]
    assert "COPIED WORD FOR WORD" in call["system"] and "Do not use outside knowledge" in call["system"]
    assert "The fee is waived." in call["messages"][0]["content"]


def test_llm_reader_uses_the_pinned_reader_tier(tmp_path):
    backend = MockBackend()
    client = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                       backend_override=backend)
    LLMReader(client=client)(query="q", answer="a", persona="p")
    assert backend.calls[0]["model"] == "scopecompose-reader"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_counts_and_failure_breakdown(inst):
    answer = inst.gold_scoped_answer
    good = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(0.95))
    bad = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(0.05))
    scores = aggregate([good, bad], threshold=0.5, threshold_tuned=False)
    assert scores.n_instances == 2
    assert scores.cca_all.point == pytest.approx(0.5)
    assert scores.cca_all_lenient.point == pytest.approx(1.0)
    assert scores.failures[Failed.NOT_ENTAILED.value] == 3
    assert scores.cca_exception.point == pytest.approx(0.5)
    assert "UNTUNED" in scores.render()


def test_untuned_threshold_is_flagged_and_tuned_is_not(inst):
    answer = inst.gold_scoped_answer
    r = score_instance(inst, answer, reader_for(inst, answer), ConstNLI(0.95))
    assert "UNTUNED" in aggregate([r], threshold_tuned=False).render()
    assert "UNTUNED" not in aggregate([r], threshold=0.62, threshold_tuned=True).render()


def test_empty_aggregate_does_not_crash():
    s = aggregate([])
    assert s.n_instances == 0 and s.cca_exception is None and s.clarification_rate == 0.0


def test_score_corpus_refuses_a_missing_answer(inst):
    with pytest.raises(KeyError, match="not a wrong one"):
        score_corpus([inst], {}, lambda **k: ReaderOutput(NOT_DETERMINED, ""), ConstNLI(0.9))


def test_score_corpus_skips_instances_without_personas_and_scores_the_rest(inst):
    no_personas = next(i for i in generate(60, seed=2) if not i.personas)
    answer = inst.gold_scoped_answer
    scores, results = score_corpus([no_personas, inst], {inst.instance_id: answer},
                                   reader_for(inst, answer), ConstNLI(0.95))
    assert len(results) == 1 and scores.cca_all.point == 1.0


def test_instance_without_personas_is_refused(inst):
    bare = inst.model_copy(update={"personas": [], "persona_pairs": []})
    with pytest.raises(ValueError, match="no personas"):
        score_instance(bare, "x", lambda **k: ReaderOutput(NOT_DETERMINED, ""), ConstNLI(0.9))
