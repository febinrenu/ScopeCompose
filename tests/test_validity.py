"""The validity battery: the eight outputs, the V1 criteria, and the rule that an
unevaluated criterion is never silently a pass."""

from __future__ import annotations

import re

import pytest

from contract.gold import GoldInstance
from contract.mock import generate
from detection.nli import NLIScores
from metrics.case_accuracy import (
    NOT_DETERMINED,
    InstanceCCA,
    PersonaVerdict,
    ReaderOutput,
    aggregate,
)
from metrics.validity import (
    CCA_MODALITY_HEDGE_MAX,
    CCA_NULL_MAX,
    CCA_UNATTACHED_HEDGE_MAX,
    FILLER,
    NULL_ANSWER,
    OUTPUTS,
    V1_CRITERIA,
    V1_CRITERIA_SHA,
    BatteryResult,
    battery_outputs,
    legacy_pr,
    run_battery,
)
from metrics.values import is_hedged


def _instances() -> list[GoldInstance]:
    out = [i for i in generate(120, seed=0) if i.personas and i.gold_scoped_answer]
    assert len(out) >= 5
    return out


class ConstNLI:
    def score_batch(self, pairs):
        return [NLIScores(0.95, 0.05, 0.0) for _ in pairs]


def honest_reader(*, query: str, answer: str, persona: str) -> ReaderOutput:
    """A careful stand-in: quotes the sentence that addresses this person, and
    says nothing if no sentence does. Deterministic, so the test is too."""
    if answer == NULL_ANSWER:
        return ReaderOutput(NOT_DETERMINED, "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    wants_exception = "-tier" in persona
    pool = [s for s in sentences if s != FILLER.split(". ")[0] + "."]
    if wants_exception:
        cands = [s for s in pool if re.search(r"waived|fee-free|may be eligible|no fee", s, re.I)]
    else:
        cands = [s for s in pool if re.search(r"\d+%|incur|fee", s, re.I) and "waived" not in s]
    s = (cands or pool)[0]
    return ReaderOutput(s, s)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def test_the_battery_has_exactly_the_eight_planned_outputs():
    assert OUTPUTS == ("gold", "selection", "concat_gold", "concat_k", "padded_gold",
                       "unattached_hedge", "modality_hedge", "null")
    assert set(battery_outputs(_instances()[0])) == set(OUTPUTS)


def test_outputs_are_deterministic():
    inst = _instances()[0]
    assert battery_outputs(inst) == battery_outputs(inst)


def test_output_contents():
    inst = _instances()[0]
    o = battery_outputs(inst)
    assert o["gold"] == inst.gold_scoped_answer
    assert o["selection"] == inst.selection_answer
    assert o["padded_gold"] == f"{inst.gold_scoped_answer} {FILLER}"
    assert o["null"] == NULL_ANSWER
    for p in inst.passages:
        assert p.text in o["concat_k"]
    for b in inst.gold_branches:
        assert b.outcome in o["unattached_hedge"]
        assert b.outcome.lower() in o["modality_hedge"].lower()  # sentence-initial capital


def test_concat_gold_pastes_only_the_grounding_passages_once():
    inst = _instances()[0]
    grounding = {b.supporting_passage for b in inst.gold_branches}
    text = battery_outputs(inst)["concat_gold"]
    for p in inst.passages:
        assert (p.text in text) is (p.id in grounding)
    assert text.count(inst.passages[0].text) == 1


def test_both_hedged_outputs_are_actually_hedged():
    """If the 'hedge' outputs were not hedged the criteria would test nothing."""
    for inst in _instances():
        o = battery_outputs(inst)
        assert is_hedged(o["unattached_hedge"])
        exception_sentence = o["modality_hedge"].split(". ", 1)[1]
        assert is_hedged(exception_sentence), exception_sentence


def test_the_unattached_hedge_names_nobody():
    inst = _instances()[0]
    text = battery_outputs(inst)["unattached_hedge"].lower()
    for b in inst.exception_branches:
        assert b.applicability.descriptor.lower() not in text


def test_the_filler_states_no_outcome_and_no_hedge():
    assert not is_hedged(FILLER)
    assert not re.search(r"\d|fee|waive", FILLER, re.I)


def test_an_instance_without_reference_answers_is_refused_not_skipped():
    bare = _instances()[0].model_copy(update={"selection_answer": None})
    with pytest.raises(ValueError, match="needs both"):
        battery_outputs(bare)
    with pytest.raises(ValueError, match="at least one"):
        run_battery([], honest_reader, ConstNLI())


# --------------------------------------------------------------------------- #
# Criteria logic, on hand-built numbers
# --------------------------------------------------------------------------- #


def _cca(points: float, n: int = 20):
    k = round(points * n)
    results = []
    for i in range(n):
        ok = i < k
        v = PersonaVerdict("p", "b0", False, "g", "r", "e", ok, ok, None if ok else None)
        results.append(InstanceCCA(f"i{i}", (v,), False))
    return aggregate(results)


def _result(pr: dict[str, float], cca: dict[str, float]) -> BatteryResult:
    return BatteryResult(n_instances=20, cca={k: _cca(v) for k, v in cca.items()}, legacy_pr=pr)


GOOD_CCA = {"gold": 1.0, "selection": 0.0, "concat_gold": 0.5, "concat_k": 0.5, "padded_gold": 1.0,
            "unattached_hedge": 0.0, "modality_hedge": 0.0, "null": 0.0}
GAMEABLE_PR = {o: 0.5 for o in OUTPUTS} | {"gold": 0.7, "concat_gold": 0.85, "concat_k": 0.9}


def test_a_well_behaved_metric_passes_v1():
    r = _result(GAMEABLE_PR, GOOD_CCA)
    assert r.v1_passed
    assert all(c.passed in (True, None) for c in r.criteria())


def test_the_legacy_criterion_is_a_replication_of_gameability():
    """'Passing' means concatenation beats gold, i.e. the legacy metric IS gameable."""
    r = _result(GAMEABLE_PR, GOOD_CCA)
    legacy = {c.name: c.passed for c in r.criteria() if c.name.startswith("legacy_pr")}
    assert legacy == {"legacy_pr/concat_gold>=gold": True, "legacy_pr/concat_k>=gold": True}
    ungameable = _result(GAMEABLE_PR | {"concat_gold": 0.2, "concat_k": 0.3}, GOOD_CCA)
    assert not ungameable.v1_passed


@pytest.mark.parametrize("key,bound", [("unattached_hedge", CCA_UNATTACHED_HEDGE_MAX),
                                       ("modality_hedge", CCA_MODALITY_HEDGE_MAX),
                                       ("null", CCA_NULL_MAX)])
def test_hedge_and_null_ceilings_are_inclusive_and_enforced(key, bound):
    at = _result(GAMEABLE_PR, GOOD_CCA | {key: bound})
    over = _result(GAMEABLE_PR, GOOD_CCA | {key: bound + 0.05})
    assert at.v1_passed, f"{key} exactly at its ceiling must pass"
    assert not over.v1_passed
    failing = [c.name for c in over.criteria() if c.passed is False]
    assert failing == [f"cca/{key}<={bound}"]


def test_cca_gold_must_not_lose_to_concatenation():
    r = _result(GAMEABLE_PR, GOOD_CCA | {"concat_k": 1.0, "gold": 0.5})
    assert not r.v1_passed
    assert any(c.name == "cca/gold>=concat_k" and c.passed is False for c in r.criteria())


def test_an_unevaluated_criterion_is_reported_and_never_counted_as_a_pass():
    r = _result(GAMEABLE_PR, GOOD_CCA)
    sp = next(c for c in r.criteria() if c.name.startswith("scoped_precision"))
    assert sp.passed is None and "not implemented" in sp.detail
    assert "n/a" in r.render() and "unevaluated criteria are not counted" in r.render()


def test_criteria_are_hashed_and_the_hash_covers_thresholds():
    assert len(V1_CRITERIA_SHA) == 64
    assert any(str(CCA_UNATTACHED_HEDGE_MAX) in k for k in V1_CRITERIA)
    assert any(str(CCA_MODALITY_HEDGE_MAX) in k for k in V1_CRITERIA)


# --------------------------------------------------------------------------- #
# End to end with a plausible reader
# --------------------------------------------------------------------------- #


def test_end_to_end_a_careful_reader_passes_v1_on_mock_instances():
    res = run_battery(_instances(), honest_reader, ConstNLI())
    cca = {k: v.cca_all.point for k, v in res.cca.items()}
    assert cca["gold"] == 1.0
    assert cca["selection"] == 0.0, "one unconditioned outcome cannot be right for everyone (Proposition 1)"
    assert cca["null"] == 0.0
    assert cca["unattached_hedge"] <= CCA_UNATTACHED_HEDGE_MAX
    assert cca["modality_hedge"] <= CCA_MODALITY_HEDGE_MAX
    assert cca["padded_gold"] == cca["gold"], "length alone must not change the score"
    assert res.v1_passed, res.render()


def test_end_to_end_the_legacy_metric_cannot_tell_concatenation_from_gold():
    res = run_battery(_instances(), honest_reader, ConstNLI())
    assert res.legacy_pr["concat_gold"] >= res.legacy_pr["gold"]


def test_legacy_pr_is_pooled_over_branches():
    insts = _instances()[:3]
    answers = {i.instance_id: i.gold_scoped_answer for i in insts}
    pr = legacy_pr(insts, answers)
    assert 0.0 <= pr <= 1.0


def test_render_lists_every_output_and_flags_an_untuned_threshold():
    text = run_battery(_instances(), honest_reader, ConstNLI()).render()
    for o in OUTPUTS:
        assert o in text
    assert "UNTUNED" in text and "criteria sha256" in text
