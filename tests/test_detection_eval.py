"""E2 detection evaluation: metrics with known answers, dev-fixed operating points, the gate
ablation, and H2's rule that the baseline is chosen on dev."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from contract.gold import GoldInstance
from contract.mock import generate
from contract.models import ScopeRelation
from detection.frames import FrameExtractor
from detection.nli import HeuristicNLI
from detection.stage1 import Stage1Filter
from experiments.detection_eval import (
    average_precision,
    choose_threshold,
    evaluate,
    f1_at,
    h2,
    is_positive,
    legacy_detectors,
    llm_pair_detector,
    nli_detector,
    prf,
    qnfa_detector,
    render,
)


def insts():
    pos = [i for i in generate(120, seed=0) if i.gold_scope_relation is ScopeRelation.REFINEMENT][:6]
    neg = [i for i in generate(120, seed=0) if i.gold_scope_relation in (ScopeRelation.DISJOINT, ScopeRelation.REDUNDANT)][:6]
    return pos, neg


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_prf_known_values():
    s, y = [0.9, 0.8, 0.4, 0.3, 0.2], [True, False, True, False, False]
    r = prf(s, y, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


def test_prf_handles_no_predictions_and_no_positives():
    assert prf([0.1, 0.2], [True, False], 0.9)["f1"] == 0.0
    assert prf([0.9], [False], 0.5)["precision"] == 0.0 and prf([], [], 0.5)["f1"] == 0.0


def test_average_precision_known_values():
    assert average_precision([0.9, 0.8, 0.7], [True, True, False]) == 1.0
    assert average_precision([0.9, 0.8, 0.7], [False, True, True]) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert average_precision([0.5], [False]) == 0.0


def test_choose_threshold_maximises_dev_f1_and_breaks_ties_conservatively():
    scores, labels = [0.9, 0.8, 0.3, 0.2], [True, True, False, False]
    t = choose_threshold(scores, labels)
    assert prf(scores, labels, t)["f1"] == 1.0
    assert t == 0.8, "ties go to the higher threshold"
    assert choose_threshold([0.1, 0.2], [False, False]) == 0.5


def test_f1_at_is_a_resamplable_statistic():
    f = f1_at(0.5)
    assert f([(0.9, True), (0.1, False)]) == 1.0 and f([(0.1, True)]) == 0.0


# --------------------------------------------------------------------------- #
# The gate ablation
# --------------------------------------------------------------------------- #


def test_removing_only_the_topicality_gate_raises_the_score_of_a_rule_and_carve_out_with_disjoint_wording():
    """The mechanism the paper claims for the pilot's 3% recall: D0 and D0-nogate are the same score
    except for the multiplier that shrinks a pair in proportion to how little vocabulary it shares."""
    from contract.gold import Applicability, Branch, GoldInstance as GI
    from contract.models import ConflictType, Construction, Domain, Passage, ScopeRelation as SR, Separation

    p0 = Passage(id="p0", text="Overseas teller machine use must incur a flat charge of 2.50.")
    p1 = Passage(id="p1", text="Reward holders enjoy fee-free worldwide access, unless the monthly cap is reached.")
    inst = GI(instance_id="gate", query="q", domain=Domain.FINANCIAL_TERMS, construction=Construction.NATURAL,
              separation=Separation.CROSS_DOCUMENT, passages=[p0, p1], gold_conflict_type=ConflictType.CONDITIONAL,
              gold_scope_relation=SR.REFINEMENT,
              gold_branches=[Branch(branch_id="b0", condition=None, outcome="a 2.50 charge applies",
                                    applicability=Applicability(descriptor="all", is_default=True),
                                    supporting_passage="p0"),
                             Branch(branch_id="b1", condition="reward holder", outcome="no charge applies",
                                    applicability=Applicability(descriptor="reward holders"),
                                    supporting_passage="p1")])
    d0, nogate = legacy_detectors(Stage1Filter(nli=HeuristicNLI(), auto_load_head=False))
    assert nogate(inst) > d0(inst), (d0(inst), nogate(inst))
    assert 0.0 <= d0(inst) <= 1.0 and 0.0 <= nogate(inst) <= 1.0


def test_the_two_legacy_scores_are_equal_when_the_passages_share_all_their_words():
    from contract.models import Passage

    same = "Transfers incur a 3% fee unless the holder is a student."
    inst = [i for i in generate(60, seed=0)][0].model_copy(update={"passages": [
        Passage(id="p0", text=same), Passage(id="p1", text=same)]})
    d0, nogate = legacy_detectors(Stage1Filter(nli=HeuristicNLI(), auto_load_head=False))
    assert d0(inst) == pytest.approx(nogate(inst))


def test_the_nli_baseline_scores_every_instance_in_unit_range():
    pos, neg = insts()
    d1 = nli_detector(HeuristicNLI())
    assert all(0.0 <= d1(i) <= 1.0 for i in pos + neg)


# --------------------------------------------------------------------------- #
# D3 and D5 on a scripted backend
# --------------------------------------------------------------------------- #


def client_with(tmp_path, scripted):
    backend = MockBackend(scripted={k: json.dumps(v) if not isinstance(v, str) else v for k, v in scripted.items()})
    return LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                     backend_override=backend), backend


def test_d3_turns_a_verbal_confidence_into_a_conditional_probability(tmp_path):
    inst = insts()[0][0]
    yes, _ = client_with(tmp_path, {"Question:": {"conditional": True, "confidence": 0.9}})
    no, _ = client_with(tmp_path, {"Question:": {"conditional": False, "confidence": 0.9}})
    assert llm_pair_detector(yes, Tier.BULK)(inst) == pytest.approx(0.9)
    assert llm_pair_detector(no, Tier.BULK)(inst) == pytest.approx(0.1)


def test_d3_treats_unparseable_output_as_no_detection(tmp_path):
    inst = insts()[0][0]
    c, _ = client_with(tmp_path, {"Question:": "not json"})
    assert llm_pair_detector(c, Tier.BULK)(inst) == 0.0


def test_d5_scores_the_strongest_conditional_pair_and_zero_when_there_is_none(tmp_path):
    from contract.models import Passage

    p0 = Passage(id="p0", text="Transfers incur a 3% fee.")
    p1 = Passage(id="p1", text="The transfer fee is waived for premium-tier cardholders.")
    inst = insts()[0][0].model_copy(update={"passages": [p0, p1]})
    sig = {"event": "transfer", "decision_kind": "flat fee"}

    def frames(*fs):
        return {"Question:": {"frames": list(fs)}}

    rule = {"passage_id": "p0", "span": "Transfers incur a 3% fee.", "decision": sig, "outcome": "a 3% fee applies",
            "applicability": {"descriptor": "everyone", "is_default": True, "attributes": []}}
    exc = {"passage_id": "p1", "span": "The transfer fee is waived for premium-tier cardholders.", "decision": sig,
           "outcome": "no fee applies",
           "applicability": {"descriptor": "premium", "is_default": False,
                             "attributes": [{"name": "tier", "kind": "categorical", "values": ["premium"]}]}}
    c, _ = client_with(tmp_path, frames(rule, exc))
    det = qnfa_detector(lambda i: FrameExtractor("banking", tier=Tier.BULK, client=c))
    assert det(inst) == pytest.approx(0.95 * 1.0 * 0.95)
    c2, _ = client_with(tmp_path, frames(rule))
    assert qnfa_detector(lambda i: FrameExtractor("banking", tier=Tier.BULK, client=c2))(inst) == 0.0


# --------------------------------------------------------------------------- #
# evaluate and H2
# --------------------------------------------------------------------------- #


def oracle_detector(noise=0.0):
    return lambda inst: (0.9 if is_positive(inst) else 0.1)


def inverted_detector():
    return lambda inst: (0.1 if is_positive(inst) else 0.9)


def constant_detector(v=0.5):
    return lambda inst: v


def test_the_operating_point_is_chosen_on_dev_and_applied_unchanged_to_test():
    pos, neg = insts()
    dev, test = pos[:3] + neg[:3], pos[3:] + neg[3:]
    res = evaluate({"oracle": oracle_detector(), "bad": inverted_detector()}, dev, test)
    assert res["oracle"].test["f1"] == 1.0 and res["oracle"].test["threshold"] == res["oracle"].threshold
    assert res["bad"].test["f1"] < 0.7


def test_render_lists_every_detector():
    pos, neg = insts()
    text = render(evaluate({"D0": oracle_detector(), "D5": constant_detector()}, pos[:3] + neg[:3], pos[3:] + neg[3:]))
    assert "D0" in text and "D5" in text and "dev F1" in text


def test_h2_chooses_the_baseline_on_dev_not_on_test():
    """B_dev_best looks best on dev; B_test_best would look best on test. H2 must compare against the former."""
    pos, neg = insts()
    dev, test = pos[:3] + neg[:3], pos[3:] + neg[3:]
    dev_ids = {i.instance_id for i in dev}

    def flip_on_test(inst):  # perfect on dev, inverted on test
        good = 0.9 if is_positive(inst) else 0.1
        return good if inst.instance_id in dev_ids else 1.0 - good

    results = evaluate({"D5": oracle_detector(), "D0": flip_on_test, "D1": inverted_detector()}, dev, test)
    r = h2(results, test, n_resamples=400)
    assert r.best_baseline == "D0", "D0 has the best DEV F1, so it is the pre-registered baseline"
    assert r.diff > 0.5, "and D5 beats it on test"


def test_h2_needs_a_margin_over_the_minimum_effect_to_be_cleared():
    pos, neg = insts()
    dev, test = pos[:3] + neg[:3], pos[3:] + neg[3:]
    strong = h2(evaluate({"D5": oracle_detector(), "D1": inverted_detector()}, dev, test), test, n_resamples=600)
    same = h2(evaluate({"D5": oracle_detector(), "D1": oracle_detector()}, dev, test), test, n_resamples=600)
    assert strong.diff > 0.15 and strong.best_baseline == "D1"
    assert not same.clears_minimum_effect and same.diff == 0
    text = same.render()
    assert text.startswith("H2: D5 F1 minus D1 F1 = +0.000") and "not cleared" in text


def test_h2_requires_d5_and_a_baseline():
    pos, neg = insts()
    res = evaluate({"D5": oracle_detector()}, pos[:3] + neg[:3], pos[3:] + neg[3:])
    with pytest.raises(KeyError):
        h2(res, pos[3:] + neg[3:])
