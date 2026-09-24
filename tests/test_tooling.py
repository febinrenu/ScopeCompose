"""Tests for the tooling layer: threshold tuning, head training, the baseline
and retrieval scorers, the WP1 probe set, and the budget estimator.

These modules were written after the core pipeline and initially shipped with
no tests at all, which was the largest gap in the repo. They are not
incidental scripts: the threshold tuner decides the API cost of detection, the
baseline scorer produces the suppression number that goes on a slide, and the
probe set is the data Member B's whole feasibility probe runs on. A silent
arithmetic error in any of them produces a plausible number rather than a
crash, which is the worst failure mode available.

Everything here runs offline.
"""

from __future__ import annotations

import json

import pytest

import settings
from contract.gold import Applicability, Branch, GoldInstance
from contract.models import (
    ConflictType,
    Construction,
    Domain,
    Passage,
    QueryRecord,
    ScopeRelation,
    SourceType,
)


# --------------------------------------------------------------------------- #
# detection/threshold.py
# --------------------------------------------------------------------------- #

from detection.threshold import (  # noqa: E402
    BandResult,
    evaluate_band,
    is_saturated,
    pareto_front,
    recommend,
    sweep,
)


def _band(low, high, n, esc, correct, resolved, s2=1.0) -> BandResult:
    return BandResult(low=low, high=high, n_pairs=n, n_escalated=esc,
                      stage1_correct=correct, stage1_resolved=resolved,
                      stage2_accuracy=s2)


def test_evaluate_band_counts_escalation_and_correctness():
    # p >= 0.5 predicts conflict. Two inside the band, two outside and correct.
    scored = [(0.05, False), (0.95, True), (0.45, True), (0.55, False)]
    r = evaluate_band(scored, 0.4, 0.6)
    assert r.n_escalated == 2
    assert r.stage1_resolved == 2
    assert r.stage1_correct == 2
    assert r.escalation_rate == 0.5


def test_zero_width_band_escalates_nothing():
    scored = [(0.2, False), (0.8, True)]
    r = evaluate_band(scored, 0.5, 0.5)
    assert r.n_escalated == 0
    assert r.escalation_rate == 0.0


def test_full_band_escalates_everything():
    scored = [(0.2, False), (0.8, True)]
    assert evaluate_band(scored, 0.0, 1.0).escalation_rate == 1.0


def test_overall_accuracy_credits_stage2_at_its_stated_accuracy():
    """Stage 2 is not assumed perfect. At 0.5 an escalated pair contributes
    half a correct answer, which is what keeps the curve honest."""
    r = _band(0.4, 0.6, n=10, esc=4, correct=6, resolved=6, s2=0.5)
    assert r.overall_accuracy == pytest.approx((6 + 4 * 0.5) / 10)

    oracle = _band(0.4, 0.6, n=10, esc=4, correct=6, resolved=6, s2=1.0)
    assert oracle.overall_accuracy == pytest.approx(1.0)


def test_stage1_accuracy_ignores_escalated_pairs():
    """A wider band flatters stage-1 accuracy by handing away the hard cases,
    so it must be computed over kept pairs only."""
    r = _band(0.4, 0.6, n=10, esc=8, correct=2, resolved=2)
    assert r.stage1_accuracy == 1.0
    assert r.overall_accuracy == 1.0


def test_stage1_accuracy_of_nothing_is_one_not_zero():
    """Dividing by zero kept-pairs must not report 0% accuracy, which would
    make an all-escalate band look terrible instead of merely expensive."""
    assert _band(0.0, 1.0, n=5, esc=5, correct=0, resolved=0).stage1_accuracy == 1.0


def test_sweep_escalation_is_monotonic_in_band_width():
    records = _mock_records(20)
    from detection.stage1 import Stage1Filter

    results = sweep(records, Stage1Filter(heuristic_nli=True, auto_load_head=False),
                    steps=6)
    rates = [r.escalation_rate for r in results]
    assert rates == sorted(rates), "a wider band cannot escalate fewer pairs"


def test_sweep_of_unlabelled_records_is_empty():
    rec = QueryRecord(query_id="q", query="q", domain=Domain.FINANCIAL_TERMS,
                      construction=Construction.NATURAL,
                      passages=_passages(2), conflict_pairs=[])
    from detection.stage1 import Stage1Filter

    assert sweep([rec], Stage1Filter(heuristic_nli=True, auto_load_head=False)) == []


def test_pareto_front_drops_dominated_bands():
    results = [
        _band(0.5, 0.5, 10, 0, 5, 10),    # 0% escalation, 0.5 accuracy
        _band(0.4, 0.6, 10, 2, 5, 8),     # 20% escalation, 0.7 -- better
        _band(0.3, 0.7, 10, 4, 3, 6),     # 40% escalation, 0.7 -- dominated
        _band(0.2, 0.8, 10, 6, 3, 4),     # 60% escalation, 0.9 -- better
    ]
    front = pareto_front(results)
    rates = [r.escalation_rate for r in front]
    assert 0.4 not in rates, "an equal-accuracy, more expensive band must be dominated"
    assert rates == sorted(rates)


def test_recommend_respects_an_escalation_budget():
    results = [
        _band(0.5, 0.5, 10, 0, 5, 10),
        _band(0.4, 0.6, 10, 2, 6, 8),
        _band(0.0, 1.0, 10, 10, 0, 0),    # perfect, but escalates everything
    ]
    pick, why = recommend(results, max_escalation=0.25)
    assert pick.escalation_rate <= 0.25
    assert "budget" in why


def test_recommend_finds_the_cheapest_band_meeting_an_accuracy_floor():
    results = [
        _band(0.5, 0.5, 10, 0, 5, 10),    # 0.5
        _band(0.4, 0.6, 10, 2, 7, 8),     # 0.9
        _band(0.2, 0.8, 10, 6, 4, 4),     # 1.0
    ]
    pick, why = recommend(results, min_accuracy=0.9)
    assert pick.escalation_rate == 0.2, "should take the cheapest band that clears the floor"
    assert "cheapest" in why


def test_recommend_says_so_when_no_band_reaches_the_floor():
    results = [_band(0.5, 0.5, 10, 0, 5, 10)]
    pick, why = recommend(results, min_accuracy=0.99)
    assert pick is not None
    assert "no band reaches" in why


def test_recommend_without_a_constraint_returns_the_knee():
    results = [
        _band(0.5, 0.5, 10, 0, 5, 10),
        _band(0.4, 0.6, 10, 2, 7, 8),
        _band(0.0, 1.0, 10, 10, 0, 0),
    ]
    pick, why = recommend(results)
    assert pick is not None
    assert "knee" in why


def test_saturated_curve_is_detected():
    """A curve where every band scores the same cannot tune anything, and the
    output otherwise reads as a spectacular result rather than a useless one."""
    flat = [_band(0.5, 0.5, 10, 0, 10, 10), _band(0.4, 0.6, 10, 2, 8, 8)]
    assert is_saturated(flat)


def test_unsaturated_curve_is_not_flagged():
    varied = [_band(0.5, 0.5, 10, 0, 5, 10), _band(0.4, 0.6, 10, 2, 7, 8)]
    assert not is_saturated(varied)


def test_write_band_persists_and_marks_tuned(tmp_path, monkeypatch):
    import yaml

    from detection.threshold import write_band

    cfg = tmp_path / "hardware.yaml"
    cfg.write_text(yaml.safe_dump({
        "active_profile": "p",
        "profiles": {"p": {
            "device": "cpu", "precision": "fp32",
            "embeddings": {"model": "m", "batch_size": 1, "max_seq_length": 8},
            "cross_encoder": {"model": "m", "batch_size": 1, "max_seq_length": 8},
            "finetune": {"model": "m", "batch_size": 1, "max_seq_length": 8},
        }},
        "retrieval": {"top_k": 5, "rrf_k": 60},
        "detection": {"low_threshold": 0.35, "high_threshold": 0.75, "tuned": False},
    }), encoding="utf-8")
    monkeypatch.setattr(settings, "HARDWARE_YAML", cfg)
    settings.reload()
    try:
        write_band(_band(0.2, 0.8, 10, 2, 8, 8), dataset="dev.jsonl")
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert raw["detection"]["low_threshold"] == 0.2
        assert raw["detection"]["high_threshold"] == 0.8
        assert raw["detection"]["tuned"] is True
        assert raw["detection"]["tuned_on"] == "dev.jsonl"
    finally:
        settings.reload()


def test_shipped_thresholds_are_marked_untuned():
    """Guards against quoting an escalation rate from two invented constants.
    Flip this in config only via `--write` on real data."""
    settings.reload()
    assert settings.hardware().thresholds_tuned is False, (
        "config/hardware.yaml claims the band is tuned. If that is now true on "
        "annotated data, update this test to assert it."
    )


# --------------------------------------------------------------------------- #
# detection/train.py
# --------------------------------------------------------------------------- #

from detection.train import _split, train  # noqa: E402


def _passages(n: int, texts: list[str] | None = None) -> list[Passage]:
    texts = texts or [f"passage text number {i} about fees" for i in range(n)]
    return [
        Passage(id=f"p{i}", text=texts[i], source_type=SourceType.OFFICIAL_POLICY,
                date="2024-01", document_id=f"doc{i}")
        for i in range(n)
    ]


def _mock_records(n: int, seed: int = 0) -> list[QueryRecord]:
    from contract.mock import generate

    return [i.to_query_record() for i in generate(n, seed=seed)]


def test_split_is_by_record_not_by_pair():
    """Two pairs from one record share passages, so splitting by pair leaks the
    dev set into training and inflates held-out accuracy."""
    records = _mock_records(10)
    train_recs, dev_recs = _split(records, 0.3)
    assert len(train_recs) + len(dev_recs) == len(records)
    ids_train = {r.query_id for r in train_recs}
    ids_dev = {r.query_id for r in dev_recs}
    assert not (ids_train & ids_dev)


def test_split_always_leaves_something_on_both_sides():
    records = _mock_records(2)
    a, b = _split(records, 0.5)
    assert a and b


def test_train_beats_the_rule_baseline_and_reports_both():
    head, report = train(_mock_records(60, seed=5), heuristic_nli=True)
    assert head.classes_
    assert report.n_train > 0
    assert report.n_dev > 0
    assert 0.0 <= report.baseline_accuracy <= 1.0
    assert 0.0 <= report.trained_accuracy <= 1.0
    assert report.improvement == pytest.approx(
        report.trained_accuracy - report.baseline_accuracy
    )


def test_train_refuses_a_single_class_corpus():
    """A conflict detector needs both conflicting and non-conflicting pairs.
    Failing loudly here is what points at a corpus missing distractors."""
    rec = QueryRecord(
        query_id="q", query="q", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=_passages(2),
        conflict_pairs=[{"doc_i": "p0", "doc_j": "p1", "is_conflict": True,
                         "type": "factual", "confidence": 0.9}],
    )
    with pytest.raises(ValueError, match="only the label|no gold-labelled"):
        train([rec, rec.model_copy(update={"query_id": "q2"})], heuristic_nli=True)


def test_trained_head_round_trips_with_a_matching_signature(tmp_path):
    from detection.head import PairHead

    head, _ = train(_mock_records(40, seed=1), heuristic_nli=True)
    path = tmp_path / "head.pkl"
    head.save(path)
    assert PairHead.load(path).classes_ == head.classes_


def test_stale_head_is_rejected_not_silently_used(tmp_path):
    """The failure this guards against does not crash: a head fitted on
    different feature semantics scores nonsense and nothing surfaces it."""
    import pickle

    from detection.head import PairHead, StaleHeadError

    head, _ = train(_mock_records(40, seed=1), heuristic_nli=True)
    path = tmp_path / "head.pkl"
    head.save(path)

    with path.open("rb") as fh:
        blob = pickle.load(fh)
    blob["feature_signature"] = "deadbeefdeadbeef"
    with path.open("wb") as fh:
        pickle.dump(blob, fh)

    with pytest.raises(StaleHeadError, match="Retrain it"):
        PairHead.load(path)
    # Explicitly opting out still works, for forensics on an old checkpoint.
    assert PairHead.load(path, strict=False).classes_ == head.classes_


def test_feature_signature_changes_when_a_cue_vocabulary_changes(monkeypatch):
    from detection import features
    from detection.head import feature_signature

    before = feature_signature()
    monkeypatch.setattr(features, "STRONG_EXCEPTION_CUES",
                        features.STRONG_EXCEPTION_CUES + ("newly added cue",))
    assert feature_signature() != before, (
        "changing a cue list changes what a feature VALUE means, so it must "
        "invalidate saved heads even though the vector shape is unchanged"
    )


# --------------------------------------------------------------------------- #
# experiments/run_baselines.py
# --------------------------------------------------------------------------- #

from experiments.run_baselines import BaselineScore, run, score_instance  # noqa: E402


def _gold_conditional(instance_id: str = "g1") -> GoldInstance:
    return GoldInstance(
        instance_id=instance_id,
        query="do I pay a fee?",
        domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL,
        passages=_passages(2, ["International transactions incur a 3% fee.",
                               "The fee is waived for premium-tier cardholders."]),
        gold_conflict_type=ConflictType.CONDITIONAL,
        gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_branches=[
            Branch(branch_id="b0", condition=None, outcome="3% fee",
                   applicability=Applicability(descriptor="all", is_default=True),
                   supporting_passage="p0"),
            Branch(branch_id="b1", condition="premium tier", outcome="no fee",
                   applicability=Applicability(descriptor="premium holders"),
                   supporting_passage="p1"),
        ],
    )


class _Output:
    def __init__(self, kept, dropped):
        self.kept = kept
        self.dropped = dropped

    @property
    def suppressed_ids(self):
        return [p.id for p in self.dropped]


def test_dropping_the_exception_passage_loses_its_branch():
    inst = _gold_conditional()
    score = BaselineScore(name="t")
    lost = score_instance(_Output(kept=[inst.passages[0]], dropped=[inst.passages[1]]),
                          inst, score)
    assert lost == ["b1"]
    assert score.n_branches_lost == 1
    assert score.n_gold_branches == 2
    assert score.branch_loss_rate == 0.5
    assert score.conditional_branch_loss_rate == 0.5


def test_keeping_everything_loses_nothing():
    inst = _gold_conditional()
    score = BaselineScore(name="t")
    assert score_instance(_Output(kept=list(inst.passages), dropped=[]), inst, score) == []
    assert score.branch_loss_rate == 0.0


def test_branch_loss_of_an_empty_run_is_zero_not_a_crash():
    assert BaselineScore(name="t").branch_loss_rate == 0.0
    assert BaselineScore(name="t").conditional_branch_loss_rate == 0.0


def test_conditional_loss_is_tracked_separately_from_overall():
    """Distractors dominate a real corpus, so overall branch loss can look fine
    while every exception is being dropped."""
    conditional = _gold_conditional("c1")
    factual = GoldInstance(
        instance_id="f1", query="q", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=_passages(2),
        gold_conflict_type=ConflictType.FACTUAL, is_distractor=True,
        gold_branches=[],
    )
    score = BaselineScore(name="t")
    score_instance(_Output(kept=[conditional.passages[0]],
                           dropped=[conditional.passages[1]]), conditional, score)
    score_instance(_Output(kept=list(factual.passages), dropped=[]), factual, score)

    assert score.n_conditional == 1
    assert score.conditional_branch_loss_rate == 0.5
    assert score.n_instances == 2


def test_standard_rag_never_drops_a_passage():
    scores = run([_gold_conditional()], heuristic_nli=True)
    by_name = {s.name: s for s in scores}
    assert by_name["standard_rag"].n_passages_dropped == 0
    assert by_name["standard_rag"].branch_loss_rate == 0.0


def test_rerank_drops_everything_after_the_first():
    scores = run([_gold_conditional()], heuristic_nli=True)
    by_name = {s.name: s for s in scores}
    assert by_name["rerank_top1"].n_passages_dropped == 1
    assert by_name["rerank_top1"].conditional_branch_loss_rate == 0.5


# --------------------------------------------------------------------------- #
# experiments/run_retrieval_eval.py
# --------------------------------------------------------------------------- #

from experiments.run_retrieval_eval import (  # noqa: E402
    build_corpus,
    duplicate_passage_rate,
)


def test_corpus_namespaces_ids_by_instance():
    """'p0' means different things in different records; a flat corpus collides
    them and silently scores the wrong passage as a hit."""
    corpus, owner = build_corpus([_gold_conditional("a"), _gold_conditional("b")])
    assert len(corpus) == 4
    assert len(set(corpus.ids)) == 4
    assert owner["a::p0"] == "a"


def test_duplicate_rate_is_one_when_every_passage_repeats():
    """Two instances with identical text: pooled retrieval cannot tell which
    copy a query meant, so recall becomes a fixture floor rather than a
    measurement."""
    assert duplicate_passage_rate([_gold_conditional("a"), _gold_conditional("b")]) == 1.0


def test_duplicate_rate_is_zero_for_distinct_passages():
    a = _gold_conditional("a")
    b = _gold_conditional("b").model_copy(update={
        "passages": _passages(2, ["Totally different first text about visas.",
                                  "Totally different second text about study."])
    })
    assert duplicate_passage_rate([a, b]) == 0.0


# --------------------------------------------------------------------------- #
# benchmark/wp1_probe_set.py
# --------------------------------------------------------------------------- #

from benchmark.wp1_probe_set import (  # noqa: E402
    ALL_CASES,
    DISTRACTORS,
    EXPLICIT,
    IMPLICIT,
    build_all,
    check_implicitness,
)


def test_every_probe_case_validates_against_the_schema():
    assert len(build_all()) == len(ALL_CASES)


def test_probe_case_ids_are_unique():
    ids = [c.id for c in ALL_CASES]
    assert len(set(ids)) == len(ids)


def test_implicit_cases_contain_no_strong_exception_cue():
    """The whole value of the set is that the implicit cases are genuinely
    implicit. A strong cue means the method never had to infer anything, and
    Member B's implicit-recovery number would be flattered by it."""
    assert check_implicitness() == []


def test_explicit_control_arm_is_non_empty():
    """Explicit and implicit recovery are reported separately and never
    averaged, so the probe needs both arms."""
    assert len(EXPLICIT) >= 5
    assert len(IMPLICIT) >= 20


def test_probe_set_covers_all_four_scope_relations():
    """A fixture missing a relation cannot catch confusions involving it."""
    rels = {c.relation for c in ALL_CASES if c.relation}
    assert rels == set(ScopeRelation)


def test_probe_set_covers_every_conflict_type():
    assert {c.conflict_type for c in ALL_CASES} == set(ConflictType)


def test_probe_set_includes_a_redundant_distractor():
    """Nested scope with an agreeing outcome. Proposing a condition there
    fabricates a branch, which inflates Preservation Rate while corrupting the
    answer -- the single most important distractor in the set."""
    assert any(c.relation is ScopeRelation.REDUNDANT for c in DISTRACTORS)


def test_probe_instances_are_marked_unverified():
    """They are model-proposed. If this ever passes with a real annotator name,
    the set has been verified and the assertion should be updated deliberately."""
    for inst in build_all():
        assert inst.annotation.annotator_a == "model-proposed"
        assert inst.annotation.annotator_b is None


def test_probe_instances_are_never_counted_as_tier_1():
    """Hand-built cases must not inflate the naturally-occurring corpus count."""
    for inst in build_all():
        assert inst.construction is Construction.SPLIT


def test_every_exception_branch_is_grounded_in_a_passage():
    for inst in build_all():
        for branch in inst.exception_branches:
            assert branch.supporting_passage is not None


def test_check_implicitness_catches_a_mislabelled_case(monkeypatch):
    from benchmark import wp1_probe_set as mod

    bad = IMPLICIT[0]
    doctored = bad.__class__(**{**bad.__dict__,
                                "other": mod.Src("This does not apply to students.",
                                                 SourceType.FAQ, "2024-01", "d")})
    monkeypatch.setattr(mod, "IMPLICIT", [doctored])
    problems = mod.check_implicitness()
    assert problems and "labelled IMPLICIT" in problems[0]


# --------------------------------------------------------------------------- #
# scripts/budget_estimate.py
# --------------------------------------------------------------------------- #


def _load_script(name: str):
    """Import a file in scripts/ as a module.

    It must be registered in ``sys.modules`` BEFORE exec: dataclass field
    resolution looks the defining module up by name, and a module that is not
    registered resolves to None and raises deep inside dataclasses.
    """
    import importlib.util
    import sys
    from pathlib import Path

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(settings.REPO_ROOT) / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _judge_volume(**kw):
    return _load_script("budget_estimate").judge_volume(**kw)


def test_batching_the_judge_roughly_halves_its_tokens():
    """The judge is ~65% of the campaign. Batching per instance rather than per
    branch sends the passages and rubric once instead of 2.5 times."""
    per_branch = _judge_volume(instances=300, configs=12, batched=False,
                               headline_configs=4, ablation_instances=None)
    batched = _judge_volume(instances=300, configs=12, batched=True,
                            headline_configs=4, ablation_instances=None)
    assert batched[1] < per_branch[1]
    assert batched[1] / per_branch[1] < 0.6


def test_subsampling_ablations_cuts_volume():
    full = _judge_volume(instances=300, configs=12, batched=True,
                         headline_configs=4, ablation_instances=None)
    sub = _judge_volume(instances=300, configs=12, batched=True,
                        headline_configs=4, ablation_instances=100)
    assert sub[0] < full[0]


def test_subsample_math_is_exact():
    # 4 headline configs on all 300 + 8 ablations on 100 = 1200 + 800.
    calls, _ = _judge_volume(instances=300, configs=12, batched=True,
                             headline_configs=4, ablation_instances=100)
    assert calls == 300 * 4 + 100 * 8


def test_headline_configs_cannot_exceed_total_configs():
    calls, _ = _judge_volume(instances=100, configs=2, batched=True,
                             headline_configs=10, ablation_instances=50)
    assert calls == 100 * 2


# --------------------------------------------------------------------------- #
# scripts/smoke_test.py and eval_descriptors.py import cleanly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", ["smoke_test", "eval_descriptors", "discover_models"])
def test_scripts_import_without_touching_the_network(script):
    """Import-time side effects are how a script that only runs under `-m`
    breaks silently. None of these may call out on import."""
    assert hasattr(_load_script(script), "main")


# --------------------------------------------------------------------------- #
# experiments/run_ablations.py
# --------------------------------------------------------------------------- #

from experiments.run_ablations import (  # noqa: E402
    ConditionalTypeAblation,
    _route_instances,
    explicitness_breakdown,
    tier_breakdown,
)


def test_removing_the_conditional_class_costs_exception_branches():
    """The headline ablation. A four-class detector emits `factual` for a
    general-rule/exception pair, which routes to selection -- and selection
    keeps one passage, so the exception branch is gone before any resolver
    sees it."""
    instances = [_gold_conditional(f"c{i}") for i in range(5)]

    baseline = _route_instances(instances, relabel_conditional_as_factual=False)
    ablated = _route_instances(instances, relabel_conditional_as_factual=True)

    assert baseline.exception_preservation_rate == 1.0
    assert ablated.exception_preservation_rate == 0.0
    assert ConditionalTypeAblation(baseline, ablated).preservation_drop == 1.0


def test_ablation_routes_relabelled_pairs_to_prior_work_not_compose():
    ablated = _route_instances([_gold_conditional()],
                               relabel_conditional_as_factual=True)
    assert ablated.composed == 0
    assert ablated.prior_work == 1


def test_ablation_leaves_non_conditional_pairs_alone():
    """Only conditional pairs are relabelled; a factual distractor routes the
    same way in both arms, so it cannot inflate the drop."""
    factual = GoldInstance(
        instance_id="f1", query="q", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=_passages(2),
        gold_conflict_type=ConflictType.FACTUAL, is_distractor=True,
        gold_branches=[],
    )
    base = _route_instances([factual], relabel_conditional_as_factual=False)
    abl = _route_instances([factual], relabel_conditional_as_factual=True)
    assert base.prior_work == abl.prior_work == 1
    assert base.exception_branches == abl.exception_branches == 0


def test_preservation_drop_is_zero_with_no_exception_branches():
    """A corpus with nothing to preserve cannot show a drop, and must not
    report one as though the ablation had been informative."""
    factual = GoldInstance(
        instance_id="f1", query="q", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=_passages(2),
        gold_conflict_type=ConflictType.FACTUAL, is_distractor=True,
        gold_branches=[],
    )
    abl = ConditionalTypeAblation(
        baseline=_route_instances([factual], relabel_conditional_as_factual=False),
        ablated=_route_instances([factual], relabel_conditional_as_factual=True),
    )
    assert abl.preservation_drop == 0.0


def test_tier_breakdown_separates_the_two_construction_tiers():
    natural = _gold_conditional("n1")
    split = _gold_conditional("s1").model_copy(
        update={"construction": Construction.SPLIT}
    )
    rows = tier_breakdown([natural, split]).rows
    assert set(rows) == {"natural", "split"}
    assert rows["natural"]["instances"] == 1
    assert rows["split"]["instances"] == 1


def test_explicitness_breakdown_counts_branches_not_instances_only():
    rows = explicitness_breakdown([_gold_conditional()]).rows
    assert rows["explicit"]["exception branches"] == 1
