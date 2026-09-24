"""Tests for A2 and A3: detection, features, and the three-variant comparison.

Everything here runs offline with the heuristic NLI stand-in. The point is not
to measure any variant's accuracy -- the stand-in makes that meaningless -- but
to verify the invariants that must hold whatever model is plugged in, above all
that no decision depends on the order passages arrived in.
"""

from __future__ import annotations

import numpy as np
import pytest

from contract.models import (
    ConflictType,
    Construction,
    Domain,
    Passage,
    QueryRecord,
    ScopeRelation,
    SourceType,
)
from detection.features import extract_pair_features, more_specific
from detection.head import HeadConfig, PairHead, interaction_features
from detection.nli import HeuristicNLI
from detection.pipeline import TwoStageDetector
from detection.stage1 import Stage1Filter, enumerate_pairs, feature_names
from detection.stage2 import Stage2Judgement, _coerce, _salvage_json
from detection.variants.lexical_nli import LexicalNLIDetector
from detection.variants.structured_entailment import (
    EntailmentEvidence,
    StructuredEntailmentDetector,
)

FEE_RULE = "International transactions incur a 3% fee."
FEE_EXCEPTION = "The fee is waived for premium-tier cardholders."
FEE_CONTRADICTION = "International transactions incur a 5% fee."
UNRELATED = "Statements are issued on the first business day of each month."


def _p(pid: str, text: str) -> Passage:
    return Passage(id=pid, text=text, source_type=SourceType.OFFICIAL_POLICY, date="2024-01")


def _record(texts: list[str], **kw) -> QueryRecord:
    return QueryRecord(
        query_id=kw.pop("query_id", "q1"),
        query=kw.pop("query", "do I pay a fee?"),
        domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL,
        passages=[_p(f"p{i}", t) for i, t in enumerate(texts)],
        conflict_pairs=kw.pop("conflict_pairs", []),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def test_exception_cue_is_detected():
    f = extract_pair_features(FEE_RULE, FEE_EXCEPTION)
    assert f.exception_cues_max >= 1


def test_restriction_asymmetry_fires_on_a_narrowed_scope():
    f = extract_pair_features(FEE_RULE, FEE_EXCEPTION)
    assert f.restriction_asymmetry >= 1


def test_numeric_clash_fires_on_a_factual_contradiction():
    f = extract_pair_features(FEE_RULE, FEE_CONTRADICTION)
    assert f.numeric_clash == 1.0


def test_unrelated_passages_share_few_tokens():
    assert extract_pair_features(FEE_RULE, UNRELATED).token_jaccard < 0.2


def test_features_are_order_invariant():
    """A feature that changed under passage reordering would break the
    order-invariance property A4 must satisfy, and would do it invisibly."""
    a = extract_pair_features(FEE_RULE, FEE_EXCEPTION)
    b = extract_pair_features(FEE_EXCEPTION, FEE_RULE)
    sym = ["exception_cues_max", "exception_cues_asymmetry", "restriction_asymmetry",
           "specificity_asymmetry", "length_ratio", "numeric_overlap", "numeric_clash",
           "year_clash", "temporal_cues_max", "opinion_cues_max", "token_jaccard"]
    for name in sym:
        assert getattr(a, name) == getattr(b, name), f"{name} is order-dependent"


def test_feature_vector_matches_declared_names():
    f = extract_pair_features(FEE_RULE, FEE_EXCEPTION)
    assert len(f.as_vector()) == len(f.names())


def test_specificity_ranking():
    assert more_specific(FEE_RULE, FEE_EXCEPTION) == -1 or \
           more_specific(FEE_EXCEPTION, FEE_RULE) == -1


def test_specificity_is_antisymmetric():
    assert more_specific(FEE_RULE, FEE_EXCEPTION) == -more_specific(FEE_EXCEPTION, FEE_RULE)


# --------------------------------------------------------------------------- #
# NLI stand-in
# --------------------------------------------------------------------------- #


def test_heuristic_nli_probabilities_sum_to_one():
    for a, b in [(FEE_RULE, FEE_EXCEPTION), (FEE_RULE, FEE_CONTRADICTION), (FEE_RULE, UNRELATED)]:
        s = HeuristicNLI().score(a, b)
        assert abs(s.entailment + s.neutral + s.contradiction - 1.0) < 1e-9


def test_conflict_signal_counts_neutral():
    """A general rule and its valid exception are NOT logically contradictory,
    so a correct NLI model calls them neutral. A detector keyed only on
    contradiction would miss the entire conditional class -- the one the
    project exists for."""
    from detection.nli import NLIScores

    neutral = NLIScores(entailment=0.1, neutral=0.8, contradiction=0.1)
    assert neutral.conflict_signal > neutral.contradiction


def test_numeric_clash_reads_as_contradiction():
    assert HeuristicNLI().score(FEE_RULE, FEE_CONTRADICTION).contradiction > 0.5


# --------------------------------------------------------------------------- #
# Pair enumeration
# --------------------------------------------------------------------------- #


def test_five_passages_make_ten_pairs():
    passages = [_p(f"p{i}", f"text {i}") for i in range(5)]
    assert len(enumerate_pairs(passages)) == 10


def test_pair_enumeration_is_order_invariant():
    passages = [_p(f"p{i}", f"text {i}") for i in range(4)]
    forward = {(a.id, b.id) for a, b in enumerate_pairs(passages)}
    backward = {(a.id, b.id) for a, b in enumerate_pairs(list(reversed(passages)))}
    assert forward == backward


def test_pairs_come_back_in_canonical_order():
    passages = [_p("p3", "c"), _p("p1", "a"), _p("p2", "b")]
    for a, b in enumerate_pairs(passages):
        assert a.id < b.id


# --------------------------------------------------------------------------- #
# Interaction features
# --------------------------------------------------------------------------- #


def test_interaction_block_has_three_times_the_width():
    e_i = np.array([[1.0, 2.0, 3.0]])
    e_j = np.array([[4.0, 5.0, 6.0]])
    assert interaction_features(e_i, e_j).shape == (1, 9)


def test_interaction_block_is_order_symmetric():
    """Swapping the two embeddings must not change the vector.

    Without this the detector can return different answers for the same pair
    depending on retrieval order, silently breaking order invariance.
    """
    e_i = np.array([[1.0, 2.0, 3.0]])
    e_j = np.array([[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        interaction_features(e_i, e_j), interaction_features(e_j, e_i)
    )


def test_interaction_block_is_symmetric_on_the_tie_case():
    """Two DIFFERENT embeddings sharing an element sum.

    This is the case the previous implementation got wrong: it sorted the two
    vectors into a canonical position on their sum, and on a tie the sort was
    arbitrary, so the vector genuinely differed under swap. The old test used
    unequal sums and never exercised it. A claimed design property that holds
    'almost always' is not a property.
    """
    e_i = np.array([[1.0, -1.0, 2.0]])   # sum 2.0
    e_j = np.array([[2.0, -1.0, 1.0]])   # sum 2.0, different vector
    assert e_i.sum() == e_j.sum()
    assert not np.allclose(e_i, e_j)
    np.testing.assert_allclose(
        interaction_features(e_i, e_j), interaction_features(e_j, e_i)
    )


def test_interaction_block_is_symmetric_on_random_pairs():
    """Property check over many random pairs, including high dimensions where
    a real encoder operates."""
    rs = np.random.RandomState(0)
    for dim in (3, 64, 384):
        a = rs.normal(size=(1, dim))
        b = rs.normal(size=(1, dim))
        np.testing.assert_allclose(
            interaction_features(a, b), interaction_features(b, a), atol=1e-12
        )


def test_interaction_block_still_distinguishes_different_pairs():
    """Symmetry must not be bought by collapsing distinct pairs together."""
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[4.0, 5.0, 6.0]])
    c = np.array([[9.0, 1.0, 0.0]])
    assert not np.allclose(interaction_features(a, b), interaction_features(a, c))


# --------------------------------------------------------------------------- #
# Learned head
# --------------------------------------------------------------------------- #


def test_head_trains_and_predicts():
    rs = np.random.RandomState(0)
    X = np.vstack([rs.normal(0, 1, (40, 5)), rs.normal(3, 1, (40, 5))])
    y = ["a"] * 40 + ["b"] * 40
    head = PairHead(HeadConfig(), feature_names=[f"f{i}" for i in range(5)]).fit(X, y)
    assert set(head.predict(X)) <= {"a", "b"}
    assert head.predict_proba(X).shape == (80, 2)


def test_head_refuses_a_single_class():
    with pytest.raises(ValueError, match="at least two classes"):
        PairHead().fit(np.zeros((10, 3)), ["only"] * 10)


def test_untrained_head_raises_clearly():
    with pytest.raises(RuntimeError, match="not trained"):
        PairHead().predict(np.zeros((1, 3)))


def test_head_round_trips_through_disk(tmp_path):
    rs = np.random.RandomState(0)
    X = np.vstack([rs.normal(0, 1, (30, 4)), rs.normal(3, 1, (30, 4))])
    y = ["a"] * 30 + ["b"] * 30
    head = PairHead(feature_names=[f"f{i}" for i in range(4)]).fit(X, y)
    path = tmp_path / "head.pkl"
    head.save(path)
    assert PairHead.load(path).predict(X) == head.predict(X)


def test_logistic_head_is_inspectable():
    """Variant (i)'s selling point is that its decisions can be read off a
    coefficient table, so this is part of the deliverable."""
    rs = np.random.RandomState(0)
    X = np.vstack([rs.normal(0, 1, (30, 4)), rs.normal(3, 1, (30, 4))])
    head = PairHead(feature_names=["a", "b", "c", "d"]).fit(X, ["x"] * 30 + ["y"] * 30)
    coefs = head.coefficients()
    assert coefs
    assert head.top_features(list(coefs)[0], n=2)


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #


@pytest.fixture
def stage1() -> Stage1Filter:
    # auto_load_head=False on purpose: a trained head in models/ would make
    # these tests pass or fail depending on whether someone had run
    # `python -m detection.train`, which is not a property of the code.
    return Stage1Filter(heuristic_nli=True, auto_load_head=False)


def test_stage1_scores_every_pair(stage1):
    rec = _record([FEE_RULE, FEE_EXCEPTION, UNRELATED])
    assert len(stage1.score_pairs(rec.passages)) == 3


def test_stage1_confidence_is_a_probability(stage1):
    rec = _record([FEE_RULE, FEE_EXCEPTION, FEE_CONTRADICTION, UNRELATED])
    for c in stage1.score_pairs(rec.passages):
        assert 0.0 <= c.confidence <= 1.0


def test_stage1_is_order_invariant(stage1):
    """The same pair must get the same verdict whichever order it arrives in."""
    texts = [FEE_RULE, FEE_EXCEPTION, UNRELATED]
    rec = _record(texts)
    flipped = QueryRecord(
        query_id="q1", query=rec.query, domain=rec.domain, construction=rec.construction,
        passages=list(reversed(rec.passages)), conflict_pairs=[],
    )
    a = {c.key: (c.is_conflict, round(c.confidence, 6)) for c in stage1.score_pairs(rec.passages)}
    b = {c.key: (c.is_conflict, round(c.confidence, 6)) for c in stage1.score_pairs(flipped.passages)}
    assert a == b


def test_unrelated_passages_score_low(stage1):
    rec = _record([FEE_RULE, UNRELATED])
    assert stage1.score_pairs(rec.passages)[0].confidence > 0.5  # confident NOT a conflict


def test_feature_names_match_the_vector(stage1):
    rec = _record([FEE_RULE, FEE_EXCEPTION])
    assert len(stage1.score_pairs(rec.passages)[0].feature_vector()) == len(feature_names())


def test_stage1_head_training(stage1):
    from contract.mock import generate

    records = [i.to_query_record() for i in generate(30, seed=0)]
    head = stage1.train_head(records)
    assert stage1.is_trained
    assert len(head.classes_) == 2


# --------------------------------------------------------------------------- #
# Stage 2 response handling
# --------------------------------------------------------------------------- #


def test_conditional_without_a_relation_is_downgraded():
    """A conditional label with no scope relation is unroutable. Downgrading
    to factual sends it to selection -- what prior work would have done --
    rather than emitting a pair the contract rejects."""
    j = _coerce({"is_conflict": True, "type": "conditional", "scope_relation": None})
    assert j.type is ConflictType.FACTUAL
    assert j.scope_relation is None


def test_relation_stripped_from_a_temporal_label():
    j = _coerce({"is_conflict": True, "type": "temporal", "scope_relation": "refinement"})
    assert j.scope_relation is None


def test_no_conflict_clears_everything():
    j = _coerce({"is_conflict": True, "type": "no_conflict", "scope_relation": "opposed"})
    assert not j.is_conflict
    assert j.scope_relation is None


def test_unparseable_type_degrades_to_no_conflict():
    assert _coerce({"type": "nonsense"}).type is ConflictType.NO_CONFLICT


def test_valid_conditional_survives():
    j = _coerce({"is_conflict": True, "type": "conditional",
                 "scope_relation": "refinement", "confidence": 0.9})
    assert j.type is ConflictType.CONDITIONAL
    assert j.scope_relation is ScopeRelation.REFINEMENT
    assert j.confidence == 0.9


def test_confidence_is_clamped():
    assert _coerce({"type": "factual", "is_conflict": True, "confidence": 5.0}).confidence == 1.0
    assert _coerce({"type": "factual", "is_conflict": True, "confidence": -1}).confidence == 0.0


@pytest.mark.parametrize("wrapped", [
    '```json\n{"type": "factual", "is_conflict": true}\n```',
    'Here is the answer: {"type": "factual", "is_conflict": true} -- hope that helps.',
    '{"type": "factual", "is_conflict": true}',
])
def test_json_is_salvaged_from_prose_and_fences(wrapped):
    assert _salvage_json(wrapped).get("type") == "factual"


def test_salvage_gives_up_gracefully():
    assert _salvage_json("no json here at all") == {}


# --------------------------------------------------------------------------- #
# Variant (i): lexical + NLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def lexical(stage1) -> LexicalNLIDetector:
    return LexicalNLIDetector(stage1=stage1)


def test_lexical_variant_spots_an_exception(lexical, stage1):
    rec = _record([FEE_RULE, FEE_EXCEPTION])
    ctype, relation = lexical.classify(stage1.score_pairs(rec.passages)[0])
    assert ctype is ConflictType.CONDITIONAL
    assert relation is not None, "a conditional label must carry a routable relation"


def test_lexical_variant_always_emits_a_routable_relation(lexical, stage1):
    """A conditional pair with no relation would be rejected by the contract
    and would leave Member B with nothing to route on."""
    from contract.mock import generate

    for inst in generate(20, seed=0):
        rec = inst.to_query_record()
        for cand in stage1.score_pairs(rec.passages):
            ctype, relation = lexical.classify(cand)
            if ctype is ConflictType.CONDITIONAL:
                assert relation is not None


def test_lexical_variant_trains(lexical):
    from contract.mock import generate

    lexical.fit([i.to_query_record() for i in generate(40, seed=0)])
    assert lexical.is_trained
    assert lexical.explain()


def test_lexical_variant_is_order_invariant(lexical, stage1):
    rec = _record([FEE_RULE, FEE_EXCEPTION])
    flipped = _record([FEE_EXCEPTION, FEE_RULE])
    a = lexical.classify(stage1.score_pairs(rec.passages)[0])
    b = lexical.classify(stage1.score_pairs(flipped.passages)[0])
    assert a == b


# --------------------------------------------------------------------------- #
# Variant (iii): structured entailment
# --------------------------------------------------------------------------- #


@pytest.fixture
def structured() -> StructuredEntailmentDetector:
    return StructuredEntailmentDetector(heuristic_nli=True)


@pytest.mark.parametrize(
    "scope_ij,scope_ji,agree,contra,expected",
    [
        (0.05, 0.05, 0.1, 0.1, ScopeRelation.DISJOINT),
        (0.9, 0.2, 0.9, 0.05, ScopeRelation.REDUNDANT),
        (0.9, 0.2, 0.05, 0.9, ScopeRelation.REFINEMENT),
        (0.35, 0.35, 0.05, 0.9, ScopeRelation.OPPOSED),
        (0.35, 0.35, 0.9, 0.05, ScopeRelation.REDUNDANT),
    ],
)
def test_structured_variant_applies_the_formal_rule(
    structured, scope_ij, scope_ji, agree, contra, expected
):
    ev = EntailmentEvidence(
        scope_i_entails_j=scope_ij, scope_j_entails_i=scope_ji,
        outcome_agreement=agree, outcome_contradiction=contra,
    )
    assert structured.relation_from_evidence(ev) is expected


def test_structured_variant_asks_scope_in_both_directions(structured):
    """Subset is asymmetric, and that asymmetry is what separates a refinement
    from an opposed pair."""
    ev = structured.gather(FEE_RULE, FEE_EXCEPTION)
    assert isinstance(ev.scope_i_entails_j, float)
    assert isinstance(ev.scope_j_entails_i, float)


def test_structured_variant_explains_itself(structured, stage1):
    rec = _record([FEE_RULE, FEE_EXCEPTION])
    explanation = structured.explain(stage1.score_pairs(rec.passages)[0])
    assert "relation" in explanation
    assert "scope_i_entails_j" in explanation


def test_opposed_is_reported_as_a_factual_contradiction(structured, stage1):
    """Overlapping scopes with neither containing the other and disagreeing
    outcomes means the passages cannot both be true. Calling that conditional
    would route a real contradiction to composition."""
    rec = _record([FEE_RULE, FEE_CONTRADICTION])
    ctype, relation = structured.classify(stage1.score_pairs(rec.passages)[0])
    if relation is ScopeRelation.OPPOSED:
        assert ctype is ConflictType.FACTUAL


# --------------------------------------------------------------------------- #
# Two-stage pipeline
# --------------------------------------------------------------------------- #


def test_detector_runs_offline_without_stage2(stage1):
    detector = TwoStageDetector(stage1=stage1, use_stage2=False)
    rec = _record([FEE_RULE, FEE_EXCEPTION, UNRELATED])
    out, stats = detector.detect(rec)
    assert len(out.conflict_pairs) == 3
    assert stats.escalated_stage2 == 0
    assert stats.resolved_stage1 == 3


def test_detector_output_validates_against_the_contract(stage1):
    """Every pair it emits must be a legal contract record -- otherwise Member
    B receives something the schema rejects."""
    from contract.mock import generate

    detector = TwoStageDetector(stage1=stage1, use_stage2=False,
                                classifier=LexicalNLIDetector(stage1=stage1))
    for inst in generate(20, seed=0):
        rec = inst.to_query_record(include_gold_pairs=False)
        out, _ = detector.detect(rec)
        QueryRecord.model_validate(out.model_dump(mode="json"))


def test_detector_is_order_invariant(stage1):
    detector = TwoStageDetector(stage1=stage1, use_stage2=False,
                                classifier=LexicalNLIDetector(stage1=stage1))
    rec = _record([FEE_RULE, FEE_EXCEPTION, UNRELATED])
    flipped = QueryRecord(
        query_id="q1", query=rec.query, domain=rec.domain, construction=rec.construction,
        passages=list(reversed(rec.passages)), conflict_pairs=[],
    )
    a = {p.key: (p.type, p.scope_relation) for p, in
         [(x,) for x in detector.detect(rec)[0].conflict_pairs]}
    b = {p.key: (p.type, p.scope_relation) for p, in
         [(x,) for x in detector.detect(flipped)[0].conflict_pairs]}
    assert a == b


def test_stage1_only_does_not_invent_a_conditional_label(stage1):
    """Stage 1 establishes only THAT two passages disagree, not HOW. An
    unearned conditional label would route to composition and could fabricate
    a branch, so the honest default is factual."""
    detector = TwoStageDetector(stage1=stage1, use_stage2=False, classifier=None)
    out, _ = detector.detect(_record([FEE_RULE, FEE_EXCEPTION]))
    for pair in out.conflict_pairs:
        if pair.is_conflict:
            assert pair.type is not ConflictType.CONDITIONAL


def test_stats_report_the_local_share(stage1):
    detector = TwoStageDetector(stage1=stage1, use_stage2=False)
    _, stats = detector.detect(_record([FEE_RULE, FEE_EXCEPTION, UNRELATED]))
    assert "resolved locally" in stats.render()
    assert stats.escalation_rate == 0.0
