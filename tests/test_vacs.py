"""VACS: the sampler steers toward the trap quadrants, and the evaluation is honest about abstention,
thresholds tuned in the baselines' favour, and what H-V and G4 actually require."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.vacs import (
    ALIGNED,
    LABELS,
    Quadrant,
    VacsItem,
    assert_no_test_documents,
    content_tokens,
    lexical_overlap,
    load,
    median_threshold,
    quadrant_counts,
    sample,
    save,
    with_labels,
)
from contract.frames import NormativeFrame
from contract.gold import Applicability
from detection.signature import make_signature
from experiments.vacs_eval import (
    OFF_DIAGONAL,
    best_threshold,
    cosine_scores,
    decision_string,
    evaluate,
    qnfa_method,
    score,
    threshold_method,
)


def item(i, overlap, label=None, proposed=None, split="train"):
    return VacsItem(f"v{i:03d}", f"text a {i}", f"text b {i}", f"da{i}", f"db{i}", overlap, proposed, label, split)


# --------------------------------------------------------------------------- #
# Overlap and quadrants
# --------------------------------------------------------------------------- #


def test_lexical_overlap_is_jaccard_over_content_words():
    assert lexical_overlap("The fee is waived for premium holders", "The fee is waived for premium holders") == 1.0
    assert lexical_overlap("overseas teller machine use", "reward holders enjoy access") == 0.0
    assert 0 < lexical_overlap("a fee applies to transfers", "no fee applies to cash") < 1
    assert lexical_overlap("", "") == 0.0 and "the" not in content_tokens("the fee")


def test_a_quadrant_comes_from_the_measured_overlap_and_the_human_label_never_the_proposal():
    thr = 0.3
    lowsame = item(1, 0.1, label="equivalent", proposed=False)       # the model proposed 'different'
    assert lowsame.quadrant(thr) is Quadrant.LO_SAME, "reported by the label, not the steering proposal"
    assert item(2, 0.5, label="distinct").quadrant(thr) is Quadrant.HI_DIFF
    assert item(3, 0.5, label="b_broader").quadrant(thr) is Quadrant.HI_SAME
    assert item(4, 0.1, label="distinct").quadrant(thr) is Quadrant.LO_DIFF
    assert item(5, 0.5).quadrant(thr) is None, "an unlabelled item is in no quadrant"
    assert {Quadrant.HI_DIFF, Quadrant.LO_SAME} == {q for q in Quadrant if q.off_diagonal}


def test_aligned_covers_equivalent_and_both_subsumption_directions():
    assert [item(i, 0.1, label=l).aligned for i, l in enumerate(LABELS)] == [True, True, True, False]
    assert ALIGNED == ("equivalent", "a_broader", "b_broader") and item(9, 0.1).aligned is None


def test_the_median_threshold_splits_the_pool_in_half():
    items = [item(i, o) for i, o in enumerate([0.1, 0.2, 0.3, 0.4])]
    assert median_threshold(items) == pytest.approx(0.25)
    assert median_threshold([item(0, 0.1), item(1, 0.5), item(2, 0.9)]) == 0.5
    with pytest.raises(ValueError):
        median_threshold([])


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def pool():
    out = []
    n = 0
    for overlap, same, count in ((0.8, True, 40), (0.8, False, 40), (0.1, True, 40), (0.1, False, 40)):
        for _ in range(count):
            out.append(item(n, overlap + (n % 7) * 0.001, proposed=same))
            n += 1
    return out


def test_sampling_oversamples_the_two_trap_quadrants():
    p = pool()
    thr = median_threshold(p)
    got = sample(p, n=40, seed=0, off_diagonal_share=0.5, threshold=thr)
    cells = [(i.overlap >= thr, i.proposed_same) for i in got]
    off = sum(1 for hi, same in cells if hi != same)
    assert len(got) == 40 and off == 20, "half of the draw targets hi/different and lo/same"
    assert sum(1 for hi, same in cells if hi and not same) == 10 and sum(1 for hi, same in cells if not hi and same) == 10


def test_sampling_is_seeded_and_ordered_and_a_shortfall_is_filled_not_hidden():
    p = pool()
    assert sample(p, n=30, seed=1) == sample(p, n=30, seed=1)
    assert [i.item_id for i in sample(p, n=30, seed=1)] == sorted(i.item_id for i in sample(p, n=30, seed=1))
    thin = [i for i in p if not (i.overlap < 0.5 and i.proposed_same)][:60]   # no lo/same items at all
    got = sample(thin, n=40, seed=0, off_diagonal_share=0.5, threshold=0.5)
    assert len(got) == 40, "filled from the other quadrants"
    assert not any(i.overlap < 0.5 and i.proposed_same for i in got)


def test_test_documents_are_refused():
    with pytest.raises(ValueError, match="train/dev"):
        assert_no_test_documents([item(1, 0.2, split="test")])
    with pytest.raises(ValueError, match="train/dev"):
        sample([item(1, 0.2, proposed=True, split="test")], n=1)


def test_sampling_ignores_items_without_a_proposal_and_validates_its_share():
    assert sample([item(1, 0.2), item(2, 0.9)], n=2, threshold=0.5) == []
    with pytest.raises(ValueError):
        sample(pool(), n=4, off_diagonal_share=1.5)


def test_labels_attach_by_id_and_unknown_labels_are_refused():
    items = [item(1, 0.2), item(2, 0.3)]
    got = with_labels(items, {"v001": "distinct"})
    assert [i.label for i in got] == ["distinct", None]
    with pytest.raises(ValueError, match="unknown VACS label"):
        with_labels(items, {"v001": "maybe"})


def test_save_and_load_round_trip(tmp_path):
    items = [item(1, 0.2, label="equivalent", proposed=True), item(2, 0.7)]
    save(items, tmp_path / "v.jsonl")
    assert load(tmp_path / "v.jsonl") == items


def test_quadrant_counts():
    items = [item(1, 0.5, "distinct"), item(2, 0.5, "distinct"), item(3, 0.1, "equivalent"), item(4, 0.1)]
    c = quadrant_counts(items, 0.3)
    assert c[Quadrant.HI_DIFF] == 2 and c[Quadrant.LO_SAME] == 1 and c[Quadrant.HI_SAME] == 0


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def test_the_best_threshold_is_the_oracle_cut_and_is_generous_to_the_baseline():
    scores, aligned = [0.9, 0.8, 0.3, 0.2], [True, True, False, False]
    t, acc = best_threshold(scores, aligned)
    assert acc == 1.0 and 0.2 < t <= 0.8
    assert best_threshold([], []) == (0.5, 0.0)


def test_cosine_scores_are_batched_and_correct():
    emb = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0], "z": [0.0, 0.0]}
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.array([emb[t] for t in texts])

    got = cosine_scores(embed, [("a", "b"), ("a", "c"), ("a", "z")])
    assert got == pytest.approx([1.0, 0.0, 0.0]) and len(calls) == 1, "one encode for every pair"
    assert cosine_scores(embed, []) == []


def test_abstention_is_scored_wrong_and_reported():
    items = [item(1, 0.5, "equivalent"), item(2, 0.5, "distinct"), item(3, 0.5, "equivalent")]
    answers = {"v001": True, "v002": None, "v003": None}
    r = score("m", lambda it: answers[it.item_id], items, overlap_threshold=0.3)
    assert r.n == 3 and r.correct == 1 and r.abstained == 2
    assert r.accuracy == pytest.approx(1 / 3) and r.abstain_rate == pytest.approx(2 / 3)


def test_unlabelled_items_have_no_right_answer_and_are_skipped():
    r = score("m", lambda it: True, [item(1, 0.5), item(2, 0.5, "equivalent")], overlap_threshold=0.3)
    assert r.n == 1 and r.correct == 1


def test_per_quadrant_accuracy_is_reported():
    items = [item(1, 0.5, "distinct"), item(2, 0.5, "distinct"), item(3, 0.1, "equivalent"), item(4, 0.1, "equivalent")]
    r = score("always_aligned", lambda it: True, items, overlap_threshold=0.3)
    assert r.quadrant_accuracy(Quadrant.HI_DIFF) == 0.0 and r.quadrant_accuracy(Quadrant.LO_SAME) == 1.0
    assert r.quadrant_accuracy(Quadrant.HI_SAME) is None
    assert r.off_diagonal_accuracy == pytest.approx(0.5) and set(OFF_DIAGONAL) == {Quadrant.HI_DIFF, Quadrant.LO_SAME}


def trap_items():
    """Like real data: similarity is informative on the easy diagonal and misleading off it.

    30 high-overlap/same and 30 low-overlap/different items (easy), plus 5 high-overlap/different and
    5 low-overlap/same (the traps). A threshold on overlap is right on all 60 easy items and wrong on all 10 traps."""
    out = []
    n = 0
    for overlap, label, count in ((0.8, "equivalent", 30), (0.1, "distinct", 30),
                                  (0.8, "distinct", 5), (0.1, "equivalent", 5)):
        for _ in range(count):
            out.append(item(n, overlap, label))
            n += 1
    return out


def test_a_similarity_method_is_wrong_on_every_trap_even_with_the_best_threshold():
    items = trap_items()
    sim = {it.item_id: it.overlap for it in items}      # a perfect similarity signal that is just overlap
    method, thr = threshold_method("overlap", lambda its: [sim[i.item_id] for i in its], items)
    r = score("overlap", method, items, overlap_threshold=0.4, threshold=thr)
    assert r.quadrant_accuracy(Quadrant.HI_DIFF) == 0.0 and r.quadrant_accuracy(Quadrant.LO_SAME) == 0.0
    assert r.quadrant_accuracy(Quadrant.HI_SAME) == 1.0 and r.quadrant_accuracy(Quadrant.LO_DIFF) == 1.0
    assert r.accuracy == pytest.approx(60 / 70), "right wherever similarity and truth agree, however well tuned"
    assert r.off_diagonal_accuracy == 0.0


def test_qnfa_method_aligns_by_signature_unification_and_abstains_when_it_cannot_decide():
    def fr(sig):
        return NormativeFrame(passage_id="p", signature=make_signature("banking", sig),
                              applicability=Applicability(descriptor="x", is_default=True), outcome="o", span="s")

    frames = {
        ("a", "v001"): [fr({"event": "cash_withdrawal", "decision_kind": "flat_fee"})],
        ("b", "v001"): [fr({"event": "atm_withdrawal", "decision_kind": "flat_fee"})],        # subsumes: aligned
        ("a", "v002"): [fr({"event": "transfer", "decision_kind": "flat_fee"})],
        ("b", "v002"): [fr({"event": "cash_withdrawal", "decision_kind": "flat_fee"})],       # distinct
        ("a", "v003"): [], ("b", "v003"): [fr({"event": "transfer"})],                        # no frames: abstain
    }
    m = qnfa_method(lambda which, it: frames[(which, it.item_id)])
    assert m(item(1, 0.1)) is True and m(item(2, 0.9)) is False and m(item(3, 0.5)) is None


def test_qnfa_abstains_when_every_pair_is_uncertain():
    from contract.frames import Signature

    free = Signature(domain="banking", slots={}, untyped_slots={"event": "lounge entry"}, schema_version="t")
    typed = make_signature("banking", {"event": "cash_withdrawal"})
    mk = lambda s: NormativeFrame(passage_id="p", signature=s, applicability=Applicability(descriptor="x"),  # noqa: E731
                                  outcome="o", span="s")
    m = qnfa_method(lambda which, it: [mk(free)] if which == "a" else [mk(typed)])
    assert m(item(1, 0.5)) is None


def test_the_decision_string_is_stable_and_readable():
    f = NormativeFrame(passage_id="p", signature=make_signature("banking", {"event": "atm_withdrawal", "location": "abroad",
                                                                           "colour": "blue"}),
                       applicability=Applicability(descriptor="x"), outcome="o", span="s")
    assert decision_string(f) == "event atm withdrawal ; location abroad ; colour blue", "typed slots (sorted), then untyped"


def test_h_v_and_g4_follow_the_per_quadrant_gains():
    items = trap_items()
    sim = {it.item_id: it.overlap for it in items}
    overlap_m, thr = threshold_method("passage_cosine", lambda its: [sim[i.item_id] for i in its], items)
    truth = lambda it: it.aligned  # noqa: E731  (a method that reads the label stands in for a perfect aligner)
    rep = evaluate({"passage_cosine": overlap_m, "variable_cosine": overlap_m, "qnfa": truth}, items,
                   thresholds={"passage_cosine": thr}, overlap_threshold=0.4)
    assert rep.g4_passes() and rep.h_v_holds() is True
    assert rep.gain("passage_cosine", Quadrant.HI_DIFF) == 1.0 and rep.gain("passage_cosine", Quadrant.HI_SAME) == 0.0
    text = rep.render()
    assert "tuned ON VACS" in text and "G4" in text and "H-V" in text and "PASS" in text


def test_g4_fails_when_qnfa_does_not_beat_the_baseline_and_h_v_is_unevaluable_without_all_quadrants():
    items = [item(1, 0.5, "equivalent"), item(2, 0.5, "distinct")]
    same = lambda it: it.aligned  # noqa: E731
    rep = evaluate({"variable_cosine": same, "passage_cosine": same, "qnfa": same}, items, overlap_threshold=0.3)
    assert not rep.g4_passes() and rep.h_v_holds() is None


def test_evaluation_requires_labelled_items():
    with pytest.raises(ValueError, match="no labelled"):
        evaluate({"m": lambda it: True}, [item(1, 0.5)])
