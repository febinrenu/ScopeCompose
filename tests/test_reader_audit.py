"""The entailment audit: flags independent of tau, offline recomputation equal to the live scorer, a blind
sample, V2 by hand, and calibration that refuses to pick a threshold it cannot justify."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmark.review import Item, ReviewStore, labels, load_items, render_entailment, review
from contract.mock import generate
from detection.nli import NLIScores
from experiments.reader_audit import (
    PRECISION_BAR,
    RECALL_BAR,
    Decision,
    calibrate_threshold,
    evaluate_v2,
    explain,
    lenient,
    sample_for_audit,
    strict_at,
    to_review_payload,
)
from metrics.case_accuracy import NOT_DETERMINED, ReaderOutput, score_persona
from metrics.validity import OUTPUTS, battery_outputs


class VariedNLI:
    """Entailment varies deterministically with the text, so thresholds actually move verdicts."""

    def score_batch(self, pairs):
        out = []
        for premise, hyp in pairs:
            e = int(hashlib.sha1((premise + hyp).encode()).hexdigest()[:6], 16) / 0xFFFFFF
            out.append(NLIScores(e, 1 - e, 0.0))
        return out


def sentence_reader(*, query, answer, persona):
    parts = [s.strip() for s in answer.replace(" However, ", ". However, ").split(". ") if s.strip()]
    if not parts or "sorry" in answer.lower():
        return ReaderOutput(NOT_DETERMINED, "")
    pick = parts[1] if ("-tier" in persona and len(parts) > 1) else parts[0]
    return ReaderOutput(pick, pick)


def insts(n=8):
    out = [i for i in generate(200, seed=0) if i.personas and i.gold_scoped_answer][:n]
    assert len(out) == n
    return out


def all_decisions():
    ds = []
    for inst in insts():
        for system, answer in battery_outputs(inst).items():
            for p in inst.personas:
                ds.append(explain(inst, p.persona_id, system, answer, sentence_reader, VariedNLI()))
    return ds


def D(item_id="i", *, determined=True, verbatim=True, hedged=False, matches=True, ent=0.9):
    return Decision(item_id, "inst", "u", "B0", "q", "a", "p", "gold", "out", "ev", determined, verbatim, hedged,
                    matches, ent)


# --------------------------------------------------------------------------- #
# Flags and the consistency property
# --------------------------------------------------------------------------- #


def test_explain_records_each_conditions_raw_outcome():
    inst = insts(1)[0]
    d = explain(inst, "u_default", "gold", inst.gold_scoped_answer, sentence_reader, VariedNLI())
    assert d.determined and d.verbatim and not d.hedged and d.matches_gold and 0.0 <= d.entailment <= 1.0
    silent = explain(inst, "u_default", "null", "I'm sorry, no.", sentence_reader, VariedNLI())
    assert not silent.determined and not silent.verbatim and silent.entailment is None


def test_the_nli_model_is_not_asked_when_the_span_is_not_verbatim():
    inst = insts(1)[0]
    liar = lambda *, query, answer, persona: ReaderOutput("no fee applies", "something the answer never said")  # noqa: E731

    class Spy(VariedNLI):
        calls = 0

        def score_batch(self, pairs):
            Spy.calls += 1
            return super().score_batch(pairs)

    d = explain(inst, "u_exception", "x", inst.gold_scoped_answer, liar, Spy())
    assert d.determined and not d.verbatim and d.entailment is None and Spy.calls == 0


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999])
def test_offline_recomputation_equals_the_live_scorer_at_every_tau(tau):
    """The property calibration depends on. If these ever disagree, a threshold chosen from the audit would
    not be the threshold the metric applies."""
    for inst in insts():
        for system, answer in battery_outputs(inst).items():
            for p in inst.personas:
                live = score_persona(inst, p, answer, sentence_reader, VariedNLI(), threshold=tau)
                d = explain(inst, p.persona_id, system, answer, sentence_reader, VariedNLI())
                assert strict_at(d, tau) == live.strict_correct, (inst.instance_id, system, p.persona_id, tau)
                assert lenient(d) == live.lenient_correct


def test_strict_is_never_more_generous_than_lenient_and_is_monotone_in_tau():
    ds = all_decisions()
    assert all(not strict_at(d, 0.5) or lenient(d) for d in ds)
    taus = [0.0, 0.25, 0.5, 0.75, 1.0]
    for d in ds:
        verdicts = [strict_at(d, t) for t in taus]
        assert verdicts == sorted(verdicts, reverse=True), "raising tau can only remove acceptances"


def test_strict_at_works_on_plain_dicts_too():
    d = D(ent=0.8)
    from dataclasses import asdict

    assert strict_at(asdict(d), 0.5) is True and strict_at(asdict(d), 0.9) is False


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_the_sample_is_stratified_deterministic_and_the_requested_size():
    ds = all_decisions()
    s = sample_for_audit(ds, n=40, tau=0.5, seed=1)
    assert len(s) == 40 and s == sample_for_audit(ds, n=40, tau=0.5, seed=1)
    assert [d.item_id for d in s] == sorted(d.item_id for d in s)
    acc = sum(strict_at(d, 0.5) for d in s)
    assert 10 <= acc <= 25, "about half is drawn from the pipeline's accepted decisions, so precision is estimable"
    assert any(d.determined and not strict_at(d, 0.5) for d in s), "false rejections are sampled"
    assert any(not d.determined for d in s)


def test_a_stratum_that_cannot_fill_its_share_is_topped_up_from_the_others():
    few = [D(f"a{i}") for i in range(3)] + [D(f"s{i}", determined=False, ent=None, verbatim=False, matches=False)
                                             for i in range(20)]
    assert len(sample_for_audit(few, n=10, tau=0.5)) == 10


def test_the_review_payload_hides_the_system_and_the_readers_output():
    d = D()
    p = to_review_payload(d)
    assert set(p) == {"item_id", "query", "answer", "persona", "claim"}
    assert p["claim"] == "gold"
    for hidden in (d.reader_outcome, d.evidence, d.system):
        assert hidden not in p.values() or hidden == p["claim"]


# --------------------------------------------------------------------------- #
# V2
# --------------------------------------------------------------------------- #


def test_v2_confusion_matrix_by_hand():
    ds = [D("a", ent=0.9), D("b", ent=0.9), D("c", ent=0.9), D("d", ent=0.2), D("e", ent=0.2), D("f", determined=False,
          verbatim=False, matches=False, ent=None)]
    truth = {"a": True, "b": True, "c": False, "d": True, "e": False, "f": False}
    r = evaluate_v2(ds, truth, tau=0.5)
    assert (r.tp, r.fp, r.fn, r.tn, r.n) == (2, 1, 1, 2, 6)
    assert r.precision.point == pytest.approx(2 / 3) and r.recall.point == pytest.approx(2 / 3)
    assert r.passes is False and "V2 reader validity: FAIL" in r.render()


def test_v2_passes_only_when_both_bars_are_met():
    ds = [D(f"t{i}") for i in range(20)] + [D(f"n{i}", ent=0.1) for i in range(5)]
    truth = {f"t{i}": True for i in range(20)} | {f"n{i}": False for i in range(5)}
    r = evaluate_v2(ds, truth, tau=0.5)
    assert r.precision.point == 1.0 and r.recall.point == 1.0 and r.passes is True
    assert PRECISION_BAR == 0.95 and RECALL_BAR == 0.85


def test_unlabelled_items_are_skipped_not_guessed_and_unevaluable_is_not_a_pass():
    ds = [D("a"), D("b")]
    r = evaluate_v2(ds, {}, tau=0.5)
    assert r.n == 0 and r.passes is None and "NOT EVALUABLE" in r.render()
    none_accepted = evaluate_v2([D("a", ent=0.1)], {"a": True}, tau=0.9)
    assert none_accepted.precision is None and none_accepted.passes is None


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_calibration_picks_the_smallest_tau_that_reaches_the_precision_bar():
    low_bad = [D(f"l{i}", ent=0.2 + i * 0.01) for i in range(10)]          # entailment 0.20-0.29: people say no
    high_good = [D(f"h{i}", ent=0.80 + i * 0.01) for i in range(10)]        # 0.80-0.89: people say yes
    truth = {d.item_id: d.item_id.startswith("h") for d in low_bad + high_good}
    tau = calibrate_threshold(low_bad + high_good, truth)
    assert tau == pytest.approx(0.80), "the most permissive threshold that is still safe"
    assert evaluate_v2(low_bad + high_good, truth, tau=tau).precision.point == 1.0


def test_the_precision_bar_tolerates_one_false_acceptance_in_twenty_one_and_calibration_uses_that():
    """0.95 is a bar, not a demand for perfection: 20 right and 1 wrong is 0.952, which clears it."""
    bad = [D("bad", ent=0.30)]
    good = [D(f"g{i}", ent=0.80 + i * 0.001) for i in range(20)]
    truth = {"bad": False} | {d.item_id: True for d in good}
    assert calibrate_threshold(bad + good, truth) == pytest.approx(0.30)
    two_bad = bad + [D("bad2", ent=0.30)] + good          # same score: no threshold can separate them
    assert calibrate_threshold(two_bad, truth | {"bad2": False}) == pytest.approx(0.80), "two errors in 22 does not"


def test_calibration_returns_none_rather_than_a_threshold_it_cannot_justify():
    ds = [D(f"x{i}", ent=0.5 + i * 0.01) for i in range(10)]
    truth = {d.item_id: i % 2 == 0 for i, d in enumerate(ds)}             # entailment carries no signal
    assert calibrate_threshold(ds, truth) is None
    assert calibrate_threshold([], {}) is None


def test_calibration_only_considers_decisions_the_other_conditions_already_accept():
    hedged = [D(f"h{i}", hedged=True, ent=0.99) for i in range(5)]
    truth = {d.item_id: False for d in hedged}
    assert calibrate_threshold(hedged, truth) is None, "no threshold can rescue a hedged span"


# --------------------------------------------------------------------------- #
# The review tool's entailment kind
# --------------------------------------------------------------------------- #


class Script:
    def __init__(self, a):
        self.a = list(a)

    def show(self, t):
        self.shown = t

    def choose(self, prompt, options):
        v = self.a.pop(0)
        assert v in options
        return v

    def text(self, prompt, default=""):
        return default


def test_the_entailment_kind_offers_yes_no_unsure_and_excludes_unsure_from_the_truth(tmp_path):
    st = ReviewStore(tmp_path, "entailment", "alice")
    items = [Item(f"e{i}", "entailment", to_review_payload(D(f"e{i}"))) for i in range(4)]
    s = review(items, st, Script(["yes", "no", "unsure", "skip"]))
    assert s.shown == 4
    assert labels(st) == {"e0": "yes", "e1": "no"}


def test_the_entailment_item_renders_the_question_and_never_the_readers_view(tmp_path):
    text = render_entailment(Item("e", "entailment", to_review_payload(D())))
    assert "Does the ANSWER, by itself, establish the CLAIM" in text and "do not use outside knowledge" in text
    assert "out" not in text.split("PERSON:")[0].lower().replace("outside", "")


def test_entailment_items_load_by_item_id(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(to_review_payload(D("z1"))) + "\n", encoding="utf-8")
    assert [i.item_id for i in load_items("entailment", p)] == ["z1"]
