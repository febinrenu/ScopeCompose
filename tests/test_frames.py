"""Frame extraction: the verbatim gate, the one re-ask, out-of-vocabulary
handling, and the one-way hedge override -- all on a scripted mock backend."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from contract.frames import Modality, NormativeFrame
from contract.models import Passage, SourceType
from detection.frames import (
    STEP,
    STEP_REASK,
    ExtractionStats,
    FrameExtractor,
    build_system_prompt,
)
from detection.signature import load_schema

P0 = Passage(id="p0", text="International cash withdrawals incur a flat 2.50 fee. Contact us for details.",
             source_type=SourceType.OFFICIAL_POLICY)
P1 = Passage(id="p1", text="The cash withdrawal fee is waived for premium-tier cardholders.",
             source_type=SourceType.PRODUCT_TERMS)
QUERY = "Will I be charged for withdrawing cash abroad?"


def frame(pid="p0", span="International cash withdrawals incur a flat 2.50 fee.", **kw):
    f = {
        "passage_id": pid, "span": span,
        "decision": {"event": "cash withdrawal", "location": "abroad", "decision_kind": "flat fee"},
        "applicability": {"descriptor": "all cardholders", "is_default": True, "attributes": []},
        "outcome": "a 2.50 fee applies", "modality": "statement", "confidence": 0.9,
    }
    f.update(kw)
    return f


def exception_frame(**kw):
    f = frame("p1", "The cash withdrawal fee is waived for premium-tier cardholders.",
              applicability={"descriptor": "premium-tier cardholders", "is_default": False,
                             "attributes": [{"name": "card_tier", "kind": "categorical",
                                             "values": ["premium"]}]},
              outcome="no fee applies")
    f.update(kw)
    return f


def extractor(tmp_path, scripted: dict[str, object]) -> tuple[FrameExtractor, MockBackend]:
    backend = MockBackend(scripted={k: json.dumps(v) if not isinstance(v, str) else v
                                    for k, v in scripted.items()})
    client = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                       backend_override=backend)
    return FrameExtractor("banking", tier=Tier.BULK, client=client), backend


FIRST = "Question: Will I be charged"   # only in the extraction prompt (the re-ask has no question)
REASK = "were not verbatim"      # appears only in the re-ask prompt


def test_happy_path_extracts_validated_normalised_frames(tmp_path):
    ex, backend = extractor(tmp_path, {FIRST: {"frames": [frame(), exception_frame()]}})
    out = ex.extract(QUERY, [P0, P1])
    assert len(out.frames) == 2 and all(isinstance(f, NormativeFrame) for f in out.frames)
    rule = next(f for f in out.frames if f.passage_id == "p0")
    assert rule.signature.slots == {"event": "cash_withdrawal", "location": "abroad", "decision_kind": "flat_fee"}
    assert rule.applicability.is_default
    exc = next(f for f in out.frames if f.passage_id == "p1")
    assert exc.applicability.attributes[0].name == "card_tier" and exc.outcome == "no fee applies"
    assert out.stats.calls == 1 and len(backend.calls) == 1
    assert out.stats.frame_yield == 1.0 and out.stats.dropped_span == 0


def test_one_call_per_query_is_batched_over_all_passages(tmp_path):
    ex, backend = extractor(tmp_path, {FIRST: {"frames": []}})
    ex.extract(QUERY, [P0, P1])
    content = backend.calls[0]["messages"][0]["content"]
    assert "[p0]" in content and "[p1]" in content and QUERY in content
    assert "COPIED WORD FOR WORD" in backend.calls[0]["system"]


def test_the_prompt_lists_the_schema_vocabulary():
    sys = build_system_prompt(load_schema("banking"))
    for needle in ("cash_withdrawal", "atm_withdrawal", "flat_fee", "abroad", "decision_kind:"):
        assert needle in sys
    assert "any_txn" not in sys, "the top element means 'unspecified' and is not offered as a value"


def test_a_non_verbatim_span_is_reasked_once_and_recovered(tmp_path):
    bad = frame(span="Cash abroad costs 2.50.")                       # paraphrase
    good = frame()
    ex, backend = extractor(tmp_path, {REASK: {"frames": [good]}, FIRST: {"frames": [bad]}})
    out = ex.extract(QUERY, [P0])
    assert [f.span for f in out.frames] == [good["span"]]
    assert out.stats.reasked == 1 and out.stats.recovered_by_reask == 1
    assert out.stats.dropped_span == 0 and out.stats.calls == 2
    assert len(backend.calls) == 2 and backend.calls[1]["messages"][0]["content"].count("were not verbatim") == 1


def test_a_span_that_is_still_wrong_after_the_reask_is_dropped_and_counted(tmp_path):
    bad = frame(span="Cash abroad costs 2.50.")
    ex, backend = extractor(tmp_path, {REASK: {"frames": [bad]}, FIRST: {"frames": [bad]}})
    out = ex.extract(QUERY, [P0])
    assert out.frames == [] and out.stats.dropped_span == 1 and out.stats.dropped_span_rate == 1.0
    assert len(backend.calls) == 2, "exactly one re-ask, never a loop"


def test_a_frame_the_reask_omits_is_dropped_too(tmp_path):
    ex, _ = extractor(tmp_path, {REASK: {"frames": []}, FIRST: {"frames": [frame(span="made up")]}})
    out = ex.extract(QUERY, [P0])
    assert out.frames == [] and out.stats.dropped_span == 1


def test_no_reask_when_every_span_is_verbatim(tmp_path):
    ex, backend = extractor(tmp_path, {FIRST: {"frames": [frame()]}})
    assert ex.extract(QUERY, [P0]).stats.reasked == 0 and len(backend.calls) == 1


def test_a_span_is_checked_against_its_own_passage_not_any_passage(tmp_path):
    cross = frame("p0", span="The cash withdrawal fee is waived for premium-tier cardholders.")  # p1's text
    ex, _ = extractor(tmp_path, {REASK: {"frames": []}, FIRST: {"frames": [cross]}})
    assert ex.extract(QUERY, [P0, P1]).frames == []


def test_whitespace_differences_do_not_make_a_span_non_verbatim(tmp_path):
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [frame(span="International  cash\nwithdrawals incur a flat 2.50 fee.")]}})
    assert len(ex.extract(QUERY, [P0]).frames) == 1


@pytest.mark.parametrize("mutation", [
    {"passage_id": "p9"}, {"passage_id": ""}, {"span": ""}, {"outcome": ""}, {"outcome": None},
])
def test_structurally_invalid_frames_are_dropped_as_other(tmp_path, mutation):
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [frame(**mutation)]}})
    out = ex.extract(QUERY, [P0])
    assert out.frames == [] and out.stats.dropped_other == 1 and out.stats.dropped_span == 0


def test_out_of_vocabulary_values_are_kept_untyped_and_counted(tmp_path):
    f = frame(decision={"event": "lounge entry", "location": "abroad", "decision_kind": "flat fee"})
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [f]}})
    out = ex.extract(QUERY, [P0])
    sig = out.frames[0].signature
    assert sig.untyped_slots == {"event": "lounge entry"} and "event" not in sig.slots
    assert out.stats.untyped_slots == 1 and out.stats.typed_slots == 2
    assert out.stats.oov_rate == pytest.approx(1 / 3)


def test_the_hedge_detector_overrides_the_model_toward_hedged_only(tmp_path):
    hedged_span = "A waiver may be granted for premium cardholders."
    p = Passage(id="p0", text=hedged_span, source_type=SourceType.PRODUCT_TERMS)
    f = frame(span=hedged_span, modality="statement", outcome="no fee applies")
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [f]}})
    out = ex.extract(QUERY, [p])
    assert out.frames[0].modality is Modality.HEDGED and out.stats.forced_hedged == 1


def test_the_model_claiming_hedged_is_respected_even_if_the_detector_sees_nothing(tmp_path):
    f = frame(modality="hedged")
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [f]}})
    out = ex.extract(QUERY, [P0])
    assert out.frames[0].modality is Modality.STATEMENT or out.frames[0].modality is Modality.HEDGED
    # the detector never *removes* a hedge; the model's own label stands when valid
    assert out.frames[0].modality is Modality.HEDGED


def test_an_invalid_modality_label_falls_back_to_the_detector(tmp_path):
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [frame(modality="maybe-ish")]}})
    assert ex.extract(QUERY, [P0]).frames[0].modality in (Modality.STATEMENT, Modality.OBLIGATION,
                                                           Modality.PERMISSION, Modality.PROHIBITION)


def test_typed_attributes_are_repaired_not_trusted(tmp_path):
    """The model says categorical and fills min_value: re-kinded, not dropped."""
    f = exception_frame(applicability={"descriptor": "older holders", "is_default": False, "attributes": [
        {"name": "age", "kind": "categorical", "values": None, "min_value": 65, "max_value": None}]})
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [f]}})
    attr = ex.extract(QUERY, [P0, P1]).frames[0].applicability.attributes[0]
    assert attr.kind.value == "numeric_range" and attr.min_value == 65


def test_a_default_frame_carries_no_attributes(tmp_path):
    f = frame(applicability={"descriptor": "everyone", "is_default": True,
                             "attributes": [{"name": "x", "kind": "categorical", "values": ["a"]}]})
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [f]}})
    assert ex.extract(QUERY, [P0]).frames[0].applicability.attributes == []


def test_duplicate_frames_collapse_and_order_is_canonical_not_the_models(tmp_path):
    ex1, _ = extractor(tmp_path, {FIRST: {"frames": [exception_frame(), frame(), frame()]}})
    out1 = ex1.extract(QUERY, [P0, P1])
    ex2, _ = extractor(tmp_path, {FIRST: {"frames": [frame(), exception_frame()]}})
    out2 = ex2.extract(QUERY, [P0, P1])
    assert [f.passage_id for f in out1.frames] == ["p0", "p1"] == [f.passage_id for f in out2.frames]
    assert out1.stats.n_kept == 2 and out1.stats.n_raw == 3


def test_garbage_replies_yield_no_frames_not_a_crash(tmp_path):
    for reply in ("not json at all", json.dumps({"frames": "nope"}), json.dumps([1, 2]), json.dumps({})):
        ex, _ = extractor(tmp_path, {FIRST: reply})
        out = ex.extract(QUERY, [P0])
        assert out.frames == [] and out.stats.n_raw == 0


def test_a_non_dict_frame_entry_is_ignored(tmp_path):
    ex, _ = extractor(tmp_path, {FIRST: {"frames": ["junk", 3, frame()]}})
    assert len(ex.extract(QUERY, [P0]).frames) == 1


def test_confidence_is_clamped(tmp_path):
    ex, _ = extractor(tmp_path, {FIRST: {"frames": [frame(confidence=7)]}})
    assert ex.extract(QUERY, [P0]).frames[0].confidence == 1.0


def test_stats_merge_and_rates_handle_zero():
    s = ExtractionStats()
    assert s.frame_yield == 0.0 and s.dropped_span_rate == 0.0 and s.oov_rate == 0.0
    a, b = ExtractionStats(n_passages=2, n_kept=3, calls=1), ExtractionStats(n_passages=4, n_kept=1, calls=2)
    m = a.merge(b)
    assert m.n_passages == 6 and m.n_kept == 4 and m.calls == 3 and m.frame_yield == pytest.approx(4 / 6)


def test_step_names_distinguish_the_reask_in_the_cost_log(tmp_path):
    assert STEP != STEP_REASK
    ex, _ = extractor(tmp_path, {REASK: {"frames": [frame()]}, FIRST: {"frames": [frame(span="nope")]}})
    ex.extract(QUERY, [P0])
    steps = [r.step for r in CostLog(tmp_path / "l.jsonl").read()]
    assert steps == [STEP, STEP_REASK]
