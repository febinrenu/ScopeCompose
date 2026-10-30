"""O3: parse + frames + partition composition, repaired against the parse -- on a scripted backend."""

from __future__ import annotations

import json

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from baselines.registry import StructuredCompact
from contract.models import Passage, SourceType
from pipelines.scopecompose import ScopeCompose

P0 = Passage(id="p0", text="Transfers incur a 3% fee. Contact us for details.", source_type=SourceType.OFFICIAL_POLICY)
P1 = Passage(id="p1", text="The transfer fee is waived for premium-tier cardholders.",
             source_type=SourceType.PRODUCT_TERMS)
Q = "Do I pay a fee on transfers?"

PARSE = "EVERY distinct outcome"       # only in the parse system prompt
FRAMES = "NORMATIVE FRAMES"            # only in the frame-extraction system prompt

SIG = {"event": "transfer", "decision_kind": "flat fee"}


def fr(pid, span, outcome, *, default=False, attrs=None, desc=None, decision=None, modality="statement"):
    return {"passage_id": pid, "span": span, "decision": decision or SIG, "outcome": outcome, "modality": modality,
            "applicability": {"descriptor": desc or ("everyone" if default else "premium-tier cardholders"),
                              "is_default": default, "attributes": attrs or []}}


RULE = fr("p0", "Transfers incur a 3% fee.", "a 3% fee applies", default=True)
EXC = fr("p1", "The transfer fee is waived for premium-tier cardholders.", "no fee applies",
         attrs=[{"name": "card_tier", "kind": "categorical", "values": ["premium"]}])

DRAFT_FULL = {"branches": [
    {"is_default": True, "condition": None, "applies_to": "everyone", "outcome": "a 3% fee applies", "source": "p0"},
    {"is_default": False, "condition": "premium card", "applies_to": "premium", "outcome": "no fee applies",
     "source": "p1"}]}
DRAFT_MISSES_EXCEPTION = {"branches": [DRAFT_FULL["branches"][0]]}


def system(tmp_path, parse, frames, domain="banking"):
    backend = MockBackend(scripted={PARSE: json.dumps(parse), FRAMES: json.dumps({"frames": frames})})
    client = LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                       backend_override=backend)
    return ScopeCompose(tier=Tier.BULK, domain=domain, client=client), backend


def test_happy_path_composes_a_rule_and_its_exception_with_attachment(tmp_path):
    o3, _ = system(tmp_path, DRAFT_FULL, [RULE, EXC])
    out = o3.answer(Q, [P0, P1])
    assert out.text == ("A 3% fee applies (per p0). However, if premium-tier cardholders, "
                        "no fee applies (per p1).")
    assert out.trace["components"][0]["regions"] == 2 and "fallback" not in out.trace


def test_cost_is_metered_across_both_calls(tmp_path):
    o3, backend = system(tmp_path, DRAFT_FULL, [RULE, EXC])
    out = o3.answer(Q, [P0, P1])
    assert out.calls == 2 == len(backend.calls)
    assert out.prompt_tokens > 0 and out.completion_tokens > 0


def test_a_parse_that_misses_the_exception_is_repaired_and_the_repair_is_logged(tmp_path):
    """The point of the hybrid: suppression by the parse is caught by the frames."""
    o3, _ = system(tmp_path, DRAFT_MISSES_EXCEPTION, [RULE, EXC])
    out = o3.answer(Q, [P0, P1])
    assert "no fee applies (per p1)" in out.text
    assert out.trace["injected"] == ["no fee applies"]
    # ... whereas B7c alone, on the same parse, would have suppressed it
    b7 = StructuredCompact(tier=Tier.BULK, client=o3.client).answer(Q, [P0, P1])
    assert "no fee applies" not in b7.text


def test_a_draft_branch_no_frame_supports_is_kept_and_logged_as_parse_only(tmp_path):
    draft = {"branches": DRAFT_FULL["branches"] + [
        {"is_default": False, "condition": "the holder is a student", "applies_to": "students",
         "outcome": "a 1% fee applies", "source": "p0"}]}
    o3, _ = system(tmp_path, draft, [RULE, EXC])
    out = o3.answer(Q, [P0, P1])
    assert "a 1% fee applies" in out.text and out.trace["parse_only"] == ["a 1% fee applies"]


def test_no_frames_falls_back_to_the_systems_own_parse(tmp_path):
    o3, _ = system(tmp_path, DRAFT_FULL, [])
    out = o3.answer(Q, [P0, P1])
    assert "fallback" in out.trace and "no_frames" in out.trace["fallback"]
    assert "no fee applies" in out.text, "never worse than the parse because a second call failed"


def test_distinct_decisions_are_never_composed_together(tmp_path):
    """A flat fee and a percentage fee are different variables: no exception relation between them,
    and the verbaliser must not present one as the other's exception."""
    pct = fr("p1", "The transfer fee is waived for premium-tier cardholders.", "no fee applies",
             decision={"event": "transfer", "decision_kind": "percentage fee"},
             attrs=[{"name": "card_tier", "kind": "categorical", "values": ["premium"]}])
    o3, _ = system(tmp_path, {"branches": []}, [RULE, pct])
    out = o3.answer(Q, [P0, P1])
    assert len(out.trace["components"]) == 2
    assert out.text == "A 3% fee applies (per p0). If premium-tier cardholders, no fee applies (per p1)."
    assert "However" not in out.text


def test_hedged_frames_are_stated_verbatim_and_never_assigned_an_outcome(tmp_path):
    hedged = fr("p1", "The transfer fee may be waived for premium-tier cardholders.", "no fee applies",
                modality="hedged")
    p1 = Passage(id="p1", text="The transfer fee may be waived for premium-tier cardholders.",
                 source_type=SourceType.PRODUCT_TERMS)
    o3, _ = system(tmp_path, {"branches": []}, [RULE, hedged])
    out = o3.answer(Q, [P0, p1])
    assert out.trace["hedged"] == ["p1"]
    assert out.text == ("A 3% fee applies (per p0). "
                        "The transfer fee may be waived for premium-tier cardholders (per p1).")


def test_equal_scope_disagreement_is_reported_as_contested_not_picked(tmp_path):
    a = fr("p0", "Transfers incur a 3% fee.", "a 3% fee applies",
           attrs=[{"name": "card_tier", "kind": "categorical", "values": ["premium"]}], desc="premium")
    b = fr("p1", "The transfer fee is waived for premium-tier cardholders.", "no fee applies",
           attrs=[{"name": "card_tier", "kind": "categorical", "values": ["premium"]}], desc="premium")
    o3, _ = system(tmp_path, {"branches": []}, [a, b])
    out = o3.answer(Q, [P0, P1])
    assert out.trace["components"][0]["contested"] == 1
    assert "the sources disagree" in out.text and "p0 states a 3% fee applies" in out.text
    assert "p1 states no fee applies" in out.text


def test_an_untyped_scope_is_shown_verbatim_with_its_condition(tmp_path):
    odd = fr("p1", "The transfer fee is waived for premium-tier cardholders.", "no fee applies",
             desc="in exceptional hardship",
             attrs=[{"name": "circumstances", "kind": "free_text", "text": "in exceptional hardship"}])
    o3, _ = system(tmp_path, {"branches": []}, [RULE, odd])
    out = o3.answer(Q, [P0, P1])
    assert out.trace["components"][0]["indeterminate"] == 1
    assert "in exceptional hardship" in out.text and "no fee applies (per p1)" in out.text


def test_the_answer_does_not_depend_on_passage_order(tmp_path):
    o3, _ = system(tmp_path, DRAFT_FULL, [EXC, RULE])
    a = o3.answer(Q, [P0, P1]).text
    o3b, _ = system(tmp_path, DRAFT_FULL, [RULE, EXC])
    assert o3b.answer(Q, [P1, P0]).text == a


def test_an_unparseable_parse_does_not_stop_composition_from_frames(tmp_path):
    o3, _ = system(tmp_path, "not json", [RULE, EXC])
    out = o3.answer(Q, [P0, P1])
    assert out.trace["parse_ok"] is False and "no fee applies (per p1)" in out.text
    assert out.trace["injected"] == ["a 3% fee applies", "no fee applies"]
