"""The system registry: B0, B3, B7c, B8 behind one interface, on a scripted backend."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient, Tier
from api_budget.costlog import CostLog
from baselines.registry import (
    ANSWER_SYSTEM,
    SYSTEMS,
    ConflictGroup,
    ConflictRAGStyle,
    FullContext,
    StructuredCompact,
    StructuredVerified,
    answer_user_prompt,
    make_system,
    oracle_conflicts,
    parse_branches,
    prompts_sha,
    verbalise_branches,
)
from contract.gold import Applicability, Branch
from contract.mock import generate
from contract.models import ConflictType, Passage, SourceType

RULE = Passage(id="p0", text="Transfers incur a 3% fee.", source_type=SourceType.OFFICIAL_POLICY, date="2024-01-01")
EXC = Passage(id="p1", text="The fee is waived for premium cardholders.", source_type=SourceType.PRODUCT_TERMS,
              date="2025-01-01")
DISTRACTOR = Passage(id="p2", text="Our branches open at nine.", source_type=SourceType.FAQ)
Q = "Do I pay a fee on transfers?"


def client_with(tmp_path, scripted=None) -> tuple[LLMClient, MockBackend]:
    backend = MockBackend(scripted={k: v if isinstance(v, str) else json.dumps(v)
                                    for k, v in (scripted or {}).items()})
    return (LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                      backend_override=backend), backend)


BRANCHES = {"branches": [
    {"is_default": True, "condition": None, "applies_to": "everyone", "outcome": "a 3% fee applies", "source": "p0"},
    {"is_default": False, "condition": "the cardholder holds a premium card", "applies_to": "premium cardholders",
     "outcome": "no fee applies", "source": "p1"},
]}


# --------------------------------------------------------------------------- #
# Uniformity
# --------------------------------------------------------------------------- #


def test_every_system_shares_one_input_shape_and_registry():
    assert set(SYSTEMS) == {"B0", "B3", "B7c", "B8"}
    for sid in SYSTEMS:
        assert make_system(sid, tier=Tier.BULK).id == sid
    with pytest.raises(KeyError, match="unknown system"):
        make_system("B99", tier=Tier.BULK)


def test_the_answer_prompt_is_neutral_about_conditions():
    """B0 must not be told to look for exceptions: that would be a friendlier prompt than the
    comparison systems get, or a strawman if it were hostile."""
    for word in ("condition", "exception", "depends", "who"):
        assert word not in ANSWER_SYSTEM.lower()


def test_prompts_are_hashed_and_the_hash_changes_with_a_prompt(monkeypatch):
    import baselines.registry as r

    before = prompts_sha()
    assert len(before) == 64
    monkeypatch.setattr(r, "ANSWER_SYSTEM", r.ANSWER_SYSTEM + " Be brief.")
    assert prompts_sha() != before


# --------------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------------- #


def test_b0_sends_every_passage_in_one_call_and_meters_tokens(tmp_path):
    c, backend = client_with(tmp_path, {"Question: Do I pay": "A 3% fee applies [p0]."})
    out = FullContext(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC, DISTRACTOR])
    assert out.text == "A 3% fee applies [p0]." and out.calls == 1 and len(backend.calls) == 1
    sent = backend.calls[0]["messages"][0]["content"]
    assert all(f"[{p.id}]" in sent for p in (RULE, EXC, DISTRACTOR))
    assert out.prompt_tokens > 0 and out.completion_tokens > 0 and out.total_tokens == out.prompt_tokens + out.completion_tokens
    assert backend.calls[0]["system"] == ANSWER_SYSTEM


# --------------------------------------------------------------------------- #
# B3
# --------------------------------------------------------------------------- #


def test_oracle_conflicts_map_conditional_onto_factual():
    inst = next(i for i in generate(60, seed=0) if i.gold_conflict_type is ConflictType.CONDITIONAL)
    (g,) = oracle_conflicts(inst)
    assert g.type == "factual" and len(g.passage_ids) == 2


def test_oracle_conflicts_are_empty_without_a_conflict():
    inst = next(i for i in generate(120, seed=0) if i.gold_conflict_type is ConflictType.NO_CONFLICT)
    assert oracle_conflicts(inst) == []


# A clear credibility winner (official + recent vs third-party + old) and an exact tie. With two
# alternatives entropy weighting puts all weight on the criteria that discriminate, so margins
# are either ~0 (a trade-off) or ~1 (one dominates); these fixtures are the two extremes.
WINNER = Passage(id="p0", text="Transfers incur a 3% fee.", source_type=SourceType.OFFICIAL_POLICY, date="2025-06-01")
LOSER = Passage(id="p1", text="The fee is waived.", source_type=SourceType.THIRD_PARTY, date="2023-01-01")
TIE_A = Passage(id="p0", text="The fee applies.", source_type=SourceType.PRODUCT_TERMS, date="2025-01-01")
TIE_B = Passage(id="p1", text="The fee is waived.", source_type=SourceType.PRODUCT_TERMS, date="2025-01-01")


def test_b3_selects_one_passage_and_the_excluded_exception_never_reaches_the_generator(tmp_path):
    """The failure under study: with a clear credibility winner, the exception is deleted."""
    c, backend = client_with(tmp_path)
    out = ConflictRAGStyle(tier=Tier.BULK, client=c).answer(
        Q, [WINNER, LOSER, DISTRACTOR], conflicts=[ConflictGroup(("p0", "p1"), "factual")])
    sent = backend.calls[0]["messages"][0]["content"]
    d = out.trace["decisions"][0]
    assert d["kind"] == "selected" and d["kept"] == ["p0"] and d["dropped"] == ["p1"]
    assert "[p1]" not in sent, "the dropped passage must not be visible to the generator"
    assert "[p0]" in sent and "[p2]" in sent, "the winner and a non-conflicting passage pass through"


def test_b3_presents_both_when_the_evidence_is_insufficient(tmp_path):
    c, backend = client_with(tmp_path)
    out = ConflictRAGStyle(tier=Tier.BULK, client=c).answer(
        Q, [TIE_A, TIE_B], conflicts=[ConflictGroup(("p0", "p1"), "factual")])
    assert out.trace["decisions"][0]["kind"] == "both"
    sent = backend.calls[0]["messages"][0]["content"]
    assert "[p0]" in sent and "[p1]" in sent


def test_a_rule_and_its_exception_can_tie_so_b3_does_not_always_delete_one():
    """Authority favours the rule, recency and specificity the exception: with the specificity
    criterion the margin is exactly 0 (B3 presents both); ablated, it selects the newest. Whether
    B3 suppresses the exception is therefore an empirical question per pair, not a given."""
    from baselines.conflictrag_topsis import select_by_credibility

    assert select_by_credibility([RULE, EXC]).kind == "both"
    assert select_by_credibility([RULE, EXC], include_specificity=False).chosen == ("p1",)


def test_b3_resolves_temporal_by_recency_and_keeps_every_opinion(tmp_path):
    c, backend = client_with(tmp_path)
    b3 = ConflictRAGStyle(tier=Tier.BULK, client=c)
    t = b3.answer(Q, [RULE, EXC], conflicts=[ConflictGroup(("p0", "p1"), "temporal")])
    assert t.trace["decisions"][0]["kind"] == "recency" and t.trace["selected"] == ["p1"]
    o = b3.answer(Q, [RULE, EXC], conflicts=[ConflictGroup(("p0", "p1"), "opinion")])
    assert o.trace["selected"] == ["p0", "p1"] and o.trace["decisions"][0]["kind"] == "synthesis"


def test_b3_without_conflicts_is_b0(tmp_path):
    c, backend = client_with(tmp_path)
    out = ConflictRAGStyle(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC, DISTRACTOR])
    assert out.trace["selected"] == ["p0", "p1", "p2"]


def test_b3_is_independent_of_retrieval_order(tmp_path):
    c, _ = client_with(tmp_path)
    g = [ConflictGroup(("p0", "p1"), "factual")]
    a = ConflictRAGStyle(tier=Tier.BULK, client=c, delta=0.05).answer(Q, [RULE, EXC, DISTRACTOR], conflicts=g)
    b = ConflictRAGStyle(tier=Tier.BULK, client=c, delta=0.05).answer(Q, [DISTRACTOR, EXC, RULE], conflicts=g)
    assert sorted(a.trace["selected"]) == sorted(b.trace["selected"])


def test_b3_ablating_specificity_is_a_switch(tmp_path):
    c, _ = client_with(tmp_path)
    assert ConflictRAGStyle(tier=Tier.BULK, client=c, include_specificity=False).include_specificity is False


# --------------------------------------------------------------------------- #
# Branch parsing and the verbaliser
# --------------------------------------------------------------------------- #


def test_parse_branches_validates_and_drops_unknown_citations():
    raw = {"branches": [
        {"is_default": True, "outcome": "a 3% fee applies", "source": "p0"},
        {"is_default": False, "condition": "premium", "outcome": "no fee applies", "source": "p99"},
        {"is_default": True, "condition": "second default", "outcome": "demoted"},
        {"outcome": ""}, "junk", {"is_default": False, "outcome": "no condition given"},
    ]}
    bs = parse_branches(raw, {"p0", "p1"})
    assert [b.outcome for b in bs] == ["a 3% fee applies", "no fee applies", "demoted", "no condition given"]
    assert bs[1].supporting_passage is None, "an unknown citation is dropped, not remapped"
    assert sum(b.is_default for b in bs) == 1, "a second default is demoted"
    assert bs[3].condition == "unspecified"


def test_parse_branches_tolerates_garbage():
    for junk in (None, [], {}, {"branches": "x"}, {"branches": [1, 2]}):
        assert parse_branches(junk, set()) == []


def test_verbaliser_attaches_each_outcome_to_who_it_is_for():
    bs = parse_branches(BRANCHES, {"p0", "p1"})
    text = verbalise_branches(bs)
    assert text == ("A 3% fee applies (per p0). However, if the cardholder holds a premium card, "
                    "no fee applies (per p1).")


def test_verbaliser_without_a_default_and_with_nothing():
    only = parse_branches({"branches": [{"is_default": False, "condition": "a", "outcome": "x", "source": "p0"},
                                        {"is_default": False, "condition": "b", "outcome": "y", "source": "p1"}]},
                          {"p0", "p1"})
    assert verbalise_branches(only) == "If a, x (per p0). If b, y (per p1)."
    assert verbalise_branches([]) == ""


# --------------------------------------------------------------------------- #
# B7c and B8
# --------------------------------------------------------------------------- #


def test_b7c_parses_then_verbalises_in_one_call(tmp_path):
    c, backend = client_with(tmp_path, {"Question: Do I pay": BRANCHES})
    out = StructuredCompact(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC])
    assert out.calls == 1 and out.trace == {"parse_ok": True, "n_draft": 2, "n_final": 2}
    assert "However, if the cardholder holds a premium card" in out.text and len(out.branches) == 2
    assert backend.calls[0]["response_format"] == {"type": "json_object"}


def test_b7c_records_a_failed_parse_instead_of_inventing_an_answer(tmp_path):
    c, _ = client_with(tmp_path, {"Question: Do I pay": "no json"})
    out = StructuredCompact(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC])
    assert out.text == "" and out.trace["parse_ok"] is False and out.branches == ()


def test_b8_adds_a_verification_call_and_uses_its_correction(tmp_path):
    fixed = {"branches": BRANCHES["branches"] + [
        {"is_default": False, "condition": "the cardholder is a student", "outcome": "a 1% fee applies",
         "source": "p1"}]}
    c, backend = client_with(tmp_path, {"Draft branches:": fixed, "Question: Do I pay": BRANCHES})
    out = StructuredVerified(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC])
    assert out.calls == 2 and len(backend.calls) == 2
    assert out.trace["n_draft"] == 2 and out.trace["n_final"] == 3 and "student" in out.text


def test_b8_never_loses_the_draft_to_an_unparseable_verification(tmp_path):
    c, _ = client_with(tmp_path, {"Draft branches:": "garbage", "Question: Do I pay": BRANCHES})
    out = StructuredVerified(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC])
    assert out.trace["n_final"] == 2 and out.calls == 2


def test_b8_skips_verification_when_there_is_nothing_to_verify(tmp_path):
    c, backend = client_with(tmp_path, {"Question: Do I pay": "no json"})
    out = StructuredVerified(tier=Tier.BULK, client=c).answer(Q, [RULE, EXC])
    assert out.calls == 1 and len(backend.calls) == 1


def test_answer_prompt_lists_every_passage_with_its_id():
    p = answer_user_prompt(Q, [RULE, EXC])
    assert p.startswith("Question: Do I pay a fee on transfers?") and "[p0] Transfers" in p and "[p1] The fee" in p
