"""The human loop: proposals become review items, and only a person's verdict changes the corpus."""

from __future__ import annotations

import json

import pytest

from api_budget.backends import MockBackend
from api_budget.cache import RequestCache
from api_budget.client import LLMClient
from api_budget.costlog import CostLog
from benchmark.human_loop import (
    apply_branches,
    apply_personas,
    main,
    merge,
    needs_branches,
    needs_personas,
    propose_branches,
    propose_persona_items,
)
from benchmark.preannotate import branch_provenance
from benchmark.review import Decision, ReviewStore
from contract.gold import GoldInstance
from contract.mock import generate


def base() -> GoldInstance:
    return next(i for i in generate(60, seed=0) if i.personas and i.gold_scoped_answer)


def unbranched(inst: GoldInstance | None = None) -> GoldInstance:
    d = (inst or base()).model_dump(mode="json")
    d.update(gold_branches=[], gold_scoped_answer=None, selection_answer=None, personas=[], persona_pairs=[])
    return GoldInstance.model_validate(d)


def unpersoned(inst: GoldInstance | None = None) -> GoldInstance:
    d = (inst or base()).model_dump(mode="json")
    d.update(personas=[], persona_pairs=[])
    return GoldInstance.model_validate(d)


def client_with(tmp_path, replies):
    it = iter(replies)

    class Seq(MockBackend):
        def complete(self, **kw):
            self.calls.append(kw)
            text = next(it)
            return self._wrap(text if isinstance(text, str) else json.dumps(text), kw["model"], "x")

    return LLMClient(cache=RequestCache(enabled=False), cost_log=CostLog(tmp_path / "l.jsonl"),
                     backend_override=Seq())


def branch_reply(inst):
    ids = [p.id for p in inst.passages]
    return {"branches": [
        {"is_default": True, "condition": None, "applies_to": "everyone else", "outcome": "a 3% fee applies",
         "source": ids[0]},
        {"is_default": False, "condition": "the holder has a premium-tier card", "applies_to": "premium holders",
         "outcome": "no fee applies", "source": ids[1]}]}


def dec(iid, verdict, edits=None, kind="branches", who="ann"):
    return Decision(iid, kind, who, verdict, edits=edits or {})


GOOD_PERSONAS = lambda inst: {  # noqa: E731
    "default": "I hold an ordinary current account and I send money abroad each month.",
    "exceptions": {inst.exception_branches[0].branch_id:
                   "I hold a premium current account and I send money abroad each month."},
    "irrelevant": {"attribute": "town of residence",
                   "statement": "I hold an ordinary current account and I send money abroad each month from my flat in Leeds."}}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_branches_are_proposed_only_for_conditional_refinements_without_them():
    assert needs_branches(unbranched()) and not needs_branches(base())
    not_conditional = [i for i in generate(120, seed=0) if i.gold_conflict_type.value != "conditional"]
    assert not_conditional and not any(needs_branches(i) for i in not_conditional)


def test_personas_are_proposed_only_where_branches_exist_and_personas_do_not():
    assert needs_personas(unpersoned()) and not needs_personas(base()) and not needs_personas(unbranched())


# --------------------------------------------------------------------------- #
# Proposing
# --------------------------------------------------------------------------- #


def test_branch_proposals_become_review_items_and_failures_are_listed_not_fatal(tmp_path):
    a, b = unbranched(), unbranched(next(i for i in generate(120, seed=1) if i.personas and i.gold_scoped_answer))
    b = b.model_copy(update={"instance_id": "other"})
    client = client_with(tmp_path, [branch_reply(a), "not json at all"])
    items, failures = propose_branches([a, b, base()], client=client)
    assert [i["instance_id"] for i in items] == [a.instance_id]
    assert items[0]["gold_branches"] and "model-proposed" in items[0]["annotation"]["notes"]
    assert list(failures) == ["other"]


def test_the_limit_counts_attempts_not_successes(tmp_path):
    a = unbranched()
    copies = [a.model_copy(update={"instance_id": f"x{i}"}) for i in range(4)]
    client = client_with(tmp_path, ["bad", "bad", "bad", "bad"])
    items, failures = propose_branches(copies, client=client, limit=2)
    assert not items and len(failures) == 2


def test_persona_proposals_carry_their_structural_problems(tmp_path):
    inst = unpersoned()
    ok_items, _ = propose_persona_items([inst], client=client_with(tmp_path, [GOOD_PERSONAS(inst)]))
    assert ok_items[0]["problems"] == [] and ok_items[0]["default"].startswith("I hold")
    bad = GOOD_PERSONAS(inst) | {"default": "branch is_default card_tier"}
    items, _ = propose_persona_items([inst], client=client_with(tmp_path, [bad, bad]))
    assert items[0]["problems"], "a proposal that fails the checks is shown with its problems"


# --------------------------------------------------------------------------- #
# Applying branch verdicts
# --------------------------------------------------------------------------- #


def proposed_item(tmp_path):
    inst = unbranched()
    items, _ = propose_branches([inst], client=client_with(tmp_path, [branch_reply(inst)]))
    return inst, items


def test_an_accepted_proposal_becomes_reviewed_gold(tmp_path):
    inst, items = proposed_item(tmp_path)
    out, rep = apply_branches(items, {inst.instance_id: dec(inst.instance_id, "accept")}, annotator="ann")
    assert rep.applied == [inst.instance_id] and branch_provenance(out[0]) == "reviewed:ann"
    assert out[0].gold_scoped_answer and out[0].selection_answer


def test_an_edit_is_applied_and_the_answers_are_rederived(tmp_path):
    inst, items = proposed_item(tmp_path)
    exc = next(b for b in items[0]["gold_branches"] if b["condition"] is not None)["branch_id"]
    d = dec(inst.instance_id, "edit", {exc: {"outcome": "the fee is refunded"}})
    out, _ = apply_branches(items, {inst.instance_id: d}, annotator="ann")
    assert "the fee is refunded" in out[0].gold_scoped_answer


def test_reject_skip_and_undecided_are_never_applied(tmp_path):
    inst, items = proposed_item(tmp_path)
    iid = inst.instance_id
    for decisions, bucket in (({iid: dec(iid, "reject")}, "rejected"), ({iid: dec(iid, "skip")}, "undecided"),
                              ({}, "undecided")):
        out, rep = apply_branches(items, decisions, annotator="ann")
        assert out == [] and getattr(rep, bucket) == [iid]


def test_a_second_annotator_can_only_hold_an_item_back(tmp_path):
    inst, items = proposed_item(tmp_path)
    iid = inst.instance_id
    mine = {iid: dec(iid, "accept")}
    for theirs in ({}, {iid: dec(iid, "reject", who="bob")}, {iid: dec(iid, "skip", who="bob")}):
        out, rep = apply_branches(items, mine, annotator="ann", second=theirs)
        assert out == [] and rep.disputed == [iid]
    out, rep = apply_branches(items, mine, annotator="ann", second={iid: dec(iid, "edit", {}, who="bob")})
    assert rep.applied == [iid]


def test_a_model_cannot_be_the_reviewer_and_a_bad_edit_is_refused_not_fatal(tmp_path):
    inst, items = proposed_item(tmp_path)
    iid = inst.instance_id
    _, rep = apply_branches(items, {iid: dec(iid, "accept")}, annotator="model")
    assert iid in rep.refused and not rep.applied
    _, rep2 = apply_branches(items, {iid: dec(iid, "edit", {"nope": {"outcome": "x"}})}, annotator="ann")
    assert iid in rep2.refused


# --------------------------------------------------------------------------- #
# Applying persona verdicts
# --------------------------------------------------------------------------- #


def persona_setup(tmp_path):
    inst = unpersoned()
    items, _ = propose_persona_items([inst], client=client_with(tmp_path, [GOOD_PERSONAS(inst)]))
    return inst, items


def test_accepted_personas_are_attached_with_counterfactual_pairs(tmp_path):
    inst, items = persona_setup(tmp_path)
    out, rep = apply_personas({inst.instance_id: inst}, items,
                              {inst.instance_id: dec(inst.instance_id, "accept", kind="personas")})
    assert rep.applied == [inst.instance_id]
    assert {p.persona_id for p in out[0].personas} >= {"u_default", "u_default_irrelevant"}
    assert {pp.flip for pp in out[0].persona_pairs} == {"relevant", "irrelevant"}


def test_an_edited_persona_is_revalidated_and_a_broken_edit_is_refused(tmp_path):
    inst, items = persona_setup(tmp_path)
    iid = inst.instance_id
    good = dec(iid, "edit", {"default": "I hold an ordinary current account and I send money abroad each week."},
               kind="personas")
    out, rep = apply_personas({iid: inst}, items, {iid: good})
    assert out and out[0].personas[0].statement.endswith("each week.")
    bad = dec(iid, "edit", {"default": "branch is_default card_tier"}, kind="personas")
    out2, rep2 = apply_personas({iid: inst}, items, {iid: bad})
    assert not out2 and iid in rep2.refused


def test_a_persona_item_for_an_unknown_instance_is_refused(tmp_path):
    inst, items = persona_setup(tmp_path)
    _, rep = apply_personas({}, items, {inst.instance_id: dec(inst.instance_id, "accept", kind="personas")})
    assert rep.refused == {inst.instance_id: "not in the corpus"}


# --------------------------------------------------------------------------- #
# Merge and the CLI
# --------------------------------------------------------------------------- #


def test_merge_replaces_by_id_preserving_order_and_rejects_strangers():
    a = generate(6, seed=0)
    changed = a[2].model_copy(update={"query": "changed?"})
    merged = merge(a, [changed])
    assert [m.instance_id for m in merged] == [x.instance_id for x in a] and merged[2].query == "changed?"
    with pytest.raises(ValueError, match="not in the corpus"):
        merge(a, [a[0].model_copy(update={"instance_id": "stranger"})])


def test_the_cli_applies_reviewed_branches_and_merges_without_overwriting(tmp_path, capsys):
    inst = unbranched()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(inst.model_dump(mode="json")) + "\n", encoding="utf-8")
    items = tmp_path / "items.jsonl"
    prop, _ = propose_branches([inst], client=client_with(tmp_path, [branch_reply(inst)]))
    items.write_text(json.dumps(prop[0]) + "\n", encoding="utf-8")
    ReviewStore(tmp_path / "rv", "branches", "ann").append(dec(inst.instance_id, "accept"))
    updated = tmp_path / "updated.jsonl"
    assert main(["apply-branches", "--items", str(items), "--annotator", "ann", "--review-dir", str(tmp_path / "rv"),
                 "--out", str(updated)]) == 0
    assert "1 applied" in capsys.readouterr().out
    assert main(["merge", "--data", str(corpus), "--updates", str(updated), "--out", str(corpus)]) == 2
    assert "must differ" in capsys.readouterr().err
    out = tmp_path / "corpus2.jsonl"
    assert main(["merge", "--data", str(corpus), "--updates", str(updated), "--out", str(out)]) == 0
    merged = GoldInstance.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert merged.gold_branches and branch_provenance(merged) == "reviewed:ann"
    assert not GoldInstance.model_validate(json.loads(corpus.read_text(encoding="utf-8"))).gold_branches
