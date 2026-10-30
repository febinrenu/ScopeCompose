"""The review loop: resumable, independent, timed -- and agreement that refuses non-independent annotators."""

from __future__ import annotations

import json

import pytest

from benchmark.annotation.agreement import AgreementError
from benchmark.review import (
    REJECT_REASONS,
    Decision,
    Item,
    ReviewStore,
    agreement,
    edit_item,
    item_id_for,
    labels,
    load_items,
    render_branches,
    render_candidate,
    render_personas,
    review,
    unique_shortcuts,
)


class Script:
    """A scripted prompter: answers come from a list, in order. ``None`` keeps a text default."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.shown: list[str] = []

    def show(self, text):
        self.shown.append(text)

    def choose(self, prompt, options):
        a = self.answers.pop(0)
        assert a in options, (a, options)
        return a

    def text(self, prompt, default=""):
        a = self.answers.pop(0)
        return default if a is None else a


def cand(i, conf=0.9):
    p = {"rule": f"Rule {i} says you must pay a fee.", "exception": f"Exception {i} says some do not pay.",
         "rule_doc_id": f"r{i}", "exception_doc_id": f"e{i}", "why": "some group", "confidence": conf}
    return Item(item_id_for(p["rule"], p["exception"]), "candidates", p)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 5.0
        return self.t


def store(tmp_path, annotator="alice", kind="candidates"):
    return ReviewStore(tmp_path, kind, annotator)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_each_verdict_is_recorded_with_its_reason_and_time(tmp_path):
    st = store(tmp_path)
    s = review([cand(1), cand(2), cand(3)], st, Script(["accept", "reject", "unclear", "skip"]), clock=Clock())
    assert (s.shown, s.accepted, s.rejected, s.skipped) == (3, 1, 1, 1)
    got = st.load()
    assert [d.verdict for d in got.values()] == ["accept", "reject", "skip"]
    assert got[cand(2).item_id].reason == "unclear" and all(d.seconds == 5.0 for d in got.values())
    assert all(d.annotator == "alice" and d.ts for d in got.values())
    assert st.hours() == pytest.approx(15 / 3600)


def test_a_decided_item_is_never_shown_again_and_a_rerun_resumes(tmp_path):
    st = store(tmp_path)
    review([cand(1), cand(2)], st, Script(["accept", "quit"]))
    sc = Script(["accept"])
    s = review([cand(1), cand(2)], st, sc)
    assert s.already_done == 1 and s.shown == 1 and len(sc.shown) == 1 and "Rule 2" in sc.shown[0]


def test_quitting_records_nothing_for_the_item_on_screen(tmp_path):
    st = store(tmp_path)
    s = review([cand(1), cand(2)], st, Script(["accept", "quit"]))
    assert s.quit_early and list(st.load()) == [cand(1).item_id]


def test_the_limit_stops_a_session(tmp_path):
    st = store(tmp_path)
    s = review([cand(i) for i in range(5)], st, Script(["accept"] * 5), limit=2)
    assert s.shown == 2 and len(st.load()) == 2


def test_candidates_offer_no_edit_and_other_kinds_do(tmp_path):
    seen = []

    class Spy(Script):
        def choose(self, prompt, options):
            seen.append(options)
            return super().choose(prompt, options)

    review([cand(1)], store(tmp_path), Spy(["accept"]))
    assert "edit" not in seen[0]
    item = Item("x", "branches", {"query": "q", "gold_branches": []})
    review([item], ReviewStore(tmp_path, "branches", "alice"), Spy(["accept"]))
    assert "edit" in seen[1]


def test_a_half_written_last_line_does_not_lose_the_earlier_decisions(tmp_path):
    st = store(tmp_path)
    review([cand(1), cand(2)], st, Script(["accept", "accept"]))
    with st.path.open("a", encoding="utf-8") as fh:
        fh.write('{"item_id": "zzz", "kind": "cand')          # killed mid-write
    assert len(st.load()) == 2


def test_the_latest_decision_for_an_item_wins(tmp_path):
    st = store(tmp_path)
    st.append(Decision("a", "candidates", "alice", "accept"))
    st.append(Decision("a", "candidates", "alice", "reject", "unclear"))
    assert st.load()["a"].verdict == "reject"


def test_store_validates_its_inputs(tmp_path):
    for bad in ("", "  ", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            ReviewStore(tmp_path, "candidates", bad)
    with pytest.raises(ValueError):
        ReviewStore(tmp_path, "nonsense", "alice")


# --------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------- #


def test_editing_branches_records_only_what_changed():
    item = Item("i", "branches", {"gold_branches": [
        {"branch_id": "b0", "condition": None, "outcome": "a 3% fee applies"},
        {"branch_id": "b1", "condition": "premium", "outcome": "no fee applies"}]})
    edits = edit_item(item, Script([None, None, "a 1% fee applies", "premium or platinum"]))
    assert edits == {"b1": {"outcome": "a 1% fee applies", "condition": "premium or platinum"}}


def test_editing_personas_records_only_what_changed():
    item = Item("i", "personas", {"default": "I hold a standard card.", "exceptions": {"b1": "I hold a premium card."},
                                  "irrelevant": "I hold a standard card in Leeds."})
    edits = edit_item(item, Script([None, "I hold a platinum card.", None]))
    assert edits == {"exceptions": {"b1": "I hold a platinum card."}}


def test_an_edit_verdict_stores_the_edits_and_counts_as_accept_for_agreement(tmp_path):
    st = ReviewStore(tmp_path, "personas", "alice")
    item = Item("p1", "personas", {"default": "d", "exceptions": {"b1": "e"}, "irrelevant": "i"})
    review([item], st, Script(["edit", None, "new exception", None]))
    d = st.load()["p1"]
    assert d.verdict == "edit" and d.edits == {"exceptions": {"b1": "new exception"}}
    assert labels(st) == {"p1": "accept"}


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def decide(tmp_path, annotator, verdicts):
    st = store(tmp_path, annotator)
    answers = []
    for v in verdicts:
        answers += [v, "unclear"] if v == "reject" else [v]
    review([cand(i) for i in range(len(verdicts))], st, Script(answers))
    return st


def test_agreement_is_cohens_kappa_over_accept_reject(tmp_path):
    a = decide(tmp_path, "alice", ["accept", "accept", "reject", "reject", "accept", "reject"])
    b = decide(tmp_path, "bob", ["accept", "accept", "reject", "accept", "accept", "reject"])
    k = agreement(a, b)
    assert k.n == 6 and k.observed_agreement == pytest.approx(5 / 6)
    assert k.kappa == pytest.approx(0.6667, abs=1e-3)


def test_a_skip_is_missing_data_not_a_disagreement(tmp_path):
    a = decide(tmp_path, "alice", ["accept", "skip", "reject", "accept"])
    b = decide(tmp_path, "bob", ["accept", "accept", "reject", "reject"])
    assert agreement(a, b).n == 3


def test_the_same_person_twice_is_refused(tmp_path):
    a = decide(tmp_path, "alice", ["accept", "reject"])
    with pytest.raises(AgreementError, match="INDEPENDENT"):
        agreement(a, store(tmp_path, "alice"))


def test_a_model_id_is_refused_as_an_annotator(tmp_path):
    a = decide(tmp_path, "alice", ["accept", "reject"])
    m = decide(tmp_path, "model-proposed", ["accept", "reject"])
    with pytest.raises(AgreementError, match="model"):
        agreement(a, m)


def test_agreement_compares_one_kind_at_a_time(tmp_path):
    with pytest.raises(ValueError, match="one kind"):
        agreement(store(tmp_path, "alice"), ReviewStore(tmp_path, "personas", "bob"))


# --------------------------------------------------------------------------- #
# Loading and rendering
# --------------------------------------------------------------------------- #


def test_candidates_load_deduplicated_and_most_confident_first(tmp_path):
    p = tmp_path / "props.json"
    dup = cand(1, 0.5).payload
    p.write_text(json.dumps({"proposals": [dup, cand(2, 0.95).payload, dup, cand(3, 0.7).payload]}), encoding="utf-8")
    items = load_items("candidates", p)
    assert len(items) == 3 and [i.payload["confidence"] for i in items] == [0.95, 0.7, 0.5]
    assert all(len(i.item_id) == 12 for i in items)


def test_other_kinds_load_one_item_per_line(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps({"instance_id": "x1"}) + "\n" + json.dumps({"instance_id": "x2"}) + "\n", encoding="utf-8")
    assert [i.item_id for i in load_items("branches", p)] == ["x1", "x2"]
    with pytest.raises(ValueError):
        load_items("nonsense", p)


def test_renderers_show_what_a_decision_needs():
    assert "RULE" in render_candidate(cand(1)) and "DIFFERENT outcome" in render_candidate(cand(1))
    b = Item("i", "branches", {"query": "Do I pay?", "gold_branches": [
        {"branch_id": "b0", "condition": None, "outcome": "fee", "supporting_passage": "p0"}]})
    assert "Do I pay?" in render_branches(b) and "default" in render_branches(b)
    pp = Item("i", "personas", {"default": "d", "exceptions": {"b1": "e"}, "irrelevant_attribute": "town",
                                "irrelevant": "x", "problems": ["too short"]})
    assert "too short" in render_personas(pp) and "town" in render_personas(pp)


def test_shortcuts_are_only_offered_when_unambiguous():
    assert unique_shortcuts(REJECT_REASONS) == {"s": "same_outcome", "u": "unclear"}
    assert "n" not in unique_shortcuts(REJECT_REASONS)
