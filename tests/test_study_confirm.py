"""The confirmatory comparison: paired on shared questions, complete only on the full population."""

from study.analysis import QScore
from study.confirm import compare


def _s(qid: str, doc: int, f1c: float) -> QScore:
    return QScore(qid, "conditional", doc, 0.0, 0.0, 0.0, f1c, 1, 0, 0, 0, True, 0, 0.0)


def test_compare_pairs_shared_questions_and_flags_partial():
    a = {f"q{i}": _s(f"q{i}", i % 5, 0.5) for i in range(20)}
    b = {f"q{i}": _s(f"q{i}", i % 5, 0.2) for i in range(15)}
    pop = frozenset(f"q{i}" for i in range(20))
    r = compare({"a": a, "b": b}, "a", "b", pop)
    assert r["n"] == 15 and not r["complete"]
    assert abs(r["diff"] - 0.3) < 1e-9 and r["low"] > 0


def test_compare_missing_cell_or_no_overlap_is_none():
    a = {"q1": _s("q1", 0, 1.0)}
    assert compare({"a": a}, "a", "b", frozenset({"q1"})) is None
    assert compare({"a": a, "b": {"q2": _s("q2", 0, 0.0)}}, "a", "b", frozenset({"q1", "q2"})) is None


def test_compare_respects_population():
    a = {"q1": _s("q1", 0, 1.0), "q2": _s("q2", 1, 1.0)}
    b = {"q1": _s("q1", 0, 0.0), "q2": _s("q2", 1, 1.0)}
    r = compare({"a": a, "b": b}, "a", "b", frozenset({"q2"}))
    assert r["n"] == 1 and r["diff"] == 0.0 and r["complete"]
