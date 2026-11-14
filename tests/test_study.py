"""The condition-loss study: data plumbing, scoring and the generator contract.

Tests that need the downloaded benchmarks skip when ``data/external`` is absent
(run ``python -m study.fetch_data``).
"""

from __future__ import annotations

import pytest

from study import reader, scoring
from study.fetch_data import path_of

HAVE_CQA = path_of("conditionalqa", "documents.json").exists()
HAVE_RAM = path_of("ramdocs", "RAMDocs_test.jsonl").exists()
needs_cqa = pytest.mark.skipif(not HAVE_CQA, reason="ConditionalQA not downloaded")
needs_ram = pytest.mark.skipif(not HAVE_RAM, reason="RAMDocs not downloaded")


# --------------------------------------------------------------------------- #
# The official scorer
# --------------------------------------------------------------------------- #


def test_gold_scores_full_marks():
    ref = [["yes", ["<p>You must be 18 or over.</p>"]]]
    s = scoring.score([("yes", ["<p>You must be 18 or over.</p>"])], ref)
    assert (s.em, s.em_cond, s.f1, s.f1_cond) == (1.0, 1.0, 1.0, 1.0)


def test_a_condition_where_gold_has_none_zeroes_the_conditional_score():
    s = scoring.score([("yes", ["<p>anything</p>"])], [["yes", []]])
    assert s.f1 == 1.0 and s.f1_cond == 0.0


def test_an_omitted_condition_zeroes_the_conditional_score():
    s = scoring.score([("yes", [])], [["yes", ["<p>c</p>"]]])
    assert s.f1 == 1.0 and s.f1_cond == 0.0


def test_an_extra_answer_is_discounted():
    s = scoring.score([("yes", []), ("no", [])], [["yes", []]])
    assert s.f1 == pytest.approx(1.0 * 2.718281828 ** (1 - 2), rel=1e-6)


def test_scoring_does_not_mutate_the_prediction():
    pred = [("yes", ["<p>c</p>"])]
    scoring.score(pred, [["yes", ["<p>c</p>"]], ["no", []]])
    assert pred == [("yes", ["<p>c</p>"])]


@needs_cqa
def test_every_gold_reference_scores_full_marks_against_itself():
    from study import cqa

    for q in cqa.questions():
        if q.kind == "unanswerable":
            assert scoring.score([], q.reference()).f1_cond == 1.0
            continue
        s = scoring.score([(a, list(c)) for a, c in q.answers], q.reference())
        assert s.f1_cond == pytest.approx(1.0), q.id


# --------------------------------------------------------------------------- #
# The generator's output contract
# --------------------------------------------------------------------------- #

EV = reader.Evidence("x", {1: "<p>rule</p>", 2: "<li>condition</li>"})


def test_conditions_map_back_to_exact_elements():
    p = reader.parse_output('{"answers": [{"answer": "yes", "evidence": [1], "conditions": [2]}]}', EV)
    assert p.ok and p.answers == (("yes", ("<li>condition</li>",)),) and p.evidence == ("<p>rule</p>",)


def test_a_citation_to_no_line_is_dropped_and_counted():
    p = reader.parse_output('{"answers": [{"answer": "no", "conditions": [7, "x"]}]}', EV)
    assert p.answers == (("no", ()),) and p.bad_citations == 2


def test_json_wrapped_in_prose_is_read():
    p = reader.parse_output('Sure.\n```json\n{"answers": [{"answer": "yes", "conditions": []}]}\n```', EV)
    assert p.ok and p.answers == (("yes", ()),)


def test_unparseable_output_is_an_empty_answer_not_an_error():
    p = reader.parse_output("I think the answer is yes.", EV)
    assert not p.ok and p.answers == ()


def test_no_more_answers_than_the_contract_allows():
    many = ", ".join('{"answer": "a%d", "conditions": []}' % i for i in range(9))
    assert len(reader.parse_output('{"answers": [%s]}' % many, EV).answers) == scoring.MAX_ANSWERS


def test_the_worked_examples_obey_the_contract():
    import json

    for _, reply in reader.DEMOS:
        items = json.loads(reply)["answers"]
        assert items and all({"answer", "evidence", "conditions"} <= set(i) for i in items)


@needs_cqa
def test_worked_examples_come_from_the_tuning_set_only():
    from study import cqa

    tuning = {q.id for q in cqa.tuning_questions()}
    assert reader.DEMO_QUESTION_IDS <= tuning
    evaluated_pages = {q.doc for q in cqa.eval_questions()}
    idx = cqa.by_id()
    assert not {idx[i].doc for i in reader.DEMO_QUESTION_IDS} & evaluated_pages


# --------------------------------------------------------------------------- #
# Chunks and gold locations
# --------------------------------------------------------------------------- #


@needs_cqa
@pytest.mark.parametrize("words, strategy", [(150, "fixed"), (100, "fixed"), (150, "section")])
def test_chunks_partition_every_page_into_whole_elements(words, strategy):
    from study import cqa

    by_doc = cqa.chunks_by_doc(words, strategy)
    for d in cqa.documents():
        got = [e for c in by_doc[d.index] for e in c.elements]
        assert tuple(got) == d.elements
        spans = [(c.start, c.end) for c in by_doc[d.index]]
        assert spans[0][0] == 0 and spans[-1][1] == len(d.elements)
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


@needs_cqa
def test_every_gold_element_is_located():
    from study import cqa

    for q in cqa.questions():
        assert not cqa.locate_gold(q).unlocated, q.id


@needs_cqa
def test_evaluation_and_tuning_pages_are_disjoint():
    from study import cqa

    assert not {q.doc for q in cqa.eval_questions()} & {q.doc for q in cqa.tuning_questions()}


@needs_cqa
def test_the_qualifier_folds_never_score_a_page_they_trained_on():
    from study import cqa, qualifier

    idx = cqa.by_id()
    plan = qualifier.fold_plan()
    for fold, scored in plan.items():
        trained = {idx[i].doc for i in qualifier.train_ids_for(fold, plan)}
        assert not trained & {idx[i].doc for i in scored}, fold


# --------------------------------------------------------------------------- #
# RAMDocs
# --------------------------------------------------------------------------- #


def test_ramdocs_scoring_needs_every_gold_and_no_wrong_answer():
    from study.ramdocs import RQuestion, score

    q = RQuestion("t", "?", (), ("AFL",), ("NBA",))
    assert score("He played in the AFL.", q).correct
    assert not score("He played in the AFL and the NBA.", q).correct
    assert not score("He played basketball.", q).correct


def test_ramdocs_match_is_on_word_boundaries():
    from study.ramdocs import contains

    assert contains("3,559 people live there", "3,559 people")
    assert not contains("the AFLX league", "AFL")


@needs_ram
def test_ramdocs_loads_all_questions():
    from study import ramdocs

    qs = ramdocs.questions()
    assert len(qs) == 500 and all(q.gold_answers for q in qs)
