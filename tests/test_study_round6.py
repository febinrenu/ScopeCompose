"""Analyses added in the sixth review round: prompt variants, retrieval designs, the clean threshold
protocol and semantic matching. Tests that need ConditionalQA skip when it is not downloaded."""

from __future__ import annotations

import json

import pytest

from study import prompts, reader
from study.fetch_data import path_of

HAVE_CQA = path_of("conditionalqa", "documents.json").exists()
needs_cqa = pytest.mark.skipif(not HAVE_CQA, reason="ConditionalQA not downloaded")


# --------------------------------------------------------------------------- #
# Prompt variants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", sorted(prompts.VARIANTS))
def test_variant_demos_keep_the_frozen_answers_and_conditions(variant):
    """Every variant shows the same four worked examples with the same answers and gold conditions."""
    assert len(prompts.DEMO_REPLIES[variant]) == len(reader.DEMOS)
    for (_, frozen), reply in zip(reader.DEMOS, prompts.DEMO_REPLIES[variant]):
        want = [(a["answer"], a["evidence"], a["conditions"]) for a in json.loads(frozen)["answers"]]
        got = [(a["answer"], a["evidence"], a["conditions"]) for a in reply["answers"]]
        assert got == want


@pytest.mark.parametrize("variant", sorted(prompts.VARIANTS))
def test_variant_output_is_read_by_the_frozen_parser(variant):
    ev = reader.Evidence("[1] a\n[2] b", {1: "<p>a</p>", 2: "<li>b</li>"})
    reply = dict(prompts.DEMO_REPLIES[variant][3])
    reply["answers"] = [{"answer": "x", "evidence": [1], "conditions": [2]}]
    parsed = reader.parse_output(json.dumps(reply), ev)
    assert parsed.ok and parsed.answers == (("x", ("<li>b</li>",)),)


def test_variants_have_distinct_prompt_hashes():
    shas = {prompts.variant_sha(v) for v in prompts.VARIANTS}
    assert len(shas) == len(prompts.VARIANTS) and reader.prompts_sha() not in shas


# --------------------------------------------------------------------------- #
# Retrieval designs
# --------------------------------------------------------------------------- #


@needs_cqa
def test_span_ids_round_trip_to_their_elements():
    from study import baselines, cqa

    d = cqa.documents()[3]
    span = baselines.Span(d.index, -1, 2, 5, d.title, "", d.elements[2:5])
    assert baselines.span_elements(span.id) == d.elements[2:5]
    assert baselines.span_elements("d3c1") is None


@needs_cqa
def test_parent_window_never_exceeds_the_room_left():
    from study import baselines, cqa

    for d in cqa.documents()[:40]:
        for i in range(0, len(d.elements), 7):
            for room in (0, 20, 80):
                unit = baselines._parent(d.index, i, room, set())
                words = sum(baselines._words(d.index, j) for j in unit)
                s, e = next((s, e) for s, e in baselines._sections(d.index) if s <= i < e)
                assert i in unit and all(s <= j < e for j in unit)
                assert words <= room or unit == [i]


# --------------------------------------------------------------------------- #
# Clean threshold protocol
# --------------------------------------------------------------------------- #


@needs_cqa
def test_clean_folds_train_on_the_other_evaluation_folds_only():
    from study import qualifier

    plan = qualifier.fold_plan()
    for k in range(qualifier.K_FOLDS):
        ids = set(qualifier.train_ids_for(f"c{k}", plan))
        assert not ids & set(plan["tuning"]) and not ids & set(plan[f"f{k}"])
        assert ids == {i for f in range(qualifier.K_FOLDS) if f != k for i in plan[f"f{f}"]}
