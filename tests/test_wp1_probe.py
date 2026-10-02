"""Tests for the WP1 go/no-go harness and the LLM entailment scorer.

The harness decides whether Member B's method contribution is the probe or the
fallback, so the tests concentrate on the ways it could return the wrong
verdict: reporting recall without its fabrication cost, averaging the easy
half of the data into the hard half, or scoring a method against an empty
answer key.
"""

from __future__ import annotations

import pytest

from contract.gold import Applicability, Branch, Explicitness, GoldInstance
from contract.models import (
    ConflictType,
    Construction,
    Domain,
    Passage,
    ScopeRelation,
)
from detection.llm_nli import LLMEntailment, _from_label, _neutral
from experiments.wp1_probe.run_probe import ProbeComparison, SystemScore, score


def _branch(bid, outcome, *, default=False, implicit=True, source="p1"):
    return Branch(
        branch_id=bid,
        condition=None if default else "some condition",
        outcome=outcome,
        applicability=Applicability(descriptor="all cases" if default else "some",
                                    is_default=default),
        explicitness=(Explicitness.IMPLICIT if implicit else Explicitness.EXPLICIT),
        supporting_passage=None if default else source,
    )


def _inst(iid="i1", *, branches=None):
    return GoldInstance(
        instance_id=iid, query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.SPLIT,
        passages=[
            Passage(id="p0", text="International transactions incur a 3% fee.",
                    document_id="d0"),
            Passage(id="p1", text="The fee is waived for premium cardholders.",
                    document_id="d0"),
        ],
        gold_conflict_type=ConflictType.CONDITIONAL,
        gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_branches=branches or [
            _branch("b0", "a 3% fee applies", default=True, source="p0"),
            _branch("b1", "no fee applies"),
        ],
    )


# --------------------------------------------------------------------------- #
# The two numbers that must travel together
# --------------------------------------------------------------------------- #


def test_recall_is_never_printed_without_its_fabrication_cost():
    """The spec is explicit: an accuracy gain bought with fabricated
    conditions is not a success. A reader who sees recall and has to go
    looking for HCR will sometimes not go looking."""
    c = ProbeComparison(
        probe=SystemScore("probe", recovered=8, missed=2,
                          implicit_recovered=8, implicit_total=10,
                          instances=10, hallucinating_instances=5),
        baseline=SystemScore("direct", recovered=4, missed=6,
                             implicit_recovered=4, implicit_total=10,
                             instances=10),
    )
    out = c.render()
    assert "recall, unmarked" in out
    assert "HCR" in out


def test_a_gain_bought_with_fabrication_is_not_a_go():
    """+40 points of recall and +50 of HCR. The spec calls this a failure, and
    a verdict function that said GO would be reading only half the result."""
    c = ProbeComparison(
        probe=SystemScore("probe", implicit_recovered=9, implicit_total=10,
                          instances=10, hallucinating_instances=6),
        baseline=SystemScore("direct", implicit_recovered=5, implicit_total=10,
                             instances=10, hallucinating_instances=1),
    )
    head, why = c.verdict()
    assert head.startswith("NO")
    assert "fabrication" in head or "fabrication" in why
    assert "fallback" in why


def test_a_clean_gain_is_a_go():
    c = ProbeComparison(
        probe=SystemScore("probe", implicit_recovered=9, implicit_total=10,
                          instances=10, hallucinating_instances=1),
        baseline=SystemScore("direct", implicit_recovered=5, implicit_total=10,
                             instances=10, hallucinating_instances=1),
    )
    assert c.verdict()[0] == "GO"


def test_no_useful_margin_invokes_the_fallback_by_name():
    """A negative result is planned for, not improvised. The verdict names the
    agreed fallback so nobody has to decide what to do under deadline."""
    c = ProbeComparison(
        probe=SystemScore("probe", implicit_recovered=5, implicit_total=10,
                          instances=10),
        baseline=SystemScore("direct", implicit_recovered=5, implicit_total=10,
                             instances=10),
    )
    head, why = c.verdict()
    assert head.startswith("NO")
    assert "grounding gate" in why and "metric suite" in why


def test_a_corpus_with_no_unmarked_conditions_is_not_a_verdict_about_the_method():
    """If every gold exception carries a cue word, the probe was never tested
    on the case it exists for. Reporting that as "no useful margin" would
    blame the method for a property of the data."""
    c = ProbeComparison(
        probe=SystemScore("probe", explicit_total=10, explicit_recovered=10,
                          instances=10),
        baseline=SystemScore("direct", explicit_total=10, explicit_recovered=10,
                             instances=10),
    )
    head, why = c.verdict()
    assert "NO UNMARKED CONDITIONS" in head
    assert "property of the corpus" in why


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_explicit_and_implicit_are_counted_separately():
    """Averaging them is dominated by the easy half, and would let a method
    that does nothing for unmarked conditions look like it works."""
    inst = _inst(branches=[
        _branch("b0", "a 3% fee applies", default=True, source="p0"),
        _branch("b1", "no fee applies", implicit=True),
        _branch("b2", "a 1% fee applies", implicit=False),
    ])
    s = score([inst], {inst.instance_id: inst.gold_branches}, name="oracle")

    assert s.implicit_total == 1
    assert s.explicit_total == 1
    assert s.implicit_recall == 1.0
    assert s.explicit_recall == 1.0


def test_the_default_branch_lands_in_neither_explicitness_bucket():
    """It has no condition to mark, so counting it would dilute both."""
    inst = _inst()
    s = score([inst], {inst.instance_id: inst.gold_branches}, name="oracle")
    assert s.implicit_total + s.explicit_total == 1, "only the exception counts"
    assert s.gold_branches == 2, "but both branches count toward recall"


def test_an_ungrounded_branch_makes_the_instance_hallucinating():
    inst = _inst()
    invented = [
        _branch("x0", "a 3% fee applies", default=True, source="p0"),
        _branch("x1", "no fee applies", source="p99"),
    ]
    s = score([inst], {inst.instance_id: invented}, name="x")
    assert s.ungrounded_branches == 1
    assert s.hcr == 1.0


def test_perfect_extraction_has_zero_hcr():
    inst = _inst()
    s = score([inst], {inst.instance_id: inst.gold_branches}, name="oracle")
    assert s.recall == 1.0
    assert s.hcr == 0.0


def test_instances_without_branch_structure_are_excluded(tmp_path):
    """An instance Member B has not annotated yet has no answer key, and
    scoring against an empty one reads as perfect recall of nothing."""
    from experiments.wp1_probe.run_probe import _load_gold

    with_branches = _inst("has")
    without = _inst("none").model_copy(update={"gold_branches": []})
    path = tmp_path / "g.jsonl"
    path.write_text(
        "\n".join(i.model_dump_json() for i in (with_branches, without)),
        encoding="utf-8")

    got = _load_gold(path)
    assert [i.instance_id for i in got] == ["has"]


# --------------------------------------------------------------------------- #
# The LLM entailment scorer
# --------------------------------------------------------------------------- #


def test_an_api_failure_scores_neutral_not_entailment():
    """The one direction a grounding gate must never fail in.

    Scoring a failed call as entailment would admit every candidate it touched
    -- the gate would silently stop gating on exactly the runs where something
    went wrong.
    """
    class Boom:
        def complete(self, **kw):
            raise RuntimeError("api down")

    got = LLMEntailment(client=Boom()).score_batch([("a", "b"), ("c", "d")])
    assert len(got) == 2
    assert all(s.neutral == 1.0 and s.entailment == 0.0 for s in got)


def test_a_missing_index_does_not_shift_later_results():
    """A reply that drops one pair must not slide the rest up a slot.

    Misalignment here scores each hypothesis against the wrong premise, and it
    is invisible downstream -- every number the gate feeds would be wrong with
    nothing to indicate it.
    """
    class Partial:
        def complete(self, **kw):
            class R:
                text = ""
                @staticmethod
                def json():
                    # index 1 is missing
                    return {"results": [{"i": 0, "label": "entailment", "confidence": 0.9},
                                        {"i": 2, "label": "contradiction", "confidence": 0.9}]}
            return R()

    got = LLMEntailment(client=Partial()).score_batch(
        [("p0", "h0"), ("p1", "h1"), ("p2", "h2")])

    assert got[0].entailment == pytest.approx(0.9)
    assert got[1].neutral == 1.0, "the gap stays a gap"
    assert got[2].contradiction == pytest.approx(0.9)


def test_an_out_of_range_index_is_ignored():
    class Rogue:
        def complete(self, **kw):
            class R:
                text = ""
                @staticmethod
                def json():
                    return {"results": [{"i": 7, "label": "entailment"}]}
            return R()

    got = LLMEntailment(client=Rogue()).score_batch([("p", "h")])
    assert got[0].neutral == 1.0


def test_labels_map_to_the_right_distribution():
    assert _from_label("entailment", 0.9).entailment == pytest.approx(0.9)
    assert _from_label("contradiction", 0.8).contradiction == pytest.approx(0.8)
    assert _from_label("neutral", 0.7).neutral == pytest.approx(0.7)
    assert _from_label("gibberish", 0.5).neutral > 0, "unknown label is not entailment"
    assert _neutral().entailment == 0.0


def test_scores_form_a_distribution():
    for label in ("entailment", "contradiction", "neutral"):
        s = _from_label(label, 0.9)
        assert s.entailment + s.neutral + s.contradiction == pytest.approx(1.0)
