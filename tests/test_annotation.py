"""Tests for the annotation tooling.

Two of these matter more than the rest, and they are the two that guard numbers
which cannot be corrected after publication: a kappa that did not measure
inter-annotator agreement, and a blind pass that was not blind.
"""

from __future__ import annotations

import pytest

from benchmark.annotation.agreement import AgreementError, cohen_kappa
from benchmark.annotation.store import AnnotationError, AnnotationStore, LabelRecord


# --------------------------------------------------------------------------- #
# Kappa: the arithmetic
# --------------------------------------------------------------------------- #


def test_kappa_matches_a_hand_computation():
    """Worked by hand so the test checks the code rather than restating it.

    Four instances, three agreements::

        (conditional, conditional) (factual, factual)
        (conditional, factual)     (no_conflict, no_conflict)

    p_o = 3/4 = 0.75.
    A's marginals: conditional 2/4, factual 1/4, no_conflict 1/4.
    B's marginals: conditional 1/4, factual 2/4, no_conflict 1/4.
    p_e = (2/4)(1/4) + (1/4)(2/4) + (1/4)(1/4) = 0.125 + 0.125 + 0.0625 = 0.3125.
    kappa = (0.75 - 0.3125) / (1 - 0.3125) = 0.4375 / 0.6875 = 0.63636...
    """
    a = {"i1": "conditional", "i2": "factual", "i3": "conditional", "i4": "no_conflict"}
    b = {"i1": "conditional", "i2": "factual", "i3": "factual", "i4": "no_conflict"}

    r = cohen_kappa(a, b, axis="type", annotator_a="jg", annotator_b="rm")

    assert r.observed_agreement == pytest.approx(0.75)
    assert r.expected_agreement == pytest.approx(0.3125)
    assert r.kappa == pytest.approx(0.63636, abs=1e-4)
    assert r.band == "substantial"


def test_perfect_agreement_on_a_varied_batch_is_one():
    a = {"i1": "conditional", "i2": "factual", "i3": "no_conflict"}
    r = cohen_kappa(a, dict(a), axis="t", annotator_a="jg", annotator_b="rm")
    assert r.kappa == pytest.approx(1.0)


def test_chance_level_agreement_is_near_zero():
    a = {f"i{i}": ("conditional" if i % 2 else "factual") for i in range(40)}
    b = {f"i{i}": ("conditional" if i % 3 else "factual") for i in range(40)}
    r = cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b="rm")
    assert abs(r.kappa) < 0.35


# --------------------------------------------------------------------------- #
# Kappa: the guards. These are the integrity tests.
# --------------------------------------------------------------------------- #


def test_one_person_cannot_produce_a_kappa_with_themselves():
    """Kappa over one observer's two passes measures self-consistency.

    It renders a plausible number and is not the statistic the word names, so
    it is refused rather than returned with a caveat somebody might not read.
    """
    a = {"i1": "conditional", "i2": "factual"}
    with pytest.raises(AgreementError, match="INDEPENDENT"):
        cohen_kappa(a, dict(a), axis="t", annotator_a="jg", annotator_b="jg")


def test_model_labels_cannot_enter_a_kappa():
    """The failure this project is most exposed to.

    Model-proposed labels are cheap and abundant, so the tempting shortcut is
    to compute agreement against them. That statistic would measure how
    self-consistent one model is, and reporting it as inter-annotator
    agreement would misrepresent the benchmark's provenance.
    """
    a = {"i1": "conditional", "i2": "factual"}
    b = {"i1": "conditional", "i2": "conditional"}
    for bad in ("model", "silver", "llm", "auto"):
        with pytest.raises(AgreementError, match="not an inter-annotator"):
            cohen_kappa(a, b, axis="t", annotator_a=bad, annotator_b="rm")
        with pytest.raises(AgreementError):
            cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b=bad)


def test_a_degenerate_batch_reports_undefined_not_perfect():
    """Both annotators label everything the same single category.

    p_o = 1 and p_e = 1, so kappa is 0/0. Returning 1.0 would present a batch
    that tested nothing as flawless annotation -- the kappa paradox, and the
    most flattering way to get this wrong.
    """
    a = {f"i{i}": "conditional" for i in range(10)}
    r = cohen_kappa(a, dict(a), axis="t", annotator_a="jg", annotator_b="rm")

    assert r.degenerate
    assert not r.is_acceptable, "unanimity on one category is not agreement"
    out = r.render()
    assert "undefined" in out
    assert "nothing to disagree about" in out


def test_no_overlap_is_refused_rather_than_scored_as_zero():
    with pytest.raises(AgreementError, match="no instances in common"):
        cohen_kappa({"i1": "factual"}, {"i2": "factual"},
                    axis="t", annotator_a="jg", annotator_b="rm")


def test_only_shared_instances_are_scored():
    """An instance one annotator skipped is missing data, not a disagreement."""
    a = {"i1": "conditional", "i2": "factual", "i3": "factual"}
    b = {"i1": "conditional", "i2": "factual"}
    r = cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b="rm")
    assert r.n == 2


def test_below_threshold_result_says_do_not_proceed():
    a = {f"i{i}": ("conditional" if i % 2 else "factual") for i in range(20)}
    b = {f"i{i}": ("conditional" if i % 3 else "factual") for i in range(20)}
    out = cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b="rm").render()
    assert "Do not start bulk annotation" in out


def test_a_dominant_disagreement_cell_is_named():
    """A single cell carrying the disagreement means one ambiguous paragraph in
    the manual, which is a fix; a scatter means the task is hard, which is not."""
    a = {f"i{i}": "conditional" for i in range(10)}
    b = {f"i{i}": ("factual" if i < 6 else "conditional") for i in range(10)}
    r = cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b="rm")
    assert r.disagreements()[0][0] == ("conditional", "factual")
    assert "One cell carries" in r.render()


# --------------------------------------------------------------------------- #
# Store: blindness and the escalation exemption
# --------------------------------------------------------------------------- #


def test_the_store_offers_no_way_to_read_everyone_at_once(tmp_path):
    """Blindness has to be structural.

    A convenience method returning all annotators' labels would be the natural
    thing to call from the labelling UI, and calling it there is exactly what
    destroys the independence the kappa rests on. So it does not exist.
    """
    store = AnnotationStore(tmp_path)
    assert not hasattr(store, "load_all")
    assert store.annotators() == []


def test_annotators_lists_ids_without_exposing_labels(tmp_path):
    store = AnnotationStore(tmp_path)
    store.append(LabelRecord(instance_id="i1", annotator="jg",
                             conflict_type="factual"))
    assert store.annotators() == ["jg"]
    assert store.load("rm") == [], "a second annotator sees nothing of the first"


def test_a_sealed_pass_cannot_be_edited(tmp_path):
    """Once a pass has been compared, revising it means revising in the light
    of the other annotator's answers."""
    store = AnnotationStore(tmp_path)
    store.append(LabelRecord(instance_id="i1", annotator="jg",
                             conflict_type="factual"))
    store.seal("jg")

    with pytest.raises(AnnotationError, match="sealed"):
        store.append(LabelRecord(instance_id="i2", annotator="jg",
                                 conflict_type="conditional",
                                 scope_relation="refinement"))


def test_an_escalated_conditional_may_carry_no_relation(tmp_path):
    """The manual says escalate rather than guess.

    If the record format demanded a relation anyway, the tool would force the
    guess the manual forbids, and a coin-flip would enter the corpus wearing a
    real label. This exemption is what makes escalation actually available.
    """
    rec = LabelRecord(instance_id="i1", annotator="jg",
                      conflict_type="conditional", scope_relation=None,
                      escalated=True, notes="scopes may or may not nest")
    assert rec.scope_relation is None

    with pytest.raises(AnnotationError, match="needs a scope relation"):
        LabelRecord(instance_id="i2", annotator="jg",
                    conflict_type="conditional", escalated=False)


def test_an_unescalated_no_conflict_cannot_carry_a_relation():
    with pytest.raises(AnnotationError, match="must not carry"):
        LabelRecord(instance_id="i1", annotator="jg",
                    conflict_type="no_conflict", scope_relation="refinement")


def test_a_typo_label_is_rejected_rather_than_becoming_a_category(tmp_path):
    """A free-text label silently becomes its own category and deflates kappa
    against an annotator who spelled it correctly."""
    with pytest.raises(ValueError):
        LabelRecord(instance_id="i1", annotator="jg", conflict_type="conditonal")


def test_relabelling_keeps_the_last_answer_and_the_history(tmp_path):
    store = AnnotationStore(tmp_path)
    store.append(LabelRecord(instance_id="i1", annotator="jg",
                             conflict_type="factual"))
    store.append(LabelRecord(instance_id="i1", annotator="jg",
                             conflict_type="conditional",
                             scope_relation="refinement"))

    loaded = store.load("jg")
    assert len(loaded) == 1
    assert loaded[0].conflict_type == "conditional"

    raw = store.path_for("jg").read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 2, "the superseded label stays on disk as history"


def test_scope_labels_omit_instances_where_the_question_was_never_asked(tmp_path):
    """A scope relation is only defined for a conditional instance.

    Defaulting the others to a placeholder would manufacture agreement on
    instances neither annotator was ever asked about.
    """
    store = AnnotationStore(tmp_path)
    store.append(LabelRecord(instance_id="i1", annotator="jg",
                             conflict_type="conditional",
                             scope_relation="refinement"))
    store.append(LabelRecord(instance_id="i2", annotator="jg",
                             conflict_type="factual"))

    assert store.labels("jg", "conflict_type") == {
        "i1": "conditional", "i2": "factual"}
    assert store.labels("jg", "scope_relation") == {"i1": "refinement"}


def test_adjudicated_labels_are_marked_as_such(tmp_path):
    """A label reviewed from a model proposal is not the same evidence as one
    reached cold, so the basis travels with the record and the datasheet can
    report the split rather than pooling them silently."""
    store = AnnotationStore(tmp_path)
    store.append(LabelRecord(instance_id="i1", annotator="jg",
                             conflict_type="factual", basis="blank"))
    store.append(LabelRecord(instance_id="i2", annotator="jg",
                             conflict_type="factual", basis="model"))

    bases = {r.instance_id: r.basis for r in store.load("jg")}
    assert bases == {"i1": "blank", "i2": "model"}


def test_the_default_basis_is_the_conservative_one():
    """An unmarked record is treated as an independent judgement, so the
    default must be the one that claims less, not more."""
    assert LabelRecord(instance_id="i1", annotator="jg",
                       conflict_type="factual").basis == "blank"


def test_progress_estimates_remaining_time_from_observed_pace(tmp_path):
    store = AnnotationStore(tmp_path)
    for i in range(4):
        store.append(LabelRecord(instance_id=f"i{i}", annotator="jg",
                                 conflict_type="factual", seconds_spent=120.0))
    out = store.progress("jg", total=100)
    assert "4 labelled of 100" in out
    assert "remaining" in out


def test_a_wide_interval_is_flagged_even_when_the_point_estimate_passes():
    """Pilot 1 returned 0.775 on 23 instances with an interval from 0.395 --
    "substantial", and cleared to proceed, on a range that includes "fair".

    A point estimate above the bar with an interval reaching well below it is
    not the same evidence as one above the bar on a large batch, and the
    Landis & Koch label hides the difference.
    """
    a = {f"i{i}": ("conditional" if i % 3 else "factual") for i in range(12)}
    b = dict(a)
    b["i1"] = "factual"

    r = cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b="rm")

    # Asserted unconditionally: a test that only checks inside an `if` passes
    # silently when the condition stops holding, which is the same class of
    # false comfort this caution exists to prevent.
    assert r.is_acceptable, "point estimate clears the bar"
    assert r.interval.low < 0.61, "interval does not"
    assert "CAUTION" in r.render()


def test_a_narrow_interval_above_the_bar_is_not_flagged():
    """The counterpart. A caution on every passing result would be noise, and
    noise is ignored."""
    a = {f"i{i}": ("conditional" if i % 3 else "factual") for i in range(200)}
    r = cohen_kappa(a, dict(a), axis="t", annotator_a="jg", annotator_b="rm")

    assert r.is_acceptable
    assert r.interval.low >= 0.61
    assert "CAUTION" not in r.render()


def test_a_numeric_disagreement_blocks_the_opinion_label():
    """`opinion` means neither passage is checkable. Two passages quoting
    figures that disagree are checkable by definition, so the label is
    impossible rather than unlikely -- and `3` and `4` are adjacent keys."""
    from benchmark.annotation.cli import _numeric_disagreement
    from contract.models import Domain, Passage, QueryRecord, SourceType

    def _rec(t0, t1):
        return QueryRecord(
            query_id="q", query="how much?", domain=Domain.FINANCIAL_TERMS,
            construction="split",
            passages=[
                Passage(id="p0", text=t0, source_type=SourceType.OFFICIAL_POLICY,
                        date="2024-01", document_id="d0"),
                Passage(id="p1", text=t1, source_type=SourceType.PRODUCT_TERMS,
                        date="2024-02", document_id="d1"),
            ])

    assert _numeric_disagreement(_rec("The application fee is 1,500.",
                                      "The application fee is 1,846."))
    assert not _numeric_disagreement(
        _rec("Withdrawals are charged at 2.50 per transaction.",
             "Your account includes unlimited fee-free withdrawals."))


def test_a_query_that_leaks_the_relationship_is_rejected():
    """A generated query fixes what is being asked; it must not hint at the
    answer. "Does the fee apply, except for premium holders?" tells the reader
    the relationship before they have read either passage."""
    from benchmark.mining import build_batch as bb

    class Leaky:
        def __init__(self, q): self.q = q
        def complete(self, **kw):
            class R:
                text = ""
                def json(inner): return {"query": self.q}
            return R()

    assert bb.write_query("a", "b", client=Leaky("Do I pay, except for members?")) is None
    assert bb.write_query("a", "b", client=Leaky("Is there a conflict here?")) is None
    assert bb.write_query("a", "b", client=Leaky("Do I pay a fee")) == "Do I pay a fee?"


def test_scoping_agreement_to_a_batch_changes_the_answer():
    """Passes accumulate across batches, so comparing whole passes pools every
    pilot ever run. Pilot 2 reported n=73 -- 54 old plus 19 new -- and the
    pooled figure hid a per-batch kappa of -0.013 behind pilot 1's 0.326.
    """
    old = {f"wp1_{i}": ("conditional" if i % 2 else "factual") for i in range(20)}
    new = {f"wp2_{i}": "conditional" for i in range(10)}

    a = {**old, **new}
    b = {**old, **{k: "no_conflict" for k in new}}

    pooled = cohen_kappa(a, b, axis="t", annotator_a="jg", annotator_b="rm")
    scoped = cohen_kappa({k: v for k, v in a.items() if k.startswith("wp2_")},
                         {k: v for k, v in b.items() if k.startswith("wp2_")},
                         axis="t", annotator_a="jg", annotator_b="rm")

    assert pooled.n == 30 and scoped.n == 10
    assert pooled.kappa > scoped.kappa, "pooling hides the failing batch"
    assert scoped.degenerate or scoped.kappa <= 0.0
