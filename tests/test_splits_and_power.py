"""Tests for split assignment and the power analysis.

Both exist to stop a specific wrong number reaching the paper: a leaked test
score, and a null result reported as equivalence. The tests check the
arithmetic against known values where one exists, and the failure mode where
it does not.
"""

from __future__ import annotations

import pytest

from benchmark.splits import assign, audit, build_groups
from contract.gold import Applicability, Branch, GoldInstance
from contract.models import (
    ConflictType,
    Construction,
    Domain,
    Passage,
    ScopeRelation,
)
from metrics.power import (
    _z,
    detectable_effect,
    paired_size,
    proportion_size,
)


def _inst(iid, doc_a, doc_b, *, split="train"):
    return GoldInstance(
        instance_id=iid, query="q?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.SPLIT, split=split,
        passages=[
            Passage(id="p0", text=f"rule text for {doc_a}", document_id=doc_a),
            Passage(id="p1", text=f"exception text for {doc_b}", document_id=doc_b),
        ],
        gold_conflict_type=ConflictType.CONDITIONAL,
        gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_branches=[
            Branch(branch_id="b0", condition=None, outcome="a fee applies",
                   applicability=Applicability(descriptor="all cases",
                                               is_default=True),
                   supporting_passage="p0"),
            Branch(branch_id="b1", condition="premium", outcome="no fee applies",
                   applicability=Applicability(descriptor="premium"),
                   supporting_passage="p1"),
        ],
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def test_instances_sharing_a_document_land_in_one_split():
    """The leakage this exists to prevent.

    A system that memorised a document's wording in train scores on test for
    having seen it. The corpus had 18 documents spanning dev and train before
    this ran, and nothing in any output would have shown it.
    """
    instances = [_inst(f"i{i}", "shared_doc", f"other{i}") for i in range(12)]
    instances += [_inst(f"j{i}", f"lone{i}", f"lone{i}b") for i in range(12)]

    mapping = assign(instances)
    shared = {mapping[f"i{i}"] for i in range(12)}
    assert len(shared) == 1, "every instance citing shared_doc must co-locate"


def test_a_chain_of_shared_documents_forms_one_group():
    """Transitivity. If A cites docs 1 and 2, and B cites 2 and 3, then doc 1
    and doc 3 are linked through B -- splitting them splits a group that
    cannot be split."""
    instances = [_inst("a", "d1", "d2"), _inst("b", "d2", "d3"),
                 _inst("c", "d3", "d4")]
    groups = build_groups(instances)
    assert len(groups) == 1
    assert groups[0].size == 3


def test_unrelated_instances_form_separate_groups():
    instances = [_inst("a", "d1", "d2"), _inst("b", "d3", "d4")]
    assert len(build_groups(instances)) == 2


def test_every_split_is_populated_on_a_reasonable_corpus():
    instances = [_inst(f"i{i}", f"d{i}a", f"d{i}b") for i in range(60)]
    mapping = assign(instances)
    counts = {s: sum(1 for v in mapping.values() if v == s)
              for s in ("train", "dev", "test")}
    assert all(n > 0 for n in counts.values()), counts
    assert counts["train"] > counts["dev"], "train should be the largest"


def test_assignment_is_deterministic_for_a_seed():
    instances = [_inst(f"i{i}", f"d{i}a", f"d{i}b") for i in range(30)]
    assert assign(instances, seed=7) == assign(instances, seed=7)


def test_the_audit_names_leaking_documents():
    leaked = [_inst("a", "shared", "x", split="train"),
              _inst("b", "shared", "y", split="test")]
    report = audit(leaked)
    assert not report.clean
    assert "shared" in report.leaked_documents
    assert "memorised" in report.render()


def test_a_clean_assignment_audits_clean():
    clean = [_inst("a", "d1", "d2", split="train"),
             _inst("b", "d3", "d4", split="test")]
    report = audit(clean)
    assert report.clean
    assert "none --" in report.render()


def test_the_audit_separates_annotation_provenance():
    """Two classes of evidence -- two independent annotators, versus
    model-proposed with one adjudicator -- must never be pooled in a headline,
    so the audit reports them per split."""
    a = _inst("a", "d1", "d2", split="train")
    b = _inst("b", "d3", "d4", split="train").model_copy(
        update={"annotation": a.annotation.model_copy(
            update={"annotator_a": "model-proposed", "annotator_b": "human"})})
    a = a.model_copy(update={"annotation": a.annotation.model_copy(
        update={"annotator_a": "johann", "annotator_b": "febin"})})

    counts = audit([a, b]).by_provenance["train"]
    assert counts["two-annotator"] == 1
    assert counts["model-proposed+adjudicated"] == 1


# --------------------------------------------------------------------------- #
# Power: checked against values that exist independently
# --------------------------------------------------------------------------- #


def test_the_normal_quantiles_match_published_values():
    assert _z(0.975) == pytest.approx(1.9600, abs=1e-3)
    assert _z(0.80) == pytest.approx(0.8416, abs=1e-3)
    assert _z(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert _z(0.5) == pytest.approx(0.0, abs=1e-6)


def test_two_proportion_size_matches_the_textbook_case():
    """0.50 against 0.70 at alpha .05, power .80 is about 93 per group in
    every standard table. A power function that cannot reproduce a known case
    should not be trusted on an unknown one."""
    assert proportion_size(0.50, 0.20) == pytest.approx(93, abs=2)


def test_smaller_effects_need_more_data():
    sizes = [paired_size(e).branches_needed for e in (0.20, 0.15, 0.10, 0.05)]
    assert sizes == sorted(sizes), "monotonic in effect size"
    assert sizes[-1] > 10 * sizes[0], "halving the effect quadruples the need"


def test_a_lower_discordant_rate_needs_fewer_branches_not_more():
    """Counterintuitive, and the opposite of what this test first asserted.

    The intuition that fails: "only discordant branches inform McNemar, so
    fewer of them must be worse." But the effect being sized is a difference
    over ALL branches, and a low discordant rate means that difference is
    concentrated in a small subset -- which makes it easier to see per
    discordant branch, not harder.

    At a 10-point total effect: 20% discordant implies a 75/25 split among
    them, while 50% discordant implies only 60/40. The lopsided case needs
    far less data.

    The practical reading for corpus planning is the reverse of the obvious
    one: two systems that mostly agree but differ sharply where they differ
    are *cheaper* to distinguish than two that disagree everywhere by a little.
    """
    concentrated = paired_size(0.10, discordant_rate=0.20).branches_needed
    diffuse = paired_size(0.10, discordant_rate=0.50).branches_needed
    assert concentrated < diffuse


def test_a_lower_discordant_rate_needs_a_larger_share_of_the_corpus():
    """The cost that does rise. Fewer branches overall, but a greater fraction
    of them must be discordant for the estimate to hold -- so an observed rate
    well below the assumed one invalidates the plan."""
    lo = paired_size(0.10, discordant_rate=0.20)
    hi = paired_size(0.10, discordant_rate=0.50)
    assert lo.discordant_needed / lo.branches_needed == pytest.approx(0.20, abs=0.02)
    assert hi.discordant_needed / hi.branches_needed == pytest.approx(0.50, abs=0.02)


def test_detectable_effect_is_consistent_with_the_size_it_inverts():
    """The inverse must agree with the forward function, or one of them is
    lying. An earlier version walked to 99.9% at 83 branches while the table
    beside it showed 20% needing 63 -- the bisection read "effect too large to
    express" as "needs more data"."""
    for n in (24, 83, 269):
        e = detectable_effect(n)
        assert paired_size(e).branches_needed <= n
        assert e < 1.0


def test_the_detectable_effect_never_exceeds_the_discordant_rate():
    """Beyond it the model breaks: every disagreement would have to favour one
    system and still fall short."""
    for rate in (0.20, 0.35, 0.50):
        assert detectable_effect(1000, discordant_rate=rate) <= rate


def test_an_effect_larger_than_the_discordant_rate_is_not_sizeable():
    assert paired_size(0.40, discordant_rate=0.35).branches_needed == 0


def test_the_plan_states_what_the_current_corpus_cannot_see():
    from metrics.power import render_plan

    out = render_plan(current_branches=83, current_instances=42)
    assert "smallest detectable difference" in out
    assert "spanning zero" in out
