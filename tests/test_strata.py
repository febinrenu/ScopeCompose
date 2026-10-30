"""v7 relabel and strata: the counts the plan stands on, and the split invariants."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmark.relabel_v7 import (
    V7Label,
    infer_opposition,
    natural_conditional_conflicts,
    relabel,
    relabel_all,
)
from benchmark.splits import audit
from benchmark.strata import (
    S1_FLOOR,
    Stratum,
    assign_stratified,
    is_llm_authored,
    quota_report,
    stratum_of,
)
from contract.gold import GoldInstance
from contract.mock import generate
from contract.models import ConflictType, Opposition, ScopeRelation

CORPUS = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "corpus.jsonl"
needs_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")


@pytest.fixture(scope="module")
def corpus() -> list[GoldInstance]:
    return [GoldInstance.model_validate(json.loads(l))
            for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# relabel
# --------------------------------------------------------------------------- #


@needs_corpus
def test_pilot_has_18_natural_conditional_conflicts_14_s1_and_4_s2(corpus):
    """Plan v4 acceptance. (An earlier count of 19 included 5 disjoint pairs,
    which are not conflicts under v8.)"""
    nat = natural_conditional_conflicts(corpus)
    assert nat["cross_document"] == 14
    assert nat["same_guide"] == 4
    assert sum(nat.values()) == 18


@needs_corpus
def test_disjoint_and_redundant_are_no_conflict_under_v7(corpus):
    labels = {r.instance_id: r for r in relabel_all(corpus)}
    for inst in corpus:
        if inst.gold_scope_relation in (ScopeRelation.DISJOINT, ScopeRelation.REDUNDANT):
            assert labels[inst.instance_id].label is V7Label.NO_CONFLICT


@needs_corpus
def test_every_label_is_derived_from_gold_and_the_corpus_is_not_modified(corpus):
    """The sidecar approach: the frozen gold carries no v7 label of its own."""
    assert all(i.contract_version == "1.2.0" for i in corpus)
    assert len(relabel_all(corpus)) == len(corpus)


@needs_corpus
def test_the_two_pilot_opposed_instances_are_crossing(corpus):
    opposed = [i for i in corpus if i.gold_scope_relation is ScopeRelation.OPPOSED]
    assert len(opposed) == 2
    for inst in opposed:
        assert inst.default_branch is None
        assert infer_opposition(inst) is Opposition.CROSSING
        assert relabel(inst).label is V7Label.CROSSING


def test_opposed_with_a_default_branch_is_undecided_not_guessed():
    """A default branch makes the other a subset -- i.e. refinement. An 'opposed'
    label on that structure contradicts it, so it is left for a person."""
    inst = next(i for i in generate(80, seed=0) if i.gold_scope_relation is ScopeRelation.OPPOSED)
    d = inst.model_dump(mode="json")
    d["gold_opposition"] = None
    d["gold_branches"][0].update(condition=None)
    d["gold_branches"][0]["applicability"].update(is_default=True, attributes=[])
    assert infer_opposition(GoldInstance.model_validate(d)) is Opposition.UNKNOWN
    assert relabel(GoldInstance.model_validate(d)).label is V7Label.OPPOSED_UNDECIDED


def test_equal_scopes_are_same_scope():
    inst = next(i for i in generate(80, seed=0) if i.gold_scope_relation is ScopeRelation.OPPOSED)
    d = inst.model_dump(mode="json")
    d["gold_opposition"] = None
    d["gold_branches"][1]["applicability"] = d["gold_branches"][0]["applicability"]
    assert infer_opposition(GoldInstance.model_validate(d)) is Opposition.SAME_SCOPE


def test_explicit_gold_opposition_wins_over_inference():
    inst = next(i for i in generate(80, seed=0) if i.gold_scope_relation is ScopeRelation.OPPOSED)
    d = inst.model_dump(mode="json")
    d["gold_opposition"] = "same_scope"
    assert infer_opposition(GoldInstance.model_validate(d)) is Opposition.SAME_SCOPE


def test_relabel_of_non_conditional_classes_is_unchanged():
    seen = set()
    for inst in generate(120, seed=1):
        if inst.gold_conflict_type is not ConflictType.CONDITIONAL:
            r = relabel(inst)
            assert r.label.value == inst.gold_conflict_type.value
            seen.add(r.label)
    assert V7Label.NO_CONFLICT in seen


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #


@needs_corpus
def test_pilot_strata_counts(corpus):
    c = Counter(stratum_of(i) for i in corpus)
    assert c[Stratum.S1] == 14
    assert c[Stratum.S2] == 4
    assert c[Stratum.S5] == 54, "all 54 LLM-authored probes are S5 whatever their label"
    assert c[Stratum.S3] == 0, "no real-text split is in the corpus yet"
    assert sum(c.values()) == 128


@needs_corpus
def test_llm_authored_is_detected_by_placeholder_host(corpus):
    assert sum(is_llm_authored(i) for i in corpus) == 54


@needs_corpus
def test_llm_authored_is_s5_even_when_it_is_a_conditional_conflict(corpus):
    probe = next(i for i in corpus if is_llm_authored(i)
                 and relabel(i).label is V7Label.CONDITIONAL_CONFLICT)
    assert stratum_of(probe) is Stratum.S5


@needs_corpus
def test_contrast_ids_route_to_s4(corpus):
    ctrl = next(i for i in corpus if stratum_of(i) is Stratum.CONTROL)
    assert stratum_of(ctrl, contrast_ids=frozenset({ctrl.instance_id})) is Stratum.S4


def test_conditional_conflict_with_unrecorded_separation_is_refused():
    base = next(i for i in generate(60, seed=0) if i.gold_scope_relation is ScopeRelation.REFINEMENT)
    real = [p.model_copy(update={"source_url": f"https://www.gov.uk/{p.id}"}) for p in base.passages]
    inst = base.model_copy(update={"separation": None, "passages": real})
    assert not is_llm_authored(inst)
    with pytest.raises(ValueError, match="unrecorded separation"):
        stratum_of(inst)


# --------------------------------------------------------------------------- #
# stratified assignment
# --------------------------------------------------------------------------- #


@needs_corpus
def test_resplit_has_no_leakage_and_is_deterministic(corpus):
    a = assign_stratified(corpus, seed=0)
    b = assign_stratified(corpus, seed=0)
    assert a == b
    after = [i.model_copy(update={"split": a[i.instance_id]}) for i in corpus]
    assert audit(after).clean


@needs_corpus
def test_resplit_keeps_s1_out_of_train_and_s5_out_of_test(corpus):
    a = assign_stratified(corpus, seed=0)
    for inst in corpus:
        s = stratum_of(inst)
        if s is Stratum.S1:
            assert a[inst.instance_id] in ("test", "dev")
        if s is Stratum.S5:
            assert a[inst.instance_id] in ("train", "dev")


@needs_corpus
def test_resplit_puts_most_s1_in_test(corpus):
    a = assign_stratified(corpus, seed=0)
    after = [i.model_copy(update={"split": a[i.instance_id]}) for i in corpus]
    q = quota_report(after)
    assert q.s1_test >= 8 and q.s5_in_test == []


@needs_corpus
def test_incremental_mode_never_moves_an_already_placed_group(corpus):
    first = assign_stratified(corpus, seed=0)
    placed = [i.model_copy(update={"split": first[i.instance_id]}) for i in corpus]
    again = assign_stratified(placed, seed=7, keep_existing=True)
    assert again == first, "an incremental run with nothing new must be a no-op, whatever the seed"


@needs_corpus
def test_incremental_mode_refuses_existing_leakage(corpus):
    from benchmark.splits import build_groups

    g = max(build_groups(corpus), key=lambda g: g.size)
    assert g.size >= 2
    ids = list(g.instance_ids)
    broken = [i.model_copy(update={"split": "train" if i.instance_id == ids[0] else "test"})
              if i.instance_id in ids[:2] else i for i in corpus]
    with pytest.raises(ValueError, match="existing leakage"):
        assign_stratified(broken, seed=0, keep_existing=True)


@needs_corpus
def test_narrowing_rule_is_computed_from_the_data(corpus):
    """With 14 S1 instances the floor cannot be met, and the report must say so
    rather than leave it to the prose."""
    a = assign_stratified(corpus, seed=0)
    after = [i.model_copy(update={"split": a[i.instance_id]}) for i in corpus]
    q = quota_report(after)
    assert q.s1_test < S1_FLOOR
    assert q.narrowing_rule_applies
    assert "BELOW FLOOR" in q.status
    assert "EXPLORATORY" in q.render()
