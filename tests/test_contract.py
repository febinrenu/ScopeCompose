"""Tests for the frozen interface contract.

These are the most load-bearing tests in the repo. The contract is what lets
two people work independently; a silent change to it is the integration-drift
failure the freeze rule exists to prevent. Every rule encoded in
``contract/models.py`` gets a test here so that breaking it is loud.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from contract import mock
from contract.gold import (
    Applicability,
    AttributeKind,
    Branch,
    GoldInstance,
    ScopeAttribute,
)
from contract.models import (
    CONTRACT_VERSION,
    ConflictPair,
    ConflictType,
    Construction,
    ContractVersionError,
    DecidedBy,
    Domain,
    MultiExceptionFlags,
    Passage,
    QueryRecord,
    ScopeRelation,
    SourceType,
    check_version,
)
from contract.routing import Action, route, route_with_flags


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


def test_five_class_taxonomy_is_exactly_five():
    assert {c.value for c in ConflictType} == {
        "no_conflict", "factual", "temporal", "opinion", "conditional",
    }


def test_scope_relation_is_four_way():
    """The four-way relation replaced an earlier two-way version.

    The two-way version conflated "do the scopes overlap" with "is this a
    conflict" and so could not tell a redundant restatement from a real
    exception. If this ever collapses back to two values, that bug is back.
    """
    assert {r.value for r in ScopeRelation} == {
        "refinement", "disjoint", "redundant", "opposed",
    }


def test_construction_tiers():
    assert {c.value for c in Construction} == {"natural", "split"}


# --------------------------------------------------------------------------- #
# ConflictPair invariants
# --------------------------------------------------------------------------- #


def _pair(**kw) -> ConflictPair:
    base = dict(
        doc_i="p0", doc_j="p1", is_conflict=True,
        type=ConflictType.CONDITIONAL, scope_relation=ScopeRelation.REFINEMENT,
        confidence=0.9,
    )
    base.update(kw)
    return ConflictPair(**base)


def test_pair_order_is_canonicalised():
    """A pair is unordered, so (p1,p0) must normalise to (p0,p1).

    This is the schema-level half of order invariance: it makes a record
    byte-identical under any permutation of the input passages.
    """
    a = _pair(doc_i="p0", doc_j="p1")
    b = _pair(doc_i="p1", doc_j="p0")
    assert a.doc_i == b.doc_i == "p0"
    assert a.doc_j == b.doc_j == "p1"
    assert a.key == b.key
    assert a.model_dump() == b.model_dump()


def test_pair_rejects_self_reference():
    with pytest.raises(ValidationError, match="two different passages"):
        _pair(doc_i="p0", doc_j="p0")


def test_conditional_pair_requires_scope_relation():
    """Member B routes on scope_relation. A conditional pair without one is
    unroutable, so the contract refuses to represent it."""
    with pytest.raises(ValidationError, match="requires a scope_relation"):
        _pair(type=ConflictType.CONDITIONAL, scope_relation=None)


def test_no_conflict_pair_rejects_scope_relation():
    with pytest.raises(ValidationError, match="must not carry a scope_relation"):
        _pair(type=ConflictType.NO_CONFLICT, is_conflict=False,
              scope_relation=ScopeRelation.REFINEMENT)


def test_no_conflict_type_contradicts_is_conflict_true():
    with pytest.raises(ValidationError, match="contradicts is_conflict=True"):
        _pair(type=ConflictType.NO_CONFLICT, is_conflict=True, scope_relation=None)


def test_is_conflict_false_requires_no_conflict_type():
    with pytest.raises(ValidationError, match="requires type='no_conflict'"):
        _pair(type=ConflictType.FACTUAL, is_conflict=False, scope_relation=None)


def test_scope_relation_rejected_on_temporal():
    with pytest.raises(ValidationError, match="only meaningful for conditional"):
        _pair(type=ConflictType.TEMPORAL, scope_relation=ScopeRelation.OPPOSED)


def test_scope_relation_allowed_on_borderline_factual():
    """The scope analyser also runs on borderline factual pairs, so the
    contract must be able to carry the result."""
    p = _pair(type=ConflictType.FACTUAL, scope_relation=ScopeRelation.OPPOSED)
    assert p.scope_relation is ScopeRelation.OPPOSED


@pytest.mark.parametrize("conf", [-0.01, 1.01])
def test_confidence_must_be_a_probability(conf):
    with pytest.raises(ValidationError):
        _pair(confidence=conf)


# --------------------------------------------------------------------------- #
# QueryRecord invariants
# --------------------------------------------------------------------------- #


def _passages(n=2, doc_ids=None):
    doc_ids = doc_ids or [f"doc{i}" for i in range(n)]
    return [
        Passage(id=f"p{i}", text=f"text {i}", source_type=SourceType.OFFICIAL_POLICY,
                date="2024-01", document_id=doc_ids[i])
        for i in range(n)
    ]


def _record(**kw) -> QueryRecord:
    base = dict(
        query_id="q1", query="do I pay a fee?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=_passages(),
        conflict_pairs=[_pair()],
    )
    base.update(kw)
    return QueryRecord(**base)


def test_record_round_trips_through_json():
    r = _record()
    again = QueryRecord.model_validate(json.loads(r.model_dump_json()))
    assert again == r


def test_record_stamps_contract_version():
    assert _record().contract_version == CONTRACT_VERSION


def test_duplicate_passage_ids_rejected():
    ps = _passages()
    ps[1] = ps[1].model_copy(update={"id": "p0"})
    with pytest.raises(ValidationError, match="duplicate passage ids"):
        _record(passages=ps, conflict_pairs=[])


def test_pair_referencing_unknown_passage_rejected():
    with pytest.raises(ValidationError, match="unknown passage"):
        _record(conflict_pairs=[_pair(doc_i="p0", doc_j="p99")])


def test_duplicate_unordered_pair_rejected():
    """(p0,p1) and (p1,p0) are the same pair; storing both would double-count
    it in every metric computed over pairs."""
    with pytest.raises(ValidationError, match="duplicate conflict_pair"):
        _record(conflict_pairs=[_pair(doc_i="p0", doc_j="p1"),
                                _pair(doc_i="p1", doc_j="p0")])


def test_natural_construction_rejects_single_document_provenance():
    """The Tier-1 yield risk, guarded at the schema level.

    A 'natural' instance claims its rule and exception came from separate
    documents. Where document_ids are present we check that claim rather than
    trusting it, so a mislabelled Tier-2 instance fails loudly instead of
    quietly inflating the headline naturally-occurring count.
    """
    ps = _passages(doc_ids=["same_doc", "same_doc"])
    with pytest.raises(ValidationError, match="that is a Tier-2"):
        _record(construction=Construction.NATURAL, passages=ps, conflict_pairs=[])


def test_split_construction_allows_single_document_provenance():
    ps = _passages(doc_ids=["same_doc", "same_doc"])
    r = _record(construction=Construction.SPLIT, passages=ps, conflict_pairs=[])
    assert r.construction is Construction.SPLIT


def test_record_accessors():
    r = _record()
    assert r.passage("p0").id == "p0"
    with pytest.raises(KeyError):
        r.passage("nope")
    assert len(r.conditional_pairs()) == 1
    assert r.pair_for("p1", "p0") is not None
    assert r.pair_for("p0", "p9") is None


def test_extra_fields_forbidden():
    """Silent field drift is how two people's schemas diverge without either
    noticing, so unknown keys are an error rather than ignored."""
    with pytest.raises(ValidationError):
        QueryRecord.model_validate({**_record().model_dump(mode="json"), "surprise": 1})


@pytest.mark.parametrize("bad", ["24-01", "2024/01", "January 2024", "2024-1"])
def test_bad_dates_rejected(bad):
    with pytest.raises(ValidationError, match="date must be"):
        Passage(id="p0", text="t", source_type=SourceType.FAQ, date=bad)


@pytest.mark.parametrize("good", ["2024", "2024-01", "2024-01-15"])
def test_good_dates_accepted(good):
    assert Passage(id="p0", text="t", source_type=SourceType.FAQ, date=good).date == good


# --------------------------------------------------------------------------- #
# Version compatibility
# --------------------------------------------------------------------------- #


def test_same_version_passes():
    check_version(CONTRACT_VERSION)


def test_minor_version_drift_tolerated():
    major = CONTRACT_VERSION.split(".")[0]
    check_version(f"{major}.99.0")


def test_major_version_drift_rejected():
    with pytest.raises(ContractVersionError, match="incompatible contract major version"):
        check_version("99.0.0")


def test_strict_mode_demands_exact_match():
    major = CONTRACT_VERSION.split(".")[0]
    with pytest.raises(ContractVersionError, match="strict mode"):
        check_version(f"{major}.99.0", strict=True)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "relation,expected",
    [
        (ScopeRelation.REFINEMENT, Action.COMPOSE),
        (ScopeRelation.DISJOINT, Action.COMPOSE),
        (ScopeRelation.REDUNDANT, Action.MERGE),
        (ScopeRelation.OPPOSED, Action.SELECT),
    ],
)
def test_conditional_routing(relation, expected):
    assert route(_pair(scope_relation=relation)).action is expected


@pytest.mark.parametrize("ctype", [ConflictType.FACTUAL, ConflictType.TEMPORAL, ConflictType.OPINION])
def test_other_types_route_to_prior_work(ctype):
    assert route(_pair(type=ctype, scope_relation=None)).action is Action.PRIOR_WORK


def test_no_conflict_passes_through():
    p = _pair(type=ConflictType.NO_CONFLICT, is_conflict=False, scope_relation=None)
    assert route(p).action is Action.PASS_THROUGH


def test_every_relation_is_routable():
    """A new scope relation added without a routing rule would fail here rather
    than silently falling through to a default."""
    for relation in ScopeRelation:
        decision = route(_pair(scope_relation=relation))
        assert decision.action in set(Action)
        assert decision.reason


@pytest.mark.parametrize(
    "flags,expected_flag",
    [
        (MultiExceptionFlags(nested=True), "nested"),
        (MultiExceptionFlags(crossed=True), "crossed"),
    ],
)
def test_second_order_structure_forces_selection(flags, expected_flag):
    """Flagged, not mis-composed. Reporting the two rates separately is itself
    a result, so the specific pattern must survive into the decision."""
    d = route_with_flags(_pair(scope_relation=ScopeRelation.REFINEMENT), flags)
    assert d.action is Action.SELECT
    assert d.flag == expected_flag


def test_flags_do_not_override_a_merge():
    """A redundant pair merges regardless; second-order flags only downgrade a
    would-be COMPOSE."""
    d = route_with_flags(_pair(scope_relation=ScopeRelation.REDUNDANT),
                         MultiExceptionFlags(nested=True))
    assert d.action is Action.MERGE


def test_unflagged_refinement_still_composes():
    d = route_with_flags(_pair(scope_relation=ScopeRelation.REFINEMENT), MultiExceptionFlags())
    assert d.action is Action.COMPOSE


# --------------------------------------------------------------------------- #
# Gold instance schema
# --------------------------------------------------------------------------- #


def _gold(**kw) -> GoldInstance:
    base = dict(
        instance_id="g1", query="q", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=_passages(),
        gold_conflict_type=ConflictType.CONDITIONAL,
        gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_branches=[
            Branch(branch_id="b0", condition=None, outcome="fee applies",
                   applicability=Applicability(descriptor="all", is_default=True),
                   supporting_passage="p0"),
            Branch(branch_id="b1", condition="premium tier", outcome="no fee",
                   applicability=Applicability(descriptor="premium holders"),
                   supporting_passage="p1"),
        ],
    )
    base.update(kw)
    return GoldInstance(**base)


def test_gold_requires_exactly_one_default_branch():
    two_defaults = [
        Branch(branch_id="b0", condition=None, outcome="x",
               applicability=Applicability(descriptor="all", is_default=True),
               supporting_passage="p0"),
        Branch(branch_id="b1", condition=None, outcome="y",
               applicability=Applicability(descriptor="all", is_default=True),
               supporting_passage="p1"),
    ]
    with pytest.raises(ValidationError, match="exactly one default branch"):
        _gold(gold_branches=two_defaults)


def test_exception_branch_must_be_grounded():
    """An ungrounded exception is exactly what the grounding gate rejects, so
    it cannot appear in gold data."""
    branches = [
        Branch(branch_id="b0", condition=None, outcome="x",
               applicability=Applicability(descriptor="all", is_default=True),
               supporting_passage="p0"),
        Branch(branch_id="b1", condition="premium", outcome="y",
               applicability=Applicability(descriptor="premium"),
               supporting_passage=None),
    ]
    with pytest.raises(ValidationError, match="no supporting_passage"):
        _gold(gold_branches=branches)


def test_branch_citing_unknown_passage_rejected():
    branches = [
        Branch(branch_id="b0", condition=None, outcome="x",
               applicability=Applicability(descriptor="all", is_default=True),
               supporting_passage="p0"),
        Branch(branch_id="b1", condition="premium", outcome="y",
               applicability=Applicability(descriptor="premium"),
               supporting_passage="p42"),
    ]
    with pytest.raises(ValidationError, match="unknown passage"):
        _gold(gold_branches=branches)


def test_redundant_instance_resolves_to_one_merged_branch():
    """The defining property of REDUNDANT.

    Recording a second branch here would assert an exception that does not
    exist -- a fabricated branch inflating Preservation Rate, which is the
    precise failure that separating 'redundant' from 'refinement' prevents.
    """
    one = [Branch(branch_id="b0", condition=None, outcome="fee applies",
                  applicability=Applicability(descriptor="all", is_default=True),
                  supporting_passage="p0")]
    inst = _gold(gold_scope_relation=ScopeRelation.REDUNDANT, gold_branches=one)
    assert len(inst.gold_branches) == 1

    with pytest.raises(ValidationError, match="exactly one merged branch"):
        _gold(gold_scope_relation=ScopeRelation.REDUNDANT)  # default fixture has 2


def test_conditioned_branch_cannot_be_default():
    with pytest.raises(ValidationError, match="cannot apply to the whole case space"):
        Branch(branch_id="b", condition="premium", outcome="x",
               applicability=Applicability(descriptor="premium", is_default=True))


def test_gold_projects_to_a_valid_wire_record():
    """The oracle path: this is how Member B builds before A's detector exists,
    and how A's own modules get scored."""
    r = _gold().to_query_record()
    assert isinstance(r, QueryRecord)
    assert len(r.conflict_pairs) == 1
    assert r.conflict_pairs[0].decided_by is DecidedBy.GOLD
    assert r.conflict_pairs[0].scope_relation is ScopeRelation.REFINEMENT
    assert r.construction is Construction.NATURAL


def test_gold_projects_to_input_only_record():
    r = _gold().to_query_record(include_gold_pairs=False)
    assert r.conflict_pairs == []


# --------------------------------------------------------------------------- #
# ScopeAttribute payload validation
# --------------------------------------------------------------------------- #


def test_categorical_requires_values():
    with pytest.raises(ValidationError, match="requires a non-empty 'values'"):
        ScopeAttribute(name="tier", kind=AttributeKind.CATEGORICAL)


def test_numeric_range_requires_a_bound():
    with pytest.raises(ValidationError, match="at least one of min_value"):
        ScopeAttribute(name="amount", kind=AttributeKind.NUMERIC_RANGE)


def test_numeric_range_rejects_inverted_bounds():
    with pytest.raises(ValidationError, match="exceeds max_value"):
        ScopeAttribute(name="amount", kind=AttributeKind.NUMERIC_RANGE,
                       min_value=100, max_value=10)


def test_boolean_requires_a_value():
    with pytest.raises(ValidationError, match="requires 'boolean_value'"):
        ScopeAttribute(name="student", kind=AttributeKind.BOOLEAN)


def test_free_text_requires_text():
    with pytest.raises(ValidationError, match="requires 'text'"):
        ScopeAttribute(name="misc", kind=AttributeKind.FREE_TEXT)


# --------------------------------------------------------------------------- #
# Mock generator
# --------------------------------------------------------------------------- #


def test_mock_is_deterministic():
    """Hard requirement: this is the shared fixture both members test against.
    Non-determinism turns a real regression into an argument about whose
    machine is wrong."""
    a = mock.generate(25, seed=7)
    b = mock.generate(25, seed=7)
    assert [x.model_dump_json() for x in a] == [x.model_dump_json() for x in b]


def test_mock_seeds_differ():
    a = mock.generate(25, seed=0)
    b = mock.generate(25, seed=1)
    assert [x.instance_id for x in a] != [x.instance_id for x in b] or \
           [x.model_dump_json() for x in a] != [x.model_dump_json() for x in b]


def test_mock_covers_every_scope_relation():
    """A fixture missing a relation cannot catch confusions involving it --
    notably redundant-vs-refinement, the most damaging one."""
    rels = {i.gold_scope_relation for i in mock.generate(40, seed=0) if i.gold_scope_relation}
    assert rels == set(ScopeRelation)


def test_mock_covers_every_conflict_type():
    types = {i.gold_conflict_type for i in mock.generate(40, seed=0)}
    assert types == set(ConflictType)


def test_mock_covers_both_multi_exception_patterns():
    insts = mock.generate(40, seed=0)
    assert any(i.gold_multi_exception_flags.nested for i in insts)
    assert any(i.gold_multi_exception_flags.crossed for i in insts)


def test_mock_covers_implicit_conditions():
    """Implicit recovery is the hard case Contrastive Scope Probing targets;
    a fixture of only explicit conditions would make it look solved."""
    insts = mock.generate(40, seed=0)
    kinds = {b.explicitness for i in insts for b in i.exception_branches}
    assert len(kinds) == 2


def test_mock_includes_distractors():
    """Without distractors the Spurious-Condition Rate has no denominator."""
    assert any(i.is_distractor for i in mock.generate(40, seed=0))


def test_mock_instances_all_validate():
    for inst in mock.generate(60, seed=3):
        GoldInstance.model_validate(inst.model_dump(mode="json"))


def test_mock_records_all_validate():
    for rec in mock.iter_records(60, seed=3):
        QueryRecord.model_validate(rec)


def test_mock_tier2_fraction_retags_and_shares_documents():
    insts = mock.generate(20, seed=0, tier2_fraction=0.5)
    split = [i for i in insts if i.construction is Construction.SPLIT]
    assert split, "tier2_fraction=0.5 produced no split instances"
    for inst in split:
        doc_ids = {p.document_id for p in inst.passages}
        assert len(doc_ids) == 1, "a Tier-2 instance comes from one source document"


def test_mock_records_survive_a_passage_permutation():
    """Order invariance at the contract level: reversing the passage list must
    not change the pair keys."""
    inst = mock.generate(10, seed=0)[0]
    rec = inst.to_query_record()
    flipped = QueryRecord.model_validate(
        {**rec.model_dump(mode="json"), "passages": list(reversed(rec.model_dump(mode="json")["passages"]))}
    )
    assert {p.key for p in rec.conflict_pairs} == {p.key for p in flipped.conflict_pairs}


# --------------------------------------------------------------------------- #
# CLI smoke tests
# --------------------------------------------------------------------------- #


def test_mock_cli_writes_valid_jsonl(tmp_path):
    out = tmp_path / "m.jsonl"
    rc = mock.main(["--n", "12", "--seed", "0", "--format", "record", "-o", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 12
    for line in lines:
        QueryRecord.model_validate(json.loads(line))


def test_validate_cli_reports_failure(tmp_path):
    from contract.validate import validate_file

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query_id": "x"}\nnot json\n', encoding="utf-8")
    rep = validate_file(bad)
    assert not rep.ok
    assert rep.total == 2
    assert rep.valid == 0
    assert len(rep.errors) == 2


def test_validate_cli_accepts_good_file(tmp_path):
    from contract.validate import validate_file

    good = tmp_path / "good.jsonl"
    mock.main(["--n", "8", "--seed", "0", "--format", "record", "-o", str(good)])
    rep = validate_file(good)
    assert rep.ok
    assert rep.valid == 8


def test_exported_schema_is_current():
    """Guards the freeze rule: a schema change committed without regenerating
    the JSON Schema would leave Member B reviewing a stale agreement."""
    from pathlib import Path

    from contract.export_schema import SCHEMA_DIR

    path = Path(SCHEMA_DIR) / f"query_record-v{CONTRACT_VERSION}.json"
    assert path.exists(), (
        f"{path.name} is missing -- run `python -m contract.export_schema` "
        "after changing the contract"
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    current = QueryRecord.model_json_schema()
    current["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    current["title"] = f"conflict-rag query_record v{CONTRACT_VERSION}"
    assert on_disk == current, (
        "the committed JSON Schema is stale -- run `python -m contract.export_schema`"
    )
