"""Contract v1.3.0: opposition, personas, counterfactual pairs, frames.

Everything new is additive, so the first group of tests is about NOT breaking
what already exists (the 128-instance corpus is stamped v1.2.0 and must keep
validating); the rest pin the new invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contract import CONTRACT_VERSION, ConflictPair, ConflictType, Opposition, ScopeRelation, check_version
from contract.frames import Modality, NormativeFrame, Signature
from contract.gold import Applicability, GoldInstance, Persona, PersonaPair
from contract.mock import generate

CORPUS = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "corpus.jsonl"


def _refinement() -> GoldInstance:
    for inst in generate(60, seed=0):
        if inst.gold_scope_relation is ScopeRelation.REFINEMENT and inst.personas:
            return inst
    raise AssertionError("mock produced no refinement instance with personas")


def _opposed() -> GoldInstance:
    for inst in generate(80, seed=0):
        if inst.gold_scope_relation is ScopeRelation.OPPOSED:
            return inst
    raise AssertionError("mock produced no opposed instance")


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #


def test_version_is_1_3_0():
    assert CONTRACT_VERSION == "1.3.0"


def test_v12_records_are_accepted_non_strict_and_rejected_strict():
    check_version("1.2.0")
    with pytest.raises(Exception):
        check_version("1.2.0", strict=True)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_every_existing_corpus_record_still_validates():
    n = 0
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            GoldInstance.model_validate(json.loads(line))
            n += 1
    assert n == 128


# --------------------------------------------------------------------------- #
# Opposition
# --------------------------------------------------------------------------- #


def test_mock_opposed_instance_is_crossing_and_projects_to_the_wire_record():
    inst = _opposed()
    assert inst.gold_opposition is Opposition.CROSSING
    pair = inst.to_query_record().conflict_pairs[0]
    assert pair.opposition is Opposition.CROSSING


def test_opposition_is_rejected_unless_the_relation_is_opposed():
    inst = _refinement()
    with pytest.raises(ValidationError, match="only meaningful"):
        GoldInstance.model_validate({**inst.model_dump(mode="json"), "gold_opposition": "crossing"})
    with pytest.raises(ValidationError, match="only meaningful"):
        ConflictPair(
            doc_i="p0", doc_j="p1", is_conflict=True, type=ConflictType.CONDITIONAL,
            scope_relation=ScopeRelation.REFINEMENT, opposition=Opposition.SAME_SCOPE, confidence=1.0,
        )


def test_unset_opposition_is_valid_for_an_opposed_pair():
    ConflictPair(
        doc_i="p0", doc_j="p1", is_conflict=True, type=ConflictType.CONDITIONAL,
        scope_relation=ScopeRelation.OPPOSED, confidence=1.0,
    )


# --------------------------------------------------------------------------- #
# Personas and counterfactual pairs
# --------------------------------------------------------------------------- #


def test_mock_personas_resolve_to_branches():
    inst = _refinement()
    assert inst.branch_for_persona("u_default").is_default
    assert not inst.branch_for_persona("u_exception").is_default
    assert {p.flip for p in inst.persona_pairs} == {"relevant", "irrelevant"}


def test_persona_pair_flip_must_match_gold_changes():
    with pytest.raises(ValidationError, match="requires gold_changes"):
        PersonaPair(base_persona_id="a", flipped_persona_id="b", flip="relevant",
                    attribute="card_tier", gold_changes=False)
    with pytest.raises(ValidationError, match="requires gold_changes"):
        PersonaPair(base_persona_id="a", flipped_persona_id="b", flip="irrelevant",
                    attribute="city", gold_changes=True)
    with pytest.raises(ValidationError, match="two different personas"):
        PersonaPair(base_persona_id="a", flipped_persona_id="a", flip="relevant",
                    attribute="x", gold_changes=True)


def test_persona_must_cite_a_real_branch():
    d = _refinement().model_dump(mode="json")
    d["personas"][0]["branch_id"] = "b99"
    with pytest.raises(ValidationError, match="unknown branch"):
        GoldInstance.model_validate(d)


def test_duplicate_persona_ids_are_rejected():
    d = _refinement().model_dump(mode="json")
    d["personas"][1]["persona_id"] = d["personas"][0]["persona_id"]
    with pytest.raises(ValidationError, match="duplicate persona_ids"):
        GoldInstance.model_validate(d)


def test_pair_gold_changes_is_checked_against_the_branches_not_trusted():
    """A hand-edited pair whose flag contradicts the branches would silently
    corrupt the reader-validity numbers, so the instance refuses it."""
    d = _refinement().model_dump(mode="json")
    # u_default and u_default_city both sit in b0, so their outcomes agree;
    # claiming the gold changes must be refused even though flip/gold_changes
    # are internally consistent ('relevant' + True).
    for pair in d["persona_pairs"]:
        if pair["flipped_persona_id"] == "u_default_city":
            pair.update(flip="relevant", gold_changes=True)
    with pytest.raises(ValidationError, match="agree in outcome"):
        GoldInstance.model_validate(d)


def test_pair_cites_unknown_persona():
    d = _refinement().model_dump(mode="json")
    d["persona_pairs"][0]["flipped_persona_id"] = "ghost"
    with pytest.raises(ValidationError, match="unknown persona"):
        GoldInstance.model_validate(d)


def test_personas_without_branches_are_rejected():
    d = _refinement().model_dump(mode="json")
    d["gold_branches"] = []
    d["gold_scope_relation"] = "refinement"
    with pytest.raises(ValidationError):
        GoldInstance.model_validate(d)


# --------------------------------------------------------------------------- #
# Signatures and frames
# --------------------------------------------------------------------------- #


def _sig(**slots) -> Signature:
    return Signature(domain="banking", slots=slots, schema_version="test-1")


def test_signature_is_hashable_and_content_equal():
    a = _sig(event="cash_withdrawal", location="abroad")
    b = _sig(location="abroad", event="cash_withdrawal")
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_signature_slot_cannot_be_both_typed_and_untyped():
    with pytest.raises(ValidationError, match="both typed and untyped"):
        Signature(domain="banking", slots={"event": "transfer"},
                  untyped_slots={"event": "wire"}, schema_version="t")


def test_signature_rejects_blank_values():
    with pytest.raises(ValidationError, match="non-blank"):
        _sig(event="  ")


def test_signature_is_frozen():
    sig = _sig(event="transfer")
    with pytest.raises(ValidationError):
        sig.domain = "immigration"  # type: ignore[misc]


def test_signature_rejects_unknown_domain():
    with pytest.raises(ValidationError):
        Signature(domain="travel", slots={}, schema_version="t")  # type: ignore[arg-type]


def _frame(span: str = "The fee is waived for premium cardholders.", **kw) -> NormativeFrame:
    return NormativeFrame(
        passage_id="p1", signature=_sig(event="cash_withdrawal"),
        applicability=Applicability(descriptor="premium cardholders"),
        outcome="no fee applies", span=span, **kw,
    )


def test_span_verbatim_check_is_whitespace_insensitive_and_exact_otherwise():
    f = _frame()
    assert f.span_is_verbatim_in("Note:  The fee is waived\nfor premium cardholders. Other text.")
    assert not f.span_is_verbatim_in("The fee is waived for platinum cardholders.")


def test_hedged_modality_does_not_determine_an_outcome():
    assert not Modality.HEDGED.determines_outcome
    for m in (Modality.OBLIGATION, Modality.PERMISSION, Modality.PROHIBITION, Modality.STATEMENT):
        assert m.determines_outcome


def test_frame_rejects_empty_span_and_bad_confidence():
    with pytest.raises(ValidationError):
        _frame(span="")
    with pytest.raises(ValidationError):
        _frame(confidence=1.5)


def test_persona_requires_statement():
    with pytest.raises(ValidationError):
        Persona(persona_id="u", statement="", branch_id="b0")
