"""The release bundle: what is withheld, what is kept, and that rehydration restores it exactly or says why not."""

from __future__ import annotations

import json

import pytest

from contract.gold import Applicability, Branch, GoldInstance
from contract.models import ConflictType, Construction, Domain, Passage, ScopeRelation
from release.bundle import (
    REDACTED_TEXT,
    DriftError,
    ReleaseError,
    build_release,
    collapse,
    licence,
    load_docs,
    locate,
    main,
    policy,
    quoted_spans,
    redact,
    rehydrate,
    verify,
)

BANK_URL = "https://www.natwest.com/terms/current-account.html"
GOV_URL = "https://www.gov.uk/guidance/fees"
BANK_DOC = ("Preamble about the account.\nForeign currency purchases normally incur a 3% fee on the amount.\n"
            "Other charges are listed below.")
GOV_DOC = "The fee is waived for premium-tier cardholders."
BANK_SENT = "Foreign currency purchases normally incur a 3% fee on the amount."
DOCS = {BANK_URL: BANK_DOC, GOV_URL: GOV_DOC}


def inst(answer="A 3% fee applies. However, if the holder has a premium-tier card, no fee applies.",
         bank_text=BANK_SENT) -> GoldInstance:
    return GoldInstance(
        instance_id="r1", query="Do I pay a fee?", domain=Domain.FINANCIAL_TERMS, construction=Construction.NATURAL,
        passages=[Passage(id="p0", text=bank_text, source_url=BANK_URL, document_id="bank"),
                  Passage(id="p1", text=GOV_DOC, source_url=GOV_URL, document_id="gov")],
        gold_conflict_type=ConflictType.CONDITIONAL, gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_scoped_answer=answer, selection_answer="A 3% fee applies.",
        gold_branches=[
            Branch(branch_id="b0", condition=None, outcome="a 3% fee applies",
                   applicability=Applicability(descriptor="everyone", is_default=True), supporting_passage="p0"),
            Branch(branch_id="b1", condition="the holder has a premium-tier card", outcome="no fee applies",
                   applicability=Applicability(descriptor="premium-tier cardholders"), supporting_passage="p1")])


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url,want", [
    ("https://www.gov.uk/x", "open"), ("https://gov.uk/x", "open"), ("https://example.invalid/doc", "open"),
    (None, "open"), ("", "open"), (BANK_URL, "redact"), ("https://www.monzo.com/legal", "redact"),
    ("https://www.gov.uk.evil.com/x", "redact"), ("https://notgov.uk/x", "redact")])
def test_only_listed_hosts_are_released_and_lookalikes_are_not(url, want):
    assert policy(url) == want


def test_the_licence_is_recorded_for_open_text():
    assert licence(GOV_URL) == "Open Government Licence v3.0"


# --------------------------------------------------------------------------- #
# Locating
# --------------------------------------------------------------------------- #


def test_a_verbatim_passage_is_located_exactly():
    s, e, exact = locate(BANK_SENT, BANK_DOC)
    assert exact and BANK_DOC[s:e] == BANK_SENT


def test_a_rewrapped_passage_is_located_by_collapsed_whitespace_and_bounds_the_original():
    doc = "Intro.\nForeign currency   purchases\nnormally incur a 3% fee\non the amount.\nEnd."
    s, e, exact = locate(BANK_SENT, doc)
    assert not exact and collapse(doc[s:e]) == BANK_SENT


def test_a_passage_absent_from_its_document_is_not_located():
    assert locate("A sentence that is not there.", BANK_DOC) is None


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_bank_text_is_withheld_and_gov_text_is_kept():
    out, recs, rep = redact([inst()], DOCS)
    assert out[0].passages[0].text == REDACTED_TEXT and out[0].passages[1].text == GOV_DOC
    assert len(recs) == 1 and recs[0].url == BANK_URL and rep.n_redacted == 1
    assert BANK_DOC[recs[0].start:recs[0].end] == BANK_SENT and recs[0].n_chars == len(BANK_SENT)
    assert rep.licences == {"Open Government Licence v3.0": 1}


def test_the_released_instance_still_validates_and_keeps_every_annotation():
    out, *_ = redact([inst()], DOCS)
    again = GoldInstance.model_validate(out[0].model_dump(mode="json"))
    assert again.gold_branches == inst().gold_branches and again.gold_scoped_answer == inst().gold_scoped_answer


def test_the_withheld_text_appears_nowhere_in_the_released_instance():
    out, *_ = redact([inst()], DOCS)
    assert "Foreign currency purchases" not in json.dumps(out[0].model_dump(mode="json"))


def test_a_passage_that_cannot_be_located_stops_the_build():
    with pytest.raises(ReleaseError, match="cannot be located"):
        redact([inst()], {BANK_URL: "a different document entirely"})
    with pytest.raises(ReleaseError, match="cannot be located"):
        redact([inst()], {})


def test_redacted_text_quoted_in_another_field_stops_the_build_unless_allowed():
    leaky = inst(answer=f"{BANK_SENT} However, if the holder has a premium-tier card, no fee applies.")
    with pytest.raises(ReleaseError, match="reappear in another field"):
        redact([leaky], DOCS)
    _, _, rep = redact([leaky], DOCS, allow_quotes=True)
    assert rep.quoted and rep.quoted[0]["passage"] == "p0"


def test_a_short_shared_phrase_is_not_a_quotation():
    assert quoted_spans("a 3% fee applies on the amount", ["A 3% fee applies. No fee otherwise."]) == []
    assert quoted_spans(BANK_SENT, [f"x {BANK_SENT} y"]) == [BANK_SENT]


# --------------------------------------------------------------------------- #
# Round trip and drift
# --------------------------------------------------------------------------- #


def test_build_then_rehydrate_restores_the_original_instances_exactly(tmp_path):
    original = [inst()]
    mf = build_release(original, DOCS, tmp_path)
    assert mf["n_redacted"] == 1 and set(mf["files"]) == {"instances.release.jsonl", "redactions.jsonl"}
    restored, rep = rehydrate(tmp_path, DOCS)
    assert restored == original and rep.restored == 1 and rep.document_changed == []


def test_a_rewrapped_document_still_restores_the_same_words(tmp_path):
    doc = BANK_DOC.replace(" normally ", "\nnormally ")
    build_release([inst()], {BANK_URL: doc, GOV_URL: GOV_DOC}, tmp_path)
    restored, _ = rehydrate(tmp_path, {BANK_URL: doc, GOV_URL: GOV_DOC})
    assert collapse(restored[0].passages[0].text) == BANK_SENT


def test_a_changed_passage_is_drift_not_a_silent_restore(tmp_path):
    build_release([inst()], DOCS, tmp_path)
    changed = BANK_DOC.replace("3% fee", "4% fee")
    with pytest.raises(DriftError, match="has changed"):
        rehydrate(tmp_path, {BANK_URL: changed, GOV_URL: GOV_DOC})


def test_a_document_changed_elsewhere_restores_and_is_reported(tmp_path):
    build_release([inst()], DOCS, tmp_path)
    elsewhere = BANK_DOC.replace("Other charges are listed below.", "Other charges are listed on page 9.")
    restored, rep = rehydrate(tmp_path, {BANK_URL: elsewhere, GOV_URL: GOV_DOC})
    assert restored[0].passages[0].text == BANK_SENT and rep.document_changed == ["r1/p0"]


def test_a_missing_document_is_named(tmp_path):
    build_release([inst()], DOCS, tmp_path)
    with pytest.raises(DriftError, match="no document supplied"):
        rehydrate(tmp_path, {})


# --------------------------------------------------------------------------- #
# Manifest integrity
# --------------------------------------------------------------------------- #


def test_editing_a_shipped_file_breaks_the_manifest(tmp_path):
    build_release([inst()], DOCS, tmp_path)
    verify(tmp_path)
    p = tmp_path / "redactions.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"start": ', '"start": 1', 1), encoding="utf-8")
    with pytest.raises(ReleaseError, match="does not match the manifest"):
        verify(tmp_path)
    with pytest.raises(ReleaseError, match="does not match the manifest"):
        rehydrate(tmp_path, DOCS)


def test_the_release_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert build_release([inst()], DOCS, a)["files"] == build_release([inst()], DOCS, b)["files"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_the_cli_round_trips_and_reports_drift_with_a_nonzero_exit(tmp_path, capsys):
    data = tmp_path / "gold.jsonl"
    data.write_text(json.dumps(inst().model_dump(mode="json")) + "\n", encoding="utf-8")
    docs = tmp_path / "docs.jsonl"
    docs.write_text("".join(json.dumps({"url": u, "text": t}) + "\n" for u, t in DOCS.items()), encoding="utf-8")
    bundle = tmp_path / "bundle"
    assert main(["build", "--data", str(data), "--docs", str(docs), "--out", str(bundle)]) == 0
    assert main(["verify", "--bundle", str(bundle)]) == 0
    out = tmp_path / "back.jsonl"
    assert main(["rehydrate", "--bundle", str(bundle), "--docs", str(docs), "--out", str(out)]) == 0
    assert GoldInstance.model_validate(json.loads(out.read_text(encoding="utf-8"))) == inst()
    assert load_docs(docs) == DOCS
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"url": BANK_URL, "text": "gone"}) + "\n"
                   + json.dumps({"url": GOV_URL, "text": GOV_DOC}) + "\n", encoding="utf-8")
    assert main(["rehydrate", "--bundle", str(bundle), "--docs", str(bad), "--out", str(out)]) == 2
    assert "could not be restored" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Typography and unrestorable passages
# --------------------------------------------------------------------------- #


def test_curly_quotes_and_nonbreaking_hyphens_do_not_defeat_location(tmp_path):
    doc = "Intro.\nWe\u2019re removing the fee for Reward\u2011tier cards (General Term 7.4).\nEnd."
    text = "We're removing the fee for Reward-tier cards (General Term 7.4)."
    s, e, exact = locate(text, doc)
    assert not exact and doc[s:e].startswith("We\u2019re") and doc[s:e].endswith("7.4).")
    i = inst(bank_text=text)
    docs = {BANK_URL: doc, GOV_URL: GOV_DOC}
    build_release([i], docs, tmp_path)
    restored, _ = rehydrate(tmp_path, docs)
    assert restored[0].passages[0].text == doc[s:e], "the document's own typography is what is restored"


def test_a_replacement_character_from_a_bad_pdf_decode_still_matches_an_apostrophe():
    assert locate("We're making changes.", "Header\nWe\ufffdre making changes.\n") is not None


def test_unlocated_passages_are_withheld_and_listed_only_when_allowed(tmp_path):
    docs = {BANK_URL: "a different document entirely", GOV_URL: GOV_DOC}
    with pytest.raises(ReleaseError, match="allow_unlocated"):
        build_release([inst()], docs, tmp_path)
    mf = build_release([inst()], docs, tmp_path, allow_unlocated=True)
    assert mf["unrestorable"] == [f"r1/p0 ({BANK_URL})"] and mf["n_redacted"] == 1
    released = (tmp_path / "instances.release.jsonl").read_text(encoding="utf-8")
    assert "Foreign currency purchases" not in released
    restored, rep = rehydrate(tmp_path, docs)
    assert rep.unrestorable == ["r1/p0"] and restored[0].passages[0].text == REDACTED_TEXT
