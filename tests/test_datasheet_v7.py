"""The generated datasheet: numbers come from the data, proposals are never counted as human labels."""

from __future__ import annotations

import json

from benchmark.datasheet_v7 import annotators_for, main, provenance_counts, render, review_rows
from benchmark.preannotate import set_branch_provenance
from benchmark.review import Decision, ReviewStore
from contract.mock import generate


def insts(n=40):
    return generate(n, seed=0)


def with_provenance(inst, value):
    d = inst.model_dump(mode="json")
    ann = d.get("annotation") or {}
    ann["notes"] = set_branch_provenance(ann.get("notes"), value)
    d["annotation"] = ann
    return type(inst).model_validate(d)


def decide(directory, kind, annotator, verdicts, seconds=30.0):
    st = ReviewStore(directory, kind, annotator)
    for iid, v in verdicts.items():
        st.append(Decision(iid, kind, annotator, v, seconds=seconds))


def test_provenance_separates_proposals_from_reviewed_and_hand_labels():
    base = [i for i in insts() if i.gold_branches][:6]
    assert len(base) == 6
    mixed = [with_provenance(base[0], "model-proposed"), with_provenance(base[1], "model-proposed"),
             with_provenance(base[2], "reviewed:alice"), base[3]]
    c = provenance_counts(mixed)
    assert c["model-proposed, not yet reviewed"] == 2 and c["reviewed by a person"] == 1
    assert c["annotated by hand (no proposal recorded)"] == 1
    no_branches = [i for i in insts() if not i.gold_branches][:2]
    assert provenance_counts(no_branches)["no branches"] == len(no_branches)


def test_review_rows_count_decisions_hours_and_kappa_per_pair(tmp_path):
    ids = {f"c{i}": v for i, v in enumerate(["accept", "accept", "reject", "accept", "reject", "accept",
                                              "reject", "accept", "accept", "reject"])}
    other = dict(ids) | {"c0": "reject"}
    decide(tmp_path, "candidates", "alice", ids, seconds=36)
    decide(tmp_path, "candidates", "bob", other, seconds=72)
    assert annotators_for(tmp_path, "candidates") == ["alice", "bob"]
    rows = review_rows(tmp_path)
    alice = next(r for r in rows if r.get("annotator") == "alice")
    assert alice["decisions"] == 10 and abs(alice["hours"] - 0.1) < 1e-9
    pair = next(r for r in rows if "pair" in r)
    assert pair["pair"] == "alice / bob" and pair["n"] == 10 and 0 < pair["kappa"] < 1 and not pair["degenerate"]


def test_one_annotator_reports_hours_but_no_agreement(tmp_path):
    decide(tmp_path, "vacs", "alice", {"v1": "equivalent", "v2": "distinct"})
    text = render(insts(), tmp_path)
    assert "only one annotator so far" in text and "| vacs | alice | 2 |" in text


def test_a_unanimous_single_category_batch_is_undefined_not_a_perfect_score(tmp_path):
    same = {f"c{i}": "accept" for i in range(8)}
    decide(tmp_path, "candidates", "alice", same)
    decide(tmp_path, "candidates", "bob", same)
    text = render(insts(), tmp_path)
    assert "undefined" in text and "1.000" not in text


def test_nothing_recorded_says_so_rather_than_printing_zeros(tmp_path):
    text = render(insts(), tmp_path)
    assert "No review decisions recorded yet." in text and "0.00" not in text


def test_the_tier_and_strata_tables_and_the_narrowing_rule_are_present(tmp_path):
    text = render(insts(), tmp_path)
    for needle in ("Tier 1 `natural`", "Strata x split", "S1 conditional conflicts in test", "Label provenance"):
        assert needle in text


def test_the_release_section_reports_withheld_passages(tmp_path):
    mf = {"n_instances": 128, "n_passages": 256, "n_redacted": 48, "unrestorable": ["a", "b"],
          "open_passages_by_licence": {"Open Government Licence v3.0": 100}}
    text = render(insts(), tmp_path, release_manifest=mf)
    assert "withheld (not redistributable): **48**" in text and "unrestorable after redaction: **2**" in text
    assert "## Release" not in render(insts(), tmp_path)


def test_the_cli_writes_the_file(tmp_path):
    data = tmp_path / "g.jsonl"
    data.write_text("".join(json.dumps(i.model_dump(mode="json")) + "\n" for i in insts(10)), encoding="utf-8")
    out = tmp_path / "d.md"
    assert main(["--data", str(data), "--out", str(out), "--review-dir", str(tmp_path / "none")]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Datasheet: generated sections")
