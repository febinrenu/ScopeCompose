"""Corpus expansion to K-passage contexts: same-site negatives, honest skips, deterministic output."""

from __future__ import annotations

import json

from benchmark.contexts import expand_corpus, main
from contract.gold import Applicability, Branch, GoldInstance
from contract.models import ConflictType, Construction, Domain, Passage, ScopeRelation

SITE = "https://www.example-bank.com"
OTHER = "https://www.other-bank.com"


def doc(doc_id, host, text, provider="p"):
    return {"doc_id": doc_id, "provider": provider, "kind": "page", "url": f"{host}/{doc_id}", "text": text}


def filler(topic, n=8):
    return " ".join(f"The {topic} number {i} is described here in some detail for customers of the bank." for i in range(n))


DOCS = [
    doc("gold_rule", SITE, "Transfers incur a 3% fee on the amount sent abroad every time."),
    doc("gold_exc", SITE, "The transfer fee is waived for premium-tier cardholders sending money abroad."),
    *[doc(f"pad{i}", SITE, filler(f"transfer fee topic {i}")) for i in range(4)],
    *[doc(f"alien{i}", OTHER, filler(f"transfer fee topic {i}")) for i in range(4)],
]


def inst(iid="i1", gold=("gold_rule", "gold_exc"), synthetic=False) -> GoldInstance:
    texts = {"gold_rule": "Transfers incur a 3% fee on the amount sent abroad every time.",
             "gold_exc": "The transfer fee is waived for premium-tier cardholders sending money abroad."}
    passages = [Passage(id=f"p{i}", text=texts[g], document_id=(None if synthetic else g),
                        source_url=("https://example.invalid/x" if synthetic else f"{SITE}/{g}"))
                for i, g in enumerate(gold)]
    return GoldInstance(
        instance_id=iid, query="Do I pay a fee on transfers abroad?", domain=Domain.FINANCIAL_TERMS,
        construction=Construction.NATURAL, passages=passages, gold_conflict_type=ConflictType.CONDITIONAL,
        gold_scope_relation=ScopeRelation.REFINEMENT,
        gold_branches=[Branch(branch_id="b0", condition=None, outcome="a 3% fee applies",
                              applicability=Applicability(descriptor="everyone", is_default=True), supporting_passage="p0"),
                       Branch(branch_id="b1", condition="the holder has a premium-tier card", outcome="no fee applies",
                              applicability=Applicability(descriptor="premium-tier cardholders"),
                              supporting_passage="p1")])


def test_instances_are_expanded_to_k_with_gold_kept_and_negatives_from_the_same_site_only():
    out, rep = expand_corpus([inst()], DOCS, k=6)
    assert rep["expanded"] == ["i1"] and len(out[0].passages) == 6
    assert {"p0", "p1"} <= {p.id for p in out[0].passages}
    negatives = [p for p in out[0].passages if p.id.startswith("n")]
    assert len(negatives) == 4 and all(p.source_url.startswith(SITE) for p in negatives)
    assert not any("gold_" in (p.document_id or "") for p in negatives), "never the gold documents"


def test_untraceable_and_short_instances_are_left_unchanged_and_counted():
    synthetic = inst("syn", synthetic=True)
    out, rep = expand_corpus([synthetic, inst("big")], DOCS, k=6)
    assert rep["untraceable"] == ["syn"] and out[0] == synthetic
    out2, rep2 = expand_corpus([inst("big")], DOCS, k=40)
    assert rep2["short"] == ["big"] and len(out2[0].passages) == 2


def test_the_expansion_is_deterministic_and_order_preserving():
    a, _ = expand_corpus([inst("a"), inst("b")], DOCS, k=6, seed=3)
    b, _ = expand_corpus([inst("a"), inst("b")], DOCS, k=6, seed=3)
    assert a == b and [i.instance_id for i in a] == ["a", "b"]


def test_a_screener_flags_negatives_for_adjudication_without_removing_them():
    out, rep = expand_corpus([inst()], DOCS, k=6, screener=lambda q, t: "topic 1" in t)
    assert rep["flagged"] and len(out[0].passages) == 6
    flagged = rep["flagged"]["i1"]
    assert all(pid in {p.id for p in out[0].passages} for pid in flagged)


def test_the_cli_writes_the_corpus_and_a_report_and_refuses_to_overwrite(tmp_path, capsys):
    data = tmp_path / "c.jsonl"
    data.write_text(json.dumps(inst().model_dump(mode="json")) + "\n", encoding="utf-8")
    docs = tmp_path / "d.jsonl"
    docs.write_text("".join(json.dumps(d) + "\n" for d in DOCS), encoding="utf-8")
    out = tmp_path / "k6.jsonl"
    assert main(["--data", str(data), "--docs", str(docs), "--out", str(out)]) == 0
    assert len(GoldInstance.model_validate(json.loads(out.read_text(encoding="utf-8"))).passages) == 6
    assert json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))["expanded"] == ["i1"]
    assert main(["--data", str(data), "--docs", str(docs), "--out", str(data)]) == 2
