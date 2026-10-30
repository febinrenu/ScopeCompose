"""Conflict audit: find the conditional conflicts hiding in a document set.

The product question behind the research one. Before anyone asks a question,
which passages in this knowledge base state a rule and an exception to it --
the pairs a selection-based RAG system will silently resolve the wrong way?

1. Candidate pairs: every pair of passages from *different* places that share
   enough vocabulary to be about the same thing (a cheap Jaccard prefilter,
   so the audit stays near-linear on real collections).
2. Stage 1 scores each candidate (NLI + the 17 pair features).
3. The strongest candidates are analysed as if conditional (A4): descriptors,
   set relation, four-way relation, rationale.

The output is a ranked list, each pair with its evidence. It is a screening
tool: stage 1 is the component the bench shows is weakest, so the report says
"worth a look", never "verified".
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from contract.models import ConflictPair, ConflictType, DecidedBy, QueryRecord, ScopeRelation
from service.events import Tracer, in_stage
from service.orchestrator import construction_of, engines

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on for is are be by with as at from this that it its any all "
            "you your we our may can will not no if".split())


#: Words that are the same thing for this purpose. An exception rarely repeats
#: the rule's wording ("fee" / "charge"), so the prefilter compares light stems
#: and a few domain synonyms, not raw strings.
_SAME = {"charge": "fee", "charges": "fee", "fees": "fee", "cost": "fee", "costs": "fee", "payable": "pay",
         "pays": "pay", "paid": "pay", "accounts": "account", "cardholders": "card", "cardholder": "card",
         "cards": "card", "payments": "payment", "withdrawals": "withdrawal", "months": "monthly",
         "month": "monthly", "visas": "visa", "applicants": "applicant", "students": "student"}


def _stem(w: str) -> str:
    w = _SAME.get(w, w)
    return w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


_NEG = re.compile(r"\b(no|not|never|none|without|waived|exempt|free of|n't)\b", re.I)


def contrast(a: str, b: str) -> float:
    """How much two passages read as "X" and "not X": one negates and the other
    does not, over shared content. 0 when both or neither negate; otherwise the
    share of the smaller passage's terms the two have in common.

    The signature of a rule and its carve-out ("a fee applies" / "Premium
    cardholders pay no fee"), and a sharper one than the stage-1 score, which
    the bench shows is near chance offline.
    """
    na, nb = len(_NEG.findall(a)) % 2, len(_NEG.findall(b)) % 2
    if na == nb:
        return 0.0
    ta, tb = _terms(a), _terms(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _risk(d, r: dict[str, Any]) -> str:
    """High: A4 calls it an exception (refinement/opposed) AND the text reads as
    a rule and its carve-out. Medium: one of the two. Low: neither."""
    exceptional = d is not None and d.relation in (ScopeRelation.REFINEMENT, ScopeRelation.OPPOSED)
    reads = r["contrast"] >= 0.5 or r["exception_cues"] >= 1
    if exceptional and reads:
        return "high"
    if exceptional or reads or r["score"] >= 0.5:
        return "medium"
    return "low"


def _position(pid: str) -> int | None:
    """Chunk index within its document (ingest ids are ``doc:index``)."""
    tail = pid.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def run_audit(passages: list, params: dict[str, Any], tracer: Tracer) -> dict[str, Any]:
    live = params.get("mode", "offline") == "live"
    max_pairs = int(params.get("max_pairs") or 300)
    analyse_top = int(params.get("analyse_top") or 12)
    min_overlap = float(params.get("min_overlap") or 0.1)
    E = engines(live)

    with in_stage(tracer, "candidates", "Find candidate pairs"):
        terms = {p.id: _terms(p.text) for p in passages}
        cands = []
        for a, b in combinations(passages, 2):
            pa, pb = _position(a.id), _position(b.id)
            if (a.document_id and a.document_id == b.document_id and pa is not None and pb is not None
                    and abs(pa - pb) > 6):
                continue  # far apart in one document: rarely a rule and its exception
            ta, tb = terms[a.id], terms[b.id]
            if not ta or not tb:
                continue
            j = len(ta & tb) / len(ta | tb)
            if j >= min_overlap:
                cands.append((j, a, b))
        cands.sort(key=lambda x: -x[0])
        total_pairs = len(passages) * (len(passages) - 1) // 2
        cands = cands[:max_pairs]
        tracer.data("candidates", "prefilter", passages=len(passages), pairs_total=total_pairs,
                    candidates=len(cands), min_overlap=min_overlap)
        tracer.log("candidates", f"{len(cands)} of {total_pairs} pair(s) share enough vocabulary to compare")

    scored = []
    with in_stage(tracer, "screen", "Stage 1 on every candidate"):
        for i in range(0, len(cands), 32):
            batch = cands[i:i + 32]
            for (j, a, b) in batch:
                c = E.stage1.score_pairs([a, b])[0]
                p = c.confidence if c.is_conflict else 1.0 - c.confidence
                exc = c.features.exception_cues_max
                k = contrast(a.text, b.text)
                # the detector's score, plus the rule/carve-out contrast and explicit
                # exception cues, so a pair the rule scorer underrates still surfaces
                rank = p + 0.45 * k + 0.15 * min(1.0, exc / 2)
                scored.append({"a": a, "b": b, "overlap": j, "score": p, "rank": rank, "contrast": k,
                               "exception_cues": exc, "escalate": c.needs_escalation,
                               "nli": {"entailment": c.nli.entailment, "contradiction": c.nli.contradiction}})
            tracer.data("screen", "progress", done=min(i + 32, len(cands)), total=len(cands))
        scored.sort(key=lambda r: -r["rank"])

    findings = []
    with in_stage(tracer, "analyse", f"Analyse the top {min(analyse_top, len(scored))} pairs (A4)"):
        for r in scored[:analyse_top]:
            a, b = r["a"], r["b"]
            pa, pb = a.model_copy(update={"id": "p0"}), b.model_copy(update={"id": "p1"})
            rec = QueryRecord(query_id="audit", query=params.get("topic") or "audit", domain="financial_terms",
                              construction=construction_of([a, b]), passages=[pa, pb], conflict_pairs=[ConflictPair(
                                  doc_i="p0", doc_j="p1", is_conflict=True, type=ConflictType.CONDITIONAL,
                                  scope_relation=ScopeRelation.REFINEMENT, confidence=r["score"],
                                  decided_by=DecidedBy.MOCK)])
            _, _, decisions = E.scope.analyse_record(rec)
            d = next(iter(decisions.values()), None)
            names = {"p0": a.id, "p1": b.id}
            finding = {
                "a": {"id": a.id, "text": a.text, "document": a.document_id, "url": a.source_url},
                "b": {"id": b.id, "text": b.text, "document": b.document_id, "url": b.source_url},
                "score": r["score"], "overlap": r["overlap"], "exception_cues": r["exception_cues"],
                "contrast": r["contrast"], "rank": r["rank"],
                "relation": d.relation.value if d else None,
                "set_relation": d.set_relation.value if d else None,
                "outcomes_agree": d.outcomes_agree if d else None,
                "evidence": d.evidence.value if d else None,
                "rationale": d.rationale if d else "",
                "default": names.get(d.default.passage_id) if d else None,
                "exception": names.get(d.exception.passage_id) if d else None,
                "risk": _risk(d, r),
            }
            findings.append(finding)
            tracer.data("analyse", "finding", **finding)

    summary = {"passages": len(passages), "candidates": len(cands), "analysed": len(findings),
               "high": sum(f["risk"] == "high" for f in findings),
               "by_relation": {k: sum(f["relation"] == k for f in findings)
                               for k in ("refinement", "disjoint", "redundant", "opposed")}}
    return summary
