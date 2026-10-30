"""Retrieval-based candidate mining for S1 (plan v4 B-2).

The whole-document proposer (``llm_proposer``) shows a model two documents truncated at
9,000 characters each and asks it to find rule/exception pairs. On the 67 harvested
pairings it found 8 candidates, the *same* 8 however many pairings were added. Long
documents bury the pair, and one model call per pairing cannot look hard at either side.

This miner turns the question around:

1. **Chunk** every document with the shared sentence-aligned chunker.
2. **Rank** cross-document chunk pairs *locally and for free*, by two cheap signals: a
   pair is promising if one chunk carries a rule cue ("must", "fee", "required") and the
   other an exception cue ("unless", "except", "exempt", "waived", "does not apply"), and
   if they speak about the same thing (content-word overlap, which only ranks and never
   filters -- QNFA exists precisely because overlap is a poor judge of alignment).
3. **Verify** only the best ``top_n`` per pairing with the evaluation model, each as a short
   prompt over two chunks, and require the quoted rule and exception sentences to appear
   **verbatim** in their chunks.

Candidates only: every proposal still goes to a human. The proposer runs on the ``EVAL``
tier, never on an evaluated model, because what it produces can reach the test set.

Pairs are formed only between chunks of *different* ``content_id``: two parts of one
guide are the S2 stratum, not S1 (``wp1_tier1_findings.md``).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from api_budget import Tier, get_client
from benchmark.chunker import Chunk, chunk_text

STEP = "retrieval_mining_verify"

_RULE_CUES = re.compile(r"\b(?:must|need(?:s)? to|required|will be charged|fee|charge|cost|pay|"
                        r"only|eligible|entitled|can(?:not)?|may not|not allowed|permitted|limit)\b", re.I)
_EXCEPTION_CUES = re.compile(r"\b(?:unless|except|exempt(?:ed)?|waived|does not apply|do not apply|"
                             r"will not (?:need|have|be charged)|not (?:need|required|charged)|"
                             r"instead|however|if you (?:are|have|hold)|only if|apart from|"
                             r"free of charge|no fee|no charge)\b", re.I)
_STOP = frozenset("""a an the and or of to in on for with at by from is are was were be been it its this that
these those as not no you your we our they their will can may must should would could has have had do does did if
then than there which who whom whose what when where how also any all each other more most such only into over
under about between per""".split())


@dataclass(frozen=True)
class Candidate:
    chunk_a: Chunk
    chunk_b: Chunk
    provider: str
    score: float
    content_id_a: str | None = None
    content_id_b: str | None = None


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in _STOP}


def pair_score(a: str, b: str) -> float:
    """Local ranking score in [0, ~2]: cue complementarity plus topical overlap.

    Complementarity -- a rule cue on one side and an exception cue on the other -- is the main
    signal. Overlap only breaks ties and is capped, so a pair is never promoted *because* the
    wording matches (the property QNFA is built not to depend on)."""
    ra, rb = len(_RULE_CUES.findall(a)) > 0, len(_RULE_CUES.findall(b)) > 0
    ea, eb = len(_EXCEPTION_CUES.findall(a)) > 0, len(_EXCEPTION_CUES.findall(b)) > 0
    comp = 1.0 if (ra and eb) or (rb and ea) else (0.4 if (ra or rb) and (ea or eb) else 0.0)
    if comp == 0.0:
        return 0.0      # no cue complementarity: overlap alone never makes a pair a candidate
    ta, tb = _tokens(a), _tokens(b)
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return comp + 0.5 * min(1.0, overlap * 2)


def candidates_for(docs: list[dict], *, top_n: int = 5, target_words: int = 140) -> list[Candidate]:
    """Ranked cross-document chunk pairs for one pairing (docs sharing a ``provider``)."""
    chunked: list[tuple[dict, list[Chunk]]] = [
        (d, chunk_text(d["doc_id"], d.get("text", ""), target_words=target_words)) for d in docs]
    scored: list[Candidate] = []
    for (da, ca), (db, cb) in ((x, y) for i, x in enumerate(chunked) for y in chunked[i + 1:]):
        cid_a, cid_b = da.get("content_id"), db.get("content_id")
        if cid_a and cid_a == cid_b:
            continue                              # one guide in two parts: S2, not S1
        for a, b in product(ca, cb):
            s = pair_score(a.text, b.text)
            if s > 0:
                scored.append(Candidate(a, b, da.get("provider", ""), round(s, 6), cid_a, cid_b))
    scored.sort(key=lambda c: (-c.score, c.chunk_a.id, c.chunk_b.id))
    return scored[:top_n]


SYSTEM = """\
You check whether two short passages from different documents contain a GENERAL RULE and a valid
EXCEPTION to it: a narrower group of people for whom the outcome is different. Return ONLY JSON:
{"pair": true or false, "rule_in": "A" or "B", "rule": "<one sentence copied exactly from that passage>",
 "exception": "<one sentence copied exactly from the other passage>",
 "why": "<one clause: who the exception covers>", "confidence": 0.0 to 1.0}
Say pair=false unless both sentences are quoted exactly and the exception genuinely narrows the rule.
Two passages that merely mention the same topic are not a pair."""


def _verbatim(sentence: str, text: str) -> bool:
    s = " ".join(sentence.split()).rstrip(".").lower()
    return len(s) >= 15 and s in " ".join(text.split()).lower()


@dataclass
class Verified:
    provider: str
    rule: str
    exception: str
    rule_doc: str
    exception_doc: str
    why: str
    confidence: float
    cross_document: bool
    rule_doc_id: str
    exception_doc_id: str
    rule_content_id: str | None
    exception_content_id: str | None
    rule_chunk: str
    exception_chunk: str
    score: float


def verify(cand: Candidate, *, client=None, tier: Tier = Tier.EVAL) -> Verified | None:
    """Ask the evaluation model; keep the pair only if both quotes are verbatim in their chunks."""
    if tier is not Tier.EVAL:
        raise ValueError("mining that can reach a test set must verify on the EVAL tier")
    c = client or get_client()
    res = c.complete(
        messages=[{"role": "user", "content": f"Passage A:\n{cand.chunk_a.text}\n\nPassage B:\n{cand.chunk_b.text}"}],
        system=SYSTEM, tier=tier, step=STEP, response_format={"type": "json_object"})
    try:
        d = json.loads(res.text)
    except (TypeError, ValueError):
        from detection.stage2 import _salvage_json

        d = _salvage_json(res.text)
    if not isinstance(d, dict) or not d.get("pair"):
        return None
    rule_in = str(d.get("rule_in", "A")).upper()[:1]
    if rule_in not in ("A", "B"):
        return None
    rc, ec = (cand.chunk_a, cand.chunk_b) if rule_in == "A" else (cand.chunk_b, cand.chunk_a)
    rule, exc = str(d.get("rule") or "").strip(), str(d.get("exception") or "").strip()
    if not (_verbatim(rule, rc.text) and _verbatim(exc, ec.text)):
        return None                                # a paraphrase is not a quote: dropped, not repaired
    cids = (cand.content_id_a, cand.content_id_b) if rule_in == "A" else (cand.content_id_b, cand.content_id_a)
    try:
        conf = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return Verified(cand.provider, rule, exc, "A" if rule_in == "A" else "B", "B" if rule_in == "A" else "A",
                    str(d.get("why", ""))[:180], conf, rc.doc_id != ec.doc_id, rc.doc_id, ec.doc_id,
                    cids[0], cids[1], rc.id, ec.id, cand.score)


def mine(docs: list[dict], *, top_n: int = 5, client=None, pace_s: float = 0.0,
         max_candidates: int | None = None, on_result=None) -> dict:
    """Run the miner over every pairing in ``docs``. Returns the same shape as ``llm_proposer``'s JSON."""
    by_provider: dict[str, list[dict]] = {}
    for d in docs:
        by_provider.setdefault(d["provider"], []).append(d)
    pairings = {p: ds for p, ds in by_provider.items() if len(ds) >= 2}
    proposals: list[Verified] = []
    attempted = failed = 0
    for provider, ds in pairings.items():
        for cand in candidates_for(ds, top_n=top_n):
            if max_candidates is not None and attempted >= max_candidates:
                break
            if attempted and pace_s:
                time.sleep(pace_s)
            attempted += 1
            try:
                v = verify(cand, client=client)
            except Exception:
                failed += 1
                continue
            if v is not None and v.cross_document:
                proposals.append(v)
                if on_result:
                    on_result(v)
    uniq = {(p.rule.lower(), p.exception.lower()): p for p in proposals}
    return {"pairings": len(pairings), "candidates_attempted": attempted, "candidates_failed": failed,
            "llm_cross_document": len(uniq), "proposals": [asdict(p) for p in uniq.values()]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=None, help="cap EVAL calls, to protect the daily quota")
    ap.add_argument("--pace", type=float, default=20.0, help="seconds between calls (tokens-per-minute cap)")
    ap.add_argument("--dry-run", action="store_true", help="rank locally and print the candidate count; no model calls")
    args = ap.parse_args(argv)

    docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.dry_run:
        by: dict[str, list[dict]] = {}
        for d in docs:
            by.setdefault(d["provider"], []).append(d)
        n = sum(len(candidates_for(ds, top_n=args.top_n)) for ds in by.values() if len(ds) >= 2)
        print(f"{sum(1 for v in by.values() if len(v) >= 2)} pairings -> {n} candidates "
              f"(~{n * 800:,} tokens on the EVAL tier)")
        return 0
    out = mine(docs, top_n=args.top_n, pace_s=args.pace, max_candidates=args.max_candidates)
    args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{out['candidates_attempted']} candidates -> {out['llm_cross_document']} verified proposals -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
