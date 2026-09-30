"""Tier 2: construct split instances from single documents.

The WP1 finding, restated: organisations state a rule and its carve-out
*together*, in one place, which is exactly why cross-document mining yields so
little. That same finding makes Tier 2 straightforward — the material is
abundant, it is just not split. 817 sentences in the mined corpus carry a
strong exception cue.

**What "split" means here, and why it is not cheating.** The proposal defines
Tier 2 as a single-document rule/exception pair deliberately divided across two
synthetic documents. The division is the intervention; the *content* is real
policy text. That isolates the resolution-operator question from the mining
question: a system is handed two passages that genuinely need composing,
without the benchmark having to find them separately published.

Tier 2 instances are never blended with Tier 1 in headline reporting
(`metrics.classification.split_for_reporting`), because a constructed
separation is not evidence that separation occurs naturally.

**Nothing here assigns a label.** A naive implementation would tag every split
`conditional`/`refinement`, since it was built from a rule and its exception —
and that is labelling by construction, which puts the answer in the data.
Annotators label these exactly as they label mined instances, and some will
turn out not to be conditional at all.

**Why an LLM does the splitting.** Cutting at the cue word produces fragments:
"(a) in breach of immigration laws," / "except where the Exceptions for
overstayers section applies." Neither stands alone, and an annotator cannot
judge scope from a fragment. The rewrite turns each half into a self-contained
statement, which is a presentation change, not a content one. Every output is
checked against the source for invented figures.

Usage::

    python -m benchmark.mining.split_constructor \\
        --docs benchmark/data/mined_docs_all.jsonl \\
        --out benchmark/data/tier2_batch.jsonl --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from contract.models import (
    Construction,
    Domain,
    Passage,
    QueryRecord,
    Separation,
    SourceType,
)

#: Shortest a half can be and still stand alone as a rule.
MIN_HALF_CHARS = 40

#: Longest source sentence worth splitting. Beyond this the sentence is usually
#: a list or a table row that lost its formatting, and the halves are not
#: statements.
MAX_SOURCE_CHARS = 340

_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%?")

#: Openings that mark a list item or clause fragment rather than a rule. The
#: text before the cue in these is not a statement, so the split cannot produce
#: one.
_FRAGMENT_START = re.compile(
    r"^\s*(?:\(?[a-z0-9]{1,3}[.)]\s|[-•*]\s|and\b|or\b|but\b|which\b|that\b|"
    r"where\b|whose\b|including\b)", re.IGNORECASE)

_SPLIT_SYSTEM = """You rewrite one policy sentence as two standalone statements.

The sentence states a general rule and a narrower case that is treated
differently. Separate them so each reads as a complete statement on its own,
as if published in two different documents.

Rules:
- Use ONLY what the sentence says. Invent no conditions, figures or outcomes.
- Each statement must stand alone: a reader seeing only one must understand it.
- Keep the original wording wherever it already stands alone.
- State the second as a positive claim about its own cases, NOT as "except..."
  or "unless...". Write what IS true for those cases.
- Do not say which statement overrides the other. Do not add commentary.
- If the sentence does not contain a general rule and a narrower case, return
  {"ok": false}.

Return JSON: {"ok": true, "general": "...", "narrower": "..."}"""


@dataclass(frozen=True)
class SplitCandidate:
    doc_id: str
    provider: str
    kind: str
    url: str
    source: str
    cue: str


def find_candidates(docs: list[dict]) -> list[SplitCandidate]:
    """Sentences that state a rule and a carve-out together."""
    from benchmark.mining.tier1_pilot import sentences
    from detection.features import STRONG_EXCEPTION_CUES

    out: list[SplitCandidate] = []
    seen: set[str] = set()
    for d in docs:
        for s in sentences(d.get("text", "")):
            if not (MIN_HALF_CHARS * 2 <= len(s) <= MAX_SOURCE_CHARS):
                continue
            low = s.lower()
            cue = next((c for c in STRONG_EXCEPTION_CUES if c in low), None)
            if cue is None:
                continue
            before = s[:low.index(cue)].strip()
            # Both halves must have enough substance to become statements, and
            # the first must not be a list item -- "(a) in breach of
            # immigration laws," is not a rule however it is rewritten.
            if len(before) < MIN_HALF_CHARS or _FRAGMENT_START.match(before):
                continue
            key = s.lower()[:90]
            if key in seen:
                continue
            seen.add(key)
            out.append(SplitCandidate(
                doc_id=d["doc_id"], provider=d["provider"], kind=d.get("kind", ""),
                url=d.get("url", ""), source=s, cue=cue))
    return out


def split_sentence(cand: SplitCandidate, *, client=None) -> tuple[str, str] | None:
    """Rewrite one sentence as two standalone statements, or return None."""
    from api_budget.client import Tier, get_client

    client = client or get_client()
    result = client.complete(
        messages=[{"role": "user", "content": f"Sentence: {cand.source}"}],
        system=_SPLIT_SYSTEM,
        tier=Tier.BULK,
        step="tier2_split",
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        raw = result.json() or {}
    except ValueError:
        return None
    if not raw.get("ok"):
        return None

    g, n = str(raw.get("general", "")).strip(), str(raw.get("narrower", "")).strip()
    if len(g) < MIN_HALF_CHARS or len(n) < MIN_HALF_CHARS:
        return None

    # Every figure in the output must appear in the source. A rewrite that
    # introduces a number has invented a rule, and in a policy benchmark a
    # wrong figure is the most damaging thing an instance can carry.
    source_nums = set(_NUM_RE.findall(cand.source))
    for half in (g, n):
        if any(x not in source_nums for x in _NUM_RE.findall(half)):
            return None

    # The narrower half restated as "except..." means the rewrite did not
    # actually separate them -- it just moved the cue.
    if n.lower().startswith(("except", "unless", "other than", "apart from")):
        return None

    # Both halves must state an OUTCOME, not just a condition.
    #
    # The failure this catches: "The applicant is relying on a student loan
    # which meets the requirements of FIN 8.3." That says who the narrower case
    # covers and never says what happens to them -- the source sentence had the
    # outcome (exemption from the 28-day rule) and the rewrite dropped it.
    #
    # Such an instance is still labellable, and that is the problem: it is
    # ambiguous because the construction was lossy rather than because the
    # policy is, and an annotator cannot tell those apart from the inside.
    if not (_states_outcome(g) and _states_outcome(n)):
        return None
    return g, n


#: Words that carry an outcome -- an obligation, a permission, a charge, or the
#: negation of one. A statement with none of these is describing a case rather
#: than saying what follows for it.
_OUTCOME_RE = re.compile(
    r"\b(?:must|may|can|will|shall|should|need|needs|required|permitted|"
    r"allowed|entitled|eligible|exempt|waived|charged|granted|refused|"
    r"applies|apply|lapses?|counts?|qualif\w+|"
    r"(?:do|does|is|are|will|can|may|need)\s+not|no\s+\w+\s+(?:is|are|applies))\b",
    re.IGNORECASE)


def _states_outcome(text: str) -> bool:
    return bool(_OUTCOME_RE.search(text or ""))


def build(
    candidates: list[SplitCandidate],
    *,
    limit: int | None = None,
    prefix: str = "t2",
    client=None,
) -> tuple[list[QueryRecord], list[dict]]:
    """Construct unlabelled Tier-2 records. No relation is assigned."""
    from benchmark.mining.build_batch import _domain, _source_type, write_query

    records: list[QueryRecord] = []
    provenance: list[dict] = []

    for cand in candidates:
        halves = split_sentence(cand, client=client)
        if halves is None:
            continue
        general, narrower = halves
        q = write_query(general, narrower, client=client)
        if q is None:
            continue

        iid = f"{prefix}_{len(records):04d}"
        # One document_id for both passages. That is what makes the instance a
        # split rather than a claim about separate publication, and the
        # contract's `natural` validator relies on it.
        doc = cand.doc_id
        records.append(QueryRecord(
            query_id=iid, query=q, domain=_domain(cand.provider),
            construction=Construction.SPLIT,
            separation=Separation.SYNTHETIC_SPLIT,
            passages=[
                Passage(id="p0", text=general,
                        source_type=_source_type(cand.kind), date=None,
                        document_id=doc, source_url=cand.url),
                Passage(id="p1", text=narrower,
                        source_type=_source_type(cand.kind), date=None,
                        document_id=doc, source_url=cand.url),
            ],
            conflict_pairs=[],
        ))
        provenance.append({
            "instance_id": iid, "doc_id": doc, "provider": cand.provider,
            "cue": cand.cue, "source_sentence": cand.source,
            "constructed": "split_from_single_document",
        })
        if limit and len(records) >= limit:
            break
    return records, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prefix", default="t2")
    ap.add_argument("--survey", action="store_true",
                    help="count candidates without spending API calls")
    args = ap.parse_args(argv)

    docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    cands = find_candidates(docs)
    print(f"{len(cands)} splittable sentences in {len(docs)} documents")

    if args.survey:
        from collections import Counter
        for cue, n in Counter(c.cue for c in cands).most_common(10):
            print(f"  {cue:<24} {n:>4}")
        return 0

    records, provenance = build(cands, limit=args.limit, prefix=args.prefix)
    if not records:
        print("no usable splits -- nothing written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    side = args.out.with_suffix(".provenance.json")
    side.write_text(json.dumps(provenance, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    print(f"\nwrote {len(records)} unlabelled Tier-2 records -> {args.out}")
    print(f"      provenance (source sentences) -> {side}")
    print()
    print("  These carry NO labels. Built from a rule and its carve-out, they")
    print("  will mostly be conditional/refinement -- but tagging them that way")
    print("  would put the answer in the data. Annotators decide, and some will")
    print("  turn out not to be conditional at all.")
    print()
    print(f"  Next:  python -m benchmark.annotation.cli annotate \\")
    print(f"             --annotator <you> --batch {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
