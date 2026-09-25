"""Turn mined candidates into a batch a human can actually label.

The gap this fills. ``llm_proposer`` emits ``{rule, exception, provider, why,
confidence}`` -- strings describing a candidate. The annotation CLI consumes
``QueryRecord`` objects with identified passages. Nothing connected the two, so
neither the kappa re-run nor WP2 corpus building could start: candidates existed
and were unlabellable.

**What this deliberately does not do.** It writes no labels. Every record comes
out with an empty ``conflict_pairs`` list and no gold anything. The proposer's
own opinion -- its ``why`` and ``confidence`` -- is dropped rather than carried
into the batch, because an annotator who can see what the model thought is no
longer an independent observer, and the kappa computed afterwards would measure
the proposal. The proposal's job ended when it decided this pair was worth a
human's attention.

The provenance is kept in a **separate sidecar file** so the audit trail
survives without travelling alongside the thing being judged.

Usage::

    python -m benchmark.mining.build_batch \\
        --proposals benchmark/data/llm_proposals_govuk.json \\
        --docs benchmark/data/govuk_docs.jsonl \\
        --out benchmark/data/pilot2_batch.jsonl --limit 30
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

#: Providers whose material is immigration rather than financial. Everything
#: else falls back to financial terms, which is the larger half of the corpus.
_IMMIGRATION_HINTS = ("gov_uk", "visa", "home_office", "immigration", "settle")

_SOURCE_BY_KIND = {
    "guide": SourceType.GOVERNMENT_GUIDANCE,
    "detailed_guide": SourceType.GOVERNMENT_GUIDANCE,
    "terms": SourceType.PRODUCT_TERMS,
    "tariff": SourceType.OFFICIAL_POLICY,
    "faq": SourceType.FAQ,
    "product": SourceType.PRODUCT_TERMS,
}

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    provider: str
    kind: str
    url: str
    text: str
    content_id: str | None = None


def load_docs(path: Path) -> list[SourceDoc]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(SourceDoc(
            doc_id=d["doc_id"], provider=d["provider"], kind=d.get("kind", ""),
            url=d.get("url", ""), text=d.get("text", ""),
            content_id=d.get("content_id"),
        ))
    return out


def _domain(provider: str) -> Domain:
    low = provider.lower()
    return (Domain.IMMIGRATION_ELIGIBILITY
            if any(h in low for h in _IMMIGRATION_HINTS)
            else Domain.FINANCIAL_TERMS)


def _source_type(kind: str) -> SourceType:
    return _SOURCE_BY_KIND.get(kind.lower(), SourceType.UNKNOWN)


def _clean(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _locate(snippet: str, docs: list[SourceDoc]) -> SourceDoc | None:
    """Which document a proposed snippet came from.

    The proposer quotes and lightly paraphrases, so exact matching fails often.
    Falls back to the document sharing the most distinctive words with the
    snippet -- good enough to attribute provenance, and the annotator sees the
    snippet itself rather than this guess.
    """
    s = _clean(snippet).lower()
    if not s:
        return None
    for d in docs:
        if s[:60] and s[:60] in _clean(d.text).lower():
            return d

    words = {w for w in re.findall(r"[a-z]{5,}", s)}
    if not words:
        return None
    best, best_score = None, 0.0
    for d in docs:
        dw = set(re.findall(r"[a-z]{5,}", _clean(d.text).lower()))
        score = len(words & dw) / len(words)
        if score > best_score:
            best, best_score = d, score
    return best if best_score >= 0.5 else None


def build(
    proposals: list[dict],
    docs: list[SourceDoc],
    *,
    limit: int | None = None,
    prefix: str = "wp2",
) -> tuple[list[QueryRecord], list[dict]]:
    """Return ``(records, provenance)``. Records carry no labels of any kind."""
    by_provider: dict[str, list[SourceDoc]] = {}
    for d in docs:
        by_provider.setdefault(d.provider, []).append(d)

    records: list[QueryRecord] = []
    provenance: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for p in proposals:
        rule, exc = _clean(p.get("rule", "")), _clean(p.get("exception", ""))
        if not rule or not exc:
            continue
        key = (rule.lower()[:80], exc.lower()[:80])
        if key in seen:
            continue
        seen.add(key)

        provider = p.get("provider", "unknown")
        pool = by_provider.get(provider, docs)
        by_id = {d.doc_id: d for d in docs}

        # Prefer the ids the proposer recorded. Word-overlap location is a
        # fallback for older proposal files, and a poor one: a long document
        # shares vocabulary with everything, so both snippets get attributed to
        # whichever document is biggest. On the first run of this builder that
        # produced "same guide 4/4" on a bank corpus, which was the locator
        # collapsing rather than a property of the data.
        located_by = "recorded"
        d_rule = by_id.get(p.get("rule_doc_id") or "")
        d_exc = by_id.get(p.get("exception_doc_id") or "")
        if d_rule is None or d_exc is None:
            located_by = "word-overlap guess"
            d_rule = d_rule or _locate(rule, pool)
            d_exc = d_exc or _locate(exc, pool)

        iid = f"{prefix}_{len(records):04d}"

        # Three distinct situations, and collapsing them loses the one the WP1
        # pilot was about.
        same_document = (
            d_rule is not None and d_exc is not None
            and d_rule.doc_id == d_exc.doc_id
        )
        same_guide = (
            not same_document
            and d_rule is not None and d_exc is not None
            and d_rule.content_id is not None
            and d_rule.content_id == d_exc.content_id
        )

        records.append(QueryRecord(
            query_id=iid,
            # A placeholder the annotator replaces. Inventing a specific
            # question here would frame the pair before anyone has read it.
            query="[to be written by the annotator]",
            domain=_domain(provider),
            construction=(Construction.NATURAL if (not same_document and not same_guide)
                          else Construction.SPLIT),
            # Two URLs inside one guide is real retrieval separation that nobody
            # manufactured -- the third category the WP1 pilot found. A pair
            # located entirely inside ONE document is neither: it never crossed
            # a retrieval boundary at all, and is tagged same_guide only so it
            # cannot be miscounted as Tier 1. Those are flagged for review
            # rather than silently admitted.
            separation=(Separation.CROSS_DOCUMENT
                        if (not same_document and not same_guide)
                        else Separation.SAME_GUIDE),
            passages=[
                Passage(id="p0", text=rule,
                        source_type=_source_type(d_rule.kind if d_rule else ""),
                        date=None,
                        document_id=(d_rule.doc_id if d_rule else f"{provider}_a"),
                        source_url=(d_rule.url if d_rule else None)),
                Passage(id="p1", text=exc,
                        source_type=_source_type(d_exc.kind if d_exc else ""),
                        date=None,
                        document_id=(d_exc.doc_id if d_exc else f"{provider}_b"),
                        source_url=(d_exc.url if d_exc else None)),
            ],
            conflict_pairs=[],
        ))

        # Everything the model thought, kept OUT of the batch.
        provenance.append({
            "instance_id": iid, "provider": provider,
            "proposer_why": p.get("why"), "proposer_confidence": p.get("confidence"),
            "rule_doc": d_rule.doc_id if d_rule else None,
            "exception_doc": d_exc.doc_id if d_exc else None,
            "same_guide": same_guide,
            "same_document": same_document,
            "located_by": located_by,
            "located": d_rule is not None and d_exc is not None,
            "needs_review": same_document or located_by != "recorded",
        })

        if limit and len(records) >= limit:
            break

    return records, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prefix", default="wp2")
    args = ap.parse_args(argv)

    raw = json.loads(args.proposals.read_text(encoding="utf-8"))
    proposals = raw.get("proposals", raw) if isinstance(raw, dict) else raw
    docs = load_docs(args.docs)

    records, provenance = build(proposals, docs, limit=args.limit,
                                prefix=args.prefix)
    if not records:
        print("no usable proposals -- nothing written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")

    side = args.out.with_suffix(".provenance.json")
    side.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    n = len(records)
    guessed = sum(1 for p in provenance if p["located_by"] != "recorded")
    same_doc = sum(1 for p in provenance if p["same_document"])
    same_guide = sum(1 for p in provenance if p["same_guide"])
    cross = n - same_doc - same_guide

    print(f"wrote {n} unlabelled records -> {args.out}")
    print(f"      provenance (kept separate)  -> {side}")
    print()
    print(f"  cross-document (Tier 1 candidate) {cross}/{n}")
    print(f"  same guide, two URLs              {same_guide}/{n}")
    print(f"  SAME DOCUMENT                     {same_doc}/{n}")
    if same_doc:
        print("      ^ these never crossed a retrieval boundary. Either the")
        print("        proposer found an intra-document pair, or provenance was")
        print("        guessed wrongly. Review before admitting them.")
    if guessed:
        print()
        print(f"  provenance GUESSED by word overlap  {guessed}/{n}")
        print("      Re-run the proposer to record real document ids; a long")
        print("      document shares vocabulary with everything, so the guess")
        print("      attributes both snippets to whichever document is biggest.")
    print()
    print("  These carry NO labels and no trace of what the proposer thought.")
    print("  That is deliberate: an annotator who can see the model's guess is")
    print("  no longer an independent observer, and the kappa would measure the")
    print("  proposal rather than the task.")
    print()
    print("  Next:  python -m benchmark.annotation.cli annotate \\")
    print(f"             --annotator <you> --batch {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
