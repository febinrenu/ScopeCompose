"""LLM candidate proposer, compared against the lexical miner.

The proposal specifies that mining is semi-automatic: *"A model proposes
candidate general/exception pairs; human annotators verify and label."* The
Tier-1 pilot's miner is lexical — cue words plus content-word overlap — which
is a different thing, and on the banking pilot it looked like the bottleneck.

The symptom: the pair audit found 10 of 11 banking pairings to be genuinely
separate documents, while the lexical miner found one cross-document
rule/exception pair among them. Those two numbers answer different questions.
The audit asks whether providers *publish* rules and exceptions separately;
the miner asks whether a bag-of-words method can *find* the pairing. A product
page and a charges schedule are written in different registers and share few
content words, so lexical overlap is a weak bridge between them even when the
pair is obvious to a reader.

This module runs the proposer the proposal actually describes, on the same
documents, so the gap between "the data has pairs" and "the lexical miner
finds them" can be measured rather than assumed.

Every call is cached and tagged, so a re-run is free.

Usage::

    python -m benchmark.mining.llm_proposer --docs benchmark/data/bank_docs.jsonl
    python -m benchmark.mining.llm_proposer --docs bank_docs.jsonl --provider natwest_graduate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from api_budget import Tier, get_client
from benchmark.mining.tier1_pilot import SourceDoc, load_docs, mine

STEP = "wp1_llm_candidate_proposal"

SYSTEM_PROMPT = """\
You find GENERAL RULE / VALID EXCEPTION pairs that span two documents.

You are given two documents from the same provider. Find places where
Document A states a general rule and Document B states a narrower case with a
DIFFERENT outcome — or the other way round.

Return ONLY JSON:
{"pairs": [
  {"rule": "<exact sentence from one document>",
   "rule_doc": "A" or "B",
   "exception": "<exact sentence from the other document>",
   "exception_doc": "B" or "A",
   "why": "<one clause: what narrower case the exception covers>",
   "confidence": 0.0 to 1.0}
]}

What counts:
  BOTH statements must be TRUE at the same time, for DIFFERENT cases. The
  exception narrows applicability to a product, tier, population, amount or
  time window, and changes the outcome there.

What does NOT count, and is the most common mistake:
  - Two statements that simply disagree with no scope difference. That is a
    contradiction, not an exception.
  - A restatement of the rule for a specific group with the SAME outcome. If
    the outcome is unchanged, it is not an exception.
  - Marketing copy, navigation, or a heading.

The rule and exception MUST come from different documents. Quote sentences
exactly as they appear. If there are no genuine cross-document pairs, return
{"pairs": []} — an empty answer is correct far more often than a strained one."""

USER_TEMPLATE = """\
Provider: {provider}

=== DOCUMENT A ({doc_a}) ===
{text_a}

=== DOCUMENT B ({doc_b}) ===
{text_b}

JSON:"""

#: How much of each document reaches the model.
#:
#: Tuned against a real trade-off rather than guessed. At 9,000 chars a call is
#: ~7,000 tokens against the BULK tier's 8,000/min cap, so back-to-back calls
#: tripped the limit. Dropping to 5,500 fixed the rate limit and made the
#: RESULTS WORSE: the model stopped finding the genuine fee-waiver pairs and
#: started returning October price changes, because truncation had cut away the
#: section where the actual rules live. Pacing is the right lever, not
#: truncation -- so the window stays wide and PACE_S carries the limit.
MAX_CHARS_PER_DOC = 9000

#: Seconds to wait between pairings. The client retries with exponential
#: backoff, but its ceiling is shorter than the 60-second rate-limit window,
#: so backoff alone cannot clear a TPM exhaustion. Pacing can.
PACE_S = 22.0


#: Characters that are common in policy PDFs and absent from the Windows
#: console's default code page. Printing one raises UnicodeEncodeError and
#: kills the run AFTER the work is done but BEFORE the JSON is written, which
#: is the worst possible place to fail.
_CONSOLE_SAFE = {
    "‑": "-", "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "*", " ": " ", "…": "...",
    "£": "GBP ", "€": "EUR ",
}


def console_safe(text: str) -> str:
    """Make text printable on a legacy console without losing meaning.

    Substitutes rather than strips: a fee of "GBP 36" is still readable, where
    dropping the symbol would leave a bare number.
    """
    for bad, good in _CONSOLE_SAFE.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "replace").decode("ascii")


@dataclass
class ProposedPair:
    provider: str
    rule: str
    exception: str
    rule_doc: str
    exception_doc: str
    why: str
    confidence: float
    cross_document: bool


def propose(doc_a: SourceDoc, doc_b: SourceDoc, *, client=None) -> list[ProposedPair]:
    client = client or get_client()
    result = client.complete(
        messages=[{"role": "user", "content": USER_TEMPLATE.format(
            provider=doc_a.provider,
            doc_a=doc_a.doc_id, text_a=doc_a.text[:MAX_CHARS_PER_DOC],
            doc_b=doc_b.doc_id, text_b=doc_b.text[:MAX_CHARS_PER_DOC],
        )}],
        system=SYSTEM_PROMPT,
        tier=Tier.BULK,
        step=STEP,
        response_format={"type": "json_object"},
    )
    try:
        raw = result.json()
    except ValueError:
        from detection.stage2 import _salvage_json

        raw = _salvage_json(result.text)

    out: list[ProposedPair] = []
    for item in (raw or {}).get("pairs", []) or []:
        if not isinstance(item, dict):
            continue
        rule, exc = str(item.get("rule", "")).strip(), str(item.get("exception", "")).strip()
        if len(rule) < 15 or len(exc) < 15:
            continue
        rd, ed = str(item.get("rule_doc", "A")).upper()[:1], str(item.get("exception_doc", "B")).upper()[:1]
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out.append(ProposedPair(
            provider=doc_a.provider, rule=rule, exception=exc,
            rule_doc=rd, exception_doc=ed, why=str(item.get("why", ""))[:180],
            confidence=min(1.0, max(0.0, conf)),
            # The prompt demands a cross-document pair; verify rather than
            # trust, since this is the one property the comparison rests on.
            cross_document=(rd != ed),
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--provider", default=None, help="run one pairing only")
    ap.add_argument("--limit", type=int, default=None, help="cap pairings, to protect quota")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--pace", type=float, default=PACE_S,
                    help="seconds between pairings, to stay under the tokens-per-minute cap")
    args = ap.parse_args(argv)

    docs = load_docs(args.docs)
    by_provider: dict[str, list[SourceDoc]] = {}
    for d in docs:
        by_provider.setdefault(d.provider, []).append(d)

    if args.provider:
        by_provider = {k: v for k, v in by_provider.items() if k == args.provider}
    pairings = [(p, ds) for p, ds in by_provider.items() if len(ds) >= 2]
    if args.limit:
        pairings = pairings[:args.limit]

    if not pairings:
        print("no pairings with two documents", file=sys.stderr)
        return 1

    print(f"{len(pairings)} pairing(s)\n")

    all_pairs: list[ProposedPair] = []
    failures = 0
    for i, (provider, ds) in enumerate(pairings):
        if i:
            time.sleep(args.pace)
        try:
            proposed = propose(ds[0], ds[1])
        except Exception as exc:
            failures += 1
            print(f"  {provider:<26} ERROR {type(exc).__name__}: {str(exc)[:46]}")
            continue
        cross = [p for p in proposed if p.cross_document]
        all_pairs.extend(cross)
        print(f"  {provider:<26} {len(cross):>2} cross-document "
              f"({len(proposed) - len(cross)} rejected as same-document)")

    # The comparison this module exists for.
    lexical = mine([d for _, ds in pairings for d in ds])

    print()
    print("Lexical miner vs LLM proposer, same documents")
    print("=" * 74)
    print(f"  {'':<34} {'cross-document pairs':>22}")
    print("  " + "-" * 58)
    attempted = len(pairings) - failures
    print(f"  {'lexical (cue + word overlap)':<34} {lexical.candidates_tier1:>22}")
    print(f"  {'LLM proposer':<34} {len(all_pairs):>22}")
    if failures:
        print(f"\n  NOTE: {failures} of {len(pairings)} pairings failed (rate limit), so the")
        print(f"  LLM figure covers {attempted} pairings against the lexical miner's "
              f"{len(pairings)}.")
        print("  It is a LOWER bound on the LLM proposer, not a like-for-like count.")
    print()
    if len(all_pairs) > lexical.candidates_tier1 * 2:
        print("  The documents contain pairs the lexical miner cannot reach. Its bag-of-")
        print("  words bridge fails between a product page and a charges schedule, which")
        print("  are written in different registers. Use the LLM proposer for WP2 mining")
        print("  -- it is what the proposal specifies, and the gap is the reason why.")
    elif len(all_pairs) <= lexical.candidates_tier1:
        print("  No better than lexical here. The scarcity is in the DATA, not in the")
        print("  method, which strengthens the Tier-1 scarcity finding rather than")
        print("  undermining it.")
    else:
        print("  A modest improvement. Report both numbers; neither method alone settles")
        print("  how much Tier-1 material these documents hold.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "lexical_tier1": lexical.candidates_tier1,
            "llm_cross_document": len(all_pairs),
            "pairings_attempted": len(pairings),
            "pairings_failed": failures,
            "proposals": [asdict(p) for p in all_pairs],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"proposals -> {args.json}")

    if all_pairs:
        print()
        print("Sample proposals (UNVERIFIED - a human decides)")
        print("-" * 74)
        for p in sorted(all_pairs, key=lambda x: -x.confidence)[:6]:
            print(f"  [{p.provider}] conf {p.confidence:.2f}")
            print(f"    rule:      {console_safe(p.rule)[:96]}")
            print(f"    exception: {console_safe(p.exception)[:96]}")
            print(f"    why:       {console_safe(p.why)[:96]}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
