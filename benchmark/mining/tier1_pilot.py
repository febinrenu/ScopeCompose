"""WP1: the Tier-1 natural multi-source mining pilot.

**Gating. Due end of week 4.**

The question this answers is narrow and load-bearing: *how often does a general
rule and its valid exception actually live in two separately retrievable
documents?*

It matters because "naturally occurring, multi-source" is the claim that
distinguishes this benchmark from ConditionalQA, which is a single-document
task. If real providers overwhelmingly state a rule and its carve-out on the
same page, the multi-source framing cannot be satisfied at the target scale,
and the corpus has to be built as an explicit Tier 1 / Tier 2 mix instead.

Better to learn that from twenty documents in week 3 than from two hundred in
week 9. The output of this script is a yield estimate and a recommendation, not
a corpus.

Usage::

    python -m benchmark.mining.tier1_pilot --docs docs.jsonl --report pilot.json
    python -m benchmark.mining.tier1_pilot --demo     # synthetic walkthrough
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from detection.features import EXCEPTION_CUES, RESTRICTION_CUES

#: The floor from the proposal: below this, Tier 2 must make up the remainder
#: of the 250-instance target and the paper states the composition explicitly.
TIER1_FLOOR = 150
TARGET_INSTANCES = 250

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SourceDoc:
    """One mined document."""

    doc_id: str
    provider: str
    """Which organisation published it. Two documents from DIFFERENT providers
    are not a rule/exception pair -- they are two unrelated policies."""
    kind: str
    """primary_terms | amendment | faq | category_page | other."""
    url: str | None = None
    text: str = ""


@dataclass
class CandidatePair:
    """A possible rule/exception pair spanning two documents."""

    provider: str
    rule_doc: str
    rule_sentence: str
    exception_doc: str
    exception_sentence: str
    cue_matched: str
    same_document: bool = False

    @property
    def is_tier1(self) -> bool:
        return not self.same_document


@dataclass
class PilotReport:
    documents_scanned: int = 0
    providers: int = 0
    candidates_total: int = 0
    candidates_tier1: int = 0
    candidates_tier2: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)

    @property
    def tier1_share(self) -> float:
        return self.candidates_tier1 / self.candidates_total if self.candidates_total else 0.0

    @property
    def yield_per_document(self) -> float:
        return self.candidates_tier1 / self.documents_scanned if self.documents_scanned else 0.0

    def documents_needed(self, target: int = TIER1_FLOOR) -> float | None:
        """How many documents to scan to hit the Tier-1 floor at this yield."""
        y = self.yield_per_document
        return (target / y) if y > 0 else None

    def recommendation(self) -> str:
        if self.documents_scanned < 5:
            return ("Too few documents scanned to estimate anything. Run the pilot on "
                    "at least 20 before drawing a conclusion.")
        if self.candidates_tier1 == 0:
            return (
                "ZERO naturally multi-source pairs found. If this holds at 20 documents, "
                "the Tier-1 framing cannot be satisfied at scale: commit to the "
                "Tier 1 / Tier 2 split now, and say so in the paper rather than "
                "presenting a single undifferentiated count."
            )
        needed = self.documents_needed()
        assert needed is not None
        if needed <= 200:
            return (
                f"Yield is {self.yield_per_document:.2f} Tier-1 pairs per document. "
                f"Reaching the {TIER1_FLOOR}-instance floor needs roughly {needed:.0f} "
                f"documents -- feasible. Proceed with Tier 1 as the primary corpus."
            )
        if needed <= 600:
            return (
                f"Yield is {self.yield_per_document:.2f} per document; roughly "
                f"{needed:.0f} documents would be needed for the {TIER1_FLOOR} floor. "
                f"Tight but possible. Plan for Tier 2 to make up the remainder of the "
                f"{TARGET_INSTANCES}-instance target, and budget the mining time honestly."
            )
        return (
            f"Yield is only {self.yield_per_document:.2f} per document; the "
            f"{TIER1_FLOOR} floor would need about {needed:.0f} documents. That is not "
            f"realistic in WP2. Commit to the Tier 1 / Tier 2 split, report the "
            f"composition explicitly, and do NOT let Tier 2 instances carry the "
            f"naturally-occurring claim."
        )

    def render(self) -> str:
        lines = [
            "Tier-1 mining pilot (WP1, gating)",
            "=" * 72,
            f"  documents scanned    {self.documents_scanned:>8,}",
            f"  providers            {self.providers:>8,}",
            f"  candidate pairs      {self.candidates_total:>8,}",
            f"    Tier 1 (separate documents)  {self.candidates_tier1:>6,}",
            f"    Tier 2 (same document)       {self.candidates_tier2:>6,}",
            f"  Tier-1 share         {self.tier1_share:>8.1%}",
            f"  yield per document   {self.yield_per_document:>8.2f}",
        ]
        needed = self.documents_needed()
        if needed is not None:
            lines.append(f"  docs for {TIER1_FLOOR}-floor    {needed:>8.0f}")
        lines += ["", "Recommendation", "-" * 72]
        for chunk in _wrap(self.recommendation(), 70):
            lines.append("  " + chunk)
        if self.examples:
            lines += ["", "Sample Tier-1 candidates", "-" * 72]
            for ex in self.examples[:5]:
                lines.append(f"  [{ex['provider']}] {ex['rule_doc']} -> {ex['exception_doc']}")
                lines.append(f"    rule:      {ex['rule_sentence'][:90]}")
                lines.append(f"    exception: {ex['exception_sentence'][:90]}")
                lines.append("")
        return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 20]


def find_exception_sentences(text: str) -> list[tuple[str, str]]:
    """Sentences that look like they carve out an exception.

    Deliberately high-recall. This is a candidate generator feeding human
    verification, so a false positive costs an annotator ten seconds and a
    false negative costs a real instance -- the asymmetry says be permissive.
    """
    out = []
    for sent in sentences(text):
        low = sent.lower()
        for cue in EXCEPTION_CUES:
            if cue in low:
                out.append((sent, cue))
                break
        else:
            # An explicit population restriction with a modal is an exception
            # in substance even with no cue word present -- which is the
            # implicit case the whole project is about.
            if any(r in low for r in RESTRICTION_CUES) and re.search(
                r"\b(may|are|is|will|shall)\b", low
            ):
                out.append((sent, "restriction_without_cue"))
    return out


def find_rule_sentences(text: str) -> list[str]:
    """Sentences that state a general rule: no exception cue, a modal or an
    outcome, and no narrowing restriction."""
    out = []
    for sent in sentences(text):
        low = sent.lower()
        if any(c in low for c in EXCEPTION_CUES):
            continue
        if any(r in low for r in RESTRICTION_CUES):
            continue
        if re.search(r"\b(incur|applies|apply|must|shall|may not|will be|are charged|"
                     r"is charged|required|prohibited|permitted)\b", low):
            out.append(sent)
    return out


def mine(docs: list[SourceDoc], *, max_examples: int = 10) -> PilotReport:
    """Find candidate rule/exception pairs and measure the Tier-1 share."""
    report = PilotReport(documents_scanned=len(docs))

    by_provider: dict[str, list[SourceDoc]] = {}
    for d in docs:
        by_provider.setdefault(d.provider, []).append(d)
    report.providers = len(by_provider)

    for provider, group in by_provider.items():
        rules = [(d, s) for d in group for s in find_rule_sentences(d.text)]
        exceptions = [(d, s, cue) for d in group for s, cue in find_exception_sentences(d.text)]

        for rule_doc, rule_sent in rules:
            for exc_doc, exc_sent, cue in exceptions:
                if rule_sent == exc_sent:
                    continue
                pair = CandidatePair(
                    provider=provider,
                    rule_doc=rule_doc.doc_id, rule_sentence=rule_sent,
                    exception_doc=exc_doc.doc_id, exception_sentence=exc_sent,
                    cue_matched=cue,
                    same_document=(rule_doc.doc_id == exc_doc.doc_id),
                )
                report.candidates_total += 1
                if pair.is_tier1:
                    report.candidates_tier1 += 1
                    report.by_provider[provider] = report.by_provider.get(provider, 0) + 1
                    if len(report.examples) < max_examples:
                        report.examples.append(asdict(pair))
                else:
                    report.candidates_tier2 += 1

    return report


def load_docs(path: Path) -> list[SourceDoc]:
    docs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(SourceDoc(**json.loads(line)))
    return docs


def demo_docs() -> list[SourceDoc]:
    """A small synthetic set, so the harness can be exercised before real
    documents are collected. Not data."""
    return [
        SourceDoc("acme_terms", "acme", "primary_terms", text=(
            "International transactions incur a 3% fee. "
            "Statements are issued on the first business day of each month. "
            "All cardholders must notify us of a change of address.")),
        SourceDoc("acme_faq", "acme", "faq", text=(
            "Do premium cardholders pay the international fee? "
            "The international transaction fee is waived for premium-tier cardholders. "
            "Contact support for more information.")),
        SourceDoc("beta_terms", "beta", "primary_terms", text=(
            "A monthly maintenance charge applies to all current accounts. "
            "The charge does not apply to accounts held by students under 23.")),
        SourceDoc("gamma_policy", "gamma", "primary_terms", text=(
            "An F-1 holder may not accept employment in the United States.")),
        SourceDoc("gamma_guidance", "gamma", "amendment", text=(
            "An F-1 holder may accept employment provided that the student has been "
            "granted economic hardship authorisation.")),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--docs", type=Path, help="JSONL of SourceDoc records")
    src.add_argument("--demo", action="store_true", help="run on a synthetic set")
    ap.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    docs = demo_docs() if args.demo else load_docs(args.docs)
    if args.demo:
        print("DEMO MODE: synthetic documents. Exercises the harness; measures nothing.\n")

    report = mine(docs)
    print(report.render())

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({**asdict(report),
                        "tier1_share": report.tier1_share,
                        "yield_per_document": report.yield_per_document,
                        "recommendation": report.recommendation()},
                       indent=2),
            encoding="utf-8",
        )
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
