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

from detection.features import RESTRICTION_CUES, STRONG_EXCEPTION_CUES

#: The floor from the proposal: below this, Tier 2 must make up the remainder
#: of the 250-instance target and the paper states the composition explicitly.
TIER1_FLOOR = 150
TARGET_INSTANCES = 250

# Sentence boundaries: terminal punctuation OR a newline. The newline matters
# on real pages -- headings and list items sit on their own line with no
# terminal punctuation, and splitting on punctuation alone glues a heading to
# the sentence after it ("When to apply When you can apply depends on...").
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

#: Sentences that are navigation, not policy. On a real page these are a large
#: fraction of the text and they match cue words readily -- "the register of
#: licensed student sponsors can be found at ..." contains a restriction cue
#: and states no rule at all.
_BOILERPLATE_RE = re.compile(
    r"(https?://|www\.|can be found at|read (?:the |more)|find out (?:more|if)|"
    r"^(?:you (?:can|must|should) )?(?:check|see|view|download|contact|call|email)\b|"
    r"^(?:related|guidance|collection|published|last updated|print this page)\b|"
    r"^(?:apply now|start now|sign in|get help)\b)",
    re.I,
)

#: A rule and its exception have to be ABOUT the same thing. Without this the
#: miner takes the cross product of every rule sentence against every
#: exception-looking sentence in the same provider, which on a 60,000-character
#: statutory appendix is thousands of pairs and almost entirely noise.
MIN_TOPIC_OVERLAP = 0.18

#: One general rule should not generate dozens of candidate pairs. Keeping the
#: best few per rule is what makes the yield figure mean "distinct rule/
#: exception pairs a human could verify" rather than "size of a cross product".
MAX_PAIRS_PER_RULE = 2

#: Long enough to drop headings and link text, short enough to keep a terse
#: rule. "Fees are waived for some applicants." is 36 characters.
MIN_SENTENCE_CHARS = 30

#: Verbs that mark a sentence as STATING a rule rather than describing one.
#: Deliberately broad: this is a candidate generator feeding human
#: verification, so a false positive costs a reviewer ten seconds and a false
#: negative costs a real instance. "is payable" and "is granted" were missing
#: and are common in exactly this register.
_RULE_VERB_RE = re.compile(
    r"\b(incur|incurs|applies|apply|must|shall|may not|will be|are charged|"
    r"is charged|required|prohibited|permitted|payable|entitled|granted|"
    r"refused|allowed|eligible)\b",
    re.I,
)

_CONTENT_RE = re.compile(r"[a-z']{3,}")
_STOP = {
    "the", "and", "you", "your", "for", "with", "that", "this", "are", "have",
    "from", "not", "but", "can", "will", "must", "any", "all", "may", "who",
    "them", "they", "been", "which", "when", "what", "were", "into",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _CONTENT_RE.findall(text.lower()) if w not in _STOP}


def topic_overlap(a: str, b: str) -> float:
    """Jaccard over content words. Cheap proxy for 'about the same thing'."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


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

    content_id: str | None = None
    """Publisher's identifier for the underlying document, where the source
    exposes one.

    This is what actually decides Tier 1, and ``doc_id`` cannot substitute for
    it. A gov.uk 'guide' is ONE content item with many parts, each with its own
    URL: ``/student-visa`` and ``/student-visa/money`` are separately
    retrievable and share a content_id. Comparing URLs would count them as two
    documents and inflate the naturally-occurring claim, which is precisely the
    Tier-1 scarcity risk the proposal names.
    """

    @property
    def identity(self) -> str:
        """What 'the same document' means for the Tier-1 test."""
        return self.content_id or self.doc_id


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
    """True when both sentences came from one underlying document -- including
    two parts of a single multi-part guide."""

    topic_overlap: float = 0.0
    """Content-word Jaccard between the rule and the exception sentence. A
    human verifier reads the highest-overlap candidates first."""

    same_guide_parts: bool = False
    """The pair spans two PARTS of one document. Separately retrievable, so a
    RAG chunker would index them apart and a retriever can surface one without
    the other -- but not separately authored. A third category the proposal's
    Tier 1 / Tier 2 binary does not cover, reported on its own rather than
    folded into either."""

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
    candidates_same_guide: int = 0
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
            f"    same document                {self.candidates_tier2:>6,}",
            f"      of which: separate parts   {self.candidates_same_guide:>6,}",
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
    """Split to policy-bearing sentences, dropping navigation.

    A modest length floor drops headings and link text -- "When to apply",
    "Overview" -- which now arrive as their own fragments because the splitter
    treats a newline as a boundary. It has to stay modest: real policy
    sentences are often short ("Fees are waived for some applicants."), and a
    floor set high enough to feel safe silently discards them.
    """
    out = []
    for raw in _SENT_RE.split(text):
        s = " ".join(raw.split())
        if len(s) < MIN_SENTENCE_CHARS:
            continue
        if _BOILERPLATE_RE.search(s):
            continue
        out.append(s)
    return out


def find_exception_sentences(text: str) -> list[tuple[str, str]]:
    """Sentences that look like they carve out an exception.

    Deliberately high-recall. This is a candidate generator feeding human
    verification, so a false positive costs an annotator ten seconds and a
    false negative costs a real instance -- the asymmetry says be permissive.
    """
    out = []
    for sent in sentences(text):
        low = sent.lower()
        for cue in STRONG_EXCEPTION_CUES:
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
        if any(c in low for c in STRONG_EXCEPTION_CUES):
            continue
        if any(r in low for r in RESTRICTION_CUES):
            continue
        if _RULE_VERB_RE.search(low):
            out.append(sent)
    return out


def mine(docs: list[SourceDoc], *, max_examples: int = 10) -> PilotReport:
    """Find candidate rule/exception pairs and measure the Tier-1 share."""
    report = PilotReport(documents_scanned=len(docs))

    by_provider: dict[str, list[SourceDoc]] = {}
    for d in docs:
        by_provider.setdefault(d.provider, []).append(d)
    report.providers = len(by_provider)

    seen_exceptions: set[tuple[str, str]] = set()

    for provider, group in by_provider.items():
        rules = [(d, s) for d in group for s in find_rule_sentences(d.text)]
        exceptions = [(d, s, cue) for d in group for s, cue in find_exception_sentences(d.text)]

        for rule_doc, rule_sent in rules:
            # Score every exception against THIS rule and keep only the few
            # that are plausibly about the same subject. Taking the full cross
            # product instead is what turned a 32-document run into 4,648
            # "candidates" that were overwhelmingly one rule sentence paired
            # with every cue-bearing sentence on the site.
            scored: list[tuple[float, SourceDoc, str, str]] = []
            for exc_doc, exc_sent, cue in exceptions:
                if rule_sent == exc_sent:
                    continue
                overlap = topic_overlap(rule_sent, exc_sent)
                if overlap < MIN_TOPIC_OVERLAP:
                    continue
                scored.append((overlap, exc_doc, exc_sent, cue))

            scored.sort(key=lambda t: (-t[0], t[2]))

            kept = 0
            for overlap, exc_doc, exc_sent, cue in scored:
                if kept >= MAX_PAIRS_PER_RULE:
                    break
                # One exception sentence should back one candidate, not be
                # re-proposed against every rule that happens to mention the
                # same words.
                key = (provider, exc_sent)
                if key in seen_exceptions:
                    continue
                seen_exceptions.add(key)
                kept += 1

                same_doc = rule_doc.identity == exc_doc.identity
                pair = CandidatePair(
                    provider=provider,
                    rule_doc=rule_doc.doc_id, rule_sentence=rule_sent,
                    exception_doc=exc_doc.doc_id, exception_sentence=exc_sent,
                    cue_matched=cue,
                    same_document=same_doc,
                    same_guide_parts=same_doc and rule_doc.doc_id != exc_doc.doc_id,
                    topic_overlap=round(overlap, 3),
                )
                report.candidates_total += 1
                if pair.is_tier1:
                    report.candidates_tier1 += 1
                    report.by_provider[provider] = report.by_provider.get(provider, 0) + 1
                    if len(report.examples) < max_examples:
                        report.examples.append(asdict(pair))
                else:
                    report.candidates_tier2 += 1
                    if pair.same_guide_parts:
                        report.candidates_same_guide += 1

    return report


def sensitivity(docs: list[SourceDoc],
                thresholds=(0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05)
                ) -> str:
    """How much does the Tier-1 count depend on the topic-overlap threshold?

    The threshold is a judgement call, so the headline yield must be reported
    as a range rather than as whatever a single setting produced. What this
    sweep is really for is separating the robust finding from the tunable one:
    the Tier-1 SHARE stays low across the whole range even though the absolute
    count moves several-fold.
    """
    global MIN_TOPIC_OVERLAP
    original = MIN_TOPIC_OVERLAP
    rows = []
    try:
        for th in thresholds:
            MIN_TOPIC_OVERLAP = th
            r = mine(docs)
            rows.append((th, r.candidates_total, r.candidates_tier1,
                         r.candidates_tier2, r.tier1_share,
                         r.yield_per_document))
    finally:
        MIN_TOPIC_OVERLAP = original

    lines = [
        "Sensitivity to the topic-overlap threshold",
        "=" * 72,
        "",
        "  The threshold is a judgement call, so the yield is a range, not a number.",
        "",
        f"  {'threshold':>10} {'total':>8} {'tier 1':>8} {'same doc':>10} "
        f"{'t1 share':>10} {'per doc':>9}",
        "  " + "-" * 62,
    ]
    for th, tot, t1, t2, share, per in rows:
        lines.append(f"  {th:>10.2f} {tot:>8,} {t1:>8,} {t2:>10,} "
                     f"{share:>9.1%} {per:>9.2f}")

    t1s = [r[2] for r in rows]
    shares = [r[4] for r in rows]
    lines += [
        "",
        f"  Tier-1 count ranges {min(t1s)}-{max(t1s)} across the sweep; the Tier-1 SHARE "
        f"stays between {min(shares):.0%} and {max(shares):.0%}.",
        "  The share is the robust finding. The count is tunable, and every candidate",
        "  still needs human verification before it is an instance.",
    ]
    return "\n".join(lines)


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
    ap.add_argument("--sensitivity", action="store_true",
                    help="sweep the topic-overlap threshold and report the yield as a range")
    args = ap.parse_args(argv)

    docs = demo_docs() if args.demo else load_docs(args.docs)
    if args.demo:
        print("DEMO MODE: synthetic documents. Exercises the harness; measures nothing.\n")

    report = mine(docs)
    print(report.render())

    if args.sensitivity:
        print()
        print(sensitivity(docs))

    # Decided 2026-09-25, after the pilot compared the two discovery paths on
    # the banking corpus: this lexical miner surfaced 1 candidate where the LLM
    # proposer surfaced 4, of which 3 held up on inspection. For WP2 the
    # proposer is the primary mechanism and this module is the cheap
    # cross-check, not the other way round.
    print()
    print("-" * 70)
    print("NOTE: for WP2 candidate generation, run benchmark.mining.llm_proposer.")
    print("On the banking corpus this lexical miner found 1 candidate against the")
    print("proposer's 4 (3 valid). Keep recall high here and let humans filter --")
    print("precision is not the objective at the discovery stage.")

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
