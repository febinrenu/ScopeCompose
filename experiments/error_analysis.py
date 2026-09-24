"""Qualitative error analysis on the confused instances.

Proposal §5.2 commits to this: the factual/conditional boundary is reported as
its own number *and* accompanied by error analysis on a sample of the confused
instances. A confusion count says how often the detector is wrong; it says
nothing about whether the errors share a cause, and that is the part that tells
you what to fix.

Errors are grouped by the feature pattern that drove them, so a systematic
failure shows up as a cluster rather than as a list. Three patterns are worth
separating, and the routing consequence differs for each:

``conditional -> factual``
    A real exception read as a plain contradiction. Routes to selection, the
    exception branch is discarded. This is the project's own failure mode
    occurring inside the project's own pipeline.
``factual -> conditional``
    A real contradiction read as an exception. Routes to composition, which
    presents two incompatible claims as if both held under different
    conditions -- inventing a scope that does not exist.
``relation errors``
    The type was right but the four-way relation was wrong. Reported by cell,
    because refinement-as-redundant loses a branch while
    redundant-as-refinement fabricates one.

Usage::

    python -m experiments.error_analysis --data gold.jsonl
    python -m experiments.error_analysis --data gold.jsonl --sample 10 --md errors.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import reproducibility
from contract.models import ConflictType, QueryRecord, ScopeRelation
from detection.stage1 import PairCandidate, Stage1Filter
from detection.variants.compare import load_records


@dataclass
class ErrorCase:
    """One misclassified pair, with the evidence that produced it."""

    query_id: str
    query: str
    pair: tuple[str, str]
    text_i: str
    text_j: str

    gold_type: str
    pred_type: str
    gold_relation: str | None = None
    pred_relation: str | None = None

    nli_entailment: float = 0.0
    nli_neutral: float = 0.0
    nli_contradiction: float = 0.0
    exception_cues: float = 0.0
    restriction_asymmetry: float = 0.0
    specificity_asymmetry: float = 0.0
    numeric_clash: float = 0.0
    token_jaccard: float = 0.0

    @property
    def kind(self) -> str:
        if self.gold_type != self.pred_type:
            return f"{self.gold_type} -> {self.pred_type}"
        return f"relation {self.gold_relation} -> {self.pred_relation}"

    def signature(self) -> str:
        """Coarse pattern label, so systematic failures cluster.

        Deliberately built from the features the detector actually sees. An
        error grouped by what drove it points at a fix; an error grouped by its
        surface text points at an anecdote.
        """
        bits = []
        bits.append("cue" if self.exception_cues >= 1 else "no-cue")
        bits.append("numeric-clash" if self.numeric_clash else "no-numeric-clash")
        if self.nli_contradiction > max(self.nli_entailment, self.nli_neutral):
            bits.append("nli-contradiction")
        elif self.nli_neutral > max(self.nli_entailment, self.nli_contradiction):
            bits.append("nli-neutral")
        else:
            bits.append("nli-entailment")
        bits.append("scope-asymmetry" if self.restriction_asymmetry >= 1
                    else "no-scope-asymmetry")
        return " / ".join(bits)


@dataclass
class ErrorReport:
    n_pairs: int = 0
    n_errors: int = 0
    cases: list[ErrorCase] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.n_pairs if self.n_pairs else 0.0

    def by_kind(self) -> Counter:
        return Counter(c.kind for c in self.cases)

    def by_signature(self, kind: str | None = None) -> Counter:
        return Counter(c.signature() for c in self.cases
                       if kind is None or c.kind == kind)

    def render(self, sample: int = 5) -> str:
        lines = [
            "Error analysis",
            "=" * 82,
            f"  pairs scored   {self.n_pairs:>8,}",
            f"  errors         {self.n_errors:>8,}   ({self.error_rate:.1%})",
        ]
        if not self.cases:
            lines.append("\n  No errors on this data. That is a fixture result, not a"
                         " system result, unless the corpus is real and large.")
            return "\n".join(lines)

        lines += ["", "By error kind", "-" * 82]
        for kind, n in self.by_kind().most_common():
            lines.append(f"  {kind:<46} {n:>5}")

        # A signature carrying most of one kind is a systematic failure with a
        # nameable cause, which is the whole reason for this analysis.
        for kind, _ in self.by_kind().most_common(3):
            sigs = self.by_signature(kind)
            total = sum(sigs.values())
            lines += ["", f"{kind}  -- dominant feature patterns", "-" * 82]
            for sig, n in sigs.most_common(4):
                share = n / total if total else 0
                flag = "   <- systematic" if share >= 0.5 and total >= 4 else ""
                lines.append(f"  {sig:<62} {n:>4} ({share:.0%}){flag}")

        lines += ["", f"Worked examples (first {sample} per kind)", "=" * 82]
        shown: Counter = Counter()
        for case in self.cases:
            if shown[case.kind] >= sample:
                continue
            shown[case.kind] += 1
            lines += [
                "",
                f"[{case.query_id} {case.pair[0]}/{case.pair[1]}]  {case.kind}",
                f"  Q:  {case.query[:78]}",
                f"  i:  {case.text_i[:78]}",
                f"  j:  {case.text_j[:78]}",
                f"  nli entail/neutral/contra  "
                f"{case.nli_entailment:.2f} / {case.nli_neutral:.2f} / "
                f"{case.nli_contradiction:.2f}",
                f"  cues={case.exception_cues:.0f}  "
                f"scope-asym={case.restriction_asymmetry:.0f}  "
                f"spec-asym={case.specificity_asymmetry:.0f}  "
                f"num-clash={case.numeric_clash:.0f}  "
                f"jaccard={case.token_jaccard:.2f}",
                f"  pattern: {case.signature()}",
            ]
        return "\n".join(lines)

    def to_markdown(self, sample: int = 10) -> str:
        """A sheet a human can annotate while reading the errors."""
        lines = [
            "# Error analysis",
            "",
            f"{self.n_errors} errors over {self.n_pairs} pairs ({self.error_rate:.1%}).",
            "",
            "For each case: is the gold label right? If yes, what would the detector",
            "have needed to see? Answering the second question across a cluster is",
            "what turns a confusion count into a fix.",
            "",
        ]
        shown: Counter = Counter()
        for case in self.cases:
            if shown[case.kind] >= sample:
                continue
            shown[case.kind] += 1
            lines += [
                f"## {case.query_id} — {case.kind}",
                "",
                f"**Question:** {case.query}",
                "",
                f"- **i:** {case.text_i}",
                f"- **j:** {case.text_j}",
                "",
                f"Gold `{case.gold_type}`"
                + (f"/`{case.gold_relation}`" if case.gold_relation else "")
                + f" — predicted `{case.pred_type}`"
                + (f"/`{case.pred_relation}`" if case.pred_relation else ""),
                "",
                f"Pattern: `{case.signature()}`",
                "",
                "- [ ] gold label is correct",
                "- [ ] detector had enough signal to get this right",
                "- Cause: ",
                "",
            ]
        return "\n".join(lines)


def analyse(
    predicted: list[QueryRecord],
    gold: list[QueryRecord],
    *,
    stage1: Stage1Filter | None = None,
) -> ErrorReport:
    stage1 = stage1 or Stage1Filter(heuristic_nli=True, auto_load_head=False)
    gold_by_id = {r.query_id: r for r in gold}
    report = ErrorReport()

    for pred in predicted:
        g = gold_by_id.get(pred.query_id)
        if g is None:
            continue
        gold_pairs = {p.key: p for p in g.conflict_pairs}
        by_key: dict[tuple[str, str], PairCandidate] = {
            c.key: c for c in stage1.score_pairs(g.passages)
        }

        for pair in pred.conflict_pairs:
            truth = gold_pairs.get(pair.key)
            if truth is None:
                continue
            report.n_pairs += 1

            type_wrong = pair.type is not truth.type
            relation_wrong = (
                truth.scope_relation is not None
                and pair.scope_relation is not truth.scope_relation
            )
            if not (type_wrong or relation_wrong):
                continue

            report.n_errors += 1
            cand = by_key.get(pair.key)
            f = cand.features if cand else None
            report.cases.append(ErrorCase(
                query_id=pred.query_id,
                query=g.query,
                pair=pair.key,
                text_i=cand.text_i if cand else "",
                text_j=cand.text_j if cand else "",
                gold_type=truth.type.value,
                pred_type=pair.type.value,
                gold_relation=truth.scope_relation.value if truth.scope_relation else None,
                pred_relation=pair.scope_relation.value if pair.scope_relation else None,
                nli_entailment=cand.nli.entailment if cand else 0.0,
                nli_neutral=cand.nli.neutral if cand else 0.0,
                nli_contradiction=cand.nli.contradiction if cand else 0.0,
                exception_cues=f.exception_cues_max if f else 0.0,
                restriction_asymmetry=f.restriction_asymmetry if f else 0.0,
                specificity_asymmetry=f.specificity_asymmetry if f else 0.0,
                numeric_clash=f.numeric_clash if f else 0.0,
                token_jaccard=f.token_jaccard if f else 0.0,
            ))
    return report


def main(argv: list[str] | None = None) -> int:
    from experiments.run_pipeline import run as run_pipeline

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=5,
                    help="worked examples printed per error kind")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", type=Path, default=None,
                    help="write an annotatable markdown sheet here")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    reproducibility.start_run(args.seed)

    gold = load_records(args.data)
    if args.limit:
        gold = gold[:args.limit]
    if not gold:
        print(f"no records in {args.data}", file=sys.stderr)
        return 1

    print(f"{len(gold)} records\n")
    analysed, _, _ = run_pipeline(gold, live=args.live)
    report = analyse(analysed, gold)
    print(report.render(sample=args.sample))

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\nannotation sheet -> {args.md}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "n_pairs": report.n_pairs,
            "n_errors": report.n_errors,
            "by_kind": dict(report.by_kind()),
            "by_signature": dict(report.by_signature()),
            "cases": [asdict(c) for c in report.cases],
        }, indent=2), encoding="utf-8")
        print(f"raw -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
