"""Evaluate A1 retrieval, and check the claim that it is held constant.

The paper says retrieval is kept identical to prior conflict-aware systems on
purpose, so that any difference in results isolates the analysis and
resolution stages. That is a claim about retrieval quality being adequate and
unbiased -- and it has to be measured, not asserted.

Two numbers matter, and the second one matters more:

**recall@k** -- of the passages that ground a gold branch, how many were
retrieved at all?

**exception recall@k** -- of the passages that ground an *exception* branch
specifically, how many were retrieved?

The second is the one that can quietly sink the project. If retrieval
systematically misses exception passages -- plausible, since an exception is
shorter, rarer, and shares fewer query terms than the general rule -- then the
suppression being measured downstream happened at *retrieval*, not at the
resolution operator. That would be the failure mode Chen et al. describe as
originating in retrieval rather than in what the pipeline does afterwards, and
it would undercut the whole framing.

Sparse, dense and hybrid are compared so the fusion choice is evidenced rather
than assumed.

Usage::

    python -m experiments.run_retrieval_eval --data benchmark.jsonl
    python -m experiments.run_retrieval_eval --data benchmark.jsonl --k 5 --sparse-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from contract.gold import GoldInstance
from contract.models import ConflictType
from retrieval.corpus import Corpus
from retrieval.pipeline import HybridRetriever


@dataclass
class RetrievalScore:
    name: str
    n_queries: int = 0

    n_relevant: int = 0
    n_retrieved_relevant: int = 0

    n_exception: int = 0
    n_retrieved_exception: int = 0

    reciprocal_ranks: list[float] = field(default_factory=list)
    misses: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.n_retrieved_relevant / self.n_relevant if self.n_relevant else 0.0

    @property
    def exception_recall(self) -> float:
        return self.n_retrieved_exception / self.n_exception if self.n_exception else 0.0

    @property
    def mrr(self) -> float:
        return sum(self.reciprocal_ranks) / len(self.reciprocal_ranks) \
            if self.reciprocal_ranks else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "recall": round(self.recall, 4),
            "exception_recall": round(self.exception_recall, 4),
            "mrr": round(self.mrr, 4),
            "n_queries": self.n_queries,
            "n_relevant": self.n_relevant,
            "n_exception": self.n_exception,
        }


def build_corpus(instances: list[GoldInstance]) -> tuple[Corpus, dict[str, str]]:
    """One corpus over every instance's passages.

    Ids are namespaced by instance, because ``p0`` means different things in
    different records. The returned map goes namespaced id -> instance id, so
    a retrieved passage can be traced back.

    Pooling every instance's passages into one corpus is what makes this a real
    retrieval test: the retriever has to find the right two passages among
    hundreds of plausible distractors from other instances, not among the four
    it was handed.
    """
    passages, owner = [], {}
    for inst in instances:
        for p in inst.passages:
            pid = f"{inst.instance_id}::{p.id}"
            passages.append(p.model_copy(update={"id": pid}))
            owner[pid] = inst.instance_id
    return Corpus(passages), owner


def evaluate(
    instances: list[GoldInstance],
    *,
    name: str,
    use_dense: bool,
    k: int,
    max_misses: int = 5,
) -> RetrievalScore:
    corpus, _ = build_corpus(instances)
    retriever = HybridRetriever(corpus, use_dense=use_dense)
    score = RetrievalScore(name=name)

    for inst in instances:
        # Passages that ground a gold branch are the relevant set.
        relevant = {
            b.supporting_passage for b in inst.gold_branches if b.supporting_passage
        }
        exception_support = {
            b.supporting_passage for b in inst.exception_branches if b.supporting_passage
        }
        if not relevant:
            continue

        score.n_queries += 1
        results = retriever.search(inst.query, k=k)
        got = {r.passage_id for r in results}
        ranks = {r.passage_id: r.rank for r in results}

        first_hit = None
        for local_id in relevant:
            full = f"{inst.instance_id}::{local_id}"
            score.n_relevant += 1
            if full in got:
                score.n_retrieved_relevant += 1
                r = ranks[full]
                first_hit = r if first_hit is None else min(first_hit, r)

        for local_id in exception_support:
            full = f"{inst.instance_id}::{local_id}"
            score.n_exception += 1
            if full in got:
                score.n_retrieved_exception += 1
            elif len(score.misses) < max_misses:
                score.misses.append({
                    "instance_id": inst.instance_id,
                    "query": inst.query,
                    "missed_passage": local_id,
                    "conflict_type": inst.gold_conflict_type.value,
                    "text": inst.passage_text(local_id) if hasattr(inst, "passage_text")
                    else next((p.text for p in inst.passages if p.id == local_id), ""),
                })

        score.reciprocal_ranks.append(1.0 / first_hit if first_hit else 0.0)

    return score


def duplicate_passage_rate(instances: list[GoldInstance]) -> float:
    """Fraction of passages whose text also appears on another passage.

    Templated fixtures reuse the same sentence across many instances. When
    that happens, pooled retrieval is being asked an unanswerable question --
    which copy of an identical sentence did this query mean? -- and recall
    collapses for reasons that have nothing to do with the retriever.
    """
    seen: dict[str, int] = {}
    for inst in instances:
        for p in inst.passages:
            key = " ".join(p.text.lower().split())
            seen[key] = seen.get(key, 0) + 1
    total = sum(seen.values())
    duplicated = sum(n for n in seen.values() if n > 1)
    return duplicated / total if total else 0.0


def render(scores: list[RetrievalScore], k: int, duplicate_rate: float = 0.0) -> str:
    lines = [
        f"A1 retrieval evaluation (k = {k})",
        "=" * 82,
        "",
        "  Every instance's passages are pooled into one corpus, so the retriever has",
        "  to find the right ones among hundreds of plausible distractors.",
        "",
        f"  {'configuration':<18} {'recall':>9} {'exception recall':>18} {'MRR':>8}",
        "  " + "-" * 58,
    ]
    for s in scores:
        lines.append(
            f"  {s.name:<18} {s.recall:>8.1%} {s.exception_recall:>17.1%} {s.mrr:>8.3f}"
        )

    lines += ["", "  Exception recall is the number that can sink the project.", ""]
    worst = min(scores, key=lambda s: s.exception_recall, default=None)
    best = max(scores, key=lambda s: s.exception_recall, default=None)

    if duplicate_rate > 0.15:
        lines += [
            f"  READ THIS FIRST: {duplicate_rate:.0%} of passages in this corpus are "
            f"near-duplicates",
            "  of another passage. No retriever can tell which instance's copy of an",
            "  identical sentence a query meant, so recall here is a floor set by the",
            "  fixture, not a measurement of the retriever.",
            "",
            "  That is what templated mock data does. Re-run on the WP2 corpus before",
            "  drawing any conclusion about retrieval quality.",
            "",
        ]

    if best and best.exception_recall < 0.9:
        lines += [
            f"  WARNING: the best configuration still misses "
            f"{1 - best.exception_recall:.0%} of exception passages.",
            "",
            "  If retrieval never surfaces the exception, the suppression measured",
            "  downstream happened HERE, not at the resolution operator -- which is the",
            "  failure Chen et al. attribute to retrieval rather than to the pipeline.",
            "  That would undercut the framing, so it has to be fixed or reported, not",
            "  left implicit.",
        ]
    elif best:
        lines += [
            f"  Exception recall is {best.exception_recall:.1%} at k={k}: retrieval is",
            "  surfacing the exception passages, so suppression measured downstream is",
            "  attributable to the resolution step rather than to retrieval.",
        ]

    if best and worst and best.name != worst.name:
        gap = best.exception_recall - worst.exception_recall
        if gap > 0.02:
            lines += [
                "",
                f"  {best.name} beats {worst.name} by {gap:.1%} on exception recall -- "
                f"evidence for the fusion choice rather than an assumption.",
            ]

    for s in scores:
        if s.misses:
            lines += ["", f"{s.name} -- missed exception passages", "-" * 82]
            for m in s.misses:
                lines += [
                    f"  query:  {m['query']}",
                    f"  missed: [{m['missed_passage']}] {m['text'][:110]}",
                    "",
                ]
    return "\n".join(lines)


def load_gold(path: Path) -> list[GoldInstance]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "gold_conflict_type" in obj:
                out.append(GoldInstance.model_validate(obj))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="JSONL of GOLD instances (branch structure is required)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sparse-only", action="store_true",
                    help="skip the dense and hybrid runs (no model download)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    instances = load_gold(args.data)
    if not instances:
        print(f"no gold instances in {args.data}. This scorer needs branch structure "
              f"to know which passages are relevant.", file=sys.stderr)
        return 1

    print(f"{len(instances)} instances, "
          f"{sum(len(i.passages) for i in instances)} passages in the pooled corpus\n")

    configs = [("bm25_only", False)]
    if not args.sparse_only:
        configs.append(("hybrid_rrf", True))

    scores = [
        evaluate(instances, name=name, use_dense=dense, k=args.k)
        for name, dense in configs
    ]
    print(render(scores, args.k, duplicate_passage_rate(instances)))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([s.as_dict() for s in scores], indent=2),
                             encoding="utf-8")
        print(f"\nresults -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
