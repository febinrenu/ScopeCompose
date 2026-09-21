"""Run Member A's pipeline end to end: A1 -> A2 -> A3 -> A4.

Two modes:

``--data FILE``
    Run over an existing corpus of records. Retrieval is skipped -- the
    passages are already chosen -- which is the right setting for every
    experiment that isolates detection and scope analysis from retrieval
    quality.

``--query TEXT --corpus FILE``
    Retrieve from a corpus, then analyse. The full path, for demos and for
    checking that A1 hands off correctly.

The default configuration runs entirely offline (heuristic NLI, rule-based
descriptors, no stage-2 escalation) so the whole pipeline can be exercised at
zero API cost before anything is scaled up. Pass ``--live`` for the real thing.

Usage::

    python -m contract.mock --n 30 --seed 0 --format record -o mock.jsonl
    python -m experiments.run_pipeline --data mock.jsonl
    python -m experiments.run_pipeline --data mock.jsonl --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contract.models import QueryRecord
from contract.routing import Action, route_with_flags
from detection.pipeline import TwoStageDetector
from detection.stage1 import Stage1Filter
from detection.variants.compare import load_records
from detection.variants.lexical_nli import LexicalNLIDetector
from scope.pipeline import ScopePipeline


def build_pipeline(*, live: bool, profile: str | None = None):
    """Assemble A2 + A3 + A4.

    Offline by default. The heuristic NLI stand-in and the rule-based
    descriptor extractor are fixtures, not models -- they exist so the wiring
    can be verified without a download, and nothing they produce is reportable.
    """
    stage1 = Stage1Filter(profile=profile, heuristic_nli=not live)
    detector = TwoStageDetector(
        stage1=stage1,
        use_stage2=live,
        classifier=LexicalNLIDetector(stage1=stage1),
    )
    scope = ScopePipeline(use_llm_extraction=live, heuristic_nli=not live)
    return detector, scope


def run(records: list[QueryRecord], *, live: bool, profile: str | None = None):
    detector, scope = build_pipeline(live=live, profile=profile)

    # A2 + A3: detect and classify. Gold pairs are stripped first -- feeding
    # the detector its own answer key would make every number meaningless.
    stripped = [r.model_copy(update={"conflict_pairs": []}) for r in records]
    detected, det_stats = detector.detect_all(stripped)

    # A4: four-way scope relation.
    analysed, scope_stats = scope.analyse_all(detected)

    return analysed, det_stats, scope_stats


def routing_summary(records: list[QueryRecord]) -> dict[str, int]:
    """What Member B would do with this output.

    The number that matters here is COMPOSE: it counts the instances where the
    new operator actually fires. If it is zero on a corpus full of conditional
    instances, the pipeline is producing labels that never reach the thing the
    project is about.
    """
    counts: dict[str, int] = {}
    for rec in records:
        for pair in rec.conflict_pairs:
            action = route_with_flags(pair, rec.multi_exception_flags).action
            counts[action.value] = counts.get(action.value, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="JSONL of records or gold instances")
    src.add_argument("--query", help="a single query to retrieve and analyse")
    ap.add_argument("--corpus", type=Path, help="passage corpus JSONL, with --query")
    ap.add_argument("--live", action="store_true",
                    help="use real models and API calls instead of the offline fixtures")
    ap.add_argument("--profile", default=None, help="hardware profile override")
    ap.add_argument("--evaluate", action="store_true",
                    help="score against the gold labels in the input file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-o", "--output", type=Path, default=None, help="write records here")
    args = ap.parse_args(argv)

    if args.query:
        if not args.corpus:
            ap.error("--query needs --corpus")
        from retrieval.corpus import Corpus
        from retrieval.pipeline import HybridRetriever

        corpus = Corpus.from_jsonl(args.corpus)
        retriever = HybridRetriever(corpus, profile=args.profile, use_dense=args.live)
        records = [retriever.to_record(args.query, query_id="q0")]
        gold = []
        print(f"retrieved {len(records[0].passages)} passages for: {args.query!r}\n")
    else:
        gold = load_records(args.data)
        if args.limit:
            gold = gold[:args.limit]
        records = gold
        print(f"loaded {len(records)} records from {args.data}\n")

    if not records:
        print("nothing to run", file=sys.stderr)
        return 1

    if not args.live:
        print("OFFLINE MODE: heuristic NLI stand-in, rule-based descriptors, no stage-2.")
        print("These outputs verify the wiring. They are not measurements. Use --live"
              " for real numbers.\n")

    analysed, det_stats, scope_stats = run(records, live=args.live, profile=args.profile)

    print(det_stats.render())
    print()
    print(scope_stats.render())
    print()

    counts = routing_summary(analysed)
    print("Routing (what Member B would do)")
    print("-" * 56)
    for action in Action:
        n = counts.get(action.value, 0)
        marker = "  <- the new operator" if action is Action.COMPOSE and n else ""
        print(f"  {action.value:<14} {n:>6,}{marker}")

    if args.evaluate and gold:
        from metrics.classification import (
            score_classification,
            score_detection,
            score_scope_relations,
            split_by_construction,
        )

        print()
        print(score_detection(analysed, gold).render())
        print()
        print(score_classification(analysed, gold).render())
        print()
        print(score_scope_relations(analysed, gold).render())

        tiers = split_by_construction(gold)
        if len(tiers) > 1:
            print("\nTier breakdown (never blend these into one headline number)")
            print("-" * 56)
            for tier, subset in sorted(tiers.items()):
                ids = {r.query_id for r in subset}
                sub_pred = [r for r in analysed if r.query_id in ids]
                s = score_classification(sub_pred, subset)
                print(f"  {tier:<10} accuracy {s.matrix.accuracy:.4f}  "
                      f"conditional-leak {s.conditional_leak_rate:.1%}  (n={s.matrix.total})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in analysed:
                fh.write(json.dumps(rec.model_dump(mode="json"), sort_keys=True) + "\n")
        print(f"\nwrote {len(analysed)} records -> {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
