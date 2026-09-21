"""Will this run fit inside the daily rate limits?

The proposal says to estimate call volume *before* committing to run anything
at scale. On a paid account that advice protects a budget. On this one it
protects the timetable: with a 200K-tokens-per-day cap, a stage that needs 20M
tokens is not expensive, it is a hundred days long, and finding that out in
week 16 is fatal in a way that finding it out in week 1 is not.

This script multiplies out the pipeline's call volume against the real limits
in ``config/models.yaml`` and reports, per stage, how many days it needs.

Usage::

    python scripts/budget_estimate.py                     # 300-instance corpus
    python scripts/budget_estimate.py --instances 50      # the WP1 pilot
    python scripts/budget_estimate.py --configs 4         # fewer ablations
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

import settings  # noqa: E402


@dataclass
class Stage:
    """One pipeline stage's call volume."""

    name: str
    tier: str
    calls_per_instance: float
    tokens_per_call: int
    scales_with_configs: bool = False
    owner: str = "A"
    note: str = ""

    def calls(self, instances: int, configs: int) -> int:
        n = self.calls_per_instance * instances
        if self.scales_with_configs:
            n *= configs
        return int(round(n))

    def tokens(self, instances: int, configs: int) -> int:
        return self.calls(instances, configs) * self.tokens_per_call


# Volume model. Derived from the pipeline as built: K=5 gives C(5,2)=10 pairs
# per query, and A4 extracts one descriptor per passage, not per pair.
STAGES = [
    Stage("A2 stage-2 detection", "BULK", 10 * 0.3, 1_000, owner="A",
          note="only the ~30% of pairs stage 1 is unsure about escalate"),
    Stage("A4 descriptor extraction", "BULK", 5.0, 800, owner="A",
          note="one per passage, cached and reused across that record's pairs"),
    Stage("B1 counterfactual questions", "BULK", 4.0, 700, owner="B"),
    Stage("B1 candidate conditions", "BULK", 16.0, 900, owner="B",
          note="4 questions x 4 other passages - the volume driver"),
    Stage("B3 scoped-answer generation", "JUDGE", 1.0, 1_500, owner="B"),
    Stage("Structured long-context baseline", "LONG_CONTEXT", 1.0, 6_000, owner="joint",
          note="the decisive experiment; one pass"),
    Stage("Second-backbone robustness", "SECOND_BACKBONE", 2.5, 2_300, owner="joint"),
]

# The metric judge is computed separately: it is the largest line item by a
# wide margin, and the two things that shrink it are structural choices worth
# making explicit rather than burying in a constant.
#
#   per-branch calls  -> one call per gold branch. The passages and the rubric
#                        are re-sent every time, so a 2.5-branch instance pays
#                        for the shared context 2.5 times.
#   batched calls     -> one call per instance, scoring every branch against a
#                        context sent once. Same judgements, ~half the tokens.
#
#   full ablations    -> every configuration scored on the whole corpus.
#   subsampled        -> headline configurations on the whole corpus, the
#                        remaining ablations on a stratified subsample. This is
#                        ordinary practice and is reported as what it is.
JUDGE_BRANCHES_PER_INSTANCE = 2.5
JUDGE_TOKENS_PER_BRANCH_CALL = 2_300
JUDGE_TOKENS_PER_BATCHED_CALL = 3_000


def judge_volume(
    *, instances: int, configs: int, batched: bool,
    headline_configs: int, ablation_instances: int | None,
) -> tuple[int, int]:
    """(calls, tokens) for the metric judge under a given strategy."""
    if ablation_instances is None:
        instance_configs = instances * configs
    else:
        headline = min(headline_configs, configs)
        rest = max(0, configs - headline)
        instance_configs = instances * headline + min(ablation_instances, instances) * rest

    if batched:
        calls = instance_configs
        tokens = calls * JUDGE_TOKENS_PER_BATCHED_CALL
    else:
        calls = int(round(instance_configs * JUDGE_BRANCHES_PER_INSTANCE))
        tokens = calls * JUDGE_TOKENS_PER_BRANCH_CALL
    return calls, tokens


def load_limits() -> dict:
    raw = yaml.safe_load(settings.MODELS_YAML.read_text(encoding="utf-8"))
    return raw.get("limits") or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=int, default=300,
                    help="corpus size (default 300, the proposal's target)")
    ap.add_argument("--configs", type=int, default=12,
                    help="number of system configurations the judge scores (default 12)")
    ap.add_argument("--cache-hit-rate", type=float, default=0.0,
                    help="assumed cache hit rate on a re-run (default 0.0, a cold first run)")
    ap.add_argument("--judge-batched", action="store_true",
                    help="score all of an instance's branches in ONE judge call, sending the "
                         "shared context once instead of per branch")
    ap.add_argument("--ablation-instances", type=int, default=None,
                    help="run non-headline ablations on a stratified subsample of this size "
                         "instead of the full corpus")
    ap.add_argument("--headline-configs", type=int, default=4,
                    help="configurations scored on the FULL corpus (default 4); "
                         "only meaningful with --ablation-instances")
    args = ap.parse_args(argv)

    settings.reload()
    cfg = settings.models()
    limits = load_limits()

    print(f"Corpus: {args.instances} instances | {args.configs} configurations "
          f"| cache hit rate {args.cache_hit_rate:.0%}\n")

    keep = 1 - args.cache_hit_rate

    # Per-tier totals.
    per_tier: dict[str, dict[str, int]] = {}
    rows = []
    for st in STAGES:
        calls = int(st.calls(args.instances, args.configs) * keep)
        toks = int(st.tokens(args.instances, args.configs) * keep)
        rows.append((st, calls, toks))
        acc = per_tier.setdefault(st.tier, {"calls": 0, "tokens": 0})
        acc["calls"] += calls
        acc["tokens"] += toks

    # The metric judge, under the chosen strategy.
    j_calls, j_tokens = judge_volume(
        instances=args.instances, configs=args.configs, batched=args.judge_batched,
        headline_configs=args.headline_configs, ablation_instances=args.ablation_instances,
    )
    j_calls, j_tokens = int(j_calls * keep), int(j_tokens * keep)
    strategy = "batched per instance" if args.judge_batched else "one call per branch"
    if args.ablation_instances:
        strategy += f", ablations on {args.ablation_instances}"
    judge_stage = Stage("Metric judge (PR/SR/HCR/SCR)", "JUDGE", 0, 0,
                        owner="B", note=strategy)
    rows.append((judge_stage, j_calls, j_tokens))
    acc = per_tier.setdefault("JUDGE", {"calls": 0, "tokens": 0})
    acc["calls"] += j_calls
    acc["tokens"] += j_tokens

    print("Per stage")
    print("-" * 94)
    print(f"  {'stage':<34} {'own':<6} {'tier':<16} {'calls':>9} {'tokens':>12}")
    for st, calls, toks in sorted(rows, key=lambda r: -r[2]):
        print(f"  {st.name:<34} {st.owner:<6} {st.tier:<16} {calls:>9,} {toks:>12,}")

    total_calls = sum(c for _, c, _ in rows)
    total_tokens = sum(t for _, _, t in rows)
    print(f"  {'TOTAL':<34} {'':<6} {'':<16} {total_calls:>9,} {total_tokens:>12,}")

    print("\nPer tier, against the real daily caps")
    print("-" * 94)
    print(f"  {'tier':<17} {'model':<26} {'tokens':>11} {'tok/day':>9} "
          f"{'req/day':>8} {'days':>7}")

    worst_days = 0.0
    bottleneck = None
    for tier_name, acc in per_tier.items():
        try:
            model = cfg.tier(tier_name).model
        except KeyError:
            model = None
        lim = limits.get(model or "", {})
        tpd = lim.get("tpd")
        rpd = lim.get("rpd")

        days_tokens = (acc["tokens"] / tpd) if tpd else 0.0
        days_calls = (acc["calls"] / rpd) if rpd else 0.0
        days = max(days_tokens, days_calls)

        if days > worst_days:
            worst_days, bottleneck = days, (tier_name, model,
                                            "tokens" if days_tokens >= days_calls else "requests")

        print(f"  {tier_name:<17} {str(model):<26} {acc['tokens']:>11,} "
              f"{(tpd or 0):>9,} {(rpd or 0):>8,} {days:>7.1f}")

    # Roll up by MODEL, which is what the limits actually apply to.
    #
    # Two tiers on the same model share one daily cap and therefore SERIALISE
    # (their days add). Two tiers on different models have independent caps and
    # therefore run CONCURRENTLY (take the max). Reporting per-tier days without
    # this distinction over-reports the split case and under-reports the shared
    # one, so the wall-clock estimate is computed here rather than eyeballed.
    by_model: dict[str, dict] = {}
    for tier_name, acc in per_tier.items():
        try:
            m = cfg.tier(tier_name).model
        except KeyError:
            continue
        if not m:
            continue
        e = by_model.setdefault(m, {"tiers": [], "calls": 0, "tokens": 0})
        e["tiers"].append(tier_name)
        e["calls"] += acc["calls"]
        e["tokens"] += acc["tokens"]

    print("\nPer model - this is what the limits actually apply to")
    print("-" * 94)
    print(f"  {'model':<26} {'tiers':<34} {'tokens':>11} {'days':>7}")

    wall_clock = 0.0
    slowest = None
    for model, e in sorted(by_model.items(), key=lambda kv: -kv[1]["tokens"]):
        lim = limits.get(model, {})
        tpd, rpd = lim.get("tpd"), lim.get("rpd")
        days = max((e["tokens"] / tpd) if tpd else 0.0,
                   (e["calls"] / rpd) if rpd else 0.0)
        if days > wall_clock:
            wall_clock, slowest = days, model
        print(f"  {model:<26} {', '.join(e['tiers']):<34} {e['tokens']:>11,} {days:>7.1f}")

    print("\nVerdict")
    print("-" * 94)
    print(f"  Wall clock for a cold full run: about {wall_clock:.0f} day(s), set by {slowest}.")
    print("  Different models have independent caps, so those lanes run concurrently;")
    print("  tiers sharing a model serialise.")

    shared = {m: e["tiers"] for m, e in by_model.items() if len(e["tiers"]) > 1}
    for model, ts in shared.items():
        print(f"    - {' + '.join(ts)} share {model}: their days ADD.")

    worst_days = wall_clock
    print()
    if worst_days <= 1:
        print("  Fits comfortably in a day. Go.")
    elif worst_days <= 5:
        print("  Multi-day but manageable. Run stages on separate days and let the")
        print("  cache make re-runs free.")
    else:
        print("  This does NOT fit a sensible schedule (WP4 is about 6 weeks). Levers,")
        print("  in the order worth pulling:")
        if not args.judge_batched:
            print("    1. --judge-batched: score all of an instance's branches in one call")
            print("       instead of re-sending the passages and rubric per branch. Same")
            print("       judgements, roughly half the tokens.")
        if args.ablation_instances is None:
            print("    2. --ablation-instances 100: headline configurations on the full")
            print("       corpus, remaining ablations on a stratified subsample. Ordinary")
            print("       practice; report it as what it is.")
        print("    3. Run the WP1 pilot first (--instances 50). Only scale once a")
        print("       component is stable - every cold re-run costs the full quota again.")
        print("    4. Keep BULK, JUDGE and SECOND_BACKBONE on DIFFERENT models. Groq's")
        print("       limits are per-model, so spreading them multiplies the budget.")

    print("\n  The request cache makes the SECOND run of anything free. These numbers")
    print("  are for a cold run; re-estimate with --cache-hit-rate after the first pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
