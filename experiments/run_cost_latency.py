"""Cost and latency per query, broken down by stage.

Proposal §6.1 requires cost and latency reported *with and without*
Contrastive Scope Probing and amortised across a realistic query mix. This
measures the part Member A owns -- retrieval, detection, scope analysis -- and
leaves the probe's own cost to Member B's instrumentation.

Three things it reports that a single average would hide:

**Local versus API time.** Stage 1 runs a cross-encoder on the GPU; stage 2
calls a hosted model. Those have different scaling properties and different
constraints -- one is bounded by VRAM, the other by a per-minute token cap --
so an average over both describes neither.

**The distribution, not the mean.** Latency is right-skewed: most queries
resolve entirely in stage 1 and a few escalate. A mean sits between the two
modes and describes no actual query, so the median and the 95th percentile are
reported alongside it.

**Cache-excluded cost.** A cached call costs nothing and takes no time, so a
run over already-cached instances reports a cost of zero. True and useless for
planning. Cold-path figures are reported separately.

Usage::

    python -m experiments.run_cost_latency --data gold.jsonl --limit 30
    python -m experiments.run_cost_latency --data gold.jsonl --live
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import reproducibility
from api_budget.costlog import CallRecord, CostLog
from contract.models import QueryRecord
from detection.variants.compare import load_records


#: A stage has to be slow enough for the skew to mean something before the
#: report calls its distribution bimodal. Below this, the p95/median ratio is
#: measuring the clock, not the pipeline.
BIMODAL_FLOOR_S = 0.010


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass
class StageTiming:
    """Wall-clock for one stage across a run."""

    name: str
    durations: list[float] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.durations)

    @property
    def mean(self) -> float:
        return self.total / len(self.durations) if self.durations else 0.0

    @property
    def median(self) -> float:
        return _percentile(self.durations, 0.5)

    @property
    def p95(self) -> float:
        return _percentile(self.durations, 0.95)


@dataclass
class CostLatencyReport:
    n_queries: int = 0
    stages: dict[str, StageTiming] = field(default_factory=dict)

    api_calls: int = 0
    api_cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    api_latency_s: float = 0.0

    def timing(self, name: str) -> StageTiming:
        return self.stages.setdefault(name, StageTiming(name))

    @property
    def total_wall(self) -> float:
        return sum(t.total for t in self.stages.values())

    @property
    def per_query(self) -> float:
        return self.total_wall / self.n_queries if self.n_queries else 0.0

    @property
    def billable_calls(self) -> int:
        return self.api_calls - self.api_cached

    @property
    def calls_per_query(self) -> float:
        return self.billable_calls / self.n_queries if self.n_queries else 0.0

    @property
    def tokens_per_query(self) -> float:
        n = self.n_queries
        return (self.prompt_tokens + self.completion_tokens) / n if n else 0.0

    def render(self) -> str:
        lines = [
            "Cost and latency (Member A's stages)",
            "=" * 76,
            f"  queries            {self.n_queries:>10,}",
            f"  wall clock         {self.total_wall:>10.2f} s",
            f"  per query          {self.per_query:>10.3f} s",
            "",
            f"  {'stage':<24} {'total s':>10} {'mean s':>9} {'median s':>10} {'p95 s':>9}",
            "  " + "-" * 66,
        ]
        for t in sorted(self.stages.values(), key=lambda x: -x.total):
            lines.append(f"  {t.name:<24} {t.total:>10.2f} {t.mean:>9.3f} "
                         f"{t.median:>10.3f} {t.p95:>9.3f}")

        lines += [
            "",
            "API",
            "-" * 76,
            f"  calls              {self.api_calls:>10,}  "
            f"({self.api_cached:,} from cache)",
            f"  billable           {self.billable_calls:>10,}",
            f"  per query          {self.calls_per_query:>10.2f}",
            f"  tokens             {self.prompt_tokens + self.completion_tokens:>10,}",
            f"  tokens per query   {self.tokens_per_query:>10.1f}",
            f"  cost               {self.cost_usd:>10.4f} USD",
        ]

        if self.api_cached and not self.billable_calls:
            lines += [
                "",
                "  EVERY API CALL WAS CACHED, so the cost above is zero and the latency",
                "  is not representative. Clear the cache, or read these as warm-path",
                "  numbers only. Cold-path cost is what a corpus run will actually pay.",
            ]

        # Latency is right-skewed: most queries resolve in stage 1 and a few
        # escalate. Say so when the two modes are far apart, because the mean
        # then describes no actual query.
        #
        # The absolute floor matters as much as the ratio. Offline, both modes
        # are sub-millisecond and the ratio between two samples of timer noise
        # runs to 50x or more -- which reported a bimodal distribution where
        # there was nothing to be bimodal about.
        for t in self.stages.values():
            if (t.durations and t.p95 >= BIMODAL_FLOOR_S
                    and t.p95 > 3 * max(t.median, 1e-9)):
                lines += [
                    "",
                    f"  {t.name}: p95 is {t.p95 / max(t.median, 1e-9):.1f}x the median.",
                    "  The distribution is bimodal -- most queries resolve locally, a few",
                    "  escalate. Report the median and p95, not the mean.",
                ]
                break

        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "wall_clock_s": round(self.total_wall, 3),
            "per_query_s": round(self.per_query, 4),
            "stages": {n: {"total_s": round(t.total, 3), "mean_s": round(t.mean, 4),
                           "median_s": round(t.median, 4), "p95_s": round(t.p95, 4)}
                       for n, t in self.stages.items()},
            "api": {"calls": self.api_calls, "cached": self.api_cached,
                    "billable": self.billable_calls,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "cost_usd": round(self.cost_usd, 6),
                    "calls_per_query": round(self.calls_per_query, 3)},
        }


def measure(
    records: list[QueryRecord],
    *,
    live: bool = False,
    use_stage2: bool = True,
    profile: str | None = None,
) -> CostLatencyReport:
    """Run the pipeline stage by stage, timing each.

    The cost log is read before and after so only calls made by THIS run are
    counted; reading the whole log would attribute every earlier experiment's
    spend to these queries.
    """
    from detection.pipeline import TwoStageDetector
    from detection.stage1 import Stage1Filter
    from detection.variants.lexical_nli import LexicalNLIDetector
    from scope.pipeline import ScopePipeline

    log = CostLog()
    before = len(list(log.read()))

    report = CostLatencyReport(n_queries=len(records))

    stage1 = Stage1Filter(profile=profile, heuristic_nli=not live)
    detector = TwoStageDetector(stage1=stage1, use_stage2=use_stage2 and live,
                                classifier=LexicalNLIDetector(stage1=stage1))
    scope = ScopePipeline(use_llm_extraction=live, heuristic_nli=not live)

    for rec in records:
        stripped = rec.model_copy(update={"conflict_pairs": []})

        t0 = time.perf_counter()
        detected, _ = detector.detect(stripped)
        report.timing("A2+A3 detection").durations.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        scope.analyse_record(detected)
        report.timing("A4 scope analysis").durations.append(time.perf_counter() - t0)

    new_calls: list[CallRecord] = list(log.read())[before:]
    for call in new_calls:
        report.api_calls += 1
        report.api_cached += int(call.cached)
        report.prompt_tokens += call.prompt_tokens
        report.completion_tokens += call.completion_tokens
        report.cost_usd += call.cost_usd
        report.api_latency_s += call.latency_s

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--live", action="store_true",
                    help="use real models and API calls; offline timings measure "
                         "the harness, not the system")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-stage2", action="store_true",
                    help="also run with stage-2 escalation disabled, to price the "
                         "escalation itself")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    reproducibility.start_run(args.seed)

    records = load_records(args.data)
    if args.limit:
        records = records[:args.limit]
    if not records:
        print(f"no records in {args.data}", file=sys.stderr)
        return 1

    print(f"{len(records)} queries, live={args.live}\n")
    if not args.live:
        print("OFFLINE: these timings measure the harness and the local encoder,")
        print("not a system anybody would deploy. Use --live for reportable numbers.\n")

    full = measure(records, live=args.live, use_stage2=True, profile=args.profile)
    print(full.render())

    payload = {"with_stage2": full.as_dict()}

    if args.compare_stage2:
        print()
        print("=" * 76)
        print()
        without = measure(records, live=args.live, use_stage2=False,
                          profile=args.profile)
        print(without.render())
        payload["without_stage2"] = without.as_dict()

        delta_t = full.per_query - without.per_query
        delta_c = full.cost_usd - without.cost_usd
        print()
        print("Price of stage-2 escalation")
        print("-" * 76)
        print(f"  extra latency per query   {delta_t:>10.3f} s")
        print(f"  extra cost per query      {delta_c / max(1, full.n_queries):>10.6f} USD")
        print(f"  extra API calls           {full.billable_calls - without.billable_calls:>10,}")
        print()
        print("  Whether that is worth paying is the question detection.threshold")
        print("  answers: it sweeps the escalation band and reports the accuracy")
        print("  bought per unit of escalation.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nresults -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
