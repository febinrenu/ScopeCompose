"""A2: tuning the stage-1 / stage-2 escalation band.

Stage 1 resolves a pair locally; pairs it is unsure about escalate to an API
call. "Unsure" is the interval ``[low, high]`` around the decision boundary,
and until this module existed those two numbers were hardcoded guesses. That
mattered more than it looks: the escalation rate IS the API cost of detection,
and on a 200K-tokens-per-day cap it is also the wall-clock cost. Quoting "70%
resolved locally" from two invented constants is not a result.

What this does is sweep the band over a labelled dev split and report the
trade-off curve: every band gives an (escalation rate, accuracy) pair, and the
useful ones are the Pareto front. You then pick a point on it deliberately --
either "cheapest band that reaches accuracy X" or "best accuracy within an
escalation budget of Y".

Stage 2 is modelled with an explicit ``stage2_accuracy`` rather than assumed
perfect. At 1.0 the curve shows the best case escalation could possibly buy;
at a measured value it shows what it actually buys. Reporting the oracle
number as if it were achievable would overstate the two-stage design.

Usage::

    python -m detection.threshold --data dev.jsonl
    python -m detection.threshold --data dev.jsonl --max-escalation 0.25
    python -m detection.threshold --data dev.jsonl --min-accuracy 0.95 --write
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from contract.models import QueryRecord
from detection.stage1 import Stage1Filter


@dataclass(frozen=True)
class BandResult:
    """One (low, high) band evaluated on the dev split."""

    low: float
    high: float
    n_pairs: int
    n_escalated: int
    stage1_correct: int
    """Correct decisions among pairs stage 1 resolved on its own."""
    stage1_resolved: int

    stage2_accuracy: float

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / self.n_pairs if self.n_pairs else 0.0

    @property
    def stage1_accuracy(self) -> float:
        """Accuracy on the pairs stage 1 kept. This is the number that matters:
        a wider band flatters it by handing the hard cases to stage 2."""
        return self.stage1_correct / self.stage1_resolved if self.stage1_resolved else 1.0

    @property
    def overall_accuracy(self) -> float:
        if not self.n_pairs:
            return 0.0
        expected_stage2 = self.n_escalated * self.stage2_accuracy
        return (self.stage1_correct + expected_stage2) / self.n_pairs

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "escalation_rate": round(self.escalation_rate, 4),
            "stage1_accuracy": round(self.stage1_accuracy, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
        })
        return d


def _labelled_pairs(
    records: list[QueryRecord], stage1: Stage1Filter
) -> list[tuple[float, bool]]:
    """Score every gold-labelled pair once: (conflict probability, gold label).

    Scored once and reused across the whole sweep. Re-running the cross-encoder
    per candidate band would make a 40-point sweep 40x more expensive for
    identical scores.
    """
    out: list[tuple[float, bool]] = []
    for rec in records:
        gold = {p.key: p.is_conflict for p in rec.conflict_pairs}
        if not gold:
            continue
        for cand in stage1.score_pairs(rec.passages):
            if cand.key not in gold:
                continue
            # Recover P(conflict) from the candidate's verdict+confidence.
            p = cand.confidence if cand.is_conflict else 1.0 - cand.confidence
            out.append((p, gold[cand.key]))
    return out


def evaluate_band(
    scored: list[tuple[float, bool]],
    low: float,
    high: float,
    *,
    stage2_accuracy: float = 1.0,
) -> BandResult:
    n_esc = n_res = correct = 0
    for p, truth in scored:
        if low <= p <= high:
            n_esc += 1
        else:
            n_res += 1
            if (p >= 0.5) == truth:
                correct += 1
    return BandResult(
        low=low, high=high, n_pairs=len(scored), n_escalated=n_esc,
        stage1_correct=correct, stage1_resolved=n_res, stage2_accuracy=stage2_accuracy,
    )


def sweep(
    records: list[QueryRecord],
    stage1: Stage1Filter,
    *,
    steps: int = 11,
    stage2_accuracy: float = 1.0,
) -> list[BandResult]:
    """Evaluate a grid of bands centred on the 0.5 decision boundary."""
    scored = _labelled_pairs(records, stage1)
    if not scored:
        return []

    results: list[BandResult] = []
    for i in range(steps):
        # Widen symmetrically from a zero-width band (no escalation) out to
        # [0, 1] (escalate everything). Symmetric because the band exists to
        # capture uncertainty, which is symmetric about the boundary.
        half = 0.5 * i / (steps - 1) if steps > 1 else 0.0
        results.append(
            evaluate_band(scored, round(0.5 - half, 4), round(0.5 + half, 4),
                          stage2_accuracy=stage2_accuracy)
        )
    return results


def pareto_front(results: list[BandResult]) -> list[BandResult]:
    """Bands not dominated by a cheaper band of equal-or-better accuracy."""
    front: list[BandResult] = []
    for r in sorted(results, key=lambda x: x.escalation_rate):
        if not front or r.overall_accuracy > front[-1].overall_accuracy + 1e-12:
            front.append(r)
    return front


def recommend(
    results: list[BandResult],
    *,
    max_escalation: float | None = None,
    min_accuracy: float | None = None,
) -> tuple[BandResult | None, str]:
    """Pick a band under an explicit constraint.

    Deliberately requires one. Without a stated budget or accuracy floor the
    "best" band is always "escalate everything", which is exactly the outcome
    the two-stage design exists to avoid.
    """
    if not results:
        return None, "no labelled pairs to tune on"

    if min_accuracy is not None:
        ok = [r for r in results if r.overall_accuracy >= min_accuracy]
        if not ok:
            best = max(results, key=lambda r: r.overall_accuracy)
            return best, (
                f"no band reaches {min_accuracy:.1%}; the best available is "
                f"{best.overall_accuracy:.1%} at {best.escalation_rate:.1%} escalation. "
                f"Either lower the target or improve stage 1 -- widening the band "
                f"further only buys stage-2 calls."
            )
        pick = min(ok, key=lambda r: r.escalation_rate)
        return pick, (
            f"cheapest band reaching {min_accuracy:.1%} accuracy: "
            f"{pick.escalation_rate:.1%} of pairs escalate"
        )

    if max_escalation is not None:
        ok = [r for r in results if r.escalation_rate <= max_escalation]
        if not ok:
            pick = min(results, key=lambda r: r.escalation_rate)
            return pick, (
                f"no band escalates at or under {max_escalation:.1%}; the narrowest "
                f"available is {pick.escalation_rate:.1%}"
            )
        pick = max(ok, key=lambda r: r.overall_accuracy)
        return pick, (
            f"best accuracy within a {max_escalation:.1%} escalation budget: "
            f"{pick.overall_accuracy:.1%}"
        )

    # No constraint given: report the knee of the curve -- the band after which
    # more escalation stops buying much accuracy.
    front = pareto_front(results)
    if len(front) < 2:
        return front[0] if front else None, "only one non-dominated band"

    best_gain, knee = -1.0, front[0]
    for prev, cur in zip(front, front[1:]):
        d_esc = cur.escalation_rate - prev.escalation_rate
        d_acc = cur.overall_accuracy - prev.overall_accuracy
        gain = d_acc / d_esc if d_esc > 1e-9 else 0.0
        if gain > best_gain:
            best_gain, knee = gain, cur
    return knee, (
        "no constraint given, so this is the knee of the curve. Re-run with "
        "--max-escalation or --min-accuracy to choose deliberately."
    )


def render(results: list[BandResult], pick: BandResult | None, why: str) -> str:
    lines = [
        "Stage-1 / stage-2 escalation band (tau_c)",
        "=" * 78,
        "",
        "  The escalation rate IS the API cost of detection, and on a capped free",
        "  tier it is also the wall-clock cost. Pick a point deliberately.",
        "",
        f"  stage-2 accuracy assumed: {results[0].stage2_accuracy:.2f}"
        + ("  (ORACLE - best case only)" if results[0].stage2_accuracy >= 1.0 else ""),
        "",
        f"  {'band':<18} {'escalated':>10} {'stage-1 acc':>12} {'overall acc':>12}",
        "  " + "-" * 56,
    ]
    front_keys = {(r.low, r.high) for r in pareto_front(results)}
    for r in results:
        mark = " *" if (r.low, r.high) in front_keys else "  "
        chosen = "  <- chosen" if pick and (r.low, r.high) == (pick.low, pick.high) else ""
        lines.append(
            f"{mark}[{r.low:.2f}, {r.high:.2f}]{'':<5} {r.escalation_rate:>9.1%} "
            f"{r.stage1_accuracy:>12.3f} {r.overall_accuracy:>12.3f}{chosen}"
        )

    lines += ["", "  * = on the Pareto front (no cheaper band does as well)", ""]

    # A saturated curve cannot tune anything, and the output looks like a
    # spectacular result rather than a useless one. Say so.
    accs = {round(r.overall_accuracy, 4) for r in results}
    zero_band = next((r for r in results if r.escalation_rate == 0.0), None)
    if len(accs) == 1 or (zero_band and zero_band.stage1_accuracy >= 0.99):
        lines += [
            "  SATURATED CURVE -- this tuning run is not usable.",
            "",
            "  Stage 1 is already at or near perfect accuracy with no escalation, so",
            "  every band scores the same and there is nothing to trade off. That is",
            "  what templated mock data looks like, not what real annotated pairs look",
            "  like: the fixture is regular enough for the head to memorise.",
            "",
            "  Retune on the WP2 corpus before quoting any escalation rate. Until then",
            "  the stage-1/stage-2 split has no measured cost basis.",
            "",
        ]
    if pick:
        lines += [
            "Recommendation",
            "-" * 78,
            f"  low_threshold  = {pick.low}",
            f"  high_threshold = {pick.high}",
            f"  {why}",
            "",
            f"  At this band {pick.escalation_rate:.1%} of pairs reach the API. On a "
            f"300-instance",
            f"  corpus (10 pairs each) that is about "
            f"{int(300 * 10 * pick.escalation_rate):,} stage-2 calls.",
        ]
    else:
        lines += ["Recommendation", "-" * 78, f"  {why}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from detection.variants.compare import load_records

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="labelled JSONL (gold instances or records) to tune on")
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--stage2-accuracy", type=float, default=1.0,
                    help="measured stage-2 accuracy; 1.0 is an oracle upper bound")
    ap.add_argument("--heuristic-nli", action="store_true",
                    help="use the lexical NLI stand-in (no model download)")
    ap.add_argument("--max-escalation", type=float, default=None,
                    help="escalation budget, e.g. 0.25")
    ap.add_argument("--min-accuracy", type=float, default=None,
                    help="accuracy floor, e.g. 0.95")
    ap.add_argument("--json", type=Path, default=None, help="write the raw curve here")
    args = ap.parse_args(argv)

    if args.max_escalation is not None and args.min_accuracy is not None:
        ap.error("give --max-escalation or --min-accuracy, not both")

    records = load_records(args.data)
    if not records:
        print(f"no records in {args.data}", file=sys.stderr)
        return 1

    stage1 = Stage1Filter(heuristic_nli=args.heuristic_nli)
    results = sweep(records, stage1, steps=args.steps,
                    stage2_accuracy=args.stage2_accuracy)
    if not results:
        print("no gold-labelled pairs found -- nothing to tune on", file=sys.stderr)
        return 1

    pick, why = recommend(results, max_escalation=args.max_escalation,
                          min_accuracy=args.min_accuracy)
    print(render(results, pick, why))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"curve": [r.as_dict() for r in results],
                        "chosen": pick.as_dict() if pick else None,
                        "rationale": why}, indent=2),
            encoding="utf-8",
        )
        print(f"\ncurve -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
