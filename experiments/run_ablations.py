"""Member A's side of the named ablations.

Three of the ablations the proposal commits to can be run without Member B's
resolver, because they are questions about *routing* -- about which branches
survive to reach a resolver at all. A branch whose supporting passage is
discarded before generation is unrecoverable, so counting those is a lower
bound on suppression that needs no LLM judge and no generation step.

**1. Route conditional -> factual.** The headline ablation. Relabel every
conditional pair as factual, which is what a four-class detector without the
new type would emit, and re-route. Every exception branch then reaches the
selection path instead of composition, and selection keeps one passage. The
delta between the two branch-preservation rates is the argument for the
conditional class existing at all. If it is near zero, the new type is not
earning its place.

**2. Tier 1 vs Tier 2.** Headline results are reported per tier and never
blended, so the split has to be computable rather than remembered.

**3. Explicit vs implicit.** Reported separately and never averaged, because
implicit recovery is what the extraction method is judged on. Member A can
compute the branch-level split; the extraction-accuracy half is Member B's.

Usage::

    python -m experiments.run_ablations --data benchmark/data/wp1_probe.jsonl
    python -m experiments.run_ablations --data gold.jsonl --json out/ablations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from contract.gold import Explicitness, GoldInstance
from contract.models import ConflictType, Construction
from contract.routing import Action, route
from experiments.run_baselines import load_gold


# --------------------------------------------------------------------------- #
# 1. Route conditional -> factual
# --------------------------------------------------------------------------- #


@dataclass
class RoutingOutcome:
    """How many gold branches survive routing under one labelling."""

    label: str
    instances: int = 0
    gold_branches: int = 0
    exception_branches: int = 0

    composed: int = 0
    merged: int = 0
    selected: int = 0
    prior_work: int = 0
    passed_through: int = 0

    exceptions_preserved: int = 0
    """An exception branch survives only where its pair was COMPOSED.
    Selection keeps one passage, so the other branch is gone."""

    @property
    def exception_preservation_rate(self) -> float:
        n = self.exception_branches
        return self.exceptions_preserved / n if n else 0.0

    @property
    def exception_loss_rate(self) -> float:
        return 1.0 - self.exception_preservation_rate


def _route_instances(
    instances: list[GoldInstance], *, relabel_conditional_as_factual: bool
) -> RoutingOutcome:
    label = "conditional -> factual (ablated)" if relabel_conditional_as_factual \
        else "full five-class (baseline)"
    out = RoutingOutcome(label=label)

    for inst in instances:
        record = inst.to_query_record(include_gold_pairs=True)
        if not record.conflict_pairs:
            continue

        out.instances += 1
        n_exc = len(inst.exception_branches)
        out.gold_branches += len(inst.gold_branches)
        out.exception_branches += n_exc

        for pair in record.conflict_pairs:
            if relabel_conditional_as_factual and pair.type is ConflictType.CONDITIONAL:
                # What a four-class detector emits: the conditional class does
                # not exist, so a general-rule/exception pair reads as a plain
                # contradiction. Drop the scope relation too -- the contract
                # forbids one on a factual pair, and a system without the
                # class would not have computed it.
                pair = pair.model_copy(
                    update={"type": ConflictType.FACTUAL, "scope_relation": None}
                )

            action = route(pair).action
            if action is Action.COMPOSE:
                out.composed += 1
                out.exceptions_preserved += n_exc
            elif action is Action.MERGE:
                out.merged += 1
            elif action is Action.SELECT:
                out.selected += 1
            elif action is Action.PRIOR_WORK:
                out.prior_work += 1
            else:
                out.passed_through += 1

    return out


@dataclass
class ConditionalTypeAblation:
    baseline: RoutingOutcome
    ablated: RoutingOutcome

    @property
    def preservation_drop(self) -> float:
        return (self.baseline.exception_preservation_rate
                - self.ablated.exception_preservation_rate)

    def render(self) -> str:
        b, a = self.baseline, self.ablated
        lines = [
            "Ablation 1: route conditional -> factual",
            "=" * 78,
            "",
            "  Relabelling every conditional pair as factual is what a four-class",
            "  detector emits. Those pairs then route to the prior-work selection",
            "  path, which keeps one passage -- so the exception branch is gone",
            "  before any resolver sees it.",
            "",
            f"  {'':<34} {'baseline':>12} {'ablated':>12}",
            "  " + "-" * 60,
            f"  {'exception branches':<34} {b.exception_branches:>12,} "
            f"{a.exception_branches:>12,}",
            f"  {'preserved (composed)':<34} {b.exceptions_preserved:>12,} "
            f"{a.exceptions_preserved:>12,}",
            f"  {'preservation rate':<34} {b.exception_preservation_rate:>11.1%} "
            f"{a.exception_preservation_rate:>11.1%}",
            "",
            f"  {'pairs composed':<34} {b.composed:>12,} {a.composed:>12,}",
            f"  {'pairs merged':<34} {b.merged:>12,} {a.merged:>12,}",
            f"  {'pairs to selection':<34} {b.selected:>12,} {a.selected:>12,}",
            f"  {'pairs to prior-work strategy':<34} {b.prior_work:>12,} {a.prior_work:>12,}",
            "",
            f"  PRESERVATION DROP: {self.preservation_drop:.1%}",
        ]
        if self.preservation_drop >= 0.5:
            lines += [
                "",
                "  Removing the conditional class discards most exception branches at the",
                "  routing step, before extraction or generation runs. This is the",
                "  argument for the class existing, and it holds without needing an LLM",
                "  judge: the branches are unreachable, not merely mis-scored.",
            ]
        elif self.preservation_drop > 0.05:
            lines += [
                "",
                f"  A {self.preservation_drop:.1%} drop is real but modest. Report the number,",
                "  do not round it up into a stronger claim than it supports.",
            ]
        else:
            lines += [
                "",
                "  NEAR ZERO. On this data the conditional class is not changing what",
                "  survives routing, so it is not earning its place. Check the corpus",
                "  actually contains refinement instances before concluding anything:",
                "  a set with no compose-eligible pairs cannot show a drop.",
            ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2 & 3. Breakdowns
# --------------------------------------------------------------------------- #


@dataclass
class Breakdown:
    title: str
    note: str
    rows: dict[str, dict[str, float]] = field(default_factory=dict)

    def render(self) -> str:
        if not self.rows:
            return f"{self.title}\n" + "-" * 78 + "\n  (no instances)"
        cols = sorted({k for r in self.rows.values() for k in r})
        width = max(len(g) for g in self.rows) + 2
        lines = [self.title, "=" * 78, "", f"  {self.note}", "",
                 "  " + " " * width + "".join(f"{c:>22}" for c in cols)]
        for group, vals in sorted(self.rows.items()):
            body = "".join(
                f"{vals[c]:>22.1%}" if isinstance(vals.get(c), float) else
                f"{vals.get(c, 0):>22,}"
                for c in cols
            )
            lines.append(f"  {group:<{width}}{body}")
        return "\n".join(lines)


def tier_breakdown(instances: list[GoldInstance]) -> Breakdown:
    b = Breakdown(
        title="Ablation 2: Tier 1 (natural) vs Tier 2 (split)",
        note=("Never blended in headline reporting. Tier 2 is legitimate and isolates "
              "the resolution question from the mining question, but it is NOT "
              "evidence of natural multi-source occurrence."),
    )
    for tier in Construction:
        subset = [i for i in instances if i.construction is tier]
        if not subset:
            continue
        outcome = _route_instances(subset, relabel_conditional_as_factual=False)
        b.rows[tier.value] = {
            "instances": len(subset),
            "exception branches": outcome.exception_branches,
            "preservation": outcome.exception_preservation_rate,
        }
    return b


def explicitness_breakdown(instances: list[GoldInstance]) -> Breakdown:
    b = Breakdown(
        title="Ablation 3: explicit vs implicit conditions",
        note=("Reported separately and never averaged -- implicit recovery is what the "
              "extraction method is judged on. Branch counts are Member A's; the "
              "extraction-accuracy half is Member B's."),
    )
    counts: dict[str, dict[str, float]] = {}
    for inst in instances:
        for branch in inst.exception_branches:
            key = branch.explicitness.value
            row = counts.setdefault(key, {"exception branches": 0, "instances": 0})
            row["exception branches"] += 1
    for inst in instances:
        kinds = {br.explicitness.value for br in inst.exception_branches}
        for k in kinds:
            counts.setdefault(k, {"exception branches": 0, "instances": 0})
            counts[k]["instances"] += 1
    b.rows = counts
    return b


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="JSONL of GOLD instances (branch structure is required)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    instances = load_gold(args.data)
    if not instances:
        print(f"no gold instances in {args.data}. These ablations need branch "
              f"structure to know what a dropped passage costs.", file=sys.stderr)
        return 1

    print(f"{len(instances)} gold instances\n")

    ablation = ConditionalTypeAblation(
        baseline=_route_instances(instances, relabel_conditional_as_factual=False),
        ablated=_route_instances(instances, relabel_conditional_as_factual=True),
    )
    print(ablation.render())
    print()
    print(tier_breakdown(instances).render())
    print()
    print(explicitness_breakdown(instances).render())

    print()
    print("-" * 78)
    print("  These measure ROUTING only: whether a branch's supporting passage")
    print("  survives to reach a resolver. That is a lower bound on suppression and")
    print("  needs no LLM judge. The full PR/SR/HCR/SCR numbers are Member B's and")
    print("  require the metric suite in metrics/preservation.py.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "conditional_type_ablation": {
                "baseline": asdict(ablation.baseline),
                "ablated": asdict(ablation.ablated),
                "preservation_drop": round(ablation.preservation_drop, 4),
            },
            "tier_breakdown": tier_breakdown(instances).rows,
            "explicitness_breakdown": explicitness_breakdown(instances).rows,
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nresults -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
