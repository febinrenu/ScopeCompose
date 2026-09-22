"""Run and score Member A's retrieval-side baselines.

Three systems, scored on the one thing that separates them: **do they still
have the exception passage when generation starts?**

That is deliberately not answer correctness. A baseline that drops the
exception can still produce a fluent, correct-looking answer from the general
rule alone -- which is the entire failure this project exists to measure.
Counting dropped passages against the gold branch structure makes the
suppression visible at the resolution step, before generation gets a chance to
paper over it.

The headline output is the **branch-loss rate**: the fraction of gold exception
branches whose only supporting passage the baseline discarded. Those branches
are unrecoverable -- no generator, however good, can preserve a branch it was
never shown.

Usage::

    python -m experiments.run_baselines --data benchmark.jsonl
    python -m experiments.run_baselines --data benchmark.jsonl --show-examples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from baselines.retrieval_side import NLIFilter, RerankTop1, StandardRAG
from contract.gold import GoldInstance
from contract.models import ConflictType


@dataclass
class BaselineScore:
    name: str
    n_instances: int = 0
    n_gold_branches: int = 0
    n_branches_lost: int = 0
    n_passages_dropped: int = 0

    # conditional instances only -- where the failure is supposed to show
    n_conditional: int = 0
    n_conditional_branches: int = 0
    n_conditional_lost: int = 0

    examples: list[dict] = field(default_factory=list)

    @property
    def branch_loss_rate(self) -> float:
        return self.n_branches_lost / self.n_gold_branches if self.n_gold_branches else 0.0

    @property
    def conditional_branch_loss_rate(self) -> float:
        n = self.n_conditional_branches
        return self.n_conditional_lost / n if n else 0.0

    @property
    def passages_dropped_per_instance(self) -> float:
        return self.n_passages_dropped / self.n_instances if self.n_instances else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "branch_loss_rate": round(self.branch_loss_rate, 4),
            "conditional_branch_loss_rate": round(self.conditional_branch_loss_rate, 4),
            "passages_dropped_per_instance": round(self.passages_dropped_per_instance, 3),
            "n_instances": self.n_instances,
            "n_gold_branches": self.n_gold_branches,
            "n_branches_lost": self.n_branches_lost,
            "n_conditional": self.n_conditional,
        }


def score_instance(output, instance: GoldInstance, score: BaselineScore) -> list[str]:
    """Update ``score`` with one instance. Returns the ids of lost branches."""
    kept = {p.id for p in output.kept}
    dropped = set(output.suppressed_ids)

    score.n_instances += 1
    score.n_passages_dropped += len(dropped)

    is_conditional = instance.gold_conflict_type is ConflictType.CONDITIONAL
    if is_conditional:
        score.n_conditional += 1

    lost: list[str] = []
    for branch in instance.gold_branches:
        score.n_gold_branches += 1
        if is_conditional:
            score.n_conditional_branches += 1

        # A branch survives only if the passage grounding it survived. The
        # default branch with no supporting passage is treated as surviving:
        # it is the claim every baseline reports anyway.
        support = branch.supporting_passage
        if support is not None and support not in kept:
            score.n_branches_lost += 1
            lost.append(branch.branch_id)
            if is_conditional:
                score.n_conditional_lost += 1
    return lost


def run(
    instances: list[GoldInstance],
    *,
    heuristic_nli: bool = False,
    show_examples: int = 0,
) -> list[BaselineScore]:
    systems = [
        StandardRAG(),
        RerankTop1(),
        NLIFilter(heuristic_nli=heuristic_nli),
    ]
    scores = [BaselineScore(name=s.name) for s in systems]

    for inst in instances:
        record = inst.to_query_record(include_gold_pairs=False)
        for system, score in zip(systems, scores):
            output = system.run(record)
            lost = score_instance(output, inst, score)

            if lost and len(score.examples) < show_examples:
                score.examples.append({
                    "instance_id": inst.instance_id,
                    "query": inst.query,
                    "dropped": output.suppressed_ids,
                    "lost_branches": lost,
                    "rationale": output.rationale,
                    "gold_scoped_answer": inst.gold_scoped_answer,
                    "selection_answer": inst.selection_answer,
                })
    return scores


def render(scores: list[BaselineScore], *, show_examples: int = 0) -> str:
    lines = [
        "Member A's retrieval-side baselines",
        "=" * 88,
        "",
        "  Scored on whether the exception passage SURVIVES to generation, not on",
        "  answer correctness. A system that drops the exception can still produce a",
        "  fluent, correct-looking answer from the general rule alone -- which is the",
        "  failure being measured. A branch whose only supporting passage was dropped",
        "  is unrecoverable: no generator can preserve what it was never shown.",
        "",
        f"  {'baseline':<18} {'branch loss':>12} {'on conditional':>15} "
        f"{'passages dropped':>18}",
        "  " + "-" * 70,
    ]
    for s in scores:
        lines.append(
            f"  {s.name:<18} {s.branch_loss_rate:>11.1%} "
            f"{s.conditional_branch_loss_rate:>14.1%} "
            f"{s.passages_dropped_per_instance:>17.2f}"
        )

    worst = max(scores, key=lambda s: s.conditional_branch_loss_rate, default=None)
    if worst and worst.conditional_branch_loss_rate > 0:
        lines += [
            "",
            f"  {worst.name} loses {worst.conditional_branch_loss_rate:.1%} of exception",
            "  branches on conditional instances -- before generation runs at all.",
        ]
        if worst.name == "nli_filter":
            lines += [
                "",
                "  This is the demonstrative case. The NLI-filter is a reasonable-looking",
                "  design somebody might actually ship: drop whichever passage looks",
                "  inconsistent with the rest. On a general-rule/valid-exception pair the",
                "  exception IS the inconsistent-looking passage, so the filter deletes",
                "  exactly the branch that applied to the user, then answers confidently",
                "  from the rule alone.",
            ]

    if show_examples:
        for s in scores:
            if not s.examples:
                continue
            lines += ["", f"{s.name} -- worked examples", "-" * 88]
            for ex in s.examples:
                lines += [
                    f"  query:      {ex['query']}",
                    f"  dropped:    {', '.join(ex['dropped']) or '(none)'}",
                    f"  lost:       {', '.join(ex['lost_branches'])}",
                    f"  because:    {ex['rationale'][:150]}",
                ]
                if ex["gold_scoped_answer"]:
                    lines.append(f"  should say: {ex['gold_scoped_answer'][:150]}")
                if ex["selection_answer"]:
                    lines.append(f"  will say:   {ex['selection_answer'][:150]}")
                lines.append("")
    return "\n".join(lines)


def load_gold(path: Path) -> list[GoldInstance]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "gold_conflict_type" not in obj:
                continue  # wire records carry no branch structure to score against
            out.append(GoldInstance.model_validate(obj))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="JSONL of GOLD instances (branch structure is required)")
    ap.add_argument("--heuristic-nli", action="store_true",
                    help="use the lexical NLI stand-in (no model download)")
    ap.add_argument("--show-examples", type=int, default=0,
                    help="print N worked examples per baseline")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    instances = load_gold(args.data)
    if not instances:
        print(f"no gold instances in {args.data}. This scorer needs branch structure, "
              f"so run contract.mock without --format record, or pass the annotated "
              f"corpus.", file=sys.stderr)
        return 1

    print(f"{len(instances)} gold instances\n")
    scores = run(instances, heuristic_nli=args.heuristic_nli,
                 show_examples=args.show_examples)
    print(render(scores, show_examples=args.show_examples))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([s.as_dict() for s in scores], indent=2), encoding="utf-8"
        )
        print(f"\nresults -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
