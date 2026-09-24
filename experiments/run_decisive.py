"""The decisive experiment: pipeline routing versus a structured long-context parse.

Proposal §6.3 and §6.5. One long-context model is given every passage at once
and asked for the same branch structure the pipeline produces. Both are scored
on the identical metric, over identical instances, and compared with a paired
test.

**The claim this settles.** The argument for a resolution *operator* is that a
model handed all the passages still cannot be relied on to preserve every
scoped branch. §6.5 commits in advance: if the baseline matches the pipeline's
Preservation Rate, the central architectural claim does not hold as stated, and
that outcome is reported as prominently as a positive one.

**What is actually being compared, stated precisely.** The pipeline's
preservation here is measured at the *routing* step: a branch survives when its
pair is routed to composition, because selection keeps one passage and the
other branch is then unreachable. The baseline's preservation is measured by
aligning the branches it emitted against gold. These are not the same
mechanism, and the comparison is only fair because both answer the same
question -- *did this gold branch survive?* -- against the same gold. Member B's
full PR/SR/HCR/SCR suite over generated answers is the eventual instrument;
this is the part that can be measured without it.

**Read README §5 before quoting any number from here.** On the zero-spend
configuration the baseline runs on open weights, so a pipeline win is weak
evidence (a reviewer attributes it to model tier) while a baseline win is
decisive falsification. The asymmetry is printed with the result.

Usage::

    python -m experiments.run_decisive --data benchmark/data/wp1_probe.jsonl --limit 20
    python -m experiments.run_decisive --data gold.jsonl --include-freetext
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import reproducibility
from baselines.structured_long_context import BaselineOutput
from baselines.structured_long_context import run as run_baseline
from contract.gold import GoldInstance
from contract.models import ConflictType
from contract.routing import Action, route
from experiments.run_baselines import load_gold
from metrics.branch_match import BranchJudgement, align, is_grounded
from metrics.stats import bootstrap_difference, mcnemar, wilson


@dataclass
class SystemScore:
    """Preservation for one system over one corpus."""

    name: str
    gold_branches: int = 0
    preserved: int = 0
    suppressed: int = 0
    distorted: int = 0

    instances: int = 0
    hallucinated: int = 0
    no_conflict_instances: int = 0
    spurious: int = 0

    #: Per-gold-branch survival, keyed (instance_id, branch_id). The two
    #: systems are scored on the same branches, so the comparison is paired.
    branch_preserved: dict[tuple[str, str], bool] = field(default_factory=dict)

    @property
    def preservation_rate(self) -> float:
        return self.preserved / self.gold_branches if self.gold_branches else 0.0

    @property
    def suppression_rate(self) -> float:
        return self.suppressed / self.gold_branches if self.gold_branches else 0.0

    @property
    def distortion_rate(self) -> float:
        return self.distorted / self.gold_branches if self.gold_branches else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / self.instances if self.instances else 0.0

    @property
    def spurious_rate(self) -> float:
        n = self.no_conflict_instances
        return self.spurious / n if n else 0.0

    def pr_ci(self, alpha: float = 0.05):
        return wilson(self.preserved, self.gold_branches, alpha=alpha)

    def sr_ci(self, alpha: float = 0.05):
        return wilson(self.suppressed, self.gold_branches, alpha=alpha)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "PR": round(self.preservation_rate, 4),
            "SR": round(self.suppression_rate, 4),
            "distortion": round(self.distortion_rate, 4),
            "HCR": round(self.hallucination_rate, 4),
            "SCR": round(self.spurious_rate, 4),
            "gold_branches": self.gold_branches,
            "instances": self.instances,
            "PR_ci": self.pr_ci().as_dict(),
        }


def score_pipeline_routing(
    instances: list[GoldInstance], *, oracle: bool = False, live: bool = False
) -> SystemScore:
    """Preservation the pipeline achieves at the routing step.

    A branch survives where its pair routes to composition. Under selection the
    other passage is discarded and the branch is unreachable regardless of how
    good the generator is.

    ``oracle=True`` routes on the GOLD conflict labels: an upper bound on what
    the routing design could achieve with perfect detection. It is not the
    pipeline's score and must never be compared against a baseline that had to
    do its own detection -- that comparison flatters the pipeline by exactly
    the detector's error rate, and it is the single easiest way to manufacture
    a result here.

    ``oracle=False`` (the default) runs the real detector and scope analyser
    and routes on what they actually predicted. That is the comparable number.
    """
    score = SystemScore(
        name="pipeline (oracle routing)" if oracle else "pipeline (real detection)"
    )

    predicted: dict[str, object] = {}
    if not oracle:
        from experiments.run_pipeline import run as run_pipeline

        inputs = [i.to_query_record(include_gold_pairs=False) for i in instances]
        analysed, _, _ = run_pipeline(inputs, live=live)
        predicted = {r.query_id: r for r in analysed}

    for inst in instances:
        score.instances += 1
        if inst.gold_conflict_type is ConflictType.NO_CONFLICT:
            score.no_conflict_instances += 1

        record = (inst.to_query_record(include_gold_pairs=True) if oracle
                  else predicted.get(inst.instance_id))
        if record is None:
            for branch in inst.gold_branches:
                score.gold_branches += 1
                score.suppressed += 1
                score.branch_preserved[(inst.instance_id, branch.branch_id)] = False
            continue

        composed = any(route(p).action is Action.COMPOSE for p in record.conflict_pairs)

        for branch in inst.gold_branches:
            score.gold_branches += 1
            # The default branch survives every route: selection reports it.
            survives = branch.is_default or composed
            score.branch_preserved[(inst.instance_id, branch.branch_id)] = survives
            if survives:
                score.preserved += 1
            else:
                score.suppressed += 1

    return score


def score_baseline(
    instances: list[GoldInstance], outputs: list[BaselineOutput], *, name: str, nli=None
) -> SystemScore:
    """Preservation the baseline achieves, by aligning its branches to gold.

    ``nli`` is strongly recommended. Without it the alignment is lexical, and
    lexical matching scores a verbose paraphrase of the right answer as a
    distortion -- which understates the baseline and inflates the pipeline's
    apparent advantage.
    """
    score = SystemScore(name=name)
    by_id = {o.instance_id: o for o in outputs}

    for inst in instances:
        out = by_id.get(inst.instance_id)
        score.instances += 1
        is_no_conflict = inst.gold_conflict_type is ConflictType.NO_CONFLICT
        if is_no_conflict:
            score.no_conflict_instances += 1

        if out is None or not out.ok:
            # A failed call preserves nothing. Counting it as missing data
            # instead would quietly flatter whichever system failed more.
            for branch in inst.gold_branches:
                score.gold_branches += 1
                score.suppressed += 1
                score.branch_preserved[(inst.instance_id, branch.branch_id)] = False
            continue

        result = align(inst.gold_branches, out.branches, nli=nli)
        for a in result.alignments:
            score.gold_branches += 1
            survived = a.judgement is BranchJudgement.PRESERVED
            score.branch_preserved[(inst.instance_id, a.gold.branch_id)] = survived
            if survived:
                score.preserved += 1
            elif a.judgement is BranchJudgement.SUPPRESSED:
                score.suppressed += 1
            else:
                score.distorted += 1

        known = {p.id for p in inst.passages}
        if any(not is_grounded(b, known) for b in result.unmatched_predictions):
            score.hallucinated += 1
        if is_no_conflict and any(not b.is_default for b in out.branches):
            score.spurious += 1

    return score


def compare(a: SystemScore, b: SystemScore, *, seed: int = 0) -> dict:
    """Paired comparison on the branches both systems were scored on."""
    shared = sorted(set(a.branch_preserved) & set(b.branch_preserved))
    if not shared:
        return {}

    outcomes_a = [a.branch_preserved[k] for k in shared]
    outcomes_b = [b.branch_preserved[k] for k in shared]
    test = mcnemar(outcomes_a, outcomes_b)

    units = list(zip(outcomes_a, outcomes_b))
    diff = bootstrap_difference(
        units,
        lambda s: sum(1 for x, _ in s if x) / len(s),
        lambda s: sum(1 for _, y in s if y) / len(s),
        n_resamples=2000, seed=seed,
    )
    return {"mcnemar": test, "pr_difference": diff, "n_shared": len(shared)}


def render(pipeline: SystemScore, baselines: list[SystemScore], comparison: dict) -> str:
    lines = [
        "THE DECISIVE EXPERIMENT",
        "=" * 78,
        "",
        "  Proposal 6.5 commits in advance: if the structured long-context baseline",
        "  matches the pipeline's Preservation Rate, the central architectural claim",
        "  does not hold as stated.",
        "",
        f"  {'system':<30} {'PR':>22} {'SR':>9} {'HCR':>7}",
        "  " + "-" * 72,
    ]
    for s in [pipeline, *baselines]:
        lines.append(f"  {s.name:<30} {s.pr_ci().render():>22} "
                     f"{s.suppression_rate:>8.1%} {s.hallucination_rate:>7.1%}")

    if comparison:
        test = comparison["mcnemar"]
        diff = comparison["pr_difference"]
        lines += [
            "",
            "Pipeline minus baseline, paired on identical gold branches",
            "-" * 78,
            f"  PR difference   {diff.render()}",
            f"  McNemar         p = {test.p_value:.4g} ({test.method})",
            f"  discordant      {test.n_discordant} of {comparison['n_shared']} branches",
            "",
        ]
        if not diff.excludes(0.0):
            lines += [
                "  THE INTERVAL CONTAINS ZERO.",
                "",
                "  On this evidence the two are not distinguishable. Per 6.5 that is the",
                "  falsifying outcome and it is reported as prominently as a positive one:",
                "  the contribution narrows to the extraction method, the metric suite and",
                "  the benchmark.",
            ]
        elif diff.point > 0:
            lines += [
                "  The pipeline preserves more, and the interval excludes zero.",
                "",
                "  Read README section 5 before quoting this. On open weights a pipeline",
                "  win is WEAK evidence -- a reviewer attributes the gap to model tier",
                "  rather than to architecture. Only a baseline win is decisive here.",
            ]
        else:
            lines += [
                "  THE BASELINE PRESERVES MORE, and the interval excludes zero.",
                "",
                "  This is decisive falsification and it is fully valid: a weaker model",
                "  beating the pipeline cannot be explained away by model tier. Report it.",
            ]

        if test.n_discordant < 10:
            lines += [
                "",
                f"  CAUTION: only {test.n_discordant} branches where the two systems differ.",
                "  The test has almost no power at this size, so a null result here is",
                "  absence of evidence, not evidence of equivalence. Scale the corpus",
                "  before drawing either conclusion.",
            ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap instances, to protect the daily token budget")
    ap.add_argument("--include-freetext", action="store_true",
                    help="also run the weaker free-text variant, which separates the "
                         "pipeline's architectural benefit from the benefit of merely "
                         "requiring structured output")
    ap.add_argument("--pace", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--oracle", action="store_true",
                    help="ALSO report the pipeline with perfect detection. An upper "
                         "bound, not a comparable score -- reported alongside, never "
                         "instead of, the real number")
    ap.add_argument("--lexical-only", action="store_true",
                    help="align branches without the NLI scorer. Faster, and it "
                         "understates every baseline")
    args = ap.parse_args(argv)

    reproducibility.start_run(args.seed)

    instances = load_gold(args.data)
    if args.limit:
        instances = instances[:args.limit]
    instances = [i for i in instances if i.gold_branches]
    if not instances:
        print(f"no gold instances with branch structure in {args.data}", file=sys.stderr)
        return 1

    print(f"{len(instances)} instances with gold branches\n")

    pipeline = score_pipeline_routing(instances)

    print("Running the structured long-context baseline")
    structured = run_baseline(instances, variant="structured", pace=args.pace)
    scored = [score_baseline(instances, structured, name="long-context (structured)")]

    if args.include_freetext:
        print("\nRunning the free-text variant")
        freetext = run_baseline(instances, variant="freetext", pace=args.pace)
        scored.append(score_baseline(instances, freetext,
                                     name="long-context (free text)"))

    comparison = compare(pipeline, scored[0], seed=args.seed)

    print()
    print(render(pipeline, scored, comparison))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": pipeline.as_dict(),
            "baselines": [s.as_dict() for s in scored],
        }
        if comparison:
            payload["comparison"] = {
                "mcnemar": comparison["mcnemar"].as_dict(),
                "pr_difference": comparison["pr_difference"].as_dict(),
                "n_shared": comparison["n_shared"],
            }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nresults -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
