"""WP1 go/no-go: does Contrastive Scope Probing beat direct extraction?

The question, from the work-package spec:

> Does Contrastive Scope Probing beat direct zero-shot extraction by a useful
> margin, **at an acceptable Hallucinated-Condition Rate**?

The second half is not a caveat. An accuracy gain bought with fabricated
conditions is not a success, and the two numbers are reported together or not
at all -- :meth:`ProbeComparison.render` will not print recall without HCR
beside it.

**The fallback is agreed in advance**, so a negative result is a finding rather
than a scramble: anchor Member B's method contribution on the grounding gate
and the metric suite, and report implicit extraction as an open problem with a
measured ceiling. A measured ceiling is publishable; a quietly dropped
component is not. :func:`verdict` states which of those two the numbers support.

**Explicit and implicit are never averaged.** Explicit conditions carry a cue
word and reduce to parsing; unmarked ones are what the method exists for.
An average over both is dominated by the easy half, and would let a method that
does nothing for unmarked conditions look like it works.

Usage::

    python -m experiments.wp1_probe.run_probe --data benchmark/data/corpus.jsonl --limit 40
    python -m experiments.wp1_probe.run_probe --data corpus.jsonl --ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import reproducibility
from contract.gold import Branch, GoldInstance
from metrics.branch_match import align, is_grounded
from metrics.stats import mcnemar, wilson


def _load_gold(path: Path, limit: int | None = None) -> list[GoldInstance]:
    out: list[GoldInstance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "gold_conflict_type" not in obj:
            continue
        inst = GoldInstance.model_validate(obj)
        # Only instances with annotated branch structure can score condition
        # recovery. An instance whose branches Member B has not filled in yet
        # has no answer key, and scoring against an empty one would read as
        # perfect recall of nothing.
        if inst.gold_branches:
            out.append(inst)
        if limit and len(out) >= limit:
            break
    return out


@dataclass
class SystemScore:
    """Condition recovery for one extraction method."""

    name: str
    recovered: int = 0
    missed: int = 0
    distorted: int = 0

    explicit_recovered: int = 0
    explicit_total: int = 0
    implicit_recovered: int = 0
    implicit_total: int = 0

    instances: int = 0
    hallucinating_instances: int = 0
    ungrounded_branches: int = 0
    api_calls: int = 0

    #: per gold branch, for the paired test
    hit: dict[tuple[str, str], bool] = field(default_factory=dict)

    @property
    def gold_branches(self) -> int:
        return self.recovered + self.missed + self.distorted

    @property
    def recall(self) -> float:
        return self.recovered / self.gold_branches if self.gold_branches else 0.0

    @property
    def explicit_recall(self) -> float:
        return (self.explicit_recovered / self.explicit_total
                if self.explicit_total else 0.0)

    @property
    def implicit_recall(self) -> float:
        return (self.implicit_recovered / self.implicit_total
                if self.implicit_total else 0.0)

    @property
    def hcr(self) -> float:
        """Hallucinated-Condition Rate: instances asserting an ungrounded branch."""
        return self.hallucinating_instances / self.instances if self.instances else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "recall": round(self.recall, 4),
            "explicit_recall": round(self.explicit_recall, 4),
            "implicit_recall": round(self.implicit_recall, 4),
            "HCR": round(self.hcr, 4),
            "gold_branches": self.gold_branches,
            "explicit_total": self.explicit_total,
            "implicit_total": self.implicit_total,
            "instances": self.instances,
            "ungrounded_branches": self.ungrounded_branches,
            "api_calls": self.api_calls,
        }


def score(
    instances: list[GoldInstance],
    predictions: dict[str, list[Branch]],
    *,
    name: str,
    api_calls: int = 0,
    nli=None,
) -> SystemScore:
    """Score one method's extracted branches against gold."""
    s = SystemScore(name=name, api_calls=api_calls)

    for inst in instances:
        pred = predictions.get(inst.instance_id, [])
        s.instances += 1

        known = {p.id for p in inst.passages}
        ungrounded = sum(1 for b in pred if not b.is_default
                         and not is_grounded(b, known))
        s.ungrounded_branches += ungrounded
        if ungrounded:
            s.hallucinating_instances += 1

        result = align(inst.gold_branches, pred, nli=nli)
        for a in result.alignments:
            key = (inst.instance_id, a.gold.branch_id)
            got = a.judgement.value == "preserved"
            s.hit[key] = got

            if got:
                s.recovered += 1
            elif a.judgement.value == "suppressed":
                s.missed += 1
            else:
                s.distorted += 1

            # Explicitness is a property of the EXCEPTION branch. The default
            # has no condition to mark, so counting it in either bucket would
            # dilute both.
            if a.gold.is_default:
                continue
            if a.gold.explicitness.value == "explicit":
                s.explicit_total += 1
                s.explicit_recovered += int(got)
            else:
                s.implicit_total += 1
                s.implicit_recovered += int(got)
    return s


@dataclass
class ProbeComparison:
    probe: SystemScore
    baseline: SystemScore

    @property
    def implicit_gain(self) -> float:
        return self.probe.implicit_recall - self.baseline.implicit_recall

    @property
    def hcr_cost(self) -> float:
        return self.probe.hcr - self.baseline.hcr

    def paired_test(self):
        keys = sorted(set(self.probe.hit) & set(self.baseline.hit))
        if not keys:
            return None
        return mcnemar([self.probe.hit[k] for k in keys],
                       [self.baseline.hit[k] for k in keys])

    def verdict(self) -> tuple[str, str]:
        """``(headline, what it means)`` -- the go/no-go, stated plainly."""
        gain, cost = self.implicit_gain, self.hcr_cost

        if self.probe.implicit_total == 0:
            return ("NO UNMARKED CONDITIONS IN THIS DATA",
                    "Every gold exception carries a cue word, so the probe has "
                    "not been tested on the case it exists for. This is a "
                    "property of the corpus, not a result about the method.")

        if gain >= 0.10 and cost <= 0.10:
            return ("GO", "The probe recovers materially more unmarked "
                          "conditions without a matching rise in fabrication.")
        if gain >= 0.10:
            return ("NO -- the gain is bought with fabrication",
                    f"Unmarked recall rises {gain:+.1%} but HCR rises "
                    f"{cost:+.1%}. The spec is explicit that this is not a "
                    "success. Invoke the fallback: anchor on the grounding "
                    "gate and the metric suite, and report the ceiling.")
        return ("NO -- no useful margin",
                f"Unmarked recall differs by {gain:+.1%}, which does not "
                "justify the probe's extra API cost over direct extraction. "
                "Invoke the fallback: anchor on the grounding gate and the "
                "metric suite, and report implicit extraction as an open "
                "problem with this ceiling.")

    def render(self) -> str:
        p, b = self.probe, self.baseline
        lines = [
            "WP1 go/no-go: Contrastive Scope Probing vs direct extraction",
            "=" * 78,
            "",
            f"  {'':<26}{'probe':>12}{'baseline':>12}{'diff':>12}",
            "  " + "-" * 62,
            f"  {'recall, unmarked':<26}{p.implicit_recall:>11.1%}"
            f"{b.implicit_recall:>12.1%}{self.implicit_gain:>+12.1%}   <- the question",
            f"  {'recall, marked':<26}{p.explicit_recall:>11.1%}"
            f"{b.explicit_recall:>12.1%}"
            f"{p.explicit_recall - b.explicit_recall:>+12.1%}",
            f"  {'recall, all':<26}{p.recall:>11.1%}{b.recall:>12.1%}"
            f"{p.recall - b.recall:>+12.1%}",
            "",
            # Printed in the same block as recall, never separately. The spec
            # requires both numbers together or neither.
            f"  {'HCR (fabrication)':<26}{p.hcr:>11.1%}{b.hcr:>12.1%}"
            f"{self.hcr_cost:>+12.1%}",
            f"  {'ungrounded branches':<26}{p.ungrounded_branches:>11}"
            f"{b.ungrounded_branches:>12}",
            f"  {'API calls':<26}{p.api_calls:>11}{b.api_calls:>12}",
        ]

        if p.implicit_total:
            ci = wilson(p.implicit_recovered, p.implicit_total)
            lines += ["", f"  probe unmarked recall 95% CI  [{ci.low:.1%}, {ci.high:.1%}]"
                          f"   on {p.implicit_total} branches"]

        test = self.paired_test()
        if test is not None:
            lines.append(f"  McNemar p = {test.p_value:.4f}  "
                         f"({test.n_discordant} discordant of {p.gold_branches})")
            if test.n_discordant < 10:
                lines.append("  Too few discordant branches to conclude from; a null "
                             "here is absence of evidence.")

        head, why = self.verdict()
        lines += ["", "=" * 78, f"  {head}", ""]
        for chunk in _wrap(why, 72):
            lines.append(f"  {chunk}")
        return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def run(
    instances: list[GoldInstance],
    *,
    live: bool,
    heuristic_nli: bool = True,
) -> ProbeComparison:
    from experiments.wp1_probe import baseline as direct
    from extraction import ContrastiveScopeProbe

    probe = ContrastiveScopeProbe(use_llm=live, heuristic_nli=heuristic_nli)

    probe_pred: dict[str, list[Branch]] = {}
    base_pred: dict[str, list[Branch]] = {}
    probe_calls = base_calls = 0

    for inst in instances:
        rec = inst.to_query_record(include_gold_pairs=False)

        result = probe.probe_record(rec)
        probe_calls += result.api_calls
        branches = [c.to_branch(f"b{i + 1}")
                    for i, c in enumerate(result.admitted)]
        # The default branch is not what either method is being judged on --
        # both are scored on the exceptions they recover. Supplying gold's
        # default to both keeps the alignment anchored so an exception cannot
        # be matched against it.
        default = next((b for b in inst.gold_branches if b.is_default), None)
        probe_pred[inst.instance_id] = ([default] if default else []) + branches

        if live:
            got, n = direct.extract(rec)
            base_calls += n
        else:
            got, n = [], 0
        base_pred[inst.instance_id] = ([default] if default else []) + got

    return ProbeComparison(
        probe=score(instances, probe_pred, name="contrastive scope probing",
                    api_calls=probe_calls),
        baseline=score(instances, base_pred, name="direct extraction",
                       api_calls=base_calls),
    )


def run_gate_ablation(instances: list[GoldInstance], *, live: bool) -> str:
    """Score the grounding gate with the local cross-encoder, then with an LLM.

    The WP1 spec names this as a decision to make on evidence. The local
    scorer is free per call and the gate is the pipeline's highest-volume
    consumer, so the LLM has to earn its cost rather than be assumed better.

    What matters is not which scorer agrees with the other -- it is what each
    ADMITS. A gate that admits more conditions raises recall and fabrication
    together, and only the pair of numbers says whether the trade is worth it.
    """
    from detection.llm_nli import LLMEntailment
    from detection.nli import get_nli
    from extraction import ContrastiveScopeProbe

    scorers = [("local cross-encoder", get_nli(heuristic=not live), 0)]
    if live:
        llm = LLMEntailment(step="wp1_gate_ablation")
        scorers.append(("LLM judge", llm, None))

    lines = ["", "Grounding-gate ablation", "=" * 78,
             f"  {'scorer':<24}{'admitted':>10}{'rejected':>10}"
             f"{'reject rate':>13}{'API calls':>11}",
             "  " + "-" * 64]

    for name, scorer, _ in scorers:
        probe = ContrastiveScopeProbe(use_llm=live, nli=scorer)
        admitted = rejected = 0
        for inst in instances:
            res = probe.probe_record(inst.to_query_record(include_gold_pairs=False))
            admitted += len(res.admitted)
            rejected += len(res.rejected)
        calls = getattr(scorer, "calls", 0)
        total = admitted + rejected
        rate = rejected / total if total else 0.0
        lines.append(f"  {name:<24}{admitted:>10}{rejected:>10}"
                     f"{rate:>12.1%}{calls:>11}")

    if not live:
        lines += ["",
                  "  OFFLINE: only the heuristic stand-in ran, so there is nothing",
                  "  to compare. Use --live for the ablation the spec asks for."]
    else:
        lines += ["",
                  "  Read the reject rates against the recall and HCR figures above.",
                  "  A scorer that rejects less raises both recall and fabrication,",
                  "  and the local one is free per call while the gate scores every",
                  "  candidate -- so the LLM has to buy a real gap, not a small one."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--live", action="store_true",
                    help="use real models; offline the baseline extracts nothing "
                         "and the comparison is meaningless")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true",
                    help="also run the grounding-gate scorer ablation")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    reproducibility.start_run(args.seed)

    instances = _load_gold(args.data, args.limit)
    if not instances:
        print(f"no gold instances with branch structure in {args.data}",
              file=sys.stderr)
        print("The probe is scored on recovering annotated conditions, so an "
              "instance whose branches are not yet filled in has no answer key.",
              file=sys.stderr)
        return 1

    print(f"{len(instances)} gold instances with branch structure, live={args.live}\n")
    if not args.live:
        print("OFFLINE: the baseline makes no API calls and extracts nothing, so")
        print("the comparison below is a harness check, not a result.\n")

    comparison = run(instances, live=args.live)
    print(comparison.render())

    if args.ablation:
        print(run_gate_ablation(instances, live=args.live))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        head, why = comparison.verdict()
        args.json.write_text(json.dumps({
            "probe": comparison.probe.as_dict(),
            "baseline": comparison.baseline.as_dict(),
            "implicit_gain": round(comparison.implicit_gain, 4),
            "hcr_cost": round(comparison.hcr_cost, 4),
            "verdict": head, "reasoning": why,
        }, indent=2), encoding="utf-8")
        print(f"\nresults -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
