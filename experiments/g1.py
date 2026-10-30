"""Gate G1: do the planned cells fit the free-tier quota? (plan v4 section 4, gate G1)

Groq caps a model at about 200K tokens a day. The plan's estimates per call were guesses
(and one of them, B7's, described a prompt that does not exist -- ``docs/v7_decisions.md``
D1.4), so before anything is committed to the primary backbone this measures **real**
tokens per instance for each system, from the client's own usage numbers, and projects the
planned matrix against the cap.

Two outputs:

* **per-system measurement** -- mean and 90th-percentile tokens and calls per instance, and
  whether each is within the estimate plus the plan's 25% tolerance;
* **a projection** -- tokens for the planned matrix, days of quota it needs, and, if it does
  not fit, the plan's pre-agreed **cut order** applied until it does. The confirmatory cells
  (B0, B3, B7c and O3 on S1) are never cut: they *are* H1 and H3.

The gate result is advisory output. It does not edit ``prereg/gates.json``; flipping a gate
is a decision a person makes with this table in front of them (``--update-gate`` does it
explicitly and records the measurement beside the status).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from baselines.registry import oracle_conflicts
from contract.gold import GoldInstance

#: The plan's estimates per *instance* (tokens), to be replaced by measurement.
ESTIMATE = {"B0": 1500, "B3": 1500, "B7c": 4000, "B8": 7500, "O3": 7500}
TOLERANCE = 0.25
DAILY_CAP = 200_000
#: Days of quota the primary backbone can use between the freeze and the analysis.
DEFAULT_AVAILABLE_DAYS = 3.5

#: The pre-agreed order in which planned work is dropped if the matrix does not fit. Items
#: protected from cutting are the confirmatory cells.
CUT_ORDER = ("persona setting on the primary backbone", "B8 on the primary backbone",
             "controls on the primary backbone", "S3 on the primary backbone")
PROTECTED = ("B0, B3 on S1", "B7c, O3 on S1")


@dataclass(frozen=True)
class SystemMeasure:
    system: str
    n: int
    mean_prompt: float
    mean_completion: float
    mean_total: float
    p90_total: float
    mean_calls: float
    estimate: int
    within_tolerance: bool


def _p90(xs: Sequence[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)] if s else 0.0


def measure(system_ids: Sequence[str], instances: Sequence[GoldInstance], *, tier, client=None,
            tolerance: float = TOLERANCE) -> dict[str, SystemMeasure]:
    """Run every system on every instance once and record its real token use."""
    from experiments.run_matrix import Cell, build_system, domain_of

    out: dict[str, SystemMeasure] = {}
    for sid in system_ids:
        totals, prompts, comps, calls = [], [], [], []
        for inst in instances:
            cell = Cell(sid, "primary" if tier.value == "BACKBONE_PRIMARY" else "local", "S5", "underspecified",
                        "dev")
            system = build_system(cell, domain_of(inst), client)
            ans = system.answer(inst.query, inst.passages,
                                conflicts=oracle_conflicts(inst) if sid == "B3" else ())
            totals.append(ans.total_tokens)
            prompts.append(ans.prompt_tokens)
            comps.append(ans.completion_tokens)
            calls.append(ans.calls)
        est = ESTIMATE[sid]
        out[sid] = SystemMeasure(sid, len(totals), mean(prompts), mean(comps), mean(totals), _p90(totals),
                                 mean(calls), est, mean(totals) <= est * (1 + tolerance))
    return out


@dataclass
class Projection:
    plan: dict[str, int]
    tokens: dict[str, float]
    total: float
    days: float
    available_days: float
    cuts: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.days <= self.available_days


def project(measures: Mapping[str, SystemMeasure], plan: Mapping[str, int], *, daily_cap: int = DAILY_CAP,
            available_days: float = DEFAULT_AVAILABLE_DAYS) -> Projection:
    """Tokens and days for ``plan`` (system -> number of instances). Uses the *measured* mean per instance."""
    missing = [s for s in plan if s not in measures]
    if missing:
        raise KeyError(f"no measurement for {missing}")
    tokens = {s: measures[s].mean_total * n for s, n in plan.items()}
    total = sum(tokens.values())
    return Projection(dict(plan), tokens, total, total / daily_cap, available_days)


def planned_matrix(n_s1: int, n_controls: int, *, persona_n: int = 0, include_b8: bool = True) -> dict[str, int]:
    """The primary-backbone matrix of the 7-day plan, as system -> instance count.

    B0 and B3 run on S1 and the controls; B7c and O3 on S1 conditionals; the persona setting
    adds one extra B0/B3 answer per persona; B8 is optional.
    """
    plan = {"B0": n_s1 + n_controls + persona_n, "B3": n_s1 + n_controls + persona_n,
            "B7c": n_s1, "O3": n_s1}
    if include_b8:
        plan["B8"] = n_s1
    return plan


def fit_to_quota(measures: Mapping[str, SystemMeasure], *, n_s1: int, n_controls: int, persona_n: int,
                 daily_cap: int = DAILY_CAP, available_days: float = DEFAULT_AVAILABLE_DAYS) -> Projection:
    """Apply the cut order, one step at a time, until the matrix fits -- never touching the protected cells."""
    persona, b8, controls = persona_n, "B8" in measures, n_controls
    cuts: list[str] = []
    while True:
        plan = planned_matrix(n_s1, controls, persona_n=persona, include_b8=b8)
        p = project({k: v for k, v in measures.items() if k in plan}, plan, daily_cap=daily_cap,
                    available_days=available_days)
        p.cuts = list(cuts)
        if p.fits:
            return p
        if persona:
            persona = 0
            cuts.append(CUT_ORDER[0])
        elif b8:
            b8 = False
            cuts.append(CUT_ORDER[1])
        elif controls:
            controls = 0
            cuts.append(CUT_ORDER[2])
        else:
            p.cuts.append("NOTHING LEFT TO CUT: the protected confirmatory cells alone exceed the quota; "
                          "change the primary backbone or the reasoning setting")
            return p


def render(measures: Mapping[str, SystemMeasure], proj: Projection | None = None) -> str:
    lines = ["G1: measured tokens per instance", "=" * 78,
             f"  {'system':<8}{'n':>4}{'prompt':>9}{'compl.':>9}{'total':>9}{'p90':>9}{'calls':>7}{'estimate':>10}"
             f"{'within +25%':>13}", "  " + "-" * 74]
    for m in measures.values():
        lines.append(f"  {m.system:<8}{m.n:>4}{m.mean_prompt:>9.0f}{m.mean_completion:>9.0f}{m.mean_total:>9.0f}"
                     f"{m.p90_total:>9.0f}{m.mean_calls:>7.1f}{m.estimate:>10}{('yes' if m.within_tolerance else 'NO'):>13}")
    if proj is not None:
        lines += ["", f"Projected: {proj.total:,.0f} tokens = {proj.days:.1f} days of a {DAILY_CAP:,}-token/day "
                      f"quota (available {proj.available_days} days): {'FITS' if proj.fits else 'DOES NOT FIT'}"]
        if proj.cuts:
            lines.append("Cuts applied, in the pre-agreed order:")
            lines += [f"  - {c}" for c in proj.cuts]
        lines.append("Protected, never cut: " + "; ".join(PROTECTED))
    return "\n".join(lines)


def update_gate(path: Path, proj: Projection, measures: Mapping[str, SystemMeasure]) -> dict:
    """Write G1's status and the measurement beside it. Explicit; never called implicitly."""
    gates = json.loads(path.read_text(encoding="utf-8"))
    gates["G1"] = {**gates.get("G1", {}), "status": "pass" if proj.fits else "fail",
                   "measured": {k: asdict(v) for k, v in measures.items()},
                   "projection": {"tokens": proj.total, "days": proj.days, "cuts": proj.cuts}}
    path.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8", newline="\n")
    return gates["G1"]


def main(argv: list[str] | None = None) -> int:
    from api_budget import Tier
    from experiments.run_matrix import load_instances

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dev instances with personas")
    ap.add_argument("--tier", choices=("BACKBONE_PRIMARY", "BACKBONE_LOCAL"), default="BACKBONE_PRIMARY")
    ap.add_argument("--systems", nargs="+", default=["B0", "B3", "B7c", "O3"])
    ap.add_argument("--limit", type=int, default=5, help="instances to measure on (uses real quota)")
    ap.add_argument("--n-s1", type=int, required=True, help="planned S1 test instances")
    ap.add_argument("--n-controls", type=int, default=20)
    ap.add_argument("--personas", type=int, default=0, help="planned persona-setting answers")
    ap.add_argument("--update-gate", action="store_true")
    args = ap.parse_args(argv)

    insts = [i for i in load_instances(args.data) if i.personas][: args.limit]
    ms = measure(args.systems, insts, tier=Tier(args.tier))
    proj = fit_to_quota(ms, n_s1=args.n_s1, n_controls=args.n_controls, persona_n=args.personas)
    print(render(ms, proj))
    if args.update_gate:
        from experiments.staging import GATES

        print("G1 ->", update_gate(GATES, proj, ms)["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
