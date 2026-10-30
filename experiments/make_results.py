"""Assemble ``results.json``: every number the paper may quote, from cell files and nothing else.

This is the single place that turns raw system answers into the analysis the pre-registration
promised. It reads the answers each cell wrote (``experiments.run_matrix``), scores them with the
pinned reader and NLI model, runs the confirmatory family, the oracle cascade and (when given) the
validity battery, and writes one JSON file. The claims ledger (``reporting.claims``) renders the
manuscript from that file, so the chain is

    cell files -> make_results -> results.json -> ledger -> manuscript

with no hand-typed number anywhere along it.

Nothing is computed here that is not specified in ``prereg/preregistration.md``. In particular
the narrowing rule is applied from the data (S1 below its floor makes every cross-document
result exploratory) and is recorded in the output, and direction is part of each hypothesis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from contract.gold import GoldInstance
from experiments.analysis import Confirmatory, Paired, cca_by_instance, confirmatory
from experiments.oracle_cascade import cascade
from metrics.case_accuracy import DEFAULT_NLI_THRESHOLD, NLIScorerLike, Reader, aggregate

SYSTEMS_PRIMARY = ("B0", "B3", "B7c", "B8", "O3")


def _plain(x: Any) -> Any:
    """JSON-safe: dataclasses and enums to plain data, NaN/inf to ``None`` (never a reportable number)."""
    import math
    from enum import Enum

    if is_dataclass(x) and not isinstance(x, type):
        return _plain(asdict(x))
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def paired_dict(p: Paired) -> dict:
    d = _plain(p)
    d["underpowered_for"] = p.underpowered_for
    # "no effect is detectable" is a result, not a missing value: it needs a form the ledger can print.
    d["mdd_text"] = ("no effect within the observed discordance is detectable at this sample size"
                     if p.mdd_points is None else f"{p.mdd_points:.1f} points")
    return d


def score_systems(
    instances: Sequence[GoldInstance],
    answers: Mapping[str, Mapping[str, str]],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
    threshold_tuned: bool = False,
) -> tuple[dict[str, dict[str, bool]], dict[str, dict]]:
    """Per system: instance -> CCA-all strict, and the aggregate scores."""
    per_instance: dict[str, dict[str, bool]] = {}
    summaries: dict[str, dict] = {}
    for system, ans in answers.items():
        scored = cca_by_instance(instances, ans, reader, nli, threshold=threshold)
        per_instance[system] = {iid: r.cca_all for iid, r in scored.items()}
        agg = aggregate(list(scored.values()), threshold=threshold, threshold_tuned=threshold_tuned)
        summaries[system] = {
            "n": agg.n_instances, "cca_all": agg.cca_all.point, "cca_all_low": agg.cca_all.low,
            "cca_all_high": agg.cca_all.high, "cca_all_lenient": agg.cca_all_lenient.point,
            "cca_exception": None if agg.cca_exception is None else agg.cca_exception.point,
            "cca_default": None if agg.cca_default is None else agg.cca_default.point,
            "over_generalisation": None if agg.over_generalisation is None else agg.over_generalisation.point,
            "clarification_rate": agg.clarification_rate, "failures": agg.failures,
        }
    return per_instance, summaries


def assemble(
    instances: Sequence[GoldInstance],
    answers: Mapping[str, Mapping[str, str]],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    comparator: str,
    s1_floor: int = 60,
    threshold: float = DEFAULT_NLI_THRESHOLD,
    threshold_tuned: bool = False,
    h2_p: float | None = None,
    cascade_system: str = "O3",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The full results dictionary for one population (S1) and one backbone."""
    per_instance, summaries = score_systems(instances, answers, reader, nli, threshold=threshold,
                                            threshold_tuned=threshold_tuned)
    s1_test = len(instances)
    conf: Confirmatory = confirmatory(per_instance, comparator=comparator, s1_test=s1_test, s1_floor=s1_floor,
                                      h2_p=h2_p)
    out: dict[str, Any] = {
        "s1": {"n_test": s1_test, "floor": s1_floor, "narrowing": conf.exploratory,
               "status": "exploratory (below floor)" if conf.exploratory else "confirmatory"},
        "nli_threshold": {"value": threshold, "tuned": threshold_tuned},
        "cca": summaries,
        "confirmatory": {
            "exploratory": conf.exploratory, "reason": conf.exploratory_reason,
            "comparator": comparator,
            "supported": {h: conf.supported(h) for h in conf.results},
            "adjusted_p": {h: p for h, (p, _) in conf.adjusted.items()},
            "reject_after_holm": {h: rej for h, (_, rej) in conf.adjusted.items()},
            "direction_as_predicted": conf.directions_supported,
        },
        "h1": paired_dict(conf.results["H1"]),
        "h3": paired_dict(conf.results["H3"]),
    }
    if cascade_system in answers:
        c = cascade(instances, answers[cascade_system], reader, nli, threshold=threshold)
        out["cascade"] = {"accuracy": {lv: c.accuracy(lv) for lv in ("L0", "L1", "L2")},
                          "ceiling_gap": c.ceiling_gap,
                          "extraction_and_alignment": paired_dict(c.deltas["L1-L0"]),
                          "composition_and_verbalisation": paired_dict(c.deltas["L2-L1"])}
    if extra:
        out.update(_plain(dict(extra)))
    return _plain(out)


def write(results: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from detection.nli import CrossEncoderNLI
    from experiments.run_matrix import Cell, load_answers, load_instances, select
    from metrics.case_accuracy import LLMReader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    ap.add_argument("--backbone", required=True, choices=("primary", "local"))
    ap.add_argument("--population", default="S1")
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--comparator", required=True, choices=("B7c", "B8"))
    ap.add_argument("--nli-threshold", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/results.json"))
    args = ap.parse_args(argv)

    if args.split == "test":
        from experiments.staging import assert_frozen

        assert_frozen(args.data)
    insts = select(load_instances(args.data),
                   Cell("B0", args.backbone, args.population, "underspecified", args.split))
    answers = {}
    for sid in SYSTEMS_PRIMARY:
        f = args.runs / Cell(sid, args.backbone, args.population, "underspecified", args.split).name
        f = f.with_suffix(".jsonl")
        if f.exists():
            answers[sid] = load_answers(f)
    res = assemble([i for i in insts if i.personas], answers, LLMReader(), CrossEncoderNLI(),
                   comparator=args.comparator, threshold=args.nli_threshold or DEFAULT_NLI_THRESHOLD,
                   threshold_tuned=args.nli_threshold is not None)
    write(res, args.out)
    print(f"wrote {args.out}: {res['s1']['n_test']} instances, {res['s1']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
