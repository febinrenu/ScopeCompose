"""Batch evaluation: the pipeline over a benchmark split, scored as it goes.

Each instance runs through the same orchestrator as a single live run, with
its own private tracer. The batch stream carries one ``item`` per instance
and a ``metrics`` snapshot after each, so a dashboard can fill in live:
detector precision / recall, routing counts, the relation confusion matrix,
and preservation per system.
"""

from __future__ import annotations

from typing import Any

from contract.models import ConflictType
from service.events import Tracer, in_stage
from service.orchestrator import RunRequest, benchmark, run_query

SPLITS = ("all", "train", "dev", "test")


def select(split: str, limit: int | None, *, branchy_only: bool = False) -> list[str]:
    ids = [i for i, inst in sorted(benchmark().items())
           if (split == "all" or inst.split == split) and (not branchy_only or inst.gold_branches)]
    return ids[:limit] if limit else ids


def run_eval(params: dict[str, Any], tracer: Tracer) -> dict[str, Any]:
    split = params.get("split", "dev")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    limit = int(params.get("limit") or 0) or None
    mode = params.get("mode", "offline")
    policy = params.get("detector_policy", "as_run")
    ids = select(split, limit, branchy_only=bool(params.get("branchy_only")))

    tp = fp = fn = tn = 0
    routing: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}
    sums: dict[str, dict[str, float]] = {}
    faithful = 0
    cost = 0.0
    llm_calls = 0

    with in_stage(tracer, "eval", f"Evaluate {len(ids)} instance(s)", total=len(ids), split=split, mode=mode,
                  detector_policy=policy):
        for n, iid in enumerate(ids, start=1):
            inst = benchmark()[iid]
            sub = Tracer(f"{tracer.run_id}:{iid}")
            try:
                s = run_query(RunRequest(source="benchmark", record_id=iid, mode=mode, detector_policy=policy), sub)
            except Exception as exc:  # one bad instance must not end the batch
                tracer.warn("eval", f"{iid} failed", id=iid, error=f"{type(exc).__name__}: {exc}"[:300])
                continue

            gold_conflict = inst.gold_conflict_type is not ConflictType.NO_CONFLICT
            verdicts = [e.payload for e in sub.events if e.stage == "detect" and e.title == "verdicts"]
            detected = any(p["is_conflict"] for p in (verdicts[0]["pairs"] if verdicts else []))
            tp += gold_conflict and detected
            fp += (not gold_conflict) and detected
            fn += gold_conflict and not detected
            tn += (not gold_conflict) and not detected

            res = s.get("resolution", "pass_through")
            routing[res] = routing.get(res, 0) + 1
            g = inst.gold_scope_relation.value if inst.gold_scope_relation else "none"
            p = s.get("relation") or "none"
            confusion.setdefault(g, {})[p] = confusion.setdefault(g, {}).get(p, 0) + 1
            faithful += bool(s.get("faithful"))
            cost += s.get("llm", {}).get("cost_usd", 0.0)
            llm_calls += s.get("llm", {}).get("calls", 0)
            for system, sc in (s.get("scores") or {}).items():
                acc = sums.setdefault(system, {"PR": 0.0, "SR": 0.0, "n": 0})
                if sc.get("PR") is not None:
                    acc["PR"] += sc["PR"]
                    acc["SR"] += sc["SR"]
                    acc["n"] += 1

            tracer.data("eval", "item", n=n, total=len(ids), id=iid, query=inst.query,
                        gold_type=inst.gold_conflict_type.value,
                        gold_relation=inst.gold_scope_relation.value if inst.gold_scope_relation else None,
                        detected=detected, override=s.get("override"), relation=s.get("relation"),
                        resolution=res, faithful=s.get("faithful"), scores=s.get("scores"),
                        answer=s.get("answer"), ms=round(sum(s.get("stage_ms", {}).values()), 1))

            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
            tracer.data("eval", "metrics", n=n, total=len(ids),
                        detector={"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                  "precision": precision, "recall": recall, "f1": f1},
                        routing=routing, confusion=confusion, faithful=faithful,
                        preservation={k: {"PR": v["PR"] / v["n"], "SR": v["SR"] / v["n"], "n": v["n"]}
                                      for k, v in sums.items() if v["n"]},
                        llm={"calls": llm_calls, "cost_usd": cost})

    return {"split": split, "n": len(ids), "detector": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "routing": routing, "confusion": confusion, "faithful": faithful,
            "preservation": {k: {"PR": v["PR"] / v["n"], "SR": v["SR"] / v["n"], "n": v["n"]}
                             for k, v in sums.items() if v["n"]},
            "llm": {"calls": llm_calls, "cost_usd": cost}}
