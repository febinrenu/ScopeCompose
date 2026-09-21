"""Append-only log of every LLM call.

Required by the shared budget discipline: nobody should discover a budget or
rate-limit problem *after* a full-corpus run. One JSONL line per call, reviewed
weekly against the volume estimate.

On the zero-spend configuration, dollar cost is 0.0 and the interesting columns
are token volume and call count -- because the binding constraint is the
provider's rate limit, not money.

One column earns its place beyond accounting: ``fell_back``. When the primary
provider rate-limits and the client silently drops to a local model, the
results change. That substitution must appear in the report rather than be
discovered later as an unexplained variance between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

DEFAULT_LOG_PATH = Path(os.environ.get("CONFLICT_RAG_COSTLOG", "logs/api_cost.jsonl"))


@dataclass
class CallRecord:
    """One LLM call."""

    timestamp: float
    step: str
    """Which pipeline component made the call, e.g. 'a2_stage2_detection'.
    Without this the log cannot attribute spend, which is the whole point."""
    tier: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    fell_back: bool = False
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostLog:
    """Append-only JSONL writer."""

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: CallRecord) -> None:
        if not self.enabled:
            return
        line = json.dumps(asdict(rec), sort_keys=True, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    def log(
        self,
        *,
        step: str,
        tier: str,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached: bool = False,
        fell_back: bool = False,
        cost_usd: float = 0.0,
        latency_s: float = 0.0,
        error: str | None = None,
        **meta: Any,
    ) -> CallRecord:
        rec = CallRecord(
            timestamp=time.time(), step=step, tier=tier, provider=provider, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cached=cached, fell_back=fell_back, cost_usd=cost_usd,
            latency_s=latency_s, error=error, meta=meta,
        )
        self.record(rec)
        return rec

    def read(self) -> Iterator[CallRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield CallRecord(**json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # a partially-written line from a killed run


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise(records: list[CallRecord]) -> dict[str, Any]:
    by_step: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "cost_usd": 0.0, "fell_back": 0, "errors": 0}
    )
    by_tier: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "cached": 0, "tokens": 0, "cost_usd": 0.0}
    )

    total = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "cost_usd": 0.0, "fell_back": 0, "errors": 0, "latency_s": 0.0}

    for r in records:
        s = by_step[r.step]
        s["calls"] += 1
        s["cached"] += int(r.cached)
        s["prompt_tokens"] += r.prompt_tokens
        s["completion_tokens"] += r.completion_tokens
        s["cost_usd"] += r.cost_usd
        s["fell_back"] += int(r.fell_back)
        s["errors"] += int(bool(r.error))

        t = by_tier[r.tier]
        t["calls"] += 1
        t["cached"] += int(r.cached)
        t["tokens"] += r.total_tokens
        t["cost_usd"] += r.cost_usd

        total["calls"] += 1
        total["cached"] += int(r.cached)
        total["prompt_tokens"] += r.prompt_tokens
        total["completion_tokens"] += r.completion_tokens
        total["cost_usd"] += r.cost_usd
        total["fell_back"] += int(r.fell_back)
        total["errors"] += int(bool(r.error))
        total["latency_s"] += r.latency_s

    total["cache_hit_rate"] = (total["cached"] / total["calls"]) if total["calls"] else 0.0
    total["billable_calls"] = total["calls"] - total["cached"]

    return {"total": total, "by_step": dict(by_step), "by_tier": dict(by_tier)}


def render(summary: dict[str, Any]) -> str:
    t = summary["total"]
    lines = [
        "API usage",
        "=" * 72,
        f"  calls            {t['calls']:>10,}   ({t['cached']:,} served from cache, "
        f"{t['cache_hit_rate']:.0%} hit rate)",
        f"  billable calls   {t['billable_calls']:>10,}",
        f"  prompt tokens    {t['prompt_tokens']:>10,}",
        f"  completion tok.  {t['completion_tokens']:>10,}",
        f"  cost             {t['cost_usd']:>10.4f} USD",
        f"  total latency    {t['latency_s']:>10.1f} s",
    ]

    if t["errors"]:
        lines.append(f"  errors           {t['errors']:>10,}")

    if t["fell_back"]:
        lines += [
            "",
            f"  !! {t['fell_back']:,} call(s) FELL BACK to the local model after rate-limiting.",
            "     Those results came from a different model than intended. This must be",
            "     reported, not averaged in silently -- see CLAUDE.md, research-integrity rules.",
        ]

    if summary["by_tier"]:
        lines += ["", "By tier", "-" * 72,
                  f"  {'tier':<20} {'calls':>8} {'cached':>8} {'tokens':>12} {'USD':>10}"]
        for name, v in sorted(summary["by_tier"].items(), key=lambda kv: -kv[1]["calls"]):
            lines.append(f"  {name:<20} {v['calls']:>8,} {v['cached']:>8,} "
                         f"{v['tokens']:>12,} {v['cost_usd']:>10.4f}")

    if summary["by_step"]:
        lines += ["", "By step", "-" * 72,
                  f"  {'step':<32} {'calls':>8} {'cached':>8} {'tokens':>12} {'USD':>9}"]
        for name, v in sorted(summary["by_step"].items(), key=lambda kv: -kv[1]["calls"]):
            tok = v["prompt_tokens"] + v["completion_tokens"]
            flag = "  <- fell back" if v["fell_back"] else ""
            lines.append(f"  {name:<32} {v['calls']:>8,} {v['cached']:>8,} "
                         f"{tok:>12,} {v['cost_usd']:>9.4f}{flag}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report on the API cost log.")
    ap.add_argument("command", choices=["report"], nargs="?", default="report")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = ap.parse_args(argv)

    log = CostLog(args.log)
    records = list(log.read())
    if not records:
        print(f"no entries in {args.log} yet")
        return 0

    summary = summarise(records)
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
