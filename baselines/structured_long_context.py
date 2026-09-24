"""The structured long-context baseline — the paper's decisive experiment.

Proposal §6.3. A long-context model is given the **full concatenated passage
set** and asked to produce an explicit branch parse in the same schema as the
gold structure: condition, outcome, applicability, and a source passage per
branch. It is scored with the identical metrics as the pipeline, so the
comparison is same-metric and direct rather than qualitative.

**Why this is the experiment and not a formality.** §6.5 says outright that if
this baseline matches the composition pipeline's Preservation Rate, the central
architectural claim does not hold as stated. The whole argument for a
resolution *operator* is that a model handed all the passages at once still
cannot be relied on to preserve every scoped branch. If it can, the pipeline is
unnecessary and the contribution narrows to the extraction method, the metric
suite and the benchmark.

Two variants, and the distinction matters:

``structured`` (primary)
    Asked for the branch schema. This is the strong form of the baseline and
    the one §6.3 specifies.
``freetext`` (secondary)
    Asked only to "mention any conditions or exceptions". Retained to separate
    two effects that are easy to confuse: how much of any observed advantage
    comes from the pipeline *architecture*, and how much comes merely from
    *requiring structured output at all*. If the free-text variant is much
    worse than the structured one, a good deal of the pipeline's apparent
    benefit is really the benefit of asking for structure.

**Read README §5 before interpreting any result from this module.** On the
zero-spend configuration this runs on open weights, which makes the outcome
interpretable in one direction only: if the baseline wins, that is decisive
falsification and fully valid; if it loses, a reviewer will attribute the gap
to model tier rather than architecture.

Usage::

    python -m baselines.structured_long_context --data gold.jsonl --limit 20
    python -m baselines.structured_long_context --data gold.jsonl --variant freetext
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from api_budget import BackendError, Tier, get_client
from contract.gold import Applicability, Branch, GoldInstance

STEP = "baseline_structured_long_context"

#: Seconds between calls. The BULK-tier lesson applies here too: the client's
#: exponential backoff tops out short of the 60-second rate-limit window, so
#: backoff alone cannot clear a tokens-per-minute exhaustion. Pacing can.
PACE_S = 8.0

STRUCTURED_SYSTEM = """\
You read a set of passages retrieved for one question and produce a complete
BRANCH STRUCTURE describing the answer.

A branch structure has exactly one DEFAULT branch — what holds absent any
further information — plus one EXCEPTION branch for every narrower case where
the outcome differs.

Return ONLY JSON:
{"branches": [
  {"is_default": true,
   "condition": null,
   "outcome": "<what happens>",
   "applicability": "<who or what this covers>",
   "source": "<passage id, e.g. p0>"},
  {"is_default": false,
   "condition": "<the condition that triggers this branch>",
   "outcome": "<what happens instead>",
   "applicability": "<the narrower group this covers>",
   "source": "<passage id>"}
]}

Rules:
- EVERY branch cites the passage id it comes from. Never invent a passage id,
  and never assert a branch no passage supports.
- Include an exception branch ONLY where the outcome genuinely DIFFERS from the
  default for a narrower group. A passage restating the default for a specific
  group is not an exception — do not create a branch for it.
- If the passages simply contradict each other for the same group, that is not
  a branch structure. Return the default branch only.
- If there are no exceptions, return the default branch alone. One branch is
  frequently the correct answer.

Be complete and be precise. A missed exception and an invented one are both
errors."""

FREETEXT_SYSTEM = """\
You read a set of passages retrieved for one question and answer it.

State any conditions or exceptions that qualify the answer.

Return ONLY JSON: {"answer": "<your answer>"}"""

USER_TEMPLATE = """\
Question: {query}

{passages}

JSON:"""


@dataclass
class BaselineOutput:
    instance_id: str
    variant: str
    branches: list[Branch] = field(default_factory=list)
    raw_text: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def n_exceptions(self) -> int:
        return sum(1 for b in self.branches if not b.is_default)


def _format_passages(inst: GoldInstance) -> str:
    return "\n\n".join(f"[{p.id}] {p.text}" for p in inst.passages)


def _parse_branches(raw: dict, inst: GoldInstance) -> list[Branch]:
    """Turn the model's JSON into validated Branch objects.

    Repairs are deliberately conservative. A branch citing a passage that does
    not exist keeps its claim but loses the citation, so it is later counted as
    ungrounded rather than quietly attributed to a real passage — attributing
    it would hide exactly the fabrication HCR measures.
    """
    known = {p.id for p in inst.passages}
    out: list[Branch] = []
    seen_default = False

    for i, item in enumerate(raw.get("branches", []) or []):
        if not isinstance(item, dict):
            continue
        outcome = str(item.get("outcome") or "").strip()
        if not outcome:
            continue

        is_default = bool(item.get("is_default", False))
        condition = item.get("condition")
        condition = str(condition).strip() if condition else None

        # The schema forbids two defaults and forbids a default with a
        # condition. Demote rather than drop: the claim was made and should be
        # scored.
        if is_default and seen_default:
            is_default = False
            condition = condition or "unspecified"
        if is_default and condition:
            condition = None
        if not is_default and not condition:
            condition = "unspecified"
        if is_default:
            seen_default = True

        source = item.get("source")
        source = str(source).strip() if source else None
        if source not in known:
            source = None

        descriptor = str(item.get("applicability") or "").strip() or (
            "all cases" if is_default else "unspecified"
        )

        try:
            out.append(Branch(
                branch_id=f"pred{i}",
                condition=condition,
                outcome=outcome[:300],
                applicability=Applicability(descriptor=descriptor[:200],
                                            is_default=is_default),
                supporting_passage=source,
            ))
        except Exception:
            continue   # a branch the schema rejects is not scoreable
    return out


def run_instance(
    inst: GoldInstance,
    *,
    variant: str = "structured",
    client=None,
    tier: Tier = Tier.LONG_CONTEXT,
) -> BaselineOutput:
    client = client or get_client()
    system = STRUCTURED_SYSTEM if variant == "structured" else FREETEXT_SYSTEM
    started = time.time()

    try:
        result = client.complete(
            messages=[{"role": "user", "content": USER_TEMPLATE.format(
                query=inst.query, passages=_format_passages(inst))}],
            system=system,
            tier=tier,
            step=f"{STEP}_{variant}",
            response_format={"type": "json_object"},
        )
    except BackendError as exc:
        return BaselineOutput(inst.instance_id, variant,
                              latency_s=time.time() - started,
                              error=f"{type(exc).__name__}: {str(exc)[:80]}")

    try:
        raw = result.json()
    except ValueError:
        from detection.stage2 import _salvage_json

        raw = _salvage_json(result.text)

    branches: list[Branch] = []
    if variant == "structured" and isinstance(raw, dict):
        branches = _parse_branches(raw, inst)

    return BaselineOutput(
        instance_id=inst.instance_id,
        variant=variant,
        branches=branches,
        raw_text=(raw.get("answer", "") if isinstance(raw, dict) else "") or result.text,
        latency_s=result.latency_s,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cached=result.cached,
    )


def run(
    instances: list[GoldInstance],
    *,
    variant: str = "structured",
    pace: float = PACE_S,
    client=None,
    progress: bool = True,
) -> list[BaselineOutput]:
    out: list[BaselineOutput] = []
    for i, inst in enumerate(instances):
        if i and pace:
            time.sleep(pace)
        result = run_instance(inst, variant=variant, client=client)
        out.append(result)
        if progress:
            state = "ok" if result.ok else (result.error or "?")[:40]
            cached = " (cached)" if result.cached else ""
            print(f"  [{i + 1:>3}/{len(instances)}] {state:<42} "
                  f"{len(result.branches)} branches{cached}")
    return out


def main(argv: list[str] | None = None) -> int:
    import reproducibility
    from experiments.run_baselines import load_gold

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="JSONL of GOLD instances")
    ap.add_argument("--variant", choices=["structured", "freetext"], default="structured")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap instances, to protect the daily token budget")
    ap.add_argument("--pace", type=float, default=PACE_S)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args(argv)

    reproducibility.start_run(args.seed)

    instances = load_gold(args.data)
    if args.limit:
        instances = instances[:args.limit]
    if not instances:
        print(f"no gold instances in {args.data}", file=sys.stderr)
        return 1

    print(f"{len(instances)} instances, variant={args.variant}\n")
    outputs = run(instances, variant=args.variant, pace=args.pace)

    ok = [o for o in outputs if o.ok]
    failed = len(outputs) - len(ok)
    tokens = sum(o.prompt_tokens + o.completion_tokens for o in ok)

    print()
    print(f"  completed        {len(ok)}/{len(outputs)}"
          + (f"   ({failed} failed)" if failed else ""))
    print(f"  tokens           {tokens:,}")
    print(f"  mean branches    {sum(len(o.branches) for o in ok) / max(1, len(ok)):.2f}")
    print(f"  mean exceptions  {sum(o.n_exceptions for o in ok) / max(1, len(ok)):.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as fh:
            for o in outputs:
                fh.write(json.dumps({
                    "instance_id": o.instance_id,
                    "variant": o.variant,
                    "branches": [b.model_dump(mode="json") for b in o.branches],
                    "raw_text": o.raw_text[:2000],
                    "error": o.error,
                    "cached": o.cached,
                    "latency_s": round(o.latency_s, 3),
                    "prompt_tokens": o.prompt_tokens,
                    "completion_tokens": o.completion_tokens,
                }, ensure_ascii=False) + "\n")
        print(f"\noutputs -> {args.output}")

    print("\n  Scoring: python -m experiments.run_decisive --data <gold.jsonl>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
