"""Direct zero-shot extraction — the baseline Contrastive Scope Probing must beat.

A ConditionalQA-style condition selector: hand the model every passage at once
and ask what conditions govern the answer. One call, no contrastive questions,
no grounding gate.

**This is the honest comparison, and it is deliberately not a straw man.** The
probe's claim is that *asking contrastively* recovers conditions a direct ask
misses, and that only means something if the direct ask is given the same
passages, the same model and a competent prompt. A baseline weakened to make
the probe look good would make the WP1 go/no-go unanswerable.

Two differences from the probe, and they are the whole experiment:

* **No presupposition.** The probe asks "under what circumstances does the rule
  *not* apply?", which is hard to answer vacuously. This asks "what conditions
  are there?", which is easy to answer with nothing.
* **No grounding gate.** Whatever the model proposes is kept. That is what makes
  the Hallucinated-Condition Rate comparison meaningful: the gate is the probe's
  cost as well as its defence, and a baseline without one shows what the gate is
  buying.
"""

from __future__ import annotations

import json

from api_budget.client import LLMClient, Tier, get_client
from contract.gold import Applicability, Branch, Explicitness
from contract.models import QueryRecord

_SYSTEM = """You extract the conditions that govern an answer.

You are given a question and several passages. Some answers only hold under
particular conditions -- a customer type, a product, a date, a threshold.

List every condition you find, with the outcome that applies when it holds.

Rules:
- Use only what the passages say.
- Cite the passage id each condition comes from.
- If an answer holds unconditionally, say so with condition null.

Return JSON:
{"branches": [{"condition": "..." or null, "outcome": "...",
               "applies_to": "...", "passage": "p0"}]}"""


def extract(record: QueryRecord, *, client: LLMClient | None = None,
            tier: Tier = Tier.BULK) -> tuple[list[Branch], int]:
    """Return ``(branches, api_calls)``. One call per record."""
    client = client or get_client()
    passages = "\n\n".join(f"[{p.id}] {p.text}" for p in record.passages)

    result = client.complete(
        messages=[{"role": "user", "content":
                   f"Question: {record.query}\n\n{passages}\n\n"
                   "List the conditions."}],
        system=_SYSTEM,
        tier=tier,
        step="wp1_baseline_direct_extraction",
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        raw = result.json() or {}
    except ValueError:
        from detection.stage2 import _salvage_json
        try:
            raw = _salvage_json(result.text) or {}
        except Exception:
            return [], 1

    known = {p.id for p in record.passages}
    branches: list[Branch] = []
    for i, item in enumerate(raw.get("branches") or []):
        if not isinstance(item, dict):
            continue
        outcome = str(item.get("outcome") or "").strip()
        if not outcome:
            continue
        cond = item.get("condition")
        cond = str(cond).strip() if cond else None
        passage = str(item.get("passage") or "").strip()
        descriptor = str(item.get("applies_to") or cond or "all cases").strip()

        try:
            branches.append(Branch(
                branch_id=f"b{i}",
                condition=cond,
                outcome=outcome,
                applicability=Applicability(
                    descriptor=("all cases" if cond is None else descriptor),
                    is_default=(cond is None)),
                explicitness=Explicitness.EXPLICIT,
                # Kept even when it cites nothing or cites a passage that does
                # not exist. That is the point of the comparison: an ungrounded
                # branch is what HCR counts, and dropping it here would hide
                # the baseline's fabrication rate behind a filter the baseline
                # does not have.
                supporting_passage=(passage if passage in known else None),
            ))
        except Exception:
            # A branch the schema rejects -- usually a condition with
            # is_default set -- is a malformed extraction, counted as a miss
            # rather than silently repaired into something the model did not
            # say.
            continue
    return branches, 1
