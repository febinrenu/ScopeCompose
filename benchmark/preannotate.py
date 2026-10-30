"""Pre-annotate gold branches with the evaluation model, for a human to correct (plan v4 B-8).

Writing a branch structure from scratch costs minutes per instance; correcting a good
proposal costs seconds, and annotation hours are the scarcest budget in the project. So
the **EVAL** model proposes and a person verifies (``benchmark.review --kind branches``).

Three rules keep this from contaminating the benchmark:

* **EVAL only.** The proposer is an evaluation-only model with *its own* prompt. The
  evaluated systems' prompts (B7c's parse, QNFA's frame extraction) are never used here,
  so a system cannot be scored against branches its own prompt wrote. ``propose`` raises
  for any other tier.
* **Proposals are marked as proposals.** ``annotation.notes`` carries ``branches=model-proposed``
  until a person's edit changes it to ``branches=reviewed:<name>``, so the datasheet can split
  model-proposed-and-checked from independently double-annotated, and never pools them.
  ``annotator_a`` / ``annotator_b`` are left alone: they record who labelled the conflict *type
  and relation*, a different axis, and overwriting them would destroy that provenance.
* **The reference answers are derived, not written.** The gold scoped answer is rendered
  from the *verified* branches by the same verbaliser every structured system uses, and the
  selection answer is just the default branch. They are therefore labelled reference
  answers, not "human-written". If a person corrects a branch the answers are re-derived,
  so they can never disagree with the branches they summarise.

Only an instance a person has adjudicated as a ``refinement`` is eligible: branches for
anything else would be inventing structure the instance does not have.
"""

from __future__ import annotations

import json
from copy import deepcopy

from api_budget import Tier, get_client
from baselines.registry import parse_branches, verbalise_branches
from contract.gold import GoldInstance
from contract.models import ConflictType, ScopeRelation

STEP = "preannotate_branches"
MODEL_PROPOSED = "model-proposed"

SYSTEM = """\
You annotate a benchmark. Given a question and two passages, write down the outcomes that apply to
the question and exactly who each applies to. One passage states a general rule; the other states a
narrower case where the outcome differs.

Return ONLY JSON:
{"branches": [
  {"is_default": true, "condition": null, "applies_to": "everyone else", "outcome": "<short phrase>",
   "source": "<passage id>"},
  {"is_default": false, "condition": "<the condition under which the exception holds>",
   "applies_to": "<short phrase naming the group>", "outcome": "<short phrase>", "source": "<passage id>"}
]}
Exactly one default branch and at least one exception. Outcomes are short phrases such as
"a 3% fee applies" or "no fee applies"; use the figures the passages state. Cite the passage each
branch comes from. Do not invent a condition the passages do not state."""


def _check_eligible(inst: GoldInstance) -> None:
    if inst.gold_conflict_type is not ConflictType.CONDITIONAL or inst.gold_scope_relation is not ScopeRelation.REFINEMENT:
        raise ValueError(f"{inst.instance_id}: only an adjudicated refinement can be pre-annotated with branches")
    if len(inst.passages) < 2:
        raise ValueError(f"{inst.instance_id}: needs at least two passages")


def derive_answers(branches) -> tuple[str, str]:
    """(gold scoped answer, selection answer), derived from the branches."""
    default = next(b for b in branches if b.is_default)
    selection = default.outcome.strip().rstrip(".")
    return verbalise_branches(branches), (selection[0].upper() + selection[1:] + ".") if selection else ""


PROVENANCE_KEY = "branches="


def set_branch_provenance(notes: str | None, value: str) -> str:
    """Replace any earlier ``branches=...`` token in the notes, keeping everything else."""
    kept = [t for t in (notes or "").split(" | ") if t and not t.startswith(PROVENANCE_KEY)]
    return " | ".join([*kept, f"{PROVENANCE_KEY}{value}"])


def branch_provenance(inst: GoldInstance) -> str | None:
    for t in (inst.annotation.notes or "").split(" | "):
        if t.startswith(PROVENANCE_KEY):
            return t[len(PROVENANCE_KEY):]
    return None


def with_branches(inst: GoldInstance, branches) -> GoldInstance:
    """Attach branches and the answers derived from them, marked as a model proposal."""
    scoped, selection = derive_answers(branches)
    d = inst.model_dump(mode="json")
    d["gold_branches"] = [b.model_dump(mode="json") for b in branches]
    d["gold_scoped_answer"] = scoped
    d["selection_answer"] = selection
    ann = d.get("annotation") or {}
    ann["notes"] = set_branch_provenance(ann.get("notes"), MODEL_PROPOSED)
    d["annotation"] = ann
    return GoldInstance.model_validate(d)


def propose(inst: GoldInstance, *, client=None, tier: Tier = Tier.EVAL) -> GoldInstance:
    """Ask the EVAL model for a branch structure. Raises if it does not yield a valid one."""
    if tier is not Tier.EVAL:
        raise ValueError("branch pre-annotation can reach the test set and must use the EVAL tier")
    _check_eligible(inst)
    body = "\n\n".join(f"[{p.id}] {p.text}" for p in inst.passages)
    c = client or get_client()
    res = c.complete(messages=[{"role": "user", "content": f"Question: {inst.query}\n\nPassages:\n{body}"}],
                     system=SYSTEM, tier=tier, step=STEP, response_format={"type": "json_object"})
    try:
        data = json.loads(res.text)
    except (TypeError, ValueError):
        from detection.stage2 import _salvage_json

        data = _salvage_json(res.text)
    branches = parse_branches(data, {p.id for p in inst.passages})
    defaults = [b for b in branches if b.is_default]
    if len(defaults) != 1 or len(branches) < 2:
        raise ValueError(f"{inst.instance_id}: the proposal needs one default and at least one exception "
                         f"(got {len(defaults)} default(s), {len(branches)} branch(es))")
    return with_branches(inst, branches)


def apply_edits(inst: GoldInstance, edits: dict, *, annotator: str) -> GoldInstance:
    """Apply a reviewer's corrections (``{branch_id: {outcome?, condition?}}``) and re-derive the answers.

    The branch provenance changes from ``model-proposed`` to ``reviewed:<annotator>``: a person has now
    checked it. The type/relation annotators are untouched. A condition edited away on an exception branch is refused (it would turn it into a
    second default)."""
    if not annotator.strip() or annotator.strip().lower() in {MODEL_PROPOSED, "model_proposed", "model"}:
        raise ValueError("a reviewed instance must carry a human annotator id")
    d = inst.model_dump(mode="json")
    by_id = {b["branch_id"]: b for b in d["gold_branches"]}
    for bid, change in edits.items():
        if bid not in by_id:
            raise KeyError(f"{inst.instance_id}: no branch {bid!r}")
        b = by_id[bid]
        if "outcome" in change:
            if not str(change["outcome"]).strip():
                raise ValueError(f"{bid}: an empty outcome")
            b["outcome"] = str(change["outcome"]).strip()
        if "condition" in change:
            cond = change["condition"]
            if b["condition"] is not None and not cond:
                raise ValueError(f"{bid}: an exception cannot lose its condition")
            if b["condition"] is not None:
                b["condition"] = str(cond).strip()
                b["applicability"]["descriptor"] = b["condition"]
    branches = [type(inst.gold_branches[0]).model_validate(b) for b in d["gold_branches"]]
    out = with_branches(inst.model_copy(update={"gold_branches": []}), branches).model_dump(mode="json")
    ann = deepcopy(out.get("annotation") or {})
    ann["notes"] = set_branch_provenance(ann.get("notes"), f"reviewed:{annotator.strip()}")
    out["annotation"] = ann
    return GoldInstance.model_validate(out)
