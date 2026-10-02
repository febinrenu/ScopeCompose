"""Author `redundant` and `opposed` instances, which mining cannot produce.

Nine kappa pilots, five mining batches, 91 gov.uk pairings: every scope
relation recovered was `refinement` or `disjoint`. Not a small-sample
artifact — a property of published policy. `redundant` needs a narrower
restatement that *agrees* with the general rule, and `opposed` needs two
crossing scopes that *disagree*. Organisations publish neither on purpose: the
first is redundant to write, the second is a drafting error.

So the corpus plan's floor of 20 each can only be met by authoring, and this is
the tool for it. Same workflow as the WP1 probe set: a model proposes, a human
verifies case by case, and both facts travel with the instance.

**The generator is not trusted about the structure it claims.** A model asked
for a `redundant` case will cheerfully return a `refinement` — nested scopes
with *different* outcomes — and nothing downstream would notice until the
four-way confusion matrix came out wrong. Every candidate is therefore checked
mechanically before it is emitted, using the same `scope.attributes.compare`
and `metrics.branch_match.outcomes_match` the pipeline itself uses:

``redundant``
    scopes nest (SUBSET or EQUAL) **and** the outcomes match.
``opposed``
    scopes OVERLAP with neither containing the other **and** the outcomes
    differ.

A candidate failing its own definition is discarded, not relabelled to
whatever it happens to be. Relabelling would quietly fill the `refinement`
bucket — which is already full — while leaving the two that are empty empty.

Usage::

    python -m benchmark.mining.relation_author --relation redundant --n 20 \\
        -o benchmark/data/authored_redundant.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contract.gold import (
    Annotation,
    Applicability,
    AttributeKind,
    Branch,
    GoldInstance,
    ScopeAttribute,
)
from contract.models import (
    ConflictType,
    Construction,
    Domain,
    MultiExceptionFlags,
    Passage,
    ScopeRelation,
    Separation,
    SourceType,
)
from metrics.branch_match import outcomes_match
from scope.attributes import SetRelation, compare

_SHARED = """You write realistic UK policy passage pairs for a research benchmark.

Both passages must read like genuine published text -- a bank's terms, a
government guidance page -- not like an exercise. Use concrete products,
routes, amounts and dates.

Return JSON:
{"query": "the question a person would ask",
 "p0": {"text": "...", "scope": "who/what it covers",
        "attribute": "dimension_name", "value": "value", "outcome": "..."},
 "p1": {"text": "...", "scope": "...", "attribute": "...", "value": "...",
        "outcome": "..."}}

`attribute` is the dimension the scope varies on (card_tier, route, age_band,
account_type, residence). `value` is this passage's value on it."""

_REDUNDANT = _SHARED + """

STRUCTURE REQUIRED -- redundant:
- p1's scope must be a SUBSET of p0's. Same dimension, narrower value.
  (p0: all cardholders. p1: premium cardholders.)
- The OUTCOMES MUST BE THE SAME. p1 restates p0's outcome for a narrower
  group. It is NOT an exception -- nothing changes for anyone.
- Write p1 as a confirmation: "...in line with the standard tariff",
  "...as for other applicants".

The trap is writing p1 with a DIFFERENT outcome. That is a refinement, and
the benchmark already has plenty of those. Same outcome, narrower scope."""

_OPPOSED = _SHARED + """

STRUCTURE REQUIRED -- opposed:
- The scopes must CROSS: each covers cases the other does not, and some cases
  fall under both. Use TWO DIFFERENT dimensions.
  (p0: Reward account holders. p1: accounts opened from January 2024.
   A Reward account opened in 2024 is in both; a 2023 Reward account only in
   the first.)
- The OUTCOMES MUST CONTRADICT on the overlap, about the SAME predicate.
- Both passages must speak to one question. "Meets the skill requirement"
  against "does not meet the requirements of this route" are DIFFERENT
  predicates and do not conflict.

The trap is making one scope a subset of the other. That is refinement.
Different dimensions, genuinely crossing."""


def _attr(name: str, value: str) -> ScopeAttribute:
    return ScopeAttribute(name=name or "scope", kind=AttributeKind.CATEGORICAL,
                          values=[value or "unspecified"])


def _verify(relation: ScopeRelation, b0: Branch, b1: Branch) -> str | None:
    """``None`` if the pair really has the claimed structure, else the reason."""
    rel = compare(b1.applicability, b0.applicability)
    same_outcome = outcomes_match(b0.outcome, b1.outcome)

    if relation is ScopeRelation.REDUNDANT:
        if rel not in (SetRelation.SUBSET, SetRelation.EQUAL):
            return f"scopes are {rel.value}, not nested"
        if not same_outcome:
            return "outcomes differ -- this is a refinement, not a restatement"
        return None

    if relation is ScopeRelation.OPPOSED:
        if rel is not SetRelation.OVERLAPPING:
            return f"scopes are {rel.value}, not crossing"
        if same_outcome:
            return "outcomes agree -- nothing is opposed"
        return None

    return f"unsupported relation {relation.value}"


def author_one(relation: ScopeRelation, index: int, *, client=None
               ) -> tuple[GoldInstance | None, str]:
    """Return ``(instance, note)``. ``instance`` is None when verification fails."""
    from api_budget.client import Tier, get_client

    client = client or get_client()
    system = _REDUNDANT if relation is ScopeRelation.REDUNDANT else _OPPOSED
    domain = Domain.FINANCIAL_TERMS if index % 2 == 0 else Domain.IMMIGRATION_ELIGIBILITY
    topic = ("a UK bank current account or card" if index % 2 == 0
             else "a UK immigration route")

    try:
        result = client.complete(
            messages=[{"role": "user", "content":
                       f"Write one {relation.value} pair about {topic}. "
                       f"Make it different from any obvious first example."}],
            system=system,
            tier=Tier.JUDGE,
            step=f"author_{relation.value}",
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        raw = result.json() or {}
    except Exception as exc:
        return None, f"generation failed: {exc}"

    p0, p1 = raw.get("p0") or {}, raw.get("p1") or {}
    query = str(raw.get("query") or "").strip()
    if not (query and p0.get("text") and p1.get("text")):
        return None, "incomplete response"

    iid = f"auth_{relation.value}_{index:03d}"
    passages = [
        Passage(id="p0", text=str(p0["text"]).strip(),
                source_type=SourceType.OFFICIAL_POLICY, date="2024-01",
                document_id=f"{iid}_a"),
        Passage(id="p1", text=str(p1["text"]).strip(),
                source_type=SourceType.PRODUCT_TERMS, date="2024-02",
                document_id=f"{iid}_b"),
    ]

    # A redundant instance resolves to ONE merged branch -- the contract
    # enforces that, and it is the whole point of separating redundant from
    # refinement. An opposed instance has two conditioned branches and no
    # default, since neither scope covers the whole frame.
    # For `redundant`, p0 is the DEFAULT branch -- the unconditioned general
    # rule -- so p1 nests inside it by construction. Giving both a categorical
    # attribute instead makes them two single-valued sets, which `compare`
    # correctly calls DISJOINT ({all} and {premium} do not intersect), and no
    # redundant case could ever verify.
    #
    # For `opposed` neither branch is the default: each covers cases the other
    # does not, which is exactly what makes the scopes cross.
    general_is_default = relation is ScopeRelation.REDUNDANT
    b0 = Branch(
        branch_id="b0",
        condition=(None if general_is_default
                   else f"the case is {p0.get('scope') or 'in the general population'}"),
        outcome=str(p0.get("outcome") or "").strip() or "the general outcome applies",
        applicability=Applicability(
            descriptor=("all cases" if general_is_default
                        else str(p0.get("scope") or "general cases")),
            is_default=general_is_default,
            attributes=([] if general_is_default else
                        [_attr(str(p0.get("attribute") or ""),
                               str(p0.get("value") or ""))])),
        supporting_passage="p0")
    b1 = Branch(
        branch_id="b1",
        condition=f"the case is {p1.get('scope') or 'in the narrower population'}",
        outcome=str(p1.get("outcome") or "").strip() or "the narrower outcome applies",
        applicability=Applicability(
            descriptor=str(p1.get("scope") or "narrower cases"),
            attributes=[_attr(str(p1.get("attribute") or ""), str(p1.get("value") or ""))]),
        supporting_passage="p1")

    problem = _verify(relation, b0, b1)
    if problem:
        return None, problem

    branches = [b0] if relation is ScopeRelation.REDUNDANT else [b0, b1]
    try:
        inst = GoldInstance(
            instance_id=iid, query=query, domain=domain,
            construction=Construction.SPLIT,
            separation=Separation.SYNTHETIC_SPLIT,
            split="train", passages=passages,
            gold_conflict_type=ConflictType.CONDITIONAL,
            gold_scope_relation=relation,
            gold_branches=branches,
            gold_multi_exception_flags=MultiExceptionFlags(),
            annotation=Annotation(
                annotator_a="model-proposed",
                # Left empty on purpose. These are authored to exhibit a
                # relation, which is labelling by construction, and the label
                # means nothing until a human has read the passages and agreed.
                annotator_b=None,
                notes=f"AUTHORED for {relation.value}. Verify the structure "
                      "before use: the generator is checked mechanically but "
                      "not for whether the text reads as real policy."),
        )
    except Exception as exc:
        return None, f"schema rejected: {exc}"
    return inst, "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relation", choices=["redundant", "opposed"], required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--attempts", type=int, default=0,
                    help="cap generation attempts (default 3x --n)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args(argv)

    relation = ScopeRelation(args.relation)
    budget = args.attempts or args.n * 3
    kept: list[GoldInstance] = []
    rejected: list[str] = []

    for i in range(budget):
        if len(kept) >= args.n:
            break
        # Indexed by attempt, not by how many were kept, so the prompt varies
        # even when a run rejects several in a row -- otherwise the generator
        # gets the same request repeatedly and returns the same case.
        inst, note = author_one(relation, i)
        if inst is None:
            rejected.append(note)
            continue
        kept.append(inst)
        print(f"  [{len(kept)}/{args.n}] {inst.instance_id}")

    if not kept:
        print("nothing survived verification", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for inst in kept:
            fh.write(inst.model_dump_json() + "\n")

    print(f"\nwrote {len(kept)} {relation.value} instances -> {args.output}")
    print(f"  rejected by structural verification: {len(rejected)}")
    if rejected:
        from collections import Counter
        for reason, n in Counter(rejected).most_common(4):
            print(f"    {n:>3}  {reason}")
        print()
        print("  Rejected candidates are discarded, never relabelled to whatever")
        print("  they turned out to be -- that would fill the refinement bucket,")
        print("  which is already full, and leave these two empty.")
    print()
    print("  MODEL-PROPOSED, NOT VERIFIED. The structure is checked mechanically;")
    print("  whether the text reads as real policy is not. Review each case the")
    print("  way the WP1 probe set was reviewed before using any of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
