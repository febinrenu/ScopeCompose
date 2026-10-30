"""Relabel the pilot corpus under the v8 definitions (proposal §4.2).

The pilot labels were written against the four-way relation of proposal v6. v8
changes what counts as a *conflict*, and nothing about the gold annotations
should be silently rewritten to follow -- the two-annotator kappa pilots were
run on the old labels. So this module **derives** the v7 label from the frozen
gold and writes a sidecar; ``corpus.jsonl`` is never touched.

The mapping:

====================================  ====================================
pilot label                           v7 label
====================================  ====================================
conditional / refinement              ``conditional_conflict``
conditional / disjoint                ``no_conflict``  (two true statements about
                                      different people; nothing is in conflict)
conditional / redundant               ``no_conflict``  (they agree; merging is
                                      right, branching would fabricate an
                                      exception)
conditional / opposed                 ``crossing`` | ``same_scope`` |
                                      ``opposed_undecided``  -- see below
no_conflict / factual / temporal /    unchanged
opinion
====================================  ====================================

**Why ``opposed`` is split.** An ``opposed`` pair whose scopes *cross* is not
default-plus-exception: reporting both non-overlapping parts as stated and the
overlap as contested is the right treatment, and counting it as a composition
success would be wrong. An ``opposed`` pair with *equal* scopes is the one case
where selection is appropriate. A pair we cannot place stays undecided and the
operator treats it as contested -- never selected silently.

A **conditional conflict** is therefore exactly one thing: a refinement.

Usage::

    python -m benchmark.relabel_v7 --data benchmark/data/corpus.jsonl
    python -m benchmark.relabel_v7 --data benchmark/data/corpus.jsonl --write
    python -m benchmark.relabel_v7 --data benchmark/data/corpus.jsonl --expect-conflicts 18
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from contract.gold import GoldInstance
from contract.models import ConflictType, Construction, Opposition, ScopeRelation, Separation

DEFAULT_SIDECAR = Path(__file__).parent / "data" / "relabel_v7.jsonl"


class V7Label(str, Enum):
    CONDITIONAL_CONFLICT = "conditional_conflict"
    NO_CONFLICT = "no_conflict"
    CROSSING = "crossing"
    SAME_SCOPE = "same_scope"
    OPPOSED_UNDECIDED = "opposed_undecided"
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    OPINION = "opinion"


@dataclass(frozen=True)
class Relabel:
    instance_id: str
    label: V7Label
    opposition: Opposition | None
    relabelled_from: str
    """``"<conflict_type>/<scope_relation>"`` of the frozen gold, for audit."""


def _scope_signature(inst: GoldInstance, branch_id: str) -> tuple:
    """A comparable fingerprint of one branch's applicability."""
    b = next(b for b in inst.gold_branches if b.branch_id == branch_id)
    if b.applicability.attributes:
        return tuple(sorted(
            (a.name, a.kind.value, tuple(sorted(a.values or [])), a.min_value, a.max_value, a.boolean_value)
            for a in b.applicability.attributes
        ))
    return (" ".join(b.applicability.descriptor.casefold().split()),)


def infer_opposition(inst: GoldInstance) -> Opposition | None:
    """Place an ``opposed`` instance as crossing / same-scope / undecided.

    An explicit v1.3.0 ``gold_opposition`` always wins. Otherwise the rule
    reads the frozen branch structure, and refuses to guess:

    * two or more conditioned branches and **no default** -- two scopes, neither
      of which is the whole case space, outcomes disagreeing: ``CROSSING``,
      unless their applicability fingerprints are identical (``SAME_SCOPE``);
    * anything else (a default branch present, or no branches recorded):
      ``UNKNOWN``. A default branch makes the other branch a subset of it,
      which is the definition of refinement, so an ``opposed`` label there
      contradicts the structure and must be looked at by a person.
    """
    if inst.gold_scope_relation is not ScopeRelation.OPPOSED:
        return None
    if inst.gold_opposition is not None:
        return inst.gold_opposition

    conditioned = [b for b in inst.gold_branches if not b.is_default]
    if inst.default_branch is not None or len(conditioned) < 2:
        return Opposition.UNKNOWN
    sigs = {_scope_signature(inst, b.branch_id) for b in conditioned}
    return Opposition.SAME_SCOPE if len(sigs) == 1 else Opposition.CROSSING


def relabel(inst: GoldInstance) -> Relabel:
    ctype = inst.gold_conflict_type
    rel = inst.gold_scope_relation
    origin = f"{ctype.value}/{rel.value if rel else '-'}"

    if ctype is ConflictType.CONDITIONAL:
        if rel is ScopeRelation.REFINEMENT:
            return Relabel(inst.instance_id, V7Label.CONDITIONAL_CONFLICT, None, origin)
        if rel in (ScopeRelation.DISJOINT, ScopeRelation.REDUNDANT):
            return Relabel(inst.instance_id, V7Label.NO_CONFLICT, None, origin)
        if rel is ScopeRelation.OPPOSED:
            opp = infer_opposition(inst)
            label = {
                Opposition.CROSSING: V7Label.CROSSING,
                Opposition.SAME_SCOPE: V7Label.SAME_SCOPE,
                Opposition.UNKNOWN: V7Label.OPPOSED_UNDECIDED,
            }[opp]
            return Relabel(inst.instance_id, label, opp, origin)
        raise ValueError(f"{inst.instance_id}: conditional instance without a relation")

    simple = {
        ConflictType.NO_CONFLICT: V7Label.NO_CONFLICT,
        ConflictType.FACTUAL: V7Label.FACTUAL,
        ConflictType.TEMPORAL: V7Label.TEMPORAL,
        ConflictType.OPINION: V7Label.OPINION,
    }
    return Relabel(inst.instance_id, simple[ctype], None, origin)


def relabel_all(instances: list[GoldInstance]) -> list[Relabel]:
    return [relabel(i) for i in instances]


def natural_conditional_conflicts(instances: list[GoldInstance]) -> Counter:
    """Natural conditional conflicts by separation: the paper's subject.

    ``cross_document`` is stratum S1 (the primary population) and
    ``same_guide`` is S2. Synthetic splits are not natural and are not counted.
    """
    out: Counter = Counter()
    for inst in instances:
        r = relabel(inst)
        if r.label is V7Label.CONDITIONAL_CONFLICT and inst.construction is Construction.NATURAL:
            out[inst.separation.value if inst.separation else "unrecorded"] += 1
        elif (
            r.label is V7Label.CONDITIONAL_CONFLICT
            and inst.separation is Separation.SAME_GUIDE
        ):
            # same_guide is stored as construction='split' (Tier 2) but is real
            # text from a real guide, so it belongs in the natural count (S2).
            out[Separation.SAME_GUIDE.value] += 1
    return out


def _load(path: Path) -> list[GoldInstance]:
    return [GoldInstance.model_validate(json.loads(l))
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def render(instances: list[GoldInstance]) -> str:
    rl = relabel_all(instances)
    by_label = Counter(r.label.value for r in rl)
    moved = Counter((r.relabelled_from, r.label.value) for r in rl
                    if r.relabelled_from.split("/")[0] == "conditional")
    nat = natural_conditional_conflicts(instances)

    lines = ["v7 relabel of the pilot corpus", "=" * 68, f"  instances: {len(instances)}", ""]
    lines.append("  v7 label                       n")
    lines.append("  " + "-" * 38)
    for label in V7Label:
        lines.append(f"  {label.value:<26}{by_label.get(label.value, 0):>8}")
    lines += ["", "  conditional-family mapping", "  " + "-" * 38]
    for (src, dst), n in sorted(moved.items()):
        lines.append(f"  {src:<26} -> {dst:<20}{n:>3}")
    lines += [
        "",
        "  natural conditional conflicts (refinement only)",
        "  " + "-" * 38,
        f"  S1 cross_document     {nat.get('cross_document', 0):>4}",
        f"  S2 same_guide         {nat.get('same_guide', 0):>4}",
        f"  total                 {sum(nat.values()):>4}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--write", nargs="?", type=Path, const=DEFAULT_SIDECAR, default=None,
                    help=f"write the sidecar (default {DEFAULT_SIDECAR.name}); corpus is never modified")
    ap.add_argument("--expect-conflicts", type=int, default=None,
                    help="exit non-zero unless the natural conditional-conflict total equals this")
    args = ap.parse_args(argv)

    instances = _load(args.data)
    print(render(instances))

    if args.write is not None:
        with args.write.open("w", encoding="utf-8", newline="\n") as fh:
            for r in relabel_all(instances):
                d = asdict(r)
                d["label"] = r.label.value
                d["opposition"] = r.opposition.value if r.opposition else None
                fh.write(json.dumps(d, sort_keys=True) + "\n")
        print(f"\nwrote sidecar -> {args.write}")

    if args.expect_conflicts is not None:
        got = sum(natural_conditional_conflicts(instances).values())
        if got != args.expect_conflicts:
            print(f"\nFAIL: expected {args.expect_conflicts} natural conditional conflicts, got {got}")
            return 1
        print(f"\nok: {got} natural conditional conflicts, as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
