"""Strata S1-S5 and the stratified, leak-free re-split (proposal v8 §3.1, §6.2).

Each instance belongs to exactly one stratum, and which claims a stratum may
support is fixed by the proposal, not chosen after seeing results:

======  =======================================  =====================================
S1      natural cross-document conditional       **primary**: every cross-document
        conflict                                 claim rests on this alone
S2      natural same-guide conditional conflict  secondary, reported separately
S3      real-text split (a real rule/exception   mechanism diagnostic ONLY; never
        deliberately split across two passages)  substitutes for S1 in a claim
S4      contrast set (authored redundant /       stress test on the rare relations
        crossing)
S5      LLM-authored probes                      development only, never evidence
CONTROL anything that is not a conditional       precision / fabrication denominators
        conflict (no_conflict, disjoint,
        redundant, factual, temporal, opinion)
======  =======================================  =====================================

**S1 minimum.** At least ``S1_TARGET`` S1 conditional instances in test; the
floor is ``S1_FLOOR``. Below the floor every cross-document hypothesis is
reported as exploratory and the title and abstract narrow (the *narrowing
rule*). It is computed here, from the data, so the paper cannot state it by
hand.

**Re-split.** The old split was assigned before any detector existed and no
model has been tuned on it, so it may be redrawn; the old assignment is
archived, never overwritten. The assignment is still by connected document
group (a document in two splits is leakage), but it is now *stratum-aware*:
S1 groups are placed into test and dev only (QNFA is prompt-and-schema based,
so there is nothing for a train split to train), and S5 groups are kept out of
test wherever a document group allows it.

Usage::

    python -m benchmark.strata --data benchmark/data/corpus.jsonl
    python -m benchmark.strata --data benchmark/data/corpus.jsonl --resplit
    python -m benchmark.strata --data benchmark/data/corpus.jsonl --resplit --apply
    python -m benchmark.strata --data benchmark/data/corpus.jsonl --resplit --incremental --apply
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from benchmark.relabel_v7 import V7Label, relabel
from benchmark.splits import Group, audit, build_groups
from contract.gold import GoldInstance
from contract.models import Separation, SourceType

S1_TARGET = 80
S1_FLOOR = 60

#: Placeholder host used by every LLM-authored probe passage.
LLM_AUTHORED_HOST = "example.invalid"


class Stratum(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    CONTROL = "control"


def is_llm_authored(inst: GoldInstance) -> bool:
    """True if any passage is model-written text rather than harvested text."""
    for p in inst.passages:
        if p.source_type is SourceType.SYNTHETIC:
            return True
        if p.source_url and LLM_AUTHORED_HOST in p.source_url:
            return True
    return False


def stratum_of(inst: GoldInstance, *, contrast_ids: frozenset[str] = frozenset()) -> Stratum:
    """Assign one instance to one stratum. Order matters and is deliberate:

    1. LLM-authored text is S5 whatever its label -- authoring is the property
       that disqualifies it as evidence, so it is checked first;
    2. human-verified contrast items are S4;
    3. a conditional *conflict* is S1 / S2 / S3 by how its passages were
       separated;
    4. everything else is a control.
    """
    if is_llm_authored(inst):
        return Stratum.S5
    if inst.instance_id in contrast_ids:
        return Stratum.S4
    label = relabel(inst).label
    if label is V7Label.CONDITIONAL_CONFLICT:
        sep = inst.separation
        if sep is Separation.CROSS_DOCUMENT:
            return Stratum.S1
        if sep is Separation.SAME_GUIDE:
            return Stratum.S2
        if sep is Separation.SYNTHETIC_SPLIT:
            return Stratum.S3
        raise ValueError(
            f"{inst.instance_id}: conditional conflict with unrecorded separation; "
            "its stratum cannot be derived and must not be guessed"
        )
    if label is V7Label.CROSSING:
        return Stratum.S4
    return Stratum.CONTROL


# --------------------------------------------------------------------------- #
# Stratified, document-grouped assignment
# --------------------------------------------------------------------------- #

#: Split shares per group class. S1 never goes to train; S5 never goes to test.
CLASS_RATIOS: dict[str, dict[str, float]] = {
    "S1": {"test": 0.70, "dev": 0.30},
    "S2": {"test": 0.40, "dev": 0.20, "train": 0.40},
    "S3": {"test": 0.50, "dev": 0.20, "train": 0.30},
    "S4": {"test": 0.50, "dev": 0.25, "train": 0.25},
    "control": {"test": 0.40, "dev": 0.25, "train": 0.35},
    "S5": {"dev": 0.35, "train": 0.65},
}

#: Which class a mixed group is assigned by (a group is indivisible, so it
#: follows the most protected stratum it contains).
_PRIORITY = ["S1", "S2", "S3", "S4", "control", "S5"]


def _group_class(group: Group, strata: dict[str, Stratum]) -> str:
    present = {strata[i].value for i in group.instance_ids}
    for cls in _PRIORITY:
        if cls in present:
            return cls
    raise AssertionError("empty group")


def assign_stratified(
    instances: list[GoldInstance],
    *,
    seed: int = 0,
    contrast_ids: frozenset[str] = frozenset(),
    keep_existing: bool = False,
) -> dict[str, str]:
    """``instance_id -> split``, by document group, stratum-aware.

    Within each class, groups go largest-first into whichever allowed split is
    furthest below its quota -- the same rule as ``benchmark.splits.assign``,
    run once per class so every stratum is spread across splits instead of the
    big groups of one stratum crowding a single split.

    ``keep_existing=True`` is the incremental mode used while mining continues:
    a document group whose instances already carry a split keeps it, and only
    groups with no assigned member are placed -- counted against the quota the
    kept groups already fill. Development against dev is then stable while new
    S1 candidates keep arriving. A group whose members disagree on a split is a
    leak that already exists, and is refused rather than papered over.
    """
    strata = {i.instance_id: stratum_of(i, contrast_ids=contrast_ids) for i in instances}
    groups = build_groups(instances)
    rng = random.Random(seed)
    rng.shuffle(groups)

    by_class: dict[str, list[Group]] = defaultdict(list)
    for g in groups:
        by_class[_group_class(g, strata)].append(g)

    current = {i.instance_id: i.split for i in instances}
    out: dict[str, str] = {}
    for cls, gs in by_class.items():
        ratios = CLASS_RATIOS[cls]
        gs.sort(key=lambda g: -g.size)
        total = sum(g.size for g in gs)
        target = {k: v * total for k, v in ratios.items()}
        have = {k: 0 for k in ratios}

        pending: list[Group] = []
        for g in gs:
            if keep_existing:
                kept = {current[i] for i in g.instance_ids}
                if len(kept) > 1:
                    raise ValueError(
                        f"document group {sorted(g.document_ids)[:2]}... already spans splits "
                        f"{sorted(kept)}: that is existing leakage, fix it before an incremental run"
                    )
                (split,) = kept
                if split in have:
                    have[split] += g.size
                    for iid in g.instance_ids:
                        out[iid] = split
                    continue
            pending.append(g)

        for g in pending:
            split = max(ratios, key=lambda k: target[k] - have[k])
            have[split] += g.size
            for iid in g.instance_ids:
                out[iid] = split
    return out


# --------------------------------------------------------------------------- #
# Quota report and the narrowing rule
# --------------------------------------------------------------------------- #


@dataclass
class Quota:
    table: dict[str, Counter] = field(default_factory=dict)
    s5_in_test: list[str] = field(default_factory=list)
    s1_test: int = 0

    @property
    def narrowing_rule_applies(self) -> bool:
        return self.s1_test < S1_FLOOR

    @property
    def status(self) -> str:
        if self.s1_test >= S1_TARGET:
            return "target met"
        if self.s1_test >= S1_FLOOR:
            return "floor met, target not met"
        return "BELOW FLOOR: narrowing rule applies"

    def render(self) -> str:
        lines = ["Strata x split", "=" * 68,
                 f"  {'stratum':<10}{'train':>8}{'dev':>8}{'test':>8}{'total':>9}", "  " + "-" * 41]
        for s in Stratum:
            c = self.table.get(s.value, Counter())
            lines.append(f"  {s.value:<10}{c['train']:>8}{c['dev']:>8}{c['test']:>8}{sum(c.values()):>9}")
        lines += [
            "",
            f"  S1 conditional conflicts in test: {self.s1_test}  (target {S1_TARGET}, floor {S1_FLOOR})",
            f"  status: {self.status}",
        ]
        if self.narrowing_rule_applies:
            lines += [
                "",
                "  Narrowing rule (proposal section 3.1): every cross-document hypothesis (H1-H3)",
                "  is reported as EXPLORATORY and the title/abstract claim narrows to",
                "  'exception suppression in RAG, with exploratory natural cross-document",
                "  evidence'. S3 never substitutes for S1.",
            ]
        if self.s5_in_test:
            lines += ["", f"  WARNING: {len(self.s5_in_test)} S5 probe(s) share a document group with a "
                          "protected stratum and landed in test; they are excluded from every test "
                          f"population by stratum: {self.s5_in_test[:4]}"]
        return "\n".join(lines)


def quota_report(instances: list[GoldInstance], *, contrast_ids: frozenset[str] = frozenset()) -> Quota:
    q = Quota()
    table: dict[str, Counter] = defaultdict(Counter)
    for inst in instances:
        s = stratum_of(inst, contrast_ids=contrast_ids)
        table[s.value][inst.split] += 1
        if s is Stratum.S5 and inst.split == "test":
            q.s5_in_test.append(inst.instance_id)
    q.table = dict(table)
    q.s1_test = table[Stratum.S1.value]["test"]
    return q


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load(path: Path) -> list[GoldInstance]:
    return [GoldInstance.model_validate(json.loads(l))
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _contrast_ids(path: Path | None) -> frozenset[str]:
    if path is None or not path.exists():
        return frozenset()
    return frozenset(
        json.loads(l)["instance_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--contrast", type=Path, default=None,
                    help="jsonl of human-verified contrast instances (stratum S4)")
    ap.add_argument("--resplit", action="store_true", help="compute a stratified assignment and report it")
    ap.add_argument("--apply", action="store_true",
                    help="with --resplit: archive the corpus as <name>.pre_v7_split and rewrite splits")
    ap.add_argument("--incremental", action="store_true",
                    help="with --resplit: keep every existing assignment, place only new document groups")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    contrast = _contrast_ids(args.contrast)
    instances = _load(args.data)

    if args.resplit:
        mapping = assign_stratified(instances, seed=args.seed, contrast_ids=contrast,
                                    keep_existing=args.incremental)
        new = [i.model_copy(update={"split": mapping[i.instance_id]}) for i in instances]
        print("BEFORE\n" + quota_report(instances, contrast_ids=contrast).render() + "\n")
        print("AFTER (stratified re-split)\n" + quota_report(new, contrast_ids=contrast).render() + "\n")
        print(audit(new).render())
        if args.apply:
            archive = args.data.with_suffix(".pre_v7_split.jsonl")
            if archive.exists():
                print(f"\nrefusing to overwrite the existing archive {archive}")
                return 1
            shutil.copy2(args.data, archive)
            with args.data.open("w", encoding="utf-8", newline="\n") as fh:
                for i in new:
                    fh.write(i.model_dump_json() + "\n")
            print(f"\narchived the old assignment -> {archive}\nrewrote splits -> {args.data}")
        else:
            print("\n(dry run: pass --apply to archive the old split and rewrite the corpus)")
        return 0

    print(quota_report(instances, contrast_ids=contrast).render())
    print()
    print(audit(instances).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
