"""The WP2 corpus target, as executable code, with progress reporting.

Agreed 2026-09-25. Written here rather than left in prose because a target that
lives only in a document is one that nobody checks against until the corpus is
finished, and a stratification plan discovered to be unmet at that point cannot
be fixed.

::

    natural cross-document        60   (20%)
    synthetic split              180   (60%)
    same-guide retrieval split    60   (20%)
                                 ---
                                 300

**Why 300 and not 150 Tier 1.** The original plan wanted 150 naturally
cross-document instances. The WP1 pilot found roughly 0.08 plausible Tier-1
pairs per curated pairing, so 150 would need on the order of 1,900 pairings --
effort that buys no proportional scientific return. Tier 1 is therefore
evidence for the natural setting, not the main corpus.

**Why 60 Tier 1 and not 20.** At 20, a reviewer can fairly say the
multi-document claim rests on a sample too small to say anything. 60 supports
independent reporting and a real error analysis. It does **not** support a
prevalence estimate, and the module refuses to compute one -- see
:func:`prevalence_estimate`.

**Why same-guide is its own row.** The pilot found 10 of 17 gov.uk pairings
were separate URLs within one underlying guide. Real separation at retrieval
time, no author intervention. Counting it as Tier 1 overstates the natural
claim; counting it as Tier 2 implies a split that never happened. It is stored
as ``construction=split`` so nothing downstream changes, and carries
``separation=same_guide`` so reporting can lift it back out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from contract.gold import Explicitness, GoldInstance
from contract.models import ConflictType, Construction, ScopeRelation, Separation

TARGET_TOTAL = 300

#: instances per reporting row
TARGETS: dict[Separation, int] = {
    Separation.CROSS_DOCUMENT: 60,
    Separation.SYNTHETIC_SPLIT: 180,
    Separation.SAME_GUIDE: 60,
}

ROW_NAMES: dict[Separation, str] = {
    Separation.CROSS_DOCUMENT: "natural cross-document",
    Separation.SYNTHETIC_SPLIT: "synthetic split",
    Separation.SAME_GUIDE: "same-guide retrieval split",
}

#: Minimum instances per cell within the synthetic-split row.
#:
#: Tier 2 is the row where the phenomenon can be controlled, so it is the row
#: that has to be stratified rather than filled to a count. Generating 180
#: instances that happen to be 170 refinements would leave the relation
#: boundaries -- which is what the paper reports -- untested.
#:
#: These are floors, not quotas: the remainder is free, and refinement will
#: dominate because it dominates reality.
TIER2_STRATA: dict[str, int] = {
    "relation:refinement": 60,
    "relation:redundant": 20,
    "relation:disjoint": 20,
    "relation:opposed": 20,
    "explicitness:implicit": 70,
    "explicitness:explicit": 30,
    "device:numeric_threshold": 15,
    "device:temporal_origination": 15,
    "device:product_or_account": 15,
    "device:eligibility": 15,
    "device:nested_exception": 10,
    "device:negative_condition": 10,
    "device:third_party": 10,
    "distractor": 40,
}

#: Below this, a row is too small for the descriptive and error analysis it is
#: meant to support.
MIN_REPORTABLE_ROW = 40


class PrevalenceClaimError(RuntimeError):
    """Raised on any attempt to turn the pilot yield into a prevalence figure."""


def prevalence_estimate(*_args, **_kwargs):
    """Refused, deliberately.

    The pilot sampled 38 curated pairings and found roughly 0.08 plausible
    Tier-1 pairs each. That is enough to decide Tier 1 cannot carry the corpus.
    It is not enough to say how common cross-document exceptions are in policy
    text, because the pairings were curated rather than sampled, drawn from two
    domains, and there are 38 of them.

    The distinction matters in one specific place -- the sentence written in the
    paper:

        ✅ "naturally occurring cross-document conditional pairs were scarce in
           the sampled sources, motivating a deliberately stratified corpus"
        ❌ "only 8% of policy document pairs contain cross-document exceptions"

    The second is a population claim this design cannot support, and it is the
    easy sentence to write by accident. Hence a function that exists only to
    refuse.
    """
    raise PrevalenceClaimError(
        "the WP1 pilot supports a go/no-go decision, not a prevalence estimate: "
        "38 curated (not sampled) pairings across two domains. Report the yield "
        "as motivation for the corpus design, never as a rate in the population."
    )


@dataclass
class CorpusProgress:
    """Where the corpus stands against the plan."""

    by_row: dict[str, int] = field(default_factory=dict)
    tier2_cells: dict[str, int] = field(default_factory=dict)
    total: int = 0
    unlabelled_separation: int = 0

    def shortfall(self) -> dict[str, int]:
        return {ROW_NAMES[s]: max(0, t - self.by_row.get(ROW_NAMES[s], 0))
                for s, t in TARGETS.items()}

    def empty_strata(self) -> list[str]:
        return [k for k, need in TIER2_STRATA.items()
                if self.tier2_cells.get(k, 0) < need]

    @property
    def complete(self) -> bool:
        return not any(self.shortfall().values()) and not self.empty_strata()

    def render(self) -> str:
        lines = [
            "WP2 corpus progress",
            "=" * 68,
            f"  {'row':<30} {'have':>6} {'target':>8} {'short':>7}",
            "  " + "-" * 54,
        ]
        for sep, target in TARGETS.items():
            name = ROW_NAMES[sep]
            have = self.by_row.get(name, 0)
            short = max(0, target - have)
            flag = "" if short == 0 else "  <-"
            lines.append(f"  {name:<30} {have:>6} {target:>8} {short:>7}{flag}")
        lines.append("  " + "-" * 54)
        lines.append(f"  {'total':<30} {self.total:>6} {TARGET_TOTAL:>8} "
                     f"{max(0, TARGET_TOTAL - self.total):>7}")

        if self.unlabelled_separation:
            lines += [
                "",
                f"  {self.unlabelled_separation} instances carry no `separation` and were",
                "  counted as synthetic split. That is the conservative fallback, but an",
                "  unlabelled Tier-1 instance is invisible in the row that matters most.",
            ]

        short_rows = [n for n, v in self.shortfall().items()
                      if v and self.by_row.get(n, 0) < MIN_REPORTABLE_ROW]
        if short_rows:
            lines += [
                "",
                f"  Below {MIN_REPORTABLE_ROW} instances these rows cannot carry the",
                "  independent reporting they exist for: " + ", ".join(short_rows),
            ]

        missing = self.empty_strata()
        lines += ["", "Tier-2 stratification", "-" * 68]
        if not missing:
            lines.append("  every stratum met")
        else:
            for k in missing:
                lines.append(f"  {k:<34} {self.tier2_cells.get(k, 0):>4} / "
                             f"{TIER2_STRATA[k]}")
            lines += [
                "",
                "  Tier 2 is the row where the phenomenon can be controlled, so it is",
                "  the one that has to be stratified rather than filled to a count.",
                "  Generating to 180 without these cells leaves the relation",
                "  boundaries -- what the paper reports -- untested.",
            ]
        return "\n".join(lines)


def _row_of(inst: GoldInstance) -> str:
    sep = inst.separation
    if sep is Separation.SAME_GUIDE:
        return ROW_NAMES[Separation.SAME_GUIDE]
    if sep is Separation.CROSS_DOCUMENT or inst.construction is Construction.NATURAL:
        return ROW_NAMES[Separation.CROSS_DOCUMENT]
    return ROW_NAMES[Separation.SYNTHETIC_SPLIT]


def measure(instances: list[GoldInstance]) -> CorpusProgress:
    """Count a corpus against the plan."""
    p = CorpusProgress(total=len(instances))
    rows: Counter = Counter()
    cells: Counter = Counter()

    for inst in instances:
        row = _row_of(inst)
        rows[row] += 1
        if inst.separation is None and inst.construction is not Construction.NATURAL:
            p.unlabelled_separation += 1

        if row != ROW_NAMES[Separation.SYNTHETIC_SPLIT]:
            continue

        if inst.is_distractor or inst.gold_conflict_type is not ConflictType.CONDITIONAL:
            cells["distractor"] += 1
        if inst.gold_scope_relation is not None:
            cells[f"relation:{inst.gold_scope_relation.value}"] += 1
        for b in inst.gold_branches:
            if not b.is_default:
                cells[f"explicitness:{b.explicitness.value}"] += 1
                break
        if inst.gold_multi_exception_flags.nested:
            cells["device:nested_exception"] += 1

    p.by_row = dict(rows)
    p.tier2_cells = dict(cells)
    return p


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="JSONL of gold instances; omit to print the plan alone")
    args = ap.parse_args(argv)

    if args.data is None:
        print("WP2 corpus plan (agreed 2026-09-25)")
        print("=" * 68)
        for sep, n in TARGETS.items():
            print(f"  {ROW_NAMES[sep]:<30} {n:>6}  ({n / TARGET_TOTAL:.0%})")
        print(f"  {'total':<30} {TARGET_TOTAL:>6}")
        print("\nTier-2 strata (floors, not quotas)")
        for k, n in TIER2_STRATA.items():
            print(f"  {k:<34} {n:>4}")
        return 0

    instances = []
    for line in args.data.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            instances.append(GoldInstance.model_validate(json.loads(line)))
    print(measure(instances).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
