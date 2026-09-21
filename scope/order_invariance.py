"""The passage-order permutation test.

A design property of A4, verified rather than asserted: role assignment and
every downstream decision depend on the applicability-and-outcome relation
between two claims, never on the order retrieval happened to return them in.

Why it matters beyond tidiness. If the answer flips when passages are
reshuffled, then two runs of the same system on the same evidence give
different answers, and every Preservation Rate number in the paper is a
function of retrieval order rather than of the method. That would not be a
small bug -- it would invalidate the main result. So this is a correctness
gate, not a nice-to-have, and it is expected to pass at 100%: anything less is
a defect in ``relation.py``, not a tolerance to be tuned.

Usage::

    python -m scope.order_invariance --data mock.jsonl --sample 50 --heuristic-nli
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from contract.models import QueryRecord
from scope.pipeline import ScopePipeline


@dataclass
class Violation:
    """One instance whose analysis changed under reordering."""

    query_id: str
    pair: tuple[str, str]
    field: str
    baseline: str
    permuted: str
    permutation: list[str]

    def render(self) -> str:
        return (
            f"  {self.query_id} pair {self.pair}: {self.field} "
            f"{self.baseline!r} -> {self.permuted!r} under order {self.permutation}"
        )


@dataclass
class InvarianceReport:
    records_tested: int = 0
    permutations_tested: int = 0
    pairs_compared: int = 0
    violations: list[Violation] = field(default_factory=list)
    non_positional_roles: int = 0
    """How often the default branch was NOT the first passage in the permuted
    order.

    This is the evidence that the check is doing real work rather than passing
    vacuously. A system that simply called the first-listed passage the default
    would also score 100% invariance on the identity permutation, but it would
    score zero here. A high count means role assignment genuinely tracked
    applicability against the positional grain.
    """

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def pass_rate(self) -> float:
        if not self.pairs_compared:
            return 1.0
        return 1.0 - len(self.violations) / self.pairs_compared

    def render(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [
            "Order-invariance check (A4)",
            "=" * 72,
            f"  result               {head}",
            f"  records tested       {self.records_tested:,}",
            f"  permutations tested  {self.permutations_tested:,}",
            f"  pair decisions       {self.pairs_compared:,}",
            f"  pass rate            {self.pass_rate:.4f}",
            f"  non-positional roles {self.non_positional_roles:,}",
        ]
        if self.violations:
            lines += [
                "",
                f"{len(self.violations)} VIOLATION(S) -- these must be fixed, not tolerated.",
                "A decision that depends on retrieval order makes every downstream metric",
                "a function of retrieval order. Look in scope/relation.py: role assignment",
                "must derive from the applicability relation alone.",
                "",
            ]
            for v in self.violations[:25]:
                lines.append(v.render())
            if len(self.violations) > 25:
                lines.append(f"  ... and {len(self.violations) - 25} more")
        elif self.pairs_compared:
            lines += [
                "",
                "  Every scope relation, branch role and confidence was identical under",
                "  every permutation tested. Role assignment is a function of the",
                "  applicability relation, as designed.",
            ]
        return "\n".join(lines)


def permute_record(record: QueryRecord, order: list[str]) -> QueryRecord:
    """Rebuild a record with its passages in a given order.

    Passage IDENTITIES are preserved -- only their position in the list
    changes. Renumbering them would make the comparison meaningless, because
    the pair keys would differ for reasons unrelated to ordering.
    """
    by_id = {p.id: p for p in record.passages}
    return record.model_copy(update={"passages": [by_id[pid] for pid in order]})


def check_record(
    record: QueryRecord,
    pipeline: ScopePipeline,
    *,
    max_permutations: int = 6,
    rng: random.Random | None = None,
) -> tuple[list[Violation], int, int, int]:
    """Analyse a record under several passage orderings and compare.

    Returns (violations, permutations_tested, pairs_compared, non_positional_roles).
    """
    rng = rng or random.Random(0)
    ids = [p.id for p in record.passages]
    if len(ids) < 2:
        return [], 0, 0, 0

    _, _, baseline = pipeline.analyse_record(record)
    if not baseline:
        return [], 0, 0, 0

    all_orders = list(itertools.permutations(ids))
    candidates = [list(o) for o in all_orders if list(o) != ids]
    if len(candidates) > max_permutations:
        candidates = rng.sample(candidates, max_permutations)
    # Reversal is the adversarial case and the one a reader will try, so it is
    # always included rather than left to the sample.
    reversed_order = list(reversed(ids))
    if reversed_order not in candidates and reversed_order != ids:
        candidates.insert(0, reversed_order)

    violations: list[Violation] = []
    pairs_compared = 0
    non_positional = 0

    for order in candidates:
        _, _, permuted = pipeline.analyse_record(permute_record(record, order))

        for key, base in baseline.items():
            other = permuted.get(key)
            if other is None:
                violations.append(Violation(
                    query_id=record.query_id, pair=key, field="pair_present",
                    baseline="analysed", permuted="missing", permutation=order,
                ))
                continue

            pairs_compared += 1
            # The default branch was not the first-listed passage, so role
            # assignment went against the positional grain. This is what
            # distinguishes a real pass from a vacuous one.
            if other.default.passage_id != order[0]:
                non_positional += 1

            for fname, b_val, o_val in (
                ("relation", base.relation.value, other.relation.value),
                ("default_branch", base.default.passage_id, other.default.passage_id),
                ("exception_branch", base.exception.passage_id, other.exception.passage_id),
                ("set_relation", base.set_relation.value, other.set_relation.value),
                ("outcomes_agree", str(base.outcomes_agree), str(other.outcomes_agree)),
                ("confidence", f"{base.confidence:.4f}", f"{other.confidence:.4f}"),
            ):
                if b_val != o_val:
                    violations.append(Violation(
                        query_id=record.query_id, pair=key, field=fname,
                        baseline=b_val, permuted=o_val, permutation=order,
                    ))

    return violations, len(candidates), pairs_compared, non_positional


def check(
    records: list[QueryRecord],
    pipeline: ScopePipeline,
    *,
    max_permutations: int = 6,
    seed: int = 0,
) -> InvarianceReport:
    rng = random.Random(seed)
    report = InvarianceReport()
    for rec in records:
        v, n_perm, n_pairs, non_positional = check_record(
            rec, pipeline, max_permutations=max_permutations, rng=rng
        )
        report.records_tested += 1
        report.permutations_tested += n_perm
        report.pairs_compared += n_pairs
        report.non_positional_roles += non_positional
        report.violations.extend(v)
    return report


def main(argv: list[str] | None = None) -> int:
    from detection.variants.compare import load_records

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="JSONL of records. Omit to generate mock records.")
    ap.add_argument("--sample", type=int, default=50, help="how many records to test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-permutations", type=int, default=6)
    ap.add_argument("--heuristic-nli", action="store_true",
                    help="use the lexical NLI stand-in: no model download, no GPU")
    ap.add_argument("--no-llm", action="store_true", default=True,
                    help="use rule-based descriptor extraction (default: on, so this "
                         "check costs nothing to run)")
    ap.add_argument("--use-llm", dest="no_llm", action="store_false",
                    help="use LLM descriptor extraction instead")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.data:
        records = load_records(args.data)[:args.sample]
    else:
        from contract.mock import generate

        records = [i.to_query_record() for i in generate(args.sample, seed=args.seed)]

    records = [r for r in records if len(r.passages) >= 2 and r.conflict_pairs]
    if not records:
        print("no usable records (need >= 2 passages and at least one labelled pair)",
              file=sys.stderr)
        return 1

    pipeline = ScopePipeline(
        use_llm_extraction=not args.no_llm,
        heuristic_nli=args.heuristic_nli,
    )
    report = check(records, pipeline,
                   max_permutations=args.max_permutations, seed=args.seed)
    print(report.render())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "passed": report.passed,
            "pass_rate": report.pass_rate,
            "records_tested": report.records_tested,
            "permutations_tested": report.permutations_tested,
            "pairs_compared": report.pairs_compared,
            "non_positional_roles": report.non_positional_roles,
            "violations": [v.__dict__ for v in report.violations],
        }, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.json}")

    # Non-zero exit on failure, so this can gate CI.
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
