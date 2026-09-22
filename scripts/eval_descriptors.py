"""Measure how often A4's descriptor extractor produces TYPED attributes.

This is the difference between A4 deciding scope arithmetically and A4 asking
a language model to do set theory in prose. A typed attribute -- an ordered
tier, a numeric range, a category set -- makes subset and disjointness exact.
Free text, or no attributes at all, sends the decision to entailment, which is
a probability dressed as an answer.

Live runs were falling through to entailment 50-100% of the time, so the rate
is worth measuring directly rather than inferring from pipeline output.

Usage::

    python scripts/eval_descriptors.py            # 12 representative passages
    python scripts/eval_descriptors.py --n 30 --data dev.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings  # noqa: E402

settings.reload()

from contract.gold import AttributeKind  # noqa: E402
from scope.descriptor import DescriptorExtractor  # noqa: E402

# Representative of the two benchmark domains, covering the shapes the
# extractor is supposed to type: tiers, categories, numeric thresholds,
# booleans, and genuine defaults.
CASES: list[tuple[str, str]] = [
    ("Do I pay an international transaction fee?",
     "International transactions incur a 3% fee."),
    ("Do I pay an international transaction fee?",
     "The fee is waived for premium-tier cardholders."),
    ("Do I pay an international transaction fee?",
     "Platinum and signature cardholders enjoy fee-free foreign currency purchases."),
    ("Is there a maintenance charge?",
     "A monthly maintenance charge of 2% applies to all current accounts."),
    ("Is there a maintenance charge?",
     "The charge does not apply to accounts held by students under 23."),
    ("Do large transfers attract a fee?",
     "The premium-tier waiver does not apply to transactions above 5,000."),
    ("Can an F-1 holder work?",
     "An F-1 holder may not accept employment in the United States."),
    ("Can an F-1 holder work?",
     "An F-1 holder may accept employment if granted economic hardship authorisation."),
    ("Can an F-1 holder work?",
     "Students enrolled less than full-time may not accept employment under any authorisation."),
    ("How long can I stay?",
     "The J-1 category permits a maximum stay of five years."),
    ("What documents do I need?",
     "A completed form and the filing fee are required for every application."),
    ("Am I eligible for the waiver?",
     "Applicants aged 65 or over are exempt from the filing fee."),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help="sample passages from a gold JSONL instead of the built-ins")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args(argv)

    cases = CASES
    if args.data:
        from experiments.run_baselines import load_gold

        cases = [
            (inst.query, p.text)
            for inst in load_gold(args.data)
            for p in inst.passages
        ][: args.n]
    cases = cases[: args.n]

    extractor = DescriptorExtractor()

    typed = untyped = defaults = 0
    kinds: dict[str, int] = {}
    rows = []

    for i, (query, text) in enumerate(cases):
        d = extractor.extract(query, f"p{i}", text)
        app = d.applicability
        attrs = app.attributes
        real = [a for a in attrs if a.kind is not AttributeKind.FREE_TEXT]

        if app.is_default:
            defaults += 1
            verdict = "default"
        elif real:
            typed += 1
            verdict = "TYPED"
        else:
            untyped += 1
            verdict = "untyped"

        for a in attrs:
            kinds[a.kind.value] = kinds.get(a.kind.value, 0) + 1

        rows.append((verdict, text[:54], app.descriptor[:34],
                     ",".join(f"{a.name}:{a.kind.value}" for a in attrs)[:44],
                     d.outcome[:26]))

    print(f"{len(cases)} passages\n")
    print(f"  {'verdict':<9} {'passage':<56} {'descriptor':<36} {'attributes':<46} outcome")
    print("  " + "-" * 176)
    for r in rows:
        print(f"  {r[0]:<9} {r[1]:<56} {r[2]:<36} {r[3]:<46} {r[4]}")

    scopeful = typed + untyped
    rate = typed / scopeful if scopeful else 0.0
    print()
    print("Typed-attribute rate on scope-bearing passages")
    print("-" * 78)
    print(f"  typed      {typed:>4}   (scope decided arithmetically -- exact)")
    print(f"  untyped    {untyped:>4}   (falls through to entailment -- a model's judgement)")
    print(f"  default    {defaults:>4}   (unconditioned; no attributes expected)")
    print(f"  RATE       {rate:>4.0%}")
    if kinds:
        print("\n  kinds produced: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    print()
    if rate >= 0.8:
        print("  Good. Most scope decisions will be exact set operations.")
    elif rate >= 0.5:
        print("  Mixed. A meaningful share of scope decisions still rest on entailment;")
        print("  report the attributes/entailment split alongside four-way accuracy.")
    else:
        print("  POOR. Most scope decisions are falling through to entailment, which")
        print("  means A4's typed attribute algebra is mostly unused and four-way")
        print("  accuracy is really measuring an NLI model's prose reasoning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
