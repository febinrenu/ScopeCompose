"""Record what the Python engine decides, so the JavaScript port can be checked.

The site reimplements ``scope/attributes.py::compare`` and
``ScopeAnalyser._combine`` in JavaScript so the detector can run in a browser
with no server. A reimplementation is a liability: it can drift from the
Python it claims to mirror, and a page that computes a *different* relation
from the pipeline is worse than a page with a screenshot, because it looks
authoritative while being wrong.

This writes the Python answer for every ordered branch pair in the corpus.
``verify_engines.js`` replays those cases through the JavaScript and fails on
any disagreement, so drift is caught rather than demonstrated to a reviewer.

    python site/tools/make_reference.py
    node site/tools/verify_engines.js
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contract.gold import GoldInstance            # noqa: E402
from scope.attributes import compare              # noqa: E402

SOURCE = ROOT / "benchmark" / "data" / "corpus.jsonl"
OUT = Path(__file__).resolve().parent / "compare_reference.json"


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"corpus not found: {SOURCE}")

    instances = [
        GoldInstance.model_validate(json.loads(line))
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cases = []
    for inst in instances:
        branches = inst.gold_branches or []
        # Both orderings: compare() is deliberately argument-relative (SUBSET
        # one way, SUPERSET the other), and a port that silently symmetrised
        # it would invert branch roles -- the general rule would be treated as
        # the exception. Checking both directions catches exactly that.
        for a in branches:
            for b in branches:
                if a.branch_id == b.branch_id:
                    continue
                cases.append({
                    "inst": inst.instance_id,
                    "a": a.branch_id,
                    "b": b.branch_id,
                    "rel": compare(a.applicability, b.applicability).value,
                })

    OUT.write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(cases)} reference cases -> {OUT.relative_to(ROOT)}")
    for rel, n in sorted(Counter(c["rel"] for c in cases).items()):
        print(f"  {rel:<12} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
