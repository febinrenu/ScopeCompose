"""Train / dev / test assignment, grouped by source document.

Two problems this fixes, both of which a reviewer checks before reading any
result.

**There was no test split.** The corpus is 89 `dev` and 39 `train`, zero
`test`. Every threshold tuned and every model selected so far has been chosen
on data that would also have to serve as the evaluation set, and a number
reported from that is a training score wearing an evaluation label.

**Instances share source documents.** 128 instances draw on 186 documents, and
`youth-mobility#overview` alone appears in five of them. Assign instances
independently and the same gov.uk guide lands in train and test, so a system
that memorises its wording scores on test for having seen train. That is
leakage, it inflates every number, and random splitting causes it silently.

So assignment is **by document group, never by instance**: every instance
touching a document goes wherever that document goes. Groups are formed by
connected components over shared documents, because an instance citing two
documents links them -- putting one in train and the other in test splits a
group that cannot be split.

The cost is honest and worth stating: exact proportions become unreachable.
Groups are indivisible, so a 70/15/15 request lands near but not on target, and
this module reports where it actually landed rather than pretending otherwise.

Usage::

    python -m benchmark.splits --data benchmark/data/corpus.jsonl --write
    python -m benchmark.splits --data benchmark/data/corpus.jsonl --audit
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from contract.gold import GoldInstance

#: Default proportions. Test is held out and must not be looked at until the
#: analysis plan says so.
DEFAULT_RATIOS = {"train": 0.60, "dev": 0.20, "test": 0.20}


@dataclass
class Group:
    """Instances tied together by shared source documents."""

    instance_ids: list[str] = field(default_factory=list)
    document_ids: set[str] = field(default_factory=set)

    @property
    def size(self) -> int:
        return len(self.instance_ids)


def build_groups(instances: list[GoldInstance]) -> list[Group]:
    """Connected components over shared documents.

    An instance citing two documents links them, so the relation is transitive
    and components -- not documents -- are the indivisible unit.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for inst in instances:
        docs = [p.document_id or f"__nodoc_{inst.instance_id}"
                for p in inst.passages]
        for d in docs[1:]:
            union(docs[0], d)

    by_root: dict[str, Group] = defaultdict(Group)
    for inst in instances:
        docs = [p.document_id or f"__nodoc_{inst.instance_id}"
                for p in inst.passages]
        g = by_root[find(docs[0])]
        g.instance_ids.append(inst.instance_id)
        g.document_ids.update(docs)
    return list(by_root.values())


def assign(
    instances: list[GoldInstance],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Return ``instance_id -> split``.

    Largest group first into whichever split is furthest below its quota. A
    purely random walk over groups leaves the small splits lumpy when one group
    carries a tenth of the corpus; placing the big ones first spreads the
    unavoidable error instead of concentrating it.
    """
    ratios = ratios or DEFAULT_RATIOS
    groups = build_groups(instances)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -g.size)

    total = sum(g.size for g in groups)
    target = {k: v * total for k, v in ratios.items()}
    have = {k: 0 for k in ratios}
    out: dict[str, str] = {}

    for g in groups:
        split = max(ratios, key=lambda k: target[k] - have[k])
        have[split] += g.size
        for iid in g.instance_ids:
            out[iid] = split
    return out


@dataclass
class SplitAudit:
    counts: dict[str, int] = field(default_factory=dict)
    ratios: dict[str, float] = field(default_factory=dict)
    leaked_documents: dict[str, set[str]] = field(default_factory=dict)
    duplicate_passages: list[tuple[str, int]] = field(default_factory=list)
    by_provenance: dict[str, Counter] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.leaked_documents

    def render(self) -> str:
        total = sum(self.counts.values()) or 1
        lines = ["Split audit", "=" * 68,
                 f"  {'split':<10}{'instances':>12}{'share':>10}",
                 "  " + "-" * 32]
        for name in ("train", "dev", "test"):
            n = self.counts.get(name, 0)
            lines.append(f"  {name:<10}{n:>12}{n / total:>9.1%}")
        lines.append(f"  {'total':<10}{total:>12}")

        lines += ["", "Document leakage", "-" * 68]
        if self.clean:
            lines.append("  none -- no source document appears in two splits")
        else:
            lines.append(f"  {len(self.leaked_documents)} documents span splits:")
            for doc, splits in list(self.leaked_documents.items())[:5]:
                lines.append(f"    {doc[:52]:<54} {sorted(splits)}")
            lines += [
                "",
                "  A system that memorised a document's wording in train scores on",
                "  test for having seen it. Every number is inflated and nothing in",
                "  the output would show it. Re-run assignment rather than reporting.",
            ]

        if self.duplicate_passages:
            lines += ["", "Repeated passages", "-" * 68]
            for text, n in self.duplicate_passages[:5]:
                lines.append(f"  x{n}  {text[:58]}")
            lines += [
                "",
                "  The same passage in several instances is not leakage by itself --",
                "  one rule legitimately has several exceptions. It does mean those",
                "  instances are not independent, so an interval over them is",
                "  narrower than the evidence supports.",
            ]

        if self.by_provenance:
            lines += ["", "Annotation provenance per split", "-" * 68,
                      "  Two classes of evidence. Never pool them in a headline.",
                      ""]
            for split in ("train", "dev", "test"):
                c = self.by_provenance.get(split)
                if c:
                    parts = ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                    lines.append(f"  {split:<8} {parts}")
        return "\n".join(lines)


def audit(instances: list[GoldInstance]) -> SplitAudit:
    """Check an assigned corpus for leakage and composition problems."""
    a = SplitAudit()
    a.counts = dict(Counter(i.split for i in instances))
    total = sum(a.counts.values()) or 1
    a.ratios = {k: v / total for k, v in a.counts.items()}

    doc_splits: dict[str, set[str]] = defaultdict(set)
    for inst in instances:
        for p in inst.passages:
            if p.document_id:
                doc_splits[p.document_id].add(inst.split)
    a.leaked_documents = {d: s for d, s in doc_splits.items() if len(s) > 1}

    texts = Counter(p.text for i in instances for p in i.passages)
    a.duplicate_passages = [(t, n) for t, n in texts.most_common() if n > 1]

    prov: dict[str, Counter] = defaultdict(Counter)
    for inst in instances:
        label = ("two-annotator"
                 if inst.annotation.annotator_a not in (None, "model-proposed")
                 else "model-proposed+adjudicated")
        prov[inst.split][label] += 1
    a.by_provenance = dict(prov)
    return a


def _load(path: Path) -> list[GoldInstance]:
    return [GoldInstance.model_validate(json.loads(l))
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--write", action="store_true",
                    help="assign splits and rewrite the file in place")
    ap.add_argument("--audit", action="store_true",
                    help="check the existing assignment without changing it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    instances = _load(args.data)
    if not instances:
        print(f"no instances in {args.data}")
        return 1

    if args.write:
        mapping = assign(instances, seed=args.seed)
        instances = [i.model_copy(update={"split": mapping[i.instance_id]})
                     for i in instances]
        with args.data.open("w", encoding="utf-8") as fh:
            for i in instances:
                fh.write(i.model_dump_json() + "\n")
        print(f"assigned splits for {len(instances)} instances -> {args.data}\n")

    print(audit(instances).render())

    if args.write:
        print()
        print("  TEST IS HELD OUT. Do not tune a threshold, select a model, or")
        print("  look at a number on it until the analysis plan says so. A test")
        print("  set consulted during development is a dev set with a misleading")
        print("  name, and the paper's headline would be a training score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
