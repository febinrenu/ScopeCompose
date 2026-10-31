"""Closing the human loop: a model proposes, people review, the verdicts land in the corpus.

``benchmark.review`` records what people decide. ``preannotate`` and ``personas`` hold the proposals. This
module is the part between them that was missing, so a verdict changes the data:

    propose-branches   conditional refinements with no branches -> model proposals -> review items
    apply-branches     reviewers' accept / edit / reject -> instances with reviewed branches
    propose-personas   instances with branches but no personas   -> proposals + structural checks -> items
    apply-personas     reviewers' verdicts -> instances with personas and counterfactual pairs
    merge              replace instances in a corpus by id, to a *new* file
    collect-candidates mined rule/exception proposals not yet in the corpus -> one file for ``--kind candidates``
    accepted-candidates the candidates a person accepted -> a proposals file for ``mining.build_batch``

Rules that keep a label honest:

* **Only a person's verdict makes a proposal gold.** Reject, skip and undecided items are never applied; an
  edit is applied *then re-validated* (an edited persona set that breaks a structural check is refused and
  reported, not attached).
* **A second annotator can only hold an item back.** If a second annotator is given, the item is applied only
  when they also decided and did not reject; otherwise it is listed as disputed. The first annotator's edits
  are the ones applied.
* **Nothing is overwritten.** ``apply-*`` write the changed instances to a file; ``merge`` writes a new corpus
  and refuses to write over its input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from benchmark.personas import PersonaProposal, generate_personas, validate, with_personas
from benchmark.preannotate import apply_edits, propose
from benchmark.review import DEFAULT_DIR, Decision, ReviewStore
from contract.gold import GoldInstance
from contract.models import ConflictType, ScopeRelation


def needs_branches(inst: GoldInstance) -> bool:
    return (inst.gold_conflict_type is ConflictType.CONDITIONAL and inst.gold_scope_relation is ScopeRelation.REFINEMENT
            and len(inst.passages) >= 2 and not inst.gold_branches)


def needs_personas(inst: GoldInstance) -> bool:
    return bool(inst.gold_branches and inst.default_branch is not None and inst.exception_branches
                and not inst.personas)


@dataclass
class Report:
    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    undecided: list[str] = field(default_factory=list)
    disputed: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"{len(self.applied)} applied, {len(self.rejected)} rejected, {len(self.undecided)} undecided, "
                 f"{len(self.disputed)} disputed or awaiting the second annotator, {len(self.refused)} refused"]
        lines += [f"  refused {k}: {v}" for k, v in sorted(self.refused.items())]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Proposing
# --------------------------------------------------------------------------- #


def propose_branches(instances: Iterable[GoldInstance], *, client=None, limit: int | None = None
                     ) -> tuple[list[dict], dict[str, str]]:
    """Review items (instance dumps with proposed branches) and ``{instance_id: reason}`` for failures."""
    items: list[dict] = []
    failures: dict[str, str] = {}
    for inst in instances:
        if not needs_branches(inst):
            continue
        if limit is not None and len(items) + len(failures) >= limit:
            break
        try:
            items.append(propose(inst, client=client).model_dump(mode="json"))
        except Exception as e:                      # one bad proposal must not stop the batch
            failures[inst.instance_id] = f"{type(e).__name__}: {e}"
    return items, failures


def propose_persona_items(instances: Iterable[GoldInstance], *, client=None, limit: int | None = None
                          ) -> tuple[list[dict], dict[str, str]]:
    """Review items for personas: the proposal and its structural problems, so a reviewer sees them."""
    items: list[dict] = []
    failures: dict[str, str] = {}
    for inst in instances:
        if not needs_personas(inst):
            continue
        if limit is not None and len(items) + len(failures) >= limit:
            break
        try:
            p = generate_personas(inst, client=client)
        except Exception as e:
            failures[inst.instance_id] = f"{type(e).__name__}: {e}"
            continue
        items.append({"instance_id": inst.instance_id, "default": p.default, "exceptions": p.exceptions,
                      "irrelevant_attribute": p.irrelevant_attribute, "irrelevant": p.irrelevant,
                      "problems": p.problems})
    return items, failures


# --------------------------------------------------------------------------- #
# Applying verdicts
# --------------------------------------------------------------------------- #


def _gate(item_id: str, primary: Mapping[str, Decision], second: Mapping[str, Decision] | None, report: Report
          ) -> Decision | None:
    """The primary decision if the item should be applied; otherwise record why not and return ``None``."""
    d = primary.get(item_id)
    if d is None or d.verdict == "skip":
        report.undecided.append(item_id)
        return None
    if d.verdict == "reject":
        report.rejected.append(item_id)
        return None
    if second is not None:
        s = second.get(item_id)
        if s is None or s.verdict in ("skip", "reject"):
            report.disputed.append(item_id)
            return None
    return d


def apply_branches(items: Sequence[dict], primary: Mapping[str, Decision], *, annotator: str,
                   second: Mapping[str, Decision] | None = None) -> tuple[list[GoldInstance], Report]:
    report = Report()
    out: list[GoldInstance] = []
    for row in items:
        iid = row["instance_id"]
        d = _gate(iid, primary, second, report)
        if d is None:
            continue
        try:
            out.append(apply_edits(GoldInstance.model_validate(row), d.edits if d.verdict == "edit" else {},
                                   annotator=annotator))
            report.applied.append(iid)
        except (ValueError, KeyError) as e:
            report.refused[iid] = str(e)
    return out, report


def _edited_proposal(row: Mapping, edits: Mapping) -> PersonaProposal:
    exceptions = dict(row.get("exceptions") or {})
    exceptions.update(edits.get("exceptions") or {})
    return PersonaProposal(row["instance_id"], str(edits.get("default", row.get("default", ""))).strip(),
                           {k: str(v).strip() for k, v in exceptions.items()}, row.get("irrelevant_attribute", ""),
                           str(edits.get("irrelevant", row.get("irrelevant", ""))).strip())


def apply_personas(instances_by_id: Mapping[str, GoldInstance], items: Sequence[dict],
                   primary: Mapping[str, Decision], *, second: Mapping[str, Decision] | None = None
                   ) -> tuple[list[GoldInstance], Report]:
    report = Report()
    out: list[GoldInstance] = []
    for row in items:
        iid = row["instance_id"]
        d = _gate(iid, primary, second, report)
        if d is None:
            continue
        inst = instances_by_id.get(iid)
        if inst is None:
            report.refused[iid] = "not in the corpus"
            continue
        prop = _edited_proposal(row, d.edits if d.verdict == "edit" else {})
        prop.problems = validate(inst, prop)
        if prop.problems:
            report.refused[iid] = "; ".join(prop.problems)
            continue
        try:
            out.append(with_personas(inst, prop))
            report.applied.append(iid)
        except ValueError as e:
            report.refused[iid] = str(e)
    return out, report


def collect_candidates(proposal_files: Iterable[Path], corpus: Sequence[GoldInstance]) -> list[dict]:
    """Cross-document proposals not already in the corpus, deduplicated, highest confidence first.

    A proposal is *already in* the corpus when both its sentences appear in the passages of some instance
    (compared on their first 60 characters, whitespace-collapsed), so a reviewer is never asked about a pair
    that has already been adjudicated."""
    from benchmark.review import item_id_for

    def flat(t: str) -> str:
        return " ".join(t.split())

    text = flat(" ".join(p.text for i in corpus for p in i.passages))
    seen: dict[str, dict] = {}
    for f in proposal_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for p in (data["proposals"] if isinstance(data, dict) else data):
            if not p.get("cross_document", True):
                continue
            if flat(p["rule"])[:60] in text and flat(p["exception"])[:60] in text:
                continue
            seen.setdefault(item_id_for(p["rule"], p["exception"]), p)
    return sorted(seen.values(), key=lambda p: -float(p.get("confidence", 0)))


def accepted_candidates(proposals: Sequence[dict], primary: Mapping[str, Decision], *,
                        second: Mapping[str, Decision] | None = None) -> tuple[list[dict], Report]:
    """The candidates a person accepted, in the proposals format ``mining.build_batch`` reads.

    Candidate ids are those of ``benchmark.review`` (a hash of the two sentences). The proposer's own
    ``why`` and ``confidence`` are dropped: ``build_batch`` drops them again, and an annotator who later sees
    what the model thought is no longer an independent observer."""
    from benchmark.review import item_id_for

    report = Report()
    out: list[dict] = []
    for p in proposals:
        iid = item_id_for(p["rule"], p["exception"])
        if _gate(iid, primary, second, report) is None:
            continue
        report.applied.append(iid)
        out.append({k: v for k, v in p.items() if k not in ("why", "confidence", "score")})
    return out, report


def merge(base: Sequence[GoldInstance], updates: Iterable[GoldInstance]) -> list[GoldInstance]:
    """``base`` with each updated instance replacing the one of the same id, order preserved."""
    new = {u.instance_id: u for u in updates}
    unknown = set(new) - {b.instance_id for b in base}
    if unknown:
        raise ValueError(f"updates for instances not in the corpus: {sorted(unknown)[:5]}")
    return [new.get(b.instance_id, b) for b in base]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load(path: Path) -> list[GoldInstance]:
    return [GoldInstance.model_validate(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("propose-branches", "propose-personas"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--limit", type=int, default=None)
    for name in ("apply-branches", "apply-personas"):
        p = sub.add_parser(name)
        p.add_argument("--items", type=Path, required=True)
        p.add_argument("--annotator", required=True)
        p.add_argument("--second", default=None, help="a second annotator who must also have accepted or edited")
        p.add_argument("--review-dir", type=Path, default=DEFAULT_DIR)
        p.add_argument("--out", type=Path, required=True)
        if name == "apply-personas":
            p.add_argument("--data", type=Path, required=True)
    c = sub.add_parser("collect-candidates")
    c.add_argument("--proposals", type=Path, nargs="+", required=True)
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("accepted-candidates")
    a.add_argument("--items", type=Path, required=True, help="the candidates file that was reviewed")
    a.add_argument("--annotator", required=True)
    a.add_argument("--second", default=None)
    a.add_argument("--review-dir", type=Path, default=DEFAULT_DIR)
    a.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--data", type=Path, required=True)
    m.add_argument("--updates", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd.startswith("propose"):
        fn = propose_branches if args.cmd == "propose-branches" else propose_persona_items
        items, failures = fn(_load(args.data), limit=args.limit)
        _write(args.out, items)
        if failures:
            args.out.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        kind = "branches" if args.cmd == "propose-branches" else "personas"
        print(f"{len(items)} proposals written to {args.out}; {len(failures)} failed")
        print(f"review with: python -m benchmark.review --kind {kind} review --items {args.out} --annotator <you>")
        return 0

    if args.cmd == "collect-candidates":
        cands = collect_candidates(args.proposals, _load(args.data))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"proposals": cands}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{len(cands)} candidates not yet in the corpus -> {args.out}")
        print(f"review with: python -m benchmark.review --kind candidates review --items {args.out} --annotator <you>")
        return 0

    if args.cmd == "accepted-candidates":
        data = json.loads(args.items.read_text(encoding="utf-8"))
        accepted, report = accepted_candidates(
            data["proposals"] if isinstance(data, dict) else data,
            ReviewStore(args.review_dir, "candidates", args.annotator).load(),
            second=ReviewStore(args.review_dir, "candidates", args.second).load() if args.second else None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"proposals": accepted}, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(report.render())
        print(f"{len(accepted)} accepted candidates -> {args.out}; build the batch with: "
              "python -m benchmark.mining.build_batch --proposals <this file> --docs <docs> --out <batch>")
        return 0

    if args.cmd == "merge":
        if args.out.resolve() == args.data.resolve():
            print("error: --out must differ from --data (nothing is overwritten)", file=sys.stderr)
            return 2
        try:
            merged = merge(_load(args.data), _load(args.updates))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        _write(args.out, (i.model_dump(mode="json") for i in merged))
        print(f"wrote {len(merged)} instances to {args.out}")
        return 0

    kind = "branches" if args.cmd == "apply-branches" else "personas"
    primary = ReviewStore(args.review_dir, kind, args.annotator).load()
    second = ReviewStore(args.review_dir, kind, args.second).load() if args.second else None
    items = _rows(args.items)
    if kind == "branches":
        updated, report = apply_branches(items, primary, annotator=args.annotator, second=second)
    else:
        updated, report = apply_personas({i.instance_id: i for i in _load(args.data)}, items, primary, second=second)
    _write(args.out, (i.model_dump(mode="json") for i in updated))
    print(report.render())
    print(f"wrote {len(updated)} instances to {args.out}; merge them with: python -m benchmark.human_loop merge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
