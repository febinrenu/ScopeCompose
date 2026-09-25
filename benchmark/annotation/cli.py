"""The annotation tool.

Four modes, matching the protocol in the manual:

``annotate``
    A blind pass. One annotator, one batch, no sight of anyone else's labels.
    This is what produces the two independent passes a kappa needs.
``adjudicate``
    Review a *proposed* label rather than create one from scratch. Roughly four
    times faster per instance, and the reason the human cost of a 300-instance
    corpus is hours rather than days. It also carries automation bias, so the
    basis is recorded on every record and reported in the datasheet: a label
    adjudicated from a proposal is not the same evidence as one judged cold.
``agreement``
    Cohen's kappa per axis, plus the disagreement cells that say which
    paragraph of the manual to rewrite.
``build-gold``
    Assemble ``GoldInstance`` records from reconciled labels.

Usage::

    python -m benchmark.annotation.cli annotate --annotator jg --batch pilot.jsonl
    python -m benchmark.annotation.cli agreement --a jg --b rm
    python -m benchmark.annotation.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmark.annotation.agreement import AgreementError, cohen_kappa
from benchmark.annotation.store import AnnotationError, AnnotationStore, LabelRecord
from contract.gold import GoldInstance
from contract.models import ConflictType, QueryRecord, ScopeRelation

TYPE_KEYS = {
    "1": ConflictType.NO_CONFLICT, "2": ConflictType.FACTUAL,
    "3": ConflictType.TEMPORAL, "4": ConflictType.OPINION,
    "5": ConflictType.CONDITIONAL,
}
RELATION_KEYS = {
    "1": ScopeRelation.REFINEMENT, "2": ScopeRelation.DISJOINT,
    "3": ScopeRelation.REDUNDANT, "4": ScopeRelation.OPPOSED,
}

TYPE_PROMPT = """
  Can both passages be true at the same time, for DIFFERENT cases?
    yes -> conditional        no -> factual / temporal / opinion

  1 no_conflict   2 factual   3 temporal   4 opinion   5 conditional

  u  NOT SURE -- genuinely undecidable, send to the third reviewer
     Use it freely. Across 91 labels each, neither annotator has ever
     marked one, and real policy text is not that clean. A guess on an
     undecidable case enters the corpus wearing a real label and cannot
     be found again; an honest `u` costs nothing.
  s skip   q save and quit"""

RELATION_PROMPT = """
  Step 1: do the applicability sets overlap, and is one nested in the other?
  Step 2: where they overlap, do the outcomes agree?

  1 refinement  (nested, outcomes differ -> compose)
  2 disjoint    (no overlap              -> compose)
  3 redundant   (same scope, same outcome-> merge)
  4 opposed     (same scope, outcomes clash -> selection fallback)
  e escalate to third reviewer"""


def _load_batch(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "gold_conflict_type" in obj:
            records.append(GoldInstance.model_validate(obj).to_query_record())
        else:
            records.append(QueryRecord.model_validate(obj))
    return records


def _show(record: QueryRecord, index: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{index}/{total}]  {record.query_id}")
    print("=" * 78)
    print(f"\n  Q: {record.query}\n")
    for p in record.passages:
        doc = f"  [{p.id}] ({p.source_type.value}, {p.date}, doc={p.document_id})"
        print(doc)
        for chunk in _wrap(p.text, 72):
            print(f"      {chunk}")
        print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def _numeric_disagreement(record: QueryRecord) -> bool:
    """Whether any two passages quote figures that disagree.

    Reuses the metric's own numeric check rather than a second regex, so the
    guard and the scorer cannot drift on what counts as a figure.
    """
    from metrics.branch_match import numbers_conflict

    texts = [p.text for p in record.passages]
    return any(numbers_conflict(texts[i], texts[j])
               for i in range(len(texts)) for j in range(i + 1, len(texts)))


def _ask(prompt: str, valid: set[str]) -> str:
    while True:
        try:
            got = input(prompt).strip().lower()
        except EOFError:
            return "q"
        if got in valid:
            return got
        print(f"  expected one of: {' '.join(sorted(valid))}")


def cmd_annotate(args: argparse.Namespace) -> int:
    store = AnnotationStore(args.dir)

    if store.is_sealed(args.annotator):
        print(f"{args.annotator}'s pass is sealed and cannot be extended.",
              file=sys.stderr)
        return 1

    batch = _load_batch(args.batch)
    done = {r.instance_id for r in store.load(args.annotator)}
    todo = [r for r in batch if r.query_id not in done]

    if not todo:
        print(f"{args.annotator} has already labelled all {len(batch)} instances.")
        print("Seal the pass with --seal once both annotators are finished.")
        return 0

    print(f"\n{args.annotator}: {len(todo)} to label ({len(done)} already done).")
    print("Judge from the passages alone. Do not consult anyone else's labels --")
    print("the kappa is only meaningful if the two passes are independent.\n")

    for i, record in enumerate(todo, start=1):
        # Every rule in the manual is relative to a question. Without one there
        # is no frame, two annotators frame differently, and the kappa measures
        # the framing. Pilot 2 ran on a batch of unfilled placeholders and
        # returned -0.013. Skipped rather than labelled.
        if record.query.startswith("[NO QUERY") or record.query.startswith("[to be"):
            print(f"\n[{i}/{len(todo)}] {record.query_id}: SKIPPED -- no query.")
            print("  The manual judges everything relative to a question, so this")
            print("  instance cannot be labelled. Rebuild the batch with queries.")
            continue

        _show(record, i, len(todo))
        started = time.perf_counter()

        print(TYPE_PROMPT)
        key = _ask("\n  conflict type> ", set(TYPE_KEYS) | {"s", "u", "e", "q"})
        if key == "q":
            break
        if key == "s":
            continue

        if key in TYPE_KEYS and TYPE_KEYS[key] is ConflictType.OPINION:
            # `opinion` means neither passage is checkable. Two passages quoting
            # figures that disagree are checkable by definition, so the label is
            # impossible rather than merely unlikely. Two pilot labels landed
            # here, and `3` (temporal) sits next to `4` (opinion) on the menu --
            # which makes a slip as likely an explanation as a judgement.
            if _numeric_disagreement(record):
                print("\n  These passages quote figures that disagree, so they ARE")
                print("  checkable and the label cannot be `opinion`. Most likely")
                print("  `factual`, or `temporal` if one has superseded the other.")
                choice = _ask("  [f]actual  [t]emporal  [o]pinion anyway  [s]kip> ",
                              {"f", "t", "o", "s"})
                if choice == "s":
                    continue
                key = {"f": "2", "t": "3", "o": "4"}[choice]

        escalated = key in ("u", "e")
        if escalated:
            ctype = ConflictType.CONDITIONAL
            relation = None
            # Optional, not required. Demanding a typed justification is
            # friction on the one action the manual most wants taken, and the
            # observed escalation rate across two pilots was zero.
            note = input("  why, in a few words (enter to skip)? ").strip() or None
        else:
            ctype = TYPE_KEYS[key]
            relation = None
            note = None
            if ctype is ConflictType.CONDITIONAL:
                print(RELATION_PROMPT)
                rkey = _ask("\n  scope relation> ", set(RELATION_KEYS) | {"e"})
                if rkey == "e":
                    escalated = True
                    note = input("  why, in a few words (enter to skip)? ").strip() or None
                else:
                    relation = RELATION_KEYS[rkey]

        try:
            store.append(LabelRecord(
                instance_id=record.query_id, annotator=args.annotator,
                conflict_type=ctype.value,
                scope_relation=relation.value if relation else None,
                escalated=escalated, notes=note,
                seconds_spent=round(time.perf_counter() - started, 1),
                basis="blank",
            ))
        except AnnotationError as exc:
            print(f"  not recorded: {exc}", file=sys.stderr)

    print("\n" + store.progress(args.annotator, total=len(batch)))
    if args.seal:
        store.seal(args.annotator)
        print(f"  {args.annotator}'s pass is now SEALED.")
    return 0


def cmd_adjudicate(args: argparse.Namespace) -> int:
    """Review model-proposed labels instead of creating labels cold.

    Roughly four times faster per instance, and that is the whole point: it is
    the difference between twenty hours of annotation and three. The cost is
    automation bias -- a plausible wrong proposal is accepted more readily than
    the same wrong answer would be reached independently -- so every record
    carries ``basis="model"`` and the datasheet reports the split. An
    adjudicated label is real human judgement, but it is not the same evidence
    as one reached cold, and the two are never pooled silently.

    Deliberately NOT usable for the kappa pilot: a shared starting point
    correlates the two annotators, which is exactly what kappa assumes is
    absent. The pilot uses ``annotate``; bulk labelling uses this.
    """
    from experiments.run_pipeline import run as run_pipeline

    store = AnnotationStore(args.dir)
    batch = _load_batch(args.batch)
    done = {r.instance_id for r in store.load(args.annotator)}
    todo = [r for r in batch if r.query_id not in done]
    if not todo:
        print(f"{args.annotator} has adjudicated all {len(batch)} instances.")
        return 0

    print(f"Generating proposals for {len(todo)} instances...")
    proposed, _, _ = run_pipeline(todo, live=args.live)
    by_id = {r.query_id: r for r in proposed}

    accepted = overridden = 0
    for i, record in enumerate(todo, start=1):
        _show(record, i, len(todo))
        started = time.perf_counter()

        pred = by_id.get(record.query_id)
        pairs = pred.conflict_pairs if pred else []
        top = max(pairs, key=lambda p: p.confidence, default=None) if pairs else None

        if top is None:
            print("  PROPOSED: no_conflict")
            p_type, p_rel = ConflictType.NO_CONFLICT, None
        else:
            p_type, p_rel = top.type, top.scope_relation
            print(f"  PROPOSED: {p_type.value}"
                  + (f" / {p_rel.value}" if p_rel else "")
                  + f"   (confidence {top.confidence:.2f})")
        print("\n  The proposal is a starting point, not a default. If you would not")
        print("  have reached it yourself, override it.")

        key = _ask("\n  [a]ccept  [c]hange  [s]kip  [e]scalate  [q]uit> ",
                   {"a", "c", "s", "e", "q"})
        if key == "q":
            break
        if key == "s":
            continue

        note = None
        escalated = key == "e"
        if escalated:
            note = input("  why is it ambiguous? ").strip() or None
            ctype, relation = p_type, None
        elif key == "a":
            ctype, relation = p_type, p_rel
            accepted += 1
        else:
            print(TYPE_PROMPT)
            k = _ask("\n  conflict type> ", set(TYPE_KEYS))
            ctype = TYPE_KEYS[k]
            relation = None
            if ctype is ConflictType.CONDITIONAL:
                print(RELATION_PROMPT)
                rk = _ask("\n  scope relation> ", set(RELATION_KEYS) | {"e"})
                if rk == "e":
                    escalated = True
                    note = input("  why is it ambiguous? ").strip() or None
                else:
                    relation = RELATION_KEYS[rk]
            overridden += 1

        try:
            store.append(LabelRecord(
                instance_id=record.query_id, annotator=args.annotator,
                conflict_type=ctype.value,
                scope_relation=relation.value if relation else None,
                escalated=escalated, notes=note,
                seconds_spent=round(time.perf_counter() - started, 1),
                basis="model",
            ))
        except AnnotationError as exc:
            print(f"  not recorded: {exc}", file=sys.stderr)

    total = accepted + overridden
    print("\n" + store.progress(args.annotator, total=len(batch)))
    if total:
        rate = accepted / total
        print(f"\n  accepted {accepted}/{total} proposals ({rate:.0%})")
        if rate > 0.95 and total >= 20:
            print("  Over 95% accepted. That is either a very good proposer or")
            print("  automation bias. Re-label a sample of these cold with")
            print("  `annotate` and compare -- if the cold labels differ, the")
            print("  adjudicated ones are recording the model, not you.")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    store = AnnotationStore(args.dir)
    ok = True

    print("\nInter-annotator agreement")
    print("=" * 78)
    print(f"  {args.a} vs {args.b}\n")

    for axis, label in (("conflict_type", "Five-class conflict type"),
                        ("scope_relation", "Four-way scope relation")):
        try:
            la, lb = store.labels(args.a, axis), store.labels(args.b, axis)

            # A case either annotator deferred is not a disagreement -- it is
            # the §1 protocol working, and scoring it as a mismatch punishes
            # them for following it. Held out, and the rate reported below.
            deferred = store.deferred(args.a) | store.deferred(args.b)
            la = {k: v for k, v in la.items() if k not in deferred}
            lb = {k: v for k, v in lb.items() if k not in deferred}

            if args.prefix:
                # Passes accumulate across batches, so comparing whole passes
                # silently pools every pilot ever run. Pilot 2 reported n=73
                # -- 54 old instances plus 19 new -- and the pooled figure hid
                # a per-batch kappa of -0.013 behind pilot 1's 0.326.
                la = {k: v for k, v in la.items() if k.startswith(args.prefix)}
                lb = {k: v for k, v in lb.items() if k.startswith(args.prefix)}
            result = cohen_kappa(
                la, lb, axis=label, annotator_a=args.a, annotator_b=args.b,
                seed=args.seed,
            )
        except AgreementError as exc:
            print(f"{label}\n" + "-" * 74)
            print(f"  NOT COMPUTED: {exc}\n")
            ok = False
            continue
        print(result.render())
        print()
        ok = ok and result.is_acceptable

    # Deferral rate, reported whether or not anything was deferred. A rate of
    # zero across a whole corpus of real policy text is not a clean corpus --
    # it means the escape hatch the manual insists on is not being used, and
    # undecidable cases are entering as guesses that nobody can find again.
    def_a, def_b = store.deferred(args.a), store.deferred(args.b)
    if args.prefix:
        def_a = {k for k in def_a if k.startswith(args.prefix)}
        def_b = {k for k in def_b if k.startswith(args.prefix)}
    n_seen = len({r.instance_id for r in store.load(args.a)
                  if not args.prefix or r.instance_id.startswith(args.prefix)})
    print("Deferrals (held out of the kappa above)")
    print("-" * 74)
    print(f"  {args.a:<12} {len(def_a):>4} of {n_seen}")
    print(f"  {args.b:<12} {len(def_b):>4} of {n_seen}")
    print(f"  {'both agreed undecidable':<12} {len(def_a & def_b):>4}")
    if not (def_a or def_b):
        print()
        print("  NOBODY DEFERRED ANYTHING. The manual (1, 4) says to escalate")
        print("  rather than guess, and real policy text is not clean enough for")
        print("  that rate to be honest. Every genuinely undecidable case was")
        print("  instead resolved by a guess -- and two independent guesses on an")
        print("  undecidable case disagree about half the time, which is a large")
        print("  part of what the kappa above is measuring.")
        print()
        print("  Press `u` freely on the next pass. A deferred case is data.")
    print()

    if not ok:
        print("=" * 78)
        print("Not clear to start bulk annotation. Fix the manual where the")
        print("disagreements cluster, then re-run the pilot on a fresh batch.")
    else:
        print("=" * 78)
        print("Both axes clear 0.61. Bulk annotation can proceed.")

    # Seal after comparing: from here on, revising a pass would mean revising
    # it in the light of the other annotator's answers.
    if args.seal:
        for who in (args.a, args.b):
            store.seal(who)
        print(f"\nSealed both passes ({args.a}, {args.b}).")
    return 0 if ok else 2


def cmd_status(args: argparse.Namespace) -> int:
    store = AnnotationStore(args.dir)
    people = store.annotators()
    if not people:
        print(f"No annotation passes in {store.root}.")
        print("Start one:  python -m benchmark.annotation.cli annotate "
              "--annotator <you> --batch <batch.jsonl>")
        return 0

    total = None
    if args.batch and args.batch.exists():
        total = len(_load_batch(args.batch))

    print(f"\nAnnotation progress ({store.root})")
    print("=" * 78)
    for who in people:
        print(store.progress(who, total=total))
        print()

    if len(people) < 2:
        print("Only one annotator has a pass. Kappa needs two independent ones.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchmark.annotation.cli", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None,
                    help="annotation directory (default benchmark/data/annotations)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("annotate", help="blind labelling pass")
    p.add_argument("--annotator", required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--seal", action="store_true", help="seal the pass when finished")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("adjudicate", help="review model-proposed labels (bulk pass)")
    p.add_argument("--annotator", required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--live", action="store_true",
                   help="use the live pipeline to propose; offline uses heuristics")
    p.set_defaults(func=cmd_adjudicate)

    p = sub.add_parser("agreement", help="Cohen's kappa per axis")
    p.add_argument("--a", required=True, help="first annotator id")
    p.add_argument("--b", required=True, help="second annotator id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default=None,
                   help="score only instance ids starting with this, e.g. wp2_ . "
                        "Without it, every batch both annotators have ever "
                        "labelled is pooled into one figure.")
    p.add_argument("--seal", action="store_true",
                   help="seal both passes after comparing")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("status", help="who has labelled what")
    p.add_argument("--batch", type=Path, default=None)
    p.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    if args.dir is None:
        from benchmark.annotation.store import DEFAULT_DIR
        args.dir = DEFAULT_DIR
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
