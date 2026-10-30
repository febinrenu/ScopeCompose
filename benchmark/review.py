"""A fast review loop for the three things humans must verify (plan v4 B-8).

Human time is the scarcest resource in this project -- roughly 25-40 hours between two
people -- so the interface is built to make a decision cost seconds, not minutes:

``candidates``
    A mined rule/exception pair. Accept it if the second sentence genuinely narrows the
    first for some group *and the outcome differs*; reject with a reason otherwise.
``branches``
    A model-proposed gold branch structure for an accepted pair. Accept, or correct it.
``personas``
    A model-proposed persona set (``benchmark.personas``). Accept, or correct a statement.
``entailment``
    The reader audit (gate V2): does this answer, by itself, establish the stated outcome for this
    person? ``yes`` / ``no`` / ``unsure``. Blind to the system and to the reader's output.
``vacs``
    Two passages: do they govern the same decision? ``equivalent``, one ``a_broader`` / ``b_broader``
    than the other, or ``distinct``. The label is the verdict.

Properties that matter for the paper, not just for convenience:

* **Append-only and resumable.** One JSONL file per (kind, annotator); an item already
  decided is never shown again, and a crash loses at most the item on screen.
* **Independent.** Two annotators write to two files and never see each other's
  verdicts. Agreement is computed afterwards, with ``benchmark.annotation.agreement``,
  which *refuses* the same annotator twice or any model id -- a kappa between one
  person's two passes is self-consistency, not agreement.
* **The time is recorded** per decision, so "annotation hours" in the paper is a
  measurement and the plan's budget can be checked against what was actually spent.
* **Accepting is not the default.** There is no "accept all"; a key press per item.

``edit`` counts as ``accept`` for agreement (the annotator judged the pair real) and its
edits are kept separately for the adjudication pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

from benchmark.annotation.agreement import KappaResult, cohen_kappa

KINDS = ("candidates", "branches", "personas", "vacs", "entailment")
VERDICTS = ("accept", "edit", "reject", "skip")
#: VACS labels are three-way alignment judgements, stored in the verdict itself.
VACS_LABELS = ("equivalent", "a_broader", "b_broader", "distinct")
OPTIONS = {
    "candidates": ("accept", "reject", "skip", "quit"),
    "branches": ("accept", "edit", "reject", "skip", "quit"),
    "personas": ("accept", "edit", "reject", "skip", "quit"),
    "vacs": (*VACS_LABELS, "skip", "quit"),
    "entailment": ("yes", "no", "unsure", "skip", "quit"),
}
REJECT_REASONS = ("not_an_exception", "same_outcome", "not_cross_document", "unclear")
DEFAULT_DIR = Path(__file__).parent / "data" / "review"


@dataclass(frozen=True)
class Item:
    item_id: str
    kind: str
    payload: dict


@dataclass(frozen=True)
class Decision:
    item_id: str
    kind: str
    annotator: str
    verdict: str
    reason: str = ""
    edits: dict = field(default_factory=dict)
    seconds: float = 0.0
    ts: str = ""


def item_id_for(rule: str, exception: str) -> str:
    return hashlib.sha1(f"{rule.strip()}||{exception.strip()}".encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_items(kind: str, path: Path) -> list[Item]:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    text = path.read_text(encoding="utf-8")
    if kind == "candidates":
        data = json.loads(text)
        props = data["proposals"] if isinstance(data, dict) else data
        seen: dict[str, Item] = {}
        for p in props:
            iid = item_id_for(p["rule"], p["exception"])
            seen.setdefault(iid, Item(iid, kind, p))
        return sorted(seen.values(), key=lambda i: -float(i.payload.get("confidence", 0)))
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    key = "item_id" if kind in ("vacs", "entailment") else "instance_id"
    return [Item(r[key], kind, r) for r in rows]


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


class ReviewStore:
    """One append-only JSONL file per (kind, annotator)."""

    def __init__(self, directory: Path, kind: str, annotator: str):
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if not annotator.strip() or "/" in annotator or "\\" in annotator:
            raise ValueError(f"invalid annotator id {annotator!r}")
        self.kind, self.annotator = kind, annotator.strip()
        self.path = Path(directory) / f"{kind}_{self.annotator}.jsonl"

    def load(self) -> dict[str, Decision]:
        out: dict[str, Decision] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        d = Decision(**json.loads(line))
                    except (TypeError, ValueError):
                        continue            # a half-written last line from a killed run
                    out[d.item_id] = d      # the latest decision for an item wins
        return out

    def append(self, d: Decision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
            fh.flush()

    def hours(self) -> float:
        return sum(d.seconds for d in self.load().values()) / 3600.0


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #


class Prompter(Protocol):
    def show(self, text: str) -> None: ...
    def choose(self, prompt: str, options: tuple[str, ...]) -> str: ...
    def text(self, prompt: str, default: str = "") -> str: ...


def unique_shortcuts(options: tuple[str, ...]) -> dict[str, str]:
    """First-letter shortcuts, only for options whose first letter is not shared.

    ``not_an_exception`` and ``not_cross_document`` both start with ``n``; offering ``n`` would
    silently pick one of them, so neither gets a shortcut and the full word is typed."""
    from collections import Counter

    firsts = Counter(o[0] for o in options)
    return {o[0]: o for o in options if firsts[o[0]] == 1}


class ConsolePrompter:
    def show(self, text: str) -> None:
        print(text.encode(sys.stdout.encoding or "ascii", "replace").decode(sys.stdout.encoding or "ascii"))

    def choose(self, prompt: str, options: tuple[str, ...]) -> str:
        shortcuts = unique_shortcuts(options)
        keyed = set(shortcuts.values())
        label = "/".join(f"({o[0]}){o[1:]}" if o in keyed else o for o in options)
        while True:
            ans = input(f"{prompt} [{label}] ").strip().lower()
            if ans in options:
                return ans
            if ans in shortcuts:
                return shortcuts[ans]
            print("  one of: " + ", ".join(options))

    def text(self, prompt: str, default: str = "") -> str:
        ans = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
        return ans or default


def render_candidate(item: Item) -> str:
    p = item.payload
    return "\n".join([
        "-" * 72,
        f"RULE      ({p.get('rule_doc_id', '?')})",
        f"  {p['rule']}",
        f"EXCEPTION ({p.get('exception_doc_id', '?')})",
        f"  {p['exception']}",
        f"why (model): {p.get('why', '')}   confidence {p.get('confidence', '?')}",
        "Is the second a genuine narrowing of the first, for some group, with a DIFFERENT outcome?",
    ])


def render_branches(item: Item) -> str:
    p = item.payload
    lines = ["-" * 72, f"QUERY: {p.get('query', '')}"]
    for b in p.get("gold_branches", []):
        tag = "default  " if b.get("condition") is None else "exception"
        lines.append(f"  [{b['branch_id']}] {tag} cond={b.get('condition')!r}  outcome={b['outcome']!r}  "
                     f"({b.get('supporting_passage')})")
    return "\n".join(lines)


def render_personas(item: Item) -> str:
    p = item.payload
    lines = ["-" * 72, f"{item.item_id}", f"  default:    {p.get('default', '')}"]
    for bid, s in (p.get("exceptions") or {}).items():
        lines.append(f"  exception {bid}: {s}")
    lines.append(f"  irrelevant ({p.get('irrelevant_attribute', '?')}): {p.get('irrelevant', '')}")
    if p.get("problems"):
        lines.append("  validator problems: " + "; ".join(p["problems"]))
    return "\n".join(lines)


def render_vacs(item: Item) -> str:
    p = item.payload
    return "\n".join([
        "-" * 72,
        f"A ({p.get('doc_a', '?')}):", f"  {p['text_a']}",
        f"B ({p.get('doc_b', '?')}):", f"  {p['text_b']}",
        "Do A and B govern the SAME decision? equivalent / a_broader (A is the more general statement) / "
        "b_broader / distinct",
    ])


def render_entailment(item: Item) -> str:
    p = item.payload
    return "\n".join([
        "-" * 72,
        f"QUESTION: {p.get('query', '')}",
        "ANSWER:",
        f"  {p['answer']}",
        f"PERSON: {p['persona']}",
        f"CLAIM:  for this person, {p['claim']}",
        "Does the ANSWER, by itself, establish the CLAIM for this person? "
        "(yes / no / unsure -- do not use outside knowledge)",
    ])


RENDER = {"candidates": render_candidate, "branches": render_branches, "personas": render_personas,
          "vacs": render_vacs, "entailment": render_entailment}


def edit_item(item: Item, prompter: Prompter) -> dict:
    """Collect corrections. Enter keeps the model's text."""
    p = item.payload
    if item.kind == "branches":
        edits: dict = {}
        for b in p.get("gold_branches", []):
            o = prompter.text(f"  [{b['branch_id']}] outcome", b["outcome"])
            c = prompter.text(f"  [{b['branch_id']}] condition", b.get("condition") or "")
            if o != b["outcome"]:
                edits.setdefault(b["branch_id"], {})["outcome"] = o
            if (c or None) != b.get("condition"):
                edits.setdefault(b["branch_id"], {})["condition"] = c or None
        return edits
    if item.kind == "personas":
        edits = {}
        d = prompter.text("  default", p.get("default", ""))
        if d != p.get("default"):
            edits["default"] = d
        for bid, s in (p.get("exceptions") or {}).items():
            n = prompter.text(f"  exception {bid}", s)
            if n != s:
                edits.setdefault("exceptions", {})[bid] = n
        irr = prompter.text("  irrelevant", p.get("irrelevant", ""))
        if irr != p.get("irrelevant"):
            edits["irrelevant"] = irr
        return edits
    return {}


@dataclass
class Summary:
    shown: int = 0
    accepted: int = 0
    edited: int = 0
    rejected: int = 0
    skipped: int = 0
    already_done: int = 0
    seconds: float = 0.0
    quit_early: bool = False


def review(
    items: Iterable[Item],
    store: ReviewStore,
    prompter: Prompter,
    *,
    limit: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
) -> Summary:
    """Show undecided items one at a time, recording each verdict and how long it took."""
    done = store.load()
    s = Summary()
    for item in items:
        if item.item_id in done:
            s.already_done += 1
            continue
        if limit is not None and s.shown >= limit:
            break
        prompter.show(RENDER[store.kind](item))
        t0 = clock()
        v = prompter.choose("verdict", OPTIONS[store.kind])
        if v == "quit":
            s.quit_early = True
            break
        reason, edits = "", {}
        if v == "reject":
            reason = prompter.choose("reason", REJECT_REASONS)
        elif v == "edit":
            edits = edit_item(item, prompter)
        d = Decision(item.item_id, store.kind, store.annotator, v, reason, edits, round(clock() - t0, 2), now())
        store.append(d)
        s.shown += 1
        s.seconds += d.seconds
        counter = {"accept": "accepted", "edit": "edited", "reject": "rejected", "skip": "skipped"}.get(
            v, "accepted")        # a VACS label is a decision, counted as one
        setattr(s, counter, getattr(s, counter) + 1)
    return s


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #


def labels(store: ReviewStore) -> dict[str, str]:
    """item -> label. accept/reject for the verification kinds (``edit`` counts as accept); the alignment
    label itself for VACS. ``skip`` is missing data, never a label."""
    out = {}
    for iid, d in store.load().items():
        if store.kind == "vacs":
            if d.verdict in VACS_LABELS:
                out[iid] = d.verdict
            continue
        if store.kind == "entailment":
            if d.verdict in ("yes", "no"):          # "unsure" is excluded from the truth, not guessed
                out[iid] = d.verdict
            continue
        if d.verdict in ("accept", "edit"):
            out[iid] = "accept"
        elif d.verdict == "reject":
            out[iid] = "reject"
    return out


def agreement(a: ReviewStore, b: ReviewStore) -> KappaResult:
    """Cohen's kappa between two annotators' accept/reject. Refuses the same person twice."""
    if a.kind != b.kind:
        raise ValueError("agreement compares one kind of review at a time")
    return cohen_kappa(labels(a), labels(b), axis=f"{a.kind}:accept_reject",
                       annotator_a=a.annotator, annotator_b=b.annotator)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review", help="review undecided items")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("--annotator", required=True)
    r.add_argument("--limit", type=int, default=None)
    k = sub.add_parser("agreement", help="kappa between two annotators")
    k.add_argument("a")
    k.add_argument("b")
    st = sub.add_parser("status", help="decisions and hours so far")
    st.add_argument("annotator")
    args = ap.parse_args(argv)

    if args.cmd == "review":
        store = ReviewStore(args.dir, args.kind, args.annotator)
        s = review(load_items(args.kind, args.items), store, ConsolePrompter(), limit=args.limit)
        print(f"\n{s.shown} decided ({s.accepted} accepted, {s.edited} edited, {s.rejected} rejected, "
              f"{s.skipped} skipped); {s.already_done} were already done; {s.seconds / 60:.1f} min this session")
        return 0
    if args.cmd == "agreement":
        res = agreement(ReviewStore(args.dir, args.kind, args.a), ReviewStore(args.dir, args.kind, args.b))
        print(res.render() if hasattr(res, "render") else res)
        return 0
    store = ReviewStore(args.dir, args.kind, args.annotator)
    ds = store.load().values()
    print(f"{len(list(ds))} decisions, {store.hours():.2f} hours recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
