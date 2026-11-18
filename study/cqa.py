"""ConditionalQA as a chunked open-retrieval testbed.

ConditionalQA (Sun et al., ACL 2022) gives, for each question, the gov.uk page
it is about, a user scenario, the answers, and for each answer the *conditions*
under which it holds. Conditions and evidence are annotated as HTML elements of
the page (``<p>``, ``<li>``, ``<tr>``, headings), quoted verbatim.

That element-level gold is what this module is built around. A chunk here is a
contiguous run of whole elements, so "which chunk holds this condition" has an
exact answer, and the benchmark's official metric -- which scores predicted
conditions by exact element match -- can be applied to a system that only ever
saw chunks.

Nothing in this file calls a model.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from study.fetch_data import path_of

DEFAULT_CHUNK_WORDS = 150
STRATEGIES = ("fixed", "section")

_TAG_RE = re.compile(r"^<(\w+)>(.*)</\1>$", re.DOTALL)
_HEADINGS = ("h1", "h2", "h3", "h4")


def split_element(element: str) -> tuple[str, str]:
    """``"<li>already claiming</li>"`` -> ``("li", "already claiming")``."""
    m = _TAG_RE.match(element.strip())
    return (m.group(1).lower(), m.group(2).strip()) if m else ("p", element.strip())


def n_words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# Documents and chunks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Doc:
    index: int
    url: str
    title: str
    elements: tuple[str, ...]          # verbatim, tags included: the unit the gold is written in

    @property
    def id(self) -> str:
        return f"d{self.index}"


@dataclass(frozen=True)
class Chunk:
    """Elements ``[start, end)`` of one document."""

    doc: int
    index: int
    start: int
    end: int
    title: str
    heading: str                        # the nearest heading above the chunk's first element
    elements: tuple[str, ...]

    @property
    def id(self) -> str:
        return f"d{self.doc}c{self.index}"

    @property
    def element_ids(self) -> tuple[tuple[int, int], ...]:
        return tuple((self.doc, i) for i in range(self.start, self.end))

    @property
    def body(self) -> str:
        return "\n".join(_render(e) for e in self.elements)

    @property
    def text(self) -> str:
        """What is indexed: the page title and section heading, then the elements.

        Without the title, chunks from different gov.uk guides that share
        boilerplate ("You must be 18 or over.") are indistinguishable to a
        retriever. Prefixing it is standard practice and is applied to every
        system alike.
        """
        head = self.title if not self.heading or self.heading == self.title else f"{self.title} - {self.heading}"
        return f"{head}\n{self.body}"

    @property
    def n_words(self) -> int:
        return sum(n_words(split_element(e)[1]) for e in self.elements)


def _render(element: str) -> str:
    tag, text = split_element(element)
    if tag == "li":
        return f"- {text}"
    if tag in _HEADINGS:
        return f"## {text}"
    return text


def chunk_doc(doc: Doc, *, words: int = DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> list[Chunk]:
    """Cut a document into chunks of whole elements.

    ``fixed``
        Greedy packing up to ``words``. Blind to structure: a chunk boundary
        falls wherever the budget runs out. This is the default fixed-size
        chunker of most RAG stacks.
    ``section``
        The same budget, but a top-level heading always starts a new chunk and
        a heading is never left as the last element of a chunk.

    An element is never split. One longer than the budget is a chunk by itself.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; one of {STRATEGIES}")
    if words < 1:
        raise ValueError("words must be positive")
    spans: list[tuple[int, int]] = []
    start, cur = 0, 0
    for i, el in enumerate(doc.elements):
        tag, text = split_element(el)
        n = n_words(text)
        breaks = i > start and (cur + n > words or (strategy == "section" and tag in ("h1", "h2")))
        if breaks:
            end = i
            if strategy == "section":            # do not strand a heading at the end of a chunk
                while end - 1 > start and split_element(doc.elements[end - 1])[0] in _HEADINGS:
                    end -= 1
            spans.append((start, end))
            start = end
            cur = sum(n_words(split_element(e)[1]) for e in doc.elements[start:i])
        cur += n
    if start < len(doc.elements):
        spans.append((start, len(doc.elements)))

    out: list[Chunk] = []
    heading, pos = "", 0
    for k, (s, e) in enumerate(spans):
        while pos < s:                               # the last heading above this chunk
            tag, text = split_element(doc.elements[pos])
            if tag in _HEADINGS:
                heading = text
            pos += 1
        out.append(Chunk(doc.index, k, s, e, doc.title, heading, doc.elements[s:e]))
    return out


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Question:
    id: str
    split: str                          # the benchmark's split: "train" | "dev"
    doc: int
    scenario: str
    question: str
    not_answerable: bool
    answers: tuple[tuple[str, tuple[str, ...]], ...]
    evidences: tuple[str, ...]

    @property
    def conditions(self) -> frozenset[str]:
        return frozenset(c for _, conds in self.answers for c in conds)

    @property
    def answer_evidence(self) -> frozenset[str]:
        """Evidence that is not itself a condition: where the answer is stated."""
        return frozenset(self.evidences) - self.conditions

    @property
    def kind(self) -> str:
        if self.not_answerable or not self.answers:
            return "unanswerable"
        return "conditional" if self.conditions else "unconditional"

    @property
    def answer_type(self) -> str:
        if self.kind == "unanswerable":
            return "none"
        return "yesno" if any(a.strip().lower() in ("yes", "no") for a, _ in self.answers) else "extractive"

    @property
    def opposite_outcomes(self) -> bool:
        """Both "yes" and "no" are gold: the closest thing here to a rule and its exception."""
        got = {a.strip().lower() for a, _ in self.answers}
        return {"yes", "no"} <= got

    def reference(self) -> list[list]:
        """The answers in the official scorer's format."""
        return [[a, list(conds)] for a, conds in self.answers]

    @property
    def retrieval_query(self) -> str:
        return f"{self.question} {self.scenario}".strip()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read(name: str):
    return json.loads(path_of("conditionalqa", name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def documents() -> tuple[Doc, ...]:
    return tuple(Doc(i, d["url"], d["title"], tuple(d["contents"])) for i, d in enumerate(_read("documents.json")))


@lru_cache(maxsize=None)
def _doc_by_url() -> dict[str, int]:
    return {d.url: d.index for d in documents()}


@lru_cache(maxsize=None)
def questions() -> tuple[Question, ...]:
    """Every labelled question (train and dev). The test split's answers are not public."""
    out: list[Question] = []
    urls = _doc_by_url()
    for split in ("train", "dev"):
        for q in _read(f"{split}.json"):
            out.append(Question(
                id=q["id"], split=split, doc=urls[q["url"]], scenario=q.get("scenario", ""),
                question=q["question"], not_answerable=bool(q.get("not_answerable")),
                answers=tuple((a, tuple(conds)) for a, conds in q.get("answers") or ()),
                evidences=tuple(q.get("evidences") or ())))
    return tuple(out)


@lru_cache(maxsize=None)
def chunks(words: int = DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> tuple[Chunk, ...]:
    """Every chunk of every document, in a stable order (the retrieval index order)."""
    return tuple(c for d in documents() for c in chunk_doc(d, words=words, strategy=strategy))


@lru_cache(maxsize=None)
def chunks_by_doc(words: int = DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> dict[int, tuple[Chunk, ...]]:
    out: dict[int, list[Chunk]] = {}
    for c in chunks(words, strategy):
        out.setdefault(c.doc, []).append(c)
    return {k: tuple(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Where the gold sits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GoldLocation:
    """The chunks of the question's page that hold its gold elements.

    An element string can occur more than once on a page (a repeated list item).
    Each gold string is assigned to **one** occurrence, preferring a chunk that
    already holds other gold, so separation is never overstated by an arbitrary
    choice between duplicates.
    """

    condition_chunks: frozenset[str]
    answer_chunks: frozenset[str]
    unlocated: tuple[str, ...] = ()
    string_chunk: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    @property
    def all_chunks(self) -> frozenset[str]:
        return self.condition_chunks | self.answer_chunks

    @property
    def condition_only_chunks(self) -> frozenset[str]:
        """Chunks that hold a condition and no other evidence."""
        return self.condition_chunks - self.answer_chunks

    @property
    def separated(self) -> bool:
        """The gold does not fit in one chunk."""
        return len(self.all_chunks) > 1


def locate_gold(q: Question, *, words: int = DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> GoldLocation:
    doc_chunks = chunks_by_doc(words, strategy)[q.doc]
    where: dict[str, list[str]] = {}
    for c in doc_chunks:
        for el in c.elements:
            where.setdefault(el, []).append(c.id)

    strings = sorted(set(q.evidences) | q.conditions)
    unlocated = tuple(s for s in strings if s not in where)
    chosen: dict[str, str] = {s: where[s][0] for s in strings if s in where and len(set(where[s])) == 1}
    for s in strings:                                 # duplicates: join gold that is already placed
        if s in where and s not in chosen:
            used = set(chosen.values())
            chosen[s] = next((cid for cid in where[s] if cid in used), where[s][0])

    return GoldLocation(
        condition_chunks=frozenset(chosen[s] for s in q.conditions if s in chosen),
        answer_chunks=frozenset(chosen[s] for s in q.answer_evidence if s in chosen),
        unlocated=unlocated, string_chunk=chosen)


# --------------------------------------------------------------------------- #
# The study's question sets
# --------------------------------------------------------------------------- #

SPLITS_FILE = Path(__file__).with_name("splits.json")
N_TUNING_DOCS = 40
N_UNCONDITIONAL = 400


def build_splits(*, seed: int = 0) -> dict:
    """The tuning and evaluation sets, disjoint by document.

    * **tuning** -- every question on ``N_TUNING_DOCS`` randomly chosen training
      pages. Prompts and thresholds are chosen here and nowhere else.
    * **eval** -- from the remaining pages: every conditional question, every
      unanswerable one, and ``N_UNCONDITIONAL`` unconditional ones sampled at
      random. No evaluated system is trained on the benchmark (the generator sees
      only four in-context demonstrations from the tuning pages), so its training
      questions are as unseen as its dev questions; using both is what
      gives the conditional stratum enough questions to test anything.

    Deterministic in ``seed``. The result is written to ``splits.json`` and
    committed: the sets are fixed before any system is run on them.
    """
    rng = random.Random(seed)
    qs = questions()
    train_docs = sorted({q.doc for q in qs if q.split == "train"})
    tuning_docs = set(rng.sample(train_docs, N_TUNING_DOCS))
    tuning = [q.id for q in qs if q.doc in tuning_docs]
    rest = [q for q in qs if q.doc not in tuning_docs]
    conditional = [q.id for q in rest if q.kind == "conditional"]
    unanswerable = [q.id for q in rest if q.kind == "unanswerable"]
    pool = sorted(q.id for q in rest if q.kind == "unconditional")
    unconditional = sorted(rng.sample(pool, min(N_UNCONDITIONAL, len(pool))))
    return {
        "seed": seed, "n_tuning_docs": N_TUNING_DOCS,
        "tuning": sorted(tuning),
        "eval": {"conditional": sorted(conditional), "unconditional": unconditional,
                 "unanswerable": sorted(unanswerable)},
    }


@lru_cache(maxsize=None)
def splits() -> dict:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"{SPLITS_FILE} is missing; run `python -m study.cqa --write-splits`")
    return json.loads(SPLITS_FILE.read_text(encoding="utf-8"))


def by_id() -> dict[str, Question]:
    return {q.id: q for q in questions()}


def eval_questions(strata: Iterable[str] = ("conditional", "unconditional", "unanswerable")) -> list[Question]:
    idx = by_id()
    return [idx[i] for s in strata for i in splits()["eval"][s]]


HOSTED_SIZES = {"conditional": 150, "unconditional": 100, "unanswerable": 50}


def hosted_questions(*, seed: int = 0) -> list[Question]:
    """The stratified evaluation subsample the hosted models answer (analysis plan, *Models*).

    Drawn from the evaluation set, so every hosted answer has a local
    counterpart on the same question.
    """
    rng = random.Random(seed)
    out: list[Question] = []
    for stratum, n in HOSTED_SIZES.items():
        ids = splits()["eval"][stratum]
        out += [by_id()[i] for i in sorted(rng.sample(ids, min(n, len(ids))))]
    return out


def tuning_questions() -> list[Question]:
    idx = by_id()
    return [idx[i] for i in splits()["tuning"]]


def folds(qs: Sequence[Question], k: int = 5, *, seed: int = 0) -> dict[int, int]:
    """``doc -> fold`` for document-disjoint cross-fitting."""
    docs = sorted({q.doc for q in qs})
    random.Random(seed).shuffle(docs)
    return {d: i % k for i, d in enumerate(docs)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-splits", action="store_true")
    args = ap.parse_args(argv)
    if args.write_splits:
        if SPLITS_FILE.exists():
            raise SystemExit(f"{SPLITS_FILE} already exists; the question sets are fixed. Delete it deliberately to redraw.")
        s = build_splits()
        SPLITS_FILE.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    s = splits()
    print(json.dumps({"tuning": len(s["tuning"]), **{k: len(v) for k, v in s["eval"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
