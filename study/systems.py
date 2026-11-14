"""The systems: each one is a rule for which evidence reaches the generator.

A system maps a question to a :class:`Context` -- an ordered list of chunks (or
none, for the closed-book control). The generator, its prompt and its output
contract are the same for all of them (``study.reader``).

Three families:

**The oracle ladder** isolates stages. Each rung removes one thing:

==================  =========================================================
``closed``          no evidence: what the model answers from memory
``gold``            exactly the chunks holding the gold evidence and conditions
``gold_nocondel``   the same chunks with the gold condition *elements* deleted
``gold_nocondchk``  the gold chunks minus those holding a condition and no
                    other evidence -- what a pipeline delivers when it finds the
                    answer and loses the separately chunked condition
``goldpad8``        the gold chunks plus top-ranked non-gold chunks up to 8:
                    perfect recall under realistic distraction
``top{k}``          the retriever's top k: the ordinary pipeline
==================  =========================================================

**Evidence-reducing operators** and **remedies** take ``top{k}`` and change it;
they are registered by ``study.operators``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from study import cqa, retrieve

DEFAULT_K = 8


@dataclass(frozen=True)
class Context:
    chunks: tuple[cqa.Chunk, ...] | None           # None = closed book
    drop_elements: frozenset[str] = frozenset()
    trace: dict = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.id for c in self.chunks or ()]


@dataclass(frozen=True)
class Env:
    """What a system may look at: one chunking and one ranker's saved lists."""

    words: int = cqa.DEFAULT_CHUNK_WORDS
    strategy: str = "fixed"
    ranker: str = "hybrid"
    #: The cell's own model, for operators that ask it something. Not part of
    #: the evidence configuration, so excluded from comparisons and hashing.
    client: object | None = field(default=None, compare=False, repr=False)
    tier: object | None = field(default=None, compare=False, repr=False)

    def ranked(self, q: cqa.Question) -> list[cqa.Chunk]:
        idx = retrieve.chunk_index(self.words, self.strategy)
        return [idx[i] for i in retrieve.load_rankings(self.ranker, self.words, self.strategy)[q.id]]

    def gold(self, q: cqa.Question) -> cqa.GoldLocation:
        return cqa.locate_gold(q, words=self.words, strategy=self.strategy)

    def chunk(self, chunk_id: str) -> cqa.Chunk:
        return retrieve.chunk_index(self.words, self.strategy)[chunk_id]

    def doc_chunks(self, doc: int) -> tuple[cqa.Chunk, ...]:
        return cqa.chunks_by_doc(self.words, self.strategy)[doc]


Builder = Callable[[cqa.Question, Env], Context]
SYSTEMS: dict[str, Builder] = {}
#: Systems whose context is built from the gold annotation. They are defined
#: only for answerable questions and are never a deployable pipeline.
ORACLE: set[str] = set()


def register(name: str, *, oracle: bool = False):
    def deco(fn: Builder) -> Builder:
        if name in SYSTEMS:
            raise ValueError(f"system {name!r} is already registered")
        SYSTEMS[name] = fn
        if oracle:
            ORACLE.add(name)
        return fn
    return deco


def _in_doc_order(chunks) -> tuple[cqa.Chunk, ...]:
    return tuple(sorted(chunks, key=lambda c: (c.doc, c.index)))


# --------------------------------------------------------------------------- #
# The oracle ladder
# --------------------------------------------------------------------------- #


@register("closed")
def closed(q: cqa.Question, env: Env) -> Context:
    return Context(None)


@register("gold", oracle=True)
def gold(q: cqa.Question, env: Env) -> Context:
    g = env.gold(q)
    return Context(_in_doc_order(env.chunk(i) for i in g.all_chunks))


@register("gold_nocondel", oracle=True)
def gold_without_condition_elements(q: cqa.Question, env: Env) -> Context:
    g = env.gold(q)
    return Context(_in_doc_order(env.chunk(i) for i in g.all_chunks), drop_elements=frozenset(q.conditions))


@register("gold_nocondchk", oracle=True)
def gold_without_condition_only_chunks(q: cqa.Question, env: Env) -> Context:
    g = env.gold(q)
    keep = g.all_chunks - g.condition_only_chunks
    return Context(_in_doc_order(env.chunk(i) for i in keep), trace={"removed": sorted(g.condition_only_chunks)})


def _gold_padded(k: int) -> Builder:
    def build(q: cqa.Question, env: Env) -> Context:
        g = env.gold(q)
        ranked = env.ranked(q)
        kept = [c for c in ranked if c.id in g.all_chunks]
        kept += [env.chunk(i) for i in sorted(g.all_chunks - {c.id for c in kept})]   # gold the retriever missed
        fill = [c for c in ranked if c.id not in g.all_chunks][: max(0, k - len(kept))]
        order = {c.id: r for r, c in enumerate(ranked)}
        chunks = sorted(kept + fill, key=lambda c: order.get(c.id, len(order)))
        return Context(tuple(chunks), trace={"gold": sorted(g.all_chunks)})
    return build


def _top(k: int) -> Builder:
    def build(q: cqa.Question, env: Env) -> Context:
        return Context(tuple(env.ranked(q)[:k]))
    return build


register(f"goldpad{DEFAULT_K}", oracle=True)(_gold_padded(DEFAULT_K))
for _k in (3, 5, 8, 12, 20):
    register(f"top{_k}")(_top(_k))


def build(system: str, q: cqa.Question, env: Env) -> Context:
    if system not in SYSTEMS:
        raise KeyError(f"unknown system {system!r}; available: {sorted(SYSTEMS)}")
    return SYSTEMS[system](q, env)
