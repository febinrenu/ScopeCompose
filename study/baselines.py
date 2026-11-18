"""Context-preserving retrieval baselines (revision, round 6; exploratory).

Does an established retrieval design already keep conditions with their answers? Four common
designs, each built from the study's own retriever (BM25 and Contriever fused by reciprocal rank),
each given the same evidence budget as ``top8``: per question, at most the number of words of that
question's top 8 chunks (the hierarchical and expanded-query variants pass 8 chunks).

``sentwin``      sentence window: rank single elements (on gov.uk, a sentence, list item or row),
                 and pass each hit with its two neighbours on either side, merged per page
``parentchild``  parent-child (small-to-big): rank single elements, pass the top-level section
                 (heading to next top-level heading) that holds each hit; a section larger than
                 the budget left is cut to the window around the hit that fits
``hier``         hierarchical: rank pages first (BM25 over the page, Contriever over its start),
                 then the chunks of the top two pages by the chunk-level hybrid score
``qexp``         query expansion with condition terms: the 150-word chunks, retrieved with the
                 question, the scenario and a fixed list of condition cue words

``hier`` and ``qexp`` are saved as rankings of the ordinary chunks, so they run as ``top8`` under
their own ranker name; ``sentwin`` and ``parentchild`` pass spans of elements, recorded with ids
``d<page>e<start>-<end>``.

    python -m study.baselines rank       # element, page and chunk rankings (GPU for the dense half)
    python -m study.baselines recall     # model-free: which gold elements each design passes on
    python -m study.baselines run        # end to end, both local generators, hosted subsample (answerable)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from contract.models import Passage
from retrieval.corpus import Corpus
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.pipeline import HybridRetriever
from study import cqa, systems
from study.retrieve import ARTIFACTS, _POOL, rankings_path, retriever

WINDOW = 2                    # sentence window: neighbours on each side
TOP_PAGES = 2                 # hierarchical: pages kept (as completion's candidate pages)
ELEMENT_DEPTH = 120
EXPANSION = "eligibility requirements conditions exceptions: only if, unless, must, provided that, except"
_TOP_LEVEL = ("h1", "h2")
_SPAN_RE = re.compile(r"^d(\d+)e(\d+)-(\d+)$")


# --------------------------------------------------------------------------- #
# Spans of elements
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Span(cqa.Chunk):
    """Elements ``[start, end)`` of one page, passed as one block."""

    @property
    def id(self) -> str:
        return f"d{self.doc}e{self.start}-{self.end}"


def span_elements(span_id: str) -> tuple[str, ...] | None:
    """The elements behind a span id, or None if ``span_id`` is an ordinary chunk id."""
    m = _SPAN_RE.match(span_id)
    if not m:
        return None
    doc, s, e = (int(x) for x in m.groups())
    return cqa.documents()[doc].elements[s:e]


def _words(doc: int, i: int) -> int:
    return cqa.n_words(cqa.split_element(cqa.documents()[doc].elements[i])[1])


@lru_cache(maxsize=None)
def _headings(doc: int) -> tuple[str, ...]:
    """The nearest heading at or above each element."""
    d, cur, out = cqa.documents()[doc], "", []
    for el in d.elements:
        tag, text = cqa.split_element(el)
        if tag in ("h1", "h2", "h3", "h4"):
            cur = text
        out.append(cur)
    return tuple(out)


@lru_cache(maxsize=None)
def _sections(doc: int) -> tuple[tuple[int, int], ...]:
    """Top-level sections: each starts at an h1/h2 element (or the page start)."""
    els = cqa.documents()[doc].elements
    starts = [0] + [i for i, el in enumerate(els) if i and cqa.split_element(el)[0] in _TOP_LEVEL]
    return tuple(zip(starts, starts[1:] + [len(els)]))


def _spans(picked: dict[int, set[int]], page_order: list[int]) -> tuple[Span, ...]:
    out: list[Span] = []
    for doc in page_order:
        idx = sorted(picked[doc])
        runs: list[list[int]] = []
        for i in idx:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        d = cqa.documents()[doc]
        for r in runs:
            out.append(Span(doc, -1, r[0], r[-1] + 1, d.title, _headings(doc)[r[0]], d.elements[r[0]:r[-1] + 1]))
    return tuple(out)


def budget_words(q: cqa.Question, env: systems.Env) -> int:
    return sum(c.n_words for c in env.ranked(q)[:systems.DEFAULT_K])


# --------------------------------------------------------------------------- #
# Indexes and rankings
# --------------------------------------------------------------------------- #


def element_units() -> list[tuple[str, str]]:
    """``(d<page>e<i>, text)``: each element, prefixed with its page title and heading like a chunk."""
    out = []
    for d in cqa.documents():
        heads = _headings(d.index)
        for i, el in enumerate(d.elements):
            h = heads[i]
            head = d.title if not h or h == d.title else f"{d.title} - {h}"
            out.append((f"d{d.index}e{i}", f"{head}\n{cqa._render(el)}"))
    return out


def _hybrid(r: HybridRetriever, query: str, depth: int, pool: int = _POOL) -> list[str]:
    lists = [r.sparse.search(query, k=pool), r.dense.search(query, k=pool)]
    return [f.passage_id for f in reciprocal_rank_fusion(lists, k=r.rrf_k, top_k=depth)]


def build_rankings() -> None:
    qs = cqa.eval_questions() + cqa.tuning_questions()
    # elements
    units = element_units()
    er = HybridRetriever(Corpus([Passage(id=i, text=t) for i, t in units]),
                         embedding_cache=ARTIFACTS / "embeddings" / "cqa_elements.npz")
    _save("elements", {q.id: _hybrid(er, q.retrieval_query, ELEMENT_DEPTH, pool=400) for q in qs})
    # pages, then chunks within the top pages
    docs = cqa.documents()
    pr = HybridRetriever(Corpus([Passage(id=d.id, text=d.title + "\n" + "\n".join(cqa._render(e) for e in d.elements))
                                 for d in docs]), embedding_cache=ARTIFACTS / "embeddings" / "cqa_pages.npz")
    cr = retriever(cqa.DEFAULT_CHUNK_WORDS, "fixed")
    n_chunks = len(cqa.chunks())
    hier: dict[str, list[str]] = {}
    pages: dict[str, list[str]] = {}
    for q in qs:
        top_pages = _hybrid(pr, q.retrieval_query, TOP_PAGES, pool=len(docs))
        pages[q.id] = top_pages
        keep = {int(p[1:]) for p in top_pages}
        full = _hybrid(cr, q.retrieval_query, n_chunks, pool=n_chunks)
        hier[q.id] = [cid for cid in full if int(cid[1:].split("c")[0]) in keep]
    _save("pages", pages)
    _save_chunks("hier", hier)
    # expanded query over the ordinary chunks
    _save_chunks("qexp", {q.id: _hybrid(cr, f"{q.retrieval_query} {EXPANSION}", 50) for q in qs})


def _save(name: str, ranking: dict) -> None:
    path = ARTIFACTS / "rankings" / f"hybrid_{name}.json"
    path.write_text(json.dumps(ranking), encoding="utf-8")
    print(f"{path.name}: {len(ranking)} questions", flush=True)


def _save_chunks(ranker: str, ranking: dict) -> None:
    path = rankings_path(ranker, cqa.DEFAULT_CHUNK_WORDS, "fixed")
    path.write_text(json.dumps(ranking), encoding="utf-8")
    print(f"{path.name}: {len(ranking)} questions", flush=True)


@lru_cache(maxsize=None)
def element_ranking() -> dict[str, list[str]]:
    return json.loads((ARTIFACTS / "rankings" / "hybrid_elements.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The two span systems
# --------------------------------------------------------------------------- #


def _fill(q: cqa.Question, env: systems.Env, unit_for) -> systems.Context:
    """Walk the element ranking; add ``unit_for(doc, i, room)`` (a set of element indices) while it fits."""
    budget = budget_words(q, env)
    picked: dict[int, set[int]] = {}
    order: list[int] = []
    used = 0
    for eid in element_ranking()[q.id]:
        doc, i = (int(x) for x in eid[1:].split("e"))
        have = picked.get(doc, set())
        if i in have:
            continue
        unit = unit_for(doc, i, budget - used, have)
        new = [j for j in unit if j not in have]
        w = sum(_words(doc, j) for j in new)
        if not new or used + w > budget:
            continue
        picked.setdefault(doc, set()).update(new)
        if doc not in order:
            order.append(doc)
        used += w
    return systems.Context(_spans(picked, order), trace={"budget_words": budget, "words": used})


def _window(doc: int, i: int, room: int, have: set[int]) -> list[int]:
    n = len(cqa.documents()[doc].elements)
    return list(range(max(0, i - WINDOW), min(n, i + WINDOW + 1)))


def _parent(doc: int, i: int, room: int, have: set[int]) -> list[int]:
    s, e = next((s, e) for s, e in _sections(doc) if s <= i < e)
    whole = list(range(s, e))
    if sum(_words(doc, j) for j in whole if j not in have) <= room:
        return whole
    # too large for what is left: the widest window around the hit, inside the section, that fits
    lo = hi = i
    w = _words(doc, i) if i not in have else 0
    while True:
        grew = False
        for j in (hi + 1, lo - 1):
            if s <= j < e and (j > hi or j < lo):
                add = _words(doc, j) if j not in have else 0
                if w + add <= room:
                    w += add
                    lo, hi = min(lo, j), max(hi, j)
                    grew = True
        if not grew:
            return list(range(lo, hi + 1))


@systems.register("sentwin")
def sentence_window(q: cqa.Question, env: systems.Env) -> systems.Context:
    return _fill(q, env, _window)


@systems.register("parentchild")
def parent_child(q: cqa.Question, env: systems.Env) -> systems.Context:
    return _fill(q, env, _parent)


HIER_ENV = systems.Env(cqa.DEFAULT_CHUNK_WORDS, "fixed", "hier")
QEXP_ENV = systems.Env(cqa.DEFAULT_CHUNK_WORDS, "fixed", "qexp")
#: name -> (system, env); every one compared with top8 under the default env
DESIGNS = {"sentwin": ("sentwin", systems.Env()), "parentchild": ("parentchild", systems.Env()),
           "hier": ("top8", HIER_ENV), "qexp": ("top8", QEXP_ENV)}


# --------------------------------------------------------------------------- #
# Model-free: which gold elements reach the generator
# --------------------------------------------------------------------------- #


def context_elements(ctx: systems.Context) -> set[str]:
    return {el for c in ctx.chunks or () for el in c.elements}


def recall() -> dict:
    """Per design, over the conditional evaluation questions: the share of gold condition and answer
    elements in the context, the share of questions whose whole gold is in it, and the words passed."""
    base = systems.Env()
    rows = {"top8": ("top8", base), "top8_section": ("top8", systems.Env(150, "section", "hybrid")), **DESIGNS}
    out: dict = {}
    qs = cqa.eval_questions(["conditional"])
    for name, (sysname, env) in rows.items():
        cond = cond_tot = ans = ans_tot = complete = words = 0
        for q in qs:
            shown = context_elements(systems.build(sysname, q, env))
            c, a = set(q.conditions), set(q.answer_evidence)
            cond, cond_tot = cond + len(c & shown), cond_tot + len(c)
            ans, ans_tot = ans + len(a & shown), ans_tot + len(a)
            complete += (c | a) <= shown
            words += sum(cqa.n_words(cqa.split_element(e)[1]) for e in shown)
        out[name] = {"cond_recall": cond / cond_tot, "answer_recall": ans / ans_tot,
                     "complete": complete / len(qs), "words": words / len(qs), "n": len(qs)}
    return out


def run() -> int:
    from study.run import run_cell

    for m in ("qwen7b", "llama8b"):
        for other in ("scopecompose-qwen7b", "scopecompose-reader"):
            subprocess.run(["ollama", "stop", other], capture_output=True)
        for name, (s, env) in DESIGNS.items():
            print(json.dumps(run_cell(s, m, "eval", env=env, questions=E2E_QUESTIONS(), progress=True)), flush=True)
    return 0


def E2E_QUESTIONS() -> list[cqa.Question]:
    """End to end, the designs answer the hosted subsample's answerable questions (their longer contexts
    take about three times as long to answer as the top 8's); the model-free comparison covers every
    conditional evaluation question."""
    return [q for q in cqa.hosted_questions() if q.kind != "unanswerable"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("rank", "recall", "run"))
    args = ap.parse_args(argv)
    if args.action == "rank":
        build_rankings()
    elif args.action == "recall":
        print(json.dumps(recall(), indent=1))
    else:
        return run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
