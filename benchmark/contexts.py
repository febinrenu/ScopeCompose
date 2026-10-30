"""Retrieval contexts of K passages: the gold pair plus hard negatives (proposal v8 section 6.4).

Two clean passages are not a RAG context, and a full-context generator handles them easily (the
first live call showed exactly that). The comparison between systems only means something when the
exception sits among distractors, so each instance is expanded to K passages: the gold passages
plus *hard negatives* -- same-provider text that is on topic but is not the rule or its exception.

Design decisions that keep the expansion honest:

* **Length-matched.** Gold passages are sentence-sized; distractors cut at paragraph size would let a
  model (or a retriever) pick the gold out by length alone. Distractors are chunked to the gold
  passages' own length.
* **Order-shuffled, deterministically.** The gold passages must not always be first. The shuffle is
  seeded per instance, so a context is reproducible and no system sees a gold-first order.
* **Hard, not random.** Negatives are ranked by BM25 against the *query*, from the same provider's
  other documents -- the passages a real retriever would also return.
* **Screened, never silently trimmed.** A negative that itself states a rule bearing on the question
  could make the gold answer wrong (an unannotated exception hiding in the distractors). The evaluation
  model flags such negatives; flagged ones are **kept in the context and listed for a person to
  adjudicate**. Removing one is an explicit act (:func:`drop_adjudicated`), never a side effect.
* **Gold inclusion is checked.** Every gold passage must be in the context; the function refuses to
  return one that is not.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from benchmark.chunker import Chunk, chunk_text
from benchmark.vacs import lexical_overlap
from contract.gold import GoldInstance
from contract.models import Passage, SourceType

DEFAULT_K = 6
#: A negative this similar to a gold passage is a duplicate of it, not a distractor.
DUPLICATE_OVERLAP = 0.6
MIN_TARGET_WORDS = 25

#: ``(query, chunk text) -> True`` if the chunk states something that bears on the answer.
Screener = Callable[[str, str], bool]


@dataclass
class ContextResult:
    instance: GoldInstance
    negatives: list[str] = field(default_factory=list)
    """Passage ids of the added negatives."""
    flagged: list[str] = field(default_factory=list)
    """Negatives the screener thinks bear on the answer. Kept; a person adjudicates."""

    @property
    def k(self) -> int:
        return len(self.instance.passages)


def target_words(passages: Sequence[Passage]) -> int:
    words = [len(p.text.split()) for p in passages]
    return max(MIN_TARGET_WORDS, round(sum(words) / len(words))) if words else MIN_TARGET_WORDS


def candidate_chunks(docs: Sequence[dict], gold_doc_ids: set[str], gold_texts: Sequence[str], *, words: int
                     ) -> list[Chunk]:
    """Length-matched chunks of the pool's documents, minus the gold documents and gold duplicates."""
    out: list[Chunk] = []
    for d in docs:
        if d["doc_id"] in gold_doc_ids:
            continue
        for c in chunk_text(d["doc_id"], d.get("text", ""), target_words=words, min_words=max(8, words // 3)):
            if any(lexical_overlap(c.text, g) >= DUPLICATE_OVERLAP for g in gold_texts):
                continue
            out.append(c)
    return out


def rank_by_bm25(query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
    """Chunks best-first against the query. Ties break by chunk id, so the order is deterministic."""
    if not chunks:
        return []
    from rank_bm25 import BM25Okapi

    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())  # noqa: E731
    bm = BM25Okapi([tok(c.text) for c in chunks])
    scores = bm.get_scores(tok(query))
    return [c for _, c in sorted(zip((-s for s in scores), chunks), key=lambda x: (x[0], x[1].id))]


def _source_type(doc: dict) -> SourceType:
    kind = (doc.get("kind") or "").lower()
    if "answer" in kind or "guide" in kind or "transaction" in kind:
        return SourceType.GOVERNMENT_GUIDANCE
    return SourceType.UNKNOWN


def build_context(
    inst: GoldInstance,
    pool_docs: Sequence[dict],
    *,
    k: int = DEFAULT_K,
    seed: int = 0,
    screener: Screener | None = None,
) -> ContextResult:
    """Expand ``inst`` to ``k`` passages. Gold passages keep their ids; negatives are ``n0, n1, ...``."""
    gold = list(inst.passages)
    if k < len(gold):
        raise ValueError(f"k={k} is smaller than the {len(gold)} gold passages")
    need = k - len(gold)
    gold_doc_ids = {p.document_id for p in gold if p.document_id}
    words = target_words(gold)
    ranked = rank_by_bm25(inst.query, candidate_chunks(pool_docs, gold_doc_ids, [p.text for p in gold], words=words))
    by_doc = {d["doc_id"]: d for d in pool_docs}

    chosen: list[Chunk] = []
    per_doc: dict[str, int] = {}
    for c in ranked:                                   # at most two per document: spread, not one dominant page
        if len(chosen) == need:
            break
        if per_doc.get(c.doc_id, 0) >= 2:
            continue
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
        chosen.append(c)
    if len(chosen) < need:
        # relax the per-document cap rather than return a short context
        for c in ranked:
            if len(chosen) == need:
                break
            if c not in chosen:
                chosen.append(c)

    negatives, flagged = [], []
    new: list[Passage] = []
    for i, c in enumerate(chosen):
        d = by_doc[c.doc_id]
        pid = f"n{i}"
        new.append(Passage(id=pid, text=c.text, source_type=_source_type(d), date=None,
                           source_url=d.get("url"), document_id=d.get("doc_id")))
        negatives.append(pid)
        if screener is not None and screener(inst.query, c.text):
            flagged.append(pid)

    allp = gold + new
    rng = random.Random(f"{inst.instance_id}|{seed}")
    rng.shuffle(allp)
    out = inst.model_copy(update={"passages": allp})
    have = {p.id for p in out.passages}
    missing = [p.id for p in gold if p.id not in have]
    if missing:
        raise AssertionError(f"{inst.instance_id}: gold passage(s) {missing} are not in the context")
    return ContextResult(out, negatives, flagged)


def drop_adjudicated(result: ContextResult, drop: Sequence[str]) -> ContextResult:
    """Remove negatives a person has confirmed state something that bears on the answer.

    Only a negative that the screener flagged can be dropped, and only by naming it: this is the one
    place a distractor is ever removed, and it leaves a record in the result."""
    bad = [d for d in drop if d not in result.flagged]
    if bad:
        raise ValueError(f"{bad} were not flagged; only flagged negatives can be dropped, and only after adjudication")
    keep = [p for p in result.instance.passages if p.id not in set(drop)]
    return ContextResult(result.instance.model_copy(update={"passages": keep}),
                         [n for n in result.negatives if n not in set(drop)],
                         [f for f in result.flagged if f not in set(drop)])


SCREEN_SYSTEM = (
    "You decide whether a passage states a rule, condition or outcome that bears on a question, i.e. "
    "something that could change or qualify the answer for some people. Return ONLY JSON: "
    '{"bears_on_answer": true or false}.'
)


def eval_screener(client=None) -> Screener:
    """A screener backed by the EVAL tier (never an evaluated model: it can reach the test set)."""
    from api_budget import Tier, get_client

    c = client or get_client()

    def screen(query: str, text: str) -> bool:
        res = c.complete(messages=[{"role": "user", "content": f"Question: {query}\n\nPassage: {text}"}],
                         system=SCREEN_SYSTEM, tier=Tier.EVAL, step="context_screen",
                         response_format={"type": "json_object"})
        try:
            return bool(json.loads(res.text).get("bears_on_answer"))
        except (ValueError, TypeError, AttributeError):
            return True        # an unreadable verdict is flagged for a person, not waved through

    return screen
