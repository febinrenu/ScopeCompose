"""One sentence-aligned chunker for every document (proposal v8 section 6.4).

Retrieval contexts, mining candidates and gold passages must all be cut the same
way, or "the chunk that contains the gold sentence" means different things in
different places. This is the single implementation.

Chunks are built by accumulating whole sentences up to ``target_words``; a sentence
is **never split**, because a rule cut in half is two non-rules. A sentence longer
than the target becomes a chunk of its own. A short trailing remainder is merged
into the previous chunk instead of standing alone, so no chunk is a fragment.

``gold_chunk`` finds the chunk containing a sentence, with whitespace normalised --
the "100% gold inclusion" check in the plan reduces to this returning a chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_TARGET_WORDS = 140
DEFAULT_MIN_WORDS = 50

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“(])")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}#c{self.index}"

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def split_sentences(text: str) -> list[str]:
    """Sentences, honouring the one-block-per-line layout of the fetched documents."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if line:
            out += [s.strip() for s in _SENT_SPLIT.split(line) if s.strip()]
    return out


def chunk_text(doc_id: str, text: str, *, target_words: int = DEFAULT_TARGET_WORDS,
               min_words: int = DEFAULT_MIN_WORDS) -> list[Chunk]:
    if target_words < 1 or min_words < 0:
        raise ValueError("target_words must be positive and min_words non-negative")
    groups: list[list[str]] = []
    cur: list[str] = []
    words = 0
    for s in split_sentences(text):
        n = len(s.split())
        if cur and words + n > target_words:
            groups.append(cur)
            cur, words = [], 0
        cur.append(s)
        words += n
    if cur:
        groups.append(cur)
    # a short remainder joins the previous chunk
    if len(groups) > 1 and sum(len(s.split()) for s in groups[-1]) < min_words:
        tail = groups.pop()      # (not `groups[-2] += groups.pop()`: the index is resolved after the pop)
        groups[-1] += tail
    return [Chunk(doc_id, i, " ".join(g)) for i, g in enumerate(groups)]


def _norm(s: str) -> str:
    return " ".join(s.split())


def gold_chunk(chunks: list[Chunk], sentence: str) -> Chunk | None:
    """The chunk containing ``sentence`` verbatim (whitespace-insensitive), or ``None``."""
    want = _norm(sentence).rstrip(".")
    if not want:
        return None
    for c in chunks:
        if want in _norm(c.text):
            return c
    return None
