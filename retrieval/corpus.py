"""Passage store.

Holds the passages a retriever searches over, together with the source
metadata that downstream stages need: ``source_type`` and ``date`` drive
temporal handling and the credibility-selection baseline, and ``document_id``
is what makes a Tier-1 "these came from separate documents" claim checkable
rather than asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from contract.models import Passage, SourceType


@dataclass
class Corpus:
    """An in-memory passage collection.

    Small by design. The benchmark targets 250-350 instances with a handful of
    passages each, so nothing here needs to scale beyond tens of thousands of
    passages, and a list plus a dict beats a vector database at that size.
    """

    passages: list[Passage] = field(default_factory=list)
    _by_id: dict[str, Passage] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {p.id: p for p in self.passages}
        if len(self._by_id) != len(self.passages):
            seen: set[str] = set()
            dupes = {p.id for p in self.passages if p.id in seen or seen.add(p.id)}
            raise ValueError(f"duplicate passage ids in corpus: {sorted(dupes)}")

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Corpus:
        passages = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    passages.append(Passage.model_validate_json(line))
        return cls(passages)

    @classmethod
    def from_records(cls, records: Iterable) -> Corpus:
        """Build from an iterable of QueryRecords or GoldInstances.

        Passage ids are namespaced by the record id, because ``p0`` means
        different things in different records and a flat corpus would collide
        them.
        """
        out: list[Passage] = []
        seen: set[str] = set()
        for rec in records:
            rid = getattr(rec, "query_id", None) or getattr(rec, "instance_id")
            for p in rec.passages:
                pid = f"{rid}::{p.id}"
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(p.model_copy(update={"id": pid}))
        return cls(out)

    @classmethod
    def from_texts(cls, texts: Iterable[str], *, prefix: str = "p") -> Corpus:
        """Quick corpus from bare strings. For tests and smoke runs."""
        return cls([
            Passage(id=f"{prefix}{i}", text=t, source_type=SourceType.UNKNOWN)
            for i, t in enumerate(texts)
        ])

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
            for p in self.passages:
                fh.write(p.model_dump_json() + "\n")

    # -- access -------------------------------------------------------------- #

    def add(self, passage: Passage) -> None:
        if passage.id in self._by_id:
            raise ValueError(f"passage id {passage.id!r} already in corpus")
        self.passages.append(passage)
        self._by_id[passage.id] = passage

    def get(self, passage_id: str) -> Passage:
        try:
            return self._by_id[passage_id]
        except KeyError:
            raise KeyError(f"no passage {passage_id!r} in corpus") from None

    @property
    def texts(self) -> list[str]:
        return [p.text for p in self.passages]

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.passages]

    def __len__(self) -> int:
        return len(self.passages)

    def __iter__(self) -> Iterator[Passage]:
        return iter(self.passages)

    def __getitem__(self, i: int) -> Passage:
        return self.passages[i]


def dump_corpus_stats(corpus: Corpus) -> dict[str, object]:
    by_source: dict[str, int] = {}
    n_dated = 0
    n_doc_ids = 0
    for p in corpus:
        by_source[p.source_type.value] = by_source.get(p.source_type.value, 0) + 1
        n_dated += int(p.date is not None)
        n_doc_ids += int(p.document_id is not None)
    lengths = [len(p.text.split()) for p in corpus] or [0]
    return {
        "passages": len(corpus),
        "by_source_type": by_source,
        "with_date": n_dated,
        "with_document_id": n_doc_ids,
        "mean_words": round(sum(lengths) / len(lengths), 1),
        "max_words": max(lengths),
    }
