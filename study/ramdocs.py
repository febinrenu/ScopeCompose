"""RAMDocs: the contrast regime, where discarding evidence is the right operation.

RAMDocs (Wang et al., 2025, "Retrieval-Augmented Generation with Conflicting
Evidence") gives each question a set of documents labelled ``correct``,
``misinfo`` or ``noise``, the gold answers, and the wrong answers the
misinformation supports. A response is correct only if it contains every gold
answer and no wrong one.

It is the mirror image of ConditionalQA for this study. There, every annotated
passage is true and the risk is dropping one; here, some passages are false and
the risk is keeping them. The document labels make an **oracle discard** system
possible -- the ceiling for any operator that removes evidence.

Caveats, stated wherever these numbers are reported: 400 of the 500 questions
have two or three valid answers (same-name entities), so the benchmark is a
mixed regime rather than a pure selection task; the misinformation is synthetic
(entity substitution); there is no training split.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from functools import lru_cache

from study.fetch_data import path_of


@dataclass(frozen=True)
class RDoc:
    id: str            # "r17d3": question 17, document 3
    text: str
    type: str          # "correct" | "misinfo" | "noise"
    answer: str


@dataclass(frozen=True)
class RQuestion:
    id: str
    question: str
    documents: tuple[RDoc, ...]
    gold_answers: tuple[str, ...]
    wrong_answers: tuple[str, ...]

    @property
    def has_misinfo(self) -> bool:
        return any(d.type == "misinfo" for d in self.documents)

    @property
    def has_noise(self) -> bool:
        return any(d.type == "noise" for d in self.documents)

    @property
    def regime(self) -> str:
        """``single`` (one valid answer) or ``multi`` (several same-name entities)."""
        return "single" if len(self.gold_answers) == 1 else "multi"


@lru_cache(maxsize=None)
def questions() -> tuple[RQuestion, ...]:
    out: list[RQuestion] = []
    lines = path_of("ramdocs", "RAMDocs_test.jsonl").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(l for l in lines if l.strip()):
        r = json.loads(line)
        docs = tuple(RDoc(f"r{i}d{j}", d["text"], d["type"], str(d.get("answer", "")))
                     for j, d in enumerate(r["documents"]))
        out.append(RQuestion(f"ram-{i}", r["question"], docs, tuple(map(str, r["gold_answers"])),
                             tuple(map(str, r.get("wrong_answers") or ()))))
    return tuple(out)


def by_id() -> dict[str, RQuestion]:
    return {q.id: q for q in questions()}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """SQuAD-style: lowercase, no punctuation, no articles, single spaces."""
    return " ".join(_ARTICLES.sub(" ", text.lower().translate(_PUNCT)).split())


def contains(response: str, answer: str) -> bool:
    a = normalise(answer)
    return bool(a) and f" {a} " in f" {normalise(response)} "


@dataclass(frozen=True)
class RScore:
    correct: bool          # every gold answer present and no wrong answer present
    gold_recall: float
    any_wrong: bool


def score(response: str, q: RQuestion) -> RScore:
    found = [contains(response, a) for a in q.gold_answers]
    wrong = any(contains(response, a) for a in q.wrong_answers)
    return RScore(all(found) and not wrong, sum(found) / len(found) if found else 0.0, wrong)
