"""The evidence-reducing operators on RAMDocs: the regime where discarding is right.

    python -m study.ram --system full oracle rr3 nli crs --model qwen7b

**Document order.** RAMDocs lists a question's documents with the correct
ones first (476 of 500 questions) and is fully sorted correct, misinformation,
noise in 285. An operator that drops "the lower-ranked" passage would read
the answer key from that order. Every system here therefore sees the documents
**in reranker order** -- what a retrieve-then-rerank pipeline would hand on --
and the dataset's own order is never used.

The generator prompt is neutral: it does not tell the model that some
documents are false. Asking it to discard misinformation would make the
plain full-context system a conflict-handling system, and the comparison
is between operators that remove documents and a generator that sees them all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import settings
from api_budget import BackendError, TruncatedResponseError
from api_budget.client import LLMClient, Tier, get_client
from study import ramdocs
from study.operators import (P, Reduction, conflictrag_select, llm_grade, llm_select, load_store, nli_filter)
from study.run import MODELS, model_id

RUNS = settings.REPO_ROOT / "study" / "runs" / "ramdocs"

SYSTEM = """\
Answer the question using the documents provided. If the question is ambiguous and the documents describe different people, places or things with the same name, give the answer for each of them.
Reply in one or two sentences."""


def prompt_sha() -> str:
    import hashlib

    return hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def ordered(qid: str) -> tuple[P, ...]:
    """The question's documents in reranker order (ties by dataset position)."""
    q = ramdocs.by_id()[qid]
    rec = load_store("ramdocs", "all")[qid]
    score = dict(zip(rec["ids"], rec["rerank"]))
    docs = sorted(q.documents, key=lambda d: (-score[d.id], d.id))
    return tuple(P(d.id, d.text) for d in docs)


def _matrix_in(order: Sequence[P], qid: str) -> list[list[float]]:
    rec = load_store("ramdocs", "all")[qid]
    pos = {pid: i for i, pid in enumerate(rec["ids"])}
    m = rec["contradiction"]
    return [[m[pos[a.id]][pos[b.id]] for b in order] for a in order]


Builder = Callable[[ramdocs.RQuestion, LLMClient, Tier], Reduction]


def _full(q, client, tier) -> Reduction:
    return Reduction(tuple(p.id for p in ordered(q.id)))


def _oracle(q, client, tier) -> Reduction:
    keep = {d.id for d in q.documents if d.type == "correct"}
    return Reduction(tuple(p.id for p in ordered(q.id) if p.id in keep))


def _rr(keep: int) -> Builder:
    def build(q, client, tier) -> Reduction:
        return Reduction(tuple(p.id for p in ordered(q.id)[:keep]))
    return build


def _nli(q, client, tier) -> Reduction:
    docs = ordered(q.id)
    return nli_filter(docs, _matrix_in(docs, q.id))


def _crs(q, client, tier) -> Reduction:
    docs = ordered(q.id)
    return conflictrag_select(docs, _matrix_in(docs, q.id))


SYSTEMS: dict[str, Builder] = {
    "full": _full,
    "oracle": _oracle,
    "rr3": _rr(3),
    "nli": _nli,
    "crs": _crs,
    "grade": lambda q, c, t: llm_grade(c, t, q.question, ordered(q.id)),
    "select": lambda q, c, t: llm_select(c, t, q.question, ordered(q.id)),
}


def cell_path(system: str, model_key: str) -> Path:
    return RUNS / f"{system}__{model_key}.jsonl"


def read_cell(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["qid"]] = r
    return out


def run_cell(system: str, model_key: str, *, client: LLMClient | None = None, limit: int | None = None,
             questions: Sequence[ramdocs.RQuestion] | None = None) -> dict:
    client = client or get_client()
    tier = MODELS[model_key]
    config = {"system": system, "model": model_id(model_key), "prompt": prompt_sha()}
    path = cell_path(system, model_key)
    done = read_cell(path)
    for r in done.values():
        if {k: r.get(k) for k in config} != config:
            raise RuntimeError(f"{path.name} was produced under a different configuration")
    todo = list(questions if questions is not None else ramdocs.questions())[:limit]
    errors, ran = 0, 0
    started = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for q in todo:
            if q.id in done:
                continue
            red = SYSTEMS[system](q, client, tier)
            by_id = {p.id: p for p in ordered(q.id)}
            docs = "\n\n".join(f"Document {i}: {by_id[d].text}" for i, d in enumerate(red.kept, start=1))
            try:
                res = client.complete(messages=[{"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {q.question}"}],
                                      system=SYSTEM, tier=tier, step=f"study_ram_{system}")
                text, pt, ct = res.text.strip(), res.prompt_tokens, res.completion_tokens
            except TruncatedResponseError as exc:
                text, pt, ct = f"[truncated] {exc}"[:300], 0, 0
            except BackendError as exc:
                errors += 1
                if errors >= 3:
                    raise RuntimeError(f"{path.stem}: {errors} provider errors, last: {exc}") from exc
                continue
            s = ramdocs.score(text, q)
            rec = {"qid": q.id, **config, "kept": list(red.kept), "text": text,
                   "correct": s.correct, "gold_recall": s.gold_recall, "any_wrong": s.any_wrong,
                   "prompt_tokens": pt + int(red.trace.get("extra_prompt_tokens", 0)),
                   "completion_tokens": ct + int(red.trace.get("extra_completion_tokens", 0)),
                   "calls": 1 + int(red.trace.get("extra_calls", 0)),
                   "trace": {k: v for k, v in red.trace.items() if k != "scores"}}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            ran += 1
    return {"cell": path.stem, "ran": ran, "seconds": round(time.time() - started, 1)}


def survival(system: str) -> dict:
    """Model-free systems only: what fraction of each document type the operator keeps."""
    kept = {"correct": [0, 0], "misinfo": [0, 0], "noise": [0, 0]}
    for q in ramdocs.questions():
        red = SYSTEMS[system](q, None, None)
        k = set(red.kept)
        for d in q.documents:
            kept[d.type][0] += d.id in k
            kept[d.type][1] += 1
    return {t: a / b for t, (a, b) in kept.items() if b}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", nargs="+", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--model", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    for m in args.model:
        for s in args.system:
            print(json.dumps(run_cell(s, m, limit=args.limit)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
