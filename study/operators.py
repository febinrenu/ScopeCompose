"""Evidence-reducing operators.

Each operator takes the passages a retriever returned and hands on fewer of
them. They are the modules RAG stacks add to cope with noise, misinformation
and conflict, reduced here to their decision: *which passages survive*.

==============  ==============================================================
``rr{a}to{b}``  cross-encoder reranking of the top ``a``, keep the best ``b``
``nli``         contradiction filter: of two passages an NLI model judges
                contradictory, the lower-ranked one is dropped
``crs``         ConflictRAG-style resolution: NLI-detected conflicts resolved
                by Entropy-TOPSIS credibility selection (``baselines``)
``grade``       pointwise relevance grading by the generator model
                (CRAG / Self-RAG style); passages graded irrelevant are dropped
``select``      listwise consolidation by the generator model (Astute-RAG
                style): one call names the passages to keep
==============  ==============================================================

An operator works on plain ``(id, text)`` passages and a query, so the same
code runs on ConditionalQA chunks and on RAMDocs documents.

Model-free operators (``rr*``, ``nli``, ``crs``) are computed once per dataset
and stored (``python -m study.operators``); the store is what both the survival
analysis and the generation cells read, so they cannot disagree. Model-based
operators run inside the cell, on the cell's own model, and report their token
cost.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

from api_budget import TruncatedResponseError
from api_budget.client import LLMClient, Tier
from study import cqa, systems
from study.retrieve import ARTIFACTS

STORE = ARTIFACTS / "operators"
POOL = systems.DEFAULT_K               # every reducer starts from the same top-8
NLI_THRESHOLD = 0.5
RERANKER = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class P:
    id: str
    text: str


@dataclass(frozen=True)
class Reduction:
    kept: tuple[str, ...]              # in the order they are handed on
    trace: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Models, loaded once
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _reranker():
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(RERANKER, device=device, max_length=512)
    if device == "cuda":
        model.model.half()
    return model


@lru_cache(maxsize=None)
def _nli():
    from detection.nli import CrossEncoderNLI

    nli = CrossEncoderNLI()
    # Two ~150-word passages are ~400 tokens; the profile's 256 would cut the second one off.
    nli.spec = dataclasses.replace(nli.spec, max_seq_length=512, batch_size=24)
    return nli


# --------------------------------------------------------------------------- #
# Model-free operators
# --------------------------------------------------------------------------- #


def rerank_scores(query: str, passages: Sequence[P]) -> list[float]:
    if not passages:
        return []
    scores = _reranker().predict([(query, p.text) for p in passages], batch_size=32, show_progress_bar=False)
    return [float(s) for s in scores]


def contradiction_matrix(passages: Sequence[P]) -> list[list[float]]:
    """``m[i][j]`` = P(contradiction) with passage i as premise and j as hypothesis."""
    n = len(passages)
    pairs = [(passages[i].text, passages[j].text) for i in range(n) for j in range(n) if i != j]
    scores = iter(_nli().score_batch(pairs))
    return [[0.0 if i == j else next(scores).contradiction for j in range(n)] for i in range(n)]


def nli_filter(passages: Sequence[P], matrix: list[list[float]], *, threshold: float = NLI_THRESHOLD) -> Reduction:
    """Of each contradicting pair, drop the lower-ranked passage (the input order is the rank order)."""
    dropped: dict[str, dict] = {}
    alive = [True] * len(passages)
    for j in range(len(passages)):
        for i in range(j):
            if not alive[i]:
                continue
            c = max(matrix[i][j], matrix[j][i])
            if c >= threshold:
                alive[j] = False
                dropped[passages[j].id] = {"against": passages[i].id, "contradiction": round(c, 4)}
                break
    return Reduction(tuple(p.id for p, a in zip(passages, alive) if a), {"dropped": dropped})


def conflictrag_select(passages: Sequence[P], matrix: list[list[float]], *,
                       threshold: float = NLI_THRESHOLD) -> Reduction:
    """NLI-detected conflicts, each resolved by the repository's Entropy-TOPSIS selection.

    All ConditionalQA passages are gov.uk guidance with no date, so authority
    and recency tie and the criterion left to discriminate is specificity; when
    the winner's margin is below the resolver's sufficiency margin it keeps
    both. That is the published design behaving as designed, not a handicap.
    """
    from baselines.conflictrag_topsis import select_by_credibility
    from contract.models import Passage, SourceType

    by_id = {p.id: Passage(id=p.id, text=p.text, source_type=SourceType.GOVERNMENT_GUIDANCE) for p in passages}
    drop: dict[str, dict] = {}
    conflicts = 0
    for j in range(len(passages)):
        for i in range(j):
            a, b = passages[i].id, passages[j].id
            if a in drop or b in drop or max(matrix[i][j], matrix[j][i]) < threshold:
                continue
            conflicts += 1
            sel = select_by_credibility([by_id[a], by_id[b]])
            for loser in {a, b} - set(sel.chosen):
                drop[loser] = {"against": (set(sel.chosen) & {a, b}).pop(), "kind": sel.kind}
    return Reduction(tuple(p.id for p in passages if p.id not in drop), {"dropped": drop, "conflicts": conflicts})


# --------------------------------------------------------------------------- #
# The store of model-free decisions
# --------------------------------------------------------------------------- #


def _store_path(dataset: str, env_tag: str) -> Path:
    return STORE / f"{dataset}_{env_tag}.json"


@lru_cache(maxsize=None)
def load_store(dataset: str = "cqa", env_tag: str = "fixed150-hybrid") -> dict[str, dict]:
    path = _store_path(dataset, env_tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m study.operators --dataset {dataset}`")
    return json.loads(path.read_text(encoding="utf-8"))


def precompute(dataset: str, items: Sequence[tuple[str, str, Sequence[P]]], env_tag: str, *,
               rerank_pool: Callable[[str], Sequence[P]] | None = None) -> dict[str, dict]:
    """Rerank scores and the contradiction matrix for every ``(id, query, passages)``.

    Stores raw scores, not decisions: thresholds and cut-offs are applied when
    read, so a sensitivity analysis costs nothing.
    """
    path = _store_path(dataset, env_tag)
    have: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    new = 0
    for qid, query, passages in items:
        if qid in have:
            continue
        rec = {"ids": [p.id for p in passages], "rerank": rerank_scores(query, passages),
               "contradiction": [[round(x, 4) for x in row] for row in contradiction_matrix(passages)]}
        if rerank_pool is not None:
            wide = list(rerank_pool(qid))
            rec["wide_ids"] = [p.id for p in wide]
            rec["wide_rerank"] = rerank_scores(query, wide)
        have[qid] = rec
        new += 1
        if new % 100 == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(have), encoding="utf-8")
            print(f"  {dataset}: {len(have)} stored", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(have), encoding="utf-8")
    load_store.cache_clear()
    return have


def stored_rerank(rec: dict, keep: int, *, wide: bool = False) -> Reduction:
    ids, scores = (rec["wide_ids"], rec["wide_rerank"]) if wide else (rec["ids"], rec["rerank"])
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))[:keep]
    return Reduction(tuple(ids[i] for i in order), {"scores": {ids[i]: round(scores[i], 3) for i in order}})


def stored_nli(rec: dict, *, threshold: float = NLI_THRESHOLD) -> Reduction:
    return nli_filter([P(i, "") for i in rec["ids"]], rec["contradiction"], threshold=threshold)


# --------------------------------------------------------------------------- #
# Model-based operators
# --------------------------------------------------------------------------- #

GRADE_SYSTEM = """\
You decide whether a passage is useful for answering a question.
A passage is useful if it contains the answer, part of the answer, or a rule, requirement or exception that affects the answer.
Reply with JSON only: {"useful": true} or {"useful": false}."""

SELECT_SYSTEM = """\
You are given a question and numbered passages retrieved for it. Some passages may be irrelevant, unreliable, or in conflict with more reliable passages.
Choose the passages that should be used to answer the question: keep the relevant and reliable ones, and when passages conflict keep the more reliable one.
Reply with JSON only: {"keep": [passage numbers]}."""


@dataclass
class LLMCost:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, res) -> None:
        self.calls += 1
        self.prompt_tokens += res.prompt_tokens
        self.completion_tokens += res.completion_tokens

    def as_dict(self) -> dict:
        return {"extra_calls": self.calls, "extra_prompt_tokens": self.prompt_tokens,
                "extra_completion_tokens": self.completion_tokens}


def _json(text: str) -> dict:
    try:
        v = json.loads(text)
    except (TypeError, ValueError):
        import re

        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        try:
            v = json.loads(m.group(0)) if m else {}
        except ValueError:
            v = {}
    return v if isinstance(v, dict) else {}


def llm_grade(client: LLMClient, tier: Tier, query: str, passages: Sequence[P]) -> Reduction:
    """One call per passage. An unparseable grade keeps the passage: the operator never drops on a failure."""
    cost = LLMCost()
    kept, grades = [], {}
    for p in passages:
        try:
            res = client.complete(messages=[{"role": "user", "content": f"Question: {query}\n\nPassage:\n{p.text}"}],
                                  system=GRADE_SYSTEM, tier=tier, step="study_op_grade",
                                  response_format={"type": "json_object"})
            cost.add(res)
            useful = _json(res.text).get("useful")
        except TruncatedResponseError:
            useful = None
        grades[p.id] = useful if isinstance(useful, bool) else None
        if useful is not False:
            kept.append(p.id)
    if not kept and passages:                       # nothing judged useful: fall back to the top-ranked passage
        kept = [passages[0].id]
    return Reduction(tuple(kept), {"grades": grades, **cost.as_dict()})


def llm_select(client: LLMClient, tier: Tier, query: str, passages: Sequence[P]) -> Reduction:
    """One call for the whole list. An unusable reply keeps every passage."""
    cost = LLMCost()
    listing = "\n\n".join(f"Passage {i}:\n{p.text}" for i, p in enumerate(passages, start=1))
    keep: list[int] | None = None
    try:
        res = client.complete(messages=[{"role": "user", "content": f"Question: {query}\n\n{listing}"}],
                              system=SELECT_SYSTEM, tier=tier, step="study_op_select",
                              response_format={"type": "json_object"})
        cost.add(res)
        raw = _json(res.text).get("keep")
        if isinstance(raw, list):
            keep = sorted({int(x) for x in raw if str(x).strip().lstrip("-").isdigit() and 1 <= int(x) <= len(passages)})
    except TruncatedResponseError:
        pass
    if not keep:
        return Reduction(tuple(p.id for p in passages), {"keep": None, **cost.as_dict()})
    return Reduction(tuple(passages[i - 1].id for i in keep), {"keep": keep, **cost.as_dict()})


# --------------------------------------------------------------------------- #
# ConditionalQA systems
# --------------------------------------------------------------------------- #


def _env_tag(env: systems.Env) -> str:
    return f"{env.strategy}{env.words}-{env.ranker}"


def _pool(q: cqa.Question, env: systems.Env) -> list[cqa.Chunk]:
    return env.ranked(q)[:POOL]


def _from_store(reduce: Callable[[dict], Reduction]) -> systems.Builder:
    def build(q: cqa.Question, env: systems.Env) -> systems.Context:
        red = reduce(load_store("cqa", _env_tag(env))[q.id])
        return systems.Context(tuple(env.chunk(i) for i in red.kept), trace=red.trace)
    return build


def _from_llm(op: Callable[[LLMClient, Tier, str, Sequence[P]], Reduction]) -> systems.Builder:
    def build(q: cqa.Question, env: systems.Env) -> systems.Context:
        if env.client is None or env.tier is None:
            raise RuntimeError("a model-based operator needs the cell's client and tier")
        pool = _pool(q, env)
        red = op(env.client, env.tier, q.retrieval_query, [P(c.id, c.text) for c in pool])
        return systems.Context(tuple(env.chunk(i) for i in red.kept), trace=red.trace)
    return build


systems.register("rr8to3")(_from_store(lambda rec: stored_rerank(rec, 3)))
systems.register("rr8to5")(_from_store(lambda rec: stored_rerank(rec, 5)))
systems.register("rr20to8")(_from_store(lambda rec: stored_rerank(rec, 8, wide=True)))
systems.register("nli")(_from_store(stored_nli))
systems.register("crs")(_from_store(lambda rec: Reduction(tuple(rec["crs_kept"]), rec.get("crs_trace", {}))))
systems.register("grade")(_from_llm(llm_grade))
systems.register("select")(_from_llm(llm_select))


def cqa_items(qs: Sequence[cqa.Question], env: systems.Env) -> list[tuple[str, str, list[P]]]:
    return [(q.id, q.retrieval_query, [P(c.id, c.text) for c in _pool(q, env)]) for q in qs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cqa", choices=("cqa", "ramdocs"))
    args = ap.parse_args(argv)
    if args.dataset == "cqa":
        env = systems.Env()
        qs = cqa.eval_questions() + cqa.tuning_questions()
        by_q = {q.id: q for q in qs}
        store = precompute("cqa", cqa_items(qs, env), _env_tag(env),
                           rerank_pool=lambda qid: [P(c.id, c.text) for c in env.ranked(by_q[qid])[:20]])
        # ConflictRAG-style selection needs the passage texts; decide once and keep the decision.
        changed = False
        for qid, rec in store.items():
            if "crs_kept" not in rec:
                red = conflictrag_select([P(c.id, c.text) for c in _pool(by_q[qid], env)], rec["contradiction"])
                rec["crs_kept"], rec["crs_trace"] = list(red.kept), red.trace
                changed = True
        if changed:
            _store_path("cqa", _env_tag(env)).write_text(json.dumps(store), encoding="utf-8")
            load_store.cache_clear()
        print(f"cqa: {len(store)} questions stored")
    else:
        from study import ramdocs

        items = [(q.id, q.question, [P(d.id, d.text) for d in q.documents]) for q in ramdocs.questions()]
        store = precompute("ramdocs", items, "all")
        print(f"ramdocs: {len(store)} questions stored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
