"""Run one cell: (system x model x question set).

    python -m study.run --system top8 --model qwen7b --set eval
    python -m study.run --system gold top8 --model qwen7b --set tuning --limit 60

A cell file is one JSON line per question, appended as it is answered, so an
interrupted cell resumes where it stopped and nothing is paid for twice. A file
that already holds answers from a different model, prompt or chunking is
refused: a cell never mixes them. Scoring is done later, from these files, by
``study.analysis`` -- running and scoring are separate so a number can be
recomputed without calling a model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import settings
from api_budget import BackendError, RateLimitError
from api_budget.client import LLMClient, Tier, get_client
from study import cqa, reader, systems

RUNS = settings.REPO_ROOT / "study" / "runs"

MODELS: dict[str, Tier] = {
    "qwen7b": Tier.STUDY_QWEN7B,
    "llama8b": Tier.STUDY_LLAMA8B,
    "oss20b": Tier.STUDY_OSS20B,
    "qwen27b": Tier.STUDY_QWEN27B,
    "oss120b": Tier.STUDY_OSS120B,
    "gemma31b": Tier.STUDY_GEMMA31B,
    "mistral4": Tier.STUDY_MISTRAL4,
}

#: Providers serving the same open weights, tried in order when one's quota is spent or it is
#: unreachable. The cell's model id stays the primary tier's; each answer records ``served_by``.
ROUTES: dict[str, tuple[Tier, ...]] = {
    "oss20b": (Tier.STUDY_OSS20B, Tier.STUDY_OSS20B_NV, Tier.STUDY_OSS20B_NV2),
    "gemma31b": (Tier.STUDY_GEMMA31B, Tier.STUDY_GEMMA31B_NV2),
    "mistral4": (Tier.STUDY_MISTRAL4, Tier.STUDY_MISTRAL4_NV2),
}
# Ollama's cloud (STUDY_*_OC tiers) also serves the gpt-oss weights but is paid-only on this
# account (HTTP 402), so it is not in the routes.
_route: dict[str, int] = {}          # model key -> route in use; sticky for the life of the process
_route_lock = threading.Lock()


def served_by(tier: Tier) -> str:
    spec = settings.models().tier(tier.value)
    return f"{spec.provider}:{spec.model}"


class CellInterrupted(RuntimeError):
    """The cell stopped early (provider errors or a token budget); rerun to resume."""


class CellMismatch(RuntimeError):
    """The cell file was produced under a different configuration."""


def model_id(model_key: str) -> str:
    return settings.models().tier(MODELS[model_key].value).model or ""


def cell_path(system: str, model_key: str, qset: str, env: systems.Env) -> Path:
    variant = "" if (env.words, env.strategy, env.ranker) == (cqa.DEFAULT_CHUNK_WORDS, "fixed", "hybrid") \
        else f"__{env.strategy}{env.words}-{env.ranker}"
    return RUNS / qset / f"{system}__{model_key}{variant}.jsonl"


def read_cell(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:      # a run stopped mid-write: that question is simply asked again
                    continue
                raise
            done[rec["qid"]] = rec
    return done


def question_set(name: str) -> list[cqa.Question]:
    if name == "tuning":       # minus the four questions shown to the generator as worked examples
        return [q for q in cqa.tuning_questions() if q.id not in reader.DEMO_QUESTION_IDS]
    if name == "eval":
        return cqa.eval_questions()
    if name == "hosted":       # the stratified subsample the hosted models answer
        return cqa.hosted_questions()
    raise ValueError(f"unknown question set {name!r}")


def applicable(system: str, qs: Iterable[cqa.Question]) -> list[cqa.Question]:
    """Oracle systems are defined on answerable questions only: an unanswerable one has no gold chunk."""
    qs = list(qs)
    return [q for q in qs if q.kind != "unanswerable"] if system in systems.ORACLE else qs


def _generate_routed(client: LLMClient, model_key: str, tier: Tier, q: cqa.Question, evidence, step: str):
    """Generate with the model's providers in turn; returns (generation, tier used).

    Raises the last error when every provider failed. Threads share ``_route``: a provider is
    skipped only by the first thread to see it fail, so parallel failures never skip a working one."""
    routes = ROUTES.get(model_key, (tier,))
    for attempt in range(len(routes)):
        with _route_lock:
            idx = _route.get(model_key, 0) % len(routes)
        use = routes[idx]
        try:
            return reader.generate(client, use, q, evidence, step=step), use
        except (RateLimitError, BackendError) as exc:
            if attempt + 1 == len(routes):
                raise
            with _route_lock:
                if _route.get(model_key, 0) % len(routes) == idx:     # nobody has moved on yet
                    _route[model_key] = idx + 1
                    print(f"  {model_key}: {served_by(use)} unavailable ({str(exc)[:90]}); "
                          f"switching to {served_by(routes[(idx + 1) % len(routes)])}", file=sys.stderr, flush=True)
    raise AssertionError("unreachable")


def run_cell(system: str, model_key: str, qset: str, *, env: systems.Env | None = None,
             client: LLMClient | None = None, questions: Sequence[cqa.Question] | None = None,
             limit: int | None = None, token_budget: int | None = None, max_errors: int = 3,
             progress: bool = False, workers: int = 1) -> dict:
    """Answer every question of the cell not yet in its file. ``workers`` > 1 sends that many requests
    at once (hosted providers); answers are appended as they arrive, each line whole."""
    client = client or get_client()
    tier = MODELS[model_key]
    env = dataclasses.replace(env or systems.Env(), client=client, tier=tier)
    config = {"system": system, "model": model_id(model_key), "prompts": reader.prompts_sha(),
              "words": env.words, "strategy": env.strategy, "ranker": env.ranker}
    path = cell_path(system, model_key, qset, env)
    if path.exists():                           # drop a line cut off by an interrupted run before appending
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            path.write_bytes(raw[: raw.rfind(b"\n") + 1])
    done = read_cell(path)
    for rec in done.values():
        got = {k: rec.get(k) for k in config}
        if got != config:
            diff = {k: (got[k], config[k]) for k in config if got[k] != config[k]}
            raise CellMismatch(f"{path.name} was produced under a different configuration: {diff}")

    todo = applicable(system, questions if questions is not None else question_set(qset))
    if limit is not None:
        todo = todo[:limit]
    summary = {"cell": path.stem, "total": len(todo), "skipped": 0, "ran": 0, "tokens": 0}
    state = {"errors": 0, "spent": 0}
    lock = threading.Lock()
    started = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)

    def answer(q: cqa.Question, fh) -> None:
        if token_budget is not None and state["spent"] >= token_budget:
            raise CellInterrupted(f"{path.stem}: token budget {token_budget} reached after {summary['ran']} new answers")
        ctx = systems.build(system, q, env)
        if system in systems.ORACLE and not ctx.chunks and not env.gold(q).all_chunks:
            with lock:
                summary["undefined"] = summary.get("undefined", 0) + 1     # no annotated evidence: the rung does not exist
            return
        evidence = None if ctx.chunks is None else reader.render_evidence(ctx.chunks, drop_elements=ctx.drop_elements)
        try:
            gen, use = _generate_routed(client, model_key, tier, q, evidence, f"study_{system}")
        except RateLimitError:
            raise
        except BackendError as exc:
            with lock:
                state["errors"] += 1
                if state["errors"] >= max_errors:
                    raise CellInterrupted(f"{path.stem}: {state['errors']} provider errors, last: {exc}") from exc
            return
        # An operator that called the model reports what that cost; it is part of the system's price.
        extra_p = int(ctx.trace.get("extra_prompt_tokens", 0))
        extra_c = int(ctx.trace.get("extra_completion_tokens", 0))
        rec = {"qid": q.id, **config, "chunks": ctx.chunk_ids, "n_lines": evidence.n_lines if evidence else 0,
               "answers": [[a, list(c)] for a, c in gen.parsed.answers], "evidence": list(gen.parsed.evidence), "ok": gen.parsed.ok,
               "bad_citations": gen.parsed.bad_citations, "text": gen.text,
               "prompt_tokens": gen.prompt_tokens + extra_p, "completion_tokens": gen.completion_tokens + extra_c,
               "calls": 1 + int(ctx.trace.get("extra_calls", 0)),
               "latency_s": round(gen.latency_s, 3), "served_by": served_by(use), "trace": ctx.trace}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            state["spent"] += rec["prompt_tokens"] + rec["completion_tokens"]
            summary["ran"] += 1
            summary["tokens"] = state["spent"]
            if progress and summary["ran"] % 50 == 0:
                rate = (time.time() - started) / summary["ran"]
                left = len(todo) - summary["skipped"] - summary["ran"]
                print(f"  {path.stem}: {summary['ran']} answered, {left} left, {rate:.1f}s each", file=sys.stderr, flush=True)

    new = [q for q in todo if q.id not in done]
    summary["skipped"] = len(todo) - len(new)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if workers <= 1:
            for q in new:
                answer(q, fh)
        else:
            ex = ThreadPoolExecutor(max_workers=workers)
            try:
                for f in as_completed([ex.submit(answer, q, fh) for q in new]):
                    f.result()                          # re-raise a spent quota or too many errors here
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
    if state["errors"]:
        raise CellInterrupted(f"{path.stem}: finished with {state['errors']} question(s) skipped after provider errors; rerun")
    summary["seconds"] = round(time.time() - started, 1)
    return summary


def main(argv: list[str] | None = None) -> int:
    from study import operators, qualifier  # noqa: F401  (register the operator and remedy systems)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", nargs="+", required=True)
    ap.add_argument("--model", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--set", dest="qset", default="eval", choices=("tuning", "eval", "hosted"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--words", type=int, default=cqa.DEFAULT_CHUNK_WORDS)
    ap.add_argument("--strategy", default="fixed", choices=cqa.STRATEGIES)
    ap.add_argument("--ranker", default="hybrid", choices=("bm25", "dense", "hybrid"))
    args = ap.parse_args(argv)
    env = systems.Env(args.words, args.strategy, args.ranker)
    status = 0
    for model_key in args.model:
        for system in args.system:
            try:
                s = run_cell(system, model_key, args.qset, env=env, limit=args.limit,
                             token_budget=args.token_budget, progress=True)
                print(json.dumps(s), flush=True)
            except CellInterrupted as exc:
                print(f"INTERRUPTED (resumable): {exc}", file=sys.stderr, flush=True)
                status = 3
    return status


if __name__ == "__main__":
    raise SystemExit(main())
