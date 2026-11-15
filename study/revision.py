"""Experiments added in revision, in answer to reviewers. All exploratory; none changes a
confirmatory result.

    python -m study.revision local      # local generators: verification pass, oracle completion, section chunks
    python -m study.revision hosted     # hosted generators: the same, plus the provider-agreement check

1. **Condition verification** (``<system>_verify``). A second call per question, after the frozen
   generator has answered: given the same numbered lines, the situation, the question and the
   answers already given, the model lists for each answer the lines stating a requirement that
   the situation does not settle. The answers are kept from the first call; only their conditions
   are replaced. It tests whether the generation loss is a failure to *notice* conditions that a
   dedicated check can recover, without touching retrieval.
2. **Oracle completion** (``qcoracle``, in ``study.qualifier``): completion with a perfect scorer.
3. **Section-aware chunks** (``top8`` and ``gold`` with the section-aware chunking): whether keeping
   page structure, rather than a fixed word budget, reduces the loss end to end.
4. **Provider agreement**: every gpt-oss-20b gold-context answer that Groq served is asked again on
   NVIDIA's API, so the two providers can be compared on the same questions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from api_budget import BackendError, RateLimitError, TruncatedResponseError
from api_budget.client import Tier, get_client
from study import cqa, operators, qualifier, reader, systems  # noqa: F401  (register systems)
from study.run import MODELS, ROUTES, RUNS, cell_path, read_cell, run_cell, served_by

VERIFY_SYSTEM = """\
You check the answers given to a person's question about UK government guidance.
You are given numbered lines copied from official guidance pages, the person's situation, their question, and the answers already given.
For each answer, list the lines that state a condition on it. A condition is a requirement that must be satisfied for the answer to be true and that the person's situation does not mention. Go through every line that states a requirement, an eligibility rule or an exception, and decide for each whether the situation already settles it. A line that only states the answer is not a condition.
Reply with JSON only, one entry per answer in the same order:
{"answers": [{"answer": "<the answer, copied exactly>", "conditions": [<line numbers>]}]}
Give "conditions": [] when the answer holds without any unsettled requirement."""


def verify_sha() -> str:
    import hashlib

    return hashlib.sha256(VERIFY_SYSTEM.encode("utf-8")).hexdigest()


def _complete_routed(client, model_key: str, messages: list[dict], system: str):
    routes = ROUTES.get(model_key, (MODELS[model_key],))
    last: Exception | None = None
    for tier in routes:
        try:
            return client.complete(messages=messages, system=system, tier=tier, step="study_verify",
                                   response_format={"type": "json_object"}), tier
        except (RateLimitError, BackendError) as exc:
            if isinstance(exc, TruncatedResponseError):
                raise
            last = exc
    raise last  # type: ignore[misc]


def verify_cell(system: str, model_key: str, qset: str, *, env: systems.Env | None = None, workers: int = 1) -> dict:
    """Second pass over the answers of cell ``system`` (which must be complete): writes ``<system>_verify``."""
    env = env or systems.Env()
    src = read_cell(cell_path(system, model_key, qset, env))
    if not src:
        raise FileNotFoundError(f"{system}__{model_key} ({qset}) has no answers to verify")
    out_path = cell_path(f"{system}_verify", model_key, qset, env)
    if out_path.exists():                       # drop a line cut off by an interrupted run before appending
        raw = out_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            out_path.write_bytes(raw[: raw.rfind(b"\n") + 1])
    done = read_cell(out_path)
    client = get_client()
    idx = cqa.by_id()
    lock = threading.Lock()
    summary = {"cell": out_path.stem, "ran": 0, "copied": 0, "failed": 0}

    def one(qid: str, rec: dict, fh) -> None:
        q = idx[qid]
        new = {**rec, "system": f"{system}_verify", "verify_prompt": verify_sha()}
        if not rec["answers"]:                       # nothing to verify (no answer was given)
            new.update({"stage2_text": "", "calls": rec.get("calls", 1)})
            with lock:
                fh.write(json.dumps(new, ensure_ascii=False) + "\n"); fh.flush(); summary["copied"] += 1
            return
        ctx = systems.build(system, q, env)
        ev = reader.render_evidence(ctx.chunks, drop_elements=ctx.drop_elements)
        given = "\n".join(f"{i}. {a}" for i, (a, _) in enumerate(rec["answers"], start=1))
        msgs = [{"role": "user", "content": f"{reader.user_prompt(q, ev)}\n\nAnswers given:\n{given}"}]
        try:
            res, tier = _complete_routed(client, model_key, msgs, VERIFY_SYSTEM)
            text, pt, ct = res.text, res.prompt_tokens, res.completion_tokens
            parsed = reader.parse_output(text, ev)
        except TruncatedResponseError as exc:
            text, pt, ct, tier, parsed = f"[truncated] {exc}"[:300], 0, 0, MODELS[model_key], reader.Parsed((), ok=False)
        except BackendError as exc:
            if "token repeat limit" not in str(exc):
                with lock:
                    summary["failed"] += 1
                return
            text, pt, ct, tier, parsed = f"[degenerate] {exc}"[:300], 0, 0, MODELS[model_key], reader.Parsed((), ok=False)
        # Answers stay those of the first call; conditions come from the check, matched by position.
        conds = [list(c) for _, c in parsed.answers]
        answers = [[a, conds[i] if i < len(conds) else []] for i, (a, _) in enumerate(rec["answers"])]
        new.update({"answers": answers, "stage1_answers": rec["answers"], "stage2_text": text, "verify_ok": parsed.ok,
                    "prompt_tokens": rec.get("prompt_tokens", 0) + pt,
                    "completion_tokens": rec.get("completion_tokens", 0) + ct,
                    "calls": rec.get("calls", 1) + 1, "stage2_served_by": served_by(tier)})
        with lock:
            fh.write(json.dumps(new, ensure_ascii=False) + "\n"); fh.flush(); summary["ran"] += 1

    todo = [(qid, rec) for qid, rec in src.items() if qid not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        if workers <= 1:
            for qid, rec in todo:
                one(qid, rec, fh)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for f in as_completed([ex.submit(one, qid, rec, fh) for qid, rec in todo]):
                    f.result()
    return summary


def provider_check(workers: int = 8) -> dict:
    """Ask NVIDIA every gpt-oss-20b gold-context question that Groq answered; same prompt, same settings."""
    src = read_cell(cell_path("gold", "oss20b", "hosted", systems.Env()))
    groq = {qid: r for qid, r in src.items() if r.get("served_by", "groq").startswith("groq")}
    out_path = RUNS / "provider_check" / "gold__oss20b_nvidia.jsonl"
    done = read_cell(out_path)
    client, idx, env = get_client(), cqa.by_id(), systems.Env()
    lock = threading.Lock()

    def one(qid: str, fh) -> None:
        q = idx[qid]
        ctx = systems.build("gold", q, env)
        ev = reader.render_evidence(ctx.chunks, drop_elements=ctx.drop_elements)
        for tier in (Tier.STUDY_OSS20B_NV, Tier.STUDY_OSS20B_NV2):
            try:
                gen = reader.generate(client, tier, q, ev, step="study_provider_check")
                break
            except (RateLimitError, BackendError):
                gen = None
        if gen is None:
            return
        rec = {**src[qid], "answers": [[a, list(c)] for a, c in gen.parsed.answers], "evidence": list(gen.parsed.evidence),
               "ok": gen.parsed.ok, "text": gen.text, "prompt_tokens": gen.prompt_tokens,
               "completion_tokens": gen.completion_tokens, "latency_s": round(gen.latency_s, 3), "served_by": served_by(tier)}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(as_completed([ex.submit(one, qid, fh) for qid in groq if qid not in done]))
    return {"groq_answers": len(groq), "rerun_on_nvidia": len(read_cell(out_path))}


SECTION_ENV = systems.Env(150, "section", "hybrid")


def local() -> int:
    for m in ("qwen7b", "llama8b"):
        subprocess.run(["ollama", "stop", "scopecompose-qwen7b"], capture_output=True)
        subprocess.run(["ollama", "stop", "scopecompose-reader"], capture_output=True)
        # Locally only the gold context is verified: the decisive case, where every condition is present.
        print(json.dumps(verify_cell("gold", m, "eval")), flush=True)
        print(json.dumps(run_cell("qcoracle", m, "eval", progress=True)), flush=True)
        for s in ("top8", "gold"):
            print(json.dumps(run_cell(s, m, "eval", env=SECTION_ENV, progress=True)), flush=True)
    return 0


def hosted(models: tuple[str, ...] = ("gemma31b", "nemotron")) -> int:
    """Run one process per model to work on both at once; never two processes on the same model."""
    if "gemma31b" in models:
        print(json.dumps(provider_check()), flush=True)
    for m in models:
        qs = cqa.hosted_questions()
        for attempt in range(6):
            try:
                print(json.dumps(run_cell("qcoracle", m, "hosted", questions=qs, workers=8, progress=True)), flush=True)
                for s in ("gold", "top8"):
                    print(json.dumps(verify_cell(s, m, "hosted", workers=8)), flush=True)
                break
            except (RateLimitError, Exception) as exc:  # noqa: BLE001  (a spent minute-quota: wait and resume)
                print(f"{m}: paused ({str(exc)[:120]})", file=sys.stderr, flush=True)
                time.sleep(120)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=("local", "hosted"))
    ap.add_argument("--model", choices=("gemma31b", "nemotron"), help="hosted only: run just this model")
    args = ap.parse_args(argv)
    if args.which == "local":
        return local()
    return hosted((args.model,) if args.model else ("gemma31b", "nemotron"))


if __name__ == "__main__":
    raise SystemExit(main())
