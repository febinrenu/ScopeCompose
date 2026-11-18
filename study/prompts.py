"""Prompt robustness (revision, round 6; exploratory): is the generation loss a property of one prompt?

The study freezes one generator prompt (``study.reader``, wording v4) so that systems differ only in
their evidence. This module asks the same generators the same questions, with the same numbered
lines and the same four worked examples, under five other condition-aware prompting strategies,
each written before any of them was run on the evaluation questions and none tuned on them:

``extract``        direct condition extraction: list every requirement line first, then answer
``pairing``        explicit answer-condition pairing: state what each answer holds only if
``coc``            chain-of-condition reasoning: identify conditions, how they combine, and whether
                   the situation satisfies, contradicts or leaves each open; then answer
``applicability``  scenario-condition verification: mark each requirement line settled or open
``evidence``       structured evidence first: label each relevant line's role, then answer

Every variant ends in the frozen output contract (``answers`` with ``evidence`` and ``conditions``
as line numbers), parsed by the frozen parser, so the official metric scores all six alike.
Extra fields (the extracted requirements, the reasoning) are kept in the record, not scored.

    python -m study.prompts local       # Qwen2.5-7B and Llama-3.1-8B, hosted subsample
    python -m study.prompts hosted      # Nemotron-3-Ultra, hosted subsample

Cells: ``study/runs/hosted/<system>_p<variant>__<model>.jsonl`` for ``gold`` and
``gold_nocondchk`` (the condition-only-chunk ablation, C2), so both the generation deficit and the
chunking cost can be compared across prompts on the same questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from api_budget import BackendError, RateLimitError, TruncatedResponseError
from api_budget.client import get_client
from study import cqa, reader, systems
from study.run import MODELS, ROUTES, RUNS, applicable, model_id, question_set, read_cell, served_by

_HEAD = """\
You answer a person's question about UK government guidance.
You are given numbered lines copied from official guidance pages, the person's situation, and their question.
"""

_ANSWER_RULES = """\
- Use only the numbered lines. If they do not contain the answer, give "answers": [].
- "answer": for a yes/no question write exactly "yes" or "no". Otherwise copy the shortest phrase from the lines that answers the question - an amount, a date, a period, a name, a form, a place.
- "evidence": the numbers of the lines that state the answer.
- Give one answer. Give two only when the answer really differs depending on something the situation leaves open - then list each answer with its own conditions."""

VARIANTS: dict[str, str] = {
    "extract": _HEAD + """
Work in two steps. Reply with JSON only, in this form:
{"requirements": [line numbers], "answers": [{"answer": "...", "evidence": [line numbers], "conditions": [line numbers]}]}

Step 1. "requirements": every line that states a requirement, an eligibility rule, a time limit or an exception that bears on the question.
Step 2. Answer the question:
""" + _ANSWER_RULES + """
- "conditions": the lines from "requirements" that must be satisfied for the answer to be true and that the person's situation does not settle. Give [] if the situation settles all of them or none applies.""",

    "pairing": _HEAD + """
Reply with JSON only, in this form:
{"answers": [{"answer": "...", "holds_if": "...", "evidence": [line numbers], "conditions": [line numbers]}]}

Rules:
""" + _ANSWER_RULES + """
- For each answer, decide whether it holds whatever else is true of the person, or only if something their situation does not settle is true.
- "holds_if": if only then, a short statement of what must be true; otherwise "".
- "conditions": the lines that state what "holds_if" says. Give [] when "holds_if" is "". A line that only states the answer is not a condition.
- If the answer would be different when that requirement is not met, give that other answer as well, with its own conditions.""",

    "coc": _HEAD + """
Think before you answer. Reply with JSON only, in this form:
{"analysis": "...", "answers": [{"answer": "...", "evidence": [line numbers], "conditions": [line numbers]}]}

"analysis" (at most 120 words): (a) name the lines that state a condition relevant to the question; (b) say how they combine - all required, any one enough, or an exception to a rule; (c) for each, say whether the situation satisfies it, contradicts it, or does not say.
Then answer:
""" + _ANSWER_RULES + """
- "conditions": the lines whose condition the situation does not say anything about and on which the answer depends. Give [] if there is none. A line that only states the answer is not a condition.""",

    "applicability": _HEAD + """
Reply with JSON only, in this form:
{"checks": [{"line": n, "settled": true}], "answers": [{"answer": "...", "evidence": [line numbers], "conditions": [line numbers]}]}

Rules:
- "checks": every line that states a requirement, an eligibility rule or an exception relevant to the question, with "settled": true if the person's situation already shows whether it is met, and false if the situation does not say.
""" + _ANSWER_RULES + """
- "conditions": the lines with "settled": false on which the answer depends. Give [] if there is none.""",

    "evidence": _HEAD + """
First collect the evidence, then answer from it. Reply with JSON only, in this form:
{"relevant": [{"line": n, "role": "answer"}], "answers": [{"answer": "...", "evidence": [line numbers], "conditions": [line numbers]}]}

Rules:
- "relevant": the lines that bear on the question, each with its role: "answer" (states the answer), "requirement" (something that must hold for the answer), "exception" (a case in which the answer does not hold) or "background".
""" + _ANSWER_RULES + """
- "evidence": the lines with role "answer".
- "conditions": the lines with role "requirement" or "exception" that the person's situation does not settle and on which the answer depends. Give [] if there is none.""",
}

#: The frozen worked examples' replies, rewritten in each variant's format. Same questions, same
#: lines, same answers and the same gold conditions as ``reader.DEMOS``.
DEMO_REPLIES: dict[str, tuple[dict, dict, dict, dict]] = {
    "extract": (
        {"requirements": [3], "answers": [{"answer": "yes", "evidence": [2], "conditions": []}]},
        {"requirements": [1], "answers": [{"answer": "within 6 weeks", "evidence": [3], "conditions": []}]},
        {"requirements": [1], "answers": [{"answer": "yes", "evidence": [1], "conditions": [1]}]},
        {"requirements": [3, 4], "answers": [{"answer": "at least 6 years", "evidence": [2], "conditions": [3, 4]}]},
    ),
    "pairing": (
        {"answers": [{"answer": "yes", "holds_if": "", "evidence": [2], "conditions": []}]},
        {"answers": [{"answer": "within 6 weeks", "holds_if": "", "evidence": [3], "conditions": []}]},
        {"answers": [{"answer": "yes", "holds_if": "the other child also has or is applying for a Child Student visa",
                      "evidence": [1], "conditions": [1]}]},
        {"answers": [{"answer": "at least 6 years", "holds_if": "the records were created on or after 1 April 2021, or still had to be kept on 31 March 2021",
                      "evidence": [2], "conditions": [3, 4]}]},
    ),
    "coc": (
        {"analysis": "Line 3 excludes sole traders and partnerships; the situation says they are a company, so it is satisfied. Line 2 says a return is still needed after a loss.",
         "answers": [{"answer": "yes", "evidence": [2], "conditions": []}]},
        {"analysis": "Line 1 requires valid invoices; the situation says all original invoices were included, so it is satisfied. Line 3 gives the time.",
         "answers": [{"answer": "within 6 weeks", "evidence": [3], "conditions": []}]},
        {"analysis": "Line 1: other children can come only if they also have or are applying for a Child Student visa. The situation does not say whether the younger daughter is.",
         "answers": [{"answer": "yes", "evidence": [1], "conditions": [1]}]},
        {"analysis": "Lines 3 and 4 are alternatives (any one is enough) for the 6-year rule in line 2. The situation does not say when the records were created.",
         "answers": [{"answer": "at least 6 years", "evidence": [2], "conditions": [3, 4]}]},
    ),
    "applicability": (
        {"checks": [{"line": 3, "settled": True}], "answers": [{"answer": "yes", "evidence": [2], "conditions": []}]},
        {"checks": [{"line": 1, "settled": True}], "answers": [{"answer": "within 6 weeks", "evidence": [3], "conditions": []}]},
        {"checks": [{"line": 1, "settled": False}], "answers": [{"answer": "yes", "evidence": [1], "conditions": [1]}]},
        {"checks": [{"line": 3, "settled": False}, {"line": 4, "settled": False}],
         "answers": [{"answer": "at least 6 years", "evidence": [2], "conditions": [3, 4]}]},
    ),
    "evidence": (
        {"relevant": [{"line": 2, "role": "answer"}, {"line": 3, "role": "exception"}],
         "answers": [{"answer": "yes", "evidence": [2], "conditions": []}]},
        {"relevant": [{"line": 1, "role": "requirement"}, {"line": 3, "role": "answer"}],
         "answers": [{"answer": "within 6 weeks", "evidence": [3], "conditions": []}]},
        {"relevant": [{"line": 1, "role": "answer"}, {"line": 2, "role": "background"}],
         "answers": [{"answer": "yes", "evidence": [1], "conditions": [1]}]},
        {"relevant": [{"line": 2, "role": "answer"}, {"line": 3, "role": "requirement"}, {"line": 4, "role": "requirement"},
                      {"line": 5, "role": "background"}],
         "answers": [{"answer": "at least 6 years", "evidence": [2], "conditions": [3, 4]}]},
    ),
}

#: Output budget: the reasoning variants need room before the answer.
MAX_TOKENS = {"extract": 768, "pairing": 768, "coc": 1536, "applicability": 1024, "evidence": 1024}
SYSTEMS = ("gold", "gold_nocondchk")
LOCAL = ("qwen7b", "llama8b")


def variant_sha(variant: str) -> str:
    blob = json.dumps({"system": VARIANTS[variant], "demos": DEMO_REPLIES[variant],
                       "demo_users": [u for u, _ in reader.DEMOS]}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def messages(q: cqa.Question, ev: reader.Evidence, variant: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    for (demo_user, _), reply in zip(reader.DEMOS, DEMO_REPLIES[variant]):
        msgs += [{"role": "user", "content": demo_user}, {"role": "assistant", "content": json.dumps(reply)}]
    msgs.append({"role": "user", "content": reader.user_prompt(q, ev)})
    return msgs


def cell_name(system: str, variant: str, model_key: str) -> str:
    return f"{system}_p{variant}__{model_key}"


def run_variant(system: str, variant: str, model_key: str, *, qset: str = "hosted", workers: int = 1,
                limit: int | None = None) -> dict:
    env = systems.Env()
    path = RUNS / qset / f"{cell_name(system, variant, model_key)}.jsonl"
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            path.write_bytes(raw[: raw.rfind(b"\n") + 1])
    done = read_cell(path)
    config = {"system": f"{system}_p{variant}", "model": model_id(model_key), "prompts": variant_sha(variant),
              "words": env.words, "strategy": env.strategy, "ranker": env.ranker}
    for rec in done.values():
        if {k: rec.get(k) for k in config} != config:
            raise RuntimeError(f"{path.name} was produced under a different configuration")
    qs = applicable(system, question_set(qset))[:limit]
    if system == "gold_nocondchk":       # only C2's population differs from ``gold``
        qs = [q for q in qs if q.kind == "conditional" and env.gold(q).condition_only_chunks]
    todo = [q for q in qs if q.id not in done]
    client = get_client()
    routes = ROUTES.get(model_key, (MODELS[model_key],))
    lock = threading.Lock()
    summary = {"cell": path.stem, "ran": 0, "failed": 0, "undefined": 0}
    started = time.time()

    def one(q: cqa.Question, fh) -> None:
        ctx = systems.build(system, q, env)
        if not ctx.chunks and not env.gold(q).all_chunks:     # as study.run: no annotated evidence at all
            with lock:
                summary["undefined"] += 1
            return
        ev = reader.render_evidence(ctx.chunks, drop_elements=ctx.drop_elements)
        res = tier = None
        text, pt, ct, lat = "", 0, 0, 0.0
        for use in routes:
            try:
                res = client.complete(messages=messages(q, ev, variant), system=VARIANTS[variant], tier=use,
                                      step=f"study_prompt_{variant}", response_format={"type": "json_object"},
                                      max_tokens=MAX_TOKENS[variant])
                tier = use
                break
            except TruncatedResponseError as exc:
                text, tier = f"[truncated] {exc}"[:300], use
                break
            except (RateLimitError, BackendError) as exc:
                if "token repeat limit" in str(exc):
                    text, tier = f"[degenerate] {exc}"[:300], use
                    break
                last = exc
        if tier is None:
            with lock:
                summary["failed"] += 1
            print(f"  {q.id}: {str(last)[:120]}", file=sys.stderr, flush=True)
            return
        if res is not None:
            text, pt, ct, lat = res.text, res.prompt_tokens, res.completion_tokens, res.latency_s
            parsed = reader.parse_output(text, ev)
        else:
            parsed = reader.Parsed((), ok=False)
        rec = {"qid": q.id, **config, "chunks": ctx.chunk_ids, "n_lines": ev.n_lines,
               "answers": [[a, list(c)] for a, c in parsed.answers], "evidence": list(parsed.evidence),
               "ok": parsed.ok, "bad_citations": parsed.bad_citations, "text": text,
               "prompt_tokens": pt, "completion_tokens": ct, "calls": 1, "latency_s": round(lat, 3),
               "served_by": served_by(tier), "trace": ctx.trace}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            summary["ran"] += 1
            if summary["ran"] % 50 == 0:
                print(f"  {path.stem}: {summary['ran']}/{len(todo)}, {(time.time() - started) / summary['ran']:.1f}s each",
                      file=sys.stderr, flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if workers <= 1:
            for q in todo:
                one(q, fh)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for f in as_completed([ex.submit(one, q, fh) for q in todo]):
                    f.result()
    summary["seconds"] = round(time.time() - started)
    return summary


def local() -> int:
    for m in LOCAL:
        for other in ("scopecompose-qwen7b", "scopecompose-reader"):
            subprocess.run(["ollama", "stop", other], capture_output=True)
        for variant in VARIANTS:
            for s in SYSTEMS:
                print(json.dumps(run_variant(s, variant, m)), flush=True)
    return 0


def hosted(model_key: str = "nemotron", workers: int = 1) -> int:
    """Repeat passes until every cell is complete: a question refused with HTTP 429 is skipped in one
    pass and asked again in the next."""
    for attempt in range(12):
        failed = 0
        try:
            for variant in VARIANTS:
                for s in SYSTEMS:
                    r = run_variant(s, variant, model_key, workers=workers)
                    failed += r["failed"]
                    print(json.dumps(r), flush=True)
        except Exception as exc:  # noqa: BLE001  (a spent minute quota: wait and resume)
            print(f"paused ({str(exc)[:120]})", file=sys.stderr, flush=True)
            failed += 1
        if not failed:
            return 0
        time.sleep(120)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=("local", "hosted", "smoke"))
    ap.add_argument("--model", default="nemotron", choices=sorted(MODELS))
    ap.add_argument("--workers", type=int, default=1, help="hosted only: requests in flight")
    args = ap.parse_args(argv)
    if args.which == "smoke":           # does every variant parse? (tuning questions only)
        for variant in VARIANTS:
            print(json.dumps(run_variant("gold", variant, args.model, qset="tuning", limit=12)), flush=True)
        return 0
    return local() if args.which == "local" else hosted(args.model, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
