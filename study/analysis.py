"""From cell files to numbers.

    python -m study.analysis --set tuning --model qwen7b        # a table of every cell present

Scores are recomputed from the saved answers every time, by the benchmark's own
script (``study.scoring``); nothing here calls a model.
"""

from __future__ import annotations

import argparse
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from study import cqa, scoring, systems
from study.run import RUNS, cell_path, read_cell

METRICS = ("em", "em_cond", "f1", "f1_cond")
STRATA = ("conditional", "unconditional", "unanswerable")


@dataclass(frozen=True)
class QScore:
    qid: str
    kind: str
    doc: int
    em: float
    em_cond: float
    f1: float
    f1_cond: float
    n_answers: int
    cond_correct: int
    cond_pred: int
    cond_gold: int
    ok: bool
    tokens: int
    latency_s: float
    cond_in_context: int = 0      # gold condition elements the generator was shown
    cond_noticed: int = 0         # ... of which it cited anywhere (evidence or conditions)


def _conditions_in_context(rec: dict, q: cqa.Question, words: int, strategy: str) -> set[str]:
    from study.retrieve import chunk_index

    from study.baselines import span_elements

    idx = chunk_index(words, strategy)
    shown = {el for cid in rec.get("chunks", []) for el in (span_elements(cid) or idx[cid].elements)}
    if rec.get("system") == "gold_nocondel":          # the oracle rung that deletes condition elements
        shown -= set(q.conditions)
    return shown & set(q.conditions)


def score_cell(path: Path) -> dict[str, QScore]:
    idx = cqa.by_id()
    out: dict[str, QScore] = {}
    for qid, rec in read_cell(path).items():
        q = idx[qid]
        pred = [(a, tuple(c)) for a, c in rec["answers"]]
        s = scoring.score(pred, q.reference())
        correct, n_pred, n_gold = scoring.condition_counts(pred, q.reference())
        in_ctx = _conditions_in_context(rec, q, rec.get("words", cqa.DEFAULT_CHUNK_WORDS), rec.get("strategy", "fixed"))
        cited = {c for _, cs in pred for c in cs} | set(rec.get("evidence", []))
        out[qid] = QScore(qid, q.kind, q.doc, s.em, s.em_cond, s.f1, s.f1_cond, len(pred), correct, n_pred, n_gold,
                          bool(rec.get("ok")), int(rec.get("prompt_tokens", 0)) + int(rec.get("completion_tokens", 0)),
                          float(rec.get("latency_s", 0.0)), len(in_ctx), len(in_ctx & cited))
    return out


def load(system: str, model_key: str, qset: str = "eval", env: systems.Env | None = None) -> dict[str, QScore]:
    return score_cell(cell_path(system, model_key, qset, env or systems.Env()))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return st.fmean(xs) if xs else float("nan")


def summarise(scores: Sequence[QScore]) -> dict:
    """Means of the four official metrics, plus how the conditions were handled."""
    if not scores:
        return {"n": 0}
    out = {"n": len(scores), **{m: mean(getattr(s, m) for s in scores) for m in METRICS}}
    correct, pred, gold = (sum(getattr(s, a) for s in scores) for a in ("cond_correct", "cond_pred", "cond_gold"))
    out["cond_precision"] = correct / pred if pred else float("nan")
    out["cond_recall"] = correct / gold if gold else float("nan")
    out["any_condition"] = mean(s.cond_pred > 0 for s in scores)       # share of answers that cite any condition
    shown = sum(s.cond_in_context for s in scores)
    out["cond_in_context"] = shown / gold if gold else float("nan")    # gold conditions that reached the generator
    out["cond_noticed"] = sum(s.cond_noticed for s in scores) / shown if shown else float("nan")
    out["no_answer"] = mean(s.n_answers == 0 for s in scores)
    out["parse_ok"] = mean(s.ok for s in scores)
    out["tokens"] = mean(s.tokens for s in scores)
    out["latency_s"] = mean(s.latency_s for s in scores)
    out["taxonomy"] = taxonomy(scores)
    return out


TAXONOMY = ("answer_wrong", "condition_omitted", "wrong_condition", "partial_conditions", "all_conditions",
            "spurious_condition", "no_condition_needed")


def error_class(s: QScore) -> str:
    """One label per answered question, from the official scores alone (analysis plan, "Error taxonomy").

    The answer counts as right when its answer-only F1 is at least 0.5, so a condition error is never
    blamed for a wrong answer."""
    if s.f1 < 0.5:
        return "answer_wrong"
    if s.kind != "conditional":
        return "spurious_condition" if s.cond_pred else "no_condition_needed"
    if s.cond_pred == 0:
        return "condition_omitted"
    if s.cond_correct == 0:
        return "wrong_condition"
    return "all_conditions" if s.cond_correct >= s.cond_gold else "partial_conditions"


def taxonomy(scores: Sequence[QScore]) -> dict[str, float]:
    """Share of answerable questions in each error class."""
    keep = [s for s in scores if s.kind != "unanswerable"]
    return {c: mean(error_class(s) == c for s in keep) for c in TAXONOMY} if keep else {}


def by_stratum(scores: dict[str, QScore], qids: Iterable[str] | None = None) -> dict[str, dict]:
    keep = [s for q, s in scores.items() if qids is None or q in set(qids)]
    out = {k: summarise([s for s in keep if s.kind == k]) for k in STRATA}
    out["all"] = summarise(keep)
    return out


def table(qset: str, model_key: str) -> str:
    rows = []
    for path in sorted((RUNS / qset).glob(f"*__{model_key}*.jsonl")):
        scores = score_cell(path)
        b = by_stratum(scores)
        name = path.stem.replace(f"__{model_key}", "")
        c, u, n, a = b["conditional"], b["unconditional"], b["unanswerable"], b["all"]

        def f(d, k):
            return f"{100 * d[k]:5.1f}" if d.get("n") else "    -"

        rows.append(
            f"{name:18s} n={a['n']:4d} | cond n={c['n']:3d} EM {f(c,'em')} F1 {f(c,'f1')} EMc {f(c,'em_cond')} F1c {f(c,'f1_cond')} "
            f"cP {f(c,'cond_precision')} cR {f(c,'cond_recall')} | uncond n={u['n']:3d} F1 {f(u,'f1')} F1c {f(u,'f1_cond')} "
            f"anyC {f(u,'any_condition')} | unans {f(n,'em')} | all F1 {f(a,'f1')} F1c {f(a,'f1_cond')} ok {f(a,'parse_ok')} tok {a['tokens']:.0f}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="qset", default="eval", choices=("tuning", "eval", "hosted"))
    ap.add_argument("--model", nargs="+", required=True)
    args = ap.parse_args(argv)
    for m in args.model:
        print(f"== {m} / {args.qset}")
        print(table(args.qset, m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
