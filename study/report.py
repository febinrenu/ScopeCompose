"""Every number the paper reports, computed from data and cell files.

    python -m study.report            # results/results.json and paper/numbers.tex

``results.json`` is the record; ``numbers.tex`` defines one LaTeX macro per
number the text uses, so no figure in the manuscript is typed by hand. A
section whose cells do not exist yet is skipped and its macros render as a
visible marker, so a draft can never silently show a stale number.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from pathlib import Path
from typing import Callable

import settings
from study import cqa, systems
from study.retrieve import RANKERS, load_rankings, rankings_path

RESULTS = settings.REPO_ROOT / "results" / "results.json"
NUMBERS = settings.REPO_ROOT / "paper" / "numbers.tex"


# --------------------------------------------------------------------------- #
# Dataset and chunking
# --------------------------------------------------------------------------- #


def dataset() -> dict:
    s = cqa.splits()
    ev = cqa.eval_questions()
    qs = cqa.questions()
    ch = cqa.chunks()
    from study import ramdocs

    rq = ramdocs.questions()
    return {
        "pages": len(cqa.documents()),
        "chunks": len(ch),
        "chunk_words": st.fmean(c.n_words for c in ch),
        "labelled_questions": len(qs),
        "labelled_conditional": sum(q.kind == "conditional" for q in qs),
        "tuning_pages": s["n_tuning_docs"],
        "tuning": len(s["tuning"]),
        "eval": len(ev),
        "eval_pages": len({q.doc for q in ev}),
        "eval_conditional": len(s["eval"]["conditional"]),
        "eval_unconditional": len(s["eval"]["unconditional"]),
        "eval_unanswerable": len(s["eval"]["unanswerable"]),
        "eval_opposite": sum(q.opposite_outcomes for q in ev),
        "ramdocs_questions": len(rq),
        "ramdocs_multi": sum(q.regime == "multi" for q in rq),
        "ramdocs_misinfo": sum(q.has_misinfo for q in rq),
    }


def separation(words: int = cqa.DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> dict:
    """How often the gold of a question does not fit in one chunk, over all labelled questions."""
    out = {}
    for kind in ("conditional", "unconditional"):
        locs = [cqa.locate_gold(q, words=words, strategy=strategy) for q in cqa.questions() if q.kind == kind]
        locs = [l for l in locs if l.all_chunks]
        row = {"n": len(locs), "separated": st.fmean(l.separated for l in locs),
               "mean_gold_chunks": st.fmean(len(l.all_chunks) for l in locs)}
        if kind == "conditional":
            row["condition_only"] = st.fmean(bool(l.condition_only_chunks) for l in locs)
            row["all_conditions_apart"] = st.fmean(bool(l.answer_chunks) and not (l.condition_chunks & l.answer_chunks)
                                                   for l in locs)
        out[kind] = row
    return out


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

KS = (3, 5, 8, 12, 20, 50)


def retrieval(ranker: str = "hybrid", words: int = cqa.DEFAULT_CHUNK_WORDS, strategy: str = "fixed") -> dict:
    """Per k: recall of answer chunks, condition chunks and condition-only chunks, on evaluation questions."""
    R = load_rankings(ranker, words, strategy)
    out: dict[str, dict] = {}
    cond = cqa.eval_questions(["conditional"])
    unc = cqa.eval_questions(["unconditional"])
    for k in KS:
        ans, cnd, only, all_c, full, lost_given_ans, u_ans = [], [], [], [], [], [], []
        for q in cond:
            g = cqa.locate_gold(q, words=words, strategy=strategy)
            top = set(R[q.id][:k])
            if g.answer_chunks:
                ans.append(len(g.answer_chunks & top) / len(g.answer_chunks))
                if g.answer_chunks <= top:
                    lost_given_ans.append(not g.condition_chunks <= top)
            cnd.append(len(g.condition_chunks & top) / len(g.condition_chunks))
            if g.condition_only_chunks:
                only.append(len(g.condition_only_chunks & top) / len(g.condition_only_chunks))
            all_c.append(g.condition_chunks <= top)
            full.append(g.all_chunks <= top)
        for q in unc:
            g = cqa.locate_gold(q, words=words, strategy=strategy)
            if g.answer_chunks:
                u_ans.append(len(g.answer_chunks & set(R[q.id][:k])) / len(g.answer_chunks))
        out[str(k)] = {"answer_recall": st.fmean(ans), "condition_recall": st.fmean(cnd),
                       "condition_only_recall": st.fmean(only), "all_conditions": st.fmean(all_c),
                       "all_gold": st.fmean(full), "condition_lost_given_answer": st.fmean(lost_given_ans),
                       "n_condition_lost_given_answer": len(lost_given_ans),
                       "unconditional_answer_recall": st.fmean(u_ans), "n_condition_only": len(only)}
    return out


def retrieval_all() -> dict:
    out = {}
    for ranker in RANKERS:
        for words, strategy in ((150, "fixed"), (100, "fixed"), (300, "fixed"), (150, "section")):
            if rankings_path(ranker, words, strategy).exists():
                out[f"{ranker}_{strategy}{words}"] = retrieval(ranker, words, strategy)
    return out


# --------------------------------------------------------------------------- #
# Operator survival (model-free operators; no generation involved)
# --------------------------------------------------------------------------- #

CQA_REDUCERS = ("rr8to3", "rr8to5", "nli", "crs")


def operator_survival() -> dict:
    """For each model-free reducer: how often it changes the top 8, and what share of the gold
    condition chunks and answer chunks that retrieval had found it removes (conditional questions)."""
    from study import operators  # noqa: F401  (registers the reducers)
    from study import ram

    env = systems.Env()
    qs = cqa.eval_questions(["conditional"])
    out: dict[str, dict] = {"cqa": {}, "ramdocs": {}}
    for name in CQA_REDUCERS:
        changed, kept_n, cond_drop, ans_drop, any_cond_lost = [], [], [], [], []
        for q in qs:
            g = env.gold(q)
            top8 = {c.id for c in env.ranked(q)[:8]}
            kept = set(systems.build(name, q, env).chunk_ids)
            changed.append(kept != top8)
            kept_n.append(len(kept))
            pc, pa = g.condition_chunks & top8, g.answer_chunks & top8
            if pc:
                cond_drop.append(len(pc - kept) / len(pc))
                any_cond_lost.append(bool(pc - kept))
            if pa:
                ans_drop.append(len(pa - kept) / len(pa))
        out["cqa"][name] = {"changed": st.fmean(changed), "kept": st.fmean(kept_n),
                            "condition_dropped": st.fmean(cond_drop), "answer_dropped": st.fmean(ans_drop),
                            "any_condition_lost": st.fmean(any_cond_lost), "n_condition": len(cond_drop),
                            "n_answer": len(ans_drop)}
    for name in ("rr3", "nli", "crs", "oracle"):
        out["ramdocs"][name] = ram.survival(name)
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

SECTIONS: dict[str, Callable[[], dict]] = {
    "dataset": dataset,
    "separation": lambda: {f"{s}{w}": separation(w, s) for w, s in ((150, "fixed"), (100, "fixed"),
                                                                    (300, "fixed"), (150, "section"))},
    "retrieval": retrieval_all,
    "operator_survival": operator_survival,
}


def build(sections: dict[str, Callable[[], dict]] | None = None) -> dict:
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for name, fn in (sections or SECTIONS).items():
        try:
            results[name] = fn()
        except FileNotFoundError as exc:          # cells not run yet: keep the section out, say so
            print(f"skip {name}: {exc}")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return results


# --------------------------------------------------------------------------- #
# numbers.tex
# --------------------------------------------------------------------------- #


def _get(results: dict, path: str):
    cur = results
    for part in path.split("."):
        cur = cur[part]
    return cur


def pct(x: float, places: int = 1) -> str:
    return f"{100 * x:.{places}f}"


def integer(x) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def one(x: float) -> str:
    return f"{x:.1f}"


#: macro name -> (path into results.json, formatter)
MACROS: dict[str, tuple[str, Callable]] = {
    "nPages": ("dataset.pages", integer),
    "nChunks": ("dataset.chunks", integer),
    "chunkWords": ("dataset.chunk_words", lambda x: f"{x:.0f}"),
    "nLabelled": ("dataset.labelled_questions", integer),
    "nLabelledCond": ("dataset.labelled_conditional", integer),
    "nTuningPages": ("dataset.tuning_pages", integer),
    "nTuning": ("dataset.tuning", integer),
    "nEval": ("dataset.eval", integer),
    "nEvalPages": ("dataset.eval_pages", integer),
    "nEvalCond": ("dataset.eval_conditional", integer),
    "nEvalUncond": ("dataset.eval_unconditional", integer),
    "nEvalUnans": ("dataset.eval_unanswerable", integer),
    "nEvalOpposite": ("dataset.eval_opposite", integer),
    "nRamQ": ("dataset.ramdocs_questions", integer),
    "nRamMulti": ("dataset.ramdocs_multi", integer),
    "nRamMisinfo": ("dataset.ramdocs_misinfo", integer),
    "sepCond": ("separation.fixed150.conditional.separated", pct),
    "sepUncond": ("separation.fixed150.unconditional.separated", pct),
    "condOnly": ("separation.fixed150.conditional.condition_only", pct),
    "condApart": ("separation.fixed150.conditional.all_conditions_apart", pct),
    "goldChunksCond": ("separation.fixed150.conditional.mean_gold_chunks", lambda x: f"{x:.2f}"),
    "goldChunksUncond": ("separation.fixed150.unconditional.mean_gold_chunks", lambda x: f"{x:.2f}"),
    "sepCondHundred": ("separation.fixed100.conditional.separated", pct),
    "sepCondThreeHundred": ("separation.fixed300.conditional.separated", pct),
    "sepCondSection": ("separation.section150.conditional.separated", pct),
    "recAnsEight": ("retrieval.hybrid_fixed150.8.answer_recall", pct),
    "recCondEight": ("retrieval.hybrid_fixed150.8.condition_recall", pct),
    "recCondOnlyEight": ("retrieval.hybrid_fixed150.8.condition_only_recall", pct),
    "recAllGoldEight": ("retrieval.hybrid_fixed150.8.all_gold", pct),
    "recLostGivenAnsEight": ("retrieval.hybrid_fixed150.8.condition_lost_given_answer", pct),
    "recUncondAnsEight": ("retrieval.hybrid_fixed150.8.unconditional_answer_recall", pct),
    "nliChanged": ("operator_survival.cqa.nli.changed", pct),
    "nliCondDrop": ("operator_survival.cqa.nli.condition_dropped", pct),
    "nliAnsDrop": ("operator_survival.cqa.nli.answer_dropped", pct),
    "crsChanged": ("operator_survival.cqa.crs.changed", pct),
    "crsCondDrop": ("operator_survival.cqa.crs.condition_dropped", pct),
    "crsAnsDrop": ("operator_survival.cqa.crs.answer_dropped", pct),
    "rrCondDrop": ("operator_survival.cqa.rr8to3.condition_dropped", pct),
    "rrAnsDrop": ("operator_survival.cqa.rr8to3.answer_dropped", pct),
    "nOpCond": ("operator_survival.cqa.nli.n_condition", integer),
    "ramRrCorrect": ("operator_survival.ramdocs.rr3.correct", pct),
    "ramRrMisinfo": ("operator_survival.ramdocs.rr3.misinfo", pct),
    "ramRrNoise": ("operator_survival.ramdocs.rr3.noise", pct),
    "ramNliCorrect": ("operator_survival.ramdocs.nli.correct", pct),
    "ramNliMisinfo": ("operator_survival.ramdocs.nli.misinfo", pct),
    "ramCrsCorrect": ("operator_survival.ramdocs.crs.correct", pct),
    "ramCrsMisinfo": ("operator_survival.ramdocs.crs.misinfo", pct),
}


def _macro_name_ok(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+", name))


def write_numbers(results: dict, extra: dict[str, str] | None = None) -> int:
    lines = ["% Generated by study/report.py from results/results.json. Do not edit by hand."]
    missing = 0
    for name, (path, fmt) in MACROS.items():
        assert _macro_name_ok(name), name
        try:
            value = fmt(_get(results, path))
        except (KeyError, TypeError):
            value, missing = r"\textbf{??}", missing + 1
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    for name, value in (extra or {}).items():
        assert _macro_name_ok(name), name
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return missing


def main() -> int:
    results = build()
    missing = write_numbers(results)
    print(f"wrote {RESULTS.relative_to(settings.REPO_ROOT)} and {NUMBERS.relative_to(settings.REPO_ROOT)}"
          f" ({len(MACROS)} macros, {missing} not yet available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
