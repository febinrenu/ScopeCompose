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
        "chunk_words_section": st.fmean(c.n_words for c in cqa.chunks(cqa.DEFAULT_CHUNK_WORDS, "section")),
        "chunk_words_100": st.fmean(c.n_words for c in cqa.chunks(100, "fixed")),
        "chunk_words_300": st.fmean(c.n_words for c in cqa.chunks(300, "fixed")),
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
        "hosted_questions": len(cqa.hosted_questions()),
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
# Prompt tuning (tuning set only; gold context)
# --------------------------------------------------------------------------- #

PROMPT_DIRS = {"v1": "tuning_prompt_v1", "v2": "tuning_prompt_v2", "v3": "tuning_prompt_v3", "v4": "tuning"}


def tuning() -> dict:
    """``gold`` on the tuning set under each prompt wording, per model; hosted models on the probe sample."""
    from study.analysis import score_cell, summarise
    from study.probe import probe_questions
    from study.run import RUNS

    probe = {q.id for q in probe_questions()}
    out: dict[str, dict] = {}
    for prompt, d in PROMPT_DIRS.items():
        for path in sorted((RUNS / d).glob("gold__*.jsonl")):
            model = path.stem.split("__")[1]
            sc = score_cell(path)
            if model in ("qwen27b", "oss20b", "oss120b"):
                sc = {k: v for k, v in sc.items() if k in probe}
            row = {}
            for kind in ("conditional", "unconditional"):
                s = summarise([v for v in sc.values() if v.kind == kind])
                row[kind] = {k: s.get(k) for k in ("n", "f1", "f1_cond", "cond_precision", "cond_recall",
                                                    "any_condition", "cond_in_context", "cond_noticed")}
            row["all"] = {k: summarise(list(sc.values())).get(k) for k in ("n", "f1", "f1_cond")}
            out.setdefault(prompt, {})[model] = row
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Generation: end-to-end cells, the confirmatory family, RAMDocs
# --------------------------------------------------------------------------- #

LOCAL_MODELS = ("qwen7b", "llama8b")
HOSTED_MODELS = ("oss20b", "gemma31b", "nemotron")


def end_to_end() -> dict:
    """Per question set, model and system: the official metrics by stratum, plus the rule-and-exception subset."""
    from study.analysis import by_stratum, score_cell, summarise
    from study.run import RUNS

    opposite = {q.id for q in cqa.questions() if q.kind == "conditional" and q.opposite_outcomes}
    out: dict = {}
    for qset, models in (("eval", LOCAL_MODELS), ("hosted", HOSTED_MODELS)):
        for m in models:
            for path in sorted((RUNS / qset).glob(f"*__{m}.jsonl")):
                scores = score_cell(path)
                if not scores:
                    continue
                row = by_stratum(scores)
                row["opposite"] = summarise([s for q, s in scores.items() if q in opposite])
                out.setdefault(qset, {}).setdefault(m, {})[path.stem.split("__")[0]] = row
    if not out:
        raise FileNotFoundError("no evaluation or hosted cells yet")
    return out


def confirmatory() -> dict:
    from study.confirm import confirmatory as run

    return run()


def ramdocs_results() -> dict:
    """RAMDocs per model and system: strict accuracy, gold recall, wrong-answer rate, overall and with misinformation,
    and each reducer's paired change in strict accuracy against full context (questions are independent here)."""
    from study import ram, ramdocs, stats

    misinfo = {q.id for q in ramdocs.questions() if q.has_misinfo}

    def summ(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        return {"n": len(rs), "correct": st.fmean(r["correct"] for r in rs),
                "gold_recall": st.fmean(r["gold_recall"] for r in rs), "any_wrong": st.fmean(r["any_wrong"] for r in rs),
                "kept": st.fmean(len(r["kept"]) for r in rs),
                "tokens": st.fmean(r["prompt_tokens"] + r["completion_tokens"] for r in rs),
                "calls": st.fmean(r["calls"] for r in rs)}

    out: dict = {}
    for path in sorted(ram.RUNS.glob("*__*.jsonl")):
        system, m = path.stem.split("__")
        recs = ram.read_cell(path)
        if recs:
            out.setdefault(m, {})[system] = {"cells": recs}
    if not out:
        raise FileNotFoundError("no RAMDocs cells yet")
    for m, systems_ in out.items():
        full = systems_.get("full", {}).get("cells", {})
        for system, d in systems_.items():
            recs = d.pop("cells")
            d["all"] = summ(list(recs.values()))
            d["misinfo"] = summ([r for q, r in recs.items() if q in misinfo])
            d["no_misinfo"] = summ([r for q, r in recs.items() if q not in misinfo])
            if system != "full" and full:
                for name, keep in (("vs_full", None), ("vs_full_misinfo", misinfo)):
                    a = {q: float(r["correct"]) for q, r in recs.items() if keep is None or q in keep}
                    b = {q: float(r["correct"]) for q, r in full.items() if keep is None or q in keep}
                    if set(a) & set(b):
                        d[name] = stats.paired(a, b, {q: i for i, q in enumerate(sorted(set(a) | set(b)))})
    return out


def providers() -> dict:
    """Which provider served each hosted answer (deviations 1-3), and gpt-oss-20b's gold-context F1c by provider."""
    from collections import Counter

    from study.analysis import score_cell
    from study.run import RUNS, read_cell

    out: dict = {}
    for m in HOSTED_MODELS:
        counts: Counter = Counter()
        for path in sorted((RUNS / "hosted").glob(f"*__{m}.jsonl")):
            counts.update(r.get("served_by", "groq:" + str(r.get("model"))).split(":")[0] for r in read_cell(path).values())
        if counts:
            out[m] = {"counts": dict(counts), "total": sum(counts.values()),
                      "groq": counts.get("groq", 0), "nvidia": counts.get("nvidia", 0) + counts.get("nvidia2", 0)}
    gold = RUNS / "hosted" / "gold__oss20b.jsonl"
    if gold.exists():
        recs, scores = read_cell(gold), score_cell(gold)
        by: dict = {}
        for qid, s in scores.items():
            if s.kind == "conditional":
                prov = recs[qid].get("served_by", "groq").split(":")[0]
                by.setdefault(prov, []).append(s.f1_cond)
        out.setdefault("oss20b", {})["gold_f1c_conditional_by_provider"] = {k: {"n": len(v), "f1_cond": st.fmean(v)}
                                                                          for k, v in by.items()}
    if not out:
        raise FileNotFoundError("no hosted cells yet")
    return out


def qualifier_section() -> dict:
    """The completion scorer on its own: the tuned threshold, pair-level recovery of missed condition
    chunks on the evaluation questions, threshold-free AUC over every candidate pair, and the
    out-of-distribution check on the project's own corpus (``study.ood``)."""
    from study import qualifier

    th = json.loads(qualifier.THRESHOLD_FILE.read_text(encoding="utf-8"))
    ev = [q.id for q in cqa.eval_questions()]
    env, idx, scores = systems.Env(), cqa.by_id(), qualifier.load_scores()
    out: dict = {"threshold": th}
    for kind in ("trained", "zeroshot"):
        row = qualifier.intrinsic(kind, ev, th[kind])
        pos, neg = [], []
        for qid in ev:
            s = scores[kind].get(qid, {})
            for p in qualifier.candidate_pairs(idx[qid], env):
                if p.candidate in s:
                    (pos if p.label else neg).append(s[p.candidate])
        neg_sorted = sorted(neg)
        import bisect
        wins = sum(bisect.bisect_left(neg_sorted, x) + 0.5 * (bisect.bisect_right(neg_sorted, x) - bisect.bisect_left(neg_sorted, x))
                   for x in pos)
        row.update({"auc": wins / (len(pos) * len(neg)) if pos and neg else float("nan"),
                    "n_pos_pairs": len(pos), "n_neg_pairs": len(neg)})
        out[kind] = row
    ood = qualifier.MODELS / "ood.json"
    if ood.exists():
        out["ood"] = json.loads(ood.read_text(encoding="utf-8"))
    return out


SECTIONS: dict[str, Callable[[], dict]] = {
    "dataset": dataset,
    "separation": lambda: {f"{s}{w}": separation(w, s) for w, s in ((150, "fixed"), (100, "fixed"),
                                                                    (300, "fixed"), (150, "section"))},
    "retrieval": retrieval_all,
    "operator_survival": operator_survival,
    "tuning": tuning,
    "end_to_end": end_to_end,
    "confirmatory": confirmatory,
    "ramdocs": ramdocs_results,
    "qualifier": qualifier_section,
    "providers": providers,
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


def _pts(x: float) -> str:
    """A signed difference in F1 points, with a typographic minus."""
    v = round(100 * x, 1)
    return "0.0" if v == 0 else f"{v:+.1f}".replace("-", "$-$")


def effect(d: dict) -> str:
    """A paired difference: points, 95% interval, and the Holm-adjusted p when the family was complete."""
    p = d.get("p_holm", d["p"])
    pv = "<0.001" if p < 0.001 else f"={p:.3f}"
    lab = r"p_{\mathrm{Holm}}" if "p_holm" in d else "p"
    return f"{_pts(d['diff'])} points (95\% CI {_pts(d['low'])} to {_pts(d['high'])}; ${lab}{pv}$)"


def lower_bound(d: dict) -> str:
    return _pts(d["low"])


#: macro name -> (path into results.json, formatter)
MACROS: dict[str, tuple[str, Callable]] = {
    "nPages": ("dataset.pages", integer),
    "nChunks": ("dataset.chunks", integer),
    "chunkWords": ("dataset.chunk_words", lambda x: f"{x:.0f}"),
    "chunkWordsSection": ("dataset.chunk_words_section", lambda x: f"{x:.0f}"),
    "chunkWordsHundred": ("dataset.chunk_words_100", lambda x: f"{x:.0f}"),
    "chunkWordsThreeHundred": ("dataset.chunk_words_300", lambda x: f"{x:.0f}"),
    "tunNoticedQwen": ("tuning.v4.qwen7b.conditional.cond_noticed", pct),
    "tunNoticedLlama": ("tuning.v4.llama8b.conditional.cond_noticed", pct),
    "tunNoticedBig": ("tuning.v4.qwen27b.conditional.cond_noticed", pct),
    "tunRecallQwen": ("tuning.v4.qwen7b.conditional.cond_recall", pct),
    "tunRecallLlama": ("tuning.v4.llama8b.conditional.cond_recall", pct),
    "tunRecallBig": ("tuning.v4.qwen27b.conditional.cond_recall", pct),
    "tunFcCondQwen": ("tuning.v4.qwen7b.conditional.f1_cond", pct),
    "tunFcCondLlama": ("tuning.v4.llama8b.conditional.f1_cond", pct),
    "tunFcCondBig": ("tuning.v4.qwen27b.conditional.f1_cond", pct),
    "tunAnyCondUncondVOne": ("tuning.v1.qwen7b.unconditional.any_condition", pct),
    "tunAnyCondUncondVOneLlama": ("tuning.v1.llama8b.unconditional.any_condition", pct),
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
    "nHostedQ": ("dataset.hosted_questions", integer),
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
    # -- generation results (study.confirm, end_to_end, qualifier) -- #
    "ramOracleEffAllQwen": ("ramdocs.qwen7b.oracle.vs_full", effect),
    "ramRerankEffAllQwen": ("ramdocs.qwen7b.rr3.vs_full", effect),
    "ramNliEffAllQwen": ("ramdocs.qwen7b.nli.vs_full", effect),
    "ramCrsEffAllQwen": ("ramdocs.qwen7b.crs.vs_full", effect),
    "ramSelectEffAllQwen": ("ramdocs.qwen7b.select.vs_full", effect),
    "ramGradeEffAllQwen": ("ramdocs.qwen7b.grade.vs_full", effect),
    "ramOracleEffAllLlama": ("ramdocs.llama8b.oracle.vs_full", effect),
    "ramRerankEffAllLlama": ("ramdocs.llama8b.rr3.vs_full", effect),
    "ramNliEffAllLlama": ("ramdocs.llama8b.nli.vs_full", effect),
    "ramCrsEffAllLlama": ("ramdocs.llama8b.crs.vs_full", effect),
    "ramSelectEffAllLlama": ("ramdocs.llama8b.select.vs_full", effect),
    "ramGradeEffAllLlama": ("ramdocs.llama8b.grade.vs_full", effect),
    "nCondOnlyEval": ("confirmatory.qwen7b.family.C2.population", integer),
    "fcClosedUncondQwen": ("end_to_end.eval.qwen7b.closed.unconditional.f1_cond", pct),
    "anyCondTopUncondQwen": ("end_to_end.eval.qwen7b.top8.unconditional.any_condition", pct),
    "fcSelectQwen": ("end_to_end.eval.qwen7b.select.conditional.f1_cond", pct),
    "fcGoldUncondQwen": ("end_to_end.eval.qwen7b.gold.unconditional.f1_cond", pct),
    "eSevenQwen": ("confirmatory.qwen7b.exploratory.E7", effect),
    "eSixQwen": ("confirmatory.qwen7b.exploratory.E6", effect),
    "nOppositeQwen": ("confirmatory.qwen7b.exploratory.E6.population", integer),
    "fcClosedUncondLlama": ("end_to_end.eval.llama8b.closed.unconditional.f1_cond", pct),
    "anyCondTopUncondLlama": ("end_to_end.eval.llama8b.top8.unconditional.any_condition", pct),
    "fcSelectLlama": ("end_to_end.eval.llama8b.select.conditional.f1_cond", pct),
    "fcGoldUncondLlama": ("end_to_end.eval.llama8b.gold.unconditional.f1_cond", pct),
    "eSevenLlama": ("confirmatory.llama8b.exploratory.E7", effect),
    "eSixLlama": ("confirmatory.llama8b.exploratory.E6", effect),
    "nOppositeLlama": ("confirmatory.llama8b.exploratory.E6.population", integer),
    "hServedGroqOss": ("providers.oss20b.groq", integer),
    "hServedNvidiaOss": ("providers.oss20b.nvidia", integer),
    "hServedTotalOss": ("providers.oss20b.total", integer),
    "hCondOnlyN": ("confirmatory.gemma31b.family.C2.population", integer),
    "hCEightOss": ("confirmatory.oss20b.C8", effect),
    "hCEightGemma": ("confirmatory.gemma31b.C8", effect),
    "hCEightNemo": ("confirmatory.nemotron.C8", effect),
    "ramFullAccQwen": ("ramdocs.qwen7b.full.all.correct", pct),
    "ramFullMiQwen": ("ramdocs.qwen7b.full.misinfo.correct", pct),
    "ramFullWrongQwen": ("ramdocs.qwen7b.full.misinfo.any_wrong", pct),
    "ramOracleAccQwen": ("ramdocs.qwen7b.oracle.all.correct", pct),
    "ramOracleMiQwen": ("ramdocs.qwen7b.oracle.misinfo.correct", pct),
    "ramOracleWrongQwen": ("ramdocs.qwen7b.oracle.misinfo.any_wrong", pct),
    "ramOracleEffQwen": ("ramdocs.qwen7b.oracle.vs_full_misinfo", effect),
    "ramRerankAccQwen": ("ramdocs.qwen7b.rr3.all.correct", pct),
    "ramRerankMiQwen": ("ramdocs.qwen7b.rr3.misinfo.correct", pct),
    "ramRerankWrongQwen": ("ramdocs.qwen7b.rr3.misinfo.any_wrong", pct),
    "ramRerankEffQwen": ("ramdocs.qwen7b.rr3.vs_full_misinfo", effect),
    "ramNliAccQwen": ("ramdocs.qwen7b.nli.all.correct", pct),
    "ramNliMiQwen": ("ramdocs.qwen7b.nli.misinfo.correct", pct),
    "ramNliWrongQwen": ("ramdocs.qwen7b.nli.misinfo.any_wrong", pct),
    "ramNliEffQwen": ("ramdocs.qwen7b.nli.vs_full_misinfo", effect),
    "ramCrsAccQwen": ("ramdocs.qwen7b.crs.all.correct", pct),
    "ramCrsMiQwen": ("ramdocs.qwen7b.crs.misinfo.correct", pct),
    "ramCrsWrongQwen": ("ramdocs.qwen7b.crs.misinfo.any_wrong", pct),
    "ramCrsEffQwen": ("ramdocs.qwen7b.crs.vs_full_misinfo", effect),
    "ramSelectAccQwen": ("ramdocs.qwen7b.select.all.correct", pct),
    "ramSelectMiQwen": ("ramdocs.qwen7b.select.misinfo.correct", pct),
    "ramSelectWrongQwen": ("ramdocs.qwen7b.select.misinfo.any_wrong", pct),
    "ramSelectEffQwen": ("ramdocs.qwen7b.select.vs_full_misinfo", effect),
    "ramGradeAccQwen": ("ramdocs.qwen7b.grade.all.correct", pct),
    "ramGradeMiQwen": ("ramdocs.qwen7b.grade.misinfo.correct", pct),
    "ramGradeWrongQwen": ("ramdocs.qwen7b.grade.misinfo.any_wrong", pct),
    "ramGradeEffQwen": ("ramdocs.qwen7b.grade.vs_full_misinfo", effect),
    "ramFullAccLlama": ("ramdocs.llama8b.full.all.correct", pct),
    "ramFullMiLlama": ("ramdocs.llama8b.full.misinfo.correct", pct),
    "ramFullWrongLlama": ("ramdocs.llama8b.full.misinfo.any_wrong", pct),
    "ramOracleAccLlama": ("ramdocs.llama8b.oracle.all.correct", pct),
    "ramOracleMiLlama": ("ramdocs.llama8b.oracle.misinfo.correct", pct),
    "ramOracleWrongLlama": ("ramdocs.llama8b.oracle.misinfo.any_wrong", pct),
    "ramOracleEffLlama": ("ramdocs.llama8b.oracle.vs_full_misinfo", effect),
    "ramRerankAccLlama": ("ramdocs.llama8b.rr3.all.correct", pct),
    "ramRerankMiLlama": ("ramdocs.llama8b.rr3.misinfo.correct", pct),
    "ramRerankWrongLlama": ("ramdocs.llama8b.rr3.misinfo.any_wrong", pct),
    "ramRerankEffLlama": ("ramdocs.llama8b.rr3.vs_full_misinfo", effect),
    "ramNliAccLlama": ("ramdocs.llama8b.nli.all.correct", pct),
    "ramNliMiLlama": ("ramdocs.llama8b.nli.misinfo.correct", pct),
    "ramNliWrongLlama": ("ramdocs.llama8b.nli.misinfo.any_wrong", pct),
    "ramNliEffLlama": ("ramdocs.llama8b.nli.vs_full_misinfo", effect),
    "ramCrsAccLlama": ("ramdocs.llama8b.crs.all.correct", pct),
    "ramCrsMiLlama": ("ramdocs.llama8b.crs.misinfo.correct", pct),
    "ramCrsWrongLlama": ("ramdocs.llama8b.crs.misinfo.any_wrong", pct),
    "ramCrsEffLlama": ("ramdocs.llama8b.crs.vs_full_misinfo", effect),
    "ramSelectAccLlama": ("ramdocs.llama8b.select.all.correct", pct),
    "ramSelectMiLlama": ("ramdocs.llama8b.select.misinfo.correct", pct),
    "ramSelectWrongLlama": ("ramdocs.llama8b.select.misinfo.any_wrong", pct),
    "ramSelectEffLlama": ("ramdocs.llama8b.select.vs_full_misinfo", effect),
    "ramGradeAccLlama": ("ramdocs.llama8b.grade.all.correct", pct),
    "ramGradeMiLlama": ("ramdocs.llama8b.grade.misinfo.correct", pct),
    "ramGradeWrongLlama": ("ramdocs.llama8b.grade.misinfo.any_wrong", pct),
    "ramGradeEffLlama": ("ramdocs.llama8b.grade.vs_full_misinfo", effect),
    "fcGoldQwen": ("end_to_end.eval.qwen7b.gold.conditional.f1_cond", pct),
    "fcNoCondElQwen": ("end_to_end.eval.qwen7b.gold_nocondel.conditional.f1_cond", pct),
    "fcPadQwen": ("end_to_end.eval.qwen7b.goldpad8.conditional.f1_cond", pct),
    "fcTopQwen": ("end_to_end.eval.qwen7b.top8.conditional.f1_cond", pct),
    "fcClosedQwen": ("end_to_end.eval.qwen7b.closed.conditional.f1_cond", pct),
    "fcRerankQwen": ("end_to_end.eval.qwen7b.rr8to3.conditional.f1_cond", pct),
    "fcNliQwen": ("end_to_end.eval.qwen7b.nli.conditional.f1_cond", pct),
    "fcCrsQwen": ("end_to_end.eval.qwen7b.crs.conditional.f1_cond", pct),
    "fcQcQwen": ("end_to_end.eval.qwen7b.qc.conditional.f1_cond", pct),
    "fcAdjQwen": ("end_to_end.eval.qwen7b.adj.conditional.f1_cond", pct),
    "fcPageQwen": ("end_to_end.eval.qwen7b.page.conditional.f1_cond", pct),
    "noticedGoldQwen": ("end_to_end.eval.qwen7b.gold.conditional.cond_noticed", pct),
    "recallGoldQwen": ("end_to_end.eval.qwen7b.gold.conditional.cond_recall", pct),
    "noticedTopQwen": ("end_to_end.eval.qwen7b.top8.conditional.cond_noticed", pct),
    "fcTopUncondQwen": ("end_to_end.eval.qwen7b.top8.unconditional.f1_cond", pct),
    "fcQcUncondQwen": ("end_to_end.eval.qwen7b.qc.unconditional.f1_cond", pct),
    "cOneQwen": ("confirmatory.qwen7b.family.C1", effect),
    "cTwoQwen": ("confirmatory.qwen7b.family.C2", effect),
    "cThreeQwen": ("confirmatory.qwen7b.family.C3", effect),
    "cFourQwen": ("confirmatory.qwen7b.family.C4", effect),
    "cFiveQwen": ("confirmatory.qwen7b.family.C5", effect),
    "cSixQwen": ("confirmatory.qwen7b.family.C6", effect),
    "cSevenQwen": ("confirmatory.qwen7b.family.C7", effect),
    "cEightQwen": ("confirmatory.qwen7b.C8", effect),
    "cEightLowQwen": ("confirmatory.qwen7b.C8", lower_bound),
    "fcGoldLlama": ("end_to_end.eval.llama8b.gold.conditional.f1_cond", pct),
    "fcNoCondElLlama": ("end_to_end.eval.llama8b.gold_nocondel.conditional.f1_cond", pct),
    "fcPadLlama": ("end_to_end.eval.llama8b.goldpad8.conditional.f1_cond", pct),
    "fcTopLlama": ("end_to_end.eval.llama8b.top8.conditional.f1_cond", pct),
    "fcClosedLlama": ("end_to_end.eval.llama8b.closed.conditional.f1_cond", pct),
    "fcRerankLlama": ("end_to_end.eval.llama8b.rr8to3.conditional.f1_cond", pct),
    "fcNliLlama": ("end_to_end.eval.llama8b.nli.conditional.f1_cond", pct),
    "fcCrsLlama": ("end_to_end.eval.llama8b.crs.conditional.f1_cond", pct),
    "fcQcLlama": ("end_to_end.eval.llama8b.qc.conditional.f1_cond", pct),
    "fcAdjLlama": ("end_to_end.eval.llama8b.adj.conditional.f1_cond", pct),
    "fcPageLlama": ("end_to_end.eval.llama8b.page.conditional.f1_cond", pct),
    "noticedGoldLlama": ("end_to_end.eval.llama8b.gold.conditional.cond_noticed", pct),
    "recallGoldLlama": ("end_to_end.eval.llama8b.gold.conditional.cond_recall", pct),
    "noticedTopLlama": ("end_to_end.eval.llama8b.top8.conditional.cond_noticed", pct),
    "fcTopUncondLlama": ("end_to_end.eval.llama8b.top8.unconditional.f1_cond", pct),
    "fcQcUncondLlama": ("end_to_end.eval.llama8b.qc.unconditional.f1_cond", pct),
    "cOneLlama": ("confirmatory.llama8b.family.C1", effect),
    "cTwoLlama": ("confirmatory.llama8b.family.C2", effect),
    "cThreeLlama": ("confirmatory.llama8b.family.C3", effect),
    "cFourLlama": ("confirmatory.llama8b.family.C4", effect),
    "cFiveLlama": ("confirmatory.llama8b.family.C5", effect),
    "cSixLlama": ("confirmatory.llama8b.family.C6", effect),
    "cSevenLlama": ("confirmatory.llama8b.family.C7", effect),
    "cEightLlama": ("confirmatory.llama8b.C8", effect),
    "cEightLowLlama": ("confirmatory.llama8b.C8", lower_bound),
    "hGoldOss": ("end_to_end.hosted.oss20b.gold.conditional.f1_cond", pct),
    "hNoCondChkOss": ("end_to_end.hosted.oss20b.gold_nocondchk.conditional.f1_cond", pct),
    "hTopOss": ("end_to_end.hosted.oss20b.top8.conditional.f1_cond", pct),
    "hQcOss": ("end_to_end.hosted.oss20b.qc.conditional.f1_cond", pct),
    "hNoticedOss": ("end_to_end.hosted.oss20b.gold.conditional.cond_noticed", pct),
    "hRecallOss": ("end_to_end.hosted.oss20b.gold.conditional.cond_recall", pct),
    "hCTwoOss": ("confirmatory.oss20b.family.C2", effect),
    "hCSevenOss": ("confirmatory.oss20b.family.C7", effect),
    "hEOneOss": ("confirmatory.oss20b.exploratory.E1", effect),
    "hGoldGemma": ("end_to_end.hosted.gemma31b.gold.conditional.f1_cond", pct),
    "hNoCondChkGemma": ("end_to_end.hosted.gemma31b.gold_nocondchk.conditional.f1_cond", pct),
    "hTopGemma": ("end_to_end.hosted.gemma31b.top8.conditional.f1_cond", pct),
    "hQcGemma": ("end_to_end.hosted.gemma31b.qc.conditional.f1_cond", pct),
    "hNoticedGemma": ("end_to_end.hosted.gemma31b.gold.conditional.cond_noticed", pct),
    "hRecallGemma": ("end_to_end.hosted.gemma31b.gold.conditional.cond_recall", pct),
    "hCTwoGemma": ("confirmatory.gemma31b.family.C2", effect),
    "hCSevenGemma": ("confirmatory.gemma31b.family.C7", effect),
    "hEOneGemma": ("confirmatory.gemma31b.exploratory.E1", effect),
    "hGoldNemo": ("end_to_end.hosted.nemotron.gold.conditional.f1_cond", pct),
    "hNoCondChkNemo": ("end_to_end.hosted.nemotron.gold_nocondchk.conditional.f1_cond", pct),
    "hTopNemo": ("end_to_end.hosted.nemotron.top8.conditional.f1_cond", pct),
    "hQcNemo": ("end_to_end.hosted.nemotron.qc.conditional.f1_cond", pct),
    "hNoticedNemo": ("end_to_end.hosted.nemotron.gold.conditional.cond_noticed", pct),
    "hRecallNemo": ("end_to_end.hosted.nemotron.gold.conditional.cond_recall", pct),
    "hCTwoNemo": ("confirmatory.nemotron.family.C2", effect),
    "hCSevenNemo": ("confirmatory.nemotron.family.C7", effect),
    "hEOneNemo": ("confirmatory.nemotron.exploratory.E1", effect),
    "qAucTrained": ("qualifier.trained.auc", lambda x: f'{x:.2f}'),
    "qPrecTrained": ("qualifier.trained.precision", pct),
    "qRecTrained": ("qualifier.trained.recall", pct),
    "qAddedTrained": ("qualifier.trained.added_per_q", lambda x: f'{x:.2f}'),
    "qAucZero": ("qualifier.zeroshot.auc", lambda x: f'{x:.2f}'),
    "qPrecZero": ("qualifier.zeroshot.precision", pct),
    "qRecZero": ("qualifier.zeroshot.recall", pct),
    "qAddedZero": ("qualifier.zeroshot.added_per_q", lambda x: f'{x:.2f}'),
    "qPosPairs": ("qualifier.trained.n_pos_pairs", integer),
    "qNegPairs": ("qualifier.trained.n_neg_pairs", integer),
    "oodAucNatural": ("qualifier.ood.natural.auc", lambda x: f'{x:.2f}'),
    "oodAucProbes": ("qualifier.ood.probes.auc", lambda x: f'{x:.2f}'),
    "oodPosNatural": ("qualifier.ood.natural.n_pos", integer),
    "oodNegNatural": ("qualifier.ood.natural.n_neg", integer),
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


TABLES = settings.REPO_ROOT / "paper" / "tables"

PROMPT_LABELS = {
    "v1": "lines stating unresolved requirements; no evidence field",
    "v2": "``usually []''; evidence field added",
    "v3": "``list every one of them''",
    "v4": "the benchmark's definition (frozen)",
}


def tuning_rows(results: dict) -> str:
    lines = []
    for p, label in PROMPT_LABELS.items():
        cells = []
        for m in ("qwen7b", "llama8b", "qwen27b"):
            row = results.get("tuning", {}).get(p, {}).get(m)
            if row is None:
                cells += ["--", "--"]
                continue
            c, u = row["conditional"]["f1_cond"], row["unconditional"]["f1_cond"]
            cells += [f"{100 * c:.1f} / {100 * u:.1f}", f"{100 * row['unconditional']['any_condition']:.0f}\\%"]
        lines.append(f"{p} & {label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


#: Table 2, written whole: \input cannot sit between rows of a tabular.
TUNING_TABLE = "\n".join([
    r"% Generated by study/report.py. Do not edit by hand.",
    r"\begin{center}",
    r"\textbf{Table 2.} Generator wording on the tuning set with the gold chunks in context: F1 with conditions (\%) "
    r"on conditional (C) and unconditional (U) questions, and the share of unconditional answers that name any "
    r"condition. Qwen3.8-27B on a fixed 60-question sample of the tuning set.\\\vspace{5pt}",
    r"{\fontsize{10}{12}\selectfont",
    r"\begin{tabular}{L{1.15cm}L{4.05cm}P{1.45cm}P{1.0cm}P{1.45cm}P{1.0cm}P{1.45cm}P{1.0cm}}",
    r"\toprule",
    r" & & \multicolumn{2}{c}{\textbf{Qwen2.5-7B}} & \multicolumn{2}{c}{\textbf{Llama-3.1-8B}} "
    r"& \multicolumn{2}{c}{\textbf{Qwen3.8-27B}} \\",
    r"\textbf{Prompt} & \textbf{Instruction for conditions} & C / U & cond. on U & C / U & cond. on U "
    r"& C / U & cond. on U \\",
    r"\midrule",
    "%ROWS%" + r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{center}",
    "",
])


#: Display names, as in Table 1.
LABEL = {
    "closed": r"\texttt{closed-book}", "gold": r"\texttt{gold}",
    "gold_nocondel": r"\texttt{gold\,$-$\,cond.\ elements}", "gold_nocondchk": r"\texttt{gold\,$-$\,cond.-only chunks}",
    "goldpad8": r"\texttt{gold\,+\,distractors}", "top3": r"\texttt{top-3}", "top8": r"\texttt{top-8}",
    "rr8to3": r"\texttt{rerank-3}", "nli": r"\texttt{NLI filter}", "crs": r"\texttt{ConflictRAG-style}",
    "grade": r"\texttt{relevance grading}", "select": r"\texttt{listwise selection}", "adj": r"\texttt{adjacent}",
    "page": r"\texttt{page}", "qc0": r"\texttt{completion (untrained)}", "qc": r"\texttt{completion} (ours)",
    "full": r"\texttt{full context}", "oracle": r"\texttt{oracle discard}", "rr3": r"\texttt{rerank-3}",
}
E2E_GROUPS = (
    ("Oracle ladder", ("closed", "gold", "gold_nocondel", "gold_nocondchk", "goldpad8")),
    ("Retrieval", ("top3", "top8")),
    ("Evidence-reducing operators", ("rr8to3", "nli", "crs", "select")),
    ("Context completion", ("adj", "page", "qc0", "qc")),
)
DASH = "--"


def _p(x, places: int = 1) -> str:
    return DASH if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{places}f}"


def _signed(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "$-$")


def _pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def e2e_rows(results: dict) -> str:
    cells = results.get("end_to_end", {}).get("eval", {})
    lines = []
    for title, systems_ in E2E_GROUPS:
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}}\\")
        for s in systems_:
            vals = []
            for m in LOCAL_MODELS:
                row = cells.get(m, {}).get(s)
                c, u = (row or {}).get("conditional", {}), (row or {}).get("unconditional", {})
                vals += [_p(c.get("f1_cond")), _p(c.get("cond_noticed")), _p(u.get("f1_cond"))]
            lines.append(f"{LABEL[s]} & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]) + "\n"


E2E_TABLE = "\n".join([
    r"% Generated by study/report.py. Do not edit by hand.",
    r"\begin{center}",
    r"\textbf{Table 3.} End-to-end results on the evaluation set: F1 with conditions (\%) on conditional (C) and "
    r"unconditional (U) questions, and the share of gold condition elements in the context that the generator "
    r"cites anywhere (noticed, \%).\\\vspace{5pt}",
    r"{\fontsize{10}{12}\selectfont",
    r"\begin{tabular}{L{4.6cm}P{1.35cm}P{1.35cm}P{1.35cm}P{1.35cm}P{1.35cm}P{1.35cm}}",
    r"\toprule",
    r" & \multicolumn{3}{c}{\textbf{Qwen2.5-7B}} & \multicolumn{3}{c}{\textbf{Llama-3.1-8B}} \\",
    r"\textbf{System} & F1c (C) & noticed & F1c (U) & F1c (C) & noticed & F1c (U) \\",
    r"\midrule",
    "%ROWS%" + r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{center}",
    "",
])

CONFIRM_LABEL = {
    "C1": ("gold vs.\\ cond.\\ elements deleted", "C"), "C2": ("gold vs.\\ cond.-only chunks removed", "C$^\\dagger$"),
    "C3": ("gold + distractors vs.\\ top-8", "C"), "C4": ("rerank-3 vs.\\ top-8", "C"),
    "C5": ("NLI filter vs.\\ top-8", "C"), "C6": ("ConflictRAG-style vs.\\ top-8", "C"),
    "C7": ("completion vs.\\ top-8", "C"), "C8": ("completion vs.\\ top-8 (non-inferiority)", "U"),
}


def confirm_rows(results: dict) -> str:
    conf = results.get("confirmatory", {})
    lines = []
    for cid, (label, pop) in CONFIRM_LABEL.items():
        vals = []
        for m in LOCAL_MODELS:
            r = conf.get(m, {})
            x = r.get("C8") if cid == "C8" else r.get("family", {}).get(cid)
            if x is None or not x.get("complete"):
                vals += [DASH, DASH]
                continue
            vals.append(f"{_signed(x['diff'])} [{_signed(x['low'])}, {_signed(x['high'])}]")
            if cid == "C8":
                vals.append("yes" if x["non_inferior"] else "no")
            else:
                vals.append(_pval(x["p_holm"]) if "p_holm" in x else DASH)
        lines.append(f"{cid} & {label} & {pop} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines) + "\n"


CONFIRM_TABLE = "\n".join([
    r"% Generated by study/report.py. Do not edit by hand.",
    r"\begin{center}",
    r"\textbf{Table 4.} The pre-registered confirmatory family: paired difference in F1 with conditions (points), "
    r"95\% BCa interval over pages, and Holm-adjusted page-level sign-flip $p$. C: conditional questions; "
    r"C$^\dagger$: those with a condition-only gold chunk; U: unconditional. C8 asks whether the lower bound "
    r"exceeds $-2$ points.\\\vspace{5pt}",
    r"{\fontsize{10}{12}\selectfont",
    r"\begin{tabular}{L{0.6cm}L{3.95cm}P{0.85cm}P{2.65cm}P{1.05cm}P{2.65cm}P{1.05cm}}",
    r"\toprule",
    r" & & & \multicolumn{2}{c}{\textbf{Qwen2.5-7B}} & \multicolumn{2}{c}{\textbf{Llama-3.1-8B}} \\",
    r"\textbf{ID} & \textbf{Comparison} & \textbf{Pop.} & diff.\ [95\% CI] & $p_{\mathrm{Holm}}$ "
    r"& diff.\ [95\% CI] & $p_{\mathrm{Holm}}$ \\",
    r"\midrule",
    "%ROWS%" + r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{center}",
    "",
])

HOSTED_SYSTEMS = ("gold", "gold_nocondchk", "top8", "qc")


def hosted_rows(results: dict) -> str:
    cells = results.get("end_to_end", {}).get("hosted", {})
    lines = []
    for s in HOSTED_SYSTEMS:
        vals = []
        for m in HOSTED_MODELS:
            row = cells.get(m, {}).get(s, {})
            c, u = row.get("conditional", {}), row.get("unconditional", {})
            full_c = c.get("n") == 150
            vals += [_p(c.get("f1_cond")) if full_c else DASH, _p(c.get("cond_noticed")) if full_c else DASH,
                     _p(u.get("f1_cond")) if u.get("n") == 100 else DASH]
        lines.append(f"{LABEL[s]} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines) + "\n"


HOSTED_TABLE = "\n".join([
    r"% Generated by study/report.py. Do not edit by hand.",
    r"\begin{center}",
    r"\textbf{Table 5.} Larger hosted generators on the stratified subsample (150 conditional, 100 unconditional "
    r"questions): F1 with conditions (\%) on conditional (C) and unconditional (U) questions, and conditions "
    r"noticed (N, \%) as in Table 3.\\\vspace{5pt}",
    r"{\fontsize{10}{12}\selectfont",
    r"\begin{tabular}{L{3.6cm}P{0.85cm}P{0.85cm}P{0.85cm}P{0.85cm}P{0.85cm}P{0.85cm}P{0.85cm}P{0.85cm}P{0.85cm}}",
    r"\toprule",
    r" & \multicolumn{3}{c}{\textbf{gpt-oss-20b}} & \multicolumn{3}{c}{\textbf{Gemma-4-31B}} "
    r"& \multicolumn{3}{c}{\textbf{Nemotron-3-Ultra}} \\",
    r"\textbf{System} & C & N & U & C & N & U & C & N & U \\",
    r"\midrule",
    "%ROWS%" + r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{center}",
    "",
])

RAM_SYSTEMS = ("full", "oracle", "rr3", "nli", "crs", "grade", "select")


def ram_rows(results: dict) -> str:
    ram = results.get("ramdocs", {})
    lines = []
    for s in RAM_SYSTEMS:
        vals = []
        for m in LOCAL_MODELS:
            d = ram.get(m, {}).get(s, {})
            full = d.get("all", {}).get("n") == 500
            vals += [_p(d["all"]["correct"]) if full else DASH,
                     _p(d["misinfo"]["correct"]) if full else DASH,
                     _p(d["misinfo"]["any_wrong"]) if full else DASH]
        lines.append(f"{LABEL[s]} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines) + "\n"


RAM_TABLE = "\n".join([
    r"% Generated by study/report.py. Do not edit by hand.",
    r"\begin{center}",
    r"\textbf{Table 6.} The same operators on RAMDocs, where discarding is correct: strict accuracy (\%) on all "
    r"questions and on those with misinformation (MI), and the share of MI answers that state a wrong answer.\\\vspace{5pt}",
    r"{\fontsize{10}{12}\selectfont",
    r"\begin{tabular}{L{4.0cm}P{1.45cm}P{1.45cm}P{1.45cm}P{1.45cm}P{1.45cm}P{1.45cm}}",
    r"\toprule",
    r" & \multicolumn{3}{c}{\textbf{Qwen2.5-7B}} & \multicolumn{3}{c}{\textbf{Llama-3.1-8B}} \\",
    r"\textbf{System} & all & MI & wrong (MI) & all & MI & wrong (MI) \\",
    r"\midrule",
    "%ROWS%" + r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{center}",
    "",
])


def write_tables(results: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, template, rows in (("tuning", TUNING_TABLE, tuning_rows), ("endtoend", E2E_TABLE, e2e_rows),
                                 ("confirm", CONFIRM_TABLE, confirm_rows), ("hosted", HOSTED_TABLE, hosted_rows),
                                 ("ramdocs", RAM_TABLE, ram_rows)):
        (TABLES / f"{name}.tex").write_text(template.replace("%ROWS%", rows(results)), encoding="utf-8")


def main() -> int:
    results = build()
    write_tables(results)
    missing = write_numbers(results)
    print(f"wrote {RESULTS.relative_to(settings.REPO_ROOT)} and {NUMBERS.relative_to(settings.REPO_ROOT)}"
          f" ({len(MACROS)} macros, {missing} not yet available)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
