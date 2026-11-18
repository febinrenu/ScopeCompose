"""Results of the analyses added in the sixth review round (deviation 5; all exploratory).

Each function reads saved cells and artifacts only, and is a section of ``results/results.json``
built by ``study.report``:

``prompt_robustness``  six generator prompts on the hosted subsample (``study.prompts``)
``baselines``          context-preserving retrieval designs (``study.baselines``)
``clean_threshold``    completion under the clean threshold protocol (``study.qualifier``)
``semantic``           condition recall under meaning-based matching (``study.semantic``)
"""

from __future__ import annotations

import bisect
import json

from study import cqa, systems

LOCAL = ("qwen7b", "llama8b")
PROMPT_MODELS = ("qwen7b", "llama8b", "nemotron")
PROMPTS = ("v4", "extract", "pairing", "coc", "applicability", "evidence")


def _stats(scores: dict, keys) -> dict:
    from study.analysis import summarise

    s = summarise([scores[k] for k in keys])
    return {x: s[x] for x in ("f1_cond", "f1", "cond_precision", "cond_recall", "any_condition", "cond_noticed",
                              "cond_in_context", "parse_ok", "no_answer", "tokens") if x in s}


def _compare(a: dict, b: dict, pop: frozenset[str]) -> dict | None:
    from study import confirm

    keys = pop & set(a) & set(b)
    if len(keys) != len(pop):
        return None                          # a cell still running reports nothing
    return confirm.compare({"a": a, "b": b}, "a", "b", pop)


def prompt_robustness() -> dict:
    from study import confirm, prompts
    from study.analysis import score_cell
    from study.run import RUNS, cell_path

    pops = {k: confirm.population("hosted", k) for k in ("conditional", "unconditional", "condition_only")}
    answerable = pops["conditional"] | pops["unconditional"]
    out: dict = {}
    for m in PROMPT_MODELS:
        base_set = "eval" if m in LOCAL else "hosted"
        cells = {("v4", s): score_cell(cell_path(s, m, base_set, systems.Env())) for s in prompts.SYSTEMS}
        for v in prompts.VARIANTS:
            for s in prompts.SYSTEMS:
                cells[(v, s)] = score_cell(RUNS / "hosted" / f"{prompts.cell_name(s, v, m)}.jsonl")
        rows: dict = {}
        for p in PROMPTS:
            gold, nochk = cells[(p, "gold")], cells[(p, "gold_nocondchk")]
            if not answerable <= set(gold):
                continue
            row = {"conditional": _stats(gold, pops["conditional"]), "unconditional": _stats(gold, pops["unconditional"]),
                   "all": _stats(gold, answerable)}
            if (c2 := _compare(gold, nochk, pops["condition_only"])) is not None:
                row["c2"] = c2
            if p != "v4":
                for pop in ("conditional", "unconditional"):
                    if (r := _compare(gold, cells[("v4", "gold")], pops[pop])) is not None:
                        row[f"vs_v4_{pop}"] = r
            rows[p] = row
        if rows:
            out[m] = rows
    if not out:
        raise FileNotFoundError("no prompt-robustness cells yet")
    recalls = [r["conditional"]["cond_recall"] for rows in out.values() for r in rows.values()]
    out["max_recall"] = max(recalls)
    out["min_recall"] = min(recalls)
    variants = [r for rows in out.values() if isinstance(rows, dict) for p, r in rows.items() if p != "v4"]
    out["max_any_uncond"] = max(r["unconditional"]["any_condition"] for r in variants)
    out["min_any_uncond"] = min(r["unconditional"]["any_condition"] for r in variants)
    c2 = [r["c2"]["diff"] for rows in out.values() if isinstance(rows, dict) for r in rows.values() if "c2" in r]
    out["c2_positive"] = sum(d > 0 for d in c2)
    out["c2_n"] = len(c2)
    gains = [r["vs_v4_conditional"] for r in variants if "vs_v4_conditional" in r]
    out["n_variant_runs"] = len(gains)
    out["n_sig_gain_cond"] = sum(g["low"] > 0 for g in gains)
    out["n_sig_loss_cond"] = sum(g["high"] < 0 for g in gains)
    out["max_gain_cond"] = max(g["diff"] for g in gains)
    out["n_sig_loss_uncond"] = sum(r["vs_v4_unconditional"]["high"] < 0 for r in variants if "vs_v4_unconditional" in r)
    local = [r["conditional"]["cond_recall"] for m in LOCAL for r in out.get(m, {}).values()]
    out["max_recall_local"] = max(local)
    return out


def baselines() -> dict:
    from study import baselines as bl
    from study import confirm
    from study.analysis import score_cell
    from study.run import cell_path

    out: dict = {"recall": bl.recall(), "e2e": {}}
    # end to end on the hosted subsample's answerable questions (study.baselines.E2E_QUESTIONS)
    pops = {k: confirm.population("hosted", k) for k in ("conditional", "unconditional")}
    for m in LOCAL:
        top = score_cell(cell_path("top8", m, "eval", systems.Env()))
        for name, (s, env) in bl.DESIGNS.items():
            a = score_cell(cell_path(s, m, "eval", env))
            row = {}
            for pop_name, pop in pops.items():
                if (r := _compare(a, top, pop)) is not None:
                    r["a_stats"], r["b_stats"] = _stats(a, pop), _stats(top, pop)
                    row[pop_name] = r
            if row:
                out["e2e"].setdefault(m, {})[name] = row
    for pop in ("conditional", "unconditional"):
        rs = [x[pop] for row in out["e2e"].values() for x in row.values() if pop in x]
        if rs:
            out[f"{pop}_min"] = min(r["diff"] for r in rs)
            out[f"{pop}_max"] = max(r["diff"] for r in rs)
            out[f"{pop}_n_sig"] = sum(r["low"] > 0 or r["high"] < 0 for r in rs)
            out[f"{pop}_n"] = len(rs)
    return out


def clean_threshold() -> dict:
    from study import qualifier
    from study.analysis import score_cell
    from study.run import cell_path

    data, fold_of = qualifier._clean()
    env, idx = systems.Env(), cqa.by_id()
    tp = fp = fn = added = 0
    pos, neg = [], []
    ev = [q.id for q in cqa.eval_questions()]
    for qid in ev:
        s = data["scores"].get(qid, {})
        t = data["threshold"][fold_of[qid]]
        pairs = qualifier.candidate_pairs(idx[qid], env)
        gold = {p.candidate for p in pairs if p.label}
        picked = {cid for _, cid in sorted(((v, k) for k, v in s.items() if v >= t), reverse=True)
                  [: qualifier.BUDGET - qualifier.ANCHOR_K]}
        tp, fp, fn, added = tp + len(picked & gold), fp + len(picked - gold), fn + len(gold - picked), added + len(picked)
        for p in pairs:
            if p.candidate in s:
                (pos if p.label else neg).append(s[p.candidate])
    neg.sort()
    wins = sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) for x in pos)
    th = list(data["threshold"].values())
    out: dict = {"thresholds": data["threshold"], "threshold_min": min(th), "threshold_max": max(th),
                 "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                 "added_per_q": added / len(ev), "auc": wins / (len(pos) * len(neg)), "e2e": {}}
    from study import confirm

    pops = {k: confirm.population("eval", k) for k in ("conditional", "unconditional")}
    for m in LOCAL:
        a = score_cell(cell_path("qcclean", m, "eval", env))
        top = score_cell(cell_path("top8", m, "eval", env))
        qc = score_cell(cell_path("qc", m, "eval", env))
        row = {}
        for pop_name, pop in pops.items():
            if (r := _compare(a, top, pop)) is not None:
                row[f"vs_top8_{pop_name}"] = r
            if (r := _compare(a, qc, pop)) is not None:
                row[f"vs_qc_{pop_name}"] = r
        if row:
            out["e2e"][m] = row
    # The clean protocol is the primary completion analysis: Holm over the confirmatory family with C7 under
    # the clean protocol in place of the pre-registered one, and C8's non-inferiority check under it.
    from metrics.stats import holm_adjust

    from study.report import RESULTS

    conf = json.loads(RESULTS.read_text(encoding="utf-8")).get("confirmatory", {}) if RESULTS.exists() else {}
    for m, row in out["e2e"].items():
        fam = conf.get(m, {}).get("family", {})
        c7 = row.get("vs_top8_conditional")
        if c7 is not None and all(f"C{i}" in fam for i in range(1, 7)):
            ps = {f"C{i}": fam[f"C{i}"]["p"] for i in range(1, 7)} | {"C7": c7["p"]}
            c7["p_holm"], c7["reject"] = holm_adjust(ps)["C7"]
        c8 = row.get("vs_top8_unconditional")
        if c8 is not None:
            c8["margin"] = confirm.NI_MARGIN
            c8["non_inferior"] = c8["low"] > -confirm.NI_MARGIN
    return out


def semantic() -> dict:
    from study import semantic as sem

    return sem.audit()


SECTIONS = {"prompt_robustness": prompt_robustness, "baselines": baselines, "clean_threshold": clean_threshold,
            "semantic": semantic}
