"""The confirmatory family of the analysis plan (``study/ANALYSIS_PLAN.md``), computed from cell files.

    python -m study.confirm                    # every model whose cells exist; prints a summary

C1-C7 are paired differences in F1 with conditions, per local model on the
evaluation set: mean difference, a page-clustered BCa interval and a page-level
sign-flip p-value, Holm-adjusted over the seven within each model. C8 is a
non-inferiority bound and is not in the Holm family. The plan states the bound
as "the lower 95% bound": we use the lower end of the two-sided 95% interval,
which is stricter than a one-sided 95% bound.

The hosted models answer a subsample with fewer systems, so only the
comparisons whose two cells exist are computed there; they are replication of
direction and carry no Holm correction (plan, "Models").

A comparison is *complete* only when both cells hold every question of its
population; Holm is applied only to a complete family, so a partial run can
never be reported as a confirmatory result.
"""

from __future__ import annotations

import json
from functools import lru_cache

from study import cqa, stats
from study.analysis import QScore, load
from study.run import question_set

#: (id, system a, system b, population) -- the plan's table, in its order.
FAMILY: tuple[tuple[str, str, str, str], ...] = (
    ("C1", "gold", "gold_nocondel", "conditional"),
    ("C2", "gold", "gold_nocondchk", "condition_only"),
    ("C3", "goldpad8", "top8", "conditional"),
    ("C4", "rr8to3", "top8", "conditional"),
    ("C5", "nli", "top8", "conditional"),
    ("C6", "crs", "top8", "conditional"),
    ("C7", "qc", "top8", "conditional"),
)
C8 = ("C8", "qc", "top8", "unconditional")
NI_MARGIN = 0.02                      # 2.0 F1c points, as a fraction (scores are in [0, 1])

#: Exploratory paired contrasts reported beside the family (no correction, labelled as such).
EXPLORATORY: tuple[tuple[str, str, str, str], ...] = (
    ("E1", "gold", "top8", "conditional"),            # the whole retrieval-side loss at equal prompt
    ("E2", "qc0", "top8", "conditional"),             # untrained scorer
    ("E3", "adj", "top8", "conditional"),             # budget-matched control for completion
    ("E4", "qc", "adj", "conditional"),
    ("E5", "qc", "top8", "opposite"),                 # the rule-and-exception subset
    ("E6", "gold", "top8", "opposite"),
    ("E7", "select", "top8", "conditional"),
    ("E8", "page", "top8", "conditional"),
)

LOCAL = ("qwen7b", "llama8b")
HOSTED = ("oss20b", "gemma31b", "nemotron")


@lru_cache(maxsize=None)
def population(qset: str, name: str) -> frozenset[str]:
    qs = question_set(qset)
    if name == "condition_only":
        return frozenset(q.id for q in qs if q.kind == "conditional" and cqa.locate_gold(q).condition_only_chunks)
    if name == "opposite":
        return frozenset(q.id for q in qs if q.kind == "conditional" and q.opposite_outcomes)
    return frozenset(q.id for q in qs if q.kind == name)


def _cells(model: str, qset: str, systems: set[str]) -> dict[str, dict[str, QScore]]:
    out = {}
    for s in systems:
        try:
            scores = load(s, model, qset)
        except FileNotFoundError:
            continue
        if scores:
            out[s] = scores
    return out


def compare(cells: dict[str, dict[str, QScore]], a: str, b: str, pop: frozenset[str], *, seed: int = 0) -> dict | None:
    if a not in cells or b not in cells:
        return None
    fa = {q: s.f1_cond for q, s in cells[a].items() if q in pop}
    fb = {q: s.f1_cond for q, s in cells[b].items() if q in pop}
    if not set(fa) & set(fb):
        return None
    doc = {q: s.doc for c in (cells[a], cells[b]) for q, s in c.items()}
    r = stats.paired(fa, fb, doc, seed=seed)
    r.update({"a_system": a, "b_system": b, "population": len(pop), "complete": r["n"] == len(pop)})
    return r


def model_results(model: str, qset: str) -> dict:
    wanted = {x for row in FAMILY + (C8,) + EXPLORATORY for x in row[1:3]}
    cells = _cells(model, qset, wanted)
    fam = {}
    for cid, a, b, pop in FAMILY:
        r = compare(cells, a, b, population(qset, pop))
        if r is not None:
            fam[cid] = r
    complete = len(fam) == len(FAMILY) and all(r["complete"] for r in fam.values())
    if qset == "eval" and complete:
        for cid, (p_adj, reject) in stats.holm_adjust({k: r["p"] for k, r in fam.items()}).items():
            fam[cid]["p_holm"], fam[cid]["reject"] = p_adj, reject
    out = {"family": fam, "family_complete": complete, "confirmatory": qset == "eval" and complete}
    cid, a, b, pop = C8
    if (r := compare(cells, a, b, population(qset, pop))) is not None:
        r["margin"] = NI_MARGIN
        r["non_inferior"] = r["low"] > -NI_MARGIN
        out["C8"] = r
    out["exploratory"] = {cid: r for cid, a, b, pop in EXPLORATORY
                          if (r := compare(cells, a, b, population(qset, pop))) is not None}
    return out


def confirmatory() -> dict:
    """The ``confirmatory`` section of results.json: local models on eval, hosted on the hosted subsample."""
    out = {}
    for m in LOCAL:
        r = model_results(m, "eval")
        if r["family"] or "C8" in r:
            out[m] = r
    for m in HOSTED:
        r = model_results(m, "hosted")
        if r["family"] or "C8" in r or r["exploratory"]:
            out[m] = r
    if not out:
        raise FileNotFoundError("no evaluation or hosted cells yet")
    return out


def _fmt(cid: str, r: dict) -> str:
    holm = f" holm {r['p_holm']:.4f}{' *' if r['reject'] else ''}" if "p_holm" in r else ""
    ni = f" NI={'yes' if r['non_inferior'] else 'no'}" if "non_inferior" in r else ""
    part = "" if r["complete"] else f" (partial {r['n']}/{r['population']})"
    return (f"  {cid} {r['a_system']:>13s} - {r['b_system']:<14s} n={r['n']:4d} "
            f"{100 * r['a']:5.1f} vs {100 * r['b']:5.1f}  diff {100 * r['diff']:+5.1f} "
            f"[{100 * r['low']:+5.1f}, {100 * r['high']:+5.1f}] p {r['p']:.4f}{holm}{ni}{part}")


def main() -> int:
    try:
        res = confirmatory()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    for m, r in res.items():
        print(f"{m}  ({'confirmatory' if r['confirmatory'] else 'not confirmatory: family incomplete or hosted'})")
        for cid, x in {**r["family"], **({"C8": r["C8"]} if "C8" in r else {}), **r["exploratory"]}.items():
            print(_fmt(cid, x))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
