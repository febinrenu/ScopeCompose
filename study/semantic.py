"""Semantic condition matching (revision, round 6; exploratory).

The official metric credits a condition only when the exact gold element is cited. The token-overlap
audit in ``study.report.metric_audit`` bounds how much that costs; this module adds a meaning-based
check. For every conditional question answered with the gold context, each gold condition the
generator did not cite as a condition is compared with each element it did cite as a condition but
is not gold, by the study's NLI cross-encoder in both directions. A missed gold condition counts as
stated in other words when a stated element and it entail each other (both probabilities >= 0.5),
and, as a lenient bound, when either entails the other. Matching is one-to-one.

    python -m study.semantic          # scores the pairs once (GPU or CPU) and saves them

The scores are saved to ``study/artifacts/semantic/nli_pairs.json`` so the report needs no model.
"""

from __future__ import annotations

import json

from study import cqa, systems
from study.retrieve import ARTIFACTS

PAIRS = ARTIFACTS / "semantic" / "nli_pairs.json"
CELLS = (("eval", "qwen7b"), ("eval", "llama8b"), ("hosted", "oss20b"), ("hosted", "gemma31b"), ("hosted", "nemotron"))
THRESHOLD = 0.5


def _text(element: str) -> str:
    return cqa.split_element(element)[1]


def missed_and_stated(rec: dict, q: cqa.Question) -> tuple[set[str], set[str]]:
    gold = {c for _, cs in q.reference() for c in cs}
    pred = {c for _, cs in rec["answers"] for c in cs}
    return gold - pred, pred - gold


def candidate_pairs() -> set[tuple[str, str]]:
    from study.run import cell_path, read_cell

    idx = cqa.by_id()
    out: set[tuple[str, str]] = set()
    for qset, m in CELLS:
        for qid, rec in read_cell(cell_path("gold", m, qset, systems.Env())).items():
            q = idx[qid]
            if q.kind != "conditional":
                continue
            missed, stated = missed_and_stated(rec, q)
            out |= {(p, g) for p in stated for g in missed}
    return out


def score() -> dict:
    from study.operators import _nli

    have = json.loads(PAIRS.read_text(encoding="utf-8")) if PAIRS.exists() else {}
    todo = sorted(p for p in candidate_pairs() if f"{p[0]}\t{p[1]}" not in have)
    nli = _nli()
    for start in range(0, len(todo), 256):
        batch = todo[start:start + 256]
        fwd = nli.score_batch([(_text(p), _text(g)) for p, g in batch])
        bwd = nli.score_batch([(_text(g), _text(p)) for p, g in batch])
        for (p, g), a, b in zip(batch, fwd, bwd):
            have[f"{p}\t{g}"] = [round(a.entailment, 4), round(b.entailment, 4)]
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(json.dumps(have, sort_keys=True), encoding="utf-8")
    return {"pairs": len(have), "new": len(todo)}


def audit() -> dict:
    """Condition recall with the gold context under exact, mutual-entailment and either-way matching."""
    from study.run import cell_path, read_cell

    if not PAIRS.exists():
        raise FileNotFoundError(f"{PAIRS} is missing; run `python -m study.semantic`")
    scores = json.loads(PAIRS.read_text(encoding="utf-8"))
    idx = cqa.by_id()
    out: dict = {}
    for qset, m in CELLS:
        n_gold = exact = 0
        added = {"mutual": 0, "either": 0}
        for qid, rec in read_cell(cell_path("gold", m, qset, systems.Env())).items():
            q = idx[qid]
            if q.kind != "conditional":
                continue
            gold = {c for _, cs in q.reference() for c in cs}
            missed, stated = missed_and_stated(rec, q)
            n_gold, exact = n_gold + len(gold), exact + len(gold - missed)
            for crit in added:
                cands = []
                for p in stated:
                    for g in missed:
                        a, b = scores[f"{p}\t{g}"]
                        ok = min(a, b) >= THRESHOLD if crit == "mutual" else max(a, b) >= THRESHOLD
                        if ok:
                            cands.append((a + b, g, p))
                used_g, used_p = set(), set()
                for _, g, p in sorted(cands, reverse=True):
                    if g not in used_g and p not in used_p:
                        used_g.add(g), used_p.add(p)
                        added[crit] += 1
        out[m] = {"exact": exact / n_gold, **{f"{c}_added": v / n_gold for c, v in added.items()},
                  **{f"{c}_recall": (exact + v) / n_gold for c, v in added.items()}}
    for c in ("mutual", "either"):
        out[f"{c}_added_max"] = max(v[f"{c}_added"] for v in out.values() if isinstance(v, dict))
        out[f"{c}_recall_max"] = max(v[f"{c}_recall"] for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    print(json.dumps(score()))
    print(json.dumps(audit(), indent=1))
