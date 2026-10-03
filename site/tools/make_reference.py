"""Record what the Python engines decide, so the JavaScript ports can be checked.

The site reimplements parts of the pipeline in JavaScript so the detector can
run in a browser with no server. A reimplementation is a liability: it can
drift from the Python it mirrors, and a page that computes a *different*
answer from the pipeline is worse than a screenshot, because it looks
authoritative while being wrong.

This records Python's answer for every case the browser will face.
``verify_engines.js`` replays them and fails on any disagreement.

    python site/tools/make_reference.py
    node site/tools/verify_engines.js

Four families:

``compare``
    Every ordered branch pair in the corpus, both directions. ``compare()`` is
    deliberately argument-relative -- SUBSET one way, SUPERSET the other -- and
    a port that silently symmetrised it would invert branch roles, making the
    general rule the exception.

``combine``
    The **full truth table** of ``ScopeAnalyser._combine``: every SetRelation
    crossed with both outcome verdicts. This exists because the corpus cannot
    exercise it. No corpus pair produces ``redundant``, so replaying only
    corpus cases leaves that arm of the four-way relation untested while
    reporting a confident pass. A truth table costs nothing and closes it.

``outcomes_agree``
    Python's **actual** verdict per corpus pair, from the NLI-backed
    ``ScopeAnalyser.outcomes_agree``. The browser cannot run that model, so it
    uses a lexical approximation -- and the two disagree on a measurable
    number of pairs. Recording Python's answer turns that from a hidden
    divergence into a reported one, and lets the browser use the real value
    for corpus instances instead of guessing.

``route``
    The routing table over every (type, relation) combination.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contract.gold import GoldInstance                              # noqa: E402
from contract.models import ConflictPair, ConflictType, ScopeRelation  # noqa: E402
from contract.routing import route                                  # noqa: E402
from scope.attributes import SetRelation, compare                   # noqa: E402
from scope.relation import ScopeAnalyser                            # noqa: E402

SOURCE = ROOT / "benchmark" / "data" / "corpus.jsonl"
OUT = Path(__file__).resolve().parent / "compare_reference.json"


def _load() -> list[GoldInstance]:
    if not SOURCE.exists():
        raise SystemExit(f"corpus not found: {SOURCE}")
    return [
        GoldInstance.model_validate(json.loads(line))
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _compare_cases(instances: list[GoldInstance]) -> list[dict]:
    cases = []
    for inst in instances:
        for a in inst.gold_branches or []:
            for b in inst.gold_branches or []:
                if a.branch_id == b.branch_id:
                    continue
                cases.append({
                    "inst": inst.instance_id, "a": a.branch_id, "b": b.branch_id,
                    "rel": compare(a.applicability, b.applicability).value,
                })
    return cases


def _combine_cases() -> list[dict]:
    """The whole truth table, including the arms the corpus never reaches."""
    cases = []
    for rel in SetRelation:
        for agree in (True, False):
            relation, _ = ScopeAnalyser._combine(rel, agree)
            cases.append({"set": rel.value, "agree": agree, "rel": relation.value})
    return cases


def _outcomes_agree_cases(instances: list[GoldInstance]) -> list[dict]:
    """Python's real verdict, which the browser cannot compute.

    Also records what a lexical equality test would have said, so the
    divergence is a number on the page rather than a surprise.
    """
    analyser = ScopeAnalyser(heuristic_nli=True)
    cases = []
    for inst in instances:
        branches = inst.gold_branches or []
        if len(branches) != 2:
            continue
        a, b = branches
        agree, _ = analyser.outcomes_agree(a.outcome, b.outcome)
        lexical = a.outcome.strip().lower() == b.outcome.strip().lower()
        cases.append({
            "inst": inst.instance_id, "agree": bool(agree), "lexical": lexical,
        })
    return cases


def _detector_cases(instances: list[GoldInstance]) -> list[dict]:
    """Stage 1 exactly as it runs offline, per instance.

    This is the centrepiece of the site's bench, so a drifted port here would
    make the headline wrong. Records the score, the verdict, and all 17
    features.
    """
    from detection.stage1 import Stage1Filter

    s1 = Stage1Filter(heuristic_nli=True, auto_load_head=False)
    cases = []
    for inst in instances:
        for cand in s1.score_pairs(inst.passages):
            p = cand.confidence if cand.is_conflict else 1.0 - cand.confidence
            cases.append({
                "inst": inst.instance_id, "i": cand.doc_i, "j": cand.doc_j,
                "score": round(p, 10),
                "conflict": bool(cand.is_conflict),
                "escalate": bool(cand.needs_escalation),
                "nli": [round(cand.nli.entailment, 10), round(cand.nli.neutral, 10),
                        round(cand.nli.contradiction, 10)],
                "f": {k: round(float(v), 10)
                      for k, v in zip(cand.features.names(), cand.features.as_vector())},
            })
    return cases


def _textual_cases(instances: list[GoldInstance]) -> list[dict]:
    """preservation.py's textual judging path, over every resolver the bench
    shows. Verifies similarity/containment and the two inline thresholds."""
    from metrics.preservation import judge_branches

    cases = []
    for inst in instances:
        if not inst.gold_branches:
            continue
        answers = {
            "selection": inst.selection_answer or "",
            "concat": " ".join(p.text for p in inst.passages),
            "gold": inst.gold_scoped_answer or "",
        }
        for name, text in answers.items():
            cases.append({
                "inst": inst.instance_id, "sys": name,
                "j": [j.value for j in judge_branches(text, inst.gold_branches)],
            })
    return cases


def _route_cases() -> list[dict]:
    """Every (type, relation) the routing table can be asked about."""
    cases = []
    for ctype in ConflictType:
        relations: list[ScopeRelation | None]
        relations = list(ScopeRelation) if ctype is ConflictType.CONDITIONAL else [None]
        if ctype is ConflictType.FACTUAL:
            relations = [None, ScopeRelation.REFINEMENT]
        for rel in relations:
            is_conflict = ctype is not ConflictType.NO_CONFLICT
            try:
                pair = ConflictPair(
                    doc_i="p0", doc_j="p1", is_conflict=is_conflict,
                    type=ctype, scope_relation=rel, confidence=0.9,
                )
            except Exception:
                continue  # the contract forbids this combination; nothing to check
            decision = route(pair)
            cases.append({
                "type": ctype.value, "rel": rel.value if rel else None,
                "action": decision.action.value,
            })
    return cases


def main() -> int:
    instances = _load()
    reference = {
        "compare": _compare_cases(instances),
        "combine": _combine_cases(),
        "outcomes_agree": _outcomes_agree_cases(instances),
        "route": _route_cases(),
        "detector": _detector_cases(instances),
        "textual": _textual_cases(instances),
    }
    OUT.write_text(json.dumps(reference, separators=(",", ":"), sort_keys=True),
                   encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  compare         {len(reference['compare']):>4} ordered branch pairs "
          f"over {sum(1 for i in instances if i.gold_branches)} annotated instances")
    for rel, n in sorted(Counter(c["rel"] for c in reference["compare"]).items()):
        print(f"      {rel:<12} {n}")
    print(f"  combine         {len(reference['combine']):>4} truth-table rows "
          f"(every SetRelation x both outcome verdicts)")
    for rel, n in sorted(Counter(c["rel"] for c in reference["combine"]).items()):
        print(f"      {rel:<12} {n}")

    oa = reference["outcomes_agree"]
    diverged = [c for c in oa if c["agree"] != c["lexical"]]
    print(f"  outcomes_agree  {len(oa):>4} two-branch instances")
    print(f"      python says agree on {sum(c['agree'] for c in oa)}, "
          f"lexical equality says {sum(c['lexical'] for c in oa)}")
    if diverged:
        print(f"      *** {len(diverged)} divergences -- the browser's lexical test is NOT")
        print(f"          the pipeline's NLI test, and the site must not imply it is.")
    print(f"  route           {len(reference['route']):>4} (type, relation) combinations")

    det = reference["detector"]
    fired = sum(c["conflict"] for c in det)
    esc = sum(c["escalate"] for c in det)
    top = max((c["score"] for c in det), default=0.0)
    print(f"  detector        {len(det):>4} pairs: {fired} flagged conflict, "
          f"{esc} escalated, max score {top:.4f}")
    print(f"  textual         {len(reference['textual']):>4} (instance, resolver) judgements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
