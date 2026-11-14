"""Out-of-distribution check of the qualifier scorer on this project's own corpus.

    python -m study.ood

The corpus (benchmark/data/corpus.jsonl) holds passage pairs annotated
independently by two people for conflict type and scope relation, from gov.uk
immigration pages and UK bank terms -- a different selection of pages, a
different annotation scheme, and for bank terms a different domain. Under the v7
labels a *conditional conflict* is a refinement: one passage states a rule, the
other an exception or condition on it.

The question is only whether the scorer, trained on ConditionalQA's condition
annotation, ranks refinement pairs above pairs that are not in conflict. Each
pair is scored in both directions (either passage as the anchor) and the higher
score kept, since the corpus does not record which passage is the rule. The
scorer is the one trained on all five evaluation folds (the "tuning" fold
model), which never saw these passages. Exploratory: the natural stratum has
18 refinement pairs.
"""

from __future__ import annotations

import json

import settings
from benchmark.relabel_v7 import V7Label, relabel
from benchmark.strata import Stratum, stratum_of
from contract.gold import GoldInstance
from study.qualifier import MAX_LEN, MODELS


def load_corpus() -> list[GoldInstance]:
    path = settings.REPO_ROOT / "benchmark" / "data" / "corpus.jsonl"
    return [GoldInstance.model_validate(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_pairs(insts: list[GoldInstance]) -> dict[str, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = MODELS / "tuning"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(device).eval()
    out = {}
    for inst in insts:
        a, b = inst.passages[0].text, inst.passages[1].text
        firsts = [f"{inst.query}\n\n{a}", f"{inst.query}\n\n{b}"]
        seconds = [b, a]
        enc = tok(firsts, seconds, truncation="only_first", max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1]
        out[inst.instance_id] = float(p.max())
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate() -> dict:
    insts = [i for i in load_corpus() if len(i.passages) >= 2]
    scores = score_pairs(insts)
    groups: dict[str, dict[str, list[float]]] = {"natural": {"pos": [], "neg": []}, "probes": {"pos": [], "neg": []}}
    for inst in insts:
        stratum = stratum_of(inst)
        label = relabel(inst).label
        group = "probes" if stratum is Stratum.S5 else "natural"
        if label is V7Label.CONDITIONAL_CONFLICT:
            groups[group]["pos"].append(scores[inst.instance_id])
        elif label in (V7Label.NO_CONFLICT, V7Label.TEMPORAL, V7Label.FACTUAL, V7Label.OPINION):
            groups[group]["neg"].append(scores[inst.instance_id])
    return {g: {"auc": auc(d["pos"], d["neg"]), "n_pos": len(d["pos"]), "n_neg": len(d["neg"]),
                "mean_pos": sum(d["pos"]) / len(d["pos"]) if d["pos"] else float("nan"),
                "mean_neg": sum(d["neg"]) / len(d["neg"]) if d["neg"] else float("nan")}
            for g, d in groups.items()}


def main() -> int:
    out = evaluate()
    path = MODELS / "ood.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
