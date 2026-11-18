"""Rescore every cell in Tables 4 and 7 with the official ConditionalQA evaluate.py, from the raw cell files and
the raw benchmark files, without the study's scoring code, and compare with the generated LaTeX tables.

    python paper/check_tables.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "third_party" / "conditionalqa" / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

refs = {}
for split in ("train", "dev"):
    for d in json.load(open(ROOT / "data" / "external" / "conditionalqa" / f"{split}.json", encoding="utf-8")):
        refs[d["id"]] = d["answers"]

LABEL = {"closed": "closed-book", "gold": "gold", "gold_nocondel": r"gold\,$-$\,cond.\ elements",
         "gold_nocondchk": r"gold\,$-$\,cond.-only chunks", "goldpad8": r"gold\,+\,distractors", "top3": "top-3",
         "top8": "top-8", "rr8to3": "rerank-3", "nli": "NLI filter", "crs": "ConflictRAG-style",
         "select": "listwise selection", "adj": "adjacent", "page": "page", "qc0": "completion (untrained)",
         "qc": r"completion (pre-reg.\ $\tau$)", "qcclean": "completion", "sentwin": "sentence window",
         "parentchild": "parent-child"}


def table_rows(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\texttt\{(.+?)\}(?: \(ours\))?(?:\$\^\\d?dagger\$)?((?:, C\$\^\\dagger\$ only)?) & (.*)\\\\", line)
        if m:
            rows[(m.group(1), bool(m.group(2)))] = [c.strip() for c in m.group(3).split("&")]
    return rows


def official(preds, ids):
    """Mean F1 with conditions over ids, per question by the official compute_metrics."""
    return 100 * sum(ev.compute_metrics(preds.get(q, []), refs[q])[3] for q in ids) / len(ids)


def cells(qset, model):
    for p in sorted((ROOT / "study" / "runs" / qset).glob(f"*__{model}.jsonl")):
        recs = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["qid"]] = [[a, list(c)] for a, c in r["answers"]]
        yield p.stem.split("__")[0], recs


def check(table, qset, models, width):
    rows = table_rows(table)
    bad = n = 0
    for mi, model in enumerate(models):
        for system, preds in cells(qset, model):
            if system not in LABEL or (LABEL[system], False) not in rows:
                continue
            cond = [q for q in preds if any(a[1] for a in refs[q])]
            unc = [q for q in preds if refs[q] and not any(a[1] for a in refs[q])]
            got = rows[(LABEL[system], False)][mi * width: mi * width + width]
            for name, ids, cell in (("C", cond, got[0]), ("U", unc, got[-1])):
                if cell == "--" or not ids:
                    continue
                v = f"{official(preds, ids):.1f}"
                n += 1
                if v != cell:
                    bad += 1
                    print(f"MISMATCH {table} {model} {system} {name}: table {cell}, official {v} (n={len(ids)})")
    print(f"{Path(table).name}: {n} values rescored, {bad} mismatches")
    return bad


bad = check(ROOT / "paper/tables/endtoend.tex", "eval", ("qwen7b", "llama8b"), 3)
bad += check(ROOT / "paper/tables/hosted.tex", "hosted", ("oss20b", "gemma31b", "nemotron"), 3)
sys.exit(1 if bad else 0)
