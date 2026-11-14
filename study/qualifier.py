"""Qualifier scoring: which chunk of a page states a condition on the answer?

Given a question and the chunk a retriever ranked highest on a page (the
*anchor*), score every other chunk of that page for whether it holds a gold
condition element. The score drives **qualifier-aware completion**: a pipeline
keeps its retrieved chunks and adds the page's chunks that qualify them.

Labels come from ConditionalQA's own annotation -- a chunk is positive if it
holds a gold condition element of the question -- so no new annotation is
needed. Unconditional questions contribute negatives only, which is what
teaches the scorer when *not* to add anything.

**Cross-fitting.** Every question is scored by a model that never saw its page.
Evaluation questions are split into five document-disjoint folds; the model for
fold *f* is trained on the other four folds plus the tuning set. The tuning set
is scored by a model trained on all five evaluation folds, and is the only
place the inclusion threshold is chosen.

    python -m study.qualifier train      # five fold models + the tuning model (GPU)
    python -m study.qualifier score      # out-of-fold scores for every candidate pair

Two baselines need no training and use the same candidates: the off-the-shelf
reranker scoring (question, candidate), and plain adjacency.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from study import cqa, systems
from study.retrieve import ARTIFACTS

BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 512
SEED = 0
K_FOLDS = 5
ANCHOR_K = 5            # completion starts from the top-5 ...
BUDGET = systems.DEFAULT_K   # ... and may add chunks up to the same 8-chunk budget as top8
MAX_PAGES = 2           # candidate pages: those of the top-ranked chunks, at most two
NEG_PER_QUESTION = 6    # negatives sampled per question for training

MODELS = ARTIFACTS / "qualifier"
SCORES = ARTIFACTS / "qualifier" / "scores.json"


@dataclass(frozen=True)
class Pair:
    qid: str
    doc: int
    anchor: str          # chunk id
    candidate: str       # chunk id
    label: int           # 1 if the candidate holds a gold condition element


def candidate_pairs(q: cqa.Question, env: systems.Env) -> list[Pair]:
    """The pairs completion would score for ``q``: same pages as the top chunks, not already retrieved."""
    ranked = env.ranked(q)
    top = ranked[:ANCHOR_K]
    have = {c.id for c in top}
    pages: list[int] = []
    for c in top:
        if c.doc not in pages:
            pages.append(c.doc)
    pages = pages[:MAX_PAGES]
    gold = env.gold(q).condition_chunks if q.kind != "unanswerable" else frozenset()
    out = []
    for doc in pages:
        anchor = next(c for c in top if c.doc == doc)
        for c in env.doc_chunks(doc):
            if c.id not in have:
                out.append(Pair(q.id, doc, anchor.id, c.id, int(c.id in gold)))
    return out


def encode_input(q: cqa.Question, env: systems.Env, pair: Pair) -> tuple[str, str]:
    """(question + scenario + anchor, candidate). The tokenizer truncates the longer side first."""
    a = env.chunk(pair.anchor)
    c = env.chunk(pair.candidate)
    return (f"{q.question} {q.scenario}\n\n{a.body}", c.text)


def training_pairs(qs: Sequence[cqa.Question], env: systems.Env, *, seed: int = SEED) -> list[Pair]:
    """All positives, and a fixed number of negatives per question.

    Positives also include the page's condition chunks that the retriever had
    already found, paired with the anchor: at training time it does not matter
    whether the retriever found them, only what a qualifying chunk looks like.
    """
    rng = random.Random(seed)
    out: list[Pair] = []
    for q in qs:
        pairs = candidate_pairs(q, env)
        if q.kind == "conditional":
            g = env.gold(q)
            seen = {p.candidate for p in pairs}
            top = env.ranked(q)[:ANCHOR_K]
            for cid in sorted(g.condition_chunks - seen):
                doc = env.chunk(cid).doc
                anchor = next((c for c in top if c.doc == doc and c.id != cid), None)
                if anchor is None:
                    others = [x for x in env.doc_chunks(doc) if x.id != cid and x.id in g.all_chunks]
                    anchor = others[0] if others else None
                if anchor is not None:
                    pairs.append(Pair(q.id, doc, anchor.id, cid, 1))
        pos = [p for p in pairs if p.label]
        neg = [p for p in pairs if not p.label]
        out += pos + rng.sample(neg, min(len(neg), NEG_PER_QUESTION))
    return out


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


def fold_plan() -> dict[str, list[str]]:
    """``fold name -> question ids it scores``; the folds partition eval and tuning by document."""
    ev = cqa.eval_questions()
    doc_fold = cqa.folds(ev, K_FOLDS, seed=SEED)
    plan: dict[str, list[str]] = {f"f{i}": [] for i in range(K_FOLDS)}
    for q in ev:
        plan[f"f{doc_fold[q.doc]}"].append(q.id)
    plan["tuning"] = [q.id for q in cqa.tuning_questions()]
    return plan


def train_ids_for(fold: str, plan: dict[str, list[str]]) -> list[str]:
    if fold == "tuning":
        return [i for f, ids in plan.items() if f != "tuning" for i in ids]
    return [i for f, ids in plan.items() if f != fold for i in ids]


# --------------------------------------------------------------------------- #
# Training (GPU)
# --------------------------------------------------------------------------- #


def train_fold(fold: str, *, epochs: int = 2, batch: int = 4, accum: int = 8, lr: float = 2e-5) -> Path:
    """Effective batch 32. Fold f0 ran as 8 x 4; the rest run as 4 x 8 (the same gradient, half the peak
    memory), because 8 x 4 outgrew the 8 GB card and Windows spilled it into system RAM, 2-3x slower."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    out_dir = MODELS / fold
    if (out_dir / "done.json").exists():
        return out_dir
    torch.manual_seed(SEED)
    env = systems.Env()
    idx = cqa.by_id()
    plan = fold_plan()
    qs = [idx[i] for i in train_ids_for(fold, plan)]
    pairs = training_pairs(qs, env)
    random.Random(SEED).shuffle(pairs)
    texts = [encode_input(idx[p.qid], env, p) for p in pairs]
    labels = [p.label for p in pairs]
    n_pos = sum(labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":                 # free the cache instead of spilling past the card
        torch.cuda.set_per_process_memory_fraction(0.85)
    progress = MODELS / f"{fold}.progress"
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Explicit float32 master weights. Transformers 5 otherwise loads the checkpoint's stored
    # precision (float16 here), and AdamW on float16 weights turns every parameter into NaN
    # within a few steps. Mixed precision is applied only inside autocast below.
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2,
                                                               dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = math.ceil(len(pairs) / (batch * accum)) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    # Positives are rare; weight them so the classifier is not rewarded for never adding anything.
    weight = torch.tensor([1.0, max(1.0, (len(labels) - n_pos) / max(n_pos, 1))], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    model.train()
    step = 0
    for epoch in range(epochs):
        order = list(range(len(pairs)))
        random.Random(SEED + epoch).shuffle(order)
        for b, start in enumerate(range(0, len(order), batch)):
            ids = order[start:start + batch]
            enc = tok([texts[i][0] for i in ids], [texts[i][1] for i in ids], truncation="only_first",
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            y = torch.tensor([labels[i] for i in ids], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), y) / accum
            loss.backward()
            if (b + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    progress.write_text(f"{step}/{steps} steps, {time.time() - t0:.0f}s, "
                                        f"{torch.cuda.max_memory_reserved() / 2**30 if device == 'cuda' else 0:.1f} GB peak",
                                        encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "done.json").write_text(json.dumps(
        {"fold": fold, "n_pairs": len(pairs), "n_pos": n_pos, "epochs": epochs, "steps": step,
         "micro_batch": batch, "accum": accum, "seconds": round(time.time() - t0),
         "train_questions": len(qs)}), encoding="utf-8")
    # The optimizer and scheduler hold the weights and Adam's state (about 3x the model); free them
    # all, or the next fold trains on top of this one's leftovers and spills out of GPU memory.
    del model, opt, sched, loss_fn
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_dir


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_fold(fold: str, plan: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """``qid -> {candidate chunk id: P(qualifies)}`` for the questions this fold scores."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    env = systems.Env()
    idx = cqa.by_id()
    path = MODELS / fold
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(device).eval()
    out: dict[str, dict[str, float]] = {}
    for qid in plan[fold]:
        q = idx[qid]
        pairs = candidate_pairs(q, env)
        out[qid] = {}
        for start in range(0, len(pairs), 32):
            chunk = pairs[start:start + 32]
            texts = [encode_input(q, env, p) for p in chunk]
            enc = tok([t[0] for t in texts], [t[1] for t in texts], truncation="only_first",
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].tolist()
            for p, s in zip(chunk, probs):
                out[qid][p.candidate] = round(float(s), 5)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def zero_shot_scores(qs: Sequence[cqa.Question]) -> dict[str, dict[str, float]]:
    """The untrained baseline: the off-the-shelf reranker on (question, candidate)."""
    from study.operators import rerank_scores

    env = systems.Env()
    out = {}
    for q in qs:
        pairs = candidate_pairs(q, env)
        scores = rerank_scores(q.retrieval_query, [_P(p.candidate, env.chunk(p.candidate).text) for p in pairs])
        out[q.id] = {p.candidate: round(1 / (1 + math.exp(-s)), 5) for p, s in zip(pairs, scores)}
    return out


@dataclass(frozen=True)
class _P:
    id: str
    text: str


@lru_cache(maxsize=None)
def load_scores() -> dict[str, dict[str, dict[str, float]]]:
    if not SCORES.exists():
        raise FileNotFoundError(f"{SCORES} is missing; run `python -m study.qualifier score`")
    return json.loads(SCORES.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The completion systems
# --------------------------------------------------------------------------- #

THRESHOLD_FILE = MODELS / "threshold.json"


def complete(q: cqa.Question, env: systems.Env, scores: dict[str, float], threshold: float) -> systems.Context:
    """Top-``ANCHOR_K``, plus the best-scoring same-page candidates above ``threshold``, up to ``BUDGET``.

    If fewer than ``BUDGET - ANCHOR_K`` candidates qualify, the remaining slots
    are **left empty** rather than filled with retrieved chunks: the system
    should add qualifiers, not context for its own sake. The budget-matched
    comparison is ``top8``, which always uses all eight.
    """
    ranked = env.ranked(q)
    top = list(ranked[:ANCHOR_K])
    picks = sorted(((s, cid) for cid, s in scores.items() if s >= threshold), reverse=True)[: BUDGET - ANCHOR_K]
    added = [env.chunk(cid) for _, cid in picks]
    # Keep added chunks next to their page's other chunks, in page order: a condition read beside its rule.
    chunks = top + added
    return systems.Context(tuple(chunks), trace={"added": [c.id for c in added],
                                                 "scores": {cid: s for s, cid in picks}})


def adjacent(q: cqa.Question, env: systems.Env) -> systems.Context:
    """Budget-matched structural baseline: top-5 plus the chunks either side of them, up to 8."""
    ranked = env.ranked(q)
    top = list(ranked[:ANCHOR_K])
    have = {c.id for c in top}
    added = []
    for c in top:
        for nb in (c.index + 1, c.index - 1):
            page = env.doc_chunks(c.doc)
            if 0 <= nb < len(page) and page[nb].id not in have and len(top) + len(added) < BUDGET:
                added.append(page[nb])
                have.add(page[nb].id)
    return systems.Context(tuple(top + added), trace={"added": [c.id for c in added]})


def fullpage(q: cqa.Question, env: systems.Env) -> systems.Context:
    """The whole page of the top-ranked chunk. A strong, simple upper reference when pages fit the window."""
    top = env.ranked(q)[0]
    return systems.Context(env.doc_chunks(top.doc))


def _threshold(kind: str) -> float:
    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(f"{THRESHOLD_FILE} is missing; run `python -m study.qualifier threshold`")
    return float(json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))[kind])


def _qc(kind: str):
    def build(q: cqa.Question, env: systems.Env) -> systems.Context:
        return complete(q, env, load_scores()[kind].get(q.id, {}), _threshold(kind))
    return build


systems.register("qc")(_qc("trained"))
systems.register("qc0")(_qc("zeroshot"))
systems.register("adj")(adjacent)
systems.register("page")(fullpage)


# --------------------------------------------------------------------------- #
# Threshold and intrinsic quality (tuning set only for the choice)
# --------------------------------------------------------------------------- #


def intrinsic(kind: str, qids: Sequence[str], threshold: float) -> dict:
    """How well a threshold recovers missed condition chunks, at the pair level."""
    env = systems.Env()
    idx = cqa.by_id()
    tp = fp = fn = added = 0
    for qid in qids:
        q = idx[qid]
        s = load_scores()[kind].get(qid, {})
        picked = {cid for sc, cid in sorted(((v, k) for k, v in s.items() if v >= threshold), reverse=True)[: BUDGET - ANCHOR_K]}
        gold = {p.candidate for p in candidate_pairs(q, env) if p.label}
        tp += len(picked & gold)
        fp += len(picked - gold)
        fn += len(gold - picked)
        added += len(picked)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "added_per_q": added / max(len(qids), 1)}


def choose_threshold(kind: str) -> dict:
    """Maximise pair-level F1 of recovering missed condition chunks, on the tuning set."""
    qids = [q.id for q in cqa.tuning_questions()]
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    best = max((intrinsic(kind, qids, t) for t in grid), key=lambda d: (d["f1"], -d["threshold"]))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("train", "score", "threshold", "report"))
    ap.add_argument("--fold", default=None, help="train only this fold")
    args = ap.parse_args(argv)
    plan = fold_plan()
    if args.action == "train":
        for fold in ([args.fold] if args.fold else plan):
            path = train_fold(fold)
            print(fold, json.loads((path / "done.json").read_text(encoding="utf-8")), flush=True)
    elif args.action == "score":
        trained: dict[str, dict[str, float]] = {}
        for fold in plan:
            trained.update(score_fold(fold, plan))
            print(f"scored {fold}", flush=True)
        idx = cqa.by_id()
        zs = zero_shot_scores([idx[i] for ids in plan.values() for i in ids])
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        SCORES.write_text(json.dumps({"trained": trained, "zeroshot": zs}), encoding="utf-8")
    elif args.action == "threshold":
        chosen = {k: choose_threshold(k) for k in ("trained", "zeroshot")}
        THRESHOLD_FILE.write_text(json.dumps({k: v["threshold"] for k, v in chosen.items()}, indent=1), encoding="utf-8")
        print(json.dumps(chosen, indent=1))
    else:
        th = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
        ev = [q.id for q in cqa.eval_questions()]
        print(json.dumps({k: intrinsic(k, ev, th[k]) for k in th}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
