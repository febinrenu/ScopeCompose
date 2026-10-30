"""The human entailment audit: gate V2 and the calibration of the entailment threshold.

CCA-strict stands or falls on one claim: when the pipeline says "the answer establishes this
person's outcome", a person reading the answer agrees. That is measured, not assumed.

**The question put to people is always the same**, and it is about the *gold* outcome:

    "Does this answer, by itself, establish that for this person: <gold outcome>?"   (yes / no)

and never "did the reader guess right". Asked this way, two annotators blind to the system and to
the reader's output provide the truth against which the pipeline's verdict -- *strict-correct or
not* -- is scored:

* **precision** -- of the decisions the pipeline accepts, the share people agree establish the
  outcome. A false acceptance inflates CCA, so the bar is high (>= 0.95);
* **recall** -- of the decisions people say do establish it, the share the pipeline accepts
  (>= 0.85). A false rejection only costs sensitivity.

**Why the flags are recorded independently of the threshold.** Whether the entailment condition
passes depends on the threshold tau, and tau is calibrated *from this audit*. So every decision is
stored as raw flags (verbatim? hedged? matches gold? what entailment score?), and
:func:`strict_at` recomputes the pipeline's verdict at any tau offline. A test holds that
recomputation equal to the live scorer, so calibration cannot quietly disagree with scoring.

**Calibration** picks the **smallest tau that reaches the precision bar** among decisions the
other three conditions already accept -- the most permissive threshold that is still safe -- and
says so plainly when none does.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from contract.gold import GoldInstance
from metrics.case_accuracy import (
    NOT_DETERMINED,
    NLIScorerLike,
    Reader,
    evidence_entails,
    evidence_is_verbatim,
)
from metrics.stats import Interval, wilson
from metrics.values import determines_outcome, values_match

PRECISION_BAR = 0.95
RECALL_BAR = 0.85


@dataclass(frozen=True)
class Decision:
    """One reader decision, as raw flags (independent of the entailment threshold)."""

    item_id: str
    instance_id: str
    persona_id: str
    system: str
    query: str
    answer: str
    persona: str
    gold_outcome: str
    reader_outcome: str
    evidence: str
    determined: bool          # the reader committed to an outcome (not NOT_DETERMINED / unparseable)
    verbatim: bool
    hedged: bool
    matches_gold: bool
    entailment: float | None  # None when the span was not verbatim (the NLI model was never asked)


def explain(inst: GoldInstance, persona_id: str, system: str, answer: str, reader: Reader,
            nli: NLIScorerLike) -> Decision:
    """Run the reader and record every condition's raw outcome, with no threshold applied."""
    persona = next(p for p in inst.personas if p.persona_id == persona_id)
    gold = inst.branch_for_persona(persona_id).outcome
    out = reader(query=inst.query, answer=answer, persona=persona.statement)
    determined = out.outcome.strip().upper() != NOT_DETERMINED and bool(out.outcome.strip())
    verbatim = hedged = matches = False
    ent: float | None = None
    if determined:
        verbatim = evidence_is_verbatim(out.evidence, answer)
        hedged = not determines_outcome(out.evidence)
        matches = values_match(gold, out.outcome)
        if verbatim:
            _, ent = evidence_entails(out.evidence, persona.statement, out.outcome, nli, threshold=0.0)
    return Decision(f"{inst.instance_id}|{persona_id}|{system}", inst.instance_id, persona_id, system,
                    inst.query, answer, persona.statement, gold, out.outcome, out.evidence,
                    determined, verbatim, hedged, matches, ent)


def strict_at(d: Decision | Mapping, tau: float) -> bool:
    """The pipeline's strict-correct verdict at threshold ``tau``, from the raw flags."""
    g = d.__getitem__ if isinstance(d, Mapping) else (lambda k: getattr(d, k))
    ent = g("entailment")
    return bool(g("determined") and g("verbatim") and ent is not None and ent >= tau
                and not g("hedged") and g("matches_gold"))


def lenient(d: Decision | Mapping) -> bool:
    g = d.__getitem__ if isinstance(d, Mapping) else (lambda k: getattr(d, k))
    return bool(g("determined") and g("matches_gold"))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_for_audit(decisions: Sequence[Decision], *, n: int = 200, tau: float = 0.5, seed: int = 0
                     ) -> list[Decision]:
    """A stratified sample, so both precision and recall are estimable.

    Strata: accepted by the pipeline at ``tau`` (needed for precision), rejected although the reader
    committed to the right outcome (the interesting false rejections -- hedges, bad quotes, weak
    entailment), and rejected with nothing determined. Half the sample goes to *accepted*; the rest is
    split between the two rejected strata. A stratum that cannot fill its share is topped up from the
    others; the shortfall is visible in the result."""
    rng = random.Random(seed)
    strata: dict[str, list[Decision]] = {"accepted": [], "rejected_committed": [], "rejected_silent": []}
    for d in sorted(decisions, key=lambda x: x.item_id):
        if strict_at(d, tau):
            strata["accepted"].append(d)
        elif d.determined:
            strata["rejected_committed"].append(d)
        else:
            strata["rejected_silent"].append(d)
    for s in strata.values():
        rng.shuffle(s)
    want = {"accepted": n // 2, "rejected_committed": (n - n // 2 + 1) // 2, "rejected_silent": (n - n // 2) // 2}
    chosen: list[Decision] = []
    for k, w in want.items():
        chosen += strata[k][:w]
        strata[k] = strata[k][w:]
    rest = [d for k in want for d in strata[k]]
    rng.shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    return sorted(chosen, key=lambda d: d.item_id)


def to_review_payload(d: Decision) -> dict:
    """What a person sees. Deliberately omits the reader's outcome, the evidence and the system: the audit is
    blind to all three, and what is asked is only whether the answer establishes the *gold* outcome."""
    return {"item_id": d.item_id, "query": d.query, "answer": d.answer, "persona": d.persona,
            "claim": d.gold_outcome}


# --------------------------------------------------------------------------- #
# V2
# --------------------------------------------------------------------------- #


@dataclass
class V2Result:
    tau: float
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: Interval | None
    recall: Interval | None
    unsure: int = 0

    @property
    def passes(self) -> bool | None:
        """``None`` when precision or recall cannot be estimated (no accepted decisions, or no positives)."""
        if self.precision is None or self.recall is None:
            return None
        return self.precision.point >= PRECISION_BAR and self.recall.point >= RECALL_BAR

    def render(self) -> str:
        def f(i: Interval | None) -> str:
            return "n/a" if i is None else f"{i.point:.3f} [{i.low:.3f}, {i.high:.3f}]"

        verdict = {True: "PASS", False: "FAIL", None: "NOT EVALUABLE"}[self.passes]
        return "\n".join([
            f"Reader audit at tau = {self.tau:.3f} over {self.n} labelled decisions ({self.unsure} unsure, excluded)",
            f"  precision (accepted decisions people agree establish the outcome)  {f(self.precision)}   need >= {PRECISION_BAR}",
            f"  recall    (established outcomes the pipeline accepts)              {f(self.recall)}   need >= {RECALL_BAR}",
            f"  confusion: tp={self.tp} fp={self.fp} fn={self.fn} tn={self.tn}",
            f"  V2 reader validity: {verdict}"])


def evaluate_v2(decisions: Sequence[Decision], truth: Mapping[str, bool], *, tau: float) -> V2Result:
    """Score the pipeline against human truth. Items with no label are skipped, never guessed."""
    tp = fp = fn = tn = 0
    for d in decisions:
        if d.item_id not in truth:
            continue
        pred, true = strict_at(d, tau), truth[d.item_id]
        tp += pred and true
        fp += pred and not true
        fn += (not pred) and true
        tn += (not pred) and (not true)
    prec = wilson(tp, tp + fp) if tp + fp else None
    rec = wilson(tp, tp + fn) if tp + fn else None
    return V2Result(tau, tp + fp + fn + tn, tp, fp, fn, tn, prec, rec)


def calibrate_threshold(decisions: Sequence[Decision], truth: Mapping[str, bool], *,
                        precision_bar: float = PRECISION_BAR) -> float | None:
    """The smallest tau at which precision reaches the bar, among decisions the other conditions accept.

    Only decisions whose *non-entailment* conditions already hold can be moved by tau, so only they are
    considered; among them, candidate thresholds are the observed entailment scores. ``None`` if no threshold
    reaches the bar (say so; do not pick one)."""
    pool = [d for d in decisions if d.item_id in truth and d.determined and d.verbatim and not d.hedged
            and d.matches_gold and d.entailment is not None]
    if not pool:
        return None
    for tau in sorted({d.entailment for d in pool}):                  # type: ignore[misc]
        accepted = [d for d in pool if d.entailment >= tau]            # type: ignore[operator]
        if accepted and sum(truth[d.item_id] for d in accepted) / len(accepted) >= precision_bar:
            return float(tau)
    return None


def as_dicts(decisions: Sequence[Decision]) -> list[dict]:
    return [asdict(d) for d in decisions]


# --------------------------------------------------------------------------- #
# Driver: draw the sample, then score it once people have labelled it
# --------------------------------------------------------------------------- #


def build_decisions(instances: Sequence[GoldInstance], answers_by_system: Mapping[str, Mapping[str, str]],
                    reader: Reader, nli: NLIScorerLike) -> list[Decision]:
    """Run the reader on every (instance, system, persona) and record the raw flags."""
    out: list[Decision] = []
    for inst in instances:
        for system, answers in answers_by_system.items():
            if inst.instance_id not in answers:
                continue
            for p in inst.personas:
                out.append(explain(inst, p.persona_id, system, answers[inst.instance_id], reader, nli))
    return out


def truth_from_reviews(a: Mapping[str, str], b: Mapping[str, str], adjudicated: Mapping[str, bool] | None = None
                       ) -> tuple[dict[str, bool], list[str]]:
    """Human truth from two annotators' yes/no labels.

    An item is true/false only where both agree; the rest are returned as ``disputed`` for adjudication
    and are *not* guessed. ``adjudicated`` resolves them. An item only one annotator labelled is skipped."""
    truth: dict[str, bool] = {}
    disputed: list[str] = []
    for k in sorted(set(a) & set(b)):
        if a[k] == b[k]:
            truth[k] = a[k] == "yes"
        elif adjudicated and k in adjudicated:
            truth[k] = adjudicated[k]
        else:
            disputed.append(k)
    return truth, disputed


def load_decisions(path) -> list[Decision]:
    import json

    return [Decision(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="run the reader, draw a stratified sample, write blind review items")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--split", default="dev", choices=("train", "dev"))
    s.add_argument("--cells", type=Path, nargs="*", default=[], help="cell files of system answers (optional)")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--tau", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", type=Path, default=Path("experiments/runs/audit"))
    e = sub.add_parser("evaluate", help="score the pipeline against two annotators and calibrate tau")
    e.add_argument("--decisions", type=Path, required=True)
    e.add_argument("--annotators", nargs=2, required=True)
    e.add_argument("--review-dir", type=Path, default=None)
    e.add_argument("--adjudicated", type=Path, default=None, help='JSON {"item_id": true|false}')
    e.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        from detection.nli import CrossEncoderNLI
        from experiments.run_matrix import load_answers, load_instances
        from experiments.validity_battery import scorable, select
        from metrics.case_accuracy import LLMReader
        from metrics.validity import battery_outputs

        insts, bad = scorable(select(load_instances(args.data), split=args.split))
        if bad:
            print(f"note: {len(bad)} instance(s) lack personas, branches or reference answers and are not audited")
        systems: dict[str, dict[str, str]] = {}
        for i in insts:
            for name, text in battery_outputs(i).items():
                systems.setdefault(f"battery:{name}", {})[i.instance_id] = text
        for path in args.cells:
            systems[path.stem] = load_answers(path)
        ds = build_decisions(insts, systems, LLMReader(step="reader_audit"), CrossEncoderNLI())
        picked = sample_for_audit(ds, n=args.n, tau=args.tau, seed=args.seed)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "decisions.jsonl").write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in as_dicts(ds)), encoding="utf-8")
        (args.out_dir / "audit_items.jsonl").write_text(
            "".join(json.dumps(to_review_payload(d), ensure_ascii=False) + "\n" for d in picked), encoding="utf-8")
        print(f"{len(ds)} decisions explained; {len(picked)} sampled for the audit -> {args.out_dir}")
        print("review with: python -m benchmark.review --kind entailment review "
              f"--items {args.out_dir / 'audit_items.jsonl'} --annotator <you>")
        return 0

    from benchmark.review import DEFAULT_DIR, ReviewStore, labels

    directory = args.review_dir or DEFAULT_DIR
    a, b = (labels(ReviewStore(directory, "entailment", n)) for n in args.annotators)
    adj = json.loads(args.adjudicated.read_text(encoding="utf-8")) if args.adjudicated else None
    truth, disputed = truth_from_reviews(a, b, adj)
    ds = load_decisions(args.decisions)
    print(f"{len(truth)} items with agreed labels; {len(disputed)} disputed (adjudicate, do not guess)")
    result = evaluate_v2(ds, truth, tau=args.tau)
    print(result.render())
    tau = calibrate_threshold(ds, truth)
    print(f"calibrated tau: {tau if tau is None else round(tau, 3)}"
          + ("  (no threshold reaches the precision bar)" if tau is None else ""))
    return 0 if result.passes else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
