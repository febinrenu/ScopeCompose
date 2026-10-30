"""Case-complete accuracy (CCA): can a reader get every person's outcome from the answer?

The harm this paper measures is invisible to single-reference metrics: a system
that returns only the general rule scores perfectly against a one-answer gold,
yet is wrong for everyone the exception protects. CCA asks the question that
exposes it. For each *persona* (a short description of one person, falling in
exactly one gold region), a **reader** that sees only the system's answer must
say what outcome holds for that person -- and prove it by quoting the answer.

The metric is only as good as the reader is hard to fool, so a determination
counts (**CCA-strict**, the primary metric) only if **all four** hold:

1. ``evidence`` is a verbatim substring of the answer -- the reader cannot
   quote something the answer never said;
2. a local NLI cross-encoder judges that ``evidence`` plus the persona
   statement *entails* "the outcome is <outcome>" -- the quoted span must
   actually support the conclusion, not merely sit near it;
3. the span is not hedged -- "may receive a fee waiver" determines nothing;
4. the outcome matches the gold outcome under deterministic normalisation
   (:func:`metrics.values.values_match`).

If any of 1-3 fails the persona is scored ``NOT_DETERMINED``, which is wrong.
**CCA-lenient** drops 1-3 and is reported next to it, so the size of the
gap-filling the safeguards remove is itself a result.

**What this deliberately is not.** There is no LLM judge anywhere in the
scoring path: the reader proposes, deterministic checks and a local NLI model
disposes. The 2026 literature finds LLM judges detect planted additions far
better than omissions, which is the direction that would hide the effect under
study; this protocol never asks a model to notice an omission.

The reader is **evaluation-only** (a local Llama, pinned, never evaluated as a
system) and may not fall back to another model: a different reader is a
different metric.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from contract.gold import GoldInstance, Persona
from metrics.stats import Interval, wilson
from metrics.values import determines_outcome, values_match

NOT_DETERMINED = "NOT_DETERMINED"

#: Entailment probability a span + persona must reach to support an outcome.
#: A PLACEHOLDER until calibrated on the dev validation battery; the calibrated
#: value is written into the pre-registration. Any CCA quoted at the default is
#: describing an invented constant, so results carry ``nli_threshold_tuned``.
DEFAULT_NLI_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #

READER_SYSTEM = (
    "You check whether a written answer settles the outcome for one specific person. "
    "Use ONLY the answer text. Do not use outside knowledge and do not guess.\n\n"
    "How to read an answer:\n"
    "- An answer may state a general rule and then exceptions, each with a condition. A person who "
    "meets an exception's condition gets that exception's outcome. A person who clearly does NOT meet "
    "any stated exception condition is covered by the general rule and gets the general rule's outcome.\n"
    "- If the answer only lists possible outcomes without saying who each one applies to, or it hedges "
    "('may', 'might', 'could', 'in some cases', 'depending on'), it does not settle the outcome for "
    "anyone.\n"
    "- If the answer says nothing that applies to this person, or the person's circumstances are not "
    "enough to tell which statement applies, the outcome is "
    f'"{NOT_DETERMINED}".\n\n'
    "Reply with one JSON object and nothing else:\n"
    '{"outcome": "<short statement of what happens to this person, or NOT_DETERMINED>", '
    '"evidence": "<the sentence or clause COPIED WORD FOR WORD from the answer that states the outcome '
    'for this person>", '
    '"condition_used": "<the condition in the answer that matches this person, or null>"}'
)

READER_USER = (
    "Question the answer responds to:\n{query}\n\n"
    "Answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
    "Person:\n{persona}\n\n"
    "What outcome does the answer give for this person?"
)

#: Hashed into the pre-registration: the reader prompt is part of the metric.
READER_PROMPT_SHA = hashlib.sha256((READER_SYSTEM + READER_USER).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReaderOutput:
    outcome: str
    evidence: str
    condition_used: str | None = None
    raw: str = ""
    parse_ok: bool = True


class Reader(Protocol):
    def __call__(self, *, query: str, answer: str, persona: str) -> ReaderOutput: ...


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_reader_json(text: str) -> ReaderOutput:
    """Tolerant JSON extraction. Anything unparseable is NOT_DETERMINED.

    A reader that cannot produce the format has determined nothing; guessing at
    its prose would be exactly the gap-filling the metric exists to prevent.
    """
    m = _JSON_RE.search(text or "")
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and isinstance(d.get("outcome"), str):
                ev = d.get("evidence")
                cu = d.get("condition_used")
                return ReaderOutput(
                    outcome=d["outcome"].strip(),
                    evidence=ev.strip() if isinstance(ev, str) else "",
                    condition_used=cu.strip() if isinstance(cu, str) and cu.strip() else None,
                    raw=text,
                )
        except json.JSONDecodeError:
            pass
    return ReaderOutput(outcome=NOT_DETERMINED, evidence="", raw=text or "", parse_ok=False)


class LLMReader:
    """The production reader: the pinned local ``READER`` tier, through the cached client."""

    def __init__(self, *, step: str = "cca_reader", client=None):
        self.step = step
        self._client = client

    def __call__(self, *, query: str, answer: str, persona: str) -> ReaderOutput:
        from api_budget import client as api

        c = self._client or api.get_client()
        res = c.complete(
            tier=api.Tier.READER,
            step=self.step,
            system=READER_SYSTEM,
            messages=[{"role": "user",
                       "content": READER_USER.format(query=query, answer=answer, persona=persona)}],
            response_format={"type": "json_object"},
        )
        return parse_reader_json(res.text)


# --------------------------------------------------------------------------- #
# The four conditions
# --------------------------------------------------------------------------- #


class Failed(str, Enum):
    NOT_DETERMINED = "reader_not_determined"
    NOT_VERBATIM = "evidence_not_verbatim"
    NOT_ENTAILED = "evidence_does_not_entail_outcome"
    HEDGED = "evidence_is_hedged"
    WRONG = "outcome_does_not_match_gold"
    UNPARSEABLE = "reader_output_unparseable"


def _squash(s: str) -> str:
    return " ".join(s.split()).casefold()


def evidence_is_verbatim(evidence: str, answer: str) -> bool:
    """Condition 1. Whitespace- and case-insensitive; otherwise exact.

    Quote marks the reader wrapped round the span and trailing ellipses are
    stripped first -- they are formatting, not content. Nothing else is forgiven:
    a paraphrase is not a quote.
    """
    ev = evidence.strip().strip("\"'“”‘’").rstrip(".… ").strip()
    return bool(ev) and _squash(ev) in _squash(answer)


class NLIScorerLike(Protocol):
    def score_batch(self, pairs): ...


def evidence_entails(
    evidence: str, persona: str, outcome: str, nli: NLIScorerLike, *, threshold: float
) -> tuple[bool, float]:
    """Condition 2: (evidence + persona) entails "the outcome is <outcome>"."""
    premise = f"{evidence.strip()} {persona.strip()}"
    hypothesis = f"The outcome for this person: {outcome.strip()}."
    (score,) = nli.score_batch([(premise, hypothesis)])
    return score.entailment >= threshold, float(score.entailment)


# --------------------------------------------------------------------------- #
# Scoring one persona, one instance, a corpus
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PersonaVerdict:
    persona_id: str
    branch_id: str
    is_exception: bool
    gold_outcome: str
    reader_outcome: str
    evidence: str
    strict_correct: bool
    lenient_correct: bool
    failed: Failed | None
    entailment: float | None = None
    over_generalised: bool = False
    """A default-branch person told the *exception's* outcome."""


def score_persona(
    inst: GoldInstance,
    persona: Persona,
    answer: str,
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
) -> PersonaVerdict:
    branch = inst.branch_for_persona(persona.persona_id)
    out = reader(query=inst.query, answer=answer, persona=persona.statement)
    base = dict(
        persona_id=persona.persona_id, branch_id=branch.branch_id,
        is_exception=not branch.is_default, gold_outcome=branch.outcome,
        reader_outcome=out.outcome, evidence=out.evidence,
    )

    if not out.parse_ok:
        return PersonaVerdict(**base, strict_correct=False, lenient_correct=False, failed=Failed.UNPARSEABLE)
    if out.outcome.strip().upper() == NOT_DETERMINED or not out.outcome.strip():
        return PersonaVerdict(**base, strict_correct=False, lenient_correct=False, failed=Failed.NOT_DETERMINED)

    matches_gold = values_match(branch.outcome, out.outcome)
    lenient = matches_gold

    # Over-generalisation: a default-branch person told an exception's outcome.
    over = False
    if branch.is_default:
        over = any(values_match(b.outcome, out.outcome) and not values_match(branch.outcome, out.outcome)
                   for b in inst.exception_branches)

    # Conditions 1-3 decide whether the reader *determined* anything; 4 whether it is right.
    failed: Failed | None = None
    entail_score: float | None = None
    if not evidence_is_verbatim(out.evidence, answer):
        failed = Failed.NOT_VERBATIM
    else:
        ok, entail_score = evidence_entails(out.evidence, persona.statement, out.outcome, nli,
                                            threshold=threshold)
        if not ok:
            failed = Failed.NOT_ENTAILED
        elif not determines_outcome(out.evidence):
            failed = Failed.HEDGED
    if failed is None and not matches_gold:
        failed = Failed.WRONG

    return PersonaVerdict(
        **base, strict_correct=failed is None, lenient_correct=lenient, failed=failed,
        entailment=entail_score,
        over_generalised=over and failed in (None, Failed.WRONG),
    )


_QUESTION_RE = re.compile(r"[^.!?]*\?")


def is_clarification_only(answer: str) -> bool:
    """True if the answer asks a question instead of giving one.

    Asking is a legitimate alternative to a scoped answer and is *not
    evaluated* here (a stated limitation) -- but it must be reported as its own
    rate, so a system that clarifies cannot silently be scored as one that
    suppressed.
    """
    text = (answer or "").strip()
    if not text:
        return False
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    questions = sum(1 for s in sentences if s.strip().endswith("?"))
    return questions > 0 and questions >= len(sentences) / 2 and not re.search(r"\d", text)


@dataclass(frozen=True)
class InstanceCCA:
    instance_id: str
    verdicts: tuple[PersonaVerdict, ...]
    clarification_only: bool

    @property
    def cca_all(self) -> bool:
        """Every persona correct, strictly. The primary endpoint."""
        return bool(self.verdicts) and all(v.strict_correct for v in self.verdicts)

    @property
    def cca_all_lenient(self) -> bool:
        return bool(self.verdicts) and all(v.lenient_correct for v in self.verdicts)


def score_instance(
    inst: GoldInstance,
    answer: str,
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
) -> InstanceCCA:
    if not inst.personas:
        raise ValueError(f"{inst.instance_id}: no personas, so case-complete accuracy is undefined")
    return InstanceCCA(
        instance_id=inst.instance_id,
        verdicts=tuple(score_persona(inst, p, answer, reader, nli, threshold=threshold)
                       for p in inst.personas),
        clarification_only=is_clarification_only(answer),
    )


@dataclass
class CCAScores:
    n_instances: int
    cca_all: Interval
    cca_all_lenient: Interval
    cca_exception: Interval | None
    cca_default: Interval | None
    over_generalisation: Interval | None
    clarification_rate: float
    failures: dict[str, int] = field(default_factory=dict)
    nli_threshold: float = DEFAULT_NLI_THRESHOLD
    nli_threshold_tuned: bool = False

    def render(self) -> str:
        def f(i: Interval | None) -> str:
            return "n/a" if i is None else f"{i.point:.3f} [{i.low:.3f}, {i.high:.3f}]"

        lines = [f"CCA over {self.n_instances} instances"
                 + ("" if self.nli_threshold_tuned
                    else f"  (NLI threshold {self.nli_threshold} is UNTUNED)"),
                 f"  CCA-all strict        {f(self.cca_all)}",
                 f"  CCA-all lenient       {f(self.cca_all_lenient)}",
                 f"  CCA-E (exception)     {f(self.cca_exception)}",
                 f"  CCA-D (default)       {f(self.cca_default)}",
                 f"  over-generalisation   {f(self.over_generalisation)}",
                 f"  clarification rate    {self.clarification_rate:.3f}"]
        if self.failures:
            lines.append("  strict failures: " + ", ".join(f"{k}={v}" for k, v in sorted(self.failures.items())))
        return "\n".join(lines)


def _ratio(num: int, den: int) -> Interval | None:
    return wilson(num, den) if den else None


def aggregate(
    results: list[InstanceCCA],
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
    threshold_tuned: bool = False,
) -> CCAScores:
    """Pool instance results. Intervals are Wilson over *instances* for CCA-all
    and over *personas* for the per-role rates; persona-level intervals treat
    personas as independent, which they are not (they share an instance), so
    the instance-clustered bootstrap in ``metrics.cluster`` is the sensitivity
    check for any comparison that matters."""
    n = len(results)
    vs = [v for r in results for v in r.verdicts]
    exc = [v for v in vs if v.is_exception]
    dflt = [v for v in vs if not v.is_exception]
    failures: dict[str, int] = {}
    for v in vs:
        if v.failed is not None:
            failures[v.failed.value] = failures.get(v.failed.value, 0) + 1
    return CCAScores(
        n_instances=n,
        cca_all=wilson(sum(r.cca_all for r in results), n),
        cca_all_lenient=wilson(sum(r.cca_all_lenient for r in results), n),
        cca_exception=_ratio(sum(v.strict_correct for v in exc), len(exc)),
        cca_default=_ratio(sum(v.strict_correct for v in dflt), len(dflt)),
        over_generalisation=_ratio(sum(v.over_generalised for v in dflt), len(dflt)),
        clarification_rate=(sum(r.clarification_only for r in results) / n) if n else 0.0,
        failures=failures,
        nli_threshold=threshold,
        nli_threshold_tuned=threshold_tuned,
    )


def score_corpus(
    instances: list[GoldInstance],
    answers: dict[str, str],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
    threshold_tuned: bool = False,
    on_instance: Callable[[InstanceCCA], None] | None = None,
) -> tuple[CCAScores, list[InstanceCCA]]:
    """Score every instance that has personas and an answer; refuse silent gaps.

    An instance with personas but no answer is an error, not a zero: a missing
    answer scored as wrong would let a crashed run masquerade as a result.
    """
    results: list[InstanceCCA] = []
    for inst in instances:
        if not inst.personas:
            continue
        if inst.instance_id not in answers:
            raise KeyError(f"no answer for {inst.instance_id}; a missing answer is not a wrong one")
        r = score_instance(inst, answers[inst.instance_id], reader, nli, threshold=threshold)
        results.append(r)
        if on_instance:
            on_instance(r)
    return aggregate(results, threshold=threshold, threshold_tuned=threshold_tuned), results
