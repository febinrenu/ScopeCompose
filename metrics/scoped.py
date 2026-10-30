"""Scoped precision: of what an answer asserts, how much is right *and attached to the right people*?

Preservation Rate counts how many gold outcomes appear somewhere in an answer. It has no precision
side, so pasting every passage scores as well as writing the answer (0.855 against 0.699 in the
pilot). Scoped precision is the missing half. Each **claim** in an answer gets one label against the
gold branches:

``aligned``
    states a gold outcome *and* is attached to the right group;
``misattached``
    states a gold outcome but to the wrong people -- the exception's outcome given to everyone, or the
    default's outcome given to the exception group. Right words, wrong person: exactly the error that
    case-complete accuracy exists to catch, and the one a coverage metric cannot see;
``unsupported``
    states an outcome that contradicts every gold branch (a different figure, the opposite polarity);
``extraneous``
    says nothing about any gold outcome (filler, boilerplate, an unrelated passage's sentence).

``scoped precision = aligned / all claims``. Pasting K passages adds extraneous claims and lowers it;
listing outcomes without saying for whom adds misattached ones.

**Deterministic first.** Labels come from the same value matcher as CCA (``metrics.values``) plus a
simple attachment test. An optional judge (the evaluation model) may be consulted *only* for claims the
deterministic path calls ``extraneous``, and a judge's verdict never overrides an ``aligned`` or
``misattached`` one -- it can only rescue a claim the matcher could not read. Whether the judge is good
enough to use is gate V3, validated against human labels.

**Attachment.** A claim is *conditioned* if it carries a restricting phrase ("if", "for ... holders",
"unless"). A default outcome must be stated unconditioned (or with the default's own scope); an
exception outcome must be conditioned *and* share content with that branch's condition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from benchmark.chunker import split_sentences
from benchmark.vacs import content_tokens
from contract.gold import Branch, GoldInstance
from metrics.branch_match import polarity_conflict
from metrics.values import canonicalise_outcome, figures_conflict, strip_citations, values_match

#: Minimum content-word overlap between a claim and an exception's condition for the claim to count as
#: attached to that group.
ATTACH_OVERLAP = 0.25


class Label(str, Enum):
    ALIGNED = "aligned"
    MISATTACHED = "misattached"
    UNSUPPORTED = "unsupported"
    EXTRANEOUS = "extraneous"


_GROUP_NOUNS = (r"holders?|cardholders?|customers?|members?|applicants?|students?|people|persons?|users?|"
                r"residents?|claimants?|employees?|workers?|accounts?|those|anyone|individuals?|parents?|carers?")
_CONDITIONED = re.compile(
    rf"\b(?:if|unless|when|where|whenever|provided|only if|who (?:is|are|has|have|holds|hold)|holders? of|"
    rf"for\b[^.;]{{0,60}}?\b(?:{_GROUP_NOUNS})|\w+[- ]tier|in the case of|in cases where|as an? \w+)\b", re.I)
_EVERYONE = re.compile(r"\b(?:all|every|everyone|any|each|anyone|regardless)\b", re.I)
_HOWEVER = re.compile(r"^\s*(?:however|but|although|except|otherwise)[,\s]+", re.I)


def split_claims(answer: str) -> list[str]:
    """Claims are sentences, with a leading connective stripped and citations removed."""
    out = []
    for s in split_sentences(answer or ""):
        s = _HOWEVER.sub("", strip_citations(s)).strip()
        if len(content_tokens(s)) >= 1 or re.search(r"\d", s):
            out.append(s)
    return out


def conditioned(claim: str) -> bool:
    return bool(_CONDITIONED.search(claim))


def _scope_text(b: Branch) -> str:
    return f"{b.condition or ''} {b.applicability.descriptor}"


def _attached(claim: str, b: Branch) -> bool:
    """Is ``claim`` attached to the group ``b`` governs?"""
    if b.is_default:
        return not conditioned(claim) or bool(_EVERYONE.search(claim))
    if not conditioned(claim):
        return False
    a, c = content_tokens(claim), content_tokens(_scope_text(b))
    return bool(c) and len(a & c) / len(c) >= ATTACH_OVERLAP


@dataclass(frozen=True)
class ClaimLabel:
    claim: str
    label: Label
    branch_id: str | None = None
    by: str = "deterministic"


Judge = Callable[[str, Sequence[Branch]], "Label | None"]


def label_claim(claim: str, branches: Sequence[Branch], *, judge: Judge | None = None) -> ClaimLabel:
    matched = [b for b in branches if values_match(b.outcome, claim)]
    if matched:
        for b in matched:
            if _attached(claim, b):
                return ClaimLabel(claim, Label.ALIGNED, b.branch_id)
        return ClaimLabel(claim, Label.MISATTACHED, matched[0].branch_id)
    cc = canonicalise_outcome(claim)
    if any(figures_conflict(b.outcome, claim) or polarity_conflict(canonicalise_outcome(b.outcome), cc)
           for b in branches):
        return ClaimLabel(claim, Label.UNSUPPORTED)
    if judge is not None:
        verdict = judge(claim, branches)
        # the judge may rescue an unreadable claim; it can never make a bad one good
        if verdict in (Label.ALIGNED, Label.MISATTACHED, Label.UNSUPPORTED):
            return ClaimLabel(claim, verdict, None, by="judge")
    return ClaimLabel(claim, Label.EXTRANEOUS)


@dataclass
class ScopedScore:
    instance_id: str
    labels: list[ClaimLabel] = field(default_factory=list)
    n_words: int = 0

    def count(self, lab: Label) -> int:
        return sum(1 for c in self.labels if c.label is lab)

    @property
    def n_claims(self) -> int:
        return len(self.labels)

    @property
    def precision(self) -> float:
        """aligned / all claims. An answer that makes no claims has precision 0, not 1: nothing was said."""
        return self.count(Label.ALIGNED) / self.n_claims if self.n_claims else 0.0

    @property
    def attachment_error_rate(self) -> float | None:
        """Of the claims that state a gold outcome, the share given to the wrong group."""
        stated = self.count(Label.ALIGNED) + self.count(Label.MISATTACHED)
        return self.count(Label.MISATTACHED) / stated if stated else None

    def covered_branches(self) -> set[str]:
        return {c.branch_id for c in self.labels if c.label is Label.ALIGNED and c.branch_id}


def score(inst: GoldInstance, answer: str, *, judge: Judge | None = None) -> ScopedScore:
    if not inst.gold_branches:
        raise ValueError(f"{inst.instance_id}: scoped precision needs gold branches")
    s = ScopedScore(inst.instance_id, n_words=len((answer or "").split()))
    s.labels = [label_claim(c, inst.gold_branches, judge=judge) for c in split_claims(answer)]
    return s


def fabricated_exception(inst: GoldInstance, answer: str) -> bool:
    """On an instance with no exception: does the answer invent one?

    True if any claim is conditioned yet states an outcome (a restriction the gold does not contain). Measured
    on controls, where the right answer has a single unconditioned outcome."""
    if inst.exception_branches:
        raise ValueError(f"{inst.instance_id}: has exceptions; fabrication is measured on controls only")
    return any(conditioned(c) and re.search(r"\d|no |not |waive|free|eligible|permit|prohibit", c, re.I)
               for c in split_claims(answer))


@dataclass
class Aggregate:
    n_instances: int
    precision: float
    attachment_error_rate: float | None
    mean_claims: float
    mean_words: float
    counts: dict[str, int]

    def render(self) -> str:
        aer = "n/a" if self.attachment_error_rate is None else f"{self.attachment_error_rate:.3f}"
        return (f"scoped precision {self.precision:.3f}  attachment errors {aer}  "
                f"({self.n_instances} instances, {self.mean_claims:.1f} claims and {self.mean_words:.0f} words each; "
                + ", ".join(f"{k} {v}" for k, v in self.counts.items()) + ")")


def aggregate(scores: Sequence[ScopedScore]) -> Aggregate:
    """Micro-averaged over claims: every claim counts once, so a long answer cannot hide its extraneous ones."""
    if not scores:
        raise ValueError("no scored instances")
    counts = {lab.value: sum(s.count(lab) for s in scores) for lab in Label}
    n = sum(s.n_claims for s in scores)
    stated = counts["aligned"] + counts["misattached"]
    return Aggregate(len(scores), counts["aligned"] / n if n else 0.0,
                     counts["misattached"] / stated if stated else None, n / len(scores),
                     sum(s.n_words for s in scores) / len(scores), counts)


def mean_precision(instances: Sequence[GoldInstance], answers: Mapping[str, str], *, judge: Judge | None = None) -> float:
    """Aggregate scoped precision for a set of answers."""
    return aggregate([score(i, answers[i.instance_id], judge=judge) for i in instances]).precision


# --------------------------------------------------------------------------- #
# The optional judge, and V3
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You label one claim from an answer against the correct outcomes for a question. Return ONLY JSON: "
    '{"label": "aligned" | "misattached" | "unsupported" | "extraneous"}. aligned: the claim states a correct '
    "outcome for the right group. misattached: it states a correct outcome but for the wrong group. "
    "unsupported: it states an outcome that is not correct for anyone. extraneous: it states no outcome at all."
)


def eval_judge(client=None) -> Judge:
    """A claim judge backed by the EVAL tier. Consulted only for claims the matcher could not read."""
    import json

    from api_budget import Tier, get_client

    c = client or get_client()

    def judge(claim: str, branches: Sequence[Branch]) -> Label | None:
        listing = "\n".join(f"- for {('everyone else' if b.is_default else (b.condition or b.applicability.descriptor))}: "
                            f"{b.outcome}" for b in branches)
        res = c.complete(messages=[{"role": "user", "content": f"Correct outcomes:\n{listing}\n\nClaim: {claim}"}],
                         system=JUDGE_SYSTEM, tier=Tier.EVAL, step="scoped_judge",
                         response_format={"type": "json_object"})
        try:
            return Label(str(json.loads(res.text).get("label", "")).strip().lower())
        except (ValueError, TypeError, AttributeError):
            return None                  # an unreadable verdict is no verdict

    return judge


V3_ACCURACY_BAR = 0.80


def validate_judge(predicted: Mapping[str, Label], human: Mapping[str, Label]) -> tuple[float | None, int]:
    """(accuracy, n) of a judge's labels against human labels on the claims both cover. ``None`` with no overlap."""
    shared = sorted(set(predicted) & set(human))
    if not shared:
        return None, 0
    return sum(predicted[k] is human[k] for k in shared) / len(shared), len(shared)
