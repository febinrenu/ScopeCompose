"""Counterfactual persona pairs: is the reader sensitive to the right thing? (gate V2)

A reader that always answers "no fee" scores well on every persona who happens
to deserve "no fee". A reader that is thrown by a person's city scores badly for
a reason that has nothing to do with the system under test. Either defect
corrupts case-complete accuracy invisibly, so each instance carries minimal
persona pairs (proposal v8 section 6.5):

* **relevant flip** -- change only the attribute that defines the exception
  region. The gold outcome *changes*; a good reader's output must change too.
* **irrelevant flip** -- change an attribute no frame constrains (a city for a
  fee rule). The gold outcome does *not* change; neither may the reader's.

Two different questions are answered here, on purpose kept apart:

**Reader sensitivity** (V2). Hold the answer fixed at the *gold* scoped answer,
which contains everything a reader could need, and vary only the persona. The
criteria are fixed in advance: an irrelevant flip changes the reader's output
in at most ``IRRELEVANT_CHANGE_MAX`` of pairs; a relevant flip changes it in at
least ``RELEVANT_CHANGE_MIN``. This validates the *instrument*.

**System consistency.** Hold the persona pair fixed and read each *system's*
answer. A pair is consistent only if the reader *determines* an outcome for both
people and the two outcomes change or hold exactly as gold says. A system that
leaves both people undetermined is not "consistently holding"; it has told
nobody anything. This scores the *system*.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract.gold import GoldInstance
from metrics.case_accuracy import (
    DEFAULT_NLI_THRESHOLD,
    Failed,
    NLIScorerLike,
    Reader,
    score_persona,
)
from metrics.stats import Interval, wilson
from metrics.values import values_match

#: V2 criteria, fixed before the first run (proposal v8 section 3.5).
IRRELEVANT_CHANGE_MAX = 0.05
RELEVANT_CHANGE_MIN = 0.90

#: Failures that mean the reader said nothing usable, as opposed to something wrong.
_UNDETERMINED = (Failed.NOT_DETERMINED, Failed.UNPARSEABLE)


def _outputs_differ(a, b) -> bool:
    """Whether the reader's output changed between two personas.

    A switch between *determined* and *undetermined* counts as a change: a
    reader whose answer evaporates when it is told a person lives in York is
    unstable, not stable.
    """
    a_none, b_none = a.failed in _UNDETERMINED, b.failed in _UNDETERMINED
    if a_none or b_none:
        return a_none != b_none
    return not values_match(a.reader_outcome, b.reader_outcome)


@dataclass
class Sensitivity:
    n_relevant: int
    n_irrelevant: int
    relevant_changed: int
    irrelevant_changed: int

    @property
    def relevant_change_rate(self) -> Interval:
        return wilson(self.relevant_changed, self.n_relevant)

    @property
    def irrelevant_change_rate(self) -> Interval:
        return wilson(self.irrelevant_changed, self.n_irrelevant)

    @property
    def passes_v2(self) -> bool | None:
        """``None`` when there are no pairs of one kind: unevaluated is not passed."""
        if not self.n_relevant or not self.n_irrelevant:
            return None
        return (self.irrelevant_changed / self.n_irrelevant <= IRRELEVANT_CHANGE_MAX
                and self.relevant_changed / self.n_relevant >= RELEVANT_CHANGE_MIN)

    def render(self) -> str:
        def f(i: Interval) -> str:
            return f"{i.point:.3f} [{i.low:.3f}, {i.high:.3f}]"

        verdict = {True: "PASS", False: "FAIL", None: "NOT EVALUATED"}[self.passes_v2]
        return "\n".join([
            "Reader sensitivity (counterfactual pairs, gold answer held fixed)",
            f"  relevant flips    changed {self.relevant_changed}/{self.n_relevant}  "
            f"{f(self.relevant_change_rate)}   need >= {RELEVANT_CHANGE_MIN}",
            f"  irrelevant flips  changed {self.irrelevant_changed}/{self.n_irrelevant}  "
            f"{f(self.irrelevant_change_rate)}   need <= {IRRELEVANT_CHANGE_MAX}",
            f"  V2 sensitivity: {verdict}",
        ])


def reader_sensitivity(
    instances: list[GoldInstance],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    answers: dict[str, str] | None = None,
    threshold: float = DEFAULT_NLI_THRESHOLD,
) -> Sensitivity:
    """Reader output change under each kind of flip, reading the *gold* answer by default."""
    s = Sensitivity(0, 0, 0, 0)
    for inst in instances:
        if not inst.persona_pairs:
            continue
        answer = (answers or {}).get(inst.instance_id) or inst.gold_scoped_answer
        if not answer:
            raise ValueError(f"{inst.instance_id}: no gold_scoped_answer to hold fixed")
        by_id = {p.persona_id: p for p in inst.personas}
        for pair in inst.persona_pairs:
            va = score_persona(inst, by_id[pair.base_persona_id], answer, reader, nli, threshold=threshold)
            vb = score_persona(inst, by_id[pair.flipped_persona_id], answer, reader, nli, threshold=threshold)
            changed = _outputs_differ(va, vb)
            if pair.flip == "relevant":
                s.n_relevant += 1
                s.relevant_changed += changed
            else:
                s.n_irrelevant += 1
                s.irrelevant_changed += changed
    return s


@dataclass(frozen=True)
class PairVerdict:
    instance_id: str
    flip: str
    consistent: bool
    determined: bool


def counterfactual_consistency(
    instances: list[GoldInstance],
    answers: dict[str, str],
    reader: Reader,
    nli: NLIScorerLike,
    *,
    threshold: float = DEFAULT_NLI_THRESHOLD,
) -> tuple[Interval | None, list[PairVerdict]]:
    """Share of persona pairs on which the system's answer leads the reader to the
    correct change or non-change (system-level; strict determination required)."""
    verdicts: list[PairVerdict] = []
    for inst in instances:
        if not inst.persona_pairs:
            continue
        if inst.instance_id not in answers:
            raise KeyError(f"no answer for {inst.instance_id}; a missing answer is not a wrong one")
        answer = answers[inst.instance_id]
        by_id = {p.persona_id: p for p in inst.personas}
        for pair in inst.persona_pairs:
            va = score_persona(inst, by_id[pair.base_persona_id], answer, reader, nli, threshold=threshold)
            vb = score_persona(inst, by_id[pair.flipped_persona_id], answer, reader, nli, threshold=threshold)
            # "Determined": the reader committed to an outcome backed by conditions
            # 1-3, right or wrong. WRONG still counts as determined; the pair is
            # then judged on whether the two outcomes move as gold says.
            determined = all(v.failed in (None, Failed.WRONG) for v in (va, vb))
            consistent = determined and (_outputs_differ(va, vb) is pair.gold_changes)
            verdicts.append(PairVerdict(inst.instance_id, pair.flip, consistent, determined))
    if not verdicts:
        return None, verdicts
    return wilson(sum(v.consistent for v in verdicts), len(verdicts)), verdicts
