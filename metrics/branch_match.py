"""Deterministic alignment of predicted branches against gold branches.

The preservation metrics need to decide, for each gold branch, whether a
system's output preserved it, suppressed it, or distorted it. Proposal §6.1
specifies an LLM judge for that, validated against human labels before it is
trusted.

This is the deterministic instrument that sits underneath. It exists for three
reasons, and it does not replace the judge:

1. **The decisive experiment has to be runnable now.** The structured
   long-context baseline emits branches in the gold schema, so alignment is a
   structural comparison, not a semantic one, and structural comparison needs
   no model.
2. **It is reproducible in a way an LLM judge is not.** Same input, same
   output, forever, with no temperature and no provider. That makes it the
   right instrument for regression-checking the pipeline between runs.
3. **It gives the judge something to be validated against.** §6.1 validates
   the judge against human labels; a deterministic baseline also shows how
   much of the judge's behaviour is doing work that string matching already
   does.

**Where it is weaker than the judge, and why that is stated rather than
hidden.** Matching is lexical. A branch phrased *"no fee applies"* against gold
*"the charge is waived"* is the same outcome and this will score it as
distorted. So the deterministic scorer gives a **lower bound** on preservation
and an **upper bound** on distortion. Report it as that, and use the judge for
the headline once it is validated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from contract.gold import Applicability, Branch
from scope.attributes import SetRelation, compare

_WORD_RE = re.compile(r"[a-z0-9']+")

#: Note what is NOT here: "no" and "not". They were stopwords in the first
#: version, and dropping them made an outcome and its negation identical --
#: "no fee applies" against "a fee applies" scored a Jaccard of 1.00. Negation
#: is the entire content of an outcome, never noise.
_STOP = {"the", "a", "an", "is", "are", "of", "to", "for", "in", "on", "and",
         "or", "be", "that", "this", "it", "with", "any", "all"}

#: Syntactic negation markers. Deliberately syntactic only -- "waived",
#: "exempt" and "free" negate semantically, but they are ordinary content words
#: whose presence does not flip a clause, and treating them as negators would
#: make "the fee is waived" conflict with "no fee applies", which is the same
#: outcome. Semantic opposition is the NLI signal's job.
_NEGATORS = {"no", "not", "never", "cannot", "neither", "nor", "without",
             "non", "nothing", "none"}

#: Outcome similarity above which two branches are treated as saying the same
#: thing. Tuned to be forgiving on wording and strict on content: "a 3% fee
#: applies" against "a 5% fee applies" shares most tokens, so numeric
#: disagreement is checked separately rather than left to the overlap score.
OUTCOME_MATCH = 0.45

#: Fraction of the gold outcome's words that must appear in the prediction.
#: Higher than the Jaccard threshold because containment is a weaker claim --
#: it ignores everything the prediction adds.
OUTCOME_CONTAINMENT = 0.6

#: Mean bidirectional entailment above which two outcomes are the same, when
#: an NLI scorer is supplied.
NLI_ENTAILMENT_MATCH = 0.5

#: Applicability similarity for the same purpose.
APPLICABILITY_MATCH = 0.35

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%?")


class BranchJudgement(str, Enum):
    """How one gold branch fared in a system's output."""

    PRESERVED = "preserved"
    SUPPRESSED = "suppressed"
    DISTORTED = "distorted"


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP}


def similarity(a: str, b: str) -> float:
    """Jaccard over content words. Deterministic and cheap."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def numbers_conflict(a: str, b: str) -> bool:
    """Whether two outcome strings quote different figures.

    Checked separately from token overlap because "a 3% fee applies" and
    "a 5% fee applies" share almost every word. Lexical similarity alone would
    call them the same outcome, which is exactly the distortion the metric
    exists to catch.
    """
    na = {n.strip() for n in _NUM_RE.findall(a or "")}
    nb = {n.strip() for n in _NUM_RE.findall(b or "")}
    return bool(na and nb and not (na & nb))


#: Comparative quantifier phrases whose "no"/"not" bounds a quantity rather
#: than negating a clause. "Employment is permitted for no more than 20 hours"
#: is a permission with a limit, not a prohibition -- but the bare negator made
#: it read as the opposite polarity to "employment is permitted", so the branch
#: it actually stated was scored as dropped.
_QUANTIFIER_RE = re.compile(
    r"\b(?:no|not)\s+(?:more|less|fewer|later|earlier|sooner|longer)\s+than\b",
    re.IGNORECASE)


def _strip_quantifiers(text: str) -> str:
    return _QUANTIFIER_RE.sub(" ", text or "")


def polarity_conflict(a: str, b: str) -> bool:
    """Whether one outcome is negated and the other is not, over the same content.

    The companion to :func:`numbers_conflict`, and it exists for the same
    reason: two outcomes can share almost every word and mean opposite things.
    "No fee applies" and "a fee applies" differ by one token that token-overlap
    scoring treats as negligible -- they scored a Jaccard of 1.00 while "no" was
    a stopword, so a system emitting the exact opposite of a gold branch was
    scored as having preserved it.

    Parity, not presence: "leave does not lapse" against "leave never lapses"
    is two negations agreeing, not a conflict. Counting markers and comparing
    parity handles that; checking "does either contain a negator" does not.

    The content-overlap guard keeps this narrow. Without it, every negated
    outcome would conflict with every unrelated positive one, and unrelated
    outcomes are already handled by the similarity thresholds -- this is only
    for the case where the two say the same thing with opposite sign.
    """
    a, b = _strip_quantifiers(a), _strip_quantifiers(b)
    ta, tb = _tokens(a), _tokens(b)
    parity_a = len(ta & _NEGATORS) % 2
    parity_b = len(tb & _NEGATORS) % 2
    if parity_a == parity_b:
        return False

    core_a, core_b = ta - _NEGATORS, tb - _NEGATORS
    if not core_a or not core_b:
        return False

    # Asymmetric, for the same reason `containment` is. One side is typically a
    # terse gold outcome and the other a clause of a generated answer carrying
    # a citation and a figure; symmetric Jaccard is dragged below the threshold
    # by those extra tokens, and "no fee applies" against "a 3% fee applies
    # (per p0)" then slips through as non-contradictory.
    overlap = len(core_a & core_b) / min(len(core_a), len(core_b))
    return overlap >= 0.6


def containment(gold: str, pred: str) -> float:
    """Fraction of the GOLD outcome's content words present in the prediction.

    Asymmetric on purpose, and this is the fix for a real mis-scoring. Gold
    outcomes are terse annotations ("a 2.50 per-transaction charge applies");
    predictions are verbose prose ("You will be charged a 2.50 fee per cash
    withdrawal plus any fee charged by the operator"). Symmetric Jaccard
    punishes the prediction for its extra words and scored that pair at 0.19 --
    marking two identical outcomes as distorted.

    What matters is whether the prediction SAYS what gold says, not whether it
    says only that.
    """
    tg, tp = _tokens(gold), _tokens(pred)
    if not tg:
        return 0.0
    return len(tg & tp) / len(tg)


def outcomes_match(gold: str, pred: str, *, nli=None) -> bool:
    """Whether two outcome statements say the same thing.

    Three signals, cheapest first:

    1. **Numeric or polarity disagreement is disqualifying.** "a 3% fee" and
       "a 5% fee" share almost every word; so do "no fee applies" and "a fee
       applies". Both are checked before any similarity score and override
       both of them.
    2. **Lexical**, via symmetric Jaccard OR asymmetric containment. Either
       clearing its threshold is enough: containment catches the verbose-
       prediction case, Jaccard catches the terse-prediction case.
    3. **Entailment**, when an NLI scorer is supplied. Paraphrase defeats
       lexical matching entirely -- "no fee applies" against "the charge is
       waived" shares nothing -- and this is the only signal that catches it.
       Optional because it costs a model load; supply it for reported numbers.
    """
    if numbers_conflict(gold, pred) or polarity_conflict(gold, pred):
        return False

    if similarity(gold, pred) >= OUTCOME_MATCH:
        return True
    if containment(gold, pred) >= OUTCOME_CONTAINMENT:
        return True

    if nli is not None:
        fwd, rev = nli.score_batch([(pred, gold), (gold, pred)])
        return (fwd.entailment + rev.entailment) / 2 >= NLI_ENTAILMENT_MATCH

    return False


def applicabilities_match(gold: Applicability, pred: Applicability) -> bool:
    """Same scope?

    Typed attributes decide it exactly where both sides have them -- EQUAL or
    SUBSET counts as a match, since a system that names a slightly narrower
    scope has still identified the branch. Otherwise fall back to descriptor
    similarity.
    """
    if gold.is_default or pred.is_default:
        return gold.is_default == pred.is_default

    rel = compare(pred, gold)
    if rel in (SetRelation.EQUAL, SetRelation.SUBSET):
        return True
    if rel in (SetRelation.DISJOINT, SetRelation.OVERLAPPING, SetRelation.SUPERSET):
        # A superset is not a match: claiming the fee is waived for everyone
        # when it is waived for premium holders is a different branch.
        return similarity(gold.descriptor, pred.descriptor) >= 0.8

    return similarity(gold.descriptor, pred.descriptor) >= APPLICABILITY_MATCH


@dataclass(frozen=True)
class BranchAlignment:
    """One gold branch and what the system did with it."""

    gold: Branch
    matched: Branch | None
    judgement: BranchJudgement
    outcome_similarity: float
    reason: str


@dataclass
class AlignmentResult:
    alignments: list[BranchAlignment]
    unmatched_predictions: list[Branch]
    """Predicted branches with no gold counterpart. Candidates for
    hallucination -- though a prediction grounded in a real passage that the
    annotator simply did not record is a benchmark gap, not a model error, so
    grounding is checked before counting."""

    @property
    def preserved(self) -> int:
        return sum(1 for a in self.alignments if a.judgement is BranchJudgement.PRESERVED)

    @property
    def suppressed(self) -> int:
        return sum(1 for a in self.alignments if a.judgement is BranchJudgement.SUPPRESSED)

    @property
    def distorted(self) -> int:
        return sum(1 for a in self.alignments if a.judgement is BranchJudgement.DISTORTED)

    @property
    def n_gold(self) -> int:
        return len(self.alignments)


def align(
    gold_branches: list[Branch],
    predicted_branches: list[Branch],
    *,
    nli=None,
) -> AlignmentResult:
    """Greedily align predicted branches to gold branches.

    Greedy on outcome similarity, best first, each prediction used once.
    Optimal assignment would need the Hungarian algorithm; with the two-to-four
    branches an instance actually carries, greedy and optimal coincide almost
    always, and greedy is inspectable.
    """
    available = list(predicted_branches)
    alignments: list[BranchAlignment] = []

    # Match the default branch first where both sides have one: it anchors the
    # instance, and letting an exception claim it would cascade.
    ordered_gold = sorted(gold_branches, key=lambda b: not b.is_default)

    for gold in ordered_gold:
        best: Branch | None = None
        best_sim = -1.0
        for cand in available:
            if gold.is_default != cand.is_default:
                continue
            # Rank candidates on the more generous of the two lexical
            # signals, so a verbose prediction is not passed over for a
            # shorter but wrong one.
            sim = max(similarity(gold.outcome, cand.outcome),
                      containment(gold.outcome, cand.outcome))
            if sim > best_sim:
                best, best_sim = cand, sim

        if best is None:
            alignments.append(BranchAlignment(
                gold, None, BranchJudgement.SUPPRESSED, 0.0,
                "no predicted branch of the same kind (default vs exception)"))
            continue

        available.remove(best)

        if not outcomes_match(gold.outcome, best.outcome, nli=nli):
            why = ("outcome figures disagree"
                   if numbers_conflict(gold.outcome, best.outcome)
                   else f"outcome similarity {best_sim:.2f} below {OUTCOME_MATCH}")
            alignments.append(BranchAlignment(
                gold, best, BranchJudgement.DISTORTED, best_sim, why))
            continue

        if not applicabilities_match(gold.applicability, best.applicability):
            alignments.append(BranchAlignment(
                gold, best, BranchJudgement.DISTORTED, best_sim,
                "right outcome, wrong scope"))
            continue

        alignments.append(BranchAlignment(
            gold, best, BranchJudgement.PRESERVED, best_sim, "outcome and scope match"))

    return AlignmentResult(alignments=alignments, unmatched_predictions=available)


def is_grounded(branch: Branch, passage_ids: set[str]) -> bool:
    """Whether a predicted branch cites a passage that exists.

    An ungrounded branch is the fabrication the Hallucinated-Condition Rate
    counts. A branch citing a real passage but absent from gold is a different
    thing -- possibly an annotation gap -- and is not counted as hallucinated.
    """
    return branch.supporting_passage is not None and branch.supporting_passage in passage_ids
