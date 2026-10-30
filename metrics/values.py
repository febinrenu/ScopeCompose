"""Outcome values and modality: the deterministic half of case-complete accuracy.

Case-complete accuracy (CCA, proposal v8 section 7.2) asks whether a reader can
determine, from a system's answer alone, the outcome that holds for a given
person. Two questions in that sentence are answered here, without a model:

1. **Do two outcome statements mean the same thing?** (:func:`values_match`)
   Gold outcomes are terse annotations ("no fee applies"); a reader's outcome is
   free text ("the charge is waived for you"). ``branch_match.outcomes_match``
   already vetoes numeric and polarity disagreement; this adds what it lacks --
   spelled-out numbers, citation stripping, and paraphrase classes (waived =
   no fee, not permitted = prohibited) so the common equivalences do not need a
   model.

2. **Does a span actually determine an outcome?** (:func:`detect_modality`)
   "Premium customers *may receive* fee waivers" does not tell a premium
   customer they pay nothing. Counting it as a determination is exactly the
   gap-filling CCA-strict exists to block. Hedged modality never determines.

**Why modal "may" needs care.** In policy prose "may" is two different words.
"Applicants *may apply* for leave" is a permission -- the outcome is stated. "A
waiver *may be granted*" is epistemic -- whether the outcome holds is left open.
Treating every "may" as a hedge would mark a correct permission as
undetermined, so "may" is classified by what follows it: a verb of the person's
own action is a permission; a verb about whether a benefit, charge or status
*occurs* ("receive", "be charged", "qualify") is a hedge. "May not" followed by
an action is a prohibition; "may not be" is a hedge. The rule is deliberately
conservative about calling things determined: when unsure it hedges, because a
false determination inflates CCA while a false hedge only costs recall -- and
recall is bounded and audited separately (gate V2).

"Normally", "usually" and "generally" are **not** hedges here. Gold answers use
them to mark the default branch ("Transfers normally incur a 3% fee. However,
the fee is waived for premium cardholders"), and an answer that states a default
and its exception explicitly has determined the outcome for both people.
"""

from __future__ import annotations

import re

from contract.frames import Modality
from metrics.branch_match import outcomes_match

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

#: "(per p0)", "[p1]", "(p0, p2)", "(source: p1)" -- provenance, not content.
_CITATION_RE = re.compile(
    r"\s*[\(\[]\s*(?:per|source|sources|see|from)?:?\s*p\d+(?:\s*[,;&]\s*(?:and\s+)?p\d+)*\s*[\)\]]",
    re.IGNORECASE,
)

_CARDINALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}

_CARDINAL_RE = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[\s-]+(" + "|".join(k for k in _CARDINALS if 1 <= _CARDINALS[k] <= 9) + r"))?"
    r"|(" + "|".join(_CARDINALS) + r"))\b"
)


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text or "")


def _cardinals_to_digits(text: str) -> str:
    """"twenty-eight days" -> "28 days"; "two years" -> "2 years"."""

    def sub(m: re.Match[str]) -> str:
        tens, unit, plain = m.group(1), m.group(2), m.group(3)
        if plain is not None:
            return str(_CARDINALS[plain])
        return str(_TENS[tens] + (_CARDINALS[unit] if unit else 0))

    return _CARDINAL_RE.sub(sub, text)


def _normalise_figures(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)            # 1,270 -> 1270
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:per\s*cent|percent|pct)\b", r"\1%", text)
    text = re.sub(r"[£$€]\s*(\d)", r"\1 ", text)                # currency sign is not a figure
    text = re.sub(r"\b\d+\.\d+\b", lambda m: _trim_decimal(m.group(0)), text)  # 2.50 -> 2.5, 28.0 -> 28
    return text


def _trim_decimal(s: str) -> str:
    whole, frac = s.split(".")
    frac = frac.rstrip("0")
    return f"{whole}.{frac}" if frac else whole


#: Words that say what a figure counts. A figure is compared only against
#: figures counting the same thing: 28 *days* against 14 *days*, an *amount*
#: against an *amount*.
_UNIT_WORDS = {
    "day": "day", "days": "day", "week": "week", "weeks": "week",
    "month": "month", "months": "month", "year": "year", "years": "year",
    "hour": "hour", "hours": "hour", "minute": "minute", "minutes": "minute",
}
_FIGURE_RE = re.compile(r"(\d+(?:\.\d+)?)(\s*%)?")


def figures(text: str) -> dict[str, set[str]]:
    """The figures in an outcome, grouped by what they count.

    ``"funds of 1270 held for 28 consecutive days"`` ->
    ``{"amount": {"1270"}, "day": {"28"}}``. The unit is the first unit word
    within two words after the figure, so "28 consecutive days" is read as days.
    """
    t = canonicalise_outcome(text)
    out: dict[str, set[str]] = {}
    for m in _FIGURE_RE.finditer(t):
        value = m.group(1)
        if m.group(2):
            unit = "%"
        else:
            # Stop at the next figure: in "1270 for 28 days" the unit "days"
            # belongs to 28, and reading it as 1270's would hide the conflict.
            tail = re.findall(r"[a-z]+", re.split(r"\d", t[m.end(): m.end() + 40])[0])[:2]
            unit = next((_UNIT_WORDS[w] for w in tail if w in _UNIT_WORDS), "amount")
        out.setdefault(unit, set()).add(value)
    return out


def figures_conflict(a: str, b: str) -> bool:
    """Whether two outcomes quote different figures for the same thing.

    Stricter than ``branch_match.numbers_conflict``, which treats *any* shared
    figure as agreement: "1270 for 28 days" against "1270 for 14 days" share
    1270 and so passed. For case-complete accuracy that is the dangerous
    direction -- a reader that gets the period wrong would be marked right.
    Here a unit with figures on both sides and no figure in common is a
    conflict, whatever else the two share.
    """
    fa, fb = figures(a), figures(b)
    return any(fa[u] and fb[u] and not (fa[u] & fb[u]) for u in fa.keys() & fb.keys())


#: Paraphrase classes, applied in order. Each maps a family of phrasings to one
#: canonical clause so that deterministic matching sees them as equal. Negative
#: phrasings come before positive ones: "not eligible" must be rewritten before
#: "eligible" gets a chance to match inside it.
_PARAPHRASES: list[tuple[re.Pattern[str], str]] = [
    # -- fees and charges -------------------------------------------------- #
    (re.compile(r"\b(?:fee|charge|cost)s?\s+(?:is|are|will be|being|gets?)?\s*(?:waived|removed|dropped|not applied|not charged|not payable)\b"), "no fee applies"),
    (re.compile(r"\b(?:waived|exempt(?:ed)?)\s+(?:from\s+)?(?:the\s+|any\s+)?(?:fee|charge|cost)s?\b"), "no fee applies"),
    (re.compile(r"\b(?:no|zero|nil)\s+(?:fee|charge|cost)s?\b(?:\s+(?:applies|applied|is charged|are charged|payable|is payable|to pay))?"), "no fee applies"),
    (re.compile(r"\b(?:free of charge|fee[- ]free|charge[- ]free|without (?:a |any )?(?:fee|charge)|isn't charged|aren't charged|pays? nothing|nothing to pay)\b"), "no fee applies"),
    (re.compile(r"\b(?:not|never)\s+(?:be\s+)?charged\b"), "no fee applies"),
    (re.compile(r"\b(?:not|never)\s+(?:pay|incur|owe)\s+(?:a\s+|any\s+|the\s+)?(?:fee|charge|cost)s?\b"), "no fee applies"),
    # -- eligibility ------------------------------------------------------- #
    (re.compile(r"\b(?:ineligible|not eligible|not entitled|(?:do|does|will|would) not qualify|(?:don't|doesn't|won't) qualify|cannot qualify|can't qualify|not qualif\w+)\b"), "is ineligible"),
    (re.compile(r"\b(?:eligible|entitled|qualif(?:y|ies|ied))\b"), "is eligible"),
    # -- obligation --------------------------------------------------------- #
    (re.compile(r"\b(?:not required|need not|no need to|(?:do|does) not need to|(?:don't|doesn't) need to|not mandatory|optional|is not obliged|are not obliged)\b"), "is not required"),
    (re.compile(r"\b(?:required|must|mandatory|obliged|obligatory|(?:have|has|need) to)\b"), "is required"),
    # -- permission -------------------------------------------------------- #
    (re.compile(r"\b(?:not permitted|not allowed|not able to|prohibited|forbidden|barred|banned|disallowed|may not|must not|cannot|can't|can not|are not allowed to|is not allowed to|no right to)\b"), "is prohibited"),
    (re.compile(r"\b(?:permitted|allowed|may|can|able to|free to)\b"), "is permitted"),
]


#: Who the outcome is about. "The applicant may apply" and "you may apply" are
#: one outcome; leaving the subject in made them disagree on a token that
#: carries no content. Removed only from the matching form.
_SUBJECT_RE = re.compile(
    r"\b(?:the\s+)?(?:you|your|yours|yourself|i|me|my|they|them|their|"
    r"applicants?|cardholders?|customers?|account holders?|holders?|users?|clients?|"
    r"claimants?|students?|residents?|people|persons?)\b")


def canonicalise_outcome(text: str) -> str:
    """The comparison form of an outcome statement.

    Lowercased, citation-free, figures normalised, spelled-out numbers turned
    into digits, and the common paraphrase classes collapsed. This is a
    *matching* form: it is never shown to a user and never fed back to a model.
    """
    t = strip_citations(text).lower().strip()
    t = _cardinals_to_digits(t)
    t = _normalise_figures(t)
    for pattern, repl in _PARAPHRASES:
        t = pattern.sub(repl, t)
    t = _SUBJECT_RE.sub(" ", t)
    return " ".join(t.replace("-", " ").split())


def values_match(gold: str, predicted: str, *, nli=None) -> bool:
    """Whether a reader's outcome says what the gold outcome says.

    Deterministic by default (no model, no provider, same answer forever), so a
    CCA number can be regenerated from the cache. ``outcomes_match`` supplies the
    numeric and polarity vetoes and the lexical thresholds; canonicalisation
    first removes the wording differences that are not differences of meaning.
    """
    if not gold.strip() or not predicted.strip():
        return False
    if figures_conflict(gold, predicted):
        return False
    return outcomes_match(canonicalise_outcome(gold), canonicalise_outcome(predicted), nli=nli)


# --------------------------------------------------------------------------- #
# Modality
# --------------------------------------------------------------------------- #

#: Cues that leave the outcome open whatever surrounds them.
_ALWAYS_HEDGE = re.compile(
    r"\b(?:might|could|perhaps|possibly|potentially|probably|likely|unlikely|"
    r"sometimes|occasionally|"
    r"in (?:some|certain|a few|limited|exceptional) (?:cases|circumstances|situations|instances)|"
    r"at (?:its|our|their|the) (?:sole )?discretion|at the discretion of|"
    r"subject to (?:approval|status|availability|conditions|review|eligibility|terms)|"
    r"depend(?:s|ing)? on|it depends|"
    r"not (?:always|necessarily|guaranteed)|"
    r"may or may not|"
    r"(?:is|are) not (?:certain|clear)|unclear|"
    r"(?:may|might|can sometimes|could) be able to|"
    r"appears? to|seems? to|reportedly|supposedly|"
    r"can be (?:waived|reduced|refunded|removed|granted) (?:in|on|at|for)|"
    r"(?:may|might) (?:also )?(?:be|get|receive|qualify|incur|owe|face|see|find))\b",
    re.IGNORECASE,
)

#: Verbs after "may" that make it epistemic: whether a benefit, charge or status
#: *occurs*, not what the person is allowed to do.
_EPISTEMIC_AFTER_MAY = {
    "be", "get", "receive", "qualify", "incur", "owe", "face", "see", "find",
    "benefit", "obtain", "need", "have", "also", "still", "then", "sometimes",
    "usually", "occasionally", "apply",  # overridden below: 'apply' is a permission
}
_EPISTEMIC_AFTER_MAY.discard("apply")

_MAY_RE = re.compile(r"\bmay(\s+not)?(?:\s+(?:also|still|then))?\s+(\w+)", re.IGNORECASE)

#: Nouns that, directly before "may", make it epistemic whatever verb follows:
#: "a fee *may apply*" is about whether the fee occurs, while "applicants *may
#: apply*" is a permission -- the verb is the same word.
_EPISTEMIC_SUBJECTS = {
    "fee", "fees", "charge", "charges", "cost", "costs", "penalty", "penalties",
    "surcharge", "surcharges", "restriction", "restrictions", "condition",
    "conditions", "rule", "rules", "requirement", "requirements", "exception",
    "exceptions", "limit", "limits", "waiver", "waivers", "discount", "discounts",
    "this", "that", "these", "those", "it", "some", "others", "other",
}

_PROHIBITION_RE = re.compile(
    r"\b(?:must not|cannot|can't|can not|not (?:permitted|allowed|able)|prohibited|"
    r"forbidden|barred|banned|disallowed|no right to)\b", re.IGNORECASE)
_OBLIGATION_RE = re.compile(
    r"\b(?:must|shall|required to|have to|has to|need(?:s)? to|obliged|mandatory)\b", re.IGNORECASE)
_PERMISSION_RE = re.compile(
    r"\b(?:can|allowed to|permitted to|permitted|entitled to|is able to|are able to)\b", re.IGNORECASE)


def _may_modality(text: str) -> Modality | None:
    """Classify the *first* "may" by what follows it; ``None`` if there is none."""
    m = _MAY_RE.search(text)
    if not m:
        return None
    negated, nxt = bool(m.group(1)), m.group(2).lower()
    before = re.findall(r"[a-z']+", text[: m.start()].lower())[-2:]
    if not negated and any(w in _EPISTEMIC_SUBJECTS for w in before):
        return Modality.HEDGED
    if negated:
        # "may not be ..." leaves it open; "may not accept ..." forbids.
        return Modality.HEDGED if nxt in ("be", "always", "necessarily") else Modality.PROHIBITION
    if nxt in _EPISTEMIC_AFTER_MAY:
        return Modality.HEDGED
    return Modality.PERMISSION


def detect_modality(span: str) -> Modality:
    """Whether and how a span commits to an outcome.

    Order matters: hedging is checked first and wins, because one hedge anywhere
    in the span is enough to leave the outcome open -- "premium customers pay
    nothing, although in some cases a fee may still apply" does not determine
    the outcome for a premium customer.
    """
    text = strip_citations(span or "")
    if not text.strip():
        return Modality.HEDGED  # nothing was said, so nothing is determined

    if _ALWAYS_HEDGE.search(text):
        return Modality.HEDGED

    may = _may_modality(text)
    if may is Modality.HEDGED:
        return Modality.HEDGED

    if may is Modality.PROHIBITION or _PROHIBITION_RE.search(text):
        return Modality.PROHIBITION
    if _OBLIGATION_RE.search(text):
        return Modality.OBLIGATION
    if may is Modality.PERMISSION or _PERMISSION_RE.search(text):
        return Modality.PERMISSION
    return Modality.STATEMENT


def is_hedged(span: str) -> bool:
    return detect_modality(span) is Modality.HEDGED


def determines_outcome(span: str) -> bool:
    """Condition 3 of CCA-strict: the span is not hedged."""
    return not is_hedged(span)
