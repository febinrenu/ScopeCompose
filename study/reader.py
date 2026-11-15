"""The answer generator, shared by every system.

Every system in the study differs in **which evidence reaches the generator**,
never in how the generator is asked. One prompt, one output contract, one
parser -- so a difference between two systems is a difference in their evidence.

The generator sees the evidence as numbered lines (one per HTML element of the
source page) and returns short answers, each with the numbers of the lines that
state a still-unresolved condition on it. Those numbers map back to the exact
element strings, which is the form the official metric scores conditions in.

The prompt was written against the tuning fold only (``study/splits.json``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from api_budget import BackendError, TruncatedResponseError
from api_budget.client import LLMClient, Tier
from study import cqa
from study.scoring import MAX_ANSWERS

SYSTEM = """\
You answer a person's question about UK government guidance.
You are given numbered lines copied from official guidance pages, the person's situation, and their question.

Reply with JSON only, in this form:
{"answers": [{"answer": "...", "evidence": [line numbers], "conditions": [line numbers]}]}

Rules:
1. Use only the numbered lines. If they do not contain the answer, reply {"answers": []}.
2. "answer": for a yes/no question write exactly "yes" or "no". Otherwise copy the shortest phrase from the lines that answers the question - an amount, a date, a period, a name, a form, a place.
3. "evidence": the numbers of the lines that state the answer.
4. "conditions": the lines stating a requirement that must be satisfied for the answer to be true and that the person's situation does not mention. Give [] if there is no such requirement. A line that only states the answer goes in "evidence", not here.
5. Give one answer. Give two only when the answer really differs depending on something the situation leaves open - then list each answer with its own conditions."""

SYSTEM_CLOSED_BOOK = """\
You answer a person's question about UK government guidance from your own knowledge. No documents are provided.

Reply with JSON only, in this form:
{"answers": [{"answer": "...", "conditions": []}]}

Rules:
1. "answer": for a yes/no question write exactly "yes" or "no". Otherwise give the shortest phrase that answers the question - an amount, a date, a period, a name, a form, a place.
2. If you cannot answer, reply {"answers": []}.
3. Give one answer. Give two only when the answer really differs depending on something the situation leaves open."""


#: Worked examples, sent as earlier turns of the conversation. All four are
#: questions from the tuning fold (train-1182, train-1409, train-1111,
#: train-792), on pages no evaluation question is about; their evidence is
#: trimmed to a few lines. They show the output contract's two halves: most
#: answers carry no condition, and a condition is cited by line number.
DEMOS: tuple[tuple[str, str], ...] = (
    ("Lines:\nPage: Company Tax Returns\n"
     "[1] Your company or association must file a Company Tax Return if you get a 'notice to deliver a Company Tax Return' from HM Revenue and Customs (HMRC).\n"
     "[2] You must still send a return if you make a loss or have no Corporation Tax to pay.\n"
     "[3] You do not send a Company Tax Return if you're self-employed as a sole trader or in a partnership - but you must send a Self Assessment return.\n\n"
     "Situation: we are a hospitality company. over the past year because of reduced visitor numbers we have recorded financial losses and have had to let some staff go.\n"
     "Question: do we have to file a return if we made a loss ?",
     '{"answers": [{"answer": "yes", "evidence": [2], "conditions": []}]}'),
    ("Lines:\nPage: Building a new home and VAT\nSection: What you need to include\n"
     "[1] VAT invoices must be valid and show the correct rate of VAT or they will not be accepted. For invoices not in sterling, convert the amounts into sterling.\n"
     "[2] ## How long it takes\n"
     "[3] HMRC will write to you about when you can expect your VAT refund if your claim is successful. They will usually write to you within 6 weeks.\n\n"
     "Situation: I have applied for a VAT refund and included all original invoices and regulation certificates. We added a conversion to our cats protection charity home in 2019.\n"
     "Question: How long will it take to get a response?",
     '{"answers": [{"answer": "within 6 weeks", "evidence": [3], "conditions": []}]}'),
    ("Lines:\nPage: Parent of a Child Student visa\nSection: Bringing family members with you\n"
     "[1] You can bring your other children with you if they also have or are applying for a Child Student visa.\n"
     "[2] You cannot bring other family members with you on this visa. They may be able to apply to come to the UK on a short-term visit visa.\n"
     "[3] Irish citizens do not need to apply for a visa or to the EU Settlement Scheme.\n\n"
     "Situation: My partner and I live abroad.We are planning to take our daughter to an independent school in the UK when she turns 5. We admire the UK National curriculum.\n"
     "Question: Can i bring other young daughter through this visa?",
     '{"answers": [{"answer": "yes", "evidence": [1], "conditions": [1]}]}'),
    ("Lines:\nPage: The National Minimum Wage and Living Wage\nSection: Employer checks\n"
     "[1] ## Keeping records\n"
     "[2] It's the employer's responsibility to keep records proving that they are paying the minimum wage. These records must be kept for at least 6 years if they:\n"
     "[3] - were created on or after 1 April 2021\n"
     "[4] - still had to be kept on 31 March 2021 under the previous rule that records must be kept for 3 years\n"
     "[5] The period records must be kept for starts from the last day of the pay reference period after the one they cover.\n\n"
     "Situation: I own and run a fruit farm in Kent, which currently employs four full-time staff and a further 16 workers on a part-time, seasonal basis. I am diligent about paying the correct wage rates and have decided to do my own accounts and record keeping to save money.\n"
     "Question: For how long am I required to keep records of my monthly payroll calculations?",
     '{"answers": [{"answer": "at least 6 years", "evidence": [2], "conditions": [3, 4]}]}'),
)
DEMO_QUESTION_IDS = frozenset({"train-1182", "train-1409", "train-1111", "train-792"})


def prompts_sha() -> str:
    blob = json.dumps({"system": SYSTEM, "closed_book": SYSTEM_CLOSED_BOOK, "demos": DEMOS}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Evidence -> numbered lines
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Evidence:
    """The lines shown to the generator, and the element behind each number."""

    text: str
    line_element: dict[int, str] = field(default_factory=dict)      # line number -> verbatim element string

    @property
    def n_lines(self) -> int:
        return len(self.line_element)


def render_evidence(chunks: Sequence[cqa.Chunk], *, drop_elements: frozenset[str] = frozenset()) -> Evidence:
    """Number every element of every chunk, in the order given.

    ``drop_elements`` removes individual elements (used only by the oracle
    ablation that deletes gold condition elements from otherwise intact chunks).
    """
    blocks: list[str] = []
    mapping: dict[int, str] = {}
    n = 0
    for c in chunks:
        head = f"Page: {c.title}" + (f"\nSection: {c.heading}" if c.heading and c.heading != c.title else "")
        lines = [head]
        for el in c.elements:
            if el in drop_elements:
                continue
            n += 1
            mapping[n] = el
            tag, text = cqa.split_element(el)
            mark = "- " if tag == "li" else ("## " if tag in ("h1", "h2", "h3", "h4") else "")
            lines.append(f"[{n}] {mark}{text}")
        if len(lines) > 1:
            blocks.append("\n".join(lines))
    return Evidence("\n\n".join(blocks), mapping)


def user_prompt(q: cqa.Question, evidence: Evidence | None) -> str:
    ask = f"Situation: {q.scenario}\nQuestion: {q.question}"
    return ask if evidence is None else f"Lines:\n{evidence.text}\n\n{ask}"


# --------------------------------------------------------------------------- #
# Output -> prediction
# --------------------------------------------------------------------------- #

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Parsed:
    answers: tuple[tuple[str, tuple[str, ...]], ...]
    ok: bool                      # the output was valid JSON in the contract's shape
    bad_citations: int = 0        # line numbers that named no line
    #: Lines cited as stating the answer (not scored by the official metric;
    #: kept so condition errors can be told apart from grounding errors).
    evidence: tuple[str, ...] = ()


def _lines(cited, mapping: dict[int, str]) -> tuple[list[str], int]:
    got, bad = [], 0
    for c in cited if isinstance(cited, list) else []:
        try:
            num = int(str(c).strip().strip("[]"))
        except ValueError:
            bad += 1
            continue
        if num in mapping:
            got.append(mapping[num])
        else:
            bad += 1
    return got, bad


def parse_output(text: str, evidence: Evidence | None) -> Parsed:
    """Model text -> ``((answer, (condition element strings)), ...)``.

    Lenient about the wrapping (code fences, prose around the object) and
    strict about the content: an answer with no text is dropped, a cited number
    that names no line is dropped and counted, never guessed.
    """
    raw = None
    for candidate in (text, *(m.group(0) for m in [_JSON_RE.search(text or "")] if m)):
        try:
            raw = json.loads(candidate)
            break
        except (TypeError, ValueError):
            continue
    items = raw.get("answers") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return Parsed((), ok=False)
    mapping = evidence.line_element if evidence is not None else {}
    out: list[tuple[str, tuple[str, ...]]] = []
    support: list[str] = []
    bad = 0
    for item in items[:MAX_ANSWERS]:
        if not isinstance(item, dict):
            continue
        answer = str(item.get("answer") or "").strip()
        if not answer:
            continue
        conds, b1 = _lines(item.get("conditions"), mapping)
        ev, b2 = _lines(item.get("evidence"), mapping)
        bad += b1 + b2
        support += ev
        out.append((answer, tuple(dict.fromkeys(conds))))
    return Parsed(tuple(out), ok=True, bad_citations=bad, evidence=tuple(dict.fromkeys(support)))


# --------------------------------------------------------------------------- #
# One call
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Generation:
    parsed: Parsed
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


def messages_for(q: cqa.Question, evidence: Evidence | None) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if evidence is not None:
        for demo_user, demo_reply in DEMOS:
            msgs += [{"role": "user", "content": demo_user}, {"role": "assistant", "content": demo_reply}]
    msgs.append({"role": "user", "content": user_prompt(q, evidence)})
    return msgs


def generate(client: LLMClient, tier: Tier, q: cqa.Question, evidence: Evidence | None, *, step: str) -> Generation:
    """One answer. An output cut off at the token limit, or aborted by the server because the
    model fell into a repetition loop, is an answer the system failed to give: it is recorded
    as such and scored, not retried or dropped. Both are deterministic at temperature 0."""
    import settings

    system = SYSTEM if evidence is not None else SYSTEM_CLOSED_BOOK
    try:
        res = client.complete(messages=messages_for(q, evidence), system=system, tier=tier, step=step,
                              response_format={"type": "json_object"})
    except TruncatedResponseError as exc:
        return Generation(Parsed((), ok=False), f"[truncated] {exc}"[:300],
                          settings.models().tier(tier.value).model or "", 0, 0, 0.0, False)
    except BackendError as exc:
        if "token repeat limit" not in str(exc):
            raise
        return Generation(Parsed((), ok=False), f"[degenerate] {exc}"[:300],
                          settings.models().tier(tier.value).model or "", 0, 0, 0.0, False)
    return Generation(parse_output(res.text, evidence), res.text, res.model, res.prompt_tokens,
                      res.completion_tokens, res.latency_s, res.cached)
