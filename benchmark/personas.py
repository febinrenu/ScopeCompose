"""Personas and counterfactual persona pairs (proposal v8 sections 6.5, 7.2).

Case-complete accuracy is only as good as its personas. Each region of an
instance needs a short first-person description of someone who falls in it, and
each conditional instance needs two *minimal pairs*:

* **relevant flip** -- the default persona with only the exception's condition
  made true. The gold outcome changes; a sound reader's output must change too.
* **irrelevant flip** -- the default persona plus one fact no branch mentions (a
  city). The gold outcome does not change; neither may the reader's.

A model (the EVAL tier, never an evaluated one) *proposes*; a person verifies.
What this module adds is the part a person should not have to do by eye: the
structural checks that make a pair a genuine minimal pair at all.

* statements are first-person, plain, and free of attribute identifiers
  (``card_tier``, ``sig:event``) -- a persona that leaks the schema lets a reader
  match on a field name instead of reading the answer;
* an exception persona must be a *minimal edit* of the default (token overlap
  above a floor), or the "only the relevant attribute changed" claim is false;
* the irrelevant fact must not appear in any branch condition, or it is not
  irrelevant;
* ``gold_changes`` is never asserted by the model. It is derived from the gold
  branches by the contract validator, so a pair whose two personas land in
  branches with the same outcome is *refused*, not trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from api_budget import Tier, get_client
from api_budget.client import LLMClient
from contract.gold import GoldInstance, Persona, PersonaPair

STEP = "personas_generate"
STEP_REPAIR = "personas_repair"

MIN_WORDS, MAX_WORDS = 6, 60
#: Token-overlap floor for "this is a minimal edit of the default persona".
MIN_EDIT_OVERLAP = 0.35
#: The irrelevant variant must keep at least this share of the default's words.
MIN_KEEP_OVERLAP = 0.70

_STOP = {"i", "a", "an", "the", "and", "or", "my", "me", "am", "is", "are", "of", "to", "in", "on", "for", "with",
         "have", "has", "at", "it", "that", "this", "be", "as", "by", "from", "was", "been"}
_SCHEMA_LEAK = re.compile(r"[a-z]+_[a-z]+|sig:|\b(?:is_default|branch|applicability)\b", re.IGNORECASE)

SYSTEM = """\
You write short first-person descriptions of people for a test. Return ONLY JSON:
{"default": "<one sentence>",
 "exceptions": {"<branch id>": "<one sentence>", ...},
 "irrelevant": {"attribute": "<name of an unrelated fact>", "statement": "<one sentence>"}}

Rules:
- Each sentence is first person ("I ..."), present tense, plain English, 8 to 30 words.
- "default": a person in the situation the question asks about who meets NONE of the exception
  conditions. Describe an ordinary circumstance; do not mention any exception condition.
- "exceptions": for each branch id, the SAME sentence as the default with ONLY that branch's condition
  made true. Change as little as possible: same wording, one fact swapped or added.
- "irrelevant": the default sentence plus ONE extra fact that none of the conditions mention
  (for example the town they live in). Name that fact in "attribute".
- Paraphrase the wording of the conditions. Never use field names, underscores or the word "branch"."""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


@dataclass
class PersonaProposal:
    instance_id: str
    default: str = ""
    exceptions: dict[str, str] = field(default_factory=dict)
    irrelevant_attribute: str = ""
    irrelevant: str = ""
    problems: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems


def user_prompt(inst: GoldInstance) -> str:
    lines = [f"Question: {inst.query}", "", "Branches (the exception conditions):"]
    for b in inst.exception_branches:
        lines.append(f"- id {b.branch_id}: condition: {b.condition}; outcome: {b.outcome}")
    d = inst.default_branch
    if d is not None:
        lines.append(f"\nThe default (everyone else): outcome: {d.outcome}")
    return "\n".join(lines)


def _parse(text: str, inst: GoldInstance) -> PersonaProposal:
    prop = PersonaProposal(inst.instance_id, raw=text)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        from detection.stage2 import _salvage_json

        data = _salvage_json(text)
    if not isinstance(data, dict):
        prop.problems.append("reply was not a JSON object")
        return prop
    prop.default = str(data.get("default") or "").strip()
    exc = data.get("exceptions")
    prop.exceptions = {str(k): str(v).strip() for k, v in (exc.items() if isinstance(exc, dict) else [])
                       if v is not None}
    irr = data.get("irrelevant") if isinstance(data.get("irrelevant"), dict) else {}
    prop.irrelevant_attribute = str(irr.get("attribute") or "").strip()
    prop.irrelevant = str(irr.get("statement") or "").strip()
    return prop


def validate(inst: GoldInstance, prop: PersonaProposal) -> list[str]:
    """Structural problems with a proposal. Empty means it is a well-formed set of minimal pairs."""
    problems: list[str] = []
    stmts = {"default": prop.default, "irrelevant": prop.irrelevant,
             **{f"exception {k}": v for k, v in prop.exceptions.items()}}
    for name, s in stmts.items():
        n = len(_words(s))
        if not s:
            problems.append(f"{name}: empty")
            continue
        if not MIN_WORDS <= n <= MAX_WORDS:
            problems.append(f"{name}: {n} words (need {MIN_WORDS}-{MAX_WORDS})")
        if _SCHEMA_LEAK.search(s):
            problems.append(f"{name}: leaks a field name or identifier")
        if not re.match(r"\s*(?:i|my|as)\b", s, re.IGNORECASE):
            problems.append(f"{name}: not first person")

    want = {b.branch_id for b in inst.exception_branches}
    if set(prop.exceptions) != want:
        problems.append(f"exceptions must cover exactly {sorted(want)}, got {sorted(prop.exceptions)}")
    if not prop.irrelevant_attribute:
        problems.append("irrelevant: the unrelated attribute is not named")
    if problems:
        return problems

    base = _content(prop.default)
    for bid, s in prop.exceptions.items():
        if s.strip().lower() == prop.default.strip().lower():
            problems.append(f"exception {bid}: identical to the default (no condition was made true)")
        elif _jaccard(base, _content(s)) < MIN_EDIT_OVERLAP:
            problems.append(f"exception {bid}: not a minimal edit of the default (overlap "
                            f"{_jaccard(base, _content(s)):.2f} < {MIN_EDIT_OVERLAP})")
    keep = len(base & _content(prop.irrelevant)) / len(base) if base else 0.0
    if keep < MIN_KEEP_OVERLAP:
        problems.append(f"irrelevant: keeps only {keep:.2f} of the default's content (need {MIN_KEEP_OVERLAP})")
    extra = _content(prop.irrelevant) - base
    cond_words = set().union(*(_content(b.condition or "") for b in inst.exception_branches)) if want else set()
    if extra & cond_words:
        problems.append(f"irrelevant: the added fact overlaps a branch condition ({sorted(extra & cond_words)})")
    if prop.irrelevant.strip().lower() == prop.default.strip().lower():
        problems.append("irrelevant: identical to the default")
    return problems


def _repair_prompt(prop: PersonaProposal) -> str:
    return ("These personas had problems:\n" + "\n".join(f"- {p}" for p in prop.problems)
            + "\n\nReturn the full corrected JSON in the same schema.")


def generate_personas(inst: GoldInstance, *, client: LLMClient | None = None, tier: Tier = Tier.EVAL
                      ) -> PersonaProposal:
    """Propose personas with the EVAL model, validate, and re-ask once on problems."""
    if not inst.exception_branches or inst.default_branch is None:
        raise ValueError(f"{inst.instance_id}: personas need a default branch and at least one exception")
    c = client or get_client()
    msgs = [{"role": "user", "content": user_prompt(inst)}]
    res = c.complete(messages=msgs, system=SYSTEM, tier=tier, step=STEP, response_format={"type": "json_object"})
    prop = _parse(res.text, inst)
    prop.problems = validate(inst, prop)
    if prop.problems:
        res2 = c.complete(messages=[*msgs, {"role": "assistant", "content": res.text},
                                    {"role": "user", "content": _repair_prompt(prop)}],
                          system=SYSTEM, tier=tier, step=STEP_REPAIR, response_format={"type": "json_object"})
        retry = _parse(res2.text, inst)
        retry.problems = validate(inst, retry)
        if len(retry.problems) <= len(prop.problems):
            prop = retry
    return prop


def with_personas(inst: GoldInstance, prop: PersonaProposal) -> GoldInstance:
    """Attach a validated proposal. The contract re-checks ``gold_changes`` against the branches.

    A relevant pair whose two personas land in branches with the *same* outcome
    would not change the gold, so it cannot be a relevant flip: the call raises
    rather than emit a pair the validator would later contradict.
    """
    if not prop.ok:
        raise ValueError(f"{inst.instance_id}: refusing to attach personas with problems: {prop.problems}")
    default_id = inst.default_branch.branch_id  # type: ignore[union-attr]
    personas = [Persona(persona_id="u_default", statement=prop.default, branch_id=default_id)]
    pairs: list[PersonaPair] = []
    for b in inst.exception_branches:
        pid = f"u_{b.branch_id}"
        personas.append(Persona(persona_id=pid, statement=prop.exceptions[b.branch_id], branch_id=b.branch_id))
        pairs.append(PersonaPair(base_persona_id="u_default", flipped_persona_id=pid, flip="relevant",
                                 attribute=(b.applicability.descriptor or b.condition or "condition"),
                                 gold_changes=True))
    personas.append(Persona(persona_id="u_default_irrelevant", statement=prop.irrelevant, branch_id=default_id))
    pairs.append(PersonaPair(base_persona_id="u_default", flipped_persona_id="u_default_irrelevant",
                             flip="irrelevant", attribute=prop.irrelevant_attribute, gold_changes=False))
    return GoldInstance.model_validate({**inst.model_dump(mode="json"),
                                        "personas": [p.model_dump(mode="json") for p in personas],
                                        "persona_pairs": [p.model_dump(mode="json") for p in pairs]})
