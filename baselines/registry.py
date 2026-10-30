"""The systems under evaluation, behind one interface (plan v4 S-1, S-2, S-3).

Every system takes the **same input** -- a query and a list of passages -- runs on
the **same backbone tier**, and returns a :class:`SystemAnswer` with its token
cost. That uniformity is not tidiness; it is what makes a comparison between
systems a comparison between *strategies*:

* the same context and the same instructions (a system that sees fewer passages,
  or a friendlier prompt, wins for reasons that have nothing to do with its
  design);
* one verbaliser, :func:`verbalise_branches`, for every structured system, so B7,
  B8 and O3 differ in what they *find*, not in how they phrase it;
* one backbone and one ``max_tokens`` / reasoning setting per cell, frozen at
  gate G1 and never mixed within a cell.

==========  ============================================================
B0          full-context generation: all passages, one neutral instruction
B3          ConflictRAG-style: resolve typed conflicts by selection, then generate
B7c         "SG-DT-style" compact structured parse, then verbalise
B8          B7c plus one self-verification pass
==========  ============================================================

**B3 is handed an oracle conflict signal.** Its published detector is trained on
other data and is not reproduced here, and a weak detector would let B3 quietly
degrade into B0 -- the objection "your strawman never fires". Given gold
conflict groups, B3 selects as often as it ever would: the *upper bound* on how
much a selection-based resolver can do. If it still suppresses valid exceptions
under perfect detection, the claim is stronger, not weaker. It is called
ConflictRAG-*style* throughout (the authors' code is not used).

**B7c is not the NormBench pipeline.** That pipeline's prompt is 18-27k tokens a
call, above the free tier's per-minute limit (``docs/v7_decisions.md`` D1.4). B7c
is a compact structured-parse prompt in the same spirit, and must be reported as
"SG-DT-style", never as the released pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from api_budget import Tier, get_client
from api_budget.client import LLMClient
from baselines.conflictrag_topsis import (
    DEFAULT_DELTA,
    select_by_credibility,
    select_by_recency,
)
from contract.gold import Applicability, Branch, GoldInstance
from contract.models import ConflictType, Passage

# --------------------------------------------------------------------------- #
# The shared answer contract
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM = (
    "Answer the user's question using only the passages provided. Be accurate and complete. "
    "Cite the passage ids you rely on in square brackets, like [p0]."
)


def format_passages(passages: Sequence[Passage]) -> str:
    return "\n\n".join(f"[{p.id}] {p.text}" for p in passages)


def answer_user_prompt(query: str, passages: Sequence[Passage]) -> str:
    return f"Question: {query}\n\nPassages:\n{format_passages(passages)}"


@dataclass(frozen=True)
class SystemAnswer:
    system_id: str
    text: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    branches: tuple[Branch, ...] = ()
    trace: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class System(Protocol):
    id: str

    def answer(self, query: str, passages: Sequence[Passage], *, conflicts: Sequence["ConflictGroup"] = ()
               ) -> SystemAnswer: ...


class _Meter:
    """Sums tokens and calls over one answer, from the client's real usage numbers."""

    def __init__(self) -> None:
        self.calls = self.prompt = self.completion = 0

    def run(self, client: LLMClient, **kw):
        res = client.complete(**kw)
        self.calls += 1
        self.prompt += res.prompt_tokens
        self.completion += res.completion_tokens
        return res


# --------------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------------- #


class FullContext:
    """B0: every passage, one neutral instruction, one call."""

    id = "B0"

    def __init__(self, *, tier: Tier, client: LLMClient | None = None):
        self.tier, self._client = tier, client

    @property
    def client(self) -> LLMClient:
        return self._client if self._client is not None else get_client()

    def answer(self, query, passages, *, conflicts=()):
        m = _Meter()
        res = m.run(self.client, messages=[{"role": "user", "content": answer_user_prompt(query, passages)}],
                    system=ANSWER_SYSTEM, tier=self.tier, step="sys_b0")
        return SystemAnswer(self.id, res.text.strip(), m.calls, m.prompt, m.completion,
                            trace={"passages": [p.id for p in passages]})


# --------------------------------------------------------------------------- #
# B3
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConflictGroup:
    """Passages in conflict, and how a ConflictRAG-style typer labelled it."""

    passage_ids: tuple[str, ...]
    type: str   # "factual" | "temporal" | "opinion"


def oracle_conflicts(inst: GoldInstance) -> list[ConflictGroup]:
    """Gold conflict groups for B3, mapped onto ConflictRAG's three-way taxonomy.

    ConflictRAG has no conditional class, so a rule-and-exception pair is typed
    ``factual`` (or ``temporal``), which is what makes selection its resolution.
    """
    if inst.gold_conflict_type is ConflictType.NO_CONFLICT or len(inst.passages) < 2:
        return []
    kind = {ConflictType.TEMPORAL: "temporal", ConflictType.OPINION: "opinion"}.get(
        inst.gold_conflict_type, "factual")
    return [ConflictGroup(tuple(sorted(p.id for p in inst.passages[:2])), kind)]


class ConflictRAGStyle:
    """B3: type the conflict, resolve factual by credibility and temporal by recency, then generate."""

    id = "B3"

    def __init__(self, *, tier: Tier, client: LLMClient | None = None, delta: float = DEFAULT_DELTA,
                 include_specificity: bool = True):
        self.tier, self._client = tier, client
        self.delta, self.include_specificity = delta, include_specificity

    @property
    def client(self) -> LLMClient:
        return self._client if self._client is not None else get_client()

    def select(self, passages: Sequence[Passage], conflicts: Sequence[ConflictGroup]):
        by_id = {p.id: p for p in passages}
        in_conflict = {pid for g in conflicts for pid in g.passage_ids if pid in by_id}
        kept = [p.id for p in passages if p.id not in in_conflict]
        decisions = []
        for g in conflicts:
            group = [by_id[pid] for pid in g.passage_ids if pid in by_id]
            if len(group) < 2:
                kept += [p.id for p in group]
                continue
            if g.type == "temporal":
                sel = select_by_recency(group)
            elif g.type == "opinion":
                kept += [p.id for p in group]          # multi-perspective: keep every view
                decisions.append({"type": g.type, "kind": "synthesis", "kept": [p.id for p in group]})
                continue
            else:
                sel = select_by_credibility(group, delta=self.delta, include_specificity=self.include_specificity)
            kept += list(sel.chosen)
            decisions.append({"type": g.type, "kind": sel.kind, "kept": list(sel.chosen),
                              "dropped": sorted(set(g.passage_ids) - set(sel.chosen)), "notes": sel.notes})
        order = {p.id: i for i, p in enumerate(passages)}
        final = [by_id[pid] for pid in sorted(set(kept), key=order.__getitem__)]
        return final, decisions

    def answer(self, query, passages, *, conflicts=()):
        final, decisions = self.select(passages, conflicts)
        m = _Meter()
        res = m.run(self.client, messages=[{"role": "user", "content": answer_user_prompt(query, final)}],
                    system=ANSWER_SYSTEM, tier=self.tier, step="sys_b3")
        return SystemAnswer(self.id, res.text.strip(), m.calls, m.prompt, m.completion,
                            trace={"selected": [p.id for p in final], "decisions": decisions})


# --------------------------------------------------------------------------- #
# Structured parse (B7c / B8) and the shared verbaliser
# --------------------------------------------------------------------------- #

PARSE_SYSTEM = """\
Read the passages and write down EVERY distinct outcome that applies to the question, and
exactly who it applies to. A general rule and a narrower exception stated in different passages
are both part of the answer.

Return ONLY JSON:
{"branches": [
  {"is_default": <true for the rule that applies to everyone unless an exception says otherwise>,
   "condition": "<the condition under which this outcome holds; null for the default>",
   "applies_to": "<short phrase: who it covers>",
   "outcome": "<short phrase: what happens>",
   "source": "<passage id such as p0>",
   "exclusion_guard": "<a limit stated in the passage that removes some people from this branch, or null>"}
]}
Rules: one branch per distinct (condition, outcome). At most one default. Cite the passage the
branch is stated in. Do not invent conditions the passages do not state."""

VERIFY_SYSTEM = """\
You are given passages and a DRAFT list of branches (condition, outcome, source) answering a question.
Check each branch against its passage. Fix wrong outcomes or scopes, add any exception or condition a
passage states that the draft is missing, and remove any branch no passage supports.
Return ONLY the corrected JSON in exactly the same schema: {"branches": [...]}."""


def parse_branches(raw, known_ids: set[str]) -> list[Branch]:
    """Model JSON -> validated :class:`Branch` objects, with conservative repairs.

    A branch citing a passage that does not exist keeps its claim but loses the
    citation: it is later counted as ungrounded instead of being quietly
    attributed to a real passage.
    """
    out: list[Branch] = []
    seen_default = False
    items = raw.get("branches") if isinstance(raw, dict) else None
    for i, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        outcome = str(item.get("outcome") or "").strip()
        if not outcome:
            continue
        is_default = bool(item.get("is_default", False))
        cond = item.get("condition")
        cond = str(cond).strip() if cond else None
        if is_default and seen_default:
            is_default, cond = False, cond or "unspecified"
        if is_default:
            cond, seen_default = None, True
        elif not cond:
            cond = "unspecified"
        src = str(item.get("source") or "").strip()
        src = src if src in known_ids else None
        scope = str(item.get("applies_to") or "").strip() or ("everyone" if is_default else cond)
        try:
            out.append(Branch(branch_id=f"b{i}", condition=cond, outcome=outcome[:300],
                              applicability=Applicability(descriptor=scope[:200], is_default=is_default),
                              supporting_passage=src))
        except Exception:
            continue
    return out


def _cite(src: str | None) -> str:
    return f" (per {src})" if src else ""


def _sentence(text: str) -> str:
    t = text.strip().rstrip(".")
    return (t[0].upper() + t[1:] + ".") if t else ""


def verbalise_branches(branches: Sequence[Branch]) -> str:
    """One deterministic rendering for every structured system.

    The default states its outcome plainly; every other branch is attached to
    its condition, so a reader can tell *who* each outcome is for -- which is
    what case-complete accuracy checks and what a bare list of outcomes lacks.
    """
    if not branches:
        return ""
    default = next((b for b in branches if b.is_default), None)
    others = [b for b in branches if not b.is_default]
    parts = []
    if default is not None:
        parts.append(_sentence(default.outcome).rstrip(".") + _cite(default.supporting_passage) + ".")
    for b in others:
        cond = (b.condition or b.applicability.descriptor).strip().rstrip(".")
        lead = "However, if" if default is not None and b is others[0] else "If"
        parts.append(f"{lead} {cond}, {b.outcome.strip().rstrip('.')}{_cite(b.supporting_passage)}.")
    return " ".join(parts)


def _json(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        from detection.stage2 import _salvage_json

        return _salvage_json(text)


class StructuredCompact:
    """B7c: a compact SG-DT-style structured parse, verbalised. Not the NormBench pipeline."""

    id = "B7c"
    verify = False

    def __init__(self, *, tier: Tier, client: LLMClient | None = None):
        self.tier, self._client = tier, client

    @property
    def client(self) -> LLMClient:
        return self._client if self._client is not None else get_client()

    def _parse(self, m: _Meter, query, passages) -> tuple[list[Branch], str]:
        res = m.run(self.client, messages=[{"role": "user", "content": answer_user_prompt(query, passages)}],
                    system=PARSE_SYSTEM, tier=self.tier, step=f"sys_{self.id.lower()}_parse",
                    response_format={"type": "json_object"})
        return parse_branches(_json(res.text), {p.id for p in passages}), res.text

    def _verify(self, m: _Meter, query, passages, draft: list[Branch]) -> list[Branch]:
        listing = json.dumps({"branches": [
            {"is_default": b.is_default, "condition": b.condition, "applies_to": b.applicability.descriptor,
             "outcome": b.outcome, "source": b.supporting_passage} for b in draft]}, indent=1)
        res = m.run(self.client, messages=[{"role": "user", "content":
                    f"{answer_user_prompt(query, passages)}\n\nDraft branches:\n{listing}"}],
                    system=VERIFY_SYSTEM, tier=self.tier, step=f"sys_{self.id.lower()}_verify",
                    response_format={"type": "json_object"})
        fixed = parse_branches(_json(res.text), {p.id for p in passages})
        return fixed or draft          # an unparseable verification never erases the draft

    def answer(self, query, passages, *, conflicts=()):
        m = _Meter()
        branches, _ = self._parse(m, query, passages)
        draft_n = len(branches)
        if self.verify and branches:
            branches = self._verify(m, query, passages, branches)
        return SystemAnswer(self.id, verbalise_branches(branches), m.calls, m.prompt, m.completion,
                            tuple(branches), {"parse_ok": draft_n > 0, "n_draft": draft_n,
                                              "n_final": len(branches)})


class StructuredVerified(StructuredCompact):
    """B8: B7c plus one self-verification pass over the draft."""

    id = "B8"
    verify = True


SYSTEMS = {"B0": FullContext, "B3": ConflictRAGStyle, "B7c": StructuredCompact, "B8": StructuredVerified}


def make_system(system_id: str, *, tier: Tier, client: LLMClient | None = None, **kw) -> System:
    if system_id not in SYSTEMS:
        raise KeyError(f"unknown system {system_id!r}; available: {sorted(SYSTEMS)}")
    return SYSTEMS[system_id](tier=tier, client=client, **kw)


def prompts_sha() -> str:
    """Hash of every instruction the systems send. Goes into the pre-registration:
    a prompt edited after the freeze is a different system."""
    import hashlib

    blob = json.dumps({"answer": ANSWER_SYSTEM, "parse": PARSE_SYSTEM, "verify": VERIFY_SYSTEM},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
