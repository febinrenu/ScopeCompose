"""Branch assembly: the glue between A4/B1 and B2.

B1 extracts *exception* branches only. Every experiment so far borrowed the
default branch from gold (``experiments/wp1_probe/run_probe.py``), which is
fine for scoring B1 but leaves nothing to compose a fresh query with. This
builds the default from the pipeline's own outputs:

- the base passage is A4's default side (applicability-derived, never order);
- its outcome is A4's descriptor outcome when it has one, else the passage's
  first sentence -- the general rule as written.
"""

from __future__ import annotations

import re

from contract.gold import Applicability, Branch, GoldInstance
from contract.models import Passage

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text: str) -> str:
    s = _SENTENCE.split(text.strip(), maxsplit=1)[0].strip()
    return s.rstrip(".") if s else text.strip()


def usable_outcome(descriptor) -> str | None:
    """A4's outcome, when it is a whole statement.

    The offline rule-based descriptor falls back to the first 80 characters of
    the passage, which is enough for A4 to compare outcomes but is not a thing
    to state in an answer ("... made in a f"). Only an LLM-extracted outcome,
    or one that ends where a clause ends, is used; otherwise the default branch
    takes the passage's own first sentence.
    """
    out = (getattr(descriptor, "outcome", "") or "").strip()
    if not out or out == "unspecified":
        return None
    if getattr(descriptor, "source", "") == "llm":
        return out
    return out if re.search(r"[.!?;:)]$|\w{3,}$", out) and len(out) < 80 else None


def _as_clause(sentence: str) -> str:
    """A sentence taken from a passage reads mid-answer ("By default, as of April ...").
    Lower-case its first letter unless it starts an acronym or a name-like token."""
    words = sentence.split(" ", 1)
    w = words[0]
    if w == "A" or (len(w) > 1 and w[0].isupper() and w[1:].islower()):
        return w[0].lower() + sentence[1:]
    return sentence


def default_branch(base: Passage, outcome: str | None = None, *, branch_id: str = "b0") -> Branch:
    out = (outcome or "").strip() or _as_clause(first_sentence(base.text))
    return Branch(
        branch_id=branch_id,
        condition=None,
        outcome=out,
        applicability=Applicability(descriptor="all cases", is_default=True),
        supporting_passage=base.id,
    )


def assemble(base: Passage, default_outcome: str | None, exceptions: list[Branch]) -> list[Branch]:
    """Default first, then exceptions renumbered b1..bn (ids must be unique)."""
    out = [default_branch(base, default_outcome)]
    for i, b in enumerate(exceptions, start=1):
        out.append(b.model_copy(update={"branch_id": f"b{i}"}))
    return out


def gold_branches(inst: GoldInstance) -> list[Branch]:
    return list(inst.gold_branches)


# --------------------------------------------------------------------------- #
# Readers: which branch is true for a given person
# --------------------------------------------------------------------------- #


def reader_dims(branches: list[Branch]) -> list[dict]:
    """The typed attributes a reader can be described by, merged across branches."""
    dims: dict[str, dict] = {}
    for b in branches:
        for a in b.applicability.attributes:
            d = dims.setdefault(a.name, {"name": a.name, "kind": a.kind.value, "values": []})
            for v in a.values or []:
                if v not in d["values"]:
                    d["values"].append(v)
            if a.min_value is not None:
                d["min"] = a.min_value
            if a.max_value is not None:
                d["max"] = a.max_value
    return list(dims.values())


def satisfies(branch: Branch, reader: dict) -> bool | None:
    """True / False when decidable from typed attributes; None when it is not.

    The default branch holds for everyone. A conditioned branch with no typed
    attributes (a free-text descriptor) cannot be decided mechanically -- that
    is reported as ``None`` rather than guessed.
    """
    if branch.is_default:
        return True
    attrs = branch.applicability.attributes
    if not attrs:
        return None
    for a in attrs:
        if a.name not in reader or reader[a.name] in (None, "", "__other__"):
            return False
        v = reader[a.name]
        if a.values is not None:
            if str(v) not in a.values:
                return False
        else:
            try:
                x = float(v)
            except (TypeError, ValueError):
                return False
            if a.min_value is not None and x < a.min_value:
                return False
            if a.max_value is not None and x > a.max_value:
                return False
    return True


def branch_for_reader(branches: list[Branch], reader: dict) -> Branch | None:
    """The most specific branch that holds: an exception if any applies, else the default."""
    exc = [b for b in branches if not b.is_default and satisfies(b, reader)]
    if exc:
        return exc[0]
    return next((b for b in branches if b.is_default), None)
