"""The claims ledger: no number reaches the manuscript except through a result file.

Every quantitative statement a paper makes is a liability: it can be mistyped, left over from
an earlier run, or simply invented. This module makes that structurally impossible rather than
a matter of care.

* A **claim** is registered once, with the *path into a results file* that holds its value and
  how to format it.
* The manuscript never contains the number. It contains ``{{claim-id}}``; :func:`render`
  substitutes the value **from the results**, so re-running an experiment changes the text and a
  stale number cannot survive.
* :func:`render` raises on a placeholder that names no registered claim, and on one whose result
  has not been produced yet ("not yet measured"). A draft cannot be silently completed with a
  guess.
* :func:`ledger` writes the table of every claim, its source and its current value -- the
  document a reviewer, or the authors a week later, can audit.

Results are plain nested dictionaries (the JSON the analysis scripts write); a path is dotted
(``"h1.p_exact"``). Lists are indexed with integers (``"detectors.0.f1"``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*(?:\|\s*([a-z0-9]+)\s*)?\}\}")


class ClaimError(RuntimeError):
    """A placeholder could not be rendered honestly."""


@dataclass(frozen=True)
class Claim:
    id: str
    source: str
    description: str
    fmt: str = "f3"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", self.id):
            raise ValueError(f"invalid claim id {self.id!r}")
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown format {self.fmt!r}; choose from {sorted(FORMATS)}")


def _p(v: float) -> str:
    return "<0.0001" if v < 0.0001 else f"{v:.4f}"


FORMATS = {
    "f1": lambda v: f"{v:.1f}",
    "f2": lambda v: f"{v:.2f}",
    "f3": lambda v: f"{v:.3f}",
    "int": lambda v: f"{int(round(v))}",
    "pct": lambda v: f"{100 * v:.1f}%",
    "pts": lambda v: f"{100 * v:+.1f}",
    "p": _p,
    "raw": lambda v: str(v),
}


def resolve(results: Mapping[str, Any], path: str) -> Any:
    """Follow a dotted path through nested dicts and lists. Raises :class:`KeyError` if absent."""
    cur: Any = results
    for part in path.split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                raise KeyError(path)
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                raise KeyError(path) from None
        else:
            raise KeyError(path)
    return cur


class Registry:
    def __init__(self, claims: list[Claim] | None = None):
        self._claims: dict[str, Claim] = {}
        for c in claims or []:
            self.add(c)

    def add(self, claim: Claim) -> None:
        if claim.id in self._claims:
            raise ValueError(f"claim {claim.id!r} is already registered")
        self._claims[claim.id] = claim

    def __getitem__(self, cid: str) -> Claim:
        if cid not in self._claims:
            raise ClaimError(f"{{{{{cid}}}}} names no registered claim; register it with its source first")
        return self._claims[cid]

    def __iter__(self):
        return iter(self._claims.values())

    def __len__(self) -> int:
        return len(self._claims)


def value_of(claim: Claim, results: Mapping[str, Any], fmt: str | None = None) -> str:
    try:
        raw = resolve(results, claim.source)
    except KeyError:
        raise ClaimError(f"{claim.id}: not yet measured (no value at {claim.source!r} in the results)") from None
    f = fmt or claim.fmt
    if f not in FORMATS:
        raise ClaimError(f"{claim.id}: unknown format {f!r}")
    if f == "raw":
        return FORMATS[f](raw)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ClaimError(f"{claim.id}: the value at {claim.source!r} is {raw!r}, not a number")
    if isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw)):
        raise ClaimError(f"{claim.id}: the value at {claim.source!r} is {raw}, which cannot be reported")
    return FORMATS[f](float(raw))


def placeholders(text: str) -> list[str]:
    """Every claim id the text refers to, in order, without duplicates."""
    return list(dict.fromkeys(m.group(1) for m in PLACEHOLDER.finditer(text)))


def render(text: str, registry: Registry, results: Mapping[str, Any]) -> str:
    """Substitute every placeholder from the results. Raises on the first one that cannot be honest."""
    def sub(m: re.Match[str]) -> str:
        return value_of(registry[m.group(1)], results, m.group(2))

    return PLACEHOLDER.sub(sub, text)


def raw_numbers(text: str, *, allow: tuple[str, ...] = ()) -> list[str]:
    """Numbers typed directly into prose (outside placeholders), so a draft can be checked for them.

    Years, section numbers and anything matching ``allow`` are ignored; everything else is a number
    that bypassed the ledger."""
    stripped = PLACEHOLDER.sub(" ", text)
    found = re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w.]*\d)", stripped)
    out = []
    for f in found:
        if re.fullmatch(r"(19|20)\d{2}", f) or any(re.fullmatch(a, f) for a in allow):
            continue
        out.append(f)
    return out


def ledger(registry: Registry, results: Mapping[str, Any]) -> str:
    """A markdown table of every claim, its source and its current value."""
    lines = ["| claim | statement | source | value |", "|---|---|---|---|"]
    for c in registry:
        try:
            v = value_of(c, results)
        except ClaimError as exc:
            v = f"**{str(exc).split(': ', 1)[-1]}**"
        lines.append(f"| `{c.id}` | {c.description} | `{c.source}` | {v} |")
    return "\n".join(lines)


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unmeasured(registry: Registry, results: Mapping[str, Any]) -> list[str]:
    """Claim ids with no value yet -- what stands between the draft and a finished paper."""
    out = []
    for c in registry:
        try:
            value_of(c, results)
        except ClaimError:
            out.append(c.id)
    return out
