"""Decision-variable signatures and their unification (proposal v8 sections 5.2-5.3).

QNFA decides whether two passages govern the **same decision** by unifying small
typed records, not by thresholding how similar their words are. A rule and its
carve-out are written by different teams in different registers, which is
exactly why a lexical or embedding signal misses them (the pilot detector found
2 of 67). A signature makes the comparison structural.

A signature has typed *slots* (``event``, ``object``/``route``,
``decision_kind``, ``location``, ``channel``). Each slot's values form a
**generality taxonomy** declared in ``qnfa/schema/<domain>.yaml``: ``any_txn`` is
broader than ``cash_withdrawal``, which is broader than ``atm_withdrawal``. An
*unspecified* slot is the top element -- broader than any value.

Unification of two signatures is decided slot by slot:

======================  ================================================
``EQUIVALENT``          every slot equal, or unspecified on both sides
``SUBSUMES``            one signature is at least as general on every slot
                        and strictly more general on one: the other is a
                        *narrower* statement of the same decision
``OVERLAP``             each is more general on a different slot (cash
                        withdrawals anywhere vs any transaction abroad)
``DISTINCT``            some slot holds two incompatible values (flat fee vs
                        percentage fee): different decisions
``UNCERTAIN``           nothing incompatible, but a slot comparison involves
                        free text that cannot be ordered
======================  ================================================

``OVERLAP`` extends proposal v8's three-valued test. v8 does not say what to do
when generality crosses; refusing to compare such pairs would drop a real class
(an ATM fee and a foreign-transaction fee both reach a foreign ATM withdrawal),
and calling them ``UNCERTAIN`` would send a decidable case to an LLM. It is
treated like ``SUBSUMES``: both differences are folded into applicability, and
the scope algebra then sees a *crossing* pair.

**Free text can never make two signatures distinct.** We cannot prove two
free-text values incompatible, so they yield ``UNCERTAIN`` and go to a
tie-break with an abstain option -- never to ``DISTINCT``, which would silently
discard a candidate pair.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from contract.frames import Signature
from contract.gold import Applicability, AttributeKind, ScopeAttribute

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "qnfa" / "schema"

#: Prefix for scope attributes synthesised from a signature difference.
SIG_ATTR_PREFIX = "sig:"


class SchemaError(ValueError):
    pass


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Schema:
    domain: str
    version: str
    sha: str
    parents: dict[str, dict[str, str | None]]
    """slot -> value -> parent value (``None`` for the slot's top element)."""
    aliases: dict[str, dict[str, str]]
    """slot -> normalised alias -> canonical value."""

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(self.parents)

    def root(self, slot: str) -> str:
        return next(v for v, p in self.parents[slot].items() if p is None)

    def ancestors(self, slot: str, value: str) -> list[str]:
        """``value`` first, then its parent chain up to the root."""
        out = [value]
        while (p := self.parents[slot][out[-1]]) is not None:
            out.append(p)
        return out

    def leq(self, slot: str, narrow: str, broad: str) -> bool:
        """``broad`` is at least as general as ``narrow`` (equal counts)."""
        return broad in self.ancestors(slot, narrow)

    def descendants_or_self(self, slot: str, value: str) -> list[str]:
        return sorted(v for v in self.parents[slot] if self.leq(slot, v, value))

    def normalise(self, slot: str, raw: str) -> str | None:
        """Canonical value for a raw string, or ``None`` if outside the vocabulary."""
        if slot not in self.parents:
            return None
        k = _key(raw)
        if k in self.parents[slot]:
            return k
        return self.aliases[slot].get(k)


def _validate(raw: dict) -> None:
    for slot, values in raw["slots"].items():
        roots = [v for v, spec in values.items() if not (spec or {}).get("parent")]
        if len(roots) != 1:
            raise SchemaError(f"slot {slot!r} needs exactly one top element, found {roots}")
        for v, spec in values.items():
            p = (spec or {}).get("parent")
            if p is not None and p not in values:
                raise SchemaError(f"{slot}.{v}: unknown parent {p!r}")
        seen: dict[str, str] = {}
        for v, spec in values.items():
            for a in [v, *(spec or {}).get("aliases", [])]:
                k = _key(a)
                if k in seen and seen[k] != v:
                    raise SchemaError(f"{slot}: {a!r} names both {seen[k]!r} and {v!r}")
                seen[k] = v
        for v in values:                                   # no cycles
            chain, cur = [], v
            while cur is not None:
                if cur in chain:
                    raise SchemaError(f"{slot}: cycle through {cur!r}")
                chain.append(cur)
                cur = (values[cur] or {}).get("parent")


@lru_cache(maxsize=None)
def load_schema(domain: str, schema_dir: Path | None = None) -> Schema:
    path = (schema_dir or SCHEMA_DIR) / f"{domain}.yaml"
    data = path.read_bytes()
    raw = yaml.safe_load(data)
    if raw.get("domain") != domain:
        raise SchemaError(f"{path.name}: declares domain {raw.get('domain')!r}, expected {domain!r}")
    _validate(raw)
    parents: dict[str, dict[str, str | None]] = {}
    aliases: dict[str, dict[str, str]] = {}
    for slot, values in raw["slots"].items():
        parents[slot] = {v: ((spec or {}).get("parent") or None) for v, spec in values.items()}
        aliases[slot] = {_key(a): v for v, spec in values.items() for a in (spec or {}).get("aliases", [])}
    return Schema(domain, raw["schema_version"], hashlib.sha256(data).hexdigest(), parents, aliases)


def make_signature(domain: str, raw_slots: dict[str, str], *, schema: Schema | None = None) -> Signature:
    """Normalise raw extractor output into a :class:`Signature`.

    A value inside the vocabulary becomes a typed slot. A value outside it, or a
    slot the schema does not know, is kept verbatim as ``untyped_slots`` -- never
    guessed into the nearest value, and never dropped. The out-of-vocabulary rate
    is a reported number.
    """
    sch = schema or load_schema(domain)
    typed: dict[str, str] = {}
    untyped: dict[str, str] = {}
    for slot, raw in raw_slots.items():
        if raw is None or not str(raw).strip():
            continue
        canon = sch.normalise(slot, str(raw))
        if canon is not None and canon != sch.root(slot):
            typed[slot] = canon                    # the top element is "unspecified"
        elif canon is None:
            untyped[slot] = str(raw).strip()
    return Signature(domain=domain, slots=typed, untyped_slots=untyped, schema_version=sch.version)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Unification
# --------------------------------------------------------------------------- #


class Kind(str, Enum):
    EQUIVALENT = "equivalent"
    SUBSUMES = "subsumes"
    OVERLAP = "overlap"
    DISTINCT = "distinct"
    UNCERTAIN = "uncertain"


class SlotRel(str, Enum):
    EQUAL = "equal"
    A_BROADER = "a_broader"
    B_BROADER = "b_broader"
    INCOMPATIBLE = "incompatible"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Unification:
    kind: Kind
    general: str | None = None
    """For ``SUBSUMES``: ``"a"`` or ``"b"`` -- the side that is the broader statement."""
    slots: dict[str, SlotRel] = field(default_factory=dict)

    def flipped(self) -> "Unification":
        flip = {SlotRel.A_BROADER: SlotRel.B_BROADER, SlotRel.B_BROADER: SlotRel.A_BROADER}
        g = {"a": "b", "b": "a", None: None}[self.general]
        return Unification(self.kind, g, {s: flip.get(r, r) for s, r in self.slots.items()})

    @property
    def incompatible_slots(self) -> list[str]:
        return [s for s, r in self.slots.items() if r is SlotRel.INCOMPATIBLE]


def _slot_rel(sch: Schema, slot: str, a: Signature, b: Signature) -> SlotRel:
    ta, tb = a.slots.get(slot), b.slots.get(slot)
    ua, ub = a.untyped_slots.get(slot), b.untyped_slots.get(slot)

    # Free text on one side and *nothing* on the other is decidable: unspecified is
    # the top element, so the side with no value is the broader statement.
    if ua is not None and ta is None and tb is None and ub is None:
        return SlotRel.B_BROADER
    if ub is not None and tb is None and ta is None and ua is None:
        return SlotRel.A_BROADER
    if ua is not None and ub is not None:
        return SlotRel.EQUAL if _key(ua) == _key(ub) else SlotRel.UNCERTAIN
    if ua is not None or ub is not None:
        return SlotRel.UNCERTAIN                      # free text against a typed value
    if ta is None and tb is None:
        return SlotRel.EQUAL
    if ta is None:
        return SlotRel.A_BROADER                      # unspecified is the top element
    if tb is None:
        return SlotRel.B_BROADER
    if ta == tb:
        return SlotRel.EQUAL
    if sch.leq(slot, tb, ta):
        return SlotRel.A_BROADER
    if sch.leq(slot, ta, tb):
        return SlotRel.B_BROADER
    return SlotRel.INCOMPATIBLE


def unify(a: Signature, b: Signature, *, schema: Schema | None = None) -> Unification:
    """Unify two signatures. See the module docstring for the five outcomes."""
    if a.domain != b.domain:
        raise ValueError(f"cannot unify signatures from different domains: {a.domain} / {b.domain}")
    sch = schema or load_schema(a.domain)
    names = sorted(set(a.slots) | set(b.slots) | set(a.untyped_slots) | set(b.untyped_slots))
    rels = {s: _slot_rel(sch, s, a, b) for s in names}

    vals = set(rels.values())
    if SlotRel.INCOMPATIBLE in vals:
        return Unification(Kind.DISTINCT, None, rels)
    if SlotRel.UNCERTAIN in vals:
        return Unification(Kind.UNCERTAIN, None, rels)
    a_b, b_b = SlotRel.A_BROADER in vals, SlotRel.B_BROADER in vals
    if a_b and b_b:
        return Unification(Kind.OVERLAP, None, rels)
    if a_b:
        return Unification(Kind.SUBSUMES, "a", rels)
    if b_b:
        return Unification(Kind.SUBSUMES, "b", rels)
    return Unification(Kind.EQUIVALENT, None, rels)


# --------------------------------------------------------------------------- #
# Folding a signature difference into applicability
# --------------------------------------------------------------------------- #


def _with_signature_constraints(app: Applicability, extra: list[ScopeAttribute], why: str) -> Applicability:
    if not extra:
        return app
    return Applicability(
        descriptor=f"{app.descriptor} ({why})" if not app.is_default else why,
        attributes=[*app.attributes, *extra],
        is_default=False,
    )


def fold_into_applicability(
    u: Unification,
    sig_a: Signature,
    sig_b: Signature,
    app_a: Applicability,
    app_b: Applicability,
    *,
    schema: Schema | None = None,
) -> tuple[Applicability, Applicability]:
    """Turn the narrower side's extra specificity into typed applicability.

    "Foreign transactions incur 2.99%" against "overseas ATM withdrawals on
    Account X are free" are two statements about *different-sounding* variables.
    Once unified, the second is the same decision *restricted to ATM cash
    withdrawals*; that restriction becomes a typed constraint on its
    applicability (``sig:event in {atm_withdrawal}``) so the scope algebra sees
    an ordinary refinement -- the case lexical gating misses.

    Only ``SUBSUMES`` and ``OVERLAP`` fold: an ``EQUIVALENT`` pair has nothing to
    fold, ``DISTINCT`` pairs are not compared, and ``UNCERTAIN`` pairs must be
    resolved first.
    """
    if u.kind not in (Kind.SUBSUMES, Kind.OVERLAP):
        return app_a, app_b
    sch = schema or load_schema(sig_a.domain)

    def constraints(sig: Signature, narrower_when: SlotRel) -> list[ScopeAttribute]:
        out = []
        for slot, rel in u.slots.items():
            if rel is narrower_when and slot in sig.slots:
                out.append(ScopeAttribute(
                    name=f"{SIG_ATTR_PREFIX}{slot}", kind=AttributeKind.CATEGORICAL,
                    values=sch.descendants_or_self(slot, sig.slots[slot])))
        return out

    # `a` is the narrower side on a slot where *b* is broader.
    extra_a = constraints(sig_a, SlotRel.B_BROADER)
    extra_b = constraints(sig_b, SlotRel.A_BROADER)
    why_a = "narrower decision: " + ", ".join(f"{c.name[len(SIG_ATTR_PREFIX):]}={c.values[0] if len(c.values or []) == 1 else '|'.join(c.values or [])}" for c in extra_a)
    why_b = "narrower decision: " + ", ".join(f"{c.name[len(SIG_ATTR_PREFIX):]}={c.values[0] if len(c.values or []) == 1 else '|'.join(c.values or [])}" for c in extra_b)
    return (_with_signature_constraints(app_a, extra_a, why_a),
            _with_signature_constraints(app_b, extra_b, why_b))


def schema_hashes(domains: Iterable[str] = ("banking", "immigration")) -> dict[str, str]:
    """Hashed into the pre-registration: the schemas are frozen from then on."""
    return {d: load_schema(d).sha for d in domains}
