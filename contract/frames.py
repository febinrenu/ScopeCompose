"""Normative frames and decision-variable signatures (proposal v8 §4.1, §5.2).

A **normative frame** is one passage's statement about one decision, reduced to
the five things composition and alignment need:

    f = (signature, applicability, outcome, modality, span)

* ``signature`` identifies *what is being decided* ("cash withdrawn abroad,
  charged as a flat fee"). Two passages are about the same decision iff their
  signatures unify (``detection.signature.unify``) -- not iff their vocabulary
  overlaps. That is the whole point: a rule and its carve-out are written by
  different teams, so lexical overlap is exactly the signal that fails.
* ``applicability`` reuses :class:`contract.gold.Applicability`, so the typed
  per-attribute algebra in ``scope/`` keeps working unchanged.
* ``modality`` records whether the passage *determines* the outcome. A hedged
  statement ("may receive a waiver") does not -- case-complete accuracy relies
  on this.

This module is deliberately independent of the detector: it is data, with the
cheap structural validation that makes a malformed frame fail at the boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contract.gold import Applicability


class Modality(str, Enum):
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    PROHIBITION = "prohibition"
    STATEMENT = "statement"
    HEDGED = "hedged"

    @property
    def determines_outcome(self) -> bool:
        """Hedged modality never settles an outcome for a person."""
        return self is not Modality.HEDGED


class Signature(BaseModel):
    """A small typed record naming the decision a frame is about.

    ``slots`` hold values normalised to the domain schema's vocabulary
    (``qnfa/schema/<domain>.yaml``). A value outside the vocabulary is kept
    verbatim in ``untyped_slots`` instead: it can never make two signatures
    *distinct* (we cannot prove incompatibility from free text), only
    *uncertain*.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: Literal["banking", "immigration"]
    slots: dict[str, str] = Field(default_factory=dict)
    untyped_slots: dict[str, str] = Field(default_factory=dict)
    schema_version: str = Field(..., min_length=1)

    @field_validator("slots", "untyped_slots")
    @classmethod
    def _no_blank_keys_or_values(cls, v: dict[str, str]) -> dict[str, str]:
        for key, val in v.items():
            if not key.strip() or not val.strip():
                raise ValueError(f"signature slots need non-blank keys and values; got {key!r}={val!r}")
        return v

    def model_post_init(self, __context) -> None:  # type: ignore[no-untyped-def]
        clash = set(self.slots) & set(self.untyped_slots)
        if clash:
            raise ValueError(
                f"slot(s) {sorted(clash)} are both typed and untyped; a slot is one or the other"
            )

    def __hash__(self) -> int:  # frozen model with dict fields: hash by content
        return hash(
            (
                self.domain,
                self.schema_version,
                tuple(sorted(self.slots.items())),
                tuple(sorted(self.untyped_slots.items())),
            )
        )


class NormativeFrame(BaseModel):
    """One passage's claim about one decision, anchored to a query."""

    model_config = ConfigDict(extra="forbid")

    passage_id: str = Field(..., min_length=1)
    signature: Signature
    applicability: Applicability
    outcome: str = Field(..., min_length=1, description="The outcome, in the decision variable's value space.")
    modality: Modality = Modality.STATEMENT
    span: str = Field(
        ...,
        min_length=1,
        description="Supporting text, verbatim in the passage. Checked against the passage by the extractor.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def span_is_verbatim_in(self, passage_text: str) -> bool:
        """Whitespace-insensitive verbatim check -- the grounding gate for a frame."""
        norm = lambda s: " ".join(s.split())  # noqa: E731
        return norm(self.span) in norm(passage_text)
