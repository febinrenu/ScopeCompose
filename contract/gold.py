"""The gold benchmark-instance schema.

Distinct from :mod:`contract.models`, and the distinction matters:

* :class:`~contract.models.QueryRecord` is the **wire format** -- what Member
  A's system emits at runtime and Member B's system consumes.
* :class:`GoldInstance` here is the **annotation format** -- everything two
  human annotators plus a third reviewer record about an instance, including
  the branch structure and reference answers that no runtime component
  produces.

A gold instance can be projected down to a wire record (see
:meth:`GoldInstance.to_query_record`), which is how Member A gets an oracle to
train and evaluate against before the real detector exists.

Annotation ownership:

* **Member A** labels conflict type, the four-way scope relation, distractors.
* **Member B** labels branch structure, gold scoped answers, the selection
  answer reference, and the ``construction`` tag.
* **Both** co-author the annotation manual, including the refinement-vs-opposed
  decision procedure, and both report kappa for their own label axis.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contract.models import (
    CONTRACT_VERSION,
    ConflictType,
    Construction,
    Domain,
    MultiExceptionFlags,
    Opposition,
    Passage,
    QueryRecord,
    ScopeRelation,
    Separation,
)


class Explicitness(str, Enum):
    """Whether the passage MARKS ITSELF as an exception to a general rule.

    The operational test, applied by ``benchmark.wp1_probe_set.check_implicitness``,
    is the presence of a strong exception cue -- "except", "unless", "only if",
    "does not apply to". Absent one, the case is ``IMPLICIT``.

    **This is narrower than it sounds, and the difference matters for how the
    number is reported.** It does not mean the condition is unstated. In
    "Where the statement balance is settled in full by the due date, purchases
    benefit from interest-free credit", the condition is stated plainly; what is
    unstated is that it *overrides* the general rule that interest accrues from
    the transaction date. So ``IMPLICIT`` means the reader must infer the
    exception RELATIONSHIP, not that they must infer the condition itself.

    Calling it "implicit condition recovery" in the paper would overclaim.
    Nearly every passage in the probe set states its own condition somewhere;
    what varies is whether the passage announces itself as a carve-out.

    Reported separately, never averaged: the unmarked cases are the hard ones
    and the ones Contrastive Scope Probing exists to address. Averaging the two
    hides exactly the number the method is judged on.
    """

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class AttributeKind(str, Enum):
    """The shape of an applicability attribute.

    Typed so the scope analyser can decide subset/disjoint arithmetically
    rather than asking a language model to do set theory in prose.
    """

    CATEGORICAL = "categorical"   # {premium, platinum} - membership in a set
    ORDERED_TIER = "ordered_tier"  # bronze < silver < gold - an ordered scale
    NUMERIC_RANGE = "numeric_range"  # age 18-65, amount > 500
    BOOLEAN = "boolean"           # is_student = true
    FREE_TEXT = "free_text"       # escape hatch; no arithmetic possible


class Applicability(BaseModel):
    """The set of cases a branch applies to.

    ``descriptor`` is the human-readable scope ("premium-tier cardholders").
    ``attributes`` is the machine-checkable decomposition, when one exists.
    An empty ``attributes`` list means the default branch (applies to all of U)
    or an unparsed scope; either way the analyser falls back to entailment.
    """

    model_config = ConfigDict(extra="forbid")

    descriptor: str = Field(..., min_length=1)
    attributes: list[ScopeAttribute] = Field(default_factory=list)
    is_default: bool = Field(
        default=False,
        description="True for the unconditioned branch, whose applicability is the whole case space U.",
    )


class ScopeAttribute(BaseModel):
    """One machine-checkable dimension of an applicability set."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="e.g. 'card_tier', 'age', 'transaction_origin'")
    kind: AttributeKind

    # Exactly one of these is populated, according to `kind`.
    values: list[str] | None = Field(default=None, description="CATEGORICAL / ORDERED_TIER members")
    min_value: float | None = Field(default=None, description="NUMERIC_RANGE lower bound, inclusive")
    max_value: float | None = Field(default=None, description="NUMERIC_RANGE upper bound, inclusive")
    boolean_value: bool | None = Field(default=None, description="BOOLEAN value")
    text: str | None = Field(default=None, description="FREE_TEXT fallback")

    @model_validator(mode="after")
    def _check_payload_matches_kind(self) -> ScopeAttribute:
        k = self.kind
        if k in (AttributeKind.CATEGORICAL, AttributeKind.ORDERED_TIER):
            if not self.values:
                raise ValueError(f"kind={k.value} requires a non-empty 'values' list")
        elif k is AttributeKind.NUMERIC_RANGE:
            if self.min_value is None and self.max_value is None:
                raise ValueError("kind=numeric_range requires at least one of min_value / max_value")
            if (
                self.min_value is not None
                and self.max_value is not None
                and self.min_value > self.max_value
            ):
                raise ValueError(f"min_value {self.min_value} exceeds max_value {self.max_value}")
        elif k is AttributeKind.BOOLEAN:
            if self.boolean_value is None:
                raise ValueError("kind=boolean requires 'boolean_value'")
        elif k is AttributeKind.FREE_TEXT:
            if not self.text:
                raise ValueError("kind=free_text requires 'text'")
        return self


class Branch(BaseModel):
    """One (condition, outcome, applicability) triple.

    A gold instance is a set of these: exactly one default branch plus zero or
    more exception branches. This is the representation the composition
    operator assembles and the metric suite scores against, branch by branch.
    """

    model_config = ConfigDict(extra="forbid")

    branch_id: str = Field(..., min_length=1)
    condition: str | None = Field(
        default=None, description="None for the default branch; the governing condition otherwise."
    )
    outcome: str = Field(..., min_length=1)
    applicability: Applicability
    explicitness: Explicitness = Explicitness.EXPLICIT
    supporting_passage: str | None = Field(
        default=None, description="Passage id this branch is grounded in. Required for exceptions."
    )

    @property
    def is_default(self) -> bool:
        return self.condition is None

    @model_validator(mode="after")
    def _check_grounding(self) -> Branch:
        if self.condition is None and not self.applicability.is_default:
            raise ValueError(
                f"branch {self.branch_id!r} has no condition, so it is the default branch "
                "and its applicability must have is_default=True"
            )
        if self.condition is not None and self.applicability.is_default:
            raise ValueError(
                f"branch {self.branch_id!r} has a condition but applicability.is_default=True; "
                "a conditioned branch cannot apply to the whole case space"
            )
        return self


class Persona(BaseModel):
    """A short first-person case description that falls in exactly one gold region.

    Personas are what case-complete accuracy (CCA) is scored over: for each one
    the reader must determine, from a system's answer alone, the outcome that
    holds for that person. ``statement`` is paraphrased from the source wording
    (never from attribute identifiers) and human-verified; ``branch_id`` names
    the gold branch that governs this person, so the gold outcome is derived
    from the instance, not stored twice.
    """

    model_config = ConfigDict(extra="forbid")

    persona_id: str = Field(..., min_length=1)
    statement: str = Field(..., min_length=1)
    branch_id: str = Field(..., min_length=1, description="The gold branch that governs this person.")
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Attribute values the persona fixes, keyed like ScopeAttribute.name. Informational.",
    )


class PersonaPair(BaseModel):
    """A minimal counterfactual pair (proposal v8 §6.5).

    ``flip='relevant'`` changes only the attribute that defines the exception
    region, so the gold outcome MUST change. ``flip='irrelevant'`` changes an
    attribute no frame constrains (a city of residence for a fee rule), so the
    gold outcome MUST NOT change. These pairs validate the reader (V2) and give
    a system-level consistency score.
    """

    model_config = ConfigDict(extra="forbid")

    base_persona_id: str = Field(..., min_length=1)
    flipped_persona_id: str = Field(..., min_length=1)
    flip: Literal["relevant", "irrelevant"]
    attribute: str = Field(..., min_length=1, description="The attribute that was flipped.")
    gold_changes: bool = Field(..., description="Whether the gold outcome differs between the two personas.")

    @model_validator(mode="after")
    def _check_flip_matches_expectation(self) -> PersonaPair:
        if self.base_persona_id == self.flipped_persona_id:
            raise ValueError("a persona pair needs two different personas")
        if (self.flip == "relevant") is not self.gold_changes:
            raise ValueError(
                f"flip={self.flip!r} requires gold_changes={self.flip == 'relevant'}, "
                f"got gold_changes={self.gold_changes}"
            )
        return self


class Annotation(BaseModel):
    """Provenance of the labels, for the three-stage protocol and kappa reporting."""

    model_config = ConfigDict(extra="forbid")

    annotator_a: str | None = None
    annotator_b: str | None = None
    disagreed: bool = Field(
        default=False, description="True if the two independent annotators differed before discussion."
    )
    resolved_by_discussion: bool = False
    third_reviewer: str | None = Field(
        default=None,
        description="Set when a procedure-ambiguous case was routed to the expert reviewer.",
    )
    notes: str | None = None


class GoldInstance(BaseModel):
    """A fully annotated benchmark instance."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = CONTRACT_VERSION
    instance_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    domain: Domain
    construction: Construction
    separation: Separation | None = Field(
        default=None,
        description=(
            "Finer-grained provenance of the document separation. Additive to "
            "'construction': same_guide instances are stored as Tier 2 but "
            "reported as their own row, because they are neither naturally "
            "cross-document nor author-split."
        ),
    )
    split: str = Field(default="train", description="train | dev | test -- fixed at release.")

    passages: list[Passage] = Field(..., min_length=1)

    # -- Member A's labels --------------------------------------------------- #
    gold_conflict_type: ConflictType
    gold_scope_relation: ScopeRelation | None = None
    gold_opposition: Opposition | None = Field(
        default=None,
        description=(
            "v1.3.0. Refines gold_scope_relation='opposed' into crossing / same_scope / "
            "unknown. Absent for every other relation, and for pre-1.3.0 records."
        ),
    )
    is_distractor: bool = Field(
        default=False,
        description=(
            "Mandatory negative: a genuine factual/temporal conflict, a disjoint "
            "non-conflicting pair, a redundant restatement, or a no-conflict instance. "
            "Without these, Spurious-Condition Rate has no denominator."
        ),
    )

    # -- Member B's labels --------------------------------------------------- #
    gold_branches: list[Branch] = Field(default_factory=list)
    gold_scoped_answer: str | None = Field(
        default=None, description="The reference answer preserving every valid branch."
    )
    selection_answer: str | None = Field(
        default=None,
        description=(
            "What a credibility-based selection system would output. The reference "
            "against which Suppression Rate is demonstrated -- without it, SR is an "
            "assertion rather than a measurement."
        ),
    )
    gold_multi_exception_flags: MultiExceptionFlags = Field(default_factory=MultiExceptionFlags)

    # -- v1.3.0: case-complete accuracy ------------------------------------- #
    personas: list[Persona] = Field(default_factory=list)
    persona_pairs: list[PersonaPair] = Field(default_factory=list)

    annotation: Annotation = Field(default_factory=Annotation)

    # -- validation ---------------------------------------------------------- #

    @model_validator(mode="after")
    def _check_labels(self) -> GoldInstance:
        if self.gold_conflict_type is ConflictType.CONDITIONAL and self.gold_scope_relation is None:
            raise ValueError(
                f"{self.instance_id!r}: a conditional instance needs a gold_scope_relation"
            )
        if self.gold_conflict_type is ConflictType.NO_CONFLICT and self.gold_scope_relation is not None:
            raise ValueError(
                f"{self.instance_id!r}: a no_conflict instance must not carry a scope_relation"
            )
        return self

    @model_validator(mode="after")
    def _check_opposition(self) -> GoldInstance:
        if self.gold_opposition is not None and self.gold_scope_relation is not ScopeRelation.OPPOSED:
            raise ValueError(
                f"{self.instance_id!r}: gold_opposition={self.gold_opposition.value!r} is only "
                f"meaningful for scope_relation='opposed', got {self.gold_scope_relation!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_personas(self) -> GoldInstance:
        """Personas must land in a real gold branch; pairs must agree with the branches.

        The pair's ``gold_changes`` is *derived* data (from the two personas'
        branches), so it is checked against that derivation here rather than
        trusted: a hand-edited pair whose flag contradicts the branches would
        otherwise silently corrupt the reader-validity numbers.
        """
        if not self.personas and not self.persona_pairs:
            return self
        if not self.gold_branches:
            raise ValueError(f"{self.instance_id!r}: personas need gold_branches to be governed by")

        branches = {b.branch_id: b for b in self.gold_branches}
        ids = [p.persona_id for p in self.personas]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.instance_id!r}: duplicate persona_ids {ids}")
        by_id = {p.persona_id: p for p in self.personas}
        for p in self.personas:
            if p.branch_id not in branches:
                raise ValueError(
                    f"{self.instance_id!r}: persona {p.persona_id!r} cites unknown branch {p.branch_id!r}"
                )
        for pair in self.persona_pairs:
            for pid in (pair.base_persona_id, pair.flipped_persona_id):
                if pid not in by_id:
                    raise ValueError(f"{self.instance_id!r}: persona pair cites unknown persona {pid!r}")
            base = branches[by_id[pair.base_persona_id].branch_id]
            flipped = branches[by_id[pair.flipped_persona_id].branch_id]
            derived = base.outcome.strip().casefold() != flipped.outcome.strip().casefold()
            if derived is not pair.gold_changes:
                raise ValueError(
                    f"{self.instance_id!r}: pair ({pair.base_persona_id}, {pair.flipped_persona_id}) "
                    f"says gold_changes={pair.gold_changes}, but the two personas' branches "
                    f"{'differ' if derived else 'agree'} in outcome"
                )
        return self

    @model_validator(mode="after")
    def _check_branches(self) -> GoldInstance:
        if not self.gold_branches:
            return self

        ids = [b.branch_id for b in self.gold_branches]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.instance_id!r}: duplicate branch_ids {ids}")

        # AT MOST one default, not exactly one.
        #
        # "Exactly one" encodes the refinement shape -- an unconditioned general
        # rule plus a carve-out nested inside it -- and silently forces every
        # other relation into that shape. For an OPPOSED or DISJOINT pair
        # neither branch is the default: "Reward account holders get lounge
        # access" against "accounts opened from January 2024 do not" has two
        # conditioned branches whose scopes overlap without nesting.
        #
        # Forcing one of them to be the default is not a cosmetic compromise.
        # A default branch's applicability is the whole case space, so the other
        # branch necessarily becomes a SUBSET of it -- which is the definition of
        # refinement. The instance then carries a relation label of 'opposed'
        # and a branch structure that says 'refinement', and the two contradict
        # each other. Four cases in the WP1 probe set were encoded that way, and
        # they were precisely the four meant to test whether a system can tell
        # refinement from the other three relations.
        defaults = [b for b in self.gold_branches if b.is_default]
        if len(defaults) > 1:
            raise ValueError(
                f"{self.instance_id!r}: at most one default branch, found {len(defaults)}"
            )
        if not defaults and self.gold_scope_relation in (
            ScopeRelation.REFINEMENT, ScopeRelation.REDUNDANT
        ):
            raise ValueError(
                f"{self.instance_id!r}: a {self.gold_scope_relation.value} instance is "
                "defined by an exception sitting inside a general rule, so it needs "
                "that general rule as a default branch"
            )

        known = {p.id for p in self.passages}
        for b in self.gold_branches:
            if not b.is_default and b.supporting_passage is None:
                raise ValueError(
                    f"{self.instance_id!r}: exception branch {b.branch_id!r} has no "
                    "supporting_passage. An ungrounded exception is exactly what the "
                    "grounding gate exists to reject, so it cannot be gold."
                )
            if b.supporting_passage is not None and b.supporting_passage not in known:
                raise ValueError(
                    f"{self.instance_id!r}: branch {b.branch_id!r} cites unknown passage "
                    f"{b.supporting_passage!r}"
                )

        # A conditional instance normally resolves to a default plus at least one
        # exception. REDUNDANT is the deliberate exception to that rule: the two
        # passages overlap but agree on outcome, so the correct resolved structure
        # is exactly ONE merged branch. Recording a second branch there would
        # assert an exception that does not exist -- which is the precise failure
        # (a fabricated branch inflating Preservation Rate) that separating
        # 'redundant' from 'refinement' exists to prevent.
        if self.gold_conflict_type is ConflictType.CONDITIONAL:
            if self.gold_scope_relation is ScopeRelation.REDUNDANT:
                if len(self.gold_branches) != 1:
                    raise ValueError(
                        f"{self.instance_id!r}: a redundant instance must resolve to exactly one "
                        f"merged branch, got {len(self.gold_branches)}"
                    )
            elif len(self.gold_branches) < 2:
                raise ValueError(
                    f"{self.instance_id!r}: a conditional instance needs a default branch plus at "
                    "least one exception; got only the default"
                )
        return self

    # -- projection ---------------------------------------------------------- #

    def to_query_record(self, *, include_gold_pairs: bool = True) -> QueryRecord:
        """Project down to the wire contract.

        With ``include_gold_pairs=True`` this is an ORACLE record: it carries
        the gold labels in the ``conflict_pairs`` field. That is what Member B
        builds against before A's detector exists, and what A's own modules are
        scored against.

        With ``include_gold_pairs=False`` the pairs are empty -- the input side
        only, for feeding a real detector.
        """
        from contract.models import ConflictPair, DecidedBy

        pairs: list[ConflictPair] = []
        if include_gold_pairs and len(self.passages) >= 2:
            # The gold label describes the relationship the instance was built
            # around, which is the first two passages. Remaining passages are
            # retrieval distractors and carry no gold pair label.
            pairs.append(
                ConflictPair(
                    doc_i=self.passages[0].id,
                    doc_j=self.passages[1].id,
                    is_conflict=self.gold_conflict_type is not ConflictType.NO_CONFLICT,
                    type=self.gold_conflict_type,
                    scope_relation=self.gold_scope_relation,
                    opposition=self.gold_opposition,
                    confidence=1.0,
                    decided_by=DecidedBy.GOLD,
                )
            )

        return QueryRecord(
            contract_version=self.contract_version,
            query_id=self.instance_id,
            query=self.query,
            domain=self.domain,
            construction=self.construction,
            separation=self.separation,
            passages=list(self.passages),
            conflict_pairs=pairs,
            multi_exception_flags=self.gold_multi_exception_flags,
        )

    def branch_for_persona(self, persona_id: str) -> Branch:
        """The gold branch governing a persona -- the source of the gold outcome."""
        branches = {b.branch_id: b for b in self.gold_branches}
        for p in self.personas:
            if p.persona_id == persona_id:
                return branches[p.branch_id]
        raise KeyError(f"no persona {persona_id!r} in instance {self.instance_id!r}")

    @property
    def exception_branches(self) -> list[Branch]:
        return [b for b in self.gold_branches if not b.is_default]

    @property
    def default_branch(self) -> Branch | None:
        for b in self.gold_branches:
            if b.is_default:
                return b
        return None


# Resolve the forward reference in Applicability.attributes.
Applicability.model_rebuild()
