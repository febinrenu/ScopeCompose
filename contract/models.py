"""The interface contract between Member A and Member B.

This is the single frozen agreement in the project. Member A produces
:class:`QueryRecord`; Member B consumes it. Everything else on either side can
change independently as long as this holds.

Changing anything here requires all three of the following in the SAME commit:

1. bump :data:`CONTRACT_VERSION`
2. update ``contract/mock.py`` so generated records still validate
3. regenerate the JSON Schema: ``python -m contract.export_schema``

...and tell the other member. A schema change only one person knows about is
exactly the integration drift the frozen contract exists to prevent.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONTRACT_VERSION = "1.1.0"
"""Semantic version of this schema. Stamped onto every record so drift is detectable."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ConflictType(str, Enum):
    """The five-class conflict taxonomy.

    Extends the prior three-way factual/temporal/opinion split with
    ``CONDITIONAL`` -- the class this project exists to handle.
    """

    NO_CONFLICT = "no_conflict"
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    OPINION = "opinion"
    CONDITIONAL = "conditional"


class ScopeRelation(str, Enum):
    """The four-way scope relation between two claims.

    Defined jointly over applicability *and* outcome. Deciding only
    applicability -- "do these scopes overlap?" -- cannot separate a redundant
    restatement from a real exception, which is the bug this four-way version
    replaced a two-way version to fix.

    REFINEMENT
        One applicability set is a proper subset of the other, and the branches
        are outcome-INCOMPATIBLE on the subset. A genuine exception. Compose.
    DISJOINT
        Applicability sets do not intersect. Outcome comparison is vacuous.
        Compose trivially.
    REDUNDANT
        Applicability sets overlap or nest, but the branches AGREE on outcome
        throughout the overlap. Not a conflict at all. Merge into one branch.
    OPPOSED
        Applicability sets overlap, neither contains the other, and outcomes
        disagree on the overlap. A real contradiction. Fall back to selection.
    """

    REFINEMENT = "refinement"
    DISJOINT = "disjoint"
    REDUNDANT = "redundant"
    OPPOSED = "opposed"


class Construction(str, Enum):
    """Benchmark tier. Never blend these two when reporting headline numbers.

    NATURAL
        Tier 1. The general rule and its exception were mined from genuinely
        separate, separately-retrievable documents. This is the primary
        evidence for the "naturally occurring, multi-source" claim that
        distinguishes this benchmark from ConditionalQA.
    SPLIT
        Tier 2. A single-document rule/exception pair deliberately split across
        two synthetic "documents". Legitimate and useful -- it isolates the
        resolution-operator question from the mining question -- but it is not
        evidence of natural multi-source occurrence.
    """

    NATURAL = "natural"
    SPLIT = "split"


class SourceType(str, Enum):
    """Provenance of a passage. Needed downstream for temporal and credibility
    handling, and for the credibility-selection baseline."""

    OFFICIAL_POLICY = "official_policy"
    PRODUCT_TERMS = "product_terms"
    AMENDMENT = "amendment"
    FAQ = "faq"
    CATEGORY_PAGE = "category_page"
    GOVERNMENT_GUIDANCE = "government_guidance"
    THIRD_PARTY = "third_party"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class Domain(str, Enum):
    """Benchmark domain. Kept deliberately narrow: financial terms and
    immigration eligibility keep stakes moderate and ground truth objective.
    Medical and legal are excluded to avoid overclaiming."""

    FINANCIAL_TERMS = "financial_terms"
    IMMIGRATION_ELIGIBILITY = "immigration_eligibility"
    SYNTHETIC = "synthetic"


class DecidedBy(str, Enum):
    """Which stage produced a pair's judgement.

    Recorded so the two-stage detector's cost/accuracy split can be reported,
    and so a run that silently fell back to a different backend is visible.
    """

    STAGE1_LOCAL = "stage1_local"
    STAGE2_API = "stage2_api"
    GOLD = "gold"
    MOCK = "mock"


_DATE_RE = re.compile(r"^\d{4}(-\d{2}){0,2}$")


# --------------------------------------------------------------------------- #
# Record components
# --------------------------------------------------------------------------- #


class Passage(BaseModel):
    """A single retrieved passage with the metadata downstream stages need."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Unique within a record, e.g. 'p0'.")
    text: str = Field(..., min_length=1)
    source_type: SourceType = SourceType.UNKNOWN
    date: str | None = Field(
        default=None,
        description="Publication date as YYYY, YYYY-MM, or YYYY-MM-DD. Drives temporal handling.",
    )

    # Tier-1 mining provenance. Optional, but without it a 'natural' record
    # cannot be audited, so the mining pipeline always fills these in.
    source_url: str | None = None
    document_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the source document. Two passages sharing a document_id "
            "did NOT come from separate documents -- which is what makes a "
            "construction='natural' claim checkable rather than asserted."
        ),
    )

    @field_validator("date")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is not None and not _DATE_RE.match(v):
            raise ValueError(f"date must be YYYY, YYYY-MM, or YYYY-MM-DD; got {v!r}")
        return v


class ConflictPair(BaseModel):
    """Member A's judgement about one pair of passages.

    Pairs are UNORDERED. ``doc_i``/``doc_j`` are stored in sorted order so that
    a record is byte-identical under any permutation of the input passages --
    the schema-level half of the order-invariance property. The semantic half
    (branch roles derive from the applicability relation, never from retrieval
    order) lives in ``scope/relation.py``.
    """

    model_config = ConfigDict(extra="forbid")

    doc_i: str
    doc_j: str
    is_conflict: bool
    type: ConflictType
    scope_relation: ScopeRelation | None = Field(
        default=None,
        description=(
            "Required when type == conditional. Optional for borderline factual "
            "pairs, where the scope analyser is also run. Must be absent when "
            "type == no_conflict."
        ),
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    decided_by: DecidedBy | None = None

    @model_validator(mode="before")
    @classmethod
    def _canonicalise_pair_order(cls, data):  # type: ignore[no-untyped-def]
        """Sort (doc_i, doc_j) before validation.

        Runs in ``before`` mode so it is impossible to construct a
        non-canonical pair at all -- which is what makes a record byte-identical
        under permutation of the input passages.
        """
        if isinstance(data, dict):
            i, j = data.get("doc_i"), data.get("doc_j")
            if isinstance(i, str) and isinstance(j, str) and i > j:
                data = {**data, "doc_i": j, "doc_j": i}
        return data

    @model_validator(mode="after")
    def _check_distinct(self) -> ConflictPair:
        if self.doc_i == self.doc_j:
            raise ValueError(
                f"a pair must reference two different passages; got {self.doc_i!r} twice"
            )
        return self

    @model_validator(mode="after")
    def _check_type_consistency(self) -> ConflictPair:
        if self.type is ConflictType.NO_CONFLICT:
            if self.is_conflict:
                raise ValueError("type='no_conflict' contradicts is_conflict=True")
            if self.scope_relation is not None:
                raise ValueError("a no_conflict pair must not carry a scope_relation")
        else:
            if not self.is_conflict:
                raise ValueError(
                    f"is_conflict=False requires type='no_conflict', got type={self.type.value!r}"
                )

        if self.type is ConflictType.CONDITIONAL and self.scope_relation is None:
            raise ValueError(
                "type='conditional' requires a scope_relation "
                "(refinement | disjoint | redundant | opposed) -- it is what Member B routes on"
            )

        if (
            self.scope_relation is not None
            and self.type not in (ConflictType.CONDITIONAL, ConflictType.FACTUAL)
        ):
            raise ValueError(
                f"scope_relation is only meaningful for conditional (or borderline factual) "
                f"pairs; got type={self.type.value!r}"
            )
        return self

    @property
    def key(self) -> tuple[str, str]:
        """Canonical unordered identity of this pair."""
        return (self.doc_i, self.doc_j)


class MultiExceptionFlags(BaseModel):
    """Second-order structure flags.

    Populated by Member B, not A: deciding these requires comparing extracted
    exception branches to EACH OTHER, not just to the default, which only B has.

    The project scope is first-order conditional conflicts. When either flag is
    set the instance is routed to the selection fallback with the specific flag
    recorded, rather than silently mis-composed. Reporting the two rates
    separately is what lets the paper say how often real data contains
    exception-to-exception structure versus co-occurring conflicting exceptions.
    """

    model_config = ConfigDict(extra="forbid")

    nested: bool = Field(
        default=False,
        description="An exception to an exception -- one exception's applicability nests inside another's.",
    )
    crossed: bool = Field(
        default=False,
        description="Two independent exceptions whose conditions co-occur and whose outcomes disagree.",
    )

    @property
    def any_flagged(self) -> bool:
        return self.nested or self.crossed


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


class QueryRecord(BaseModel):
    """One record per query. The unit that crosses the A/B boundary."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = CONTRACT_VERSION
    query_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    domain: Domain
    construction: Construction
    passages: list[Passage] = Field(..., min_length=1)
    conflict_pairs: list[ConflictPair] = Field(default_factory=list)
    multi_exception_flags: MultiExceptionFlags = Field(default_factory=MultiExceptionFlags)

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> QueryRecord:
        ids = [p.id for p in self.passages]
        dupes = {x for x in ids if ids.count(x) > 1}
        if dupes:
            raise ValueError(f"duplicate passage ids: {sorted(dupes)}")

        known = set(ids)
        seen: set[tuple[str, str]] = set()
        for pair in self.conflict_pairs:
            missing = {pair.doc_i, pair.doc_j} - known
            if missing:
                raise ValueError(
                    f"conflict_pair references unknown passage id(s) {sorted(missing)}; "
                    f"record has {sorted(known)}"
                )
            if pair.key in seen:
                raise ValueError(
                    f"duplicate conflict_pair for {pair.key} -- pairs are unordered, "
                    "so (p0,p1) and (p1,p0) are the same pair"
                )
            seen.add(pair.key)
        return self

    @model_validator(mode="after")
    def _check_natural_provenance(self) -> QueryRecord:
        """A 'natural' record claims its passages came from separate documents.

        Where document_ids are present, check that claim instead of trusting it.
        This is the schema-level guard on the Tier-1 yield risk: it makes a
        mislabelled Tier-2 instance fail loudly rather than quietly inflate the
        headline naturally-occurring count.
        """
        if self.construction is not Construction.NATURAL:
            return self
        doc_ids = [p.document_id for p in self.passages if p.document_id is not None]
        if len(doc_ids) >= 2 and len(set(doc_ids)) == 1:
            raise ValueError(
                f"construction='natural' but all passages share document_id "
                f"{doc_ids[0]!r} -- that is a Tier-2 (split) instance, not Tier 1"
            )
        return self

    # -- convenience accessors used across both members' code ---------------- #

    def passage(self, passage_id: str) -> Passage:
        for p in self.passages:
            if p.id == passage_id:
                return p
        raise KeyError(f"no passage {passage_id!r} in record {self.query_id!r}")

    def conflicting_pairs(self) -> list[ConflictPair]:
        return [p for p in self.conflict_pairs if p.is_conflict]

    def conditional_pairs(self) -> list[ConflictPair]:
        return [p for p in self.conflict_pairs if p.type is ConflictType.CONDITIONAL]

    def pair_for(self, a: str, b: str) -> ConflictPair | None:
        key = (a, b) if a < b else (b, a)
        for p in self.conflict_pairs:
            if p.key == key:
                return p
        return None


# --------------------------------------------------------------------------- #
# Version compatibility
# --------------------------------------------------------------------------- #


class ContractVersionError(RuntimeError):
    """Raised when a record was written against an incompatible schema version."""


def check_version(record_version: str, *, strict: bool = False) -> None:
    """Validate a record's ``contract_version`` against this module's.

    Same major version is accepted by default, because additive minor changes
    are backwards compatible. ``strict=True`` demands an exact match -- use it
    when reproducing reported numbers, where a silent schema difference would
    be a correctness problem rather than an inconvenience.
    """
    if record_version == CONTRACT_VERSION:
        return
    if strict:
        raise ContractVersionError(
            f"record is contract v{record_version}, this code is v{CONTRACT_VERSION} (strict mode)"
        )
    major_r = record_version.split(".", 1)[0]
    major_c = CONTRACT_VERSION.split(".", 1)[0]
    if major_r != major_c:
        raise ContractVersionError(
            f"incompatible contract major version: record v{record_version} "
            f"vs code v{CONTRACT_VERSION}. Regenerate the data or check out the matching commit."
        )
