"""The routing table, as executable code.

Written down once, here, rather than reimplemented on each side of the
boundary. Member B's composition operator calls :func:`route` on Member A's
output; Member A's tests call the same function to check that a produced
record routes the way the design intends.

The routing rule itself is the contribution in miniature: it is the point in
the pipeline where prior work unconditionally calls ``Selection`` and this
project sometimes calls ``Compose`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contract.models import ConflictPair, ConflictType, MultiExceptionFlags, ScopeRelation


class Action(str, Enum):
    """What Member B's resolver does with a pair."""

    COMPOSE = "compose"
    """Assemble a scoped answer preserving every branch. The new operator."""

    MERGE = "merge"
    """Two passages say the same thing where they overlap. One branch, not two."""

    SELECT = "select"
    """Genuine contradiction. Fall back to credibility selection, exactly as
    prior work does, and report it as a selection."""

    PRIOR_WORK = "prior_work"
    """Factual / temporal / opinion. Apply the matching published strategy;
    this project makes no claim about these classes."""

    PASS_THROUGH = "pass_through"
    """No conflict. Nothing to resolve."""


@dataclass(frozen=True)
class RoutingDecision:
    action: Action
    reason: str
    flag: str | None = None
    """Set to 'nested' or 'crossed' when selection was forced by second-order
    structure rather than by a first-order contradiction. Reported separately,
    because the two mean different things about the data."""


def route(pair: ConflictPair) -> RoutingDecision:
    """Route one pair on Member A's first-order judgement.

    This handles everything decidable from a (default, exception) pair alone.
    Second-order structure -- two exceptions that interact with each other --
    needs :func:`route_with_flags`, because it requires comparing extracted
    branches to each other, which only Member B can do.
    """
    if not pair.is_conflict or pair.type is ConflictType.NO_CONFLICT:
        return RoutingDecision(Action.PASS_THROUGH, "no conflict detected")

    if pair.type is not ConflictType.CONDITIONAL:
        return RoutingDecision(
            Action.PRIOR_WORK,
            f"{pair.type.value} conflict -- handled by the published strategy for that class",
        )

    match pair.scope_relation:
        case ScopeRelation.REFINEMENT:
            return RoutingDecision(
                Action.COMPOSE,
                "a valid exception narrows the default's scope and changes the outcome there; "
                "both branches are true and both are preserved",
            )
        case ScopeRelation.DISJOINT:
            return RoutingDecision(
                Action.COMPOSE,
                "applicability sets do not intersect, so no case falls under both; compose trivially",
            )
        case ScopeRelation.REDUNDANT:
            return RoutingDecision(
                Action.MERGE,
                "the branches agree on outcome throughout their overlap -- a restatement, "
                "not a conflict; composing them as two branches would be wrong",
            )
        case ScopeRelation.OPPOSED:
            return RoutingDecision(
                Action.SELECT,
                "applicability overlaps, neither set contains the other, and the outcomes "
                "disagree on the overlap -- a real contradiction; selection is the honest answer",
            )
        case None:
            raise ValueError(
                f"conditional pair {pair.key} has no scope_relation; "
                "the contract requires one and routing cannot proceed without it"
            )

    raise AssertionError(f"unhandled scope_relation {pair.scope_relation!r}")


def route_with_flags(pair: ConflictPair, flags: MultiExceptionFlags) -> RoutingDecision:
    """Route including Member B's second-order flags.

    The project scope is first-order conditional conflicts. When two or more
    surviving exception branches interact -- one nested inside another, or two
    that co-occur and disagree -- the instance is routed to selection with the
    specific pattern recorded, rather than silently mis-composed.

    Flagging is deliberately not treated as a failure. Reporting how often real
    data contains each pattern is a result in its own right: nested structure
    is the recursive case that Span-Grounded Deontic Trees already handles in
    the single-document setting, and crossed structure is a different pattern
    that recursive framing does not directly name.
    """
    base = route(pair)

    if base.action is not Action.COMPOSE:
        return base

    if flags.nested:
        return RoutingDecision(
            Action.SELECT,
            "an exception is itself carved back by a further exception (second-order "
            "structure, outside this project's first-order scope); flagged, not mis-composed",
            flag="nested",
        )
    if flags.crossed:
        return RoutingDecision(
            Action.SELECT,
            "two independent exceptions can co-occur and disagree with each other; "
            "flagged, not mis-composed",
            flag="crossed",
        )
    return base


# Static table, for documentation, tests, and the zeroth-review slide.
ROUTING_TABLE: dict[tuple[str, str | None], Action] = {
    (ConflictType.CONDITIONAL.value, ScopeRelation.REFINEMENT.value): Action.COMPOSE,
    (ConflictType.CONDITIONAL.value, ScopeRelation.DISJOINT.value): Action.COMPOSE,
    (ConflictType.CONDITIONAL.value, ScopeRelation.REDUNDANT.value): Action.MERGE,
    (ConflictType.CONDITIONAL.value, ScopeRelation.OPPOSED.value): Action.SELECT,
    (ConflictType.FACTUAL.value, None): Action.PRIOR_WORK,
    (ConflictType.TEMPORAL.value, None): Action.PRIOR_WORK,
    (ConflictType.OPINION.value, None): Action.PRIOR_WORK,
    (ConflictType.NO_CONFLICT.value, None): Action.PASS_THROUGH,
}
