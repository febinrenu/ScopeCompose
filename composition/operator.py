"""B2: the composition operator.

This is the step that replaces selection. Prior conflict-aware RAG reaches this
point, ranks the sources, and emits the winner. This assembles a structure that
keeps every branch that is true, each attached to the scope it is true under.

The algorithm, per the module spec:

1. Route each pair on A4's four-way relation (``contract.routing``).
2. Any ``opposed`` pair -> selection, reported as a selection.
3. ``redundant`` -> merge into the default. **Not** a second branch: a
   restatement recorded as an exception is a fabricated branch, and it inflates
   Preservation Rate with something that was never there.
4. For every remaining pair of exception branches, compare their applicability:
   nested -> flag ``nested``; overlapping with disagreeing outcomes -> flag
   ``crossed``; overlapping and agreeing -> merge.
5. Any flag -> selection, with the flag recorded.
6. Otherwise compose.

**Why flagging rather than composing.** The project scope is first-order
conflicts: a default and its direct exceptions. Second-order structure -- an
exception carved back by a further exception, or two exceptions that co-occur
and disagree -- is detected and reported, never composed recursively and never
silently picked between. Reporting how often real data contains each pattern is
a result; guessing at it is not.

**Why two flags rather than one.** ``nested`` is the recursive case that
span-grounded deontic trees already handle in the single-document setting;
``crossed`` is a different pattern that a recursive-tree framing does not
directly name. Collapsing them into one "complex" flag discards the distinction
that makes the future-work argument concrete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from contract.gold import Applicability, Branch
from contract.models import ConflictPair, MultiExceptionFlags, QueryRecord
from contract.routing import Action, RoutingDecision, route
from scope.attributes import SetRelation, compare


class Resolution(str, Enum):
    """What the operator did with an instance."""

    COMPOSED = "composed"
    MERGED = "merged"
    SELECTED = "selected"
    PASS_THROUGH = "pass_through"
    PRIOR_WORK = "prior_work"


#: Outcome-similarity above which two exception branches covering the same
#: cases are treated as saying the same thing. Reuses the alignment metric's
#: threshold so the operator and the scorer cannot drift apart on what "same
#: outcome" means.
from metrics.branch_match import outcomes_match  # noqa: E402


@dataclass
class ComposedAnswer:
    """The resolved branch structure for one instance."""

    query_id: str
    resolution: Resolution
    branches: list[Branch] = field(default_factory=list)
    flags: MultiExceptionFlags = field(default_factory=MultiExceptionFlags)
    reason: str = ""
    selected_passage: str | None = None
    """Set when the operator fell back to selection, so the write-up can say
    which branch survived and why."""

    merged_pairs: int = 0
    """How many redundant restatements were folded into the default rather than
    recorded as separate branches."""

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    @property
    def is_composition(self) -> bool:
        return self.resolution is Resolution.COMPOSED

    def default_branch(self) -> Branch | None:
        return next((b for b in self.branches if b.is_default), None)

    def exceptions(self) -> list[Branch]:
        return [b for b in self.branches if not b.is_default]


@dataclass
class CompositionStats:
    instances: int = 0
    composed: int = 0
    merged: int = 0
    selected: int = 0
    pass_through: int = 0
    prior_work: int = 0

    nested_flagged: int = 0
    crossed_flagged: int = 0
    redundant_merges: int = 0
    branches_preserved: int = 0

    def merge(self, a: ComposedAnswer) -> None:
        self.instances += 1
        self.branches_preserved += a.n_branches
        self.redundant_merges += a.merged_pairs
        self.nested_flagged += int(a.flags.nested)
        self.crossed_flagged += int(a.flags.crossed)
        match a.resolution:
            case Resolution.COMPOSED:
                self.composed += 1
            case Resolution.MERGED:
                self.merged += 1
            case Resolution.SELECTED:
                self.selected += 1
            case Resolution.PASS_THROUGH:
                self.pass_through += 1
            case Resolution.PRIOR_WORK:
                self.prior_work += 1

    @property
    def composition_rate(self) -> float:
        return self.composed / self.instances if self.instances else 0.0

    @property
    def nested_rate(self) -> float:
        return self.nested_flagged / self.instances if self.instances else 0.0

    @property
    def crossed_rate(self) -> float:
        return self.crossed_flagged / self.instances if self.instances else 0.0

    def render(self) -> str:
        return "\n".join([
            "Composition operator (B2)",
            "-" * 62,
            f"  instances            {self.instances:>8,}",
            f"  composed             {self.composed:>8,}   ({self.composition_rate:.1%})",
            f"  merged (redundant)   {self.merged:>8,}",
            f"  selection fallback   {self.selected:>8,}",
            f"  prior-work classes   {self.prior_work:>8,}",
            f"  no conflict          {self.pass_through:>8,}",
            "",
            "  second-order structure (reported separately, never summed)",
            f"    nested flagged     {self.nested_flagged:>8,}   ({self.nested_rate:.1%})",
            f"    crossed flagged    {self.crossed_flagged:>8,}   ({self.crossed_rate:.1%})",
            "",
            f"  redundant restatements merged, not branched  {self.redundant_merges:>6,}",
            f"  branches preserved   {self.branches_preserved:>8,}",
        ])


def classify_exception_pair(a: Branch, b: Branch) -> tuple[str | None, str]:
    """Compare two exception branches. Returns ``(flag, reason)``.

    ``flag`` is ``"nested"``, ``"crossed"``, or ``None`` when the two can
    coexist without second-order structure.

    Both determinations are made separately -- do the scopes interact, and do
    the outcomes agree where they do -- for the same reason A4 separates them.
    Collapsing "these overlap" into "these conflict" is the error the four-way
    relation exists to prevent, and it is just as wrong one level down.
    """
    rel = compare(a.applicability, b.applicability)

    if rel is SetRelation.DISJOINT:
        return None, "exception scopes do not intersect; both stand"

    if rel in (SetRelation.SUBSET, SetRelation.SUPERSET):
        # One exception's cases sit entirely inside the other's, so a case in
        # the inner set is governed by both. Which wins is exception-to-
        # exception precedence: second-order, out of scope, flagged.
        if outcomes_match(a.outcome, b.outcome):
            return None, "nested scopes but the same outcome; no precedence question arises"
        return "nested", (
            "one exception's applicability is contained in the other's and they "
            "disagree there, so resolving it needs exception-to-exception "
            "precedence -- second-order structure"
        )

    if rel is SetRelation.EQUAL:
        if outcomes_match(a.outcome, b.outcome):
            return None, "same scope, same outcome; a restatement"
        return "crossed", "identical scope with disagreeing outcomes"

    if rel is SetRelation.OVERLAPPING:
        if outcomes_match(a.outcome, b.outcome):
            return None, "overlapping scopes agreeing on the overlap; compatible"
        return "crossed", (
            "two independent exceptions can both apply to one case and disagree "
            "there, with neither taking precedence by scope"
        )

    # UNKNOWN: the typed attributes could not decide it. Treated as compatible
    # rather than flagged -- flagging on ignorance would inflate the nested and
    # crossed rates, which are reported as findings about the data.
    return None, "scope relation undecidable from typed attributes; not flagged"


class CompositionOperator:
    """B2 end to end."""

    def __init__(self, *, credibility_order: list[str] | None = None):
        """
        Parameters
        ----------
        credibility_order
            Source types, most credible first, used only on the selection
            fallback. Selection is prior work's operator; this project uses it
            where composition is not the honest answer, and reports it as such.
        """
        self.credibility_order = credibility_order or [
            "official_policy", "government_guidance", "product_terms",
            "news", "third_party", "user_generated",
        ]

    def compose(
        self,
        record: QueryRecord,
        branches: list[Branch],
        *,
        pairs: list[ConflictPair] | None = None,
    ) -> ComposedAnswer:
        """Resolve one instance into a branch structure.

        ``branches`` is B1's output (plus the default). ``pairs`` defaults to
        the record's own conflict pairs, which is A4's judgement.
        """
        pairs = record.conflict_pairs if pairs is None else pairs
        conflicting = [p for p in pairs if p.is_conflict]

        if not conflicting:
            # The default branch alone, not everything that was handed in.
            #
            # No conflict detected means the system is going to answer with the
            # general rule and say nothing about any exception. Carrying every
            # branch through here would credit the instance with preserving
            # branches the answer never states -- and this is exactly the case
            # where that matters most, because a *missed* conditional conflict
            # is the suppression the project exists to measure. Inflating PR
            # here would hide the detector's own failures behind the scorer.
            default = next((b for b in branches if b.is_default), None)
            kept = [default] if default is not None else branches[:1]
            return ComposedAnswer(
                record.query_id, Resolution.PASS_THROUGH, kept,
                reason="no conflict detected; the general rule is answered alone")

        decisions = [(p, route(p)) for p in conflicting]

        # Step 2: any opposed pair settles the instance. A genuine contradiction
        # anywhere means the answer cannot be presented as uniformly true.
        for pair, d in decisions:
            if d.action is Action.SELECT:
                return self._select(record, branches, d, reason=d.reason)

        if all(d.action is Action.PRIOR_WORK for _, d in decisions):
            return ComposedAnswer(
                record.query_id, Resolution.PRIOR_WORK, branches,
                reason=decisions[0][1].reason)

        # Step 3: fold redundant restatements into the default.
        merges = sum(1 for _, d in decisions if d.action is Action.MERGE)
        kept = branches
        if merges:
            kept = self._merge_redundant(branches)
            if not any(d.action is Action.COMPOSE for _, d in decisions):
                return ComposedAnswer(
                    record.query_id, Resolution.MERGED, kept,
                    reason="every pair was a restatement; one merged branch, not several",
                    merged_pairs=merges)

        # Step 4/5: second-order structure between surviving exceptions.
        exceptions = [b for b in kept if not b.is_default]
        flags = MultiExceptionFlags()
        flag_reason = ""
        for i in range(len(exceptions)):
            for j in range(i + 1, len(exceptions)):
                flag, why = classify_exception_pair(exceptions[i], exceptions[j])
                if flag == "nested" and not flags.nested:
                    flags = MultiExceptionFlags(nested=True, crossed=flags.crossed)
                    flag_reason = why
                elif flag == "crossed" and not flags.crossed:
                    flags = MultiExceptionFlags(nested=flags.nested, crossed=True)
                    flag_reason = flag_reason or why

        if flags.nested or flags.crossed:
            name = "nested" if flags.nested else "crossed"
            # Reduce to the selected branch, exactly as the opposed path does.
            # Routing to selection while still reporting every branch would
            # credit the instance with preserving branches the answer never
            # states -- inflating Preservation Rate with the cases the operator
            # explicitly declined to resolve.
            winner = self._most_credible(record)
            survived = [b for b in kept if b.supporting_passage == winner] or kept[:1]
            return ComposedAnswer(
                record.query_id, Resolution.SELECTED, survived, flags=flags,
                reason=f"{name}: {flag_reason}",
                selected_passage=winner,
                merged_pairs=merges)

        # Step 6: compose.
        return ComposedAnswer(
            record.query_id, Resolution.COMPOSED, kept, flags=flags,
            reason="every branch is true within its own scope and all are preserved",
            merged_pairs=merges)

    # -- helpers ---------------------------------------------------------------- #

    def _merge_redundant(self, branches: list[Branch]) -> list[Branch]:
        """Drop exception branches that restate the default.

        A restatement recorded as an exception is a branch that does not exist.
        It inflates Preservation Rate -- the system is credited with preserving
        something the gold data never contained -- which is precisely why
        'redundant' is a separate relation from 'refinement'.
        """
        default = next((b for b in branches if b.is_default), None)
        if default is None:
            return branches
        return [default] + [
            b for b in branches
            if not b.is_default and not outcomes_match(default.outcome, b.outcome)
        ]

    def _select(self, record: QueryRecord, branches: list[Branch],
                decision: RoutingDecision, *, reason: str) -> ComposedAnswer:
        winner = self._most_credible(record)
        kept = [b for b in branches if b.supporting_passage == winner] or branches[:1]
        return ComposedAnswer(
            record.query_id, Resolution.SELECTED, kept,
            reason=reason, selected_passage=winner)

    def _most_credible(self, record: QueryRecord) -> str | None:
        """Prior work's operator, used only where composition is not honest."""
        def rank(p) -> tuple[int, str]:
            st = p.source_type.value
            return (self.credibility_order.index(st)
                    if st in self.credibility_order else len(self.credibility_order),
                    p.date or "")
        return min(record.passages, key=rank).id if record.passages else None

    def compose_all(
        self, records: list[QueryRecord], branches_by_id: dict[str, list[Branch]]
    ) -> tuple[list[ComposedAnswer], CompositionStats]:
        stats = CompositionStats()
        out: list[ComposedAnswer] = []
        for r in records:
            a = self.compose(r, branches_by_id.get(r.query_id, []))
            out.append(a)
            stats.merge(a)
        return out, stats
