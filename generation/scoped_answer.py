"""B3: conflict-aware generation.

Verbalise a resolved branch structure into a scoped answer with per-branch
attribution:

    "The fee applies by default, but is waived for premium-tier cardholders
    (per p1)."

**The one rule: the model explains the resolved structure, it does not decide
which branch is true.** Truth was settled upstream, by A4's scope relation and
B2's composition. A generator given room to adjudicate will adjudicate -- it
will notice that two branches disagree, pick the one that sounds more
authoritative, and quietly undo the branch preservation the whole pipeline
exists to achieve. The failure would be invisible in answer-correctness scoring,
which is the same blind spot the project is about, reappearing at the last step.

Three things enforce that here:

* The prompt states it, and gives the branches as settled facts rather than as
  evidence to weigh.
* :func:`render_template` produces the answer with no model at all. It is the
  default, it is free, and on a structure this constrained it is often as good
  as the model -- the model earns its place only where fluency matters.
* :func:`check_faithfulness` verifies afterwards that every branch actually
  reached the answer, because a rule stated in a prompt is a request and a
  rule checked in code is a constraint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from api_budget.client import LLMClient, Tier, get_client
from composition.operator import ComposedAnswer, Resolution
from contract.gold import Branch
from metrics.branch_match import outcomes_match

#: Answers are assembled as clauses joined by semicolons, and each branch
#: occupies one. Splitting on sentence and clause boundaries is what lets a
#: branch be matched against the part of the answer that claims to state it,
#: rather than against a paragraph where its negation may live elsewhere.
#:
#: The digit guards matter: a plain ``[;.]`` split tears "a 2.75% fee applies"
#: into "a 2" and "75% fee applies", so the branch matched neither fragment and
#: every answer quoting a decimal was reported as having dropped its own
#: branches.
_CLAUSE_RE = re.compile(r";\s*|(?<!\d)\.(?!\d)\s*")

_SYSTEM = """You verbalise an already-resolved answer structure. You do NOT decide what is true.

You are given branches. Each is TRUE within its own stated scope. They do not
compete; they apply to different cases.

Rules:
- State every branch. Never omit one, and never present one as overriding
  another unless the input says so.
- Attach each branch to the cases it applies to.
- Cite the supporting passage id in parentheses after each branch.
- Do not add conditions, figures, or caveats that are not in the input.
- Do not say which source is more reliable. That is not the question.
- Two or three sentences. Plain language.

Write only the answer."""


@dataclass
class GeneratedAnswer:
    query_id: str
    text: str
    method: str
    """``"template"`` or ``"llm"``. Recorded because the two are different
    systems and their numbers are never pooled."""

    branches_cited: int = 0
    api_calls: int = 0
    faithfulness: FaithfulnessReport | None = None

    @property
    def is_faithful(self) -> bool:
        return self.faithfulness is None or self.faithfulness.ok


@dataclass
class FaithfulnessReport:
    """Whether the answer says what the structure says -- no more, no less."""

    missing_branches: list[str] = field(default_factory=list)
    """Branches absent from the answer. This is suppression happening at the
    generation step, after composition did its job correctly."""

    missing_attribution: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    """Figures in the answer that appear in no branch. The cheapest detectable
    form of invention, and the most damaging in a rule-governed domain."""

    adjudicated: bool = False
    """The answer ranked the sources instead of scoping them."""

    @property
    def ok(self) -> bool:
        return not (self.missing_branches or self.unsupported_numbers or self.adjudicated)

    def render(self) -> str:
        if self.ok:
            return "  faithful: every branch present, nothing invented"
        lines = ["  UNFAITHFUL"]
        if self.missing_branches:
            lines.append(f"    branches dropped in generation: {self.missing_branches}")
        if self.unsupported_numbers:
            lines.append(f"    figures in no branch: {self.unsupported_numbers}")
        if self.adjudicated:
            lines.append("    the answer ranked sources rather than scoping them; "
                         "generation adjudicated, which is upstream's job")
        if self.missing_attribution:
            lines.append(f"    uncited branches: {self.missing_attribution}")
        return "\n".join(lines)


#: Phrases that indicate the generator weighed the sources rather than scoping
#: them. Checked as a guard, not as a style preference: credibility language in
#: a composed answer means a branch was ranked, and ranking is selection.
_ADJUDICATION_CUES = (
    "more reliable", "more authoritative", "more credible", "less reliable",
    "should be trusted", "takes precedence", "outweighs", "more up to date",
    "the correct answer is", "we recommend relying",
)

#: The lookbehind is load-bearing. Without it this matches the digit inside a
#: passage id, so "(per p0)" reported a figure the branches did not contain and
#: every correctly-attributed answer was flagged as having invented a number --
#: including answers from the template renderer, which cannot invent anything.
#: That the deterministic path failed the check is what exposed it.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%?")


def render_template(answer: ComposedAnswer) -> str:
    """Deterministic verbalisation. No model, no cost, no room to adjudicate.

    The default for a reason beyond cost: it cannot drop a branch or invent a
    figure, so it is the reference the LLM path is judged against. If the model
    does not beat this on fluency by enough to matter, it is not worth its
    tokens or its risk.
    """
    branches = answer.branches
    if not branches:
        return "No answer could be resolved from the retrieved passages."

    if answer.resolution is Resolution.PASS_THROUGH:
        # State the rule rather than announcing that the sources agree. "There
        # is no conflict to resolve" is a non-answer, and it made every
        # pass-through instance fail the faithfulness check for stating none of
        # its own branches -- correctly, since it stated nothing at all.
        b = branches[0]
        cite = f" (per {b.supporting_passage})" if b.supporting_passage else ""
        return f"{_sentence(b.outcome)}{cite}."

    if answer.resolution is Resolution.SELECTED:
        b = branches[0]
        flag = ""
        if answer.flags.nested or answer.flags.crossed:
            which = "nested" if answer.flags.nested else "crossed"
            flag = (f" The exceptions here interact ({which}), which is outside "
                    "what this system resolves, so a single source was used.")
        cite = f" (per {b.supporting_passage})" if b.supporting_passage else ""
        return f"{_sentence(b.outcome)}{cite}.{flag}"

    default = answer.default_branch()
    exceptions = answer.exceptions()
    parts: list[str] = []

    if default is not None:
        cite = f" (per {default.supporting_passage})" if default.supporting_passage else ""
        parts.append(f"By default, {_clause(default.outcome)}{cite}")
    for b in exceptions:
        cite = f" (per {b.supporting_passage})" if b.supporting_passage else ""
        scope = b.applicability.descriptor or b.condition or "certain cases"
        parts.append(f"for {scope}, {_clause(b.outcome)}{cite}")

    if not parts:
        return "No answer could be resolved from the retrieved passages."
    if len(parts) == 1:
        return _sentence(parts[0]) + "."
    return _sentence(parts[0]) + "; " + "; ".join(parts[1:]) + "."


def _clause(text: str) -> str:
    return (text or "").strip().rstrip(".")


def _sentence(text: str) -> str:
    t = _clause(text)
    return t[:1].upper() + t[1:] if t else t


def check_faithfulness(text: str, branches: list[Branch]) -> FaithfulnessReport:
    """Verify the answer says what the structure says.

    Not a style check. Each finding is a specific failure:

    * a missing branch is suppression occurring *after* composition preserved
      it -- the pipeline's own failure mode, at the last step;
    * an unsupported figure is invention, and in a rule-governed domain a wrong
      number is the most harmful thing the system can emit;
    * adjudication language means generation ranked the sources, which is
      selection under another name.
    """
    report = FaithfulnessReport()
    low = (text or "").lower()
    clauses = [c for c in _CLAUSE_RE.split(text or "") if c.strip()]

    for b in branches:
        # Matched clause by clause through outcomes_match rather than by
        # containment against the whole answer.
        #
        # Containment against the whole text cannot see negation. "No fee
        # applies" is 2/3 contained in "A 3% fee applies (per p0)" -- they share
        # "fee" and "applies" -- so an answer stating the exact opposite of a
        # branch was scored as having stated the branch. outcomes_match carries
        # the polarity and numeric vetoes; the clause split is what lets it
        # apply, since polarity parity over a whole multi-branch answer is
        # meaningless.
        if not any(outcomes_match(b.outcome, c) for c in clauses):
            report.missing_branches.append(b.branch_id)
        elif b.supporting_passage and b.supporting_passage.lower() not in low:
            report.missing_attribution.append(b.branch_id)

    allowed = {n for b in branches for n in _NUM_RE.findall(b.outcome or "")}
    allowed |= {n for b in branches
                for n in _NUM_RE.findall(b.applicability.descriptor or "")}
    allowed |= {n for b in branches for n in _NUM_RE.findall(b.condition or "")}
    for n in _NUM_RE.findall(text or ""):
        if n not in allowed:
            report.unsupported_numbers.append(n)

    report.adjudicated = any(cue in low for cue in _ADJUDICATION_CUES)
    return report


@dataclass
class GenerationStats:
    instances: int = 0
    llm_calls: int = 0
    unfaithful: int = 0
    dropped_branches: int = 0
    invented_numbers: int = 0
    adjudications: int = 0

    def merge(self, a: GeneratedAnswer) -> None:
        self.instances += 1
        self.llm_calls += a.api_calls
        f = a.faithfulness
        if f and not f.ok:
            self.unfaithful += 1
            self.dropped_branches += len(f.missing_branches)
            self.invented_numbers += len(f.unsupported_numbers)
            self.adjudications += int(f.adjudicated)

    def render(self) -> str:
        lines = [
            "Conflict-aware generation (B3)",
            "-" * 62,
            f"  instances            {self.instances:>8,}",
            f"  unfaithful answers   {self.unfaithful:>8,}",
            f"    branches dropped   {self.dropped_branches:>8,}",
            f"    figures invented   {self.invented_numbers:>8,}",
            f"    adjudications      {self.adjudications:>8,}",
            f"  API calls            {self.llm_calls:>8,}",
        ]
        if self.adjudications:
            lines += [
                "",
                "  The generator ranked sources instead of scoping them. That is the",
                "  pipeline's own failure mode reappearing at the last step, and it",
                "  is invisible to answer-correctness scoring. Tighten the prompt or",
                "  fall back to the template renderer.",
            ]
        return "\n".join(lines)


class ScopedAnswerGenerator:
    """B3 end to end."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        tier: Tier = Tier.JUDGE,
        use_llm: bool = False,
        fall_back_on_unfaithful: bool = True,
    ):
        """
        Parameters
        ----------
        use_llm
            Off by default. The template renderer cannot drop a branch or
            invent a figure; the model can do both. Turn it on when fluency is
            being measured, and report the two as separate systems.
        fall_back_on_unfaithful
            When the model drops a branch or invents a figure, emit the
            template answer instead and record it. Preserving a branch matters
            more than reading well, and a silent unfaithful answer is exactly
            the failure the metric suite exists to catch.
        """
        self.client = client or (get_client() if use_llm else None)
        self.tier = tier
        self.use_llm = use_llm
        self.fall_back_on_unfaithful = fall_back_on_unfaithful

    def generate(self, answer: ComposedAnswer, query: str = "") -> GeneratedAnswer:
        template = render_template(answer)
        if not self.use_llm:
            return GeneratedAnswer(
                answer.query_id, template, "template",
                branches_cited=answer.n_branches,
                faithfulness=check_faithfulness(template, answer.branches))

        result = self.client.complete(
            messages=[{"role": "user", "content":
                       f"Question: {query}\n\n"
                       f"Resolved branches:\n{self._describe(answer)}\n\n"
                       "Write the answer."}],
            system=_SYSTEM,
            tier=self.tier,
            step="b3_generation",
            temperature=0.0,
        )
        text = (result.text or "").strip()
        report = check_faithfulness(text, answer.branches)

        if not report.ok and self.fall_back_on_unfaithful:
            return GeneratedAnswer(
                answer.query_id, template, "template_after_unfaithful_llm",
                branches_cited=answer.n_branches, api_calls=1,
                faithfulness=check_faithfulness(template, answer.branches))

        return GeneratedAnswer(answer.query_id, text, "llm",
                               branches_cited=answer.n_branches, api_calls=1,
                               faithfulness=report)

    @staticmethod
    def _describe(answer: ComposedAnswer) -> str:
        lines = []
        for b in answer.branches:
            scope = "all cases" if b.is_default else (
                b.applicability.descriptor or b.condition or "certain cases")
            lines.append(f"- applies to: {scope}\n"
                         f"  outcome: {b.outcome}\n"
                         f"  source: {b.supporting_passage or 'unattributed'}")
        return "\n".join(lines)

    def generate_all(
        self, answers: list[ComposedAnswer], queries: dict[str, str] | None = None
    ) -> tuple[list[GeneratedAnswer], GenerationStats]:
        queries = queries or {}
        stats = GenerationStats()
        out: list[GeneratedAnswer] = []
        for a in answers:
            g = self.generate(a, queries.get(a.query_id, ""))
            out.append(g)
            stats.merge(g)
        return out, stats
