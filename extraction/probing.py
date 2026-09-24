"""B1: Contrastive Scope Probing, with the grounding gate.

The problem this solves. An explicit condition ("for premium-tier
cardholders...") reduces to semantic parsing. An **implicit** one does not: a
passage can narrow a rule's applicability without saying that it is doing so,
and there is nothing in its surface form to parse.

    p0: "International transactions incur a 3% fee."
    p1: "Our Premier account includes fee-free spending worldwide."

Nothing in p1 says "except" or "unless". The condition -- *that the account is
a Premier account* -- has to be recovered by asking what would have to be true
of a case for p1 to hold of it rather than p0.

That is what the probe does. Rather than asking a model "what is the condition
here?", which invites it to invent a plausible one, it asks a **contrastive**
question: given that the rule says X, under what circumstances does this other
passage say something different? A question with a presupposition is harder to
answer vacuously than an open one.

**The grounding gate is the load-bearing part.** A probe that proposes
conditions is a hallucination engine unless something independent checks that
each proposal is actually supported by the passage it claims to come from. The
gate scores ``passage |= (condition AND outcome)`` with a local NLI
cross-encoder and drops anything below threshold. It is never disabled outside
its own ablation, and the rejection rate is reported -- a gate that rejects
nothing is not a gate, and a gate that rejects everything has a broken
threshold rather than a clean corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from api_budget.client import LLMClient, Tier, get_client
from contract.gold import Applicability, Branch, Explicitness
from contract.models import Passage, QueryRecord
from detection.nli import NLIScorer, get_nli

#: Entailment score below which a proposed condition is treated as ungrounded.
#: Deliberately not near-1.0: the hypothesis is a conjunction of a condition and
#: an outcome, phrased by a model rather than lifted from the passage, so even a
#: well-grounded proposal rarely scores above 0.8. Tuned on the WP1 pilot --
#: until then this is a PLACEHOLDER, flagged the same way the detector's
#: escalation band is.
TAU_GROUND = 0.5

#: The gate is meant to bite. If it rejects nothing across a run, it is not
#: doing its job and the number it produces is decorative.
SUSPICIOUS_REJECTION_RATE = 0.0

_QUESTION_SYSTEM = """You write contrastive probe questions for a legal/policy analyst.

Given a general rule, write questions of the form "under what circumstances
would <the opposite outcome> apply instead?".

Rules:
- Each question must presuppose that the general rule holds by default.
- Ask about CIRCUMSTANCES (who, what product, what timing, what amount), never
  about whether the rule is correct.
- Do not invent specific conditions. Ask what the circumstances are.

Return JSON: {"questions": ["...", "..."]}"""

_CANDIDATE_SYSTEM = """You recover the applicability condition a passage implies.

You are given a question about when a general rule does NOT apply, and one
passage. Decide whether the passage answers it.

Rules:
- Use ONLY what the passage states. Do not use outside knowledge.
- If the passage does not narrow the rule's applicability, return null. Most
  passages do not. Returning null is the correct and common answer.
- The condition must be a property of a CASE (a customer, product, date,
  amount), not a property of the document.
- Quote the span of the passage the condition comes from.

Return JSON:
{"condition": "..." or null,
 "outcome": "...",
 "applicability": "...",
 "evidence_span": "...",
 "explicitly_marked": true/false}

"explicitly_marked" is true only if the passage uses an exception marker
("except", "unless", "only if", "does not apply to")."""


@dataclass(frozen=True)
class CandidateCondition:
    """One proposed condition, before the gate."""

    condition: str
    outcome: str
    applicability: str
    evidence_span: str
    source_passage: str
    explicitly_marked: bool = False
    question: str = ""

    entailment: float = 0.0
    admitted: bool = False
    rejection_reason: str = ""

    @property
    def explicitness(self) -> Explicitness:
        return Explicitness.EXPLICIT if self.explicitly_marked else Explicitness.IMPLICIT

    def to_branch(self, branch_id: str) -> Branch:
        return Branch(
            branch_id=branch_id,
            condition=self.condition,
            outcome=self.outcome,
            applicability=Applicability(descriptor=self.applicability),
            explicitness=self.explicitness,
            supporting_passage=self.source_passage,
        )


@dataclass
class ProbeResult:
    """Everything one probe produced, admitted and rejected alike.

    Rejected candidates are kept, not discarded. The rejection rate is a
    reported metric, and a rejected candidate is the only evidence of what the
    gate is actually catching -- which is what tells you whether the threshold
    is right.
    """

    query_id: str
    admitted: list[CandidateCondition] = field(default_factory=list)
    rejected: list[CandidateCondition] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    api_calls: int = 0

    @property
    def n_candidates(self) -> int:
        return len(self.admitted) + len(self.rejected)

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / self.n_candidates if self.n_candidates else 0.0


@dataclass
class ProbeStats:
    """Aggregated across a run."""

    instances: int = 0
    candidates: int = 0
    admitted: int = 0
    rejected: int = 0
    explicit_admitted: int = 0
    implicit_admitted: int = 0
    api_calls: int = 0
    parse_failures: int = 0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.candidates if self.candidates else 0.0

    def merge(self, r: ProbeResult) -> None:
        self.instances += 1
        self.candidates += r.n_candidates
        self.admitted += len(r.admitted)
        self.rejected += len(r.rejected)
        self.api_calls += r.api_calls
        for c in r.admitted:
            if c.explicitly_marked:
                self.explicit_admitted += 1
            else:
                self.implicit_admitted += 1

    def render(self) -> str:
        lines = [
            "Contrastive Scope Probing (B1)",
            "-" * 62,
            f"  instances            {self.instances:>8,}",
            f"  candidates proposed  {self.candidates:>8,}",
            f"  admitted             {self.admitted:>8,}",
            f"    explicitly marked  {self.explicit_admitted:>8,}",
            f"    unmarked           {self.implicit_admitted:>8,}   <- the hard case",
            f"  rejected by gate     {self.rejected:>8,}",
            f"  rejection rate       {self.rejection_rate:>8.4f}",
            f"  API calls            {self.api_calls:>8,}",
        ]
        if self.parse_failures:
            lines.append(f"  unparseable replies  {self.parse_failures:>8,}")

        if self.candidates and self.rejection_rate <= SUSPICIOUS_REJECTION_RATE:
            lines += [
                "",
                "  The gate rejected NOTHING. Either the proposer is unusually",
                "  disciplined or tau_ground is too low to bite. Check a sample of",
                "  admitted candidates against their passages before trusting HCR:",
                "  a gate that never fires cannot be evidence that nothing was",
                "  hallucinated.",
            ]
        if self.candidates and self.rejection_rate >= 0.95:
            lines += [
                "",
                "  The gate rejected almost EVERYTHING. That is a threshold problem,",
                "  not a clean corpus -- check tau_ground before concluding the",
                "  proposer is useless.",
            ]
        # Reported separately, never averaged: the unmarked cases are what the
        # method is judged on, and an average over both is dominated by the easy
        # half.
        if self.admitted and not self.implicit_admitted:
            lines += [
                "",
                "  Every admitted condition was explicitly marked. On this data the",
                "  probe has not demonstrated anything a cue-word matcher could not.",
            ]
        return "\n".join(lines)


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from detection.stage2 import _salvage_json
        try:
            return _salvage_json(text)
        except Exception:
            return None


class ContrastiveScopeProbe:
    """B1 end to end."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        nli: NLIScorer | None = None,
        tier: Tier = Tier.BULK,
        tau_ground: float = TAU_GROUND,
        max_questions: int = 3,
        use_llm: bool = True,
        heuristic_nli: bool = False,
    ):
        """
        Parameters
        ----------
        use_llm
            False runs the rule-based fallback proposer, which recovers only
            explicitly-marked conditions. Useful offline and as the ablation
            baseline the probe has to beat -- if it does not beat cue matching,
            the method has not earned its API cost.
        tau_ground
            Grounding threshold. A PLACEHOLDER until tuned on the WP1 pilot.
        """
        self.client = client or (get_client() if use_llm else None)
        self.nli = nli if nli is not None else get_nli(heuristic=heuristic_nli)
        self.tier = tier
        self.tau_ground = tau_ground
        self.max_questions = max_questions
        self.use_llm = use_llm

    # -- step 2: counterfactual questions -------------------------------------- #

    def questions_for(self, rule_text: str, query: str) -> tuple[list[str], int]:
        """Contrastive probe questions for one base rule."""
        if not self.use_llm:
            return ([f"Under what circumstances does this not apply: {rule_text}"], 0)

        result = self.client.complete(
            messages=[{"role": "user", "content":
                       f"Question being answered: {query}\n\n"
                       f"General rule: {rule_text}\n\n"
                       f"Write up to {self.max_questions} contrastive probe questions."}],
            system=_QUESTION_SYSTEM,
            tier=self.tier,
            step="b1_counterfactual_questions",
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = _parse_json(result.text) or {}
        qs = [q for q in (raw.get("questions") or []) if isinstance(q, str) and q.strip()]
        return qs[:self.max_questions] or [
            f"Under what circumstances does this not apply: {rule_text}"], 1

    # -- step 3: candidate proposal -------------------------------------------- #

    def propose(self, question: str, passage: Passage,
                rule_text: str) -> tuple[CandidateCondition | None, int]:
        if not self.use_llm:
            return self._propose_by_rule(passage), 0

        result = self.client.complete(
            messages=[{"role": "user", "content":
                       f"General rule: {rule_text}\n\n"
                       f"Question: {question}\n\n"
                       f"Passage [{passage.id}]: {passage.text}"}],
            system=_CANDIDATE_SYSTEM,
            tier=self.tier,
            step="b1_propose_candidate_condition",
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = _parse_json(result.text)
        if not raw or not raw.get("condition"):
            return None, 1

        return CandidateCondition(
            condition=str(raw["condition"]).strip(),
            outcome=str(raw.get("outcome") or "").strip(),
            applicability=str(raw.get("applicability") or raw["condition"]).strip(),
            evidence_span=str(raw.get("evidence_span") or "").strip(),
            source_passage=passage.id,
            explicitly_marked=bool(raw.get("explicitly_marked")),
            question=question,
        ), 1

    def _propose_by_rule(self, passage: Passage) -> CandidateCondition | None:
        """Cue-based fallback. Recovers explicit conditions only, by design.

        This is the baseline the probe must beat. Keeping it in the same class
        rather than in a separate ablation file makes the comparison hard to
        avoid running.
        """
        from detection.features import STRONG_EXCEPTION_CUES

        low = passage.text.lower()
        hit = next((c for c in STRONG_EXCEPTION_CUES if c in low), None)
        if hit is None:
            return None
        idx = low.index(hit)
        return CandidateCondition(
            condition=passage.text[idx:].strip().rstrip("."),
            outcome=passage.text[:idx].strip().rstrip(".") or passage.text,
            applicability=passage.text[idx + len(hit):].strip().rstrip("."),
            evidence_span=passage.text[idx:].strip(),
            source_passage=passage.id,
            explicitly_marked=True,
            question="rule-based fallback",
        )

    # -- step 4: the grounding gate -------------------------------------------- #

    def gate(self, candidates: list[CandidateCondition],
             passages: dict[str, str]) -> list[CandidateCondition]:
        """Score ``passage |= (condition AND outcome)`` and admit or reject.

        Scored against the passage the candidate cites, not against the whole
        record. A condition that is entailed by some *other* passage is not
        grounded in the one it claims, and letting that pass would admit
        conditions assembled from two documents at once.
        """
        if not candidates:
            return []

        pairs = []
        for c in candidates:
            premise = passages.get(c.source_passage, "")
            hypothesis = f"{c.condition}, and in that case {c.outcome}".strip()
            pairs.append((premise, hypothesis))

        scores = self.nli.score_batch(pairs)
        out: list[CandidateCondition] = []
        for c, s in zip(candidates, scores):
            admitted = s.entailment >= self.tau_ground
            reason = "" if admitted else (
                f"entailment {s.entailment:.2f} below tau_ground {self.tau_ground:.2f}; "
                "the cited passage does not support this condition"
            )
            out.append(CandidateCondition(
                **{**c.__dict__, "entailment": s.entailment,
                   "admitted": admitted, "rejection_reason": reason}))
        return out

    # -- end to end ------------------------------------------------------------ #

    def probe_record(self, record: QueryRecord, *,
                     base_passage_id: str | None = None) -> ProbeResult:
        """Probe one record.

        The base rule defaults to the first passage. When A4 has already
        identified which passage carries the default branch, pass its id --
        probing the exception as if it were the rule inverts every condition.
        """
        if not record.passages:
            return ProbeResult(query_id=record.query_id)

        by_id = {p.id: p for p in record.passages}
        base = by_id.get(base_passage_id or "", record.passages[0])
        others = [p for p in record.passages if p.id != base.id]

        questions, calls = self.questions_for(base.text, record.query)
        candidates: list[CandidateCondition] = []

        for p in others:
            for q in questions:
                cand, n = self.propose(q, p, base.text)
                calls += n
                if cand is not None:
                    candidates.append(cand)
                if not self.use_llm:
                    break  # the fallback ignores the question; one pass is enough

        candidates = self._dedupe(candidates)
        scored = self.gate(candidates, {p.id: p.text for p in record.passages})

        return ProbeResult(
            query_id=record.query_id,
            admitted=[c for c in scored if c.admitted],
            rejected=[c for c in scored if not c.admitted],
            questions=questions,
            api_calls=calls,
        )

    @staticmethod
    def _dedupe(candidates: list[CandidateCondition]) -> list[CandidateCondition]:
        """Several questions often recover the same condition from one passage.

        Counting those separately would inflate both the candidate count and
        the apparent recall, so they collapse on (passage, condition).
        """
        seen: dict[tuple[str, str], CandidateCondition] = {}
        for c in candidates:
            key = (c.source_passage, c.condition.lower().strip())
            if key not in seen:
                seen[key] = c
        return list(seen.values())

    def probe_all(self, records: list[QueryRecord]) -> tuple[list[ProbeResult], ProbeStats]:
        stats = ProbeStats()
        out: list[ProbeResult] = []
        for r in records:
            res = self.probe_record(r)
            out.append(res)
            stats.merge(res)
        return out, stats
