"""Member A's baselines: the retrieval and detection side.

Three systems, in increasing order of how directly they demonstrate the
problem:

**Standard RAG** concatenates the top-K passages and lets the generator sort it
out. No conflict handling at all. The floor.

**Rerank** keeps only the single highest-ranked passage. A selection paradigm
in its crudest form.

**NLI-filter** drops whichever passage an entailment model judges inconsistent
with the rest. This is the vivid one, and it is included precisely because it
is a reasonable-looking design that a practitioner might actually ship: on a
general-rule/valid-exception pair, the exception is the passage that looks
inconsistent, so the filter deletes it. The system then answers confidently
using only the general rule, scores well on answer correctness, and has
silently erased the branch that applied to the user.

That is the project's central failure mode, reproducible in about thirty lines.
It is the baseline to put on a slide.
"""

from __future__ import annotations

from dataclasses import dataclass

from contract.models import Passage, QueryRecord
from detection.nli import NLIScorer, get_nli


@dataclass(frozen=True)
class BaselineOutput:
    """What a baseline hands to a generator."""

    name: str
    kept: list[Passage]
    dropped: list[Passage]
    rationale: str

    @property
    def context(self) -> str:
        return "\n\n".join(f"[{p.id}] {p.text}" for p in self.kept)

    @property
    def suppressed_ids(self) -> list[str]:
        """Passages this baseline discarded.

        Handed to the metric suite: a gold branch grounded only in a dropped
        passage cannot possibly be preserved, which is how Suppression Rate
        gets attributed to the resolution step rather than to generation.
        """
        return [p.id for p in self.dropped]


class StandardRAG:
    """Concatenate the top-K passages. No conflict handling."""

    name = "standard_rag"

    def __init__(self, k: int | None = None):
        self.k = k

    def run(self, record: QueryRecord) -> BaselineOutput:
        kept = record.passages[: self.k] if self.k else list(record.passages)
        return BaselineOutput(
            name=self.name,
            kept=kept,
            dropped=[p for p in record.passages if p not in kept],
            rationale="all retrieved passages concatenated; no conflict detection",
        )


class RerankTop1:
    """Keep only the top-ranked passage.

    Selection at its simplest. Retrieval order is taken as the credibility
    order, which is what makes this a fair stand-in for rank-and-pick systems
    without reimplementing a full credibility model.
    """

    name = "rerank_top1"

    def run(self, record: QueryRecord) -> BaselineOutput:
        if not record.passages:
            return BaselineOutput(self.name, [], [], "no passages retrieved")
        kept = [record.passages[0]]
        return BaselineOutput(
            name=self.name,
            kept=kept,
            dropped=list(record.passages[1:]),
            rationale="single highest-ranked passage kept; everything else discarded",
        )


class NLIFilter:
    """Drop the passage judged most inconsistent with the others.

    The demonstrative baseline. On a conditional conflict the exception is, by
    construction, the passage that reads as inconsistent with the general rule
    -- so this filter reliably deletes exactly the branch the user needed.
    """

    name = "nli_filter"

    def __init__(
        self,
        *,
        nli: NLIScorer | None = None,
        profile: str | None = None,
        heuristic_nli: bool = False,
        threshold: float = 0.5,
    ):
        self._nli = nli
        self._profile = profile
        self._heuristic = heuristic_nli
        self.threshold = threshold

    @property
    def nli(self) -> NLIScorer:
        if self._nli is None:
            self._nli = get_nli(self._profile, heuristic=self._heuristic)
        return self._nli

    def run(self, record: QueryRecord) -> BaselineOutput:
        passages = list(record.passages)
        if len(passages) < 2:
            return BaselineOutput(self.name, passages, [], "fewer than two passages; nothing to filter")

        # Score each passage against every other. Highest total contradiction
        # is the odd one out.
        pairs, index = [], []
        for i, a in enumerate(passages):
            for j, b in enumerate(passages):
                if i != j:
                    pairs.append((b.text, a.text))   # does the rest contradict a?
                    index.append(i)

        scores = self.nli.score_batch(pairs)
        totals = [0.0] * len(passages)
        for idx, s in zip(index, scores):
            totals[idx] += s.contradiction

        worst = max(range(len(passages)), key=lambda i: (totals[i], passages[i].id))
        if totals[worst] < self.threshold:
            return BaselineOutput(
                name=self.name, kept=passages, dropped=[],
                rationale=(
                    f"no passage exceeded the inconsistency threshold "
                    f"(max {totals[worst]:.3f} < {self.threshold})"
                ),
            )

        dropped = passages[worst]
        return BaselineOutput(
            name=self.name,
            kept=[p for i, p in enumerate(passages) if i != worst],
            dropped=[dropped],
            rationale=(
                f"dropped {dropped.id} as inconsistent (contradiction score "
                f"{totals[worst]:.3f}). On a conditional conflict this is the exception "
                f"branch, and discarding it is the suppression the project measures."
            ),
        )


ALL_BASELINES = {
    StandardRAG.name: StandardRAG,
    RerankTop1.name: RerankTop1,
    NLIFilter.name: NLIFilter,
}
