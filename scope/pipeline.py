"""A4 end to end: annotate a record's conditional pairs with scope relations.

Takes a record whose ``conflict_pairs`` have been labelled by A2/A3 and fills
in the ``scope_relation`` field for every conditional pair -- and for
borderline factual pairs, where the analyser runs too because the
factual/conditional boundary is the one the detector is least reliable on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contract.models import ConflictPair, ConflictType, QueryRecord
from scope.descriptor import ClaimDescriptor, DescriptorExtractor, rule_based_descriptor
from scope.relation import Evidence, ScopeAnalyser, ScopeDecision


@dataclass
class ScopeStats:
    pairs_analysed: int = 0
    by_relation: dict[str, int] = field(default_factory=dict)
    by_evidence: dict[str, int] = field(default_factory=dict)
    rule_based_descriptors: int = 0

    def merge(self, other: ScopeStats) -> None:
        self.pairs_analysed += other.pairs_analysed
        self.rule_based_descriptors += other.rule_based_descriptors
        for src, dst in ((other.by_relation, self.by_relation),
                         (other.by_evidence, self.by_evidence)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v

    def render(self) -> str:
        lines = ["Scope analysis (A4)", "-" * 56,
                 f"  pairs analysed       {self.pairs_analysed:>8,}"]
        if self.by_relation:
            lines.append("  by relation:")
            for k, v in sorted(self.by_relation.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<16} {v:>6,}")
        if self.by_evidence:
            lines.append("  decided by:")
            for k, v in sorted(self.by_evidence.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<16} {v:>6,}")
            n_ent = self.by_evidence.get(Evidence.ENTAILMENT.value, 0)
            if n_ent and self.pairs_analysed:
                lines.append(
                    f"  NOTE: {n_ent / self.pairs_analysed:.0%} of decisions rested on "
                    "entailment rather than typed attributes."
                )
                lines.append(
                    "        Those are model judgements, not exact set operations -- "
                    "report the split."
                )
        if self.rule_based_descriptors:
            lines.append(
                f"  !! {self.rule_based_descriptors:,} descriptor(s) came from the coarse "
                "rule-based extractor, not the LLM."
            )
        return "\n".join(lines)


class ScopePipeline:
    """A4: descriptor extraction followed by four-way relation analysis."""

    def __init__(
        self,
        *,
        extractor: DescriptorExtractor | None = None,
        analyser: ScopeAnalyser | None = None,
        use_llm_extraction: bool = True,
        analyse_borderline_factual: bool = True,
        heuristic_nli: bool = False,
    ):
        self.extractor = extractor or DescriptorExtractor()
        self.analyser = analyser or ScopeAnalyser(heuristic_nli=heuristic_nli)
        self.use_llm_extraction = use_llm_extraction
        self.analyse_borderline_factual = analyse_borderline_factual

    # -- descriptors ----------------------------------------------------------- #

    def describe(self, record: QueryRecord) -> dict[str, ClaimDescriptor]:
        """Extract a descriptor per passage.

        Done once per record and reused across pairs. The same passage appears
        in up to four pairs at K = 5, so re-extracting per pair would multiply
        the API cost by four for no benefit.
        """
        out: dict[str, ClaimDescriptor] = {}
        for p in record.passages:
            if self.use_llm_extraction:
                try:
                    out[p.id] = self.extractor.extract(record.query, p.id, p.text)
                    continue
                except Exception:
                    # A descriptor we cannot extract must not abort a corpus
                    # run; the coarse extractor is marked in the stats.
                    pass
            out[p.id] = rule_based_descriptor(p.id, p.text)
        return out

    # -- analysis --------------------------------------------------------------- #

    def _should_analyse(self, pair: ConflictPair) -> bool:
        if not pair.is_conflict:
            return False
        if pair.type is ConflictType.CONDITIONAL:
            return True
        return self.analyse_borderline_factual and pair.type is ConflictType.FACTUAL

    def analyse_record(
        self, record: QueryRecord
    ) -> tuple[QueryRecord, ScopeStats, dict[tuple[str, str], ScopeDecision]]:
        stats = ScopeStats()
        if not any(self._should_analyse(p) for p in record.conflict_pairs):
            return record, stats, {}

        descriptors = self.describe(record)
        stats.rule_based_descriptors = sum(
            1 for d in descriptors.values() if d.source == "rules"
        )

        texts = {p.id: p.text for p in record.passages}
        decisions: dict[tuple[str, str], ScopeDecision] = {}
        updated: list[ConflictPair] = []

        for pair in record.conflict_pairs:
            if not self._should_analyse(pair):
                updated.append(pair)
                continue

            d_i, d_j = descriptors.get(pair.doc_i), descriptors.get(pair.doc_j)
            if d_i is None or d_j is None:
                updated.append(pair)
                continue

            decision = self.analyser.analyse(
                d_i, d_j, text_a=texts.get(pair.doc_i, ""), text_b=texts.get(pair.doc_j, "")
            )
            decisions[pair.key] = decision

            stats.pairs_analysed += 1
            stats.by_relation[decision.relation.value] = (
                stats.by_relation.get(decision.relation.value, 0) + 1
            )
            stats.by_evidence[decision.evidence.value] = (
                stats.by_evidence.get(decision.evidence.value, 0) + 1
            )

            updated.append(pair.model_copy(update={"scope_relation": decision.relation}))

        return record.model_copy(update={"conflict_pairs": updated}), stats, decisions

    def analyse_all(self, records: list[QueryRecord]) -> tuple[list[QueryRecord], ScopeStats]:
        out, total = [], ScopeStats()
        for rec in records:
            r, s, _ = self.analyse_record(rec)
            out.append(r)
            total.merge(s)
        return out, total
