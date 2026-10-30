"""One query, end to end, with every step reported.

    retrieve -> detect (stage 1, stage 2) -> type -> policy -> scope (A4)
      -> extract (B1) -> branches -> compose (B2) -> generate (B3) -> verify
      -> arena (baselines) -> reader -> score (against gold, when there is gold)

Each stage is the project's own module, called as the experiments call it;
this file adds no logic of its own beyond the two pieces that never existed:
assembling a default branch (``service.branches``) and the detector-miss
policy. Both are reported as what they are.

Offline mode is deterministic and needs no model download: heuristic NLI,
rule-based descriptors, cue-based extraction, template generation. Live mode
loads DeBERTa and Contriever and calls the LLM tiers through ``api_budget``;
every call shows up in the run as an ``llm`` event with its tokens and cost.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from contract.gold import Branch, GoldInstance
from contract.models import (ConflictPair, ConflictType, Construction, DecidedBy, Domain, Passage,
                             QueryRecord, ScopeRelation, SourceType)
from service import branches as B
from service.events import Tracer, in_stage, use_tracer

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "benchmark" / "data" / "corpus.jsonl"

SOURCES = ("benchmark", "collection", "passages")
MODES = ("offline", "live")
POLICIES = ("as_run", "override")


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


@dataclass
class RunRequest:
    query: str = ""
    source: str = "benchmark"
    record_id: str | None = None
    collection_id: str | None = None
    passages: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "offline"
    detector_policy: str = "override"
    generation: str = "template"
    reader: dict[str, Any] = field(default_factory=dict)
    k: int = 5
    long_context_baseline: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRequest:
        names = {f for f in cls.__dataclass_fields__}
        req = cls(**{k: v for k, v in d.items() if k in names and v is not None})
        req.validate()
        return req

    def validate(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.detector_policy not in POLICIES:
            raise ValueError(f"detector_policy must be one of {POLICIES}")
        if self.generation not in ("template", "llm"):
            raise ValueError("generation must be 'template' or 'llm'")
        if self.source == "benchmark" and not self.record_id:
            raise ValueError("a benchmark run needs record_id")
        if self.source == "collection" and not (self.collection_id and self.query.strip()):
            raise ValueError("a collection run needs collection_id and a query")
        if self.source == "passages":
            if not self.query.strip():
                raise ValueError("a passages run needs a query")
            if len([p for p in self.passages if str(p.get("text", "")).strip()]) < 2:
                raise ValueError("give at least two passages")
        if not 2 <= int(self.k) <= 10:
            raise ValueError("k must be between 2 and 10")


# --------------------------------------------------------------------------- #
# Shared, expensive things: the benchmark, retrievers, engines
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_benchmark: dict[str, GoldInstance] | None = None
_engines: dict[bool, Any] = {}
_retrievers: dict[tuple, Any] = {}


def benchmark() -> dict[str, GoldInstance]:
    global _benchmark
    with _lock:
        if _benchmark is None:
            out: dict[str, GoldInstance] = {}
            with BENCHMARK.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "gold_conflict_type" in obj:
                        inst = GoldInstance.model_validate(obj)
                        out[inst.instance_id] = inst
            _benchmark = out
        return _benchmark


class Engines:
    """The pipeline's modules, configured for one mode. Built once per mode."""

    def __init__(self, live: bool):
        from composition.operator import CompositionOperator
        from detection.pipeline import TwoStageDetector
        from detection.stage1 import Stage1Filter
        from detection.variants.lexical_nli import LexicalNLIDetector
        from extraction.probing import ContrastiveScopeProbe
        from generation.scoped_answer import ScopedAnswerGenerator
        from scope.pipeline import ScopePipeline
        from service import tracing_client

        tracing_client.install()  # before anything captures the default client
        self.live = live
        self.stage1 = Stage1Filter(heuristic_nli=not live)
        self.detector = TwoStageDetector(stage1=self.stage1, use_stage2=live,
                                         classifier=LexicalNLIDetector(stage1=self.stage1))
        self.scope = ScopePipeline(use_llm_extraction=live, heuristic_nli=not live)
        self.probe = ContrastiveScopeProbe(use_llm=live, heuristic_nli=not live)
        self.operator = CompositionOperator()
        self.template = ScopedAnswerGenerator(use_llm=False)
        self.llm_generator = ScopedAnswerGenerator(use_llm=True) if live else None

    def describe(self) -> dict[str, str]:
        return {
            "nli": "DeBERTa-v3 cross-encoder (cross-encoder/nli-deberta-v3-base)" if self.live
                   else "heuristic NLI (offline fixture, not a model)",
            "stage1_scorer": "trained head" if self.stage1.head is not None
                             else "hand-weighted rule scorer (no trained head)",
            "stage2": "LLM adjudication of the uncertain band" if self.live else "off (offline)",
            "descriptors": "LLM descriptor extraction, rule fallback" if self.live else "rule-based descriptors",
            "extraction": "contrastive scope probing (LLM) + NLI grounding gate" if self.live
                          else "cue-word proposer + grounding gate",
            "generation": "template; LLM on request" if self.live else "template",
        }


def engines(live: bool) -> Engines:
    with _lock:
        if live not in _engines:
            _engines[live] = Engines(live)
        return _engines[live]


def retriever_for(key: tuple, passages: list[Passage], live: bool):
    """A cached hybrid retriever (sparse always; dense in live mode)."""
    from retrieval.corpus import Corpus
    from retrieval.pipeline import HybridRetriever

    import hashlib

    digest = hashlib.sha1("\n".join(f"{p.id}\t{p.text}" for p in passages).encode()).hexdigest()[:12]
    full = (*key, live, digest)
    with _lock:
        if full not in _retrievers:
            # dense embeddings persist per exact passage set, so a restart does not re-encode
            cache = ROOT / "logs" / "embeddings" / f"{'-'.join(map(str, key))}-{digest}.npz"
            _retrievers[full] = HybridRetriever(Corpus(list(passages)), use_dense=live, embedding_cache=cache)
        return _retrievers[full]


def benchmark_retriever(live: bool):
    from retrieval.corpus import Corpus

    corpus = Corpus.from_records(benchmark().values())
    return retriever_for(("benchmark",), corpus.passages, live)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _excerpt(text: str, n: int = 220) -> str:
    return text if len(text) <= n else text[:n].rstrip() + " …"


def _rank_tables(retr, query: str, show: int = 8) -> dict[str, Any]:
    from retrieval.fusion import reciprocal_rank_fusion

    pool = max(retr.top_k * 4, 20)
    sparse = retr.sparse.search(query, k=pool)
    rankings = [sparse]
    dense = []
    if retr.use_dense:
        dense = retr.dense.search(query, k=pool)
        rankings.append(dense)
    fused = reciprocal_rank_fusion(rankings, k=retr.rrf_k, top_k=max(retr.top_k, show))
    text = lambda pid: _excerpt(retr.corpus.get(pid).text, 160)  # noqa: E731
    return {
        "bm25": [{"id": s.passage_id, "rank": s.rank, "score": s.score, "text": text(s.passage_id)}
                 for s in sparse[:show]],
        "dense": [{"id": s.passage_id, "rank": s.rank, "score": s.score, "text": text(s.passage_id)}
                  for s in dense[:show]],
        "fused": [{"id": f.passage_id, "rank": f.rank, "score": f.score, "sparse_rank": f.sparse_rank,
                   "dense_rank": f.dense_rank, "both": f.found_by_both, "text": text(f.passage_id)}
                  for f in fused[:show]],
        "rrf_k": retr.rrf_k, "top_k": retr.top_k, "pool": pool, "dense_on": retr.use_dense,
        "_fused_ids": [f.passage_id for f in fused],
    }


def construction_of(passages: list[Passage]) -> Construction:
    """Provenance, as the contract defines it: passages all from one document are
    a split (Tier 2) record, never a natural (Tier 1) one."""
    docs = {p.document_id for p in passages}
    return Construction.SPLIT if len(docs) == 1 and None not in docs else Construction.NATURAL


def _passage_view(p: Passage) -> dict[str, Any]:
    return {"id": p.id, "text": p.text, "source_type": p.source_type.value, "date": p.date,
            "url": p.source_url, "document_id": p.document_id}


def _branch_view(b: Branch, origin: str = "") -> dict[str, Any]:
    return {"id": b.branch_id, "default": b.is_default, "condition": b.condition, "outcome": b.outcome,
            "applies_to": b.applicability.descriptor, "passage": b.supporting_passage,
            "explicitness": b.explicitness.value,
            "attributes": [a.model_dump(mode="json", exclude_none=True) for a in b.applicability.attributes],
            "origin": origin}


def _pair_view(p: ConflictPair) -> dict[str, Any]:
    return {"pair": [p.doc_i, p.doc_j], "is_conflict": p.is_conflict, "type": p.type.value,
            "relation": p.scope_relation.value if p.scope_relation else None,
            "confidence": p.confidence, "decided_by": p.decided_by.value}


def _credibility_pick(passages: list[Passage], order: list[str]) -> Passage:
    """Prior work's selection: most credible source type, then the earliest date."""
    rank = {s: i for i, s in enumerate(order)}
    return min(passages, key=lambda p: (rank.get(p.source_type.value, len(order)), p.date or "9999", p.id))


_CLAUSE = __import__("re").compile(r"(?<=[.;!?])\s+|;\s*")


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE.split(text) if c and c.strip()]


def _states(outcome: str, text: str) -> bool:
    """Does any clause of ``text`` state ``outcome``? The project's own matcher,
    which vetoes on clashing numbers and opposite polarity -- word overlap alone
    would read "no monthly fee applies" into "a monthly fee applies"."""
    from metrics.branch_match import outcomes_match

    return any(outcomes_match(outcome, c) for c in _clauses(text))


def _judge_text(text: str, gold: list[Branch]) -> list:
    from metrics.branch_match import BranchJudgement, similarity

    out = []
    for b in gold:
        if _states(b.outcome, text):
            out.append(BranchJudgement.PRESERVED)
        elif any(similarity(b.outcome, c) >= 0.25 for c in _clauses(text)):
            out.append(BranchJudgement.DISTORTED)
        else:
            out.append(BranchJudgement.SUPPRESSED)
    return out


def _rates(judgements: list) -> dict[str, Any]:
    n = len(judgements)
    pres = sum(1 for j in judgements if j.value == "preserved")
    supp = sum(1 for j in judgements if j.value == "suppressed")
    dist = sum(1 for j in judgements if j.value == "distorted")
    return {"n": n, "preserved": pres, "suppressed": supp, "distorted": dist,
            "PR": pres / n if n else None, "SR": supp / n if n else None,
            "judgements": [j.value for j in judgements]}


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run_query(req: RunRequest, tracer: Tracer, *, store=None) -> dict[str, Any]:
    """Run one request, emitting events into ``tracer``. Returns the summary."""
    req.validate()
    live = req.mode == "live"
    with use_tracer(tracer):
        return _run(req, tracer, live, store)


def _run(req: RunRequest, tracer: Tracer, live: bool, store) -> dict[str, Any]:
    from contract.routing import route_with_flags
    from generation.scoped_answer import check_faithfulness, render_template
    from metrics.preservation import judge_branches
    from retrieval.pipeline import record_from_passages

    tracer.data("run", "request", request=asdict(req))
    summary: dict[str, Any] = {"mode": req.mode, "source": req.source, "detector_policy": req.detector_policy}

    # ---- setup ------------------------------------------------------------------
    with in_stage(tracer, "setup", "Load the pipeline"):
        E = engines(live)
        tracer.data("setup", "components", components=E.describe(), live=live,
                    thresholds={"low": E.stage1.low, "high": E.stage1.high, "decide": 0.5,
                                "tuned": E.stage1.thresholds_tuned},
                    tau_ground=E.probe.tau_ground)
        for k, v in E.describe().items():
            tracer.log("setup", f"{k}: {v}")

    inst: GoldInstance | None = None
    query = req.query.strip()

    # ---- ingest -----------------------------------------------------------------
    with in_stage(tracer, "ingest", "Gather the documents"):
        if req.source == "benchmark":
            inst = benchmark().get(req.record_id or "")
            if inst is None:
                raise KeyError(f"no benchmark record {req.record_id!r}")
            query = query or inst.query
            tracer.data("ingest", "benchmark record", id=inst.instance_id, query=inst.query,
                        domain=inst.domain.value, construction=inst.construction.value,
                        separation=inst.separation.value if inst.separation else None, split=inst.split,
                        gold_type=inst.gold_conflict_type.value,
                        gold_relation=inst.gold_scope_relation.value if inst.gold_scope_relation else None,
                        gold_branches=len(inst.gold_branches))
            tracer.log("ingest", f"benchmark record {inst.instance_id} ({inst.construction.value}, {inst.split} split)")
        elif req.source == "collection":
            if store is None:
                raise RuntimeError("collections need the service store")
            coll = store.get_collection(req.collection_id or "")
            if coll is None:
                raise KeyError(f"no collection {req.collection_id!r}")
            tracer.data("ingest", "collection", id=coll["id"], name=coll["name"],
                        documents=coll["documents"], passages=coll["passages"])
            tracer.log("ingest", f"collection “{coll['name']}”: {len(coll['documents'])} document(s), "
                                 f"{coll['passages']} passage(s)")
        else:
            tracer.data("ingest", "inline passages", count=len(req.passages))
            tracer.log("ingest", f"{len(req.passages)} passage(s) given inline")

    # ---- retrieve ---------------------------------------------------------------
    with in_stage(tracer, "retrieve", "Retrieve passages (BM25 + dense, RRF)"):
        if req.source == "benchmark":
            assert inst is not None
            retr = benchmark_retriever(live)
            tables = _rank_tables(retr, query)
            own = {f"{inst.instance_id}::{p.id}" for p in inst.passages}
            fused_ids = tables.pop("_fused_ids")
            for row in tables["fused"] + tables["bm25"] + tables["dense"]:
                row["is_instance"] = row["id"] in own
            hits = [pid for pid in fused_ids[:retr.top_k] if pid in own]
            tracer.data("retrieve", "rankings over the benchmark corpus", corpus_size=len(retr.corpus), **tables)
            tracer.data("retrieve", "decision", mode="oracle",
                        note="Benchmark runs answer from the instance's own passages, as every experiment does; "
                             "the rankings above show whether hybrid retrieval would have found them.",
                        found=len(hits), of=len(own))
            tracer.log("retrieve", f"hybrid retrieval found {len(hits)} of {len(own)} instance passage(s) in the top {retr.top_k}")
            record = inst.to_query_record(include_gold_pairs=False)
        elif req.source == "collection":
            pool = store.passages(req.collection_id)
            if len(pool) < 2:
                raise ValueError("the collection needs at least two passages")
            retr = retriever_for(("collection", req.collection_id), pool, live)
            retr.top_k = min(int(req.k), len(pool))
            tables = _rank_tables(retr, query)
            fused_ids = tables.pop("_fused_ids")[:retr.top_k]
            tracer.data("retrieve", "rankings over the collection", corpus_size=len(pool), **tables)
            chosen = [retr.corpus.get(pid) for pid in fused_ids]
            record = record_from_passages(query, chosen, query_id=tracer.run_id,
                                          construction=construction_of(chosen))
            tracer.data("retrieve", "decision", mode="retrieved", kept=[
                {"as": f"p{i}", "from": p.id} for i, p in enumerate(chosen)])
            tracer.log("retrieve", f"kept the top {len(chosen)} fused passages")
        else:
            given = []
            for i, p in enumerate(x for x in req.passages if str(x.get("text", "")).strip()):
                st = p.get("source_type") or "unknown"
                given.append(Passage(id=f"p{i}", text=str(p["text"]).strip(),
                                     source_type=SourceType(st) if st in SourceType._value2member_map_ else SourceType.UNKNOWN,
                                     date=p.get("date") or None, source_url=p.get("url") or None))
            record = record_from_passages(query, given, query_id=tracer.run_id)
            tracer.data("retrieve", "decision", mode="given", note="Passages were given; retrieval is skipped.")
            tracer.data("retrieve", "skipped", reason="passages given inline")
            tracer.log("retrieve", "passages given inline: retrieval skipped")
        tracer.data("retrieve", "passages", passages=[_passage_view(p) for p in record.passages])
        tracer.data("contract", "QueryRecord after A1", record=record)

    by_id = {p.id: p for p in record.passages}

    # ---- detect -----------------------------------------------------------------
    with in_stage(tracer, "detect", "Detect conflicts (stage 1, stage 2)"):
        cands = E.stage1.score_pairs(record.passages)
        for c in cands:
            p = c.confidence if c.is_conflict else 1.0 - c.confidence
            tracer.data("detect", f"{c.doc_i} × {c.doc_j}", pair=[c.doc_i, c.doc_j],
                        nli={"entailment": c.nli.entailment, "neutral": c.nli.neutral,
                             "contradiction": c.nli.contradiction},
                        features=asdict(c.features), score=p, is_conflict=c.is_conflict,
                        escalate=c.needs_escalation)
        esc = [c for c in cands if c.needs_escalation]
        tracer.log("detect", f"{len(cands)} pair(s) scored locally; {len(esc)} in the uncertain band "
                             f"[{E.stage1.low:.2f}, {E.stage1.high:.2f}]")
        if esc and not live:
            tracer.log("detect", "stage 2 is off offline: uncertain pairs keep stage 1's verdict", "warn")
        record, dstats = E.detector.detect_from_candidates(record, cands)
        tracer.data("detect", "verdicts", pairs=[_pair_view(p) for p in record.conflict_pairs], stats=dstats)

    with in_stage(tracer, "type", "Classify the conflict (A3)"):
        for p in record.conflict_pairs:
            tracer.log("type", f"{p.doc_i} × {p.doc_j}: {p.type.value} "
                               f"({p.decided_by.value}, confidence {p.confidence:.2f})")
        tracer.data("type", "types", pairs=[_pair_view(p) for p in record.conflict_pairs])
        tracer.data("contract", "QueryRecord after A2/A3", record=record)

    # ---- detector-miss policy -----------------------------------------------------
    override: dict[str, Any] | None = None
    with in_stage(tracer, "policy", "Detector-miss policy"):
        conflicting = [p for p in record.conflict_pairs if p.is_conflict]
        if conflicting:
            tracer.data("policy", "no override needed", conflicts=len(conflicting))
            tracer.log("policy", f"the detector found {len(conflicting)} conflict(s)")
        elif req.detector_policy == "as_run":
            tracer.warn("policy", "No conflict detected: answering with the general rule (as-run)",
                        policy="as_run")
        elif inst is not None and inst.gold_conflict_type is ConflictType.NO_CONFLICT:
            tracer.data("policy", "gold agrees: no conflict", policy="override", applied=False)
            tracer.log("policy", "gold also says no conflict: nothing to override")
        else:
            if inst is not None:
                a, b = inst.passages[0].id, inst.passages[1].id
                ctype, rel, kind = inst.gold_conflict_type, inst.gold_scope_relation, "gold"
                decided = DecidedBy.GOLD
            else:
                best = max(cands, key=lambda c: c.confidence if c.is_conflict else 1.0 - c.confidence)
                a, b = best.doc_i, best.doc_j
                ctype, rel, kind = ConflictType.CONDITIONAL, ScopeRelation.REFINEMENT, "assumed"
                decided = DecidedBy.MOCK
            forced = ConflictPair(doc_i=a, doc_j=b, is_conflict=True, type=ctype,
                                  scope_relation=rel if ctype in (ConflictType.CONDITIONAL, ConflictType.FACTUAL) else None,
                                  confidence=1.0, decided_by=decided)
            record = record.model_copy(update={"conflict_pairs": [
                forced if p.key == forced.key else p for p in record.conflict_pairs]})
            override = {"kind": kind, "pair": [forced.doc_i, forced.doc_j], "type": ctype.value,
                        "provisional_relation": rel.value if rel else None}
            tracer.warn("policy", "Detector missed: continuing as if detected", policy="override", **override,
                        badge=("oracle: gold label" if kind == "gold" else "assumed conflict"))
    summary["override"] = override

    # ---- scope (A4) -----------------------------------------------------------------
    decisions: dict = {}
    with in_stage(tracer, "scope", "Scope relation (A4)"):
        if any(E.scope._should_analyse(p) for p in record.conflict_pairs):
            descriptors = E.scope.describe(record)
            tracer.data("scope", "descriptors", descriptors=[{
                "passage": pid, "outcome": d.outcome, "applies_to": d.applicability.descriptor,
                "default": d.applicability.is_default, "source": d.source,
                "attributes": [x.model_dump(mode="json", exclude_none=True) for x in d.applicability.attributes],
            } for pid, d in descriptors.items()])
            record, sstats, decisions = E.scope.analyse_record(record, descriptors=descriptors)
            for key, d in decisions.items():
                tracer.data("scope", f"{key[0]} × {key[1]}", pair=list(key), relation=d.relation.value,
                            set_relation=d.set_relation.value, outcomes_agree=d.outcomes_agree,
                            evidence=d.evidence.value, confidence=d.confidence, rationale=d.rationale,
                            default_passage=d.default.passage_id, exception_passage=d.exception.passage_id,
                            roles_swapped=d.roles_swapped,
                            gold_relation=(inst.gold_scope_relation.value if inst and inst.gold_scope_relation else None))
                tracer.log("scope", f"{key[0]} × {key[1]}: {d.relation.value} ({d.evidence.value}) — {d.rationale}")
            tracer.data("contract", "QueryRecord after A4", record=record)
        else:
            tracer.log("scope", "no conditional conflict to analyse: skipped")
            tracer.data("scope", "skipped", reason="no conditional conflict to analyse")

    main = next((p for p in record.conflict_pairs if p.is_conflict), None)
    decision = decisions.get(main.key) if main is not None else None
    if main is not None:
        summary["relation"] = main.scope_relation.value if main.scope_relation else None
        summary["type"] = main.type.value

    # ---- extract (B1) -----------------------------------------------------------------
    probe = None
    with in_stage(tracer, "extract", "Extract conditions (B1)"):
        if main is None:
            tracer.log("extract", "no conflict: nothing to extract")
            tracer.data("extract", "skipped", reason="no conflict to extract conditions for")
        else:
            base_id = decision.default.passage_id if decision else main.doc_i
            sub = record.model_copy(update={"passages": [p for p in record.passages if p.id in main.key]})
            probe = E.probe.probe_record(sub, base_passage_id=base_id)
            tracer.data("extract", "probe", base_passage=base_id, questions=probe.questions,
                        tau_ground=E.probe.tau_ground,
                        candidates=[{**{k: v for k, v in c.__dict__.items()}, "explicitness": c.explicitness.value}
                                    for c in probe.admitted + probe.rejected],
                        admitted=len(probe.admitted), rejected=len(probe.rejected), api_calls=probe.api_calls)
            tracer.log("extract", f"{len(probe.questions)} question(s); {len(probe.admitted)} condition(s) admitted, "
                                  f"{len(probe.rejected)} rejected by the grounding gate")

    # ---- branches ---------------------------------------------------------------------
    with in_stage(tracer, "branches", "Assemble branches"):
        extracted = [c.to_branch(f"x{i}") for i, c in enumerate(probe.admitted)] if probe else []
        if override and override["kind"] == "gold" and inst is not None and inst.gold_branches:
            branches = B.gold_branches(inst)
            origin = "oracle: gold branches"
            if extracted:
                from metrics.branch_match import align

                res = align([b for b in inst.gold_branches if not b.is_default], extracted)
                tracer.data("branches", "B1 against gold", recovered=sum(
                    1 for a in res.alignments if a.judgement.value == "preserved"),
                    gold_exceptions=len(inst.exception_branches))
        elif main is not None:
            base_id = decision.default.passage_id if decision else main.doc_i
            branches = B.assemble(by_id[base_id], B.usable_outcome(decision.default) if decision else None, extracted)
            origin = "default from A4 + conditions from B1"
        else:
            branches = [B.default_branch(record.passages[0])]
            origin = "general rule only (no conflict)"
        tracer.data("branches", "branches", origin=origin, branches=[_branch_view(b) for b in branches],
                    dims=B.reader_dims(branches))
        tracer.log("branches", f"{len(branches)} branch(es): {origin}")

    # ---- compose (B2) -----------------------------------------------------------------
    with in_stage(tracer, "compose", "Compose (B2)"):
        composed = E.operator.compose(record, branches)
        routed = route_with_flags(main, composed.flags) if main is not None else None
        tracer.data("compose", "composition", resolution=composed.resolution.value, reason=composed.reason,
                    flags=composed.flags, selected_passage=composed.selected_passage,
                    merged_pairs=composed.merged_pairs, kept=[b.branch_id for b in composed.branches],
                    route=({"action": routed.action.value, "reason": routed.reason, "flag": routed.flag}
                           if routed else None))
        tracer.log("compose", f"{composed.resolution.value}: {composed.reason}")
        summary["resolution"] = composed.resolution.value

    # ---- generate (B3) + verify ----------------------------------------------------------
    with in_stage(tracer, "generate", "Generate the answer (B3)"):
        gen = E.llm_generator if (live and req.generation == "llm" and E.llm_generator) else E.template
        answer = gen.generate(composed, query)
        tracer.data("generate", "answer", text=answer.text, method=answer.method, api_calls=answer.api_calls,
                    citations=sorted({b.supporting_passage for b in composed.branches if b.supporting_passage}))
        tracer.log("generate", f"{answer.method} answer, {len(answer.text)} characters")

    with in_stage(tracer, "verify", "Verify faithfulness"):
        f = answer.faithfulness or check_faithfulness(answer.text, composed.branches)
        grounded = all((b.supporting_passage in by_id) for b in composed.branches if not b.is_default)
        tracer.data("verify", "faithfulness", ok=f.ok, missing_branches=f.missing_branches,
                    missing_attribution=f.missing_attribution, unsupported_numbers=f.unsupported_numbers,
                    adjudicated=f.adjudicated, grounded=grounded)
        tracer.log("verify", "faithful: every kept branch stated, attributed, no invented figure"
                   if f.ok else "unfaithful: " + f.render().splitlines()[0], "info" if f.ok else "warn")
    summary.update(answer=answer.text, method=answer.method, faithful=f.ok)

    # ---- arena: the same input through other systems -------------------------------------
    arena: list[dict[str, Any]] = [{"system": "scopecompose", "name": "ScopeCompose", "text": answer.text,
                                    "kept": [b.branch_id for b in composed.branches], "method": answer.method}]
    with in_stage(tracer, "arena", "Baselines on the same input"):
        concat = " ".join(p.text for p in record.passages)
        arena.append({"system": "concat", "name": "Standard RAG (all passages, unresolved)",
                      "text": concat, "kept": [b.branch_id for b in branches], "method": "concatenation"})
        pool = [by_id[i] for i in main.key] if main is not None else record.passages
        chosen = _credibility_pick(pool, E.operator.credibility_order)
        sel_branches = [b for b in branches if b.supporting_passage == chosen.id] or branches[:1]
        from composition.operator import ComposedAnswer, Resolution

        sel_text = render_template(ComposedAnswer(record.query_id, Resolution.SELECTED, sel_branches,
                                                  reason="selection", selected_passage=chosen.id))
        arena.append({"system": "selection", "name": f"Selection (keeps {chosen.id}, most credible)",
                      "text": sel_text, "kept": [b.branch_id for b in sel_branches], "method": "credibility selection"})
        if live and req.long_context_baseline:
            from baselines.structured_long_context import run_instance

            lc = run_instance(SimpleNamespace(instance_id=record.query_id, query=query,
                                              passages=record.passages), variant="structured")
            arena.append({"system": "long_context", "name": "Long-context LLM (structured)",
                          "text": lc.raw_text if lc.ok else f"(failed: {lc.error})",
                          "kept": [], "method": "LLM", "branches": [_branch_view(b) for b in lc.branches]})
        tracer.data("arena", "answers", systems=arena)

    # ---- reader ---------------------------------------------------------------------------
    with in_stage(tracer, "reader", "Who is this answer right for?"):
        dims = B.reader_dims(branches)
        if req.reader:
            target = B.branch_for_reader(branches, req.reader)
            verdicts = []
            for sys_ in arena:
                right = None if target is None else _states(target.outcome, sys_["text"])
                verdicts.append({"system": sys_["system"], "right": right})
            tracer.data("reader", "verdicts", reader=req.reader, dims=dims,
                        branch=target.branch_id if target else None,
                        membership={b.branch_id: B.satisfies(b, req.reader) for b in branches},
                        verdicts=verdicts)
        else:
            tracer.data("reader", "dims", dims=dims, reader=None)
            tracer.log("reader", "no reader set: set one to see who each answer is right for")
            tracer.data("reader", "skipped", reason="no reader set")

    # ---- score against gold --------------------------------------------------------------
    with in_stage(tracer, "score", "Score against gold"):
        if inst is not None and inst.gold_branches:
            rows = []
            for sys_ in arena:
                if sys_["system"] == "scopecompose":
                    j = judge_branches(sys_["text"], inst.gold_branches, predicted_branches=composed.branches)
                    instrument = "structural (branch alignment)"
                else:
                    j = _judge_text(sys_["text"], inst.gold_branches)
                    instrument = "textual (clause matcher, number and polarity veto)"
                rows.append({"system": sys_["system"], "name": sys_["name"], "instrument": instrument, **_rates(j)})
            tracer.data("score", "preservation", rows=rows, gold_branches=len(inst.gold_branches),
                        gold_answer=inst.gold_scoped_answer)
            summary["scores"] = {r["system"]: {"PR": r["PR"], "SR": r["SR"]} for r in rows}
        else:
            tracer.log("score", "no gold branches for this input: nothing to score against")
            tracer.data("score", "skipped", reason="no gold branches for this input")

    summary.update(query=query, branches=[_branch_view(b) for b in composed.branches],
                   llm=dict(tracer.llm), stage_ms={k: round(v, 1) for k, v in tracer.stage_ms.items()})
    return summary
