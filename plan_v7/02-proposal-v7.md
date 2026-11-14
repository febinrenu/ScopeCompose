# Beyond Source Selection: Detecting and Measuring Cross-Document Exception Suppression in Conflict-Aware Retrieval-Augmented Generation

**Research proposal — Revision 7 (final after two internal review rounds), 2026-10-30. Supersedes v6.**

This revision is rebuilt around what the ScopeCompose pilot actually measured; `01-critical-review-and-gap-analysis.md` explains why. It was then revised again after an independent strict review (`04-second-round-review-and-responses.md`). It is written in the register of the eventual journal paper so that sections can be lifted into it directly.

> **What changed from v6**
>
> 1. **The contribution moves from the resolution operator to the stage that fails, and to the harm it causes.** In the pilot, the operator preserves 81 of 83 gold branches when it is given gold inputs; the detector recovers 2 of 67 conflicts.
> 2. **The headline metric is replaced.** Containment-based Preservation Rate is gamed by verbosity. The new primary endpoint is *case-complete accuracy*: from the answer alone, can a reader determine the correct outcome for a user in every branch's scope? It is reported alongside scoped precision and answer length.
> 3. **The benchmark becomes a RAG benchmark built on real text.** It has chunk-level K-passage contexts, branch annotation on real passages, personas, and contrast sets. Test construction never uses any detector we evaluate.
> 4. **The system the paper argues against is actually run.** ConflictRAG-style selection is reimplemented with a fidelity check. Strong prompting baselines are run on the same backbones with the same inputs.
> 5. **New prior art is handled explicitly.** This covers MDCR, Xie & Du, InsLogicBench, TransplantQA, CitizenQuery-UK, and the 2026 results on omission-blind LLM judges. SG-DT's own limitations sentence states the cross-document gap in its authors' words.

---

## Abstract

Conflict-aware retrieval-augmented generation (RAG) systems detect disagreement among retrieved passages, type it, and resolve factual conflicts by selecting the most credible source. This design can silently delete a common and consequential kind of disagreement: a general rule, and a valid exception to it, stated in separately retrievable documents. An example is *"international transactions incur a 3% fee"* alongside *"fees are waived for premium-tier cardholders."* Both statements are true. Selecting either returns an answer that is wrong for an identifiable group of users, and single-reference answer-correctness metrics do not register the loss. Chen et al. (KDD 2026) named the within-provision form of this failure *Silent Scope Omission*. We study its cross-document form.

We contribute four things:

1. **A formalisation.** Passages are represented as query-anchored *normative frames*: a decision variable, an applicability set and an outcome. This yields a scope-relation taxonomy that states exactly when source selection is the correct operation (identical scopes) and when it is necessarily lossy.
2. **XScope**, a benchmark built from real UK government guidance and UK bank terms. It provides:
   - chunk-level retrieval contexts;
   - double-annotated branch structures;
   - user personas whose correct answers differ by branch;
   - human-verified contrast sets for relations that published policy almost never contains.

   No detector we evaluate is used to construct the test set.
3. **A judge-light evaluation protocol.** It centres on *case-complete accuracy*: a reader model outside the evaluated families must recover, from the answer alone, the correct outcome for a persona in every branch's scope. Scoped precision and length are reported alongside. The protocol is validated against human labels and against adversarial hedging and concatenation.
4. **QNFA and an empirical study.** QNFA is a detection method that aligns passages by the decision variable they govern rather than by surface overlap; it has a distilled 184M-parameter variant. The study covers nine baselines and three pipeline variants across three open-weight backbones (7B–120B). An oracle cascade attributes the end-to-end error to individual stages.

A pilot on 128 instances, 42 of them LLM-authored probes, motivates the design:

- A lexical-NLI detector recovered 2 of 67 conflict-labelled pairs.
- Given gold routing, composition preserved 81 of 83 branches (probes).
- A single-passage reranker and an NLI filter deleted 49% and 6% of gold branches respectively (probes).
- The containment metric common to this literature scored verbatim concatenation above the human-written answer.

We pre-register three confirmatory hypotheses and a gated fourth, and size the real-text test set for a minimum detectable paired difference of about 10–13 points.

---

## 1. Introduction

### 1.1 The failure

A RAG system answering *"Will I be charged for withdrawing cash abroad?"* may retrieve a tariff stating a per-withdrawal charge and a product page stating that one account type includes fee-free withdrawals. The two passages disagree on the outcome without disagreeing on the facts: the second narrows the scope of the first.

Current conflict-aware pipelines have no type for this case. ConflictRAG (Wang et al., 2026, arXiv) types inter-document conflicts as follows:

| Type | Resolution |
|---|---|
| Factual | Entropy-TOPSIS credibility selection |
| Temporal | Recency |
| Opinion | Multi-perspective synthesis |

A rule-and-exception pair is typed factual (or temporal) and resolved by selection. Whichever passage wins, the answer is wrong for one class of users.

We call this **cross-document exception suppression**. Chen et al.'s benchmark is explicitly "scoped to intra-provision defeasible scope (not precedent, multi-document synthesis, or open-textured standards)" (their §7). Our setting is the multi-document case they set aside: the rule and its exception are retrieved separately. A pipeline must therefore first *recognise* that two differently worded passages govern the same decision before anything can preserve the exception.

### 1.2 Why it is invisible

Answer correctness compares an output with a single reference. A dataset that stores one answer per question stores the general rule, so a system that suppresses the exception scores perfectly. The users harmed are exactly those the exception was written for.

Measuring the harm therefore needs an instrument that:

1. checks every case, not one reference; and
2. cannot be satisfied by pasting the retrieved passages.

Our pilot found that the containment-based preservation metric used to argue for composition fails the second test. It scored a verbatim dump of both passages (0.855) above the human-written scoped answer (0.699).

### 1.3 Where the remedy breaks

We built the obvious remedy end to end — detect, type, relate scopes, extract conditions, compose — and measured each stage. Given gold routing and gold branches, composition preserved 81 of 83 branches on the 42 branch-annotated probe instances. The detector in front of it recovered 2 of 67 conflict-labelled pairs across all 128 pilot instances, and fired on none of the 42 probes.

The mechanism is specific:

- The detector gates on lexical topicality.
- A rule and its carve-out are typically written by different teams for different audiences, so they share little vocabulary.
- The gate therefore suppresses exactly the pairs it exists to find.

One exploratory head-to-head run was also made: the pipeline against a structured-prompt baseline, on 12 probe instances. It scored the two arms with different code paths and is inconclusive (−12.5 points, 95% CI [−37.5, +12.5]). We report it rather than discard it.

The lesson is that composing branches is easy. Recognising that two passages speak to the same decision is hard. This paper targets that recognition step, together with an instrument that measures what failing at it costs users.

### 1.4 Contributions

- **C1 — Formalisation (§4).** We define normative frames, conditional conflict, and a scope-relation taxonomy with an explicit *same-scope* case, the only case where selection is correct. We prove that selection is lossy on refinement pairs (Proposition 1). We state when region-partition composition is sound and complete given correct frames (Proposition 2). Both propositions are elementary; their role is to locate the remaining difficulty in frame recovery.
- **C2 — XScope benchmark (§6).** Real UK government text (Open Government Licence) and UK bank terms (released as pointers). It has:
  - four strata, never pooled silently;
  - K = 6 chunk-level contexts;
  - personas;
  - human-verified contrast sets;
  - double annotation, with an external annotator on a 15% sample.

  The test set is constructed without any evaluated detector.
- **C3 — Evaluation protocol (§7).**
  - Case-complete accuracy, measured with a reader from outside the evaluated families.
  - Per-branch accuracy, scoped precision, attachment errors, and fabricated exceptions.
  - Validation against human labels, plus adversarial checks for unattached hedging, concatenation, and null answers.
- **C4 — QNFA and the empirical study (§5, §9).**
  - A detector that aligns frames by query-anchored decision variable.
  - A distilled DeBERTa-v3 variant and a confidence cascade, sized for a 6 GB GPU.
  - A study of nine baselines (including a fidelity-checked ConflictRAG-style selector) and three pipeline variants on three backbones.
  - An oracle cascade that attributes end-to-end loss to retrieval, detection, scope relation, extraction and generation.

We do **not** claim:

- a new defeasible logic;
- the relation names (specificity, *lex specialis* and rule-base verification already use them);
- the discovery of within-provision scope omission.

---

## 2. Related work and positioning

### 2.1 Conflict-aware RAG

**CONFLICTS** (Cattan et al., 2025, arXiv:2506.08500) defines five inter-context conflict types with an expected behaviour for each, and releases an expert-annotated benchmark. The types are: no conflict, complementary information, conflicting opinions, outdated information, and misinformation.

**ConflictRAG** (Wang et al., 2026, arXiv:2605.17301) types conflicts as factual, temporal and opinion. It routes them as described in §1.1. Its detector is a frozen sentence encoder with an MLP, falling back to an LLM below a confidence threshold. It is trained on pairs re-derived from Xie et al.'s ConflictQA (ICLR 2024). That dataset is distinct from the "ConflictQA" of Zhao et al. (arXiv:2604.11209).

Other related systems:

| System | What it does |
|---|---|
| TCR (Ye et al., 2026, arXiv:2601.06842) | Makes context–memory conflict handling transparent |
| CARE-RAG (arXiv:2507.01281) | Synthesises one answer by conflict-driven summarisation |
| ArbGraph (arXiv:2604.18362) | Arbitrates evidence for long-form answers |
| MADAM-RAG (arXiv:2504.13079) | Aggregates the answers of separate agents |
| EvidentialRAG (arXiv:2607.10491) | Routes on Dempster–Shafer uncertainty |
| ConflictScore (arXiv:2606.26437) | Measures whether a response *acknowledges* conflict |
| Behavioural contracts (Zeng et al., EMNLP 2026) | Test compliance under evidence interventions |
| Yu et al. (arXiv:2605.27157) | Show that acknowledging contradictory evidence does not constrain the final recommendation |

**Where we sit.** None of these has a type in which both passages are true for different populations and one *defeats* the other inside its scope. The closest is CONFLICTS' *complementary information*, so the difference must be exact:

- **Complementary answers are additive and symmetric.** Two acronym expansions, or a city-dependent fact, are both kept, and consolidating them is correct. In our taxonomy these are `disjoint` pairs.
- **A conditional conflict is asymmetric.** The exception overrides the rule for a sub-population. Consolidating the two without attaching the condition produces the wrong answer for one group.

We test empirically whether CONFLICTS-style taxonomy-aware prompting handles disjoint pairs but suppresses refinement pairs (§3, H-T). TransplantQA (Li et al., arXiv:2605.29084) found that 75% of inter-source pairs in medical RAG are "complementary (differing in detail or scope)" and fewer than 0.1% are contradictory. It audits rather than resolves, and it has no defeat relation, but it supports our premise.

### 2.2 Conditional and defeasible question answering

**ConditionalQA** (Sun et al., ACL 2022) studies answers that hold only under conditions, over single UK-government documents with user scenarios. Rule and condition are always in view, so no resolution stage can discard either. Our persona device follows its scenario idea.

**MDCR** (Chen et al., Findings EMNLP 2024, arXiv:2406.11784) extends ConditionalQA across documents. It models relations between the eligibility conditions of *different* items to answer optimisation questions. It does not model one rule defeated by another for the same decision variable, has no conflict-resolution stage, and does not measure suppression.

Other related work:

| Work | Focus | Why it differs |
|---|---|---|
| CondAmbigQA (arXiv:2502.01523) | Conditions arising from query ambiguity | Not inter-document defeat |
| InsLogicBench (ACL 2026) | Defeasible insurance-policy logic; LLMs fail on exception clauses | Single-policy adjudication, no retrieval |
| Defeasible NLI (Rudinger et al., Findings EMNLP 2020) | Defeasible inference | No retrieval |
| DERELAB (EMNLP 2026) | Defeasible reasoning | No retrieval |
| DeFAb (arXiv:2606.18557) | Defeasible abduction | No retrieval |

### 2.3 Defeasible scope parsing

**SG-DT / NormBench** (Chen et al., KDD 2026, arXiv:2606.08932) parses one provision into a span-grounded deontic tree with explicit exclusion guards. It reports *Recursion Decay* as defeater depth grows, and classifies 37% of NormBench provisions as Level 3+ (recursive). Its motivation discusses RAG legal agents, but its task is "given a provision X, output an SG-DT", and its limitations exclude multi-document synthesis.

The two lines of work are complementary:

- **SG-DT** asks how the defeat structure of one provision should be represented.
- **We** ask how a RAG pipeline finds, across separately retrieved passages, the pairs that form such a structure, and what users lose when it does not.

SG-DT trees are a natural internal representation for frames (§12).

### 2.4 The symbolic lineage

Exceptions are classical knowledge representation:

- default logic (Reiter 1980);
- specificity-based override (Poole 1985), the `A → C` versus `A ∧ B → ¬C` pattern;
- defeasible logic and argumentation (Pollock 1987; Prakken & Sartor 1997; Dung 1995);
- *lex specialis* in AI and law;
- rule-base verification, which already names disjoint, redundant and conflicting rule pairs;
- preference handling over inconsistent knowledge bases (Bienvenu et al., KR 2025).

Xie & Du (arXiv:2604.17340) translate clinical guidelines into logic and check them with a SAT solver offline.

We operationalise these relations **at query time, over retrieved natural-language passages, inside a RAG resolution stage**, and evaluate their consequence for answers. The `opposed` relation (overlapping, non-nested scopes with disagreeing outcomes) is the case specificity does not settle. Our operator presents it as contested.

### 2.5 Completeness evaluation and judge validity

**Claim-level evaluation.**
- FActScore (EMNLP 2023), ALCE (EMNLP 2023), RAGChecker (NeurIPS 2024 D&B) and nugget evaluation (Pradeep et al., SIGIR 2025) evaluate claims and completeness.
- CitizenQuery-UK (arXiv:2602.04064) applies FActScore-style factuality to 22k synthetic gov.uk queries. It shares our data source but has no conflict or exception structure.

**Judge validity.**
- Judging the Judges (GEM 2025), JUDGE-BENCH (ACL 2025) and Rao & Callison-Burch (arXiv:2606.00093) cover judge validation and agreement reporting.
- **Two 2026 results matter directly for suppression measurement:**
  - LLM judges detect added or altered content far better than omissions (Fox et al., NeurIPS 2026, arXiv:2608.31016).
  - LLMs judge better than they *enumerate*, and LLM-written enumerations silently omit items (arXiv:2608.01000).

Our design responds to both. Omissions are never left for a judge to notice. The reader is asked one positive question per gold-derived persona, and the persona list comes from annotated gold, not from a model's enumeration.

### 2.6 Positioning matrix (criteria defined in the paper's appendix)

| Work | Cross-document | Query-time over retrieved text | Defeat represented | Resolution changes the answer | Harm to exception-scope users measured |
|---|:-:|:-:|:-:|:-:|:-:|
| CONFLICTS | ✓ | ✓ | — | prompt-level | — |
| ConflictRAG | ✓ | ✓ | — | selection | — |
| ConditionalQA | — | — | conditions | — | scenarios |
| MDCR | ✓ | — | condition relations across items | — | partial |
| SG-DT / NormBench | — | — | ✓ | — | — |
| Xie & Du | ✓ | — (offline) | ✓ | — | — |
| **This work** | ✓ | ✓ | ✓ | composition | ✓ |

### 2.7 Naming

Gokul et al. (arXiv:2504.00180) use "conditional contradiction" for a different phenomenon: a third document that *creates* a contradiction. We use "conditional conflict" only as defined in §4, and prefer the descriptive term "cross-document exception suppression" for the failure.

---

## 3. Research questions and hypotheses

| | Question | Primary measure |
|---|---|---|
| **RQ1** (harm) | How much do selection-style and published conflict-aware resolvers suppress valid exceptions on real text? Is the loss invisible to single-reference correctness, and does it concentrate on under-specified queries? | Case-complete accuracy (CCA-all); per-branch CCA |
| **RQ2** (attribution) | Where does end-to-end loss arise: retrieval, detection, scope relation, extraction or generation? | Oracle-cascade deltas |
| **RQ3** (detection) | Does aligning passages by query-anchored decision variable raise conditional-conflict detection on natural cross-document pairs, and can a distilled model keep most of the gain at a fraction of the cost? | Conditional-class F1 at dev-fixed operating points; tokens per query |
| **RQ4** (resolution) | On the same backbone and inputs, does explicit scope checking beat the best end-to-end structured prompting? Does the benefit change with model scale? | CCA-all; scale interaction |
| **RQ5** (measurement) | Are the metrics valid: in agreement with humans, robust to concatenation and hedging, and not reliant on omission-blind judging? | Validity gates V1–V3 |

**Primary endpoint.** **CCA-all**: the proportion of conditional instances for which the reader recovers the gold outcome for *every* persona from the system's under-specified answer.

Why this endpoint:
- It credits a system only when every branch is usable.
- Selection that favours either branch is penalised, whichever one it picks.
- Unattached hedging is penalised by construction (§7.2).

**Confirmatory family** (pre-registered; H1–H3 Holm-adjusted; H4 tested only if H3 is rejected).

- **H1 (harm).** On real-text conditional test instances, the ConflictRAG-style resolver (B3) has lower CCA-all than full-context generation (B0) on the primary backbone (gpt-oss-120b).
- **H2 (detection).** On natural (S1+S2) test pairs, QNFA's conditional-class F1, at an operating point fixed on dev, exceeds that of the best non-QNFA detector (D0–D4, chosen on dev). The 15-point minimum effect of interest is reported as a one-sided 95% lower confidence bound.
- **H3 (resolution).** On the primary backbone, the hybrid pipeline O3 has higher CCA-all than the better of B7 and B8. That comparator is chosen on dev and fixed in the pre-registration.
- **H4 (scale; gated on H3).** The O3 − B7 CCA-all difference is larger at gpt-oss-20b than at gpt-oss-120b (same model family). It is tested as a risk-difference interaction.

**Secondary hypotheses** (pre-specified, unadjusted, reported with CIs).

- **H-T (taxonomy gap).** Measured against O3, B5 (CONFLICTS-style taxonomy-aware prompting) has a larger CCA-all deficit on `refinement` instances than on `disjoint` instances (difference-in-differences). This is the empirical evidence that the existing taxonomy lacks a defeat type.
- **H-C (concentration).** The B3 − B0 gap is smaller in the persona setting, where user attributes are stated, than in the under-specified setting.
- **H-F (frugality).** Distilled QNFA retains at least 80% of QNFA's F1 at no more than 15% of its API tokens.

**Validity gates.** These are checked **before** confirmatory testing, on held-out validation data built like test.

- **V1 (verbosity).** Each metric has its own criterion:
  - Under legacy containment PR, concatenation ≥ gold (a replication of the pilot).
  - Under CCA-all, gold ≥ concatenation. Concatenation may be determinate, and CCA does not claim to penalise it.
  - Under scoped precision, gold > concatenation.
  - Unattached-hedge answers score CCA-all ≤ 0.05.
- **V2 (reader).** The reader agrees with two humans at ≥ 0.90. NOT_DETERMINED recall is ≥ 0.90 on an adversarial set of unattached hedges and partial answers. The reader returns NOT_DETERMINED for ≥ 0.95 of personas on null answers.
- **V3 (judge).** The enumerate-then-verify judge used for scoped precision reaches ≥ 0.80 accuracy on human-labelled claim decisions.

H1 and V1–V3 are expected to hold. H2 is the main methodological bet. H3 and H4 are open; §11 states what the paper becomes under each outcome.

---

## 4. Problem formulation

### 4.1 Normative frames

Let *q* be a query and *D_q = (d₁, …, d_K)* the retrieved passages. A **normative frame** is *f = (v, A, o, m, s)*, where:

| Symbol | Meaning |
|---|---|
| *v* | A **decision variable** that *q* asks about, e.g. "charge for a cash withdrawal abroad". Variables are anchored to the query. |
| *A ⊆ U* | An **applicability set** over a case space *U* of user and situation attributes. It is given as typed constraints (categorical, numeric range, boolean, date), or as ⊤ for an unconditioned default. |
| *o ∈ Y_v* | An **outcome** in the variable's value space: an amount, a boolean permission, or a category. |
| *m* | A **modality**: obligation, permission, prohibition or statement. |
| *s* | The **supporting span**, which must be verbatim text from some *d_i*. |

A frame extractor *Φ(q, d)* maps a passage to zero or more frames. A passage that discusses withdrawals but not charges yields no frame for this *q*.

### 4.2 Relations

Two frames *f_i, f_j* with *v_i = v_j* are in exactly one of these relations.

| Relation | Applicability | Outcomes | Correct behaviour |
|---|---|---|---|
| **Refinement** | *A_j ⊊ A_i* | differ on *A_j* | Compose: *o_i*, except under *A_j*, where *o_j* holds (specificity) |
| **Disjoint** | *A_i ∩ A_j = ∅* | — | Keep both; no conflict (CONFLICTS' "complementary") |
| **Redundant** | overlap or nested | agree on the overlap | Merge. Branching here **fabricates** an exception. |
| **Opposed (crossing)** | overlap, neither contains the other | differ on the overlap | Partition: non-overlapping parts as stated, the overlap reported as contested |
| **Same-scope conflict** | *A_i = A_j* | differ | Factual or temporal conflict. **The only case where source selection — by credibility or recency — is the right operation.** |

A pair is a **conditional conflict** iff its relation is refinement or crossing opposition. Disjoint and redundant pairs are **not** conflicts.

**Correspondence with the pilot's labels.** The pilot's label set:
- labelled disjoint and redundant pairs as type `conditional`;
- folded same-scope and crossing pairs into one label, `opposed`;
- sent undecidable scope to `opposed`, and from there to selection.

v7 relabels deterministically:

| Pilot label | v7 label |
|---|---|
| `conditional` ∧ `disjoint` | `no_conflict` (complementary) |
| `conditional` ∧ `redundant` | `no_conflict` (redundant) |
| `opposed` | split by an `opposition` sub-label: `equal` / `crossing` / `unknown` |

Undecidable scope is presented as contested rather than selected. Routing uncertainty into selection would reintroduce exactly the suppression we measure; the old policy is kept as an ablation.

Under the v7 definitions, the pilot contains **18 natural conditional conflicts**.

**Multi-exception structure.** Several frames on one variable induce a partition of *U* into regions (Venn cells).
- **Nested exceptions** form chains of strict subsets, and each region takes its most specific frame.
- **Crossed exceptions** are incomparable sets with differing outcomes, and create a contested region.

### 4.3 Two propositions

**Proposition 1 (selection is lossy on refinement pairs).** Consider a refinement pair with *A_e ⊊ A_d* and *o_e ≠ o_d*, and any answer that asserts a single unconditioned outcome for *v*. That answer is wrong on every case in *A_e* or on every case in *A_d \ A_e*. Its case-complete accuracy is therefore 0.

*Proof.* One outcome cannot equal both *o_e* and *o_d*. ∎

The proposition is elementary. Its consequences are empirical:
- whether real resolvers in fact output a single outcome;
- which branch they choose;
- whether harm concentrates where the query leaves the user's attributes unstated (H-C).

**Proposition 2 (partition composition).** Suppose:
- the frames are correct;
- their constraints are decidable over typed attributes.

Then assigning each region its most specific applicable frame's outcome, and marking incomparable disagreements as contested, is:
- **sound**: no region receives an outcome from a frame that does not apply to it, or that a more specific applicable frame overrides;
- **complete**: every non-empty region is covered;
- **order-invariant**: the output depends only on the multiset of frames.

*Proof sketch.* The partition and the per-region assignment are functions of the set of constraints. Specificity is a partial order on the sets containing a region, and each region has either a unique minimal applicable frame or incomparable minimal frames. ∎

**What the propositions establish.** The remaining difficulty is entirely in recovering correct frames, which is why this paper is about detection and measurement and not about the operator.

### 4.4 Scope of the claims

- English; UK government and UK banking text.
- Applicability is decided arithmetically when it is typed, and by entailment otherwise. Every decision records which.
- First-order frames. Nested and crossed structure is resolved by partition where attributes are typed, and flagged otherwise.

---

## 5. Method

### 5.1 Pipeline

```
query q, context D_q (K = 6 chunks)
  └─► [A] Frame extraction Φ(q, d): one batched call per query, cached, on the SAME backbone as the cell
       └─► [B] Decision-variable alignment → candidate pairs sharing v
            └─► [C] Conflict test (outcome mismatch) + scope relation (typed algebra ▸ entailment fallback)
                 └─► [D] Implicit-condition recovery (Contrastive Scope Probing) for A = ⊤ frames that lose a refinement test
                      └─► [E] Region-partition composition (specificity; contested regions flagged)
                           └─► [F] Verbalisation (template | backbone-constrained) + coverage and faithfulness checks
```

Each stage has a gold-input oracle; experiment E4 uses these oracles for attribution. In every resolution experiment, **frames are extracted by the backbone of the cell being evaluated**, so that a 7B cell receives no help from a larger model. The detection experiment (E2) uses one fixed extractor (gpt-oss-20b) for all detectors that need one.

### 5.2 Query-anchored frame extraction (A)

**One call per query.** A single call receives *q* and all K passages and returns, per passage, a JSON list of frames:
- `variable`;
- `applicability`: typed attributes plus a descriptor;
- `outcome`: type, normalised value and unit;
- `modality`;
- `span`.

**Validation.**
- Every span must be a verbatim substring of its passage. A non-verbatim span triggers one re-ask; if it fails again, the frame is dropped and counted.
- Outcome values are normalised by deterministic code (currency, percentages, durations, polarity, a small per-domain categorical vocabulary).

**Cost.** Cost is *O(K)* tokens per query, rather than *O(K²)* pairwise calls.

### 5.3 Decision-variable alignment (B)

A pair of frames is a candidate when:
1. the Contriever cosine similarity of their `variable` strings is at least a dev-tuned threshold;
2. both frames answer *q*.

An LLM tie-break is used only inside an uncertainty band. This replaces lexical-topicality gating, the identified cause of the pilot's failure. A rule and its exception share a decision variable far more reliably than they share vocabulary.

### 5.4 Conflict test and scope relation (C)

- **Conflict:** an outcome mismatch on an aligned pair.
- **Scope relation:** the project's typed attribute algebra (`compare`, `intersects`, `narrower`) decides it, with bidirectional entailment over descriptors as the fallback.
- **Type** follows from the relation (§4.2). Undecidable scope becomes `opposed/unknown`, which is presented as contested.

**Scores.** Every detector must output a score so that operating points can be fixed on dev:
- **QNFA:** alignment similarity × outcome-mismatch confidence.
- **LLM-label detectors:** a calibrated verbal confidence elicited in the same call.

### 5.5 Region-partition composition (E)

The operator enumerates the regions induced by typed constraints. Each region takes the outcome of its most specific applicable frame; incomparable disagreements are marked contested, with both sources cited. When constraints are untyped, it falls back to the pilot operator: compose, merge or select, with nested and crossed flags.

We claim no novelty for the operator beyond its operationalisation.

### 5.6 Distilled detector and cascade

The deployment target is a 6 GB GPU with capped API access.

**Model.** A DeBERTa-v3-base cross-encoder over `[q] [SEP] d_i [SEP] d_j`, with seq 512 and pair-aware truncation, so that the end of each passage is preserved. It has two heads:
- **type** (3-way): conditional conflict / same-scope conflict / no conflict;
- **relation** (5-way).

The five-class pilot type set is collapsed because its factual and opinion cells hold 4 and 2 examples.

**Training data.**
- Gold train instances.
- Silver labels from QNFA over mined pairs, *from train and dev document groups only*. Mined pairs have no query, so a query is synthesised from the rule passage's decision variable and recorded.

**Cascade.** Following the confidence-cascade practice of recent conflict-typing work, pairs inside a dev-tuned band escalate to QNFA. We report the full cost–F1 curve.

### 5.7 Implicit conditions: Contrastive Scope Probing (D)

When an *A = ⊤* frame contradicts a co-referential frame, the condition is implicit; a product page, for example, never says "unlike the standard tariff". Contrastive Scope Probing (implemented in `extraction/probing.py`) works in three steps:
1. generate counterfactual questions;
2. propose candidate conditions;
3. admit a candidate only through an entailment grounding gate.

It is evaluated as a component, on explicit and implicit instances separately, with fabrication reported next to recall. Its pre-registered go/no-go (WP1) decides whether it enters O2 and O3.

### 5.8 Our systems

| System | Definition |
|---|---|
| **O1 ScopeCompose-local** | Distilled detector + cascade, partition composition, template verbalisation. Frugal. |
| **O2 ScopeCompose-LLM** | QNFA on the cell's backbone, partition composition, backbone-constrained verbalisation, with a **coverage check**: every branch outcome value must appear. One retry, then template fallback (logged). |
| **O3 ScopeCompose-hybrid** | The backbone first produces the B7 structured parse. The parse is then checked against QNFA frames from the same backbone: missing aligned frames are injected, redundant branches merged, misattached conditions repaired. Finally it is composed and verbalised. Each repair is logged and can be ablated. O3 isolates the value of explicit scope checking *on top of* the strongest prompting strategy. |

---

## 6. The XScope benchmark

### 6.1 Sources and licence

| Domain | Source | Release |
|---|---|---|
| Immigration and benefits eligibility | gov.uk via the Content API, which yields `content_id`, `document_type` and `public_updated_at` | Open Government Licence v3.0. Text released with attribution. |
| Consumer banking terms | Public pages of UK banks | Copyright. Released as URL, character offsets and SHA-256 of the span, with a rehydration script. |

A US federal public-domain slice was considered and **dropped** to protect the annotation budget (§12).

### 6.2 Strata

**Rules for reporting strata:**
- Every pooled figure is accompanied by its per-stratum breakdown.
- S4 and S5 are never pooled with S1–S3.

| Stratum | Definition | Pilot (v7 labels) | Role |
|---|---|---:|---|
| **S1 natural cross-document** | Rule and exception in different content items (distinct `content_id`) | 14 conditional conflicts | Primary for detection (H2); part of the real-text pool |
| **S2 natural same-guide** | Different parts of one gov.uk guide (one `content_id`) | 4 | As S1 |
| **S3 real-text split** | A real sentence stating rule and carve-out, split at the carve-out into two retrievable units | 45 built, unlabelled; ≈ 256 splittable sentences in the mined pool | Part of the real-text pool; controlled |
| **S4 contrast** | Human-verified minimal edits of S1–S3 instances that create `redundant` (exception outcome aligned to the default's) or crossing `opposed` (scope shifted to cross the default's) | 4 authored, unverified | Diagnostic for rare relations; never counted toward prevalence |
| **S5 authored probe** | The 54 LLM-authored WP1 probes | 54 | **Dev and diagnostics only** |

Stratum is derived from `separation` and annotation provenance. It is **not** derived from `construction`: the pilot's same-guide records are tagged `construction=split`.

### 6.3 Test construction without evaluated detectors

The test set must not be built by any system it evaluates.

**Mining the test set (S1/S2).** Candidates are generated by retrieval: each rule-bearing sentence retrieves cross-document chunks with BM25 + Contriever. A **verifier model that is not evaluated anywhere** (Qwen ≈ 27B, with its own prompt) filters candidates for human verification. QNFA, D3, and any backbone are excluded from test construction.

**Re-splitting.** The pilot's split places only 1 natural conditional conflict in test. Because no model has yet been trained or tuned on the pilot splits, the natural document groups are re-split before the freeze, and the old assignment is archived. Newly mined document groups go to test until the targets in §6.7 are met.

**QNFA-assisted mining** is permitted only for train, dev and silver data.

### 6.4 Contexts and passage granularity

**Granularity.** Pilot passages are single sentences (mean 17.7 words), which makes detection unrealistically easy.

- **Primary setting: chunk level.** All source documents are chunked once by one chunker (≈ 120–160 words, sentence-aligned). The chunk containing each gold sentence is the gold passage. There is no gold-centred chunking, which would leave a positional artefact.
- **S3 halves** are placed in two chunks that the same chunker produces from synthetic two-section documents.
- **Sentence level** is retained for comparison with the pilot.

**Context.** K = 6: the gold passages, plus hard negatives retrieved by BM25 + Contriever RRF from the same provider pool.

**Screening negatives.** The same non-evaluated verifier screens the negatives. Flagged negatives are **adjudicated by a human, never silently deleted**; confirmed conflicts are labelled. A random 20% of finished contexts is audited by a human, and the residual unlabelled-conflict rate is reported. K ∈ {4, 6, 10} is a sensitivity analysis.

**Settings.**
- **Oracle-inclusion** (primary): gold passages are guaranteed present.
- **End-to-end retrieval** (secondary): retrieval loss is recorded.

### 6.5 Personas

Every conditional instance has at least one persona per region; nested or crossed structure adds more. A persona is:
- a natural-language **statement** of attributes ("I hold a Global Traveller account");
- a gold short outcome.

**How personas are made.**
- Personas are derived from gold branch applicability, mapping attributes to branch with the existing `branch_for_reader`.
- Statements are paraphrased from the **source passage wording**, not from attribute identifiers, so the reader cannot match tokens with a system's output.
- They are verified by a human. The share of hand-written personas, and the lexical overlap between persona statements and each system's answers, are reported.

### 6.6 Annotation and agreement

**Annotation passes.**
- **Type and relation:** double-blind, using the existing tool and manual.
- **Branch structures and gold scoped answers:**
  - pre-annotated by the non-evaluated Qwen ≈ 27B;
  - corrected by annotator A, with the correction distance recorded;
  - re-annotated **from scratch** by annotator B on a stratified 30%, to measure anchoring.
- **External annotator:** a course peer labels 15% of instances on all three axes.
- **Adjudication:** by discussion, with a third reviewer for cases the manual leaves ambiguous.

**Agreement reported.**
- κ per axis **per stratum**, with informative-n, p_e and label composition. The pilot's 54 single-annotator probes are never pooled.
- Branch-level F1 between annotators.
- Krippendorff's α on persona outcomes.
- Pass-plausibility checks, which have already caught one invalid pass.

### 6.7 Size and power

The primary endpoint is binary per instance and paired across systems. For a paired McNemar test at α = 0.0167 (Holm's first step) and power 0.8, the minimum detectable difference (MDD) depends on the discordance rate *d*:

| Real-text conditional test instances | MDD at *d* = 0.15 | *d* = 0.25 | *d* = 0.35 |
|---:|---:|---:|---:|
| 60 | 16 | 20 | 24 |
| 100 | 12 | 16 | 19 |
| 160 | 10 | 13 | 15 |
| **200** | **9** | **11** | **13** |

Before the freeze, `power_sim.py` reproduces these by simulation at the instance level, using the discordance observed on dev.

**Targets for the test split.**

| Component | Target |
|---|---|
| Real-text conditional conflicts (S1–S3) | **≥ 200** |
| …of which natural (S1+S2) | **≥ 60** — the H2 population, and an external-validity subset for H1/H3 (descriptive, MDD ≈ 20) |
| Non-conflict controls (disjoint, redundant, no-conflict, same-scope) | ≥ 120 |
| Contrast instances (S4) | ≥ 40: ≥ 20 redundant, ≥ 20 crossing opposed |

**Dev** has ≈ 100 real-text instances built like test, plus S5. Dev is used for operating points and validity gates (§3).

### 6.8 Datasheet and release

- A datasheet (Gebru et al., 2021) with counts generated from the data.
- The annotation manual and its decision procedure.
- Pinned model identifiers and dates for every LLM-produced artefact.
- All cached model outputs, so every number regenerates offline.
- The κ pilot history.
- Snapshot dates of source pages.

---

## 7. Evaluation protocol

### 7.1 Answer settings

| Setting | Query | Gold |
|---|---|---|
| **Under-specified** (primary) | *q* as written | Branch set and scoped answer |
| **Persona** (secondary, for H-C) | *q* plus the persona statement | Short outcome |

**Shared conditions.**
- All systems receive the same context, the same output instructions, the same `max_tokens`, and the same reasoning setting.
- All answers are scored by the same code.
- Systems are instructed to answer rather than ask. Clarification-only answers are reported **separately**, as a rate per system, and excluded from CCA denominators in a sensitivity analysis.

### 7.2 Case-complete accuracy (primary)

**The reader.** A fixed reader model receives only the answer and one persona statement, and is asked: *"Based only on this answer, what is the outcome for this person? If the answer does not determine it, reply NOT_DETERMINED."* Its reply is matched deterministically against the gold outcome value.

The reader is a local 8B instruction model from a family **not** among the backbones (Llama-3.1-8B-Instruct), with `num_ctx` set explicitly and logged. It is pre-registered and is not changed after the freeze.

**Metrics.**

| Metric | Definition |
|---|---|
| **CCA-all** | Instance is correct iff every persona is correct (primary) |
| **CCA-E** | Exception personas |
| **CCA-D** | Default personas |
| **Over-generalisation rate** | Default persona answered with the exception's outcome |

**How CCA behaves on hedging and concatenation.**
- An answer listing both outcomes without saying who each applies to ("it may be £2.50 or free") determines no persona. It is scored NOT_DETERMINED.
- Verbatim concatenation of gold passages *can* be determinate. CCA does not claim to penalise it; conciseness is measured by §7.3, and the two are always reported together.

**Relation to prior work.** This is not Fox et al.'s intervention. It follows the same principle: per-item positive checks derived from gold.

### 7.3 Scoped precision and per-branch scores (secondary)

Answers are split into claims (sentence and clause splitting) and labelled:

| Label | Meaning |
|---|---|
| *aligned* | outcome and condition match a gold branch |
| *misattached* | outcome matches but the condition does not |
| *extraneous* | supported by the context but not in gold |
| *unsupported* | not supported by any passage |

Labelling is deterministic first (value and attribute matching), then the enumerate-then-verify judge (Qwen ≈ 27B, non-evaluated) for the residue, on a stratified 40% of answers.

**Metrics.**

| Metric | Definition |
|---|---|
| Scoped precision | aligned / all claims |
| Branch recall | branches with an aligned claim / branches |
| Attachment error | misattached / (aligned + misattached) |
| Fabricated-exception rate | Exceptions asserted on redundant and no-conflict controls |
| Unsupported-claim rate | unsupported / all claims |
| Answer length | words |

**Single-reference answer correctness** (the default outcome contained in the answer) is reported **descriptively only**, to show what one-answer datasets would record. It is not a hypothesis, because its blindness to suppression holds by construction.

### 7.4 Detection and relation metrics

**Two candidate sets, reported separately.**
- **Natural hard-candidate set** (primary for H2). All S1+S2 gold conditional pairs in test, plus curated same-provider, same-topic non-conflict pairs.
- **All within-context pairs.** C(K, 2) pairs per context. This set reflects false-alarm volume and cost.

**Metrics.**
- Conditional-class F1 at operating points fixed on dev, with precision and recall;
- PR-AUC for score-based detectors;
- the conditional/same-scope confusion cell;
- relation accuracy (separately on S4);
- the share of relation decisions made by typed attributes;
- tokens per query, calls per query, and wall-clock time on the 6 GB profile.

### 7.5 Validity studies (E5)

- **Verbosity and hedge battery.** Seven outputs per validation instance:
  - gold;
  - selection;
  - concatenation of the gold passages;
  - concatenation of all K passages;
  - gold padded with extraneous sentences;
  - unattached hedge;
  - null.

  Each metric is checked against its own V1 criterion.
- **Human labels.** Both annotators label 200 reader decisions, 300 claim decisions and 200 branch decisions, blind to system identity. The external annotator labels 100 of them.
- **Reported statistics:** accuracy, κ, per-class recall (including NOT_DETERMINED recall), and Kendall τ between system rankings under human and automatic scoring.
- **Stability.** Two repeated runs of the API systems on 50 instances; per-instance agreement and SD are reported.

---

## 8. Systems compared

### 8.1 Systems

All systems receive the same K-context, backbone, output instructions and reasoning setting.

| ID | System | Represents |
|---|---|---|
| B0 | Full-context generation | Vanilla RAG |
| B1 | Rerank → top-1 → generate | Crude selection |
| B2 | NLI filter → generate | What a careful engineer ships |
| **B3** | **ConflictRAG-style** (details below) | The design we critique |
| B4 | Conflict-driven summarisation (CARE-style prompt) | Synthesis into one answer |
| B5 | Taxonomy-aware prompting (CONFLICTS) | The best published prompt-level conflict handling |
| B6 | Free-text "state any conditions or exceptions" | Naive hedging |
| **B7** | Structured branch parse (SG-DT-style JSON, rendered). Schema and instructions adapted from the released NormBench prompts where applicable; deviations listed. | The strongest end-to-end alternative |
| B8 | B7 + self-verification (enumerate passages, check each branch, revise) | Guards against a strawman B7 |
| O1–O3 | Our pipeline variants (§5.8) | — |

**B3 in detail.**
- **Detector.** A ConflictRAG-style detector — a frozen MiniLM with an MLP and LLM fallback — trained as that paper describes: three types plus no-conflict, on pairs re-derived from ConflictQA. It is **not** trained on our labels, so conditional pairs are typed as its taxonomy dictates.
- **Routing:** factual → Entropy-TOPSIS selection; temporal → recency; opinion → multi-perspective synthesis.
- **TOPSIS criteria:**
  - authority, from `source_type`, enriched from gov.uk `document_type`;
  - recency, from `date`, enriched from `public_updated_at`;
  - relevance;
  - specificity;
  - consistency.
- **Fidelity.** A fidelity check against the paper's reported detector accuracy is run. Specificity is ablated, because it can favour the exception passage. Every choice the source paper leaves unspecified is listed.

### 8.2 Backbones and evaluation models

**Backbones** (open weights; identifiers pinned in the pre-registration manifest):

| Backbone | Size | Role |
|---|---|---|
| Qwen2.5-7B-Instruct (Q4, local) | 7B | Small |
| gpt-oss-20b (Groq) | 20B | Medium |
| gpt-oss-120b (Groq) | 120B | Large, **primary** |

The two gpt-oss backbones share a family, so H4 is not confounded by family.

**Evaluation and construction models** (never evaluated as systems):

| Model | Role |
|---|---|
| Qwen ≈ 27B (Groq) | Mining verifier, negative screener, branch pre-annotator, claim judge |
| Llama-3.1-8B-Instruct (local) | Reader |

**Compute matching.** O3 makes more calls than B7. We report tokens and calls per system, and B8 is the compute-closer comparator. H3's comparator is the better of B7 and B8, chosen on dev.

**Optional closed frontier check (P2).** If a free tier is available at run time, run B7, B8 and O3 on the S1–S3 conditional test set. Without it, all conclusions are stated for open-weight models.

---

## 9. Experiments

| ID | RQ | Design | Output |
|---|---|---|---|
| **E1 Harm** | RQ1 | Primary backbone: B0, B3, B4, B5, B7, B8, O3 on real-text conditional test (under-specified); B0, B3 in the persona setting; all 12 systems on the 7B backbone | CCA-all, CCA-E, CCA-D, single-reference AC (descriptive); H1, H-C, H-T |
| **E2 Detection** | RQ3 | D0 (pilot heuristic), D1 (NLI), D2 (ConflictRAG-style), D3 (zero-shot LLM pair), D4 (fine-tuned on gold), D5 (QNFA), D6 (distilled), D7 (cascade); fixed extractor | F1 at dev-fixed points, PR-AUC, cost curve; H2, H-F |
| **E2b QNFA ablations** | RQ3 | Remove one at a time: query anchoring; variable alignment (whole-passage embedding instead); typed attributes (entailment only); outcome normalisation | Which component carries the gain |
| **E3 Resolution & scale** | RQ4 | B0, B3, B7, B8, O3 on gpt-oss-20b and gpt-oss-120b; all systems on 7B | H3, H4; Scoped precision; length |
| **E4 Oracle cascade** | RQ2 | O2 with gold substituted stage by stage (retrieval → detection → relation → extraction → generation); 7B and 20b | Loss attributed per stage; order-averaged as a robustness check |
| **E5 Validity** | RQ5 | §7.5 | V1–V3 |
| **E6 Robustness** | — | On 7B, unless noted: order permutation (existing gate); K ∈ {4, 6, 10}; sentence vs chunk; explicit vs implicit; S4; end-to-end retrieval; unknown-scope policy; B3 without specificity | Sensitivity tables |
| **E7 Cost** | — | All systems | Tokens, calls, latency; Pareto front of CCA-all vs cost |
| **E8 External (P1, gated)** | — | Cross-passage variant of ConditionalQA (rule and condition placed in separate passages), if a week-3 feasibility check finds that ≥ 40% of sampled items map to refinement structure with both outcomes stated, and its licence permits | Transfer of E1/E3 findings to an established dataset |

---

## 10. Statistical analysis plan

**Units.** The unit is the **instance**. All comparisons are paired on identical instances.

**Primary tests.**

| Hypothesis | Test |
|---|---|
| H1, H3 | Exact McNemar test on instance-level CCA-all |
| H2 | Instance-cluster bootstrap of the paired F1 difference (10,000 resamples, BCa intervals). Operating points are fixed on dev, so no threshold is re-estimated inside the bootstrap. H2 is rejected if the two-sided p < Holm α. The 15-point minimum effect of interest is assessed by the one-sided 95% lower confidence bound. |
| H4 | Bootstrap of the difference-in-differences on the risk-difference scale. A logit scale would manufacture interactions near the ceiling. Tested at α = 0.05 only if H3 is rejected. |

**Multiplicity.** Holm–Bonferroni over H1–H3, with fixed-sequence gatekeeping for H4. Secondary hypotheses and E2b–E8 are labelled exploratory.

**Sensitivity analyses.**
- Logistic GEE with instance clusters and stratum and backbone fixed effects, using **bias-reduced (Mancl–DeRouen) sandwich errors** because clusters are few.
- An instance-level sign-flip permutation test.

**Pre-registration.** `prereg/preregistration.md` and a manifest are committed and tagged **before any test-set answer exists**. The manifest records:
- the SHA-256 of the test file;
- the prompts;
- the thresholds;
- the reader;
- the H3 comparator;
- the model identifiers.

Deviations are logged with reasons.

**Run variance.** Stability (E5) is reported, but there is no ad-hoc rule that overrides the pre-registered tests.

**Power.** See §6.7. H4 is powered only for large interactions, and a null H4 is pre-registered as inconclusive.

---

## 11. Expected outcomes and what the paper becomes

| Outcome | The paper's claim | Natural venue emphasis |
|---|---|---|
| H1–H3 hold | Published conflict-aware designs cause measurable, reference-invisible harm. Decision-variable alignment repairs the detection bottleneck. Explicit scope checking improves on the best structured prompting. | Method + benchmark + measurement |
| H1, H2 hold; H3 fails | Harm is real and detection is fixable. With the right prompt, strong models compose as well as an explicit operator, so the operator's value is cost (O1) and transparency. | Benchmark + detection + negative result |
| H1 holds; H2 fails | Harm is real, and recognising cross-document defeat is the open problem; E4 locates it. | Resource / evaluation paper |
| H1 fails | Published resolvers do not harm users on real text (e.g., they keep exceptions in practice). A clean null with a validated instrument. | Evaluation paper |
| A V-gate fails | The persona setting (judge-free) becomes the headline. The reader/judge failure is reported as a methods finding that extends the 2026 omission-blindness results to RAG answers. | — |

Every row is a publishable paper. The pilot lacked this property: the only result it could report was "our system wins".

---

## 12. Limitations and ethics

- **Coverage.** English, UK, two domains. A US public-domain slice was dropped to protect annotation quality. Policy text is not representative of conversational, scientific or medical evidence.
- **Open weights.** All backbones are open-weight. Without the optional frontier check, no claim is made about frontier models.
- **Pre-annotation.** LLM pre-annotation, by a non-evaluated model, can anchor annotators. The from-scratch second pass measures this, and the external annotator bounds author bias.
- **Rare relations.** `redundant` and crossing `opposed` come mostly from contrast sets, so their prevalence is not estimated. Their natural rarity is reported, together with its drafting-practice explanation.
- **First-order frames.** Deep recursive defeat (NormBench Level 3+) is out of scope beyond what partition composition covers over typed attributes. SG-DT is the named extension path.
- **Clarification.** Systems are not evaluated on asking clarifying questions, which is a legitimate alternative for under-specified queries.
- **Licence.** Bank text is released as pointers only; gov.uk text is released under OGL.
- **Harm framing and personas.** Systems are research artefacts, not advisors. Personas are synthetic attribute bundles with no personal data.
- **Annotators.** The annotators include the authors. This is disclosed, and κ on independent passes plus the external annotator are the safeguards.

---

## 13. Timeline (detail in `03-implementation-plan-v3.md`)

| Weeks (from 2 Nov 2026) | Milestone |
|---|---|
| 1–2 | Metric repairs and validity battery; contract v1.3.0; reader; frame extractor; ConflictRAG-style detector training data; per-call token measurement |
| 2–5 | QNFA on dev; detection baselines; contexts; personas; mining sprint (test without evaluated detectors); re-split |
| 3–7 | Annotation to test targets; contrast sets; pre-registration |
| 7 (Sun 20 Dec) | **Freeze test set; tag `prereg-v1`** |
| 8–14 | Rate-limited runs (scheduler; unattended over the break). Validity study in weeks 10–11. |
| 12–15 | Analysis and writing |
| 16 (≈ 19 Feb 2027) | Internal strict review; submission |

---

## 14. Target venues

Listed in submission order. Quartiles are SCImago 2024–25; re-check them at submission.

1. **Information Processing & Management** (Elsevier, Q1). Best fit for an evaluation-and-method RAG paper.
2. **Knowledge-Based Systems** (Q1), or **Natural Language Processing Journal** (Elsevier; low open-access APC).
3. **If the paper leads with the benchmark** (H2 fails): **Language Resources and Evaluation**.
4. **Q2 options with broad remit:** Journal of Intelligent Information Systems, Applied Intelligence, Data & Knowledge Engineering.

---

## References (verified identifiers; full bibliography in the paper)

- Cattan et al. 2025. *DRAGged into Conflicts.* arXiv:2506.08500.
- Wang et al. 2026. *ConflictRAG.* arXiv:2605.17301.
- Ye et al. 2026. *Seeing through the Conflict* (TCR). arXiv:2601.06842.
- Chen, Li, Wan, Yuan 2026. *From Statute to Control Flow: Span-Grounded Deontic Trees for Defeasible Scope Parsing.* KDD 2026; arXiv:2606.08932.
- Sun, Cohen, Salakhutdinov 2022. *ConditionalQA.* ACL 2022.
- Chen, Zhang, Liu, Gupta, Kim, Cafarella 2024. *MDCR.* Findings of EMNLP 2024; arXiv:2406.11784.
- Rudinger et al. 2020. *Thinking Like a Skeptic.* Findings of EMNLP 2020.
- Xie & Du 2026. *Neuro-Symbolic Resolution of Recommendation Conflicts in Multimorbidity Clinical Guidelines.* arXiv:2604.17340.
- Majithia et al. 2026. *The CitizenQuery Benchmark.* arXiv:2602.04064.
- Li, Padman, Krishnan 2026. *Same Question, Different Source, Different Answer.* arXiv:2605.29084.
- Liu et al. 2026. *InsLogicBench.* ACL 2026.
- Yu et al. 2026. *Detecting Is Not Resolving.* arXiv:2605.27157.
- Kan 2026. *Claim-Selective Certification for High-Risk Medical RAG.* arXiv:2605.21949.
- *Rethinking All Evidence: Conflict-Driven Summarization* (CARE-RAG). arXiv:2507.01281.
- Niu et al. 2026. *ArbGraph.* arXiv:2604.18362.
- Li et al. 2026. *S2G-RAG.* ACL 2026.
- Fox et al. 2026. *LLM Judges Verify Presence, Not Absence.* NeurIPS 2026; arXiv:2608.31016.
- *Judging Is Not Enumerating: Silent Omissions in LLM-Authored Acceptable Sets.* arXiv:2608.01000.
- Ru et al. 2024. *RAGChecker.* NeurIPS 2024 D&B.
- Min et al. 2023. *FActScore.* EMNLP 2023.
- Gao et al. 2023. *ALCE.* EMNLP 2023.
- Pradeep et al. 2025. *The Great Nugget Recall.* SIGIR 2025.
- Gokul et al. 2025. *Contradiction Detection in RAG Systems.* arXiv:2504.00180.
- Reiter 1980; Poole 1985; Pollock 1987; Prakken & Sartor 1997; Dung 1995; Bienvenu et al., KR 2025.
- Gebru et al. 2021. *Datasheets for Datasets.* CACM.

*Removed after verification failed:* two October 2026 arXiv items cited in an earlier draft of this revision (a conflict-typing cascade paper and a "Don't Pick a Winner" paper). Neither could be confirmed by search; re-add them only after reading the abstract page.
