# Beyond Source Selection: Detecting and Measuring Cross-Document Exception Suppression in Retrieval-Augmented Generation

**Research proposal — Revision 8, 2026-10-30. Supersedes v7.**

This revision answers the third-round review (`05-third-round-review-and-responses.md`). The five changes the reviewer prioritised are made:

1. **Prior-art comparison.** MDCR is now compared formally, with a planned annotation-level mapping study. ConfRAG, Ragability and the WWW 2026 Conflict-Aware RAG paper are added. A six-dimension comparison table covers eight works.
2. **Natural cross-document evidence is separated.** S1 carries the cross-document claims on its own, with an explicit minimum size and a pre-declared narrowing rule if that minimum is not met.
3. **QNFA is fully specified.** This covers its name, definition, structured decision-variable signatures with a unification test (not a similarity threshold), handling of uncertainty, the full training procedure, and an intrinsic alignment evaluation against plain semantic similarity.
4. **CCA validation is strengthened.** The reader must quote the supporting span from the answer, counterfactual persona pairs test its sensitivity, and humans audit whether the answer *entails* the outcome, not just whether the reader agreed with gold.
5. **The experiment set is tiered.** There are Essential, Important and Optional tiers, and expansion beyond Essential depends on gates.

Also fixed:

- The selection claim is softened.
- Crossing opposition is its own relation.
- Proposition 2 now covers incomplete attributes, inconsistent constraints and empty regions.
- A no-deletion property has been added.
- B7 uses the released NormBench SG-DT pipeline and is checked for fidelity.
- TCR is now cited at its venue, AAAI 2026.

---

## Abstract

Conflict-aware retrieval-augmented generation (RAG) systems detect disagreement among retrieved passages, type it, and resolve factual conflicts by selecting the most credible source. This design can silently delete a common and consequential kind of disagreement: a general rule and a valid exception to it, stated in *separately retrievable documents*. Take *"international transactions incur a 3% fee"* alongside *"fees are waived for premium-tier cardholders."* Both are true. Selecting either returns an answer that is wrong for an identifiable group of users, and single-reference answer-correctness metrics do not register the loss.

Chen et al. (KDD 2026) named the within-provision form of this failure *Silent Scope Omission*. We study its cross-document form, which arises before any provision-level parsing can help: the system must first recognise that two differently worded passages govern the same decision.

We contribute four elements and evaluate them together:

1. **A formalisation.** Passages are represented as query-anchored *normative frames*: a decision variable, an applicability set, an outcome, a modality and a supporting span. Scope relations, including a separate *crossing* relation and a *same-scope* relation, are defined over frames. We prove that selection is lossy on refinement pairs, and give a no-deletion guarantee for the composition operator.
2. **QNFA (Query-anchored Normative Frame Alignment).** QNFA decides whether two passages govern the same decision by unifying structured decision-variable signatures, rather than thresholding semantic similarity. A distilled 184M-parameter cross-encoder runs it on a 6 GB GPU.
3. **XScope.** A benchmark of real UK government guidance and UK bank terms. Its *natural cross-document* stratum carries every cross-document claim on its own. It provides chunk-level retrieval contexts, branch annotation, personas, and contrast sets. No evaluated system or detector is used to build the test set.
4. **Case-complete accuracy (CCA).** This evaluation protocol asks whether a non-evaluated reader can determine, *with quoted evidence from the answer alone*, the correct outcome for a user in every branch's scope. It is validated against human entailment judgements, counterfactual persona pairs, and adversarial hedges and concatenations.

**Pilot (128 instances).** The pilot motivates the design:

- A lexical-NLI detector recovered 2 of 67 conflict-labelled pairs.
- Given gold routing, composition preserved 81 of 83 gold branches on 42 LLM-authored probes.
- The containment metric used in prior conflict work scored verbatim concatenation above the human-written answer.

---

## 1. Introduction

### 1.1 The failure

A RAG system answering *"Will I be charged for withdrawing cash abroad?"* may retrieve a tariff that states a per-withdrawal charge, and a product page that states one account type includes fee-free withdrawals. The passages disagree on the outcome without disagreeing on the facts: the second narrows the scope of the first.

ConflictRAG (Wang et al., 2026, arXiv) types inter-document conflicts as factual, temporal or opinion:

| Type | Resolution |
|---|---|
| Factual | Entropy-TOPSIS credibility selection |
| Temporal | Recency |
| Opinion | Multi-perspective synthesis |

A rule-and-exception pair is typed as factual or temporal and is resolved by selection. Whichever passage wins, the answer is wrong for one class of users. We call this **cross-document exception suppression**.

Chen et al.'s NormBench is explicitly "scoped to intra-provision defeasible scope (not precedent, multi-document synthesis, or open-textured standards)" (their §7; verified in the PDF). Ours is the multi-document case.

### 1.2 Why it is invisible

A dataset that stores one answer per question stores the general rule, so a system that suppresses the exception scores perfectly. Measuring the harm needs an instrument with three properties:

1. it checks every case;
2. it cannot be satisfied by pasting the retrieved passages;
3. it cannot be satisfied by listing every outcome without saying who each applies to.

The third property separates our protocol from coverage metrics in recent conflict benchmarks (§2.1).

### 1.3 Where the remedy breaks

We built the obvious remedy end to end and measured each stage. Composition preserved 81 of 83 gold branches when given gold routing (on 42 LLM-authored probes). The pilot detector recovered 2 of 67 conflict-labelled pairs across all 128 pilot instances. It gates on lexical topicality, and a rule and its carve-out are typically written by different teams with little shared vocabulary.

A single exploratory head-to-head run (12 probes, arms scored by different code paths) was inconclusive at −12.5 points (95% CI [−37.5, +12.5]). We disclose it.

The lesson: composing branches is easy once they are found. Recognising that two passages govern the same decision is hard, and its failure harms the users the exception was written for.

### 1.4 Contributions and claims

| Contribution | Claim it supports | Evidence |
|---|---|---|
| **C1 Formalisation** (§4) | Selection is the right operation only in a narrow set of cases. Composition cannot delete a frame. | Propositions 1–3 |
| **C2 QNFA** (§5) | Decision-variable unification detects cross-document defeat better than semantic matching. | E2 on S1; intrinsic VACS study (E2c) |
| **C3 XScope** (§6) | Cross-document exception suppression occurs in real retrieval contexts and is measurable. | S1 stratum; taxonomy-coverage study (E9) |
| **C4 CCA protocol** (§7) | The harm is invisible to single-reference and coverage metrics, but measurable with evidence-grounded per-case checks. | V1–V3; E5 |

We do **not** claim:

- a new defeasible logic;
- the relation names, which specificity, *lex specialis* and rule-base verification already use;
- that the composition operator is novel;
- the discovery of within-provision scope omission.

---

## 2. Related work and positioning

### 2.1 Multi-source conflict benchmarks and resolvers

**CONFLICTS** (Cattan et al., 2025, arXiv:2506.08500; data released as `google-research-datasets/rag_conflicts`). It defines five inter-context types, each with an expected behaviour: no conflict, complementary information, conflicting opinions, outdated information and misinformation. Its *complementary information* category covers compatible partial answers that should be consolidated.

**ConfRAG** (Yuan et al., ACL 2026). 1,814 real questions with an average of 9.58 retrieved web paragraphs; 57.2% exhibit explicit contradictions. It evaluates three things:

- *answer clustering* (NMI over document-to-viewpoint assignments);
- *answer coverage*, where a gold cluster counts as matched if any of its judge keywords appears as a substring of a predicted answer;
- *reason coverage*.

By that definition, a response that lists "yes" and "no" without saying who each applies to receives full answer coverage. Its unit is the *viewpoint cluster*, not an applicability-scoped branch.

**Ragability** (Gross, Petrak & Krenn, LREC 2026). Inter-context conflicts that require implicit reasoning. Real entities are replaced with fantasy entities to remove parametric knowledge. It finds that models detect contradiction but struggle with content questions.

**ConflictRAG** (Wang et al., 2026, arXiv:2605.17301). Typed routing, as described in §1.1. Its detector is a frozen encoder with an MLP and an LLM fallback, trained on pairs re-derived from Xie et al.'s ConflictQA (ICLR 2024).

**Conflict-Aware RAG with multi-stage conflict signals** (ACM Web Conference 2026, doi:10.1145/3774904.3792289; the exact title will be confirmed against the ACM DL before submission). It uses conflict signals in training, preference optimisation and reranking: a learning-to-be-robust approach, not a resolution operator.

**Other resolvers and diagnostics.**

| Work | What it does |
|---|---|
| **TCR** (Ye et al., AAAI 2026) | Transparent context–memory conflict handling |
| **CARE-RAG** (arXiv:2507.01281) | Conflict-driven summarisation into one answer |
| **ArbGraph** (arXiv:2604.18362) | Evidence arbitration |
| **MADAM-RAG** (arXiv:2504.13079) | Multi-agent aggregation |
| **EvidentialRAG** (arXiv:2607.10491) | Uncertainty routing |
| **ConflictScore** (arXiv:2606.26437) | Measures conflict *acknowledgement* |
| **Yu et al.** (arXiv:2605.27157) | Detecting contradiction does not constrain the recommendation |
| **TransplantQA** (arXiv:2605.29084) | 75% of inter-source pairs are "complementary (differing in detail or scope)"; fewer than 0.1% are contradictory |

### 2.2 Conditional reasoning across documents: MDCR, examined formally

**MDCR** (Chen et al., Findings of EMNLP 2024) is the closest QA-side work, so we state the relationship in our own formalism rather than assert a difference.

In MDCR, each document describes the eligibility conditions for **its own outcome**: one scholarship or one job. Relations are annotated between **conditions** in different documents:

| MDCR relation | Example | Our set relation over applicability |
|---|---|---|
| Conflicting | "first-class honours" vs "expecting a high-school diploma" | disjoint |
| Equivalent | "expecting a high-school diploma" ≡ "graduating high-school senior" | equal |
| Inclusive | "post-secondary degree" ⊇ "full-time STEM undergraduate" | subset |

Questions ask whether a user can obtain at least one, all, or the maximum number of the outcomes. Gold answers come from boolean satisfiability over the documents' AND/OR condition expressions.

In frame terms, two MDCR documents yield frames with **distinct decision variables** (v₁ = *eligible for Croucher*, v₂ = *eligible for Coca-Cola*), and MDCR relates their applicability sets. XScope pairs instead share **one** decision variable and differ in *outcome* on nested or overlapping sets.

Formally:

- MDCR's three relations are our applicability algebra (`compare`) restricted to *different* variables.
- XScope adds the outcome dimension on a *shared* variable. Defeat, and therefore suppression, can only arise there.

An MDCR pair can therefore never be a refinement. A system can also solve MDCR without ever deciding whether two passages govern the same outcome. That decision is exactly what QNFA addresses.

**This is a prediction, and we test it.** The taxonomy-coverage study (E9, §9) annotates 60 MDCR document pairs with our frame scheme and reports the distribution of relations. We predict that 0% are same-variable refinements.

### 2.3 Other conditional and defeasible QA

| Work | What it covers | How it differs from XScope |
|---|---|---|
| **ConditionalQA** (Sun et al., ACL 2022) | Single UK-government documents with user scenarios | Both the rule and the condition are always in view. Our persona device follows its scenario idea. |
| **CondAmbigQA** (arXiv:2502.01523) | Conditions arising from query ambiguity | Not cross-source defeat |
| **InsLogicBench** (ACL 2026) | Defeasible insurance-policy logic; LLMs fail on exception clauses | Single-policy adjudication, not multi-source retrieval |
| **Defeasible NLI** (Rudinger et al., Findings of EMNLP 2020), **DERELAB** (EMNLP 2026), **DeFAb** (arXiv:2606.18557) | Defeasible inference | No retrieval |

### 2.4 Defeasible scope parsing: SG-DT / NormBench

**SG-DT** (Chen et al., KDD 2026; code and data released under CC BY 4.0) parses one provision into a span-grounded deontic tree with explicit exclusion guards. It names Silent Scope Omission and reports that structured output helps on exception-active cases, with mixed aggregate gains that depend on parser fidelity. 37% of NormBench provisions are classified Level 3+.

We use SG-DT as **the strongest end-to-end comparator**: B7 runs the released NormBench stage-1 pipeline, adapted to multi-passage input, with a fidelity check (§8). We do not use it as a strawman.

The two works ask complementary questions:

- **SG-DT:** how should one provision's defeat structure be represented?
- **This work:** how does a RAG pipeline find, across separately retrieved passages, the pairs that form such a structure, and what do users lose when it does not?

### 2.5 The symbolic lineage

The underlying concepts are established:

- default logic (Reiter 1980);
- specificity (Poole 1985);
- defeasible logic and argumentation (Pollock 1987; Prakken & Sartor 1997; Dung 1995);
- *lex specialis*;
- rule-base verification, which already uses disjoint, redundant and conflicting rules;
- KR preference handling (Bienvenu et al., KR 2025);
- Xie & Du's offline SAT checking of guideline conflicts (arXiv:2604.17340).

We operationalise these at query time over retrieved text and measure the consequences for answers.

### 2.6 Completeness evaluation and judge validity

Related evaluation work:

- FActScore (EMNLP 2023);
- ALCE (EMNLP 2023);
- RAGChecker (NeurIPS 2024 D&B);
- nugget evaluation (Pradeep et al., SIGIR 2025);
- CitizenQuery-UK (arXiv:2602.04064), which shares our data source but has no exception structure;
- judge validation work: Judging the Judges (GEM 2025), JUDGE-BENCH (ACL 2025), Rao & Callison-Burch (arXiv:2606.00093).

Two 2026 results bear directly on our protocol:

- LLM judges detect added or altered content far better than omissions (Fox et al., NeurIPS 2026, arXiv:2608.31016).
- LLMs judge better than they enumerate, and their enumerations silently omit items (arXiv:2608.01000).

Our protocol never asks a judge to notice an omission. Personas come from annotated gold, not from a model's enumeration.

### 2.7 Six-dimension comparison (required before submission; filled from each paper and, where released, its data)

| Dimension | CONFLICTS | ConfRAG | Ragability | MDCR | ConditionalQA | SG-DT/NormBench | InsLogicBench | **XScope** |
|---|---|---|---|---|---|---|---|---|
| 1. Unit of annotation | query × result set: conflict type | question: viewpoint clusters + reasons | context set: conflict yes/no + content answer | document: conditions, AND/OR; cross-doc condition relations | document + scenario: answer + conditions | provision: deontic tree | policy + claim: adjudication | query × K-context: **frames, branches, scope relation, personas** |
| 2. Conditional applicability modelled | — | — | — | ✓ (eligibility conditions) | ✓ | ✓ | ✓ | ✓ |
| 3. Default-vs-exception asymmetry (defeat) | — ("complementary" is symmetric) | — | — | — (distinct outcomes) | — | ✓ (within provision) | ✓ (within policy) | ✓ (**across sources**) |
| 4. Persona-dependent correctness | — | — | — | ✓ (scenarios) | ✓ (scenarios) | — | ✓ (claim facts) | ✓ (**per branch region**) |
| 5. Cross-document source separation | ✓ | ✓ | ✓ | ✓ | — | — | — | ✓ (**natural S1 stratum**) |
| 6. Detects omitted exception branches | — | — (substring coverage credits unattached listing) | — | partial (missing conditions in conditional answers) | partial | ✓ (structure fidelity) | partial | ✓ (**CCA with evidence; attachment errors**) |

The table makes the gap precise. Several works have cross-document separation, and several have defeat. None has both, and none measures whether the *exception-scope user* can still obtain their answer.

### 2.8 Naming

Gokul et al. (arXiv:2504.00180) use "conditional contradiction" for a different phenomenon. We use "cross-document exception suppression" for the failure, and "refinement" for the relation.

---

## 3. Research questions, claims and hypotheses

### 3.1 Which stratum supports which claim

| Stratum | Role | Claims it may support |
|---|---|---|
| **S1 natural cross-document** | **Primary** | Every claim about *cross-document* suppression, detection and resolution |
| S2 natural same-guide | Secondary | Cross-section retrieval (reported separately) |
| S3 real-text split | Controlled diagnostic | Claims about the *resolution mechanism* given a rule and exception in context, explicitly scoped as such |
| S4 contrast | Stress test | Relation classification on rare relations |
| S5 LLM-authored probes | Development only | None |

**S1 minimum.** At least **80** S1 conditional instances in test; the floor is 60.

- **Below 60:** every cross-document hypothesis (H1–H3) is reported as *exploratory*, and the paper's title and abstract claim narrows to "exception suppression in RAG, with exploratory natural cross-document evidence".
- **S3 never substitutes for S1** in a cross-document claim.

### 3.2 Research questions

| | Question | Primary measure | Population |
|---|---|---|---|
| **RQ1** (harm) | Do selection-style and published conflict-aware resolvers suppress valid exceptions on natural cross-document text, invisibly to single-reference and coverage metrics? | CCA-all | S1 |
| **RQ2** (attribution) | Where does end-to-end loss arise? | Oracle-cascade deltas | S1, S3 |
| **RQ3** (detection) | Does decision-variable unification (QNFA) detect cross-document defeat better than semantic matching? | Conditional-class F1 at dev-fixed operating points; VACS alignment accuracy | S1; VACS |
| **RQ4** (resolution) | On the same backbone and inputs, does explicit scope checking beat the best structured prompting? | CCA-all | S1 (primary), S3 (mechanism) |
| **RQ5** (measurement) | Is CCA valid: evidence-grounded, insensitive to irrelevant persona attributes, in agreement with human entailment judgements? | V1–V3 | Dev validation set |

**Primary endpoint: CCA-all (strict).** An instance counts as correct only if the reader returns the gold outcome for *every* persona, **with a quoted span from the answer that a human-validated entailment check accepts** (§7.2).

### 3.3 Confirmatory hypotheses

The confirmatory family is H1–H3, Holm-adjusted, all on **S1**:

- **H1 (harm).** The ConflictRAG-style resolver (B3) has lower CCA-all than full-context generation (B0) on the primary backbone (gpt-oss-120b).
- **H2 (detection).** QNFA's conditional-class F1 on natural S1 hard-candidate pairs, at a dev-fixed operating point, exceeds the best non-QNFA detector (D0–D4, chosen on dev). The 15-point minimum effect of interest is assessed with a one-sided 95% lower confidence bound.
- **H3 (resolution).** On the primary backbone, O3 has higher CCA-all than the better of B7 and B8 (the comparator is chosen on dev and fixed in the pre-registration).

**Mechanism family (S3, separate α = 0.05, H4 gated on H3-S3).**

- **H3-S3.** As H3, on S3.
- **H4 (scale).** The O3 − B7 difference is larger at gpt-oss-20b than at gpt-oss-120b (same model family). It is tested as a risk-difference interaction.

### 3.4 Secondary hypotheses (pre-specified, unadjusted, reported with CIs)

- **H-T (taxonomy gap).** B5 (CONFLICTS-style taxonomy-aware prompting) has a larger CCA-all deficit relative to O3 on refinement instances than on disjoint ones.
- **H-C (concentration).** The B3 − B0 gap is smaller in the persona setting than in the under-specified setting.
- **H-F (frugality).** Distilled QNFA retains at least 80% of QNFA's F1 at no more than 15% of its API tokens.
- **H-V (alignment).** On VACS, signature unification beats variable-string cosine similarity and whole-passage cosine similarity in alignment accuracy, with the largest gain in the low-lexical / same-variable and high-lexical / different-variable quadrants.
- **H-K (coverage blindness).** ConfRAG-style answer coverage scores unattached-hedge answers at least as high as gold scoped answers, while CCA-all scores them at most 0.05.

### 3.5 Validity gates

These are checked first, on dev-built validation data.

**V1 — verbosity and hedging.** Each metric has its own criterion:

| Metric | Criterion |
|---|---|
| Legacy containment PR | concatenation ≥ gold (replicates the pilot) |
| CCA-all strict | gold ≥ concatenation; unattached hedge ≤ 0.05; modality hedge ("may receive") ≤ 0.10 |
| Scoped precision | gold > concatenation |

**V2 — reader.**
- Against the **human entailment audit**: reader determination precision ≥ 0.95 (a false determination inflates CCA) and recall ≥ 0.85.
- **Counterfactual personas:** irrelevant-attribute flips change the reader's output in ≤ 5% of cases; relevant-attribute flips change it in ≥ 90%.
- **Null answers:** NOT_DETERMINED in ≥ 95% of cases.

**V3 — judge.** The enumerate-then-verify judge used for scoped precision reaches ≥ 0.80 accuracy on human-labelled claims.

---

## 4. Problem formulation

### 4.1 Normative frames

Let *q* be a query and *D_q = (d₁, …, d_K)* the retrieved passages. A **normative frame** is *f = (σ, A, o, m, s)*:

- ***σ*** is a **decision-variable signature** (§5.2): a typed record that identifies what is being decided, e.g. `{event: cash_withdrawal, location: abroad, charge_kind: flat_fee}`. It is anchored to *q*.
- ***A ⊆ U*** is an **applicability set**. It is a finite conjunction of per-attribute constraints over a typed case space *U* = ∏ᵢ Dᵢ, whose domains are categorical, ordered, numeric-interval, boolean or date. An attribute a frame does not constrain is ⊤ on that dimension. Free-text constraints are *untyped*.
- ***o ∈ Y_σ*** is the **outcome**, in the variable's value space.
- ***m*** is the **modality**: obligation, permission, prohibition, statement, or **hedged** (e.g. "may").
- ***s*** is a **supporting span**, verbatim in some *d_i*.

### 4.2 Relations

Two frames are **co-referential** if their signatures unify (§5.3). For co-referential frames *f_i, f_j*:

| Relation | Applicability | Outcomes | Treatment |
|---|---|---|---|
| **Refinement** | *A_j ⊊ A_i* | differ on *A_j* | Compose: *o_i*, except under *A_j*, where *o_j* holds |
| **Redundant** | overlap | agree on the overlap | Merge. Branching would fabricate an exception. |
| **Disjoint** | *A_i ∩ A_j = ∅* | — | Keep both (CONFLICTS' "complementary") |
| **Crossing** | overlap, neither contains the other | differ on the overlap | **Not** a default-plus-exception structure. Report the non-overlapping parts as stated and the overlap as *contested*. |
| **Same-scope conflict** | *A_i = A_j* | differ | The **principal** case in which credibility- or recency-based selection is an appropriate strategy, *subject to evidence sufficiency and source reliability*. When neither condition holds, the correct behaviour is to abstain or present both, with sources. |

A pair is a **conditional conflict** iff its relation is **refinement**. Crossing is reported as its own class and is never counted as a composition success. Disjoint and redundant pairs are not conflicts. Frames whose signatures do not unify are **not compared**: they are about different decisions. This is the MDCR case (§2.2).

**Relabelling the pilot** (`relabel_v7.py`):

| Pilot label | New label |
|---|---|
| `conditional/disjoint` | no conflict |
| `conditional/redundant` | no conflict |
| `opposed` | split into crossing / same-scope / unknown |

Under these definitions the pilot contains **18 natural conditional conflicts**: 14 in S1 and 4 in S2.

### 4.3 Propositions

**Proposition 1 (selection is lossy on refinement pairs).** For a refinement pair, any answer asserting one unconditioned outcome for the decision is wrong on all of *A_j* or on all of *A_i \ A_j*. Its CCA-all is therefore 0. *Proof:* immediate.

**Proposition 2 (partition composition over typed constraints).** Let *F* be a set of co-referential frames whose constraints are all typed. Then:

1. **Internally inconsistent frames.** A frame whose constraints are unsatisfiable (*A = ∅*) is discarded and logged.
2. **Regions.** Each sign vector over *F*, i.e. each choice of inside/outside for every frame, defines a candidate region: the conjunction of the frame constraints and their complements. Unsatisfiable conjunctions (empty regions) are dropped. Satisfiability is decidable per attribute, because each domain is finite, an interval or ordered.
3. **Outcome per region.** For each non-empty region *R*, take the frames containing it, *F_R*.
   - If *F_R* has a unique ⊆-minimal element, *R* takes its outcome (specificity).
   - Otherwise, if all ⊆-minimal elements agree, *R* takes that outcome.
   - Otherwise *R* is **contested**.
   - If *F_R = ∅*, *R* is **uncovered**: no source speaks to it.

Then the output is:

- a **partition** of *U* (every case lies in exactly one region);
- **sound** (no region receives an outcome from a frame that does not contain it, or that a strictly more specific containing frame overrides);
- **order-invariant**.

*Proof sketch.* Sign vectors partition *U* by construction. Assignment depends only on set inclusion among the frames containing the region. ∎

**Incomplete attributes.** An attribute a frame does not mention is ⊤ for it, which is well defined. When *some* constraints are untyped, region satisfiability is undecidable. Regions involving them are labelled **indeterminate**: each such frame is presented with its condition verbatim and its source, and no outcome is inferred for the uncertain overlap.

**Proposition 3 (no deletion).** Under the default policy (unknown scope is presented as contested), the operator never omits a frame that has non-empty applicability and that is not merged as redundant. This holds **for any relation decisions, including wrong ones**. Selection happens only for same-scope pairs that pass the evidence-sufficiency rule.

Relation errors can therefore cause *misattachment*, which CCA penalises, but never silent deletion. *Proof:* by inspection of the operator's three exits: compose, merge (only when outcomes agree), select (only same-scope). ∎

Propositions 2 and 3 are elementary. Their role is to show that the remaining difficulty lies in recovering frames and aligning variables.

---

## 5. QNFA: Query-anchored Normative Frame Alignment

### 5.1 Definition and what is new

**Definition.** Given *q* and *D_q*, QNFA has three steps:

1. extract, for each passage, the frames relevant to *q*;
2. align frames across passages by **unifying their decision-variable signatures**;
3. for each unified pair, decide conflict (outcome mismatch over a commensurable outcome space) and scope relation (§4.2).

It outputs typed, scored pair decisions.

**What is new.** Frame extraction by an LLM is not new. The contribution is:

- **(a) A query-anchored, structured unit of comparison.** The decision-variable signature replaces passage-level or claim-level similarity.
- **(b) A three-valued unification test with a separate *uncertain* outcome.** The test returns equivalent, subsumes or distinct; uncertain is reserved for undecidable slots and escalated to an LLM tie-break. A rule and its exception are recognised because they share a signature, not because they share vocabulary.
- **(c) An outcome-commensurability check.** Two frames conflict only if their outcomes live in the same value space (same unit and type). This prevents "fee" vs "conversion rate" false conflicts.
- **(d) Distillation of the whole decision** into a 6 GB-deployable cross-encoder, with a calibrated cascade.

### 5.2 Decision-variable signatures

A signature is a small typed record. Its slots are fixed per domain and declared in a versioned schema, `qnfa/schema/{banking,immigration}.yaml`:

| Slot | Example values (banking) | Example values (immigration) |
|---|---|---|
| `event` | cash_withdrawal, card_payment, transfer, any_foreign_txn | application, extension, work, study |
| `object` | account, card, overdraft | visa_category, dependant |
| `charge_kind` / `decision_kind` | flat_fee, percentage, interest_rate / eligibility, permission, requirement, fee | |
| `qualifiers` | location: abroad / UK; channel: ATM / online | in_country / out_of_country |

Slot values are normalised to a per-domain vocabulary. A value outside it is kept as free text and marked untyped.

### 5.3 Unification

Two signatures *σ₁, σ₂*:

- **Equivalent** if every slot is equal or both slots are unspecified.
- **σ₁ subsumes σ₂** if every slot of σ₁ is equal to, or more general than, the corresponding slot of σ₂ in the slot's taxonomy. For example, `event: any_foreign_txn ⊒ cash_withdrawal`, `location: ⊤ ⊒ abroad`.
- **Distinct** if some typed slot is incompatible, e.g. `charge_kind: flat_fee` vs `percentage`.
- **Uncertain** if no typed slot is incompatible but some slot comparison involves free text. Uncertain pairs go to an LLM tie-break with an abstain option.

**Consequence of subsumption.** Frames whose signatures are related by subsumption are compared as co-referential, with the *narrower signature folded into applicability*. A variable-level refinement such as "foreign transactions incur 2.99%" vs "overseas ATM withdrawals on Account X are free" therefore becomes an applicability refinement, which is the case lexical gating misses.

**Worked ambiguity example** (the reviewer's):

| Passage variable | Signature |
|---|---|
| "International cash withdrawal fee" | `{event: cash_withdrawal, location: abroad, charge_kind: flat_fee}` |
| "Foreign currency transaction charge" | `{event: any_foreign_currency_txn, charge_kind: percentage}` |
| "Overseas ATM conversion fee" | `{event: cash_withdrawal, location: abroad, channel: ATM, charge_kind: percentage}` |

- Rows 1 and 2 are **distinct** (`charge_kind`). Both charges can apply cumulatively, so they are not in conflict.
- Row 2 **subsumes** row 3 (`event` and `channel` are more general, `charge_kind` is equal). A differing outcome between rows 2 and 3 is therefore a candidate refinement.
- Rows 1 and 3 are **distinct** (`charge_kind`).

No similarity threshold over these strings separates these cases reliably. That claim is tested in the VACS study below.

### 5.4 Algorithm

```
QNFA(q, D):
  F ← ⋃_d Φ(q, d)                       # one batched call; schema-validated; spans verbatim
  for (f_i, f_j) in pairs(F), source(f_i) ≠ source(f_j):
      u ← unify(σ_i, σ_j)                # equivalent | subsumes | distinct | uncertain
      if u = uncertain: u ← tiebreak_LLM(q, f_i, f_j) with abstain → 'uncertain' kept
      if u ∈ {distinct}: continue
      if not commensurable(o_i, o_j): continue
      A_i', A_j' ← fold_signature_difference_into_applicability(u, f_i, f_j)
      rel ← scope_relation(A_i', A_j', values_match(o_i, o_j))   # typed algebra ▸ entailment fallback
      score ← s_align(u) · s_mismatch(o_i, o_j) · s_rel
      emit (pair, type(rel), rel, score)
```

**Scores.**
- `s_align` is 1.0 for typed equivalence or subsumption, and the tie-break confidence otherwise.
- `s_mismatch` comes from value normalisation: 1.0 for exact typed values, lower for text values.
- `s_rel` is 0.95 for attribute-decided relations, and the entailment confidence otherwise.

All scores are calibrated on dev by isotonic regression, and ECE is reported.

### 5.5 Distilled detector and cascade (deployment on a 6 GB GPU)

**Model.** DeBERTa-v3-base cross-encoder over `[q] [SEP] d_i [SEP] d_j`, at seq 512 with pair-aware truncation. It has three heads:

| Head | Classes |
|---|---|
| Alignment | equivalent-or-subsumes / distinct |
| Type | conditional / same-scope / crossing / none |
| Relation | 5 classes |

**Labels.**
- Gold: from annotated train instances.
- Silver: QNFA decisions on mined pairs from **train and dev document groups only**. A query is synthesised from the rule passage's signature and recorded. A silver label is kept only if QNFA's calibrated score is at least 0.8 (positives) or at most 0.2 (negatives).

**Splits.** Document-group connected components, as in `benchmark/splits.py`. Silver never touches test groups; this is asserted in code.

**Class balance.** Negatives are downsampled to 3:1. Class-weighted loss is used, and per-class F1 is reported.

**Hard negatives.**
- (i) Same-provider, same-topic, *distinct-signature* pairs.
- (ii) High-lexical, different-variable pairs mined by BM25 similarity among distinct-signature pairs.
- (iii) Redundant and disjoint pairs.

**Calibration and cascade.** Temperature scaling on dev, with ECE reported. Pairs inside a dev-tuned band are escalated to full QNFA, and the full cost–F1 curve is reported.

**Comparisons.**
- Frozen MiniLM embedding cosine + logistic regression (the simplest embedding baseline).
- The ConflictRAG-style MiniLM+MLP (detector D2).
- The gold-only fine-tuned model (D4).

### 5.6 Composition, verbalisation and our systems

**Composition** follows Proposition 2. Untyped constraints fall back to the pilot operator under Proposition 3.

| System | Definition |
|---|---|
| **O1** | Distilled cascade + composition + template |
| **O2** | QNFA on the cell's backbone + composition + backbone-constrained verbalisation, with a coverage check: every branch outcome value must appear. |
| **O3** | The B7 parse, checked against same-backbone QNFA frames. Missing aligned frames are injected, redundant branches merged, and misattachments repaired. Then composition and verbalisation. Each repair is logged. |

**Contrastive Scope Probing** recovers implicit conditions. It enters O2 and O3 only if its pre-registered go/no-go passes.

---

## 6. XScope

### 6.1 Sources and licence

| Domain | Source | Licence | Release |
|---|---|---|---|
| Immigration and benefits | gov.uk, via the Content API and **Search API** (topic harvesting across distinct content items: guides, detailed guidance, the Immigration Rules appendices, policy papers). Records `content_id`, `document_type` and `public_updated_at`. | OGL v3.0 | Text with attribution |
| Banking | Public pages of UK banks | Copyright | URL, offsets and SHA-256, with a rehydration script |

### 6.2 Strata and targets (test split)

| Stratum | Pilot count (v7 labels) | Test target | Role |
|---|---:|---:|---|
| **S1 natural cross-document** | 14 (≈ 1 in the current test split) | **≥ 80** (floor 60) | Primary |
| S2 natural same-guide | 4 | ≥ 20 | Secondary |
| S3 real-text split | 45 built | ≥ 140 | Mechanism diagnostic |
| S4 contrast | 4 authored | ≥ 40 (≥ 20 redundant, ≥ 20 crossing) | Stress test |
| Controls (disjoint, redundant, same-scope, no-conflict) | 50+ | ≥ 120 | Precision and fabrication |
| S5 LLM probes | 54 | 0 (dev only) | — |

**Dev:** about 100 real-text instances built like test, plus S5.

### 6.3 Test construction without evaluated systems

**Mining.** Candidates come from retrieval: each rule-bearing chunk retrieves cross-document chunks. A **non-evaluated verifier** (Qwen ≈ 27B, with its own prompt) filters them, and humans verify every candidate.

**Exclusions.** QNFA, D3 and the backbones play no part in mining, screening or pre-annotation of test data. QNFA-assisted mining is limited to train and dev.

**Splits.** Natural document groups are re-split before the freeze (no model has been tuned on the pilot split; the old assignment is archived). New groups go to test until the targets are met.

### 6.4 Contexts

- **Chunking.** One sentence-aligned chunker (≈ 120–160 words) is applied to all documents. The gold passage is the chunk that contains the gold sentence.
- **Context size.** K = 6: gold chunks plus same-provider RRF hard negatives.
- **Screening.** The non-evaluated verifier screens the negatives. Flagged negatives are **adjudicated by a human, never silently removed**. A random 20% of contexts is audited.
- **Settings.** Two settings:
  - **oracle-inclusion** (primary);
  - **end-to-end retrieval**, with retrieval loss recorded.

### 6.5 Personas and counterfactual pairs

**Personas.** Each region gets at least one persona. Persona statements are paraphrased from source wording, not from attribute identifiers, and human-verified.

**Counterfactual pairs.** For every conditional instance we add two minimal persona pairs:

- **Relevant flip:** change only the attribute that defines the exception region. The gold outcome changes.
- **Irrelevant flip:** change an attribute no frame constrains, e.g. a city of residence for a fee rule. The gold outcome does not change.

These pairs serve both reader validation (V2) and a system-level **consistency** score: the share of pairs where a system's answer leads the reader to the correct change or non-change.

### 6.6 Annotation and agreement

**Annotation.**
- **Type and relation:** double-blind.
- **Branches:** pre-annotated by the non-evaluated model, corrected by one annotator, and re-annotated from scratch by the other on 30% of instances.
- **External annotator:** labels 15% of instances on all axes.
- **Adjudication:** by discussion, with a third reviewer when needed.

**Agreement reported.**
- κ per axis **per stratum**, with informative-n, p_e and composition.
- Branch-level F1.
- Krippendorff's α on persona outcomes.
- Author-versus-external agreement.

### 6.7 Size and power

The primary endpoint is binary per instance and paired. The minimum detectable difference (MDD) uses a McNemar test at power 0.8, for two levels of discordance *d*:

| n | MDD at α = 0.0167, *d* = 0.15 | at α = 0.0167, *d* = 0.35 | at α = 0.05, *d* = 0.25 |
|---:|---:|---:|---:|
| 60 | 16 | 24 | 18 |
| 80 | 14 | 21 | 15 |
| 140 | 10 | 16 | 12 |
| 220 | 8 | 13 | 9 |

**What these numbers mean for each claim:**

- **H1 (S1).** Selection versus full context is plausibly a large effect, so it is adequately powered.
- **H3 on S1.** This is powered for differences of roughly 14–21 points. A smaller true difference will appear inconclusive on S1, and the paper says so.
- **H3-S3 and H4.** These are the mechanism tests, with S3 supplying the power.

The function `power_sim.py` re-derives these figures by simulation, using the discordance observed on dev, before the freeze.

### 6.8 Taxonomy-coverage study (data for §2.7 and E9)

Using released data, we annotate three samples with our frame scheme. Annotation is double-blind on 30% of items.

| Sample | Size | Prediction |
|---|---:|---|
| MDCR document pairs | 60 | ≈ 0% same-variable refinement |
| ConfRAG questions labelled contradictory | 60 | — |
| CONFLICTS "complementary information" instances | 60 | — |

**What we report:** the distribution of relations (distinct-variable / disjoint / redundant / refinement / crossing / same-scope) and how often a refinement is present.

Two findings make the case for XScope necessary rather than duplicative:

- a non-trivial refinement share inside ConfRAG or CONFLICTS (suppression occurs in the wild, beyond UK policy, but those benchmarks do not label or score it);
- MDCR's predicted 0%.

### 6.9 Datasheet and release

The release includes:

- a generated datasheet;
- the annotation manual;
- the frame schemas;
- VACS;
- pinned model IDs;
- all cached outputs;
- the κ history;
- snapshot dates.

---

## 7. Evaluation protocol

### 7.1 Settings

| Setting | Query | Gold |
|---|---|---|
| Under-specified (primary) | *q* | branches, scoped answer, personas |
| Persona (secondary) | *q* + persona statement | short outcome |

**Rules for every system:**

- Every system receives the same context, instructions, `max_tokens` and reasoning setting.
- One scorer is used for all systems.
- Clarification-only answers are reported as a separate rate.

### 7.2 CCA with evidence

**Reader.** The reader is Llama-3.1-8B-Instruct, local, non-evaluated and pre-registered, with `num_ctx` logged. It receives only the system's answer and one persona statement, and returns JSON:

```
{"outcome": <value | "NOT_DETERMINED">,
 "evidence": "<verbatim span copied from the answer>",
 "condition_used": "<the condition in the answer that matches this person, or null>"}
```

**A determination counts only if all four conditions hold:**

1. `evidence` is a verbatim substring of the answer;
2. a local NLI cross-encoder judges that (`evidence` + persona statement) entails "the outcome is `outcome`", above a dev-calibrated threshold;
3. the outcome is not hedged: modality "may", "might", "could" or "in some cases" does not determine;
4. `outcome` matches the gold value under deterministic normalisation.

If any of conditions 1–3 fails, the determination is scored **NOT_DETERMINED**. That is CCA-strict, the primary metric. CCA-lenient, which drops conditions 1–3, is reported for comparison.

**This blocks gap-filling.** Take the reviewer's example: answer "Premium customers may receive fee waivers", persona "I have a premium account". This fails condition 3 (hedged) and usually condition 2. If the answer had said "Premium-tier cardholders pay no fee" but the persona held a product the answer never names, condition 2 fails.

**Metrics.**

| Metric | Definition |
|---|---|
| **CCA-all** | instance correct iff every persona is correct (primary) |
| **CCA-E** | accuracy on exception personas |
| **CCA-D** | accuracy on default personas |
| **Over-generalisation rate** | default persona answered with the exception's outcome |
| **Counterfactual consistency** | from the persona pairs (§6.5) |

### 7.3 Scoped precision and coverage comparators (secondary)

**Scoped precision.** Claims are labelled aligned / misattached / extraneous / unsupported:

1. deterministic matching first;
2. the remainder by an enumerate-then-verify judge (non-evaluated Qwen ≈ 27B) on a stratified 40% sample.

**Also reported:**
- attachment error rate;
- fabricated-exception rate (on controls);
- unsupported-claim rate;
- answer length.

**Comparator metrics, reported to show what they miss (H-K):**
- ConfRAG-style answer coverage (keyword substring), reimplemented from its paper;
- legacy containment PR;
- single-reference answer correctness.

### 7.4 Detection metrics

**Primary set.** On the **natural S1 hard-candidate set** (all S1 gold conditional pairs, plus curated same-provider, same-topic non-conflict pairs, including high-lexical different-variable pairs):

- F1 at dev-fixed operating points, with P and R;
- PR-AUC;
- the conditional / same-scope / crossing confusion matrix.

**Also reported.** The same metrics on all within-context pairs, plus tokens, calls and latency on the 6 GB profile.

**Intrinsic alignment (E2c): the Variable-Alignment Challenge Set (VACS).** VACS has 200 human-annotated passage pairs, stratified into four quadrants of lexical similarity × same-variable. Each pair is labelled equivalent, subsumes or distinct, and double-annotated. The set is built from train/dev document groups only. Five methods are compared:

- whole-passage Contriever cosine;
- variable-string cosine;
- NLI;
- LLM direct judgement;
- QNFA unification.

### 7.5 Validity studies (E5)

**Validation battery.** Eight outputs are generated per validation instance:

| # | Output |
|---|---|
| 1 | gold |
| 2 | selection |
| 3 | concat-gold |
| 4 | concat-K |
| 5 | padded gold |
| 6 | unattached hedge |
| 7 | modality hedge |
| 8 | null |

**Human entailment audit.** On 300 reader decisions, two annotators (blind to system and to the reader's output) judge *"Does this answer, by itself, establish the outcome for this person?"* The external annotator covers 100 of them. This audit determines V2's precision and recall.

**Counterfactual pairs.** Pass rates for V2.

**Judge validation.** 300 claim decisions, for V3.

**Stability.** Two repeats on 50 instances.

---

## 8. Systems

### 8.1 Systems by tier

| ID | System | Tier |
|---|---|---|
| B0 | Full-context generation | **Essential** |
| **B3** | ConflictRAG-style resolver (details below) | **Essential** |
| **B7** | **SG-DT parse via the released NormBench stage-1 pipeline** (CC BY 4.0), adapted to multi-passage input: passages are concatenated with passage-ID-prefixed spans, and exclusion guards are enforced by NormBench's validator. The tree is compiled to branches by their compiler, then verbalised as in O2. **Fidelity check:** the same pipeline on 100 NormBench English items, scored with *their* evaluator, compared with their reported figures for the closest model. Every adaptation is listed. | **Essential** |
| B8 | B7 + self-verification | **Essential** |
| **O3** | Hybrid (§5.6) | **Essential** |
| B5 | Taxonomy-aware prompting (CONFLICTS) | Important (for H-T) |
| O1, O2 | Our frugal and full pipelines | Important |
| B1, B2, B4, B6 | rerank-top1, NLI filter, CARE-style, free-text | Optional (run on 7B only unless budget allows) |

**B3 in detail.**
- **Detector:** trained as ConflictRAG describes, on ConflictQA-derived data, *not* on our labels. A fidelity replication against the paper's reported detector accuracy is P0.
- **Metadata:** source type and date are enriched from the gov.uk Content API.
- **Ablation:** the specificity criterion is ablated.
- **Labelling:** called "ConflictRAG-style" throughout, and the authors are asked for their code.

### 8.2 Backbones and evaluation-only models

**Backbones:**

| Backbone | Size | Notes |
|---|---|---|
| Qwen2.5-7B (local) | 7B | |
| gpt-oss-20b (Groq) | 20B | |
| gpt-oss-120b (Groq) | 120B | **Primary** |

Frames are always extracted by the cell's own backbone.

**Evaluation-only models (never evaluated as systems):**
- Qwen ≈ 27B: verifier, screener, pre-annotator and judge.
- Llama-3.1-8B: reader.

**Optional (P2):** a free frontier tier for B7, B8 and O3 on S1, if one is available at run time.

---

## 9. Experiments (tiered)

| ID | Tier | Design |
|---|---|---|
| **E1 Harm** | Essential | B0, B3, B7, B8, O3 × gpt-oss-120b × S1 (under-specified). B0 and B3 in the persona setting. Same systems on S3. |
| **E2 Detection** | Essential | D0 heuristic, D1 NLI, D2 ConflictRAG-style, D3 zero-shot LLM pair, D4 fine-tuned on gold, **D5 QNFA**, D6 distilled, D7 cascade, on the S1 hard set. D6 and D7 are Important. |
| **E2c VACS** | Essential | Intrinsic alignment, H-V |
| **E4 Oracle cascade** | Essential | O2 with gold substituted stage by stage, on the 7B and 20b backbones, for S1 and S3 |
| **E5 Validity** | Essential | §7.5; V1–V3; H-K |
| E3 Scale | Important | B7, O3 (+ B0, B3) × gpt-oss-20b on S3 and S1 → H3-S3, H4 |
| E9 Taxonomy coverage | Important | §6.8 |
| E6 Robustness | Important | Order permutation; K ∈ {4, 6, 10}; end-to-end retrieval; unknown-scope policy; B3 without specificity — run on the 7B backbone |
| E2b QNFA ablations | Important | No query anchoring; no signature unification (string cosine instead); no typed attributes; no outcome commensurability |
| E7 Cost | Important | Tokens, calls, latency; Pareto front |
| E8 External (ConditionalQA cross-passage) | Optional, gated | |
| E10 Preference study, frontier check, full ablation grid, all systems on all backbones | Optional | |

**Staged expansion.** Important experiments start only after:

- G4 (detection works on dev);
- G-YIELD (S1 is on track);
- G1 (token budget measured).

Optional experiments start only if the Essential set is complete by week 11.

---

## 10. Statistical analysis plan

**Units.** The unit of analysis is the instance; all comparisons are paired.

**Primary tests.**
- **H1, H3, H3-S3:** exact McNemar test on CCA-all.
- **H2:** instance-cluster BCa bootstrap of the F1 difference at dev-fixed operating points. The minimum effect of interest is checked with a one-sided 95% lower confidence bound.
- **H4:** bootstrap of the risk-difference interaction, run only if H3-S3 is rejected.

**Multiplicity.**
- Holm across H1–H3 (S1).
- H3-S3 is tested at α = 0.05, with H4 behind fixed-sequence gatekeeping.
- Everything else is secondary or exploratory, and labelled as such.

**Sensitivity analyses.** GEE with bias-reduced (Mancl–DeRouen) errors, and an instance-level sign-flip permutation test.

**Pre-registration.** Committed and tagged before any test answer exists. It records:

- the test file hash;
- prompts;
- thresholds;
- the reader;
- the H3 comparator;
- model IDs;
- the S1 narrowing rule;
- the `power_sim` output.

**Run variance.** Reported descriptively, with no override rule.

---

## 11. Outcomes and what the paper becomes

| Outcome | Claim |
|---|---|
| H1–H3 hold on S1 | Published conflict-aware designs cause measurable, metric-invisible cross-document harm. Signature unification repairs detection. Explicit scope checking beats the best structured prompting. |
| H1, H2 hold; H3 inconclusive on S1, H3-S3 holds | Harm is real and detection is fixable. The resolution advantage is established as a mechanism on controlled splits, and is underpowered on natural pairs. That is stated plainly. |
| H1 holds; H2 fails | Harm is real, and cross-document defeat recognition is the open problem. E4 and E9 are the evidence. This becomes a resource and evaluation paper. |
| S1 below 60 | Every cross-document claim is exploratory; the title and abstract narrow (§3.1). |
| A validity gate fails | The persona setting (judge-free, deterministic matching) becomes the headline, and the reader result is reported as a methods finding. |

---

## 12. Limitations and ethics

- **Scope of data.** English, UK, two domains, open-weight models.
- **S1 size.** It may be small. The narrowing rule makes the consequence explicit.
- **Annotation anchoring.** LLM pre-annotation by a non-evaluated model is used. The from-scratch second pass and the external annotator bound its effect.
- **Constructed contrast sets.** S4 is constructed, so its prevalence is not estimated.
- **First-order frames.** SG-DT trees are the named extension.
- **Signature schemas.** These are per-domain and hand-designed. Porting QNFA to a new domain needs a new schema; E2b measures what the schema contributes.
- **Clarification not evaluated.** Asking a clarifying question is a legitimate alternative to a scoped answer. We do not evaluate it.
- **Personas.** They are synthetic. Systems are research artefacts, not advisors.
- **Annotators.** The authors are annotators, which is disclosed. Safeguards are an external annotator and independent passes.

---

## 13. Timeline and venues

The timeline is in `03-implementation-plan-v4.md`. In outline:

- Essential work freezes on **20 Dec 2026**.
- API runs continue unattended over the break.
- Submission is around **19 Feb 2027**.

**Venues, in order:**

1. Information Processing & Management (Q1)
2. Knowledge-Based Systems or Natural Language Processing Journal
3. Language Resources and Evaluation, if the resource framing leads
4. Q2 options with broad remit: Journal of Intelligent Information Systems, Applied Intelligence

---

## References (verified unless marked)

- Cattan et al. 2025. *DRAGged into Conflicts.* arXiv:2506.08500. Data: `google-research-datasets/rag_conflicts`.
- Yuan, Kong, Li, Li, Liu 2026. *Benchmarking LLM's Capability in Reasoning over Conflicting Web References* (ConfRAG). ACL 2026, pp. 303–322.
- Gross, Petrak, Krenn 2026. *Ragability Benchmark: A Dataset and Library to Test LLMs on Inter-context Conflicts.* LREC 2026, pp. 2323–2333.
- *Conflict-Aware RAG with multi-stage conflict signals.* ACM Web Conference 2026, doi:10.1145/3774904.3792289. **Exact title and authors to be confirmed against the ACM DL.**
- Wang et al. 2026. *ConflictRAG.* arXiv:2605.17301.
- Ye et al. 2026. *Seeing through the Conflict: Transparent Knowledge Conflict Handling in RAG* (TCR). AAAI 2026, 40(40):34423–34431.
- Chen, Li, Wan, Yuan 2026. *From Statute to Control Flow: Span-Grounded Deontic Trees for Defeasible Scope Parsing.* KDD 2026; arXiv:2606.08932. Code: `github.com/AlexJJJChen/NormBench` (CC BY 4.0).
- Chen, Zhang, Liu, Gupta, Kim, Cafarella 2024. *MDCR: A Dataset for Multi-Document Conditional Reasoning.* Findings of EMNLP 2024, pp. 11407–11424.
- Sun, Cohen, Salakhutdinov 2022. *ConditionalQA.* ACL 2022.
- Liu et al. 2026. *InsLogicBench.* ACL 2026.
- Rudinger et al. 2020. *Thinking Like a Skeptic.* Findings of EMNLP 2020.
- Xie & Du 2026. arXiv:2604.17340.
- Li, Padman, Krishnan 2026. *TransplantQA.* arXiv:2605.29084.
- Majithia et al. 2026. *CitizenQuery-UK.* arXiv:2602.04064.
- Yu et al. 2026. arXiv:2605.27157.
- CARE-RAG, arXiv:2507.01281.
- ArbGraph, arXiv:2604.18362.
- MADAM-RAG, arXiv:2504.13079.
- EvidentialRAG, arXiv:2607.10491.
- ConflictScore, arXiv:2606.26437.
- Fox et al. 2026. NeurIPS 2026; arXiv:2608.31016.
- *Judging Is Not Enumerating.* arXiv:2608.01000.
- FActScore (EMNLP 2023).
- ALCE (EMNLP 2023).
- RAGChecker (NeurIPS 2024 D&B).
- Pradeep et al. (SIGIR 2025).
- Gokul et al. 2025. arXiv:2504.00180.
- Reiter 1980; Poole 1985; Pollock 1987; Prakken & Sartor 1997; Dung 1995; Bienvenu et al., KR 2025.
- Gebru et al. 2021.
