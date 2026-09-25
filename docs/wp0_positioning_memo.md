# WP0 — Prior-art review and positioning memo

**Status: SG-DT verified against the source, 2026-09-25. Closed-venue sweep still open.**
**Date of model sweep: 2026-09-24. Human verification of SG-DT: 2026-09-25.**

---

## Verdict

> **The bridging claim survives — but the distinction is narrower than the
> first draft of this memo stated, and must be worded precisely.**

This project targets the **resolution operator** in conflict-aware RAG: the step
deciding whether two separately retrieved, credibility-ranked passages are
selected between or composed, for the *conditional* case.

Two things narrow it, and both belong in §4 in the paper's own voice:

1. **Chen et al. named the failure first.** Silent Scope Omission — a system
   applying a general rule while silently dropping a nested exception — is
   their term, arrived at independently.
2. **They are not retrieval-innocent.** Verified against the source: the paper
   *does* discuss RAG and retrieval in its motivation, explicitly noting that
   retrieval can determine whether a decisive exception is visible at all. Its
   downstream statutory-reasoning experiment supplies a statute snippet as
   evidence alongside a case fact pattern.

### The sentence to write, and the one not to

> ❌ **"SG-DT does not involve retrieval"** / "prior work does not consider
> retrieval" / "the pipeline never has two documents."
>
> A reviewer points at their RAG discussion and the distinction collapses. The
> first draft of this memo made exactly this overclaim.

> ✅ **"Span-Grounded Deontic Trees address silent scope omission within an
> identified provision, by explicitly representing defeaters and their
> attachment to source spans. Our setting introduces an additional
> retrieval-level challenge: the governing rule and its scope-narrowing
> exception may be surfaced from separately retrieved documents. The system
> must therefore first recognise that the passages constitute a conditional
> relation, and then resolve their scope relation before composing an answer."**

The precise distinction, stated once:

| | SG-DT | this project |
|---|---|---|
| structural target | an **already-supplied** provision or executable unit | K separately retrieved passages |
| retrieval's role | discussed as a surrounding issue; may determine what is visible | **part of the phenomenon being benchmarked** |
| benchmark task | intra-provision defeasible scope parsing | cross-document scope resolution |

Their own §4 describes NormBench as benchmarking *intra-provision defeasible
scope parsing*, and their task definition is explicit: *given a provision X,
systems output a Span-Grounded Deontic Tree*, with nodes grounded to spans of
that current text scope (a full provision, or an executable unit when
segmented). That is the load-bearing fact, and it is verified.

**Confidence: high on SG-DT, unchanged elsewhere.** The closed-venue sweep
(§4) remains the open risk.

---


## How this pass was conducted, and what it is not

Every citation carried over from the proposal was checked against arXiv or the
ACL Anthology, and the title, authors and claims were compared against what the
memo asserted. That found four discrepancies, recorded below.

**What this pass could not do:**

- **The closed-venue sweep is not discharged.** KR, TPLP, AAAI, AIJ and JAR
  proceedings are not reachable by web search in the way arXiv is. The
  checklist below remains open, and it is the part most likely to contain a
  classical-KR anticipation of the four-way scope relation.
- **Full texts were mostly not read.** Conclusions rest on abstracts and
  anthology pages. The exception is SG-DT, the one paper where the answer
  decides the framing: it was read against the source on 2026-09-25, and that
  read corrected an overclaim this memo had made (see the Verdict).
- **A model reviewed a model's citations.** Shared failure modes; a claim we
  both misread in the same direction survives unchallenged. The SG-DT
  correction is a worked example — the model pass concluded "the pipeline never
  has two documents", which the paper's own RAG discussion contradicts.

---

## 1. Citation verification

All seven verified as real, with titles and authors matching. No fabricated
references.

| cited as | verified | status |
|---|---|---|
| Chen et al., SG-DT/NormBench, arXiv:2606.08932 | ✅ Jian Chen, Siyuan Li, Chucheng Wan, Zixuan Yuan | exact |
| ConditionalQA, arXiv:2110.06884 | ✅ Haitian Sun, William W. Cohen, Ruslan Salakhutdinov | exact |
| ConflictRAG, Wang et al., arXiv:2605.17301 | ✅ Chenyu Wang, Yueyuan Li, Yingmin Liu, Yang Shu | exact |
| DRAGged into Conflicts, arXiv:2506.08500 | ✅ Cattan, Jacovi, Ram, Herzig, Aharoni, Goldshtein, Ofek, Szpektor, Caciularu | exact |
| TCR, Ye et al., arXiv:2601.06842 | ✅ Hua Ye et al. | **see §2a** |
| Detecting Is Not Resolving, arXiv:2605.27157 | ✅ Zhe Yu, Wenpeng Xing, Chen Ye, Xuyang Teng, Bo Yang, Changting Lin, Meng Han | exact |
| Claim-Selective Certification, arXiv:2605.21949 | ✅ Shao Kan | exact |
| Defeasible NLI, Rudinger et al. | ✅ Rudinger, Shwartz, Hwang, Bhagavatula, Forbes, Le Bras, Smith, Choi — Findings of EMNLP 2020, pp. 4661–4675 | full author list recovered |
| S2G-RAG, Li et al., ACL 2026 | ✅ Li, Zou, Lv, Zhang, Zhou — arXiv:2604.23783, ACL 2026 (2026.acl-long.1185) | **ID recovered; memo had none** |

### Four discrepancies to fix in the write-up

**a. TCR is described imprecisely.** The memo calls it "makes the detect-resolve
process contestable." The paper is *"Seeing through the Conflict: Transparent
Knowledge Conflict Handling in RAG"*, and it addresses **internal (parametric)
knowledge versus retrieved external content** using dual contrastive encoders.
That is a *context–memory* conflict, not an inter-source conflict. The memo's
"orthogonal to which operator is used" is right, but for a stronger reason than
stated: it is a different conflict type.

**b. The "37%" figure is VERIFIED — with a wording constraint.** Confirmed
against the dataset-statistics table on 2026-09-25:

| complexity | count | share |
|---|---:|---:|
| Level 1 — simple | 834 | 36% |
| Level 2 — nested | 610 | 27% |
| **Level 3+ — recursive** | **846** | **37%** |

The paper states directly that *a significant portion of the dataset is
recursive (Level 3+, 37%)*. Citable.

The constraint is on phrasing. Their table classifies items by level; it does
not assert that 37% *require* deep resolution to answer.

- ❌ "37% of their benchmark needs deep exception nesting"
- ✅ "37% of NormBench items are classified as Level 3+ (recursive),
  corresponding to deep/counter-exception structures"

**c. Two further named findings, both usable.** The memo had only "Recursion
Decay". The abstract adds the **Auditability Trap** — *"models retrieve relevant
spans but fail to assemble correct control flow"* — and the results section
frames the same dissociation as the **Structure-Grounding Gap**: finding the
relevant span and attaching it to the correct logical parent are separate
abilities, and models have the first without the second.

That is this project's thesis observed *inside* a single provision, by an
independent group, and it is the strongest external support available for the
argument that locating evidence and composing it correctly come apart. Cite it
as support rather than treating it as competition.

**d. ConflictRAG's credibility mechanism.** The memo says "Entropy-TOPSIS". The
abstract describes an "entropy-based source credibility assessment" and a
diagnostic metric, **CARS**. TOPSIS is not confirmed. Verify before the related-
work section describes the mechanism.

---

## 2. The gating question, answered

> Does SG-DT anywhere define a resolution operator over *separately retrieved*
> competing sources?

**No — verified against the source, 2026-09-25.** Their task is defined as
*given a provision X, output a Span-Grounded Deontic Tree*, with nodes grounded
to spans of that current text scope; §4 calls NormBench a benchmark for
*intra-provision defeasible scope parsing*. Competing separately retrieved
documents are never the input unit.

**But the first version of this section overclaimed and has been corrected.**
It read: *"the pipeline never has two, because the retrieval in its Auditability
Trap is retrieval of spans inside the provision."* The paper discusses RAG and
retrieval directly in its motivation, and its downstream statutory-reasoning
experiment supplies a statute snippet as evidence. Writing "SG-DT does not
involve retrieval" would hand a reviewer an easy correction.

The surviving distinction is about **what the benchmark task takes as its
input unit**, not about whether retrieval is mentioned:

| | SG-DT | this project |
|---|---|---|
| structural input | an already-supplied provision or executable unit | K passages from retrieval, separately ranked |
| retrieval | discussed as a surrounding concern | part of the phenomenon benchmarked |
| failure | exception present in the text, dropped during parsing | exception present in the corpus, **discarded before parsing** |
| fix | structured intermediate representation | composition operator replacing selection |

A perfect SG-DT parser still exhibits this project's failure, because the
credibility ranker discards the exception passage before the parser sees both
together. That sentence is the positioning argument, and it is worth putting in
the paper close to verbatim.

**Confirmed 2026-09-25.** SG-DT §3–§4 read against the source. Input is a
single provision or executable unit; the bridging claim stands, with the wording
tightened as above.

---

## 3. New work found in this sweep

Five items not in the proposal. None anticipates the claim; two must be cited.

### CARE-RAG — the nearest neighbour, and the one a reviewer will raise
*Rethinking All Evidence: Enhancing Trustworthy RAG via Conflict-Driven
Summarization*, arXiv:2507.01281.

**Conflict-driven summarization** is the closest published thing to
"composition", and the name alone will prompt the question. The distinction is
real and must be stated:

- CARE-RAG **synthesises one reliable answer** from all evidence, having removed
  what it judges misleading. The output is a single answer with conflict handled
  *during* construction.
- This project **preserves each branch with its applicability condition
  attached**, and the output is structurally plural. Nothing is removed for being
  in tension with something else — that tension is the content.

CARE-RAG also mixes parametric and retrieved evidence, so like TCR it is partly
a context–memory method. **Cite it in related work; do not let a reviewer raise
it first.**

### Conflict-Aware RAG (ConScore) — ACM Web Conference 2026
doi:10.1145/3774904.3792289. Multi-stage learning with conflict signals; ConScore
compares generative probabilities across sources. A *detection and training*
signal, not a resolution operator. Cite alongside ConflictRAG.

### Ragability Benchmark — LREC 2026
ACL Anthology 2026.lrec-1.182. Inter-context conflicts requiring implicit
reasoning. Checked specifically for the conditional case: **it does not
distinguish conflicts where both sources are true under different scopes from
flat contradictions.** That absence supports the benchmark contribution and is
worth one sentence.

### ArbGraph — arXiv:2604.18362
*Conflict-Aware Evidence Arbitration for Reliable Long-Form RAG*. "Arbitration"
is selection vocabulary. Likely a direct instance of the operator this project
argues against — **worth reading properly**, as it may be a strong baseline or a
sharp foil.

### DeFAb — arXiv:2606.18557
*A Verifiable Benchmark for Defeasible Abduction in Foundation Models*.
Defeasible reasoning benchmark, no retrieval component. Relevant to the
defeasible-logic lineage rather than the RAG line.

---

## 4. Venues still to sweep — NOT discharged

Web search does not reach these. Someone with library access must do it.

- [ ] **KR** (Principles of Knowledge Representation and Reasoning)
- [ ] **IJCAI / IJCAI-ECAI 2026** — the KR track has argumentation and
      logic-programming sessions; accepted-papers list is public and worth a
      title scan
- [ ] **TPLP**
- [ ] **AAAI**
- [ ] **Artificial Intelligence Journal**
- [ ] **Journal of Automated Reasoning**
- [ ] **ACL Anthology** — beyond what general search surfaces

**Highest risk sits here.** The four-way scope relation (refinement / disjoint /
redundant / opposed) is a *knowledge-representation* contribution, and classical
KR has reasoned about specificity-based override since Reiter (1980). If the
four-way distinction exists under another name in the defeasible-logic
literature, the contribution narrows from "we define it" to "we operationalise
it for retrieval". That is survivable and should be planned for.

Classical lineage to check while there: default logic (Reiter 1980), defeasible
logic (Pollock; Prakken & Sartor), argumentation frameworks (Dung 1995),
truth-maintenance systems.

---

## 5. Consequences for the paper

1. **§4 must credit Chen et al. for Silent Scope Omission explicitly**, in the
   project's own voice, and then draw the retrieval-stage distinction. Reads as
   scholarship when volunteered; reads as concealment when found by a reviewer.
2. **Add a related-work paragraph on conflict-driven summarization** (CARE-RAG),
   distinguishing synthesis-into-one from composition-preserving-branches.
3. **Cite the 37% figure using their terminology** — "Level 3+ (recursive)",
   not "needs deep nesting". Verified; see §1b.
4. **Never write that SG-DT does not involve retrieval.** It does, in the
   motivation. The distinction is about the benchmark's input unit. The exact
   sentence to use is in the Verdict.
5. **Fix the TCR description** — it is context–memory, not inter-source.
6. **Verify ConflictRAG's mechanism** before describing it as Entropy-TOPSIS.
7. **Use the Structure-Grounding Gap as support, not competition.** An
   independent group found that locating the relevant span and attaching it to
   the right logical parent are separate abilities. That is this project's
   thesis, observed inside a single provision, and it is the strongest external
   evidence available for it.
8. **Claim the benchmark gap explicitly** — CONFLICTS and Ragability both taxonomise
   conflicts without separating "both true under different scopes" from
   "contradiction". That is the hole this corpus fills.

---

## 6. What would overturn this verdict

Stated up front so the fallback is not a retreat:

- ~~**SG-DT turns out to evaluate over a retrieved set**~~ — **checked and
  ruled out** on 2026-09-25. Its structural input is a single provision or
  executable unit. This was the largest single risk and it is closed.
- **A KR/TPLP paper already formalises the four-way relation** → the relation is
  not the contribution; the operationalisation and the benchmark are. **Now the
  largest remaining risk**, and it sits behind the closed venues in §4.
- **ArbGraph already composes rather than arbitrates** (the name suggests not,
  but the name is not the paper) → the operator claim narrows to the conditional
  case specifically.

None of the three would sink the project. All three would change §3 and §4, and
all three are cheap to check now and expensive to discover in week 14.

---

## 7. Outstanding actions, ranked

1. ~~Read SG-DT §3–§4~~ — **done 2026-09-25.** Single-provision input
   confirmed; overclaim corrected.
2. ~~Confirm the NormBench 37% figure~~ — **done.** Verified, with a wording
   constraint (§1b).
3. **Sweep the closed venues in §4.** Now the largest remaining risk.
4. Read ArbGraph (arXiv:2604.18362) — potential foil or baseline.
5. Confirm ConflictRAG's credibility mechanism.
6. Read CARE-RAG (arXiv:2507.01281) closely enough to write the distinguishing
   paragraph.

Item 3 is the real work and carries the residual risk. Items 4–6 are two papers
and one abstract check, an afternoon between them.
