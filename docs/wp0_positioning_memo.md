# WP0 — Prior-art review and positioning memo

**Status: first pass complete, model-conducted. Verdict below is provisional.**
**Date of sweep: 2026-09-24.**

---

## Verdict

> **The bridging claim survives, narrowed.**

This project appears to be the first to target the **resolution operator** in
conflict-aware RAG — the step that decides whether two separately retrieved,
credibility-ranked passages are selected between or composed — for the
*conditional* case specifically.

The narrowing is on novelty of the *problem*, not the *operator*. Chen et al.
(SG-DT, below) independently named the same failure — a system applying a
general rule while silently dropping a nested exception — and got there first.
The paper must say so plainly in §4, in its own voice, rather than leaving a
reviewer to discover it.

**Confidence: moderate, not high.** See "What would overturn this" at the end.
The decisive fact — that SG-DT assumes its provision is already identified and
never models competition between separately retrieved sources — is drawn from
the abstract and a partial PDF read, and one automated read of that PDF
returned demonstrably wrong answers about the paper's own terminology. A human
needs to confirm it against the methods section. It is one section of one paper,
and everything downstream rests on it.

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
- **Full texts were mostly not read.** Conclusions rest on abstracts,
  anthology pages and partial PDF reads. For SG-DT — the one paper where the
  answer decides the framing — that is not good enough on its own.
- **A model reviewed a model's citations.** Shared failure modes; a claim we
  both misread in the same direction survives unchallenged.

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

**b. The "37%" figure is unverified.** The memo asserts 37% of NormBench needs
Level-3-or-deeper resolution. Not present in the abstract, and not confirmed in
the partial PDF read. **Do not cite this number until someone reads the paper.**

**c. SG-DT's second pathology is missing.** The memo names "Recursion Decay"
(verified verbatim). The abstract names a second: the **"Auditability Trap,
where models retrieve relevant spans but fail to assemble correct control
flow."** That is directly relevant — it is evidence that *locating* the right
text and *composing* it correctly are dissociated, which is this project's thesis
observed inside a single provision.

**d. ConflictRAG's credibility mechanism.** The memo says "Entropy-TOPSIS". The
abstract describes an "entropy-based source credibility assessment" and a
diagnostic metric, **CARS**. TOPSIS is not confirmed. Verify before the related-
work section describes the mechanism.

---

## 2. The gating question, answered

> Does SG-DT anywhere define a resolution operator over *separately retrieved*
> competing sources?

**Evidence says no.** SG-DT takes an already-identified provision as input and
parses scope *within* it. The "retrieval" in its Auditability Trap is retrieval
of **spans inside the provision**, not retrieval of competing documents. There
is no selection-versus-composition decision over two independently retrieved
passages, because the pipeline never has two.

**Therefore the distinction in the proposal holds, and it is sharper than
originally written.** The two failures compose rather than overlap:

| | SG-DT | this project |
|---|---|---|
| input | one identified provision | K passages from retrieval, separately ranked |
| failure | exception present in text, dropped during parsing | exception present in corpus, **discarded before parsing** |
| fix | structured intermediate representation | composition operator replacing selection |

A perfect SG-DT parser still exhibits this project's failure, because the
credibility ranker discards the exception passage before the parser sees both
together. That sentence is the positioning argument, and it is worth putting in
the paper close to verbatim.

**What a human must confirm:** read SG-DT §3 (method) and §4 (experimental
setup) and check that the input is a single provision. If SG-DT evaluates over a
retrieved *set*, the bridging claim collapses to the fallback.

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
3. **Drop the unverified 37% figure** until confirmed.
4. **Fix the TCR description** — it is context–memory, not inter-source.
5. **Verify ConflictRAG's mechanism** before describing it as Entropy-TOPSIS.
6. **Use the Auditability Trap** as independent support: locating the right text
   and composing it correctly are dissociated, shown inside a single provision.
7. **Claim the benchmark gap explicitly** — CONFLICTS and Ragability both taxonomise
   conflicts without separating "both true under different scopes" from
   "contradiction". That is the hole this corpus fills.

---

## 6. What would overturn this verdict

Stated up front so the fallback is not a retreat:

- **SG-DT turns out to evaluate over a retrieved set** → bridging claim gone;
  fall back to extraction method + formal scope-relation definition + validated
  metric suite.
- **A KR/TPLP paper already formalises the four-way relation** → the relation is
  not the contribution; the operationalisation and the benchmark are.
- **ArbGraph already composes rather than arbitrates** (the name suggests not,
  but the name is not the paper) → the operator claim narrows to the conditional
  case specifically.

None of the three would sink the project. All three would change §3 and §4, and
all three are cheap to check now and expensive to discover in week 14.

---

## 7. Outstanding actions, ranked

1. Read SG-DT §3–§4. Confirm single-provision input. **Everything rests on this.**
2. Read ArbGraph (arXiv:2604.18362) — potential foil or baseline.
3. Sweep the closed venues in §4.
4. Confirm the NormBench 37% figure, or drop it.
5. Confirm ConflictRAG's credibility mechanism.
6. Read CARE-RAG (arXiv:2507.01281) closely enough to write the distinguishing
   paragraph.

Items 1–2 are two papers and would take an afternoon. Item 3 is the real work
and the one with the most residual risk.
