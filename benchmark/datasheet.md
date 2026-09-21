# Datasheet — Conditional Conflict RAG Benchmark

**Status: template. Completed during WP2 (weeks 4–11). Joint deliverable.**

Following Gebru et al. (2021). The benchmark is treated as a first-class
deliverable, not an afterthought, because it may outlive the specific method it
accompanies — a released corpus with a datasheet keeps citation value even if
the central architectural claim is falsified.

---

## Motivation

- **Why was it created?** Existing conflict-aware RAG benchmarks do not
  distinguish *conditional* conflicts — a general rule and a valid exception,
  both true under different scopes — from factual contradictions. Without
  instances of that class, the suppression this project measures cannot be
  measured at all.
- **Who created it?** Two-member final-year project team.
- **Funding?** None. Built on one consumer GPU and a free-tier LLM API; this
  constrains what the accompanying experiments could run (see README §5).

## Composition

- **What do instances represent?** A query, a set of retrieved passages, and
  annotations describing how the passages relate: five-class conflict type,
  four-way scope relation, branch structure, and reference answers.
- **How many?** Target 250–350 human-verified conditional instances, plus
  mandatory distractors. *Record the final count here, broken out by tier.*
- **Tier composition — fill in, do not blend:**

  | Tier | Definition | Count |
  |---|---|---|
  | Tier 1 `natural` | rule and exception in separately retrievable documents | TBD |
  | Tier 2 `split` | single-document pair split across two synthetic documents | TBD |

  Tier 2 is legitimate and isolates the resolution-operator question from the
  mining question, but it is **not** evidence of natural multi-source
  occurrence and must never carry that claim.
- **Distractors included?** Yes, mandatory: genuine factual conflicts,
  temporal conflicts, disjoint non-conflicting pairs, redundant restatements,
  and no-conflict instances. Without them the Spurious-Condition Rate has no
  denominator.
- **Domains?** Consumer financial terms; immigration/visa eligibility. Medical
  and legal are deliberately excluded to keep stakes moderate and ground truth
  objective, rather than overclaiming.
- **Is any data confidential or personal?** No. All source documents are
  publicly published policy and terms pages. *Confirm at release.*

## Collection

- **How was it collected?** Semi-automatic mining from public provider
  documents. A model proposes candidate general/exception pairs; humans verify
  and label. Tier-1 yield was estimated first by the WP1 pilot
  (`benchmark/mining/tier1_pilot.py`) rather than assumed.
- **Sampling strategy?** *Record here.* Note any provider or domain skew.
- **Who collected it?** The two project members.
- **Over what timeframe?** WP2, weeks 4–11. *Record actual dates.*

## Preprocessing / labelling

- **Annotation protocol.** Three stages, adapted from Cattan et al.
  (arXiv:2506.08500): two annotators label independently, disagreements
  resolved by discussion, a third expert reviewer performs a final pass.
  Procedure-ambiguous refinement-vs-opposed cases route to the third reviewer
  rather than to a coin flip.
- **Agreement.** Cohen's kappa reported **separately** for the five-class
  conflict-type label and the four-way scope-relation label. *Record both.*
  The scope-relation label is expected to be the harder of the two; a blended
  number would hide exactly that.
- **Manual.** `benchmark/manual/annotation_manual.md`, including the explicit
  refinement-vs-opposed decision procedure and worked examples.
- **Is raw source text retained?** *Record.* Needed for reproducibility and for
  auditing a `natural` provenance claim.

## Uses

- **Intended use.** Evaluating whether a conflict-aware RAG system preserves or
  suppresses validly-scoped exceptions. Suitable for detection, classification,
  scope-relation, extraction and resolution tasks.
- **Uses to avoid.** Not a source of financial or immigration advice. Not a
  test of factual accuracy about any named provider — passages are snapshots
  and go stale. Not suitable for training a system that gives regulated advice.
- **Known limitations.** Two domains only; English only; a snapshot in time;
  Tier-2 instances are constructed rather than naturally multi-source; the
  accompanying experiments ran on open-weights models under a zero-spend
  constraint (README §5).

## Distribution

- **License.** Intended CC BY 4.0 for the annotations. *Source-document
  licensing must be cleared during WP2 — do not assume it.*
- **Versioning.** Versioned releases with a fixed train/dev/test split.
  `contract/schema/` carries the schema version the release was built against.
- **Where?** *Record the repository or archive at release.*

## Maintenance

- **Who maintains it?** *Record contact.*
- **Will it be updated?** *State the policy.* If passages go stale, say so
  rather than silently refreshing them — a benchmark that changes underneath
  published numbers is worse than a stale one.
