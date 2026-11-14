"""The condition-loss study (November 2026 pivot).

Stage-wise attribution of applicability-condition loss in chunked
retrieval-augmented generation, measured on public human-annotated data:

* ConditionalQA v1.0 (Sun et al., ACL 2022) -- the primary testbed;
* RAMDocs (Wang et al., 2025) -- the contrast regime, where discarding
  evidence is the correct operation;
* this repository's own double-annotated corpus -- out-of-distribution.

Nothing here needs new human annotation. Every gold label is either a public
benchmark's or one this project's two annotators already agreed on.
"""
