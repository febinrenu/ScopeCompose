# v7 package: what is in here and what to do first

**Date:** 2026-10-30.
**Written against:** `ScopeCompose/` at commit `7438728` (520 tests, 128-instance corpus).

## Reading order

1. **`01-critical-review-and-gap-analysis.md`**
   - An outside-reviewer reading of proposal v6 and of what the repository has actually measured.
   - Verdict: as things stand, the paper would be rejected. The reason is not effort. The measurements point somewhere other than where v6 aims.
2. **`02-proposal-v7.md`**
   - The rewritten proposal, in the register of the journal paper.
   - The centre of gravity moves from the composition operator, which preserves 81 of 83 branches when given gold inputs, to three things:
     - **detection**: the pilot detector recovered 2 of 67 conflict-labelled pairs;
     - **a valid harm measurement**;
     - **a real-text RAG benchmark**.
3. **`03-implementation-plan-v3.md`**
   - File-by-file edits to the existing repository, with owners, priorities and acceptance tests.
   - Includes the token budget against Groq's free tier, a realistic annotation budget, decision gates, and a week-by-week schedule to a submission around 19 Feb 2027.
4. **`04-second-round-review-and-responses.md`**
   - An independent strict review of the first v7 draft, every claim checked against the repository data, and the change made for each point.

## The ten things to do in week 1 (2–8 Nov)

1. **Read 01 and 02 together, then agree the reframing with your supervisor.** The paper is now about cross-document exception suppression: detecting it, measuring it, and comparing designs. It is no longer "our operator wins".
2. **Freeze `site/` and the Studio.** Bug fixes only until submission.
3. **Line up the external annotator** (a course peer). About 12 hours in total, spread over weeks 5–11.
4. **Contract v1.3.0 + `relabel_v7.py`.** Check that it reproduces 18 natural conditional conflicts (14 cross-document, 4 same-guide).
5. **Add the READER tier** (local Llama-3.1-8B, `num_ctx` set) and the **EVAL tier** (Qwen ≈ 27B). Remove Qwen from the backbones.
6. **Measure tokens per call** for B0, B7, B8 and O3 on 20 instances at one fixed reasoning setting. This is gate G1, and it decides the whole run matrix.
7. **Build `metrics/values.py`** and start `case_accuracy.py`. Write the V1–V3 criteria into `prereg/` **before** running the validity battery.
8. **Start S-9:** rebuild the ConflictRAG training pairs from ConflictQA, and search for, or ask the authors for, their code.
9. **Run `experiments/wp1_probe --live` on S5.** It has been answerable since October 2, and its verdict is gate G5.
10. **Re-verify every 2026 arXiv citation** you plan to keep. Two from the literature sweep could not be verified and were removed.

## Folder convention

The earlier versions (v1–v6, plan v1–v2) stay one level up for history. This folder is the current plan of record. When the paper is written, copy the parts of `02` you use into the manuscript, rather than editing `02` into a paper in place.
