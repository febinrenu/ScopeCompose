"""Freeze the pre-registration: export the test set and record everything that must not change.

Run once, after development on dev data is finished and before any test answer
exists::

    python -m prereg.freeze --corpus benchmark/data/corpus.jsonl --h3-comparator B7c
    git add prereg/manifest.json benchmark/data/test.jsonl && git commit -m "Freeze the pre-registration"
    python -m prereg.freeze --tag          # tags that commit prereg-v1

The manifest pins, by hash or value: the test file; every prompt that shapes a
system or the metric; the signature schemas; the concrete model behind each
pinned tier; the reader's Modelfile; the thresholds; the power calculation; the
H3 comparator; and the narrowing rule's outcome. A later run recomputes these and
:func:`experiments.staging.assert_frozen` refuses to start if the test file or
the tag does not match.

**The H3 comparator must be chosen on dev first** (the better of B7c and B8) and
passed explicitly: picking it after seeing test results would make the
confirmatory test a choice between two outcomes, not a test. The freeze refuses
to run without it.

**S5 never enters the test file.** LLM-authored probes are development material;
they are excluded here by stratum, whatever split they landed in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import settings
from baselines.conflictrag_topsis import AUTHORITY, DEFAULT_DELTA
from baselines.registry import prompts_sha
from benchmark.strata import S1_FLOOR, S1_TARGET, Stratum, quota_report, stratum_of
from contract.gold import GoldInstance
from contract.models import CONTRACT_VERSION
from detection.signature import schema_hashes
from experiments.staging import MANIFEST, PREREG_DIR, PREREG_TAG, REPO_ROOT, file_sha256
from metrics.case_accuracy import DEFAULT_NLI_THRESHOLD, READER_PROMPT_SHA
from metrics.power_sim import mdd_table
from metrics.validity import V1_CRITERIA_SHA

PINNED_TIERS = ("BACKBONE_PRIMARY", "BACKBONE_LOCAL", "READER", "EVAL")
H3_COMPARATORS = ("B7c", "B8")
DEFAULT_TEST_PATH = REPO_ROOT / "benchmark" / "data" / "test.jsonl"


def export_test(corpus: Path, out: Path) -> list[GoldInstance]:
    """Write the test file: split == test, S5 excluded, sorted by id, canonical JSON."""
    insts = [GoldInstance.model_validate(json.loads(l))
             for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    test = sorted((i for i in insts if i.split == "test" and stratum_of(i) is not Stratum.S5),
                  key=lambda i: i.instance_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i in test:
            fh.write(json.dumps(i.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n")
    return test


def _git_head(repo: Path) -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return r.stdout.strip()


def _repo_relative(path: Path, repo: Path) -> str:
    """The path as the repo sees it, or the absolute path if the file lives elsewhere."""
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def tier_record(name: str) -> dict:
    t = settings.models().tier(name)
    return {"provider": t.provider, "model": t.model, "temperature": t.temperature,
            "max_tokens": t.max_tokens, "reasoning_effort": t.reasoning_effort,
            "fallback": t.allow_fallback}


def build_manifest(test: list[GoldInstance], test_path: Path, *, h3_comparator: str,
                   nli_threshold: float | None = None, discordance: float | None = None,
                   repo: Path = REPO_ROOT) -> dict:
    if h3_comparator not in H3_COMPARATORS:
        raise ValueError(f"the H3 comparator must be one of {H3_COMPARATORS}, chosen on dev; got {h3_comparator!r}")
    q = quota_report(test)
    by_stratum = {s: sum(c.values()) for s, c in q.table.items()}
    ollama = {p.name: file_sha256(p) for p in sorted((REPO_ROOT / "config" / "ollama").glob("*.Modelfile"))}
    return {
        "version": PREREG_TAG,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_head(repo),
        "contract_version": CONTRACT_VERSION,
        "test": {"path": _repo_relative(test_path, repo),
                 "sha256": file_sha256(test_path), "n": len(test), "by_stratum": by_stratum,
                 "s1_test": q.s1_test, "s1_target": S1_TARGET, "s1_floor": S1_FLOOR,
                 "narrowing_rule_applies": q.narrowing_rule_applies, "s1_status": q.status},
        "test_sha256": file_sha256(test_path),
        "prompts": {"systems": prompts_sha(), "reader": READER_PROMPT_SHA, "validity_criteria": V1_CRITERIA_SHA},
        "schemas": schema_hashes(),
        "models": {n: tier_record(n) for n in PINNED_TIERS},
        "modelfiles": ollama,
        "thresholds": {
            "nli_threshold": DEFAULT_NLI_THRESHOLD if nli_threshold is None else nli_threshold,
            "nli_threshold_tuned": nli_threshold is not None,
            "b3_delta": DEFAULT_DELTA,
            "b3_authority": {k.value: v for k, v in AUTHORITY.items()},
        },
        "hypotheses": {
            "confirmatory_family": ["H1", "H2", "H3"], "population": "S1",
            "multiplicity": "Holm over H1-H3", "test": "exact McNemar (force_exact=True) on CCA-all strict",
            "h3_comparator": h3_comparator,
            "mechanism_family": ["H3-S3", "H4"],
        },
        "power": {"mdd_points": {str(n): r for n, r in mdd_table().items()}, "observed_discordance": discordance},
    }


def create_tag(repo: Path = REPO_ROOT, tag: str = PREREG_TAG, manifest_path: Path = MANIFEST) -> None:
    """Tag HEAD, but only if the manifest and test file are committed and unmodified."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [manifest_path.resolve().relative_to(repo.resolve()).as_posix(), manifest["test"]["path"]]
    st = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=repo, capture_output=True, text=True)
    if st.stdout.strip():
        raise RuntimeError("commit prereg/manifest.json and the test file first; refusing to tag:\n" + st.stdout)
    r = subprocess.run(["git", "tag", "-a", tag, "-m", "Pre-registration frozen"], cwd=repo,
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--test-out", type=Path, default=DEFAULT_TEST_PATH)
    ap.add_argument("--h3-comparator", choices=H3_COMPARATORS, default=None)
    ap.add_argument("--nli-threshold", type=float, default=None, help="the dev-calibrated entailment threshold")
    ap.add_argument("--discordance", type=float, default=None, help="observed dev discordance d")
    ap.add_argument("--tag", action="store_true", help="tag the already-committed freeze")
    args = ap.parse_args(argv)

    if args.tag:
        create_tag()
        print(f"tagged {PREREG_TAG}")
        return 0
    if args.corpus is None or args.h3_comparator is None:
        print("freeze needs --corpus and --h3-comparator (chosen on dev, before any test answer)", file=sys.stderr)
        return 2
    test = export_test(args.corpus, args.test_out)
    manifest = build_manifest(test, args.test_out, h3_comparator=args.h3_comparator,
                              nli_threshold=args.nli_threshold, discordance=args.discordance)
    PREREG_DIR.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    t = manifest["test"]
    print(f"wrote {args.test_out} ({t['n']} instances; S1 test = {t['s1_test']}; {t['s1_status']})")
    print(f"wrote {MANIFEST}\nnext: commit both, then `python -m prereg.freeze --tag`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
