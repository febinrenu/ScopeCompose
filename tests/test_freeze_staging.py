"""The freeze and the gates, against a real temporary git repository -- the tag logic is
exercised for real, not mocked, because it is the logic that decides whether a result counts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.staging import (
    EXPERIMENTS,
    FreezeError,
    GateError,
    assert_frozen,
    can_run,
    file_sha256,
    load_gates,
    require,
    tag_commit,
)
from prereg.freeze import PINNED_TIERS, build_manifest, create_tag, export_test

CORPUS = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "corpus.jsonl"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                          cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture()
def repo(tmp_path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "prereg").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


def frozen_repo(repo: Path, *, tag: bool = True, commit: bool = True) -> tuple[Path, Path]:
    test = repo / "data" / "test.jsonl"
    test.write_text('{"instance_id": "a"}\n{"instance_id": "b"}\n', encoding="utf-8")
    manifest = repo / "prereg" / "manifest.json"
    manifest.write_text(json.dumps({"test_sha256": file_sha256(test), "test": {"path": "data/test.jsonl"}}),
                        encoding="utf-8")
    if commit:
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "freeze")
        if tag:
            git(repo, "tag", "-a", "prereg-v1", "-m", "frozen")
    return test, manifest


# --------------------------------------------------------------------------- #
# assert_frozen
# --------------------------------------------------------------------------- #


def test_a_committed_tagged_untouched_test_file_passes(repo):
    test, manifest = frozen_repo(repo)
    assert assert_frozen(test, manifest_path=manifest, repo=repo)["test"]["path"] == "data/test.jsonl"
    assert tag_commit("prereg-v1", repo=repo) is not None


def test_a_missing_manifest_is_refused(repo):
    test, manifest = frozen_repo(repo)
    manifest.unlink()
    with pytest.raises(FreezeError, match="manifest"):
        assert_frozen(test, manifest_path=manifest, repo=repo)


def test_a_missing_tag_is_refused_even_with_a_matching_manifest(repo):
    test, manifest = frozen_repo(repo, tag=False)
    with pytest.raises(FreezeError, match="tag"):
        assert_frozen(test, manifest_path=manifest, repo=repo)


def test_a_test_file_edited_after_the_freeze_is_refused(repo):
    test, manifest = frozen_repo(repo)
    test.write_text(test.read_text(encoding="utf-8") + '{"instance_id": "c"}\n', encoding="utf-8")
    with pytest.raises(FreezeError, match="changed since the freeze"):
        assert_frozen(test, manifest_path=manifest, repo=repo)


def test_an_edit_that_keeps_the_manifest_hash_in_sync_still_differs_from_the_tag(repo):
    """Re-hashing the manifest after editing the test file must not launder the edit."""
    test, manifest = frozen_repo(repo)
    test.write_text('{"instance_id": "z"}\n', encoding="utf-8")
    manifest.write_text(json.dumps({"test_sha256": file_sha256(test), "test": {"path": "data/test.jsonl"}}),
                        encoding="utf-8")
    with pytest.raises(FreezeError, match="differs from its committed version"):
        assert_frozen(test, manifest_path=manifest, repo=repo)


def test_a_test_file_that_was_never_committed_is_refused(repo):
    """`git diff` ignores untracked files; the explicit tag-membership check must catch it."""
    keep, _ = frozen_repo(repo)
    new = repo / "data" / "other_test.jsonl"
    new.write_text('{"instance_id": "q"}\n', encoding="utf-8")
    manifest = repo / "prereg" / "manifest.json"
    manifest.write_text(json.dumps({"test_sha256": file_sha256(new)}), encoding="utf-8")
    with pytest.raises(FreezeError, match="never committed"):
        assert_frozen(new, manifest_path=manifest, repo=repo)


def test_a_missing_test_file_is_refused(repo):
    test, manifest = frozen_repo(repo)
    test.unlink()
    with pytest.raises(FreezeError, match="does not exist"):
        assert_frozen(test, manifest_path=manifest, repo=repo)


# --------------------------------------------------------------------------- #
# create_tag
# --------------------------------------------------------------------------- #


def test_create_tag_refuses_while_the_freeze_is_uncommitted_and_tags_once_it_is(repo):
    test, manifest = frozen_repo(repo, tag=False, commit=False)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")  # dirty
    with pytest.raises(RuntimeError, match="commit"):
        create_tag(repo, manifest_path=manifest)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "freeze")
    create_tag(repo, manifest_path=manifest)
    assert tag_commit("prereg-v1", repo=repo) is not None
    assert_frozen(test, manifest_path=manifest, repo=repo)


# --------------------------------------------------------------------------- #
# Gates and tiers
# --------------------------------------------------------------------------- #


def test_essential_experiments_have_no_gate_and_lower_tiers_do():
    for e, (tier, needs) in EXPERIMENTS.items():
        assert (tier == "essential") == (needs == ()), e


def test_an_unknown_or_open_gate_blocks_an_important_experiment():
    ok, why = can_run("E3", {})
    assert not ok and "G1=open" in why and "G4=open" in why and "G-YIELD=open" in why
    ok, why = can_run("E3", {"G1": "pass", "G4": "fail", "G-YIELD": "pass"})
    assert not ok and "G4=fail" in why and "G1" not in why


def test_all_gates_passed_unblocks_it_and_require_raises_otherwise():
    assert can_run("E3", {"G1": "pass", "G4": "pass", "G-YIELD": "pass"})[0]
    require("E1", {})
    with pytest.raises(GateError, match="blocked"):
        require("E8", {"G-OPT": "open"})
    require("E8", {"G-OPT": "pass"})


def test_an_unknown_experiment_is_an_error():
    with pytest.raises(KeyError):
        can_run("E99", {})


def test_the_shipped_gates_file_starts_with_every_gate_open():
    gates = load_gates()
    assert gates and set(gates.values()) == {"open"}
    assert {"G1", "G4", "G-YIELD", "G-FREEZE", "G-OPT"} <= set(gates)


# --------------------------------------------------------------------------- #
# The freeze itself
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_the_exported_test_file_excludes_s5_and_other_splits(tmp_path):
    from benchmark.strata import Stratum, stratum_of

    out = tmp_path / "test.jsonl"
    test = export_test(CORPUS, out)
    assert test and all(i.split == "test" for i in test)
    assert all(stratum_of(i) is not Stratum.S5 for i in test)
    ids = [json.loads(l)["instance_id"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert ids == sorted(ids)
    assert export_test(CORPUS, tmp_path / "again.jsonl") and file_sha256(out) == file_sha256(tmp_path / "again.jsonl"), \
        "the export is deterministic, so its hash is meaningful"


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_the_manifest_pins_prompts_schemas_models_and_the_comparator(tmp_path):
    out = tmp_path / "test.jsonl"
    test = export_test(CORPUS, out)
    m = build_manifest(test, out, h3_comparator="B7c")
    assert m["test_sha256"] == file_sha256(out) == m["test"]["sha256"]
    assert set(m["models"]) == set(PINNED_TIERS) and all(v["fallback"] is False for v in m["models"].values())
    assert len(m["prompts"]["systems"]) == len(m["prompts"]["reader"]) == 64
    assert set(m["schemas"]) == {"banking", "immigration"}
    assert m["hypotheses"]["h3_comparator"] == "B7c" and m["hypotheses"]["multiplicity"].startswith("Holm")
    assert "force_exact" in m["hypotheses"]["test"]
    assert m["thresholds"]["nli_threshold_tuned"] is False, "an untuned threshold is flagged, not hidden"
    assert set(m["modelfiles"]) == {"backbone7b.Modelfile", "reader.Modelfile"}


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_the_manifest_computes_the_narrowing_rule_from_the_data(tmp_path):
    out = tmp_path / "test.jsonl"
    m = build_manifest(export_test(CORPUS, out), out, h3_comparator="B8")
    assert m["test"]["narrowing_rule_applies"] is True and m["test"]["s1_test"] < m["test"]["s1_floor"]


def test_the_freeze_refuses_a_comparator_that_was_not_chosen_from_the_two(tmp_path):
    out = tmp_path / "t.jsonl"
    out.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="H3 comparator"):
        build_manifest([], out, h3_comparator="B0")


def test_the_cli_refuses_to_freeze_without_a_comparator():
    from prereg.freeze import main

    assert main(["--corpus", str(CORPUS)]) == 2
