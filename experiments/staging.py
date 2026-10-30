"""Gates that are enforced by code, not by good intentions (plan v4 E-10, G-FREEZE).

Two rules decide whether a result can be believed, and both are cheap to break
by accident:

**No test answer before the pre-registration is frozen.** A result obtained
after seeing the test set, and then "pre-registered", is not pre-registered.
:func:`assert_frozen` refuses to let any test-split run start unless

1. ``prereg/manifest.json`` exists and records the SHA-256 of the test file;
2. the git tag ``prereg-v1`` exists; and
3. the test file in the working tree is byte-identical to the one committed at
   that tag, so an edit made after the freeze cannot slip through.

**No lower-tier work before its gate passes.** Important and Optional
experiments are advisory unless something refuses them. :func:`can_run` reads
``prereg/gates.json`` and refuses to start one while a gate it depends on is
open or failed; "unknown" is treated as open, never as passed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_DIR = REPO_ROOT / "prereg"
MANIFEST = PREREG_DIR / "manifest.json"
GATES = PREREG_DIR / "gates.json"
PREREG_TAG = "prereg-v1"


class FreezeError(RuntimeError):
    """A test-split run was attempted without a valid freeze."""


class GateError(RuntimeError):
    """An experiment was attempted before a gate it depends on passed."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def tag_commit(tag: str = PREREG_TAG, *, repo: Path = REPO_ROOT) -> str | None:
    r = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}", repo=repo)
    return r.stdout.strip() or None if r.returncode == 0 else None


def assert_frozen(test_path: Path, *, manifest_path: Path = MANIFEST, repo: Path = REPO_ROOT,
                  tag: str = PREREG_TAG) -> dict:
    """Raise :class:`FreezeError` unless the pre-registration is frozen and the test file untouched."""
    test_path = Path(test_path)
    if not manifest_path.exists():
        raise FreezeError(f"{manifest_path} is missing: run prereg/freeze.py before any test-split run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not test_path.exists():
        raise FreezeError(f"test file {test_path} does not exist")
    actual = file_sha256(test_path)
    if manifest.get("test_sha256") != actual:
        raise FreezeError(
            f"the test file changed since the freeze (manifest {str(manifest.get('test_sha256'))[:12]}, "
            f"now {actual[:12]}); a result on an edited test set is not pre-registered")

    if tag_commit(tag, repo=repo) is None:
        raise FreezeError(f"git tag {tag!r} does not exist: the freeze has not been committed and tagged")

    rel = test_path.resolve().relative_to(repo.resolve()).as_posix()
    # `git diff` ignores untracked files, so first require that the file exists in the tagged commit.
    if _git("cat-file", "-e", f"{tag}:{rel}", repo=repo).returncode != 0:
        raise FreezeError(f"{rel} is not part of the commit tagged {tag!r}: it was never committed")
    # Then compare working tree with the tagged blob; an unmodified file has no diff.
    diff = _git("diff", "--quiet", tag, "--", rel, repo=repo)
    if diff.returncode != 0:
        raise FreezeError(f"{rel} differs from its committed version at {tag!r}")
    return manifest


# --------------------------------------------------------------------------- #
# Tiers and gates
# --------------------------------------------------------------------------- #

#: experiment -> (tier, gates that must have passed before it may start)
EXPERIMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "E1": ("essential", ()), "E2": ("essential", ()), "E2c": ("essential", ()),
    "E4": ("essential", ()), "E5": ("essential", ()),
    "E3": ("important", ("G1", "G4", "G-YIELD")), "E9": ("important", ("G1", "G4", "G-YIELD")),
    "E6": ("important", ("G1", "G4", "G-YIELD")), "E2b": ("important", ("G1", "G4", "G-YIELD")),
    "E7": ("important", ("G1", "G4", "G-YIELD")),
    "E8": ("optional", ("G-OPT",)), "E-pref": ("optional", ("G-OPT",)),
}

#: Statuses a gate can hold. Anything else, including absence, counts as open.
PASSED = "pass"


def load_gates(path: Path = GATES) -> dict[str, str]:
    if not path.exists():
        return {}
    return {k: (v.get("status") if isinstance(v, dict) else v)
            for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def can_run(experiment: str, gates: dict[str, str] | None = None) -> tuple[bool, str]:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {experiment!r}; known: {sorted(EXPERIMENTS)}")
    tier, needs = EXPERIMENTS[experiment]
    g = load_gates() if gates is None else gates
    blocked = [n for n in needs if g.get(n) != PASSED]
    if blocked:
        states = ", ".join(n + "=" + str(g.get(n, "open")) for n in blocked)
        return False, f"{experiment} ({tier}) is blocked: gate(s) {states} have not passed"
    return True, f"{experiment} ({tier}) may run"


def require(experiment: str, gates: dict[str, str] | None = None) -> None:
    ok, why = can_run(experiment, gates)
    if not ok:
        raise GateError(why)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", help="experiment id to check (omit to list every status)")
    args = ap.parse_args(argv)
    gates = load_gates()
    ids = [args.experiment] if args.experiment else sorted(EXPERIMENTS)
    rc = 0
    for e in ids:
        ok, why = can_run(e, gates)
        print(("ok     " if ok else "BLOCKED"), why)
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
