"""Download the public benchmarks the study uses, and pin them by hash.

    python -m study.fetch_data            # download what is missing, verify everything
    python -m study.fetch_data --verify   # verify only; never touches the network

Files land in ``data/external/<benchmark>/`` (gitignored: the benchmarks are
redistributed by their authors, not by this repository). Each file's SHA-256 is
recorded in ``study/data_manifest.json`` the first time it is fetched and
checked on every later run, so a result can name the exact bytes it was
computed from and a silently changed upstream file is an error, not a new
number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import settings

DATA_ROOT = settings.REPO_ROOT / "data" / "external"
MANIFEST = Path(__file__).with_name("data_manifest.json")

_CQA = "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master"
_RAM = "https://huggingface.co/datasets/HanNight/RAMDocs/resolve/main"

#: benchmark -> {local file name: url}
SOURCES: dict[str, dict[str, str]] = {
    "conditionalqa": {
        "train.json": f"{_CQA}/v1_0/train.json",
        "dev.json": f"{_CQA}/v1_0/dev.json",
        "documents.json": f"{_CQA}/v1_0/documents.json",
        "evaluate.py.txt": f"{_CQA}/evaluate.py",     # kept as text: read, then vendored by hand
        "LICENSE.txt": f"{_CQA}/LISCENSE",
        "README.md": f"{_CQA}/README.md",
    },
    "ramdocs": {
        "RAMDocs_test.jsonl": f"{_RAM}/RAMDocs_test.jsonl",
        "README.md": f"{_RAM}/README.md",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_of(benchmark: str, name: str) -> Path:
    return DATA_ROOT / benchmark / name


def load_manifest() -> dict[str, dict[str, str]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def fetch(*, verify_only: bool = False) -> dict[str, dict[str, str]]:
    manifest = load_manifest()
    problems: list[str] = []
    for bench, files in SOURCES.items():
        for name, url in files.items():
            dest = path_of(bench, name)
            if not dest.exists():
                if verify_only:
                    problems.append(f"missing: {dest}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(url, headers={"User-Agent": "research-data-fetch"})
                with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (fixed https URLs)
                    dest.write_bytes(resp.read())
            digest = sha256(dest)
            pinned = manifest.setdefault(bench, {}).get(name)
            if pinned is None:
                manifest[bench][name] = digest
            elif pinned != digest:
                problems.append(f"hash mismatch: {dest} is {digest[:12]}, manifest pins {pinned[:12]}")
    if problems:
        raise SystemExit("\n".join(problems))
    if not verify_only:
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check hashes only; no network")
    args = ap.parse_args(argv)
    manifest = fetch(verify_only=args.verify)
    for bench, files in manifest.items():
        for name, digest in files.items():
            print(f"{bench}/{name}  {digest[:16]}  {path_of(bench, name).stat().st_size:>9,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
