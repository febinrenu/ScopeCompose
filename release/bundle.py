"""The public release bundle: share the benchmark without redistributing text we have no right to.

Passages come from three kinds of source:

* **authored** (``example.invalid`` and passages with no ``source_url``): written by us or generated
  for the benchmark. Released as is;
* **open** (``gov.uk``): Open Government Licence v3.0. Released as is, with the attribution in the manifest;
* **everything else** (bank terms and conditions): copyright holders' text. **Not released.** The instance
  keeps a placeholder and the bundle records, per passage, *where the text is* (URL, character offsets in
  the document) and *what it is* (SHA-256 of the exact text, SHA-256 of the whole document).

``rehydrate`` rebuilds the original instances from a copy of the documents the user fetches themselves, and
**checks the hashes**. A page that has changed since the benchmark was built is reported as drift for that
passage, never silently accepted: scores computed on text that is not the benchmark's text are not the
benchmark's scores.

Rules enforced here, because each one is a way a release quietly goes wrong:

* a passage that cannot be *located* in its document (so could never be rehydrated) stops the build;
* redacted text must not reappear in another field of the same instance (a gold answer or outcome that
  quotes a long run of it): such spans are listed and stop the build unless explicitly allowed;
* the manifest records the SHA-256 of every file it ships, and ``verify`` recomputes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

from contract.gold import GoldInstance
from contract.models import Passage

RELEASE_VERSION = "1"
REDACTED_TEXT = "[text withheld: rehydrate from the source URL, see redactions.jsonl]"

#: host -> licence under which the text may be redistributed. Hosts not listed are redacted.
OPEN_HOSTS = {
    "www.gov.uk": "Open Government Licence v3.0",
    "gov.uk": "Open Government Licence v3.0",
    "example.invalid": "authored for this benchmark",
}
#: A span of this many consecutive words from redacted text, found in another field, counts as quoting it.
QUOTE_WORDS = 10

_WS = re.compile(r"\s+")


class ReleaseError(ValueError):
    """The release cannot be built or verified as asked."""


class DriftError(ReleaseError):
    """A document no longer contains the text the benchmark recorded."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def policy(url: str | None) -> str:
    """``open`` or ``redact`` for a passage with this source URL."""
    if not url:
        return "open"                                  # no source: authored or generated for the benchmark
    return "open" if (urlparse(url).netloc.lower() in OPEN_HOSTS) else "redact"


def licence(url: str | None) -> str:
    return OPEN_HOSTS.get(urlparse(url or "").netloc.lower(), "authored for this benchmark")


# --------------------------------------------------------------------------- #
# Locating a passage inside its document
# --------------------------------------------------------------------------- #


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"', 0xFFFD: "'"}


def _fold_char(ch: str) -> str:
    """One character in the form used to compare text that differs only in typography."""
    return unicodedata.normalize("NFKC", ch).translate(_DASHES).translate(_QUOTES)


def flatten(text: str) -> str:
    """Whitespace-collapsed, typography-folded text: what a passage is compared as when not verbatim."""
    return collapse("".join(_fold_char(c) for c in text))


def _flat_with_map(text: str) -> tuple[str, list[int]]:
    """:func:`flatten` of ``text`` and, for each character of it, its index in the original."""
    out, idx, prev_ws = [], [], True
    for i, ch in enumerate(text):
        for f in _fold_char(ch) or " ":
            if f.isspace():
                if not prev_ws:
                    out.append(" ")
                    idx.append(i)
                prev_ws = True
            else:
                out.append(f)
                idx.append(i)
                prev_ws = False
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def locate(passage_text: str, doc_text: str) -> tuple[int, int, bool] | None:
    """``(start, end, exact)`` of the passage in the document, or ``None``.

    ``exact`` is True when the passage is a verbatim slice. Otherwise it matches only after whitespace is
    collapsed and typography is folded (re-wrapped lines, curly quotes, non-breaking hyphens), and the
    offsets still bound the passage in the original."""
    at = doc_text.find(passage_text)
    if at >= 0:
        return at, at + len(passage_text), True
    flat_doc, idx = _flat_with_map(doc_text)
    needle = flatten(passage_text)
    at = flat_doc.find(needle) if needle else -1
    if at < 0:
        return None
    return idx[at], idx[at + len(needle) - 1] + 1, False


@dataclass(frozen=True)
class Redaction:
    instance_id: str
    passage_id: str
    url: str
    start: int
    end: int
    exact: bool
    text_sha256: str          # of the passage text (flattened when not exact)
    document_sha256: str
    n_chars: int
    located: bool = True      # False: not found in the stored document, so it cannot be restored


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _strings(inst: GoldInstance, skip_passage_ids: set[str]) -> list[str]:
    """Every authored string in the instance that is not itself a redacted passage."""
    out = [inst.query, inst.gold_scoped_answer or "", inst.selection_answer or ""]
    for b in inst.gold_branches:
        out += [b.outcome, b.condition or "", b.applicability.descriptor]
    for p in inst.passages:
        if p.id not in skip_passage_ids:
            out.append(p.text)
    for ps in inst.personas:
        out.append(ps.statement)
    return [s for s in out if s]


def quoted_spans(text: str, others: Iterable[str], n: int = QUOTE_WORDS) -> list[str]:
    """Runs of ``n`` words of ``text`` that appear verbatim (whitespace-insensitively) in ``others``."""
    hay = collapse(" \n ".join(others)).lower()
    words = collapse(text).split()
    found: list[str] = []
    i = 0
    while i + n <= len(words):
        span = " ".join(words[i:i + n])
        if span.lower() in hay:
            j = i + n
            while j < len(words) and " ".join(words[i:j + 1]).lower() in hay:
                j += 1
            found.append(" ".join(words[i:j]))
            i = j
        else:
            i += 1
    return found


@dataclass
class BuildReport:
    n_instances: int = 0
    n_passages: int = 0
    n_redacted: int = 0
    licences: dict[str, int] = field(default_factory=dict)
    quoted: list[dict] = field(default_factory=list)
    unlocated: list[str] = field(default_factory=list)


def redact(instances: Sequence[GoldInstance], docs_by_url: Mapping[str, str], *, allow_quotes: bool = False,
           allow_unlocated: bool = False) -> tuple[list[GoldInstance], list[Redaction], BuildReport]:
    """Withhold the text of non-open passages; return released instances, the redaction records and a report."""
    report = BuildReport()
    released: list[GoldInstance] = []
    records: list[Redaction] = []
    for inst in instances:
        report.n_instances += 1
        new_passages: list[Passage] = []
        redacted_ids: dict[str, str] = {}
        for p in inst.passages:
            report.n_passages += 1
            if policy(p.source_url) == "open":
                lic = licence(p.source_url)
                report.licences[lic] = report.licences.get(lic, 0) + 1
                new_passages.append(p)
                continue
            doc = docs_by_url.get(p.source_url or "")
            loc = locate(p.text, doc) if doc else None
            if loc is None:
                report.unlocated.append(f"{inst.instance_id}/{p.id} ({p.source_url})")
                records.append(Redaction(inst.instance_id, p.id, p.source_url or "", -1, -1, False,
                                         sha256(flatten(p.text)), sha256(doc) if doc else "", len(p.text), False))
                redacted_ids[p.id] = p.text
                new_passages.append(p.model_copy(update={"text": REDACTED_TEXT}))
                report.n_redacted += 1
                continue
            start, end, exact = loc
            body = p.text if exact else flatten(p.text)
            records.append(Redaction(inst.instance_id, p.id, p.source_url or "", start, end, exact, sha256(body),
                                     sha256(doc), end - start))
            redacted_ids[p.id] = p.text
            new_passages.append(p.model_copy(update={"text": REDACTED_TEXT}))
            report.n_redacted += 1
        for pid, text in redacted_ids.items():
            for span in quoted_spans(text, _strings(inst, set(redacted_ids))):
                report.quoted.append({"instance": inst.instance_id, "passage": pid, "span": span})
        released.append(inst.model_copy(update={"passages": new_passages}))
    if report.unlocated and not allow_unlocated:
        raise ReleaseError("passages that cannot be located in their documents (so could never be rehydrated): "
                           + "; ".join(report.unlocated[:6]) + (" ..." if len(report.unlocated) > 6 else "")
                           + ". Pass allow_unlocated to withhold them knowingly; they are then listed as unrestorable.")
    if report.quoted and not allow_quotes:
        first = report.quoted[0]
        raise ReleaseError(f"{len(report.quoted)} redacted span(s) of {QUOTE_WORDS}+ words reappear in another "
                           f"field, e.g. {first['instance']}/{first['passage']}: {first['span']!r}. "
                           f"Reword the field, or pass allow_quotes to ship it knowingly.")
    return released, records, report


def build_release(instances: Sequence[GoldInstance], docs_by_url: Mapping[str, str], out_dir: Path, *,
                  allow_quotes: bool = False, allow_unlocated: bool = False) -> dict:
    """Write ``instances.release.jsonl``, ``redactions.jsonl`` and ``manifest.json`` to ``out_dir``."""
    released, records, report = redact(instances, docs_by_url, allow_quotes=allow_quotes,
                                       allow_unlocated=allow_unlocated)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_path, red_path = out_dir / "instances.release.jsonl", out_dir / "redactions.jsonl"
    inst_path.write_text("".join(json.dumps(i.model_dump(mode="json"), ensure_ascii=False) + "\n"
                                 for i in released), encoding="utf-8")
    red_path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    manifest = {
        "release_version": RELEASE_VERSION,
        "n_instances": report.n_instances, "n_passages": report.n_passages, "n_redacted": report.n_redacted,
        "open_passages_by_licence": report.licences,
        "redaction_policy": "text of passages whose source host is not in open_hosts is withheld",
        "open_hosts": OPEN_HOSTS,
        "quoted_spans_shipped": report.quoted if allow_quotes else [],
        "unrestorable": report.unlocated,
        "files": {inst_path.name: sha256(inst_path.read_text(encoding="utf-8")),
                  red_path.name: sha256(red_path.read_text(encoding="utf-8"))},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify(out_dir: Path) -> None:
    """Recompute the hash of every file the manifest lists."""
    mf = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, want in mf["files"].items():
        got = sha256((out_dir / name).read_text(encoding="utf-8"))
        if got != want:
            raise ReleaseError(f"{name} does not match the manifest (edited after the release was built)")


# --------------------------------------------------------------------------- #
# Rehydrate
# --------------------------------------------------------------------------- #


@dataclass
class RehydrateReport:
    restored: int = 0
    document_changed: list[str] = field(default_factory=list)
    unrestorable: list[str] = field(default_factory=list)
    """Passages recorded as not locatable at build time: they stay withheld and their instances are unusable."""


def rehydrate(out_dir: Path, docs_by_url: Mapping[str, str]) -> tuple[list[GoldInstance], RehydrateReport]:
    """Rebuild the original instances from documents the caller fetched.

    Raises :class:`DriftError` if a document is missing or any passage's text no longer hashes to the recorded
    value. A document whose *other* text changed but whose passage did not is restored, and listed."""
    verify(out_dir)
    records = [json.loads(l) for l in (out_dir / "redactions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = {(r["instance_id"], r["passage_id"]): r for r in records}
    report = RehydrateReport()
    out: list[GoldInstance] = []
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line in (out_dir / "instances.release.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        inst = GoldInstance.model_validate(json.loads(line))
        passages = []
        for p in inst.passages:
            r = by_key.get((inst.instance_id, p.id))
            if r is None:
                passages.append(p)
                continue
            seen.add((inst.instance_id, p.id))
            if not r.get("located", True):
                report.unrestorable.append(f"{inst.instance_id}/{p.id}")
                passages.append(p)
                continue
            doc = docs_by_url.get(r["url"])
            if doc is None:
                problems.append(f"{inst.instance_id}/{p.id}: no document supplied for {r['url']}")
                passages.append(p)
                continue
            text = doc[r["start"]:r["end"]]
            body = text if r["exact"] else flatten(text)
            if sha256(body) != r["text_sha256"]:
                problems.append(f"{inst.instance_id}/{p.id}: the text at {r['url']}[{r['start']}:{r['end']}] has changed")
                passages.append(p)
                continue
            if sha256(doc) != r["document_sha256"]:
                report.document_changed.append(f"{inst.instance_id}/{p.id}")
            passages.append(p.model_copy(update={"text": text}))
            report.restored += 1
        out.append(inst.model_copy(update={"passages": passages}))
    missing = set(by_key) - seen
    if missing:
        problems.append(f"{len(missing)} redaction record(s) match no passage in the release")
    if problems:
        raise DriftError(f"{len(problems)} passage(s) could not be restored: " + "; ".join(problems[:6])
                         + (" ..." if len(problems) > 6 else ""))
    return out, report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_docs(*paths: Path) -> dict[str, str]:
    """``url -> text`` from JSONL files with ``url`` and ``text`` fields."""
    docs: dict[str, str] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("url") and d.get("text"):
                    docs[d["url"]] = d["text"]
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="redact and write a release bundle")
    b.add_argument("--data", type=Path, required=True, help="gold instances (JSONL)")
    b.add_argument("--docs", type=Path, nargs="+", required=True, help="document JSONL files with url and text")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--allow-quotes", action="store_true")
    b.add_argument("--allow-unlocated", action="store_true",
                   help="withhold passages not found in their documents (listed as unrestorable)")
    r = sub.add_parser("rehydrate", help="rebuild instances from documents you fetched")
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--docs", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify", help="check the manifest hashes")
    v.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "build":
            from experiments.run_matrix import load_instances

            mf = build_release(load_instances(args.data), load_docs(*args.docs), args.out,
                               allow_quotes=args.allow_quotes, allow_unlocated=args.allow_unlocated)
            print(f"released {mf['n_instances']} instances; {mf['n_redacted']} of {mf['n_passages']} passages withheld")
        elif args.cmd == "verify":
            verify(args.bundle)
            print("manifest hashes match")
        else:
            insts, rep = rehydrate(args.bundle, load_docs(*args.docs))
            args.out.write_text("".join(json.dumps(i.model_dump(mode="json"), ensure_ascii=False) + "\n"
                                        for i in insts), encoding="utf-8")
            print(f"restored {rep.restored} passages into {len(insts)} instances"
                  + (f"; {len(rep.document_changed)} document(s) changed elsewhere" if rep.document_changed else "")
                  + (f"; {len(rep.unrestorable)} passage(s) unrestorable" if rep.unrestorable else ""))
    except ReleaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
