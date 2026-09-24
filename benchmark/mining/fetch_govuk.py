"""Fetch gov.uk pages through the official Content API.

Uses ``https://www.gov.uk/api/content/<path>`` rather than scraping HTML. It
is the publisher's own interface, returns clean structured text, and does not
pretend to be a browser.

**The reason this module exists rather than a ten-line `requests` loop.** The
API returns a ``content_id``, and that field decides whether a mined pair is
Tier 1 at all. Many gov.uk visa pages are *guides*: one CMS document with a
dozen parts, each with its own URL. ``/student-visa`` and
``/student-visa/money`` look like two documents in a browser and share a
single ``content_id``. Pairing them and calling the result "naturally
occurring, multi-source" would be false — and it is exactly the mistake the
proposal's Tier-1 scarcity risk anticipates.

So every fetched passage records where it really came from, and the miner can
then separate three cases the proposal's binary does not cover:

``distinct``
    Two different content items. Genuine Tier 1.
``same_guide``
    Two parts of one guide. Separately *retrievable* -- a RAG chunker would
    index them apart, and a retriever can surface one without the other -- but
    not separately *authored*. This is a real third category, weaker than
    Tier 1 and stronger than Tier 2, and it should be reported as its own
    thing rather than folded into either.
``broken``
    404, or a redirect to the other member of the pair.

Usage::

    python -m benchmark.mining.fetch_govuk --pairs pairs.txt -o docs.jsonl
    python -m benchmark.mining.fetch_govuk --pairs pairs.txt --audit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

API = "https://www.gov.uk/api/content"
UA = "conflict-rag-research/0.1 (final-year project; contact via repository)"

#: gov.uk asks for considerate use of the Content API. One request every
#: 400ms is well under anything that could matter and still fetches 30 pages
#: in under a minute.
DELAY_S = 0.4

_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "div", "tr", "br", "section"}


class _TextExtractor(HTMLParser):
    """Flatten gov.uk body HTML to text, one block element per line.

    Line-per-block matters downstream: the miner works sentence by sentence,
    and a run-together wall of text merges a rule and its carve-out into one
    unsplittable string.
    """

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.chunks.append(data)

    def text(self) -> str:
        raw = unescape("".join(self.chunks))
        lines = [" ".join(ln.split()) for ln in raw.split("\n")]
        return "\n".join(ln for ln in lines if ln)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html or "")
    return p.text()


@dataclass
class FetchedPage:
    """One retrievable unit: a whole document, or one part of a guide."""

    url: str
    base_path: str
    content_id: str
    schema: str
    title: str
    part_slug: str | None
    text: str
    updated: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.text) > 200

    @property
    def doc_id(self) -> str:
        """Identity for mining. Distinct per URL, so guide parts stay separate."""
        return f"{self.base_path.strip('/')}" + (f"#{self.part_slug}" if self.part_slug else "")


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{API}{path}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch(url: str, cache: dict[str, dict]) -> FetchedPage:
    """Fetch one URL, resolving a guide part to its own text.

    The base document is cached, so a guide fetched for five of its parts
    costs one request, not five.
    """
    path = urlparse(url).path.rstrip("/")
    segments = [s for s in path.split("/") if s]

    # A guide part is <base>/<slug>. Try the full path first; if the API
    # returns a guide, the part slug is the last segment.
    base_path = path
    part_slug = None

    try:
        if path not in cache:
            cache[path] = _get(path)
            time.sleep(DELAY_S)
        doc = cache[path]
    except urllib.error.HTTPError as exc:
        return FetchedPage(url=url, base_path=path, content_id="", schema="",
                           title="", part_slug=None, text="",
                           error=f"HTTP {exc.code}")
    except Exception as exc:  # network, timeout, malformed JSON
        return FetchedPage(url=url, base_path=path, content_id="", schema="",
                           title="", part_slug=None, text="",
                           error=f"{type(exc).__name__}: {exc}")

    schema = doc.get("schema_name", "")
    if schema == "redirect":
        return FetchedPage(url=url, base_path=path, content_id=doc.get("content_id", ""),
                           schema=schema, title=doc.get("title", ""), part_slug=None,
                           text="", error="redirect")

    details = doc.get("details", {}) or {}
    parts = details.get("parts") or []

    if parts:
        # The API returns the whole guide for any of its part URLs, so the
        # base path is the guide root and the part is identified by slug.
        if len(segments) > 1 and any(p.get("slug") == segments[-1] for p in parts):
            part_slug = segments[-1]
            base_path = "/" + "/".join(segments[:-1])
        else:
            part_slug = parts[0].get("slug")
            base_path = path

        chosen = next((p for p in parts if p.get("slug") == part_slug), parts[0])
        text = html_to_text(chosen.get("body", ""))
        title = f"{doc.get('title', '')} — {chosen.get('title', '')}".strip(" —")
    else:
        text = html_to_text(details.get("body", "") or "")
        title = doc.get("title", "")

    return FetchedPage(
        url=url, base_path=base_path, content_id=doc.get("content_id", ""),
        schema=schema, title=title, part_slug=part_slug, text=text,
        updated=(doc.get("public_updated_at") or "")[:10] or None,
    )


# --------------------------------------------------------------------------- #
# Pair auditing
# --------------------------------------------------------------------------- #


@dataclass
class PairAudit:
    provider: str
    urls: tuple[str, str]
    verdict: str          # distinct | same_guide | broken
    detail: str
    content_ids: tuple[str, str] = ("", "")


@dataclass
class AuditReport:
    pairs: list[PairAudit] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.pairs:
            out[p.verdict] = out.get(p.verdict, 0) + 1
        return out

    def render(self) -> str:
        c = self.counts()
        n = len(self.pairs)
        distinct = c.get("distinct", 0)
        lines = [
            "Pair audit — are these genuinely two documents?",
            "=" * 88,
            "",
            "  gov.uk 'guides' are ONE content item with many parts, each with its own",
            "  URL. Two parts of one guide look like two documents in a browser and",
            "  share a content_id. Pairing them and calling the result naturally",
            "  occurring multi-source would be false.",
            "",
            f"  {'provider':<32} {'verdict':<12} detail",
            "  " + "-" * 84,
        ]
        for p in sorted(self.pairs, key=lambda x: (x.verdict, x.provider)):
            lines.append(f"  {p.provider:<32} {p.verdict:<12} {p.detail[:40]}")

        lines += [
            "",
            f"  distinct documents (Tier 1 candidates)  {distinct:>3} / {n}",
            f"  same guide, different parts             {c.get('same_guide', 0):>3} / {n}",
            f"  broken (404 or redirect)                {c.get('broken', 0):>3} / {n}",
        ]
        return "\n".join(lines)


def audit_pairs(pairs: dict[str, list[str]], pages: dict[str, FetchedPage]) -> AuditReport:
    report = AuditReport()
    for provider, urls in pairs.items():
        if len(urls) < 2:
            report.pairs.append(PairAudit(provider, (urls[0], ""), "broken",
                                          "only one URL given"))
            continue
        a, b = pages.get(urls[0]), pages.get(urls[1])
        if a is None or b is None or a.error or b.error:
            why = (a.error if a and a.error else None) or \
                  (b.error if b and b.error else None) or "not fetched"
            report.pairs.append(PairAudit(provider, (urls[0], urls[1]), "broken", why))
            continue
        if a.content_id and a.content_id == b.content_id:
            same_part = a.part_slug == b.part_slug
            report.pairs.append(PairAudit(
                provider, (urls[0], urls[1]), "same_guide",
                "identical page" if same_part
                else f"parts '{a.part_slug}' and '{b.part_slug}' of one guide",
                (a.content_id, b.content_id),
            ))
            continue
        report.pairs.append(PairAudit(provider, (urls[0], urls[1]), "distinct",
                                      f"{a.schema} + {b.schema}",
                                      (a.content_id, b.content_id)))
    return report


def parse_pairs(path: Path) -> dict[str, list[str]]:
    """Read ``provider <whitespace> url`` lines, preserving order."""
    pairs: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        bits = raw.split()
        if len(bits) < 2:
            continue
        pairs.setdefault(bits[0], []).append(bits[1])
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="text file of 'provider<space>url' lines")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write SourceDoc JSONL here, for the Tier-1 miner")
    ap.add_argument("--audit", action="store_true",
                    help="only report whether each pair is really two documents")
    ap.add_argument("--raw", type=Path, default=None,
                    help="also dump the fetched pages with full text")
    args = ap.parse_args(argv)

    pairs = parse_pairs(args.pairs)
    urls = list(dict.fromkeys(u for us in pairs.values() for u in us))
    print(f"{len(pairs)} pairings, {len(urls)} unique URLs\n")

    cache: dict[str, dict] = {}
    pages: dict[str, FetchedPage] = {}
    for i, url in enumerate(urls, 1):
        page = fetch(url, cache)
        pages[url] = page
        status = "ok" if page.ok else (page.error or "empty")
        print(f"  [{i:>2}/{len(urls)}] {status:<12} {len(page.text):>6} chars  {url}")

    report = audit_pairs(pairs, pages)
    print()
    print(report.render())

    if args.raw:
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        with args.raw.open("w", encoding="utf-8", newline="\n") as fh:
            for p in pages.values():
                fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        print(f"\nraw pages -> {args.raw}")

    if args.audit:
        return 0

    if args.output:
        # Emit in the miner's SourceDoc shape. `provider` is the PAIRING, so
        # the miner looks for rule/exception pairs within each intended pair
        # rather than across every gov.uk page at once.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with args.output.open("w", encoding="utf-8", newline="\n") as fh:
            for provider, us in pairs.items():
                for url in us:
                    page = pages.get(url)
                    if page is None or not page.ok:
                        continue
                    fh.write(json.dumps({
                        "doc_id": page.doc_id,
                        "provider": provider,
                        "kind": page.schema,
                        "url": url,
                        "text": page.text,
                        "content_id": page.content_id,
                    }, ensure_ascii=False) + "\n")
                    n += 1
        print(f"\nwrote {n} documents -> {args.output}")
        print(f"next: python -m benchmark.mining.tier1_pilot --docs {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
