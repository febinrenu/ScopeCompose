"""Fetch arbitrary public web pages and PDFs for Tier-1 mining.

The gov.uk fetcher uses that publisher's Content API. Commercial sites have no
such interface, so this is the general path: plain HTTP, HTML or PDF, one
request per URL.

Three things it does that a ten-line loop would not:

**Respects robots.txt, per host, before every fetch.** Not a formality — one
of the URLs in the first banking batch pointed at an origin-server hostname
that robots.txt explicitly disallows. Skipping it is the correct outcome, and
the skip is reported rather than silently dropped.

**Extracts the main content region.** A bank product page is mostly navigation,
cookie notice, and footer. Feeding all of it to the miner buries the four
sentences that state a rule. This prefers ``<main>`` / ``<article>`` and falls
back to the densest block.

**Hashes the extracted text.** Commercial sites expose no ``content_id``, so
the same-document test has nothing authoritative to compare. Two URLs whose
extracted text is identical are one document — which happens more than you
would expect, because several product pages can share one charges page.

Usage::

    python -m benchmark.mining.fetch_web --pairs pairs.txt -o docs.jsonl
    python -m benchmark.mining.fetch_web --pairs pairs.txt --audit
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

UA = "conflict-rag-research/0.1 (final-year academic project)"
DELAY_S = 1.0
"""One second between requests. These are commercial sites being read for
research, not an API that invited traffic."""

TIMEOUT_S = 45

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript", "svg",
              "form", "button", "select", "aside"}
_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "div", "tr", "td",
               "br", "section", "dd", "dt"}
_MAIN_TAGS = {"main", "article"}


class _Extractor(HTMLParser):
    """Pull readable text, preferring a main-content region if one is marked."""

    def __init__(self) -> None:
        super().__init__()
        self.all_chunks: list[str] = []
        self.main_chunks: list[str] = []
        self._skip_depth = 0
        self._main_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in _MAIN_TAGS or dict(attrs).get("role") == "main":
            self._main_depth += 1
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in _MAIN_TAGS and self._main_depth:
            self._main_depth -= 1
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._emit(data)

    def _emit(self, s: str) -> None:
        self.all_chunks.append(s)
        if self._main_depth:
            self.main_chunks.append(s)

    @staticmethod
    def _clean(chunks: list[str]) -> str:
        raw = unescape("".join(chunks))
        lines = [" ".join(ln.split()) for ln in raw.split("\n")]
        return "\n".join(ln for ln in lines if ln)

    def text(self) -> str:
        main = self._clean(self.main_chunks)
        everything = self._clean(self.all_chunks)
        # Use the marked main region only if it actually carries the content.
        # Some templates mark <main> around a shell and render the body
        # elsewhere, which would silently return almost nothing.
        return main if len(main) > 0.25 * len(everything) and len(main) > 400 else everything


def html_to_text(html: str) -> str:
    p = _Extractor()
    p.feed(html)
    return p.text()


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    raw = "\n".join(pages)
    lines = [" ".join(ln.split()) for ln in raw.split("\n")]
    return "\n".join(ln for ln in lines if ln)


@dataclass
class FetchedDoc:
    url: str
    provider: str
    kind: str = ""
    text: str = ""
    content_hash: str = ""
    bytes_len: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.text) > 400


class RobotsCache:
    """One robots.txt per host, fetched once.

    A host whose robots.txt cannot be read is treated as DISALLOWED. The
    conservative direction is the correct one when the question is whether a
    publisher wants to be crawled.
    """

    def __init__(self) -> None:
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> tuple[bool, str]:
        host = urlparse(url).netloc
        if host not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"https://{host}/robots.txt")
            try:
                rp.read()
                self._cache[host] = rp
            except Exception:
                self._cache[host] = None
        rp = self._cache[host]
        if rp is None:
            return False, "robots.txt unreadable — treated as disallowed"
        return (True, "") if rp.can_fetch(UA, url) else (False, "disallowed by robots.txt")


def fetch(url: str, provider: str, robots: RobotsCache) -> FetchedDoc:
    allowed, why = robots.allowed(url)
    if not allowed:
        return FetchedDoc(url=url, provider=provider, error=why)

    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/pdf,*/*",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            data = r.read()
    except urllib.error.HTTPError as exc:
        return FetchedDoc(url=url, provider=provider, error=f"HTTP {exc.code}")
    except Exception as exc:
        return FetchedDoc(url=url, provider=provider,
                          error=f"{type(exc).__name__}: {str(exc)[:70]}")
    finally:
        time.sleep(DELAY_S)

    try:
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text, kind = pdf_to_text(data), "pdf"
        else:
            text, kind = html_to_text(data.decode("utf-8", errors="replace")), "html"
    except Exception as exc:
        return FetchedDoc(url=url, provider=provider, kind=ctype,
                          bytes_len=len(data),
                          error=f"parse failed: {type(exc).__name__}")

    return FetchedDoc(
        url=url, provider=provider, kind=kind, text=text, bytes_len=len(data),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


@dataclass
class PairAudit:
    provider: str
    verdict: str          # distinct | same_document | broken
    detail: str


@dataclass
class AuditReport:
    pairs: list[PairAudit] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.pairs:
            out[p.verdict] = out.get(p.verdict, 0) + 1
        return out

    def render(self) -> str:
        c, n = self.counts(), len(self.pairs)
        lines = [
            "Pair audit — are these genuinely two documents?",
            "=" * 88,
            "",
            "  No content_id on commercial sites, so identity is the hash of the",
            "  extracted text. Two URLs rendering the same text are one document.",
            "",
            f"  {'provider':<28} {'verdict':<14} detail",
            "  " + "-" * 84,
        ]
        for p in sorted(self.pairs, key=lambda x: (x.verdict, x.provider)):
            lines.append(f"  {p.provider:<28} {p.verdict:<14} {p.detail[:42]}")
        lines += [
            "",
            f"  distinct documents (Tier 1 candidates)  {c.get('distinct', 0):>3} / {n}",
            f"  same document                           {c.get('same_document', 0):>3} / {n}",
            f"  broken or blocked                       {c.get('broken', 0):>3} / {n}",
        ]
        return "\n".join(lines)


def audit(pairs: dict[str, list[str]], docs: dict[str, FetchedDoc]) -> AuditReport:
    rep = AuditReport()
    for provider, urls in pairs.items():
        if len(urls) < 2:
            rep.pairs.append(PairAudit(provider, "broken", "only one URL given"))
            continue
        a, b = docs.get(urls[0]), docs.get(urls[1])
        if a is None or b is None or not a.ok or not b.ok:
            why = next((d.error for d in (a, b) if d is not None and d.error),
                       "extracted too little text")
            rep.pairs.append(PairAudit(provider, "broken", why))
            continue
        if a.content_hash == b.content_hash:
            rep.pairs.append(PairAudit(provider, "same_document",
                                       "identical extracted text"))
            continue
        rep.pairs.append(PairAudit(provider, "distinct", f"{a.kind} + {b.kind}"))
    return rep


def parse_pairs(path: Path) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        bits = raw.split()
        if len(bits) >= 2:
            pairs.setdefault(bits[0], []).append(bits[1])
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args(argv)

    pairs = parse_pairs(args.pairs)
    urls = list(dict.fromkeys(u for us in pairs.values() for u in us))
    owner = {u: p for p, us in pairs.items() for u in us}
    print(f"{len(pairs)} pairings, {len(urls)} unique URLs\n")

    robots = RobotsCache()
    docs: dict[str, FetchedDoc] = {}
    for i, url in enumerate(urls, 1):
        d = fetch(url, owner[url], robots)
        docs[url] = d
        state = "ok" if d.ok else (d.error or "too short")
        print(f"  [{i:>2}/{len(urls)}] {state[:34]:<36} {len(d.text):>7} chars  {url[:66]}")

    print()
    print(audit(pairs, docs).render())

    if args.audit:
        return 0

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with args.output.open("w", encoding="utf-8", newline="\n") as fh:
            for provider, us in pairs.items():
                for url in us:
                    d = docs.get(url)
                    if d is None or not d.ok:
                        continue
                    fh.write(json.dumps({
                        "doc_id": urlparse(url).path.strip("/")[-80:] or url,
                        "provider": provider,
                        "kind": d.kind,
                        "url": url,
                        "text": d.text,
                        "content_id": d.content_hash,
                    }, ensure_ascii=False) + "\n")
                    n += 1
        print(f"\nwrote {n} documents -> {args.output}")
        print(f"next: python -m benchmark.mining.tier1_pilot --docs {args.output} --sensitivity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
