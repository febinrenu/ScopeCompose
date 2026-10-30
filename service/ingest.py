"""Bring your own documents: text, PDF or a URL, cut into passages.

Passages are sentence-aware chunks of roughly a paragraph -- the unit the
detector and the composition operator were built around (the benchmark's
passages average about a hundred characters, a sentence or two). Each keeps
its provenance: document id, URL, a source type guessed from where it came
from, and a date when the text states one.
"""

from __future__ import annotations

import html
import io
import re
from html.parser import HTMLParser

from contract.models import Passage, SourceType

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_DATE_ISO = re.compile(r"\b(20\d\d|19\d\d)-(0[1-9]|1[0-2])(?:-(0[1-9]|[12]\d|3[01]))?\b")
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_DATE_TEXT = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d\d)\b", re.I)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def chunk_text(text: str, *, max_words: int = 70) -> list[str]:
    """Pack sentences into chunks of up to ``max_words``; paragraphs never merge.

    A rule and its exception usually sit in different sentences, often in
    different paragraphs. Keeping chunks small keeps them apart, so the
    detector sees them as the pair they are rather than as one passage.
    """
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n|\r\n\s*\r\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        cur: list[str] = []
        words = 0
        for s in split_sentences(para) or [para]:
            n = len(s.split())
            if cur and words + n > max_words:
                chunks.append(" ".join(cur))
                cur, words = [], 0
            cur.append(s)
            words += n
        if cur:
            chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= 4]


def guess_source_type(name: str = "", text: str = "") -> SourceType:
    key = (name or "").lower()
    low = text[:400].lower()
    if ".gov" in key or "gov.uk" in key or "government" in low:
        return SourceType.GOVERNMENT_GUIDANCE
    if "faq" in key or "frequently asked" in low:
        return SourceType.FAQ
    if any(w in key for w in ("terms", "conditions", "tariff", "fees")) or "terms and conditions" in low:
        return SourceType.PRODUCT_TERMS
    if "amend" in key or "amendment" in low:
        return SourceType.AMENDMENT
    if "policy" in key or "policy" in low:
        return SourceType.OFFICIAL_POLICY
    return SourceType.UNKNOWN


def find_date(text: str) -> str | None:
    if (m := _DATE_ISO.search(text)):
        return m.group(0)
    if (m := _DATE_TEXT.search(text)):
        day, month, year = int(m.group(1)), _MONTHS[m.group(2).lower()], m.group(3)
        return f"{year}-{month:02d}-{day:02d}"
    return None


def passages_from_text(text: str, *, doc_id: str, name: str = "", source_url: str | None = None,
                       source_type: SourceType | None = None, start: int = 0) -> list[Passage]:
    st = source_type or guess_source_type(name or source_url or "", text)
    date = find_date(text)
    return [
        Passage(id=f"{doc_id}:{start + i}", text=chunk, source_type=st, date=date,
                source_url=source_url, document_id=doc_id)
        for i, chunk in enumerate(chunk_text(text))
    ]


def read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


class _TextOnly(HTMLParser):
    """Visible text from HTML, paragraph breaks kept, scripts and chrome dropped."""

    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "tr", "div", "section", "article", "br"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(raw: str) -> tuple[str, str]:
    p = _TextOnly()
    p.feed(raw)
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text)
    return text.strip(), " ".join(p.title.split())


def fetch_url(url: str, *, timeout: float = 15.0) -> tuple[str, str]:
    """(text, title) of a web page. Network access is the caller's choice."""
    import httpx

    r = httpx.get(url, timeout=timeout, follow_redirects=True,
                  headers={"User-Agent": "ScopeCompose-Studio/1.0 (research prototype)"})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return read_pdf(r.content), url.rsplit("/", 1)[-1]
    return html_to_text(r.text)
