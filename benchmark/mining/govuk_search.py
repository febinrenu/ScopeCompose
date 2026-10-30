"""Harvest distinct gov.uk content items per topic through the Search API.

**Why this exists.** The pilot's binding constraint is S1 yield: natural
cross-document conditional conflicts, where the rule and its exception sit in
*separately authored* documents. The pilot found 0.34 candidates per pairing
across 91 hand-picked pairings, and 10 of 17 early "pairs" were one guide under
two URLs (``wp1_tier1_findings.md``). Hand-picking pairs does not scale.

The pairs that worked share a shape -- a consumer-facing page (``/graduate-visa``)
set against the rules or caseworker guidance that govern the same route
(``/guidance/immigration-rules/...``, ``/government/publications/...``) -- and
those are exactly the items the Search API can enumerate by topic. Each result
carries a ``content_id``, so distinctness is **checked at harvest time**, not
discovered after a pair has been mined and annotated.

Records ``content_id``, ``document_type`` and ``public_updated_at`` (proposal v8
section 6.1), because they are the provenance that makes a ``natural`` claim
auditable.

Politeness: one request per ``DELAY_S``, an identifying User-Agent, and every
response cached on disk so a re-run costs nothing.

Usage::

    python -m benchmark.mining.govuk_search --topics immigration -o benchmark/data/govuk_search_pairs.txt
    python -m benchmark.mining.govuk_search --topics immigration --report
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

API = "https://www.gov.uk/api/search.json"
UA = "conflict-rag-research/0.1 (final-year project; contact via repository)"
DELAY_S = 0.4

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "logs" / "govuk_search_cache.json"

FIELDS = ("title", "link", "format", "content_id", "content_store_document_type", "public_timestamp")

#: Pages written for the person asking. A rule they state is usually the
#: general one.
CONSUMER_TYPES = frozenset({"guide", "answer", "transaction", "smart_answer", "simple_smart_answer"})

#: Pages that govern the same decision from the other side: the rules, the
#: caseworker guidance, the policy. Where the carve-out usually lives.
RULE_TYPES = frozenset({
    "detailed_guide", "guidance", "statutory_guidance", "manual", "manual_section",
    "policy_paper", "html_publication", "regulation", "statutory_instrument",
    "form", "impact_assessment", "notice", "official_statistics",
})

#: Topic seed lists. Routes are named as gov.uk names them, so the search
#: returns the route's own page rather than a news story about it.
TOPICS: dict[str, list[str]] = {
    "immigration": [
        "Graduate visa", "Student visa", "Skilled Worker visa", "Health and Care Worker visa",
        "Global Talent visa", "Innovator Founder visa", "Scale-up Worker visa",
        "High Potential Individual visa", "Youth Mobility Scheme visa", "Standard Visitor visa",
        "UK family visa partner", "UK family visa child", "UK family visa parent",
        "Seasonal Worker visa", "Temporary Worker Creative Worker visa",
        "Charity Worker visa", "Religious Worker visa", "International Sportsperson visa",
        "Domestic Worker in a Private Household visa", "Graduate Trainee visa",
        "Indefinite leave to remain", "Apply for British citizenship", "Immigration Health Surcharge",
        "Biometric residence permit", "Switch to a different visa", "Sponsor licence",
        "Frontier Worker permit", "Representative of an overseas business visa",
        "Turing scheme graduate", "Ukraine visa scheme",
    ],
}


@dataclass(frozen=True)
class Hit:
    """One distinct gov.uk content item."""

    link: str
    title: str
    content_id: str
    document_type: str
    updated: str | None
    topic: str
    score: float = 0.0

    @property
    def url(self) -> str:
        return f"https://www.gov.uk{self.link}"

    @property
    def role(self) -> str:
        if self.document_type in CONSUMER_TYPES:
            return "consumer"
        if self.document_type in RULE_TYPES:
            return "rule"
        return "other"


@dataclass
class Candidate:
    topic: str
    consumer: Hit
    rule: Hit

    def provider(self) -> str:
        slug = "".join(c if c.isalnum() else "-" for c in self.topic.lower()).strip("-")
        return f"govsearch-{slug}"

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.consumer.link, self.rule.link)))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# HTTP, with an on-disk cache
# --------------------------------------------------------------------------- #

FetchJSON = Callable[[str], dict]


class CachedFetcher:
    """GET JSON with a persistent cache and a polite delay between live calls."""

    def __init__(self, cache_path: Path | None = DEFAULT_CACHE, delay_s: float = DELAY_S):
        self.cache_path = cache_path
        self.delay_s = delay_s
        self._cache: dict[str, dict] = {}
        self.live_calls = 0
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {}  # a corrupt cache is a cache miss, never a crash

    def __call__(self, url: str) -> dict:
        if url in self._cache:
            return self._cache[url]
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        self._cache[url] = data
        self.live_calls += 1
        time.sleep(self.delay_s)
        return data

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Search and pairing
# --------------------------------------------------------------------------- #


def search_url(query: str, *, doc_types: list[str] | None = None, count: int = 20) -> str:
    params: list[tuple[str, str]] = [("q", query), ("count", str(count))]
    params += [("fields", f) for f in FIELDS]
    for dt in doc_types or []:
        params.append(("filter_content_store_document_type", dt))
    return f"{API}?{urllib.parse.urlencode(params)}"


def _to_hit(raw: dict, topic: str) -> Hit | None:
    link, cid = raw.get("link"), raw.get("content_id")
    if not link or not cid or not link.startswith("/"):
        return None  # external or malformed result: not a gov.uk content item we can audit
    return Hit(
        link=link,
        title=raw.get("title", ""),
        content_id=cid,
        document_type=raw.get("content_store_document_type") or raw.get("format") or "",
        updated=(raw.get("public_timestamp") or "")[:10] or None,
        topic=topic,
        score=float(raw.get("es_score") or 0.0),
    )


def harvest_topic(topic: str, fetch: FetchJSON, *, count: int = 20) -> list[Hit]:
    """Distinct content items for one topic, consumer and rule side searched separately.

    Searching the two sides separately matters: a single mixed query is
    dominated by whichever side has more pages on the topic, and the rules
    appendix would never surface behind forty consumer pages.
    """
    hits: dict[str, Hit] = {}
    for types in (sorted(CONSUMER_TYPES), sorted(RULE_TYPES)):
        data = fetch(search_url(topic, doc_types=types, count=count))
        for raw in data.get("results", []):
            hit = _to_hit(raw, topic)
            if hit and hit.content_id not in hits:
                hits[hit.content_id] = hit
    return list(hits.values())


def pair_up(
    hits: list[Hit],
    *,
    exclude: set[tuple[str, str]] | None = None,
    max_per_topic: int = 8,
) -> list[Candidate]:
    """Consumer x rule pairings with different ``content_id``s, best-scored first.

    ``content_id`` distinctness is the entire point (a guide's parts share one),
    so it is enforced here. ``exclude`` removes pairs already mined, so a
    re-harvest only proposes new ones.
    """
    exclude = exclude or set()
    consumers = [h for h in hits if h.role == "consumer"]
    rules = [h for h in hits if h.role == "rule"]
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for c, r in sorted(product(consumers, rules), key=lambda cr: -(cr[0].score + cr[1].score)):
        if c.content_id == r.content_id:
            continue
        cand = Candidate(topic=c.topic, consumer=c, rule=r)
        if cand.key() in exclude or cand.key() in seen:
            continue
        seen.add(cand.key())
        out.append(cand)
        if len(out) >= max_per_topic:
            break
    return out


def existing_pairs(pairs_dir: Path) -> set[tuple[str, str]]:
    """Every (path, path) pairing already mined, from the ``*_pairs*.txt`` files."""
    done: set[tuple[str, str]] = set()
    for f in sorted(pairs_dir.glob("govuk*pairs*.txt")):
        groups: dict[str, list[str]] = {}
        for raw in f.read_text(encoding="utf-8").splitlines():
            bits = raw.split()
            if len(bits) >= 2 and not raw.lstrip().startswith("#"):
                groups.setdefault(bits[0], []).append(urllib.parse.urlparse(bits[1]).path.rstrip("/"))
        for urls in groups.values():
            for a, b in product(urls, urls):
                if a < b:
                    done.add((a, b))
    return done


@dataclass
class HarvestReport:
    topics: int = 0
    items: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    live_calls: int = 0

    def render(self) -> str:
        lines = ["gov.uk Search harvest", "=" * 68,
                 f"  topics searched          {self.topics}",
                 f"  distinct content items   {self.items}",
                 f"  new candidate pairings   {len(self.candidates)}",
                 f"  live API calls           {self.live_calls}", "", "  by document type", "  " + "-" * 38]
        for t, n in sorted(self.by_type.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {t:<30}{n:>6}")
        return "\n".join(lines)


def harvest(
    topics: list[str],
    fetch: FetchJSON,
    *,
    exclude: set[tuple[str, str]] | None = None,
    max_per_topic: int = 8,
    count: int = 20,
) -> HarvestReport:
    rep = HarvestReport(topics=len(topics))
    all_hits: dict[str, Hit] = {}
    for topic in topics:
        hits = harvest_topic(topic, fetch, count=count)
        for h in hits:
            all_hits.setdefault(h.content_id, h)
        rep.candidates += pair_up(hits, exclude=exclude, max_per_topic=max_per_topic)
    rep.items = len(all_hits)
    for h in all_hits.values():
        rep.by_type[h.document_type] = rep.by_type.get(h.document_type, 0) + 1
    # one pairing may be proposed by two topics; keep the first
    uniq: dict[tuple[str, str], Candidate] = {}
    for c in rep.candidates:
        uniq.setdefault(c.key(), c)
    rep.candidates = list(uniq.values())
    return rep


def write_pairs(candidates: list[Candidate], path: Path) -> None:
    """Emit the ``provider url`` format ``benchmark.mining.fetch_govuk`` reads."""
    lines = [
        "# gov.uk Search API harvest. Each provider is one (consumer page, rule page) pairing",
        "# whose content_ids DIFFER, checked at harvest time.",
        "",
    ]
    for i, c in enumerate(candidates):
        prov = f"{c.provider()}-{i:03d}"
        lines.append(f"{prov} {c.consumer.url}")
        lines.append(f"{prov} {c.rule.url}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_provenance(candidates: list[Candidate], path: Path) -> None:
    path.write_text(
        json.dumps([{"topic": c.topic, "consumer": asdict(c.consumer), "rule": asdict(c.rule)}
                    for c in candidates], indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", choices=sorted(TOPICS), default="immigration")
    ap.add_argument("-o", "--output", type=Path, default=None, help="write the pairs file here")
    ap.add_argument("--report", action="store_true", help="print the report and write nothing")
    ap.add_argument("--max-per-topic", type=int, default=8)
    ap.add_argument("--pairs-dir", type=Path, default=REPO_ROOT / "benchmark" / "data")
    args = ap.parse_args(argv)

    fetcher = CachedFetcher()
    exclude = existing_pairs(args.pairs_dir)
    try:
        rep = harvest(TOPICS[args.topics], fetcher, exclude=exclude, max_per_topic=args.max_per_topic)
    finally:
        fetcher.save()
    rep.live_calls = fetcher.live_calls
    print(f"{len(exclude)} pairings already mined are excluded\n")
    print(rep.render())

    if args.output and not args.report:
        write_pairs(rep.candidates, args.output)
        prov = args.output.with_suffix(".provenance.json")
        write_provenance(rep.candidates, prov)
        print(f"\nwrote {len(rep.candidates)} pairings -> {args.output}\nprovenance -> {prov}")
        print(f"next: python -m benchmark.mining.fetch_govuk --pairs {args.output} -o <docs.jsonl>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
