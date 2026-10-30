"""The gov.uk Search harvester, offline: a fake fetcher stands in for the API."""

from __future__ import annotations

import json
import urllib.parse

import pytest

from benchmark.mining.govuk_search import (
    CachedFetcher,
    Candidate,
    Hit,
    existing_pairs,
    harvest,
    harvest_topic,
    pair_up,
    search_url,
    write_pairs,
)


def _raw(link, cid, dtype, title="t", score=1.0, ts="2024-05-01T10:00:00Z"):
    return {"link": link, "content_id": cid, "content_store_document_type": dtype,
            "format": dtype, "title": title, "es_score": score, "public_timestamp": ts}


def fake_api(consumers, rules):
    """Answer by which side of the search the URL asks for."""

    def fetch(url: str) -> dict:
        types = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(
            "filter_content_store_document_type", [])
        return {"results": consumers if "guide" in types and "detailed_guide" not in types else rules}

    return fetch


def test_search_url_requests_the_provenance_fields_and_filters():
    url = search_url("Graduate visa", doc_types=["guide", "answer"], count=7)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["q"] == ["Graduate visa"] and q["count"] == ["7"]
    assert {"content_id", "public_timestamp", "content_store_document_type"} <= set(q["fields"])
    assert q["filter_content_store_document_type"] == ["guide", "answer"]


def test_harvest_topic_searches_both_sides_and_keeps_distinct_content_ids():
    consumers = [_raw("/graduate-visa", "c1", "guide"), _raw("/graduate-visa/stay", "c1", "guide")]
    rules = [_raw("/guidance/rules", "r1", "manual_section")]
    seen_urls = []

    def fetch(url):
        seen_urls.append(url)
        return {"results": consumers} if "filter_content_store_document_type=answer" in url else {"results": rules}

    hits = harvest_topic("Graduate visa", fetch)
    assert len(seen_urls) == 2, "consumer and rule sides are searched separately"
    assert [h.content_id for h in hits].count("c1") <= 1


def test_results_without_a_content_id_or_with_external_links_are_dropped():
    rules = [_raw("/ok", "r1", "guidance"),
             {"link": "/no-id", "content_store_document_type": "guidance", "title": "x"},
             _raw("https://example.com/x", "r2", "guidance")]
    hits = harvest_topic("t", lambda url: {"results": rules})
    assert {h.link for h in hits} == {"/ok"}


def _hit(link, cid, dtype, score=1.0):
    return Hit(link=link, title=link, content_id=cid, document_type=dtype, updated="2024-01-01",
               topic="Graduate visa", score=score)


def test_roles_come_from_document_type():
    assert _hit("/a", "1", "guide").role == "consumer"
    assert _hit("/a", "1", "manual_section").role == "rule"
    assert _hit("/a", "1", "news_story").role == "other"


def test_a_pair_never_joins_two_parts_of_one_content_item():
    """The whole point: a guide's parts share a content_id, and pairing them
    would call one document two."""
    hits = [_hit("/g", "same", "guide"), _hit("/rules", "same", "guidance"),
            _hit("/other", "different", "guidance")]
    cands = pair_up(hits)
    assert [(c.consumer.link, c.rule.link) for c in cands] == [("/g", "/other")]


def test_pairs_already_mined_are_excluded_and_results_are_best_first():
    hits = [_hit("/a", "1", "guide", 2.0), _hit("/b", "2", "guide", 0.1),
            _hit("/r1", "3", "guidance", 2.0), _hit("/r2", "4", "guidance", 0.1)]
    cands = pair_up(hits, exclude={("/a", "/r1")})
    assert ("/a", "/r1") not in {c.key() for c in cands}
    assert cands[0].key() == tuple(sorted(("/a", "/r2"))) or cands[0].key() == tuple(sorted(("/b", "/r1")))


def test_max_per_topic_caps_the_output():
    hits = [_hit(f"/c{i}", f"c{i}", "guide") for i in range(5)] + \
           [_hit(f"/r{i}", f"r{i}", "guidance") for i in range(5)]
    assert len(pair_up(hits, max_per_topic=3)) == 3


def test_harvest_report_counts_items_and_dedupes_pairings_across_topics():
    api = fake_api([_raw("/g", "c1", "guide")], [_raw("/r", "r1", "guidance")])
    rep = harvest(["Topic A", "Topic B"], api)
    assert rep.items == 2 and len(rep.candidates) == 1, "the same pairing from two topics counts once"
    assert rep.by_type == {"guide": 1, "guidance": 1}


def test_write_pairs_emits_the_format_the_fetcher_reads(tmp_path):
    from benchmark.mining.fetch_govuk import parse_pairs

    cand = Candidate("Graduate visa", _hit("/graduate-visa", "c1", "guide"), _hit("/rules/x", "r1", "guidance"))
    out = tmp_path / "pairs.txt"
    write_pairs([cand], out)
    parsed = parse_pairs(out)
    assert len(parsed) == 1
    (urls,) = parsed.values()
    assert urls == ["https://www.gov.uk/graduate-visa", "https://www.gov.uk/rules/x"]


def test_existing_pairs_reads_every_mined_pairing(tmp_path):
    (tmp_path / "govuk_pairs_9.txt").write_text(
        "# c\nprov https://www.gov.uk/b\nprov https://www.gov.uk/a/\nother https://www.gov.uk/x\n",
        encoding="utf-8")
    assert existing_pairs(tmp_path) == {("/a", "/b")}


def test_cached_fetcher_hits_the_network_once_and_survives_a_corrupt_cache(tmp_path, monkeypatch):
    cache = tmp_path / "c.json"
    cache.write_text("{not json", encoding="utf-8")
    calls = []

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return b'{"results": []}'

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: calls.append(req) or FakeResp())
    f = CachedFetcher(cache, delay_s=0.0)
    f("https://x/1"); f("https://x/1")
    assert len(calls) == 1 and f.live_calls == 1
    f.save()
    assert json.loads(cache.read_text(encoding="utf-8")) == {"https://x/1": {"results": []}}


def test_the_real_topic_list_is_nonempty_and_uses_route_names():
    from benchmark.mining.govuk_search import TOPICS

    assert len(TOPICS["immigration"]) >= 25
    assert all(isinstance(t, str) and t for t in TOPICS["immigration"])
