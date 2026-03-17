"""
Tests for data_ingestion/scrapers/news_sites.py

HTTP calls are mocked — no real network requests made.
"""

import pytest
from unittest.mock import MagicMock, patch
from data_ingestion.scrapers.news_sites import (
    _parse_rss_date,
    _is_politics_related,
    scrape_rss_feed,
    scrape_all_news,
    RSS_FEEDS,
)

# Both items are politics-related so they pass the keyword filter.
SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Rappler Nation</title>
    <item>
      <title>Senate Approves Education Bill</title>
      <link>https://rappler.com/nation/senate-education-bill</link>
      <pubDate>Mon, 13 Jan 2025 10:00:00 +0800</pubDate>
      <description>&lt;p&gt;The Senate voted 20-3 to approve the bill.&lt;/p&gt;</description>
    </item>
    <item>
      <title>COMELEC Sets Filing Period</title>
      <link>https://rappler.com/nation/comelec-filing</link>
      <pubDate>Tue, 14 Jan 2025 09:00:00 +0800</pubDate>
      <description>Filing period announced.</description>
    </item>
  </channel>
</rss>"""

# Feed with one non-political item that should be filtered out.
MIXED_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Senator Calls for Budget Review</title>
      <link>https://example.com/1</link>
      <pubDate>Mon, 13 Jan 2025 10:00:00 +0800</pubDate>
      <description>Senator proposes new budget.</description>
    </item>
    <item>
      <title>Local Sports Team Wins Championship</title>
      <link>https://example.com/2</link>
      <pubDate>Mon, 13 Jan 2025 11:00:00 +0800</pubDate>
      <description>The team won 3-0.</description>
    </item>
  </channel>
</rss>"""


class TestParseRssDate:
    def test_parses_rss_format(self):
        result = _parse_rss_date("Mon, 13 Jan 2025 10:00:00 +0800")
        assert result == "2025-01-13"

    def test_parses_iso_format(self):
        result = _parse_rss_date("2025-01-13T10:00:00+08:00")
        assert result == "2025-01-13"

    def test_returns_today_for_unrecognized_format(self):
        from datetime import datetime
        result = _parse_rss_date("not-a-date")
        datetime.strptime(result, "%Y-%m-%d")

    def test_handles_utc_timezone_label(self):
        result = _parse_rss_date("Mon, 13 Jan 2025 10:00:00 GMT")
        assert result == "2025-01-13"


class TestIsPoliticsRelated:
    def test_matches_english_institution_keyword(self):
        assert _is_politics_related("Senate passes new law", "") is True

    def test_matches_keyword_in_description(self):
        assert _is_politics_related("Breaking news", "Marcos signs executive order") is True

    def test_matches_comelec(self):
        assert _is_politics_related("COMELEC announces filing period", "") is True

    def test_matches_filipino_keyword(self):
        assert _is_politics_related("Halalan 2025 updates", "") is True

    def test_rejects_unrelated_article(self):
        assert _is_politics_related("Local sports team wins", "Championship game recap") is False

    def test_case_insensitive(self):
        assert _is_politics_related("SENATOR files bill", "") is True

    def test_empty_strings_return_false(self):
        assert _is_politics_related("", "") is False

    def test_matches_high_profile_name(self):
        assert _is_politics_related("Duterte faces new charges", "") is True

    def test_matches_keyword_in_combined_text(self):
        assert _is_politics_related("Unrelated title", "President signs decree") is True


# Shared mock for the seen-URL tracker: all URLs appear fresh (not yet seen).
_FRESH_SEEN = {"seen.return_value": False, "mark.return_value": None}


class TestScrapeRssFeed:
    def _mock_response(self, text):
        resp = MagicMock()
        resp.text = text
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_empty_list_when_request_fails(self):
        with patch("data_ingestion.scrapers.news_sites._get", return_value=None):
            result = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        assert result == []

    def test_returns_documents_from_valid_rss(self):
        mock_resp = self._mock_response(SAMPLE_RSS)
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""), \
             patch("data_ingestion.scrapers.news_sites._seen", **_FRESH_SEEN):
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        assert len(docs) == 2

    def test_document_has_required_metadata_fields(self):
        mock_resp = self._mock_response(SAMPLE_RSS)
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""), \
             patch("data_ingestion.scrapers.news_sites._seen", **_FRESH_SEEN):
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        doc = docs[0]
        assert doc["source"] == "rappler.com"
        assert doc["source_type"] == "news"
        assert doc["title"] == "Senate Approves Education Bill"
        assert doc["url"] == "https://rappler.com/nation/senate-education-bill"
        assert doc["date"] == "2025-01-13"

    def test_respects_max_items_limit(self):
        mock_resp = self._mock_response(SAMPLE_RSS)
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""), \
             patch("data_ingestion.scrapers.news_sites._seen", **_FRESH_SEEN):
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/", max_items=1)
        assert len(docs) == 1

    def test_full_article_text_replaces_description_when_available(self):
        mock_resp = self._mock_response(SAMPLE_RSS)
        full_text = "Full article body with much more detail about the education bill."
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=full_text), \
             patch("data_ingestion.scrapers.news_sites._seen", **_FRESH_SEEN):
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        assert docs[0]["text"] == full_text

    def test_description_used_when_article_fetch_returns_empty(self):
        mock_resp = self._mock_response(SAMPLE_RSS)
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""), \
             patch("data_ingestion.scrapers.news_sites._seen", **_FRESH_SEEN):
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        assert "Senate voted" in docs[0]["text"]

    def test_returns_empty_list_on_malformed_xml(self):
        mock_resp = self._mock_response("<<<not xml>>>")
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""):
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        assert docs == []

    def test_filters_out_non_political_articles(self):
        mock_resp = self._mock_response(MIXED_RSS)
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""), \
             patch("data_ingestion.scrapers.news_sites._seen", **_FRESH_SEEN):
            docs = scrape_rss_feed("example.com", "https://example.com/feed/")
        assert len(docs) == 1
        assert "Senator" in docs[0]["title"]

    def test_skips_already_seen_url(self):
        mock_resp = self._mock_response(SAMPLE_RSS)
        with patch("data_ingestion.scrapers.news_sites._get", return_value=mock_resp), \
             patch("data_ingestion.scrapers.news_sites._fetch_article_text", return_value=""), \
             patch("data_ingestion.scrapers.news_sites._seen") as mock_seen:
            mock_seen.seen.return_value = True  # every URL already seen
            docs = scrape_rss_feed("rappler.com", "https://rappler.com/feed/")
        assert docs == []


class TestScrapeAllNews:
    def test_aggregates_results_from_all_feeds(self):
        mock_docs = [
            {"source": "rappler.com", "title": "Article 1", "text": "T1",
             "url": "https://rappler.com/1", "date": "2025-01-01", "source_type": "news",
             "politician": ""}
        ]
        with patch("data_ingestion.scrapers.news_sites.scrape_rss_feed", return_value=mock_docs) as mock_fn:
            docs = scrape_all_news(max_items_per_source=5)

        assert mock_fn.call_count == len(RSS_FEEDS)
        assert len(docs) == len(RSS_FEEDS)  # 1 doc per feed

    def test_returns_empty_list_when_all_feeds_fail(self):
        with patch("data_ingestion.scrapers.news_sites.scrape_rss_feed", return_value=[]):
            docs = scrape_all_news()
        assert docs == []

    def test_passes_max_items_to_each_feed(self):
        with patch("data_ingestion.scrapers.news_sites.scrape_rss_feed", return_value=[]) as mock_fn:
            scrape_all_news(max_items_per_source=7)
        for call in mock_fn.call_args_list:
            assert call.kwargs.get("max_items") == 7 or call.args[2] == 7

    def test_rss_feeds_has_all_five_sources(self):
        expected = {"rappler.com", "philstar.com",
                    "bworldonline.com", "gmanetwork.com", "pcij.org"}
        assert set(RSS_FEEDS.keys()) == expected

    def test_all_feed_urls_are_https(self):
        for name, url in RSS_FEEDS.items():
            assert url.startswith("https://"), f"{name} feed URL should use HTTPS"
