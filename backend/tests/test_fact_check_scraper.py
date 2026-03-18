"""Tests for the fact_check scraper (Vera Files, Tsek.ph)."""

import pytest
from unittest.mock import MagicMock, patch

VERA_FILES_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>VERA Files</title>
    <item>
      <title>Fact Check: Marcos claim on GDP growth is misleading</title>
      <link>https://verafiles.org/articles/fact-check-marcos-gdp</link>
      <pubDate>Wed, 19 Mar 2026 08:00:00 +0800</pubDate>
      <description>President Marcos claimed the Philippines has the highest GDP growth in ASEAN.</description>
    </item>
    <item>
      <title>VERA Files: Duterte drug war death toll revisited</title>
      <link>https://verafiles.org/articles/duterte-drug-war-revisited</link>
      <pubDate>Tue, 18 Mar 2026 09:00:00 +0800</pubDate>
      <description>An investigation into official figures versus reported deaths.</description>
    </item>
  </channel>
</rss>"""

TSEK_PH_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Tsek.ph</title>
    <item>
      <title>Hindi totoo: Walang inaprubahang budget increase para sa CHR</title>
      <link>https://tsek.ph/hindi-totoo-chr-budget</link>
      <pubDate>Mon, 17 Mar 2026 10:00:00 +0800</pubDate>
      <description>Sinuri ng Tsek.ph ang viral na post tungkol sa CHR budget.</description>
    </item>
  </channel>
</rss>"""

EMPTY_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Empty</title></channel></rss>"""


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


class TestScrapeFactChecks:
    def test_vera_files_rss_returns_documents(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response(VERA_FILES_RSS)):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed", max_items=10)
        assert len(docs) == 2
        assert docs[0]["title"] == "Fact Check: Marcos claim on GDP growth is misleading"
        assert docs[0]["url"] == "https://verafiles.org/articles/fact-check-marcos-gdp"

    def test_tsek_ph_rss_returns_documents(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response(TSEK_PH_RSS)):
            docs = scrape_rss_feed("tsek.ph", "https://tsek.ph/feed/", max_items=10)
        assert len(docs) == 1
        assert "tsek.ph" == docs[0]["source"]

    def test_http_error_returns_empty(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=None):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed")
        assert docs == []

    def test_document_has_fact_check_source_type(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response(VERA_FILES_RSS)):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed")
        assert all(d["source_type"] == "fact_check" for d in docs)

    def test_document_has_required_fields(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response(VERA_FILES_RSS)):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed")
        doc = docs[0]
        for field in ("source", "source_type", "date", "politician", "title", "url", "text"):
            assert field in doc, f"Missing field: {field}"

    def test_respects_max_items(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response(VERA_FILES_RSS)):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed", max_items=1)
        assert len(docs) <= 1

    def test_parse_error_returns_empty(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response("not xml")):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed")
        assert docs == []

    def test_scrape_all_fact_checks_aggregates_sources(self):
        from data_ingestion.scrapers.fact_check import scrape_all_fact_checks

        def side_effect(url, *args, **kwargs):
            if "verafiles" in url:
                return _mock_response(VERA_FILES_RSS)
            return _mock_response(TSEK_PH_RSS)

        with patch("data_ingestion.scrapers.fact_check._get", side_effect=side_effect):
            docs = scrape_all_fact_checks(max_items=10)
        assert len(docs) == 3  # 2 vera + 1 tsek

    def test_scrape_all_fact_checks_returns_empty_on_all_errors(self):
        from data_ingestion.scrapers.fact_check import scrape_all_fact_checks
        with patch("data_ingestion.scrapers.fact_check._get", return_value=None):
            docs = scrape_all_fact_checks()
        assert docs == []

    def test_date_is_parsed(self):
        from data_ingestion.scrapers.fact_check import scrape_rss_feed
        with patch("data_ingestion.scrapers.fact_check._get", return_value=_mock_response(VERA_FILES_RSS)):
            docs = scrape_rss_feed("verafiles.org", "https://verafiles.org/feed")
        assert docs[0]["date"] == "2026-03-19"
