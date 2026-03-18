"""Tests for the research scraper (PIDS, ADB, UNDP, IMF, TI, UP CIDS)."""

import pytest
from unittest.mock import MagicMock, patch

PIDS_HTML = """
<html><body>
  <div class="publication-list">
    <div class="publication-item">
      <h3><a href="https://pids.gov.ph/publication/dp2026-01">
        Discussion Paper: Fiscal Sustainability in the Philippines</a></h3>
      <span class="pub-date">March 2026</span>
      <p class="abstract">This paper examines the long-term fiscal trajectory.</p>
    </div>
    <div class="publication-item">
      <h3><a href="https://pids.gov.ph/publication/rps2026-02">
        RPS: Impact of CCT on Poverty Reduction</a></h3>
      <span class="pub-date">February 2026</span>
      <p class="abstract">Evaluates the 4Ps conditional cash transfer program.</p>
    </div>
  </div>
</body></html>"""

ADB_HTML = """
<html><body>
  <ul class="publications-list">
    <li class="pub-item">
      <a href="https://www.adb.org/publications/ph-country-diagnostics-2026">
        Philippines: Country Diagnostics 2026</a>
      <time datetime="2026-02-15">15 Feb 2026</time>
    </li>
    <li class="pub-item">
      <a href="https://www.adb.org/publications/ph-transport-sector-review">
        Philippines Transport Sector Review</a>
      <time datetime="2026-01-20">20 Jan 2026</time>
    </li>
  </ul>
</body></html>"""

EMPTY_HTML = "<html><body></body></html>"


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


class TestResearchScraper:
    def test_pids_publications_parsed(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(PIDS_HTML)):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications", max_items=10)
        assert len(docs) >= 1
        assert any("Fiscal" in d["title"] or "CCT" in d["title"] or "Poverty" in d["title"] for d in docs)

    def test_adb_publications_parsed(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(ADB_HTML)):
            docs = _scrape_publications("adb.org", "https://www.adb.org/countries/philippines/publications", max_items=10)
        assert len(docs) >= 1
        assert any("Philippines" in d["title"] for d in docs)

    def test_http_error_returns_empty(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=None):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications")
        assert docs == []

    def test_source_type_is_research(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(PIDS_HTML)):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications")
        assert all(d["source_type"] == "research" for d in docs)

    def test_document_has_required_fields(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(PIDS_HTML)):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications")
        doc = docs[0]
        for field in ("source", "source_type", "date", "politician", "title", "url", "text"):
            assert field in doc

    def test_respects_max_items(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(PIDS_HTML)):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications", max_items=1)
        assert len(docs) <= 1

    def test_empty_page_returns_empty_list(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(EMPTY_HTML)):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications")
        assert docs == []

    def test_scrape_all_research_aggregates_sources(self):
        from data_ingestion.scrapers.research import scrape_all_research

        def side_effect(url, *args, **kwargs):
            if "pids" in url:
                return _mock_response(PIDS_HTML)
            if "adb" in url:
                return _mock_response(ADB_HTML)
            return _mock_response(EMPTY_HTML)

        with patch("data_ingestion.scrapers.research._get", side_effect=side_effect):
            docs = scrape_all_research(max_items=10)
        assert len(docs) >= 2  # at least PIDS + ADB

    def test_scrape_all_research_returns_empty_on_all_errors(self):
        from data_ingestion.scrapers.research import scrape_all_research
        with patch("data_ingestion.scrapers.research._get", return_value=None):
            docs = scrape_all_research()
        assert docs == []

    def test_url_is_absolute(self):
        from data_ingestion.scrapers.research import _scrape_publications
        with patch("data_ingestion.scrapers.research._get", return_value=_mock_response(PIDS_HTML)):
            docs = _scrape_publications("pids.gov.ph", "https://pids.gov.ph/publications")
        assert all(d["url"].startswith("http") for d in docs)
