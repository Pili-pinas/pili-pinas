"""Tests for the oversight scraper (COA, Ombudsman, Sandiganbayan, CSC)."""

import pytest
from unittest.mock import MagicMock, patch

# Minimal HTML mimicking a press-release listing page.
COA_PRESS_RELEASES_HTML = """
<html><body>
  <div class="itemListView">
    <article>
      <h2><a href="https://www.coa.gov.ph/press-releases/coa-report-2025">
        COA Flags P12B Irregularities in DPWH Projects</a></h2>
      <span class="date">March 18, 2026</span>
    </article>
    <article>
      <h2><a href="https://www.coa.gov.ph/press-releases/lgu-audit-2025">
        COA: 34 LGUs Failed to Submit Financial Reports</a></h2>
      <span class="date">March 15, 2026</span>
    </article>
  </div>
</body></html>"""

OMBUDSMAN_HTML = """
<html><body>
  <ul class="news-list">
    <li>
      <a href="https://www.ombudsman.gov.ph/news/case-filed-2026">
        Ombudsman Files Charges Against Ex-DILG Official for Graft</a>
      <span class="news-date">March 17, 2026</span>
    </li>
    <li>
      <a href="https://www.ombudsman.gov.ph/news/dismissed-2026">
        3 DPWH Engineers Dismissed for Neglect of Duty</a>
      <span class="news-date">March 10, 2026</span>
    </li>
  </ul>
</body></html>"""

EMPTY_HTML = "<html><body><p>No records found.</p></body></html>"


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


class TestScrapeOversight:
    def test_coa_parses_press_releases(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(COA_PRESS_RELEASES_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/index.php/press-releases", max_items=10)
        assert len(docs) == 2
        assert "COA Flags" in docs[0]["title"]

    def test_ombudsman_parses_announcements(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(OMBUDSMAN_HTML)):
            docs = _scrape_press_releases("ombudsman.gov.ph", "https://www.ombudsman.gov.ph/news", max_items=10)
        assert len(docs) == 2
        assert "Ombudsman" in docs[0]["title"]

    def test_http_error_returns_empty(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=None):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases", max_items=10)
        assert docs == []

    def test_source_type_is_oversight(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(COA_PRESS_RELEASES_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases")
        assert all(d["source_type"] == "oversight" for d in docs)

    def test_document_schema_has_required_fields(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(COA_PRESS_RELEASES_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases")
        doc = docs[0]
        for field in ("source", "source_type", "date", "politician", "title", "url", "text"):
            assert field in doc, f"Missing field: {field}"

    def test_source_field_matches_domain(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(COA_PRESS_RELEASES_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases")
        assert all(d["source"] == "coa.gov.ph" for d in docs)

    def test_empty_page_returns_empty_list(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(EMPTY_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases")
        assert docs == []

    def test_respects_max_items(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(COA_PRESS_RELEASES_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases", max_items=1)
        assert len(docs) <= 1

    def test_scrape_all_oversight_aggregates_all_sources(self):
        from data_ingestion.scrapers.oversight import scrape_all_oversight

        def side_effect(url, *args, **kwargs):
            if "coa" in url:
                return _mock_response(COA_PRESS_RELEASES_HTML)
            if "ombudsman" in url:
                return _mock_response(OMBUDSMAN_HTML)
            return _mock_response(EMPTY_HTML)

        with patch("data_ingestion.scrapers.oversight._get", side_effect=side_effect):
            docs = scrape_all_oversight(max_items=10)
        # At least COA (2) + Ombudsman (2) = 4
        assert len(docs) >= 4

    def test_scrape_all_oversight_returns_empty_on_all_errors(self):
        from data_ingestion.scrapers.oversight import scrape_all_oversight
        with patch("data_ingestion.scrapers.oversight._get", return_value=None):
            docs = scrape_all_oversight()
        assert docs == []

    def test_url_extracted_from_link(self):
        from data_ingestion.scrapers.oversight import _scrape_press_releases
        with patch("data_ingestion.scrapers.oversight._get", return_value=_mock_response(COA_PRESS_RELEASES_HTML)):
            docs = _scrape_press_releases("coa.gov.ph", "https://www.coa.gov.ph/press-releases")
        assert docs[0]["url"].startswith("http")
