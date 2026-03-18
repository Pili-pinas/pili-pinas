"""Tests for the financial scraper (PhilGEPS procurement, SOCE campaign finance)."""

import pytest
from unittest.mock import MagicMock, patch

PHILGEPS_HTML = """
<html><body>
  <table id="bodyContent">
    <tr class="odd">
      <td><a href="https://notices.philgeps.gov.ph/notice/12345">
        Supply and Delivery of Medical Equipment for DOH</a></td>
      <td>Department of Health</td>
      <td>March 18, 2026</td>
      <td>PHP 50,000,000.00</td>
    </tr>
    <tr class="even">
      <td><a href="https://notices.philgeps.gov.ph/notice/12346">
        Construction of Farm-to-Market Road in Nueva Ecija</a></td>
      <td>Department of Agriculture</td>
      <td>March 15, 2026</td>
      <td>PHP 12,500,000.00</td>
    </tr>
  </table>
</body></html>"""

SOCE_HTML = """
<html><body>
  <div class="soce-list">
    <ul>
      <li>
        <a href="https://comelec.gov.ph/soce/marcos-bongbong-2022.pdf">
          SOCE: Marcos, Ferdinand Jr. — 2022 Presidential Election</a>
        <span class="date">Filed: July 12, 2022</span>
      </li>
      <li>
        <a href="https://comelec.gov.ph/soce/robredo-leni-2022.pdf">
          SOCE: Robredo, Leni — 2022 Presidential Election</a>
        <span class="date">Filed: July 10, 2022</span>
      </li>
    </ul>
  </div>
</body></html>"""

EMPTY_HTML = "<html><body></body></html>"


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


class TestFinancialScraper:
    def test_philgeps_notices_parsed(self):
        from data_ingestion.scrapers.financial import scrape_philgeps
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(PHILGEPS_HTML)):
            docs = scrape_philgeps(max_items=10)
        assert len(docs) >= 1
        assert any("Medical" in d["title"] or "Road" in d["title"] for d in docs)

    def test_philgeps_source_type_is_procurement(self):
        from data_ingestion.scrapers.financial import scrape_philgeps
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(PHILGEPS_HTML)):
            docs = scrape_philgeps(max_items=10)
        assert all(d["source_type"] == "procurement" for d in docs)

    def test_philgeps_http_error_returns_empty(self):
        from data_ingestion.scrapers.financial import scrape_philgeps
        with patch("data_ingestion.scrapers.financial._get", return_value=None):
            docs = scrape_philgeps()
        assert docs == []

    def test_philgeps_empty_page_returns_empty(self):
        from data_ingestion.scrapers.financial import scrape_philgeps
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(EMPTY_HTML)):
            docs = scrape_philgeps()
        assert docs == []

    def test_soce_links_parsed(self):
        from data_ingestion.scrapers.financial import scrape_soce
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(SOCE_HTML)):
            docs = scrape_soce(max_items=10)
        assert len(docs) >= 1
        assert any("Marcos" in d["title"] or "SOCE" in d["title"] for d in docs)

    def test_soce_source_type_is_campaign_finance(self):
        from data_ingestion.scrapers.financial import scrape_soce
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(SOCE_HTML)):
            docs = scrape_soce(max_items=10)
        assert all(d["source_type"] == "campaign_finance" for d in docs)

    def test_soce_http_error_returns_empty(self):
        from data_ingestion.scrapers.financial import scrape_soce
        with patch("data_ingestion.scrapers.financial._get", return_value=None):
            docs = scrape_soce()
        assert docs == []

    def test_document_has_required_fields(self):
        from data_ingestion.scrapers.financial import scrape_philgeps
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(PHILGEPS_HTML)):
            docs = scrape_philgeps(max_items=10)
        if docs:
            doc = docs[0]
            for field in ("source", "source_type", "date", "politician", "title", "url", "text"):
                assert field in doc

    def test_scrape_all_financial_aggregates(self):
        from data_ingestion.scrapers.financial import scrape_all_financial

        def side_effect(url, *args, **kwargs):
            if "philgeps" in url:
                return _mock_response(PHILGEPS_HTML)
            if "comelec" in url:
                return _mock_response(SOCE_HTML)
            return _mock_response(EMPTY_HTML)

        with patch("data_ingestion.scrapers.financial._get", side_effect=side_effect):
            docs = scrape_all_financial(max_items=10)
        assert len(docs) >= 2

    def test_scrape_all_financial_returns_empty_on_all_errors(self):
        from data_ingestion.scrapers.financial import scrape_all_financial
        with patch("data_ingestion.scrapers.financial._get", return_value=None):
            docs = scrape_all_financial()
        assert docs == []

    def test_philgeps_respects_max_items(self):
        from data_ingestion.scrapers.financial import scrape_philgeps
        with patch("data_ingestion.scrapers.financial._get", return_value=_mock_response(PHILGEPS_HTML)):
            docs = scrape_philgeps(max_items=1)
        assert len(docs) <= 1
