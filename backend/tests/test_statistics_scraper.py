"""Tests for the statistics scraper (PSA, BSP, DBM, NEDA, World Bank API)."""

import pytest
from unittest.mock import MagicMock, patch

PSA_HTML = """
<html><body>
  <div class="view-content">
    <div class="views-row">
      <span class="field-content">
        <a href="https://psa.gov.ph/content/gdp-grows-56-2025">
          GDP Grows 5.6% in Q4 2025</a>
      </span>
      <span class="date-display-single">March 12, 2026</span>
    </div>
    <div class="views-row">
      <span class="field-content">
        <a href="https://psa.gov.ph/content/inflation-2026">
          Inflation Rate Eases to 3.2% in February 2026</a>
      </span>
      <span class="date-display-single">March 5, 2026</span>
    </div>
  </div>
</body></html>"""

BSP_HTML = """
<html><body>
  <table class="table">
    <tr>
      <td><a href="https://www.bsp.gov.ph/releases/2026/pr20260318">
        BSP Keeps Policy Rate at 5.75% in March 2026</a></td>
      <td>18 March 2026</td>
    </tr>
    <tr>
      <td><a href="https://www.bsp.gov.ph/releases/2026/pr20260301">
        Gross International Reserves Rise to $107 Billion</a></td>
      <td>01 March 2026</td>
    </tr>
  </table>
</body></html>"""

# Minimal World Bank API response for a single indicator
WB_API_RESPONSE = [
    {"page": 1, "pages": 1, "per_page": 5, "total": 3},
    [
        {
            "indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
            "country": {"id": "PH", "value": "Philippines"},
            "date": "2024",
            "value": 5.6,
        },
        {
            "indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
            "country": {"id": "PH", "value": "Philippines"},
            "date": "2023",
            "value": 5.5,
        },
    ],
]

EMPTY_HTML = "<html><body></body></html>"


def _mock_response(data, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if isinstance(data, (list, dict)):
        resp.json = MagicMock(return_value=data)
        resp.text = ""
    else:
        resp.text = data
        resp.json = MagicMock(side_effect=ValueError("not JSON"))
    return resp


class TestStatisticsScraper:
    def test_psa_press_releases_parsed(self):
        from data_ingestion.scrapers.statistics import _scrape_press_releases
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(PSA_HTML)):
            docs = _scrape_press_releases("psa.gov.ph", "https://psa.gov.ph/statistics/press-releases")
        assert len(docs) >= 1
        assert any("GDP" in d["title"] for d in docs)

    def test_bsp_press_releases_parsed(self):
        from data_ingestion.scrapers.statistics import _scrape_press_releases
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(BSP_HTML)):
            docs = _scrape_press_releases("bsp.gov.ph", "https://www.bsp.gov.ph/releases")
        assert len(docs) >= 1
        assert any("BSP" in d["title"] for d in docs)

    def test_http_error_returns_empty(self):
        from data_ingestion.scrapers.statistics import _scrape_press_releases
        with patch("data_ingestion.scrapers.statistics._get", return_value=None):
            docs = _scrape_press_releases("psa.gov.ph", "https://psa.gov.ph/statistics/press-releases")
        assert docs == []

    def test_source_type_is_statistics(self):
        from data_ingestion.scrapers.statistics import _scrape_press_releases
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(PSA_HTML)):
            docs = _scrape_press_releases("psa.gov.ph", "https://psa.gov.ph/press-releases")
        assert all(d["source_type"] == "statistics" for d in docs)

    def test_document_has_required_fields(self):
        from data_ingestion.scrapers.statistics import _scrape_press_releases
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(PSA_HTML)):
            docs = _scrape_press_releases("psa.gov.ph", "https://psa.gov.ph/press-releases")
        doc = docs[0]
        for field in ("source", "source_type", "date", "politician", "title", "url", "text"):
            assert field in doc

    def test_world_bank_api_returns_indicator_docs(self):
        from data_ingestion.scrapers.statistics import scrape_world_bank
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(WB_API_RESPONSE)):
            docs = scrape_world_bank()
        assert len(docs) >= 1

    def test_world_bank_text_includes_indicator_name_and_value(self):
        from data_ingestion.scrapers.statistics import scrape_world_bank
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(WB_API_RESPONSE)):
            docs = scrape_world_bank()
        text = docs[0]["text"]
        assert "GDP" in text or "5.6" in text

    def test_world_bank_source_type_is_statistics(self):
        from data_ingestion.scrapers.statistics import scrape_world_bank
        with patch("data_ingestion.scrapers.statistics._get", return_value=_mock_response(WB_API_RESPONSE)):
            docs = scrape_world_bank()
        assert all(d["source_type"] == "statistics" for d in docs)

    def test_world_bank_http_error_returns_empty(self):
        from data_ingestion.scrapers.statistics import scrape_world_bank
        with patch("data_ingestion.scrapers.statistics._get", return_value=None):
            docs = scrape_world_bank()
        assert docs == []

    def test_scrape_all_statistics_aggregates_sources(self):
        from data_ingestion.scrapers.statistics import scrape_all_statistics

        def side_effect(url, *args, **kwargs):
            if "worldbank" in url:
                return _mock_response(WB_API_RESPONSE)
            return _mock_response(PSA_HTML)

        with patch("data_ingestion.scrapers.statistics._get", side_effect=side_effect):
            docs = scrape_all_statistics(max_items=5)
        assert len(docs) >= 1

    def test_scrape_all_statistics_returns_empty_on_all_errors(self):
        from data_ingestion.scrapers.statistics import scrape_all_statistics
        with patch("data_ingestion.scrapers.statistics._get", return_value=None):
            docs = scrape_all_statistics()
        assert docs == []
