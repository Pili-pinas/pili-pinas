"""
Tests for data_ingestion/scrapers/official_gazette.py

HTTP calls are mocked — no real network requests made.

The scraper now uses the Supreme Court E-Library
(elibrary.judiciary.gov.ph) since officialgazette.gov.ph is
Cloudflare-blocked. It:
  1. GETs the RA index page to extract a CSRF token
  2. POSTs to /republic_acts/fetch_ra with DataTable pagination params
  3. Optionally GETs each law's detail page for full text
"""

import pytest
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup

from data_ingestion.scrapers.official_gazette import (
    scrape_laws,
    _parse_law_row,
    _fetch_index,
    _fetch_ra_page,
    _fetch_law_text,
)


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

# Minimal index HTML — contains CSRF token in the DataTable ajax config
INDEX_HTML_WITH_CSRF = """
<html><body>
<script>
jQuery(function($){
  $('#ra').DataTable({
    "ajax": {
      url: "https://elibrary.judiciary.gov.ph/republic_acts/fetch_ra",
      type: "POST",
      data: { 'csrf_test_name': 'abc123deadbeef' },
    },
  });
});
</script>
</body></html>
"""

INDEX_HTML_NO_CSRF = "<html><body><p>No token here.</p></body></html>"

# DataTable JSON response rows
ROW_1 = [
    "REPUBLIC ACT NO. 12312",
    "2025-10-23",
    "<a href='https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/2/100174'>"
    "AN ACT BANNING AND DECLARING ILLEGAL OFFSHORE GAMING OPERATIONS</a>",
]
ROW_2 = [
    "REPUBLIC ACT NO. 12313",
    "2025-10-23",
    "<a href='https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/2/100182'>"
    "AN ACT INSTITUTIONALIZING THE LIFELONG LEARNING FRAMEWORK</a>",
]
ROW_3 = [
    "REPUBLIC ACT NO. 12310",
    "2025-10-03",
    "<a href='https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/2/99967'>"
    "AN ACT EXPANDING THE PHILIPPINE SCIENCE HIGH SCHOOL</a>",
]

API_PAGE_1 = {
    "draw": 1,
    "recordsTotal": 3,
    "recordsFiltered": 3,
    "data": [ROW_1, ROW_2],
}

API_PAGE_2 = {
    "draw": 2,
    "recordsTotal": 3,
    "recordsFiltered": 3,
    "data": [ROW_3],
}

# Detail page HTML — law text is in div.single_content
DETAIL_HTML = """
<html><body>
<div class="single_content">
  <center><h2>[ REPUBLIC ACT NO. 12312, October 23, 2025 ]</h2>
  <h3>AN ACT BANNING AND DECLARING ILLEGAL OFFSHORE GAMING OPERATIONS</h3>
  </center>
  <div align="justify">
    <i>Be it enacted by the Senate and House of Representatives...</i>
  </div>
  <div>SECTION 1. Short Title. — This Act shall be known as the "POGO Ban Act".</div>
</div>
</body></html>
"""

DETAIL_HTML_NO_CONTENT = "<html><body><p>Not found.</p></body></html>"


def _mock_resp(data, status=200):
    resp = MagicMock()
    resp.status_code = status
    if isinstance(data, dict):
        resp.json.return_value = data
        resp.text = ""
    else:
        resp.text = data
        resp.content = data.encode() if isinstance(data, str) else data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _parse_law_row
# ---------------------------------------------------------------------------

class TestParseLawRow:
    def test_extracts_short_title(self):
        doc = _parse_law_row(ROW_1)
        assert "REPUBLIC ACT NO. 12312" in doc["title"]

    def test_extracts_date(self):
        doc = _parse_law_row(ROW_1)
        assert doc["date"] == "2025-10-23"

    def test_extracts_url_from_link(self):
        doc = _parse_law_row(ROW_1)
        assert "showdocs/2/100174" in doc["url"]

    def test_extracts_long_title_as_text(self):
        doc = _parse_law_row(ROW_1)
        assert "OFFSHORE GAMING" in doc["text"]

    def test_source_is_elibrary(self):
        doc = _parse_law_row(ROW_1)
        assert doc["source"] == "elibrary.judiciary.gov.ph"

    def test_source_type_is_law(self):
        doc = _parse_law_row(ROW_1)
        assert doc["source_type"] == "law"

    def test_politician_is_empty_string(self):
        doc = _parse_law_row(ROW_1)
        assert doc["politician"] == ""

    def test_missing_link_yields_empty_url(self):
        row = ["REPUBLIC ACT NO. 99999", "2025-01-01", "No link here"]
        doc = _parse_law_row(row)
        assert doc["url"] == ""

    def test_title_includes_short_title(self):
        doc = _parse_law_row(ROW_1)
        assert "REPUBLIC ACT NO. 12312" in doc["title"]


# ---------------------------------------------------------------------------
# _fetch_index
# ---------------------------------------------------------------------------

class TestFetchIndex:
    def test_returns_session_and_token_on_success(self):
        mock_session = MagicMock()
        mock_session.get.return_value = _mock_resp(INDEX_HTML_WITH_CSRF)
        token = _fetch_index(mock_session)
        assert token == "abc123deadbeef"

    def test_returns_none_when_request_fails(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Connection error")
        token = _fetch_index(mock_session)
        assert token is None

    def test_returns_none_when_no_csrf_in_html(self):
        mock_session = MagicMock()
        mock_session.get.return_value = _mock_resp(INDEX_HTML_NO_CSRF)
        token = _fetch_index(mock_session)
        assert token is None


# ---------------------------------------------------------------------------
# _fetch_ra_page
# ---------------------------------------------------------------------------

class TestFetchRaPage:
    def test_returns_data_on_success(self):
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_resp(API_PAGE_1)
        result = _fetch_ra_page(mock_session, "abc123", start=0, length=10)
        assert result["recordsTotal"] == 3
        assert len(result["data"]) == 2

    def test_returns_none_when_request_fails(self):
        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("timeout")
        result = _fetch_ra_page(mock_session, "abc123", start=0, length=10)
        assert result is None

    def test_sends_csrf_in_post_data(self):
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_resp(API_PAGE_1)
        _fetch_ra_page(mock_session, "mytoken", start=0, length=5)
        call_kwargs = mock_session.post.call_args
        post_data = call_kwargs[1].get("data") or call_kwargs[0][1]
        assert post_data["csrf_test_name"] == "mytoken"

    def test_sends_correct_start_offset(self):
        mock_session = MagicMock()
        mock_session.post.return_value = _mock_resp(API_PAGE_1)
        _fetch_ra_page(mock_session, "tok", start=20, length=10)
        call_kwargs = mock_session.post.call_args
        post_data = call_kwargs[1].get("data") or call_kwargs[0][1]
        assert int(post_data["start"]) == 20


# ---------------------------------------------------------------------------
# _fetch_law_text
# ---------------------------------------------------------------------------

class TestFetchLawText:
    def test_returns_text_from_single_content_div(self):
        mock_session = MagicMock()
        mock_session.get.return_value = _mock_resp(DETAIL_HTML)
        text = _fetch_law_text(mock_session, "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/2/100174")
        assert "Be it enacted" in text
        assert "POGO Ban Act" in text

    def test_returns_empty_string_when_request_fails(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("timeout")
        text = _fetch_law_text(mock_session, "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/2/1")
        assert text == ""

    def test_returns_empty_string_when_no_content_div(self):
        mock_session = MagicMock()
        mock_session.get.return_value = _mock_resp(DETAIL_HTML_NO_CONTENT)
        text = _fetch_law_text(mock_session, "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/2/1")
        assert text == ""


# ---------------------------------------------------------------------------
# scrape_laws
# ---------------------------------------------------------------------------

class TestScrapeLaws:
    def test_returns_empty_when_index_fails(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value=None):
            docs = scrape_laws(max_items=10)
        assert docs == []

    def test_returns_documents_from_api(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page",
                   side_effect=[API_PAGE_1, API_PAGE_2]), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=3)
        assert len(docs) == 3

    def test_respects_max_items(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value=API_PAGE_1), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=1)
        assert len(docs) == 1

    def test_stops_when_all_records_fetched(self):
        single_page = {**API_PAGE_1, "recordsTotal": 2, "data": [ROW_1, ROW_2]}
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok") as _, \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page",
                   return_value=single_page) as mock_post, \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=100)
        assert mock_post.call_count == 1

    def test_documents_have_required_fields(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value={
                 **API_PAGE_1, "recordsTotal": 2, "data": [ROW_1, ROW_2]
             }), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=5)
        doc = docs[0]
        assert doc["source"] == "elibrary.judiciary.gov.ph"
        assert doc["source_type"] == "law"
        assert "date" in doc
        assert "politician" in doc
        assert "title" in doc
        assert "url" in doc
        assert "text" in doc

    def test_uses_detail_text_when_available(self):
        full_text = "Be it enacted by the Senate. SECTION 1."
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value={
                 **API_PAGE_1, "recordsTotal": 2, "data": [ROW_1]
             }), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=full_text):
            docs = scrape_laws(max_items=1)
        assert docs[0]["text"] == full_text

    def test_falls_back_to_long_title_when_detail_fails(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value={
                 **API_PAGE_1, "recordsTotal": 1, "data": [ROW_1]
             }), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=1)
        assert "OFFSHORE GAMING" in docs[0]["text"]

    def test_returns_empty_when_api_page_fails(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value=None):
            docs = scrape_laws(max_items=10)
        assert docs == []

    # --- from_year ---

    def test_from_year_excludes_laws_older_than_cutoff(self):
        old_row = ["REPUBLIC ACT NO. 9000", "2004-06-15",
                   "<a href='https://elibrary.judiciary.gov.ph/x'>Old Act</a>"]
        page = {"recordsTotal": 3, "data": [ROW_1, ROW_2, old_row]}
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value=page), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=10, from_year=2006)
        titles = [d["title"] for d in docs]
        assert "REPUBLIC ACT NO. 9000" not in titles
        assert "REPUBLIC ACT NO. 12312" in titles

    def test_from_year_stops_pagination_when_cutoff_reached(self):
        old_row = ["REPUBLIC ACT NO. 9000", "2004-06-15",
                   "<a href='https://elibrary.judiciary.gov.ph/x'>Old Act</a>"]
        page = {"recordsTotal": 100, "data": [ROW_1, old_row]}
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page",
                   return_value=page) as mock_page, \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            scrape_laws(max_items=100, from_year=2006)
        # must not request a second page after hitting an old law
        assert mock_page.call_count == 1

    def test_from_year_none_does_not_stop_early(self):
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page",
                   side_effect=[API_PAGE_1, API_PAGE_2]), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=3, from_year=None)
        assert len(docs) == 3

    def test_from_year_includes_laws_exactly_on_boundary(self):
        boundary_row = ["REPUBLIC ACT NO. 9001", "2006-01-01",
                        "<a href='https://elibrary.judiciary.gov.ph/y'>Boundary Act</a>"]
        page = {"recordsTotal": 1, "data": [boundary_row]}
        with patch("data_ingestion.scrapers.official_gazette._fetch_index", return_value="tok"), \
             patch("data_ingestion.scrapers.official_gazette._fetch_ra_page", return_value=page), \
             patch("data_ingestion.scrapers.official_gazette._fetch_law_text", return_value=""):
            docs = scrape_laws(max_items=10, from_year=2006)
        assert len(docs) == 1
