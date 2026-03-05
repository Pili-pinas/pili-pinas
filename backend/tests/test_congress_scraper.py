"""
Tests for data_ingestion/scrapers/congress.py

HTTP calls are mocked — no real network requests made.
"""

import pytest
from unittest.mock import MagicMock, patch
from data_ingestion.scrapers.congress import (
    scrape_house_bills,
    scrape_members,
    _hb_to_doc,
    _parse_members_table,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_HB = {
    "id": "abc456",
    "type": "bill",
    "subtype": "HB",
    "name": "HBN-06571",
    "bill_number": 6571,
    "congress": 19,
    "title": "THE RIGHT-OF-WAY ACT",
    "long_title": "AN ACT PROVIDING FOR THE EXPEDITIOUS ACQUISITION OF RIGHT-OF-WAY",
    "date_filed": "2022-12-07",
    "scope": "NATIONAL",
    "subjects": ["Infrastructure", "Land Acquisition"],
    "authors_raw": None,  # HB data often lacks authors
    "senate_website_permalink": None,
    "download_url_sources": [],
    "authors": [],
}

BETTERGOV_PAGE_1 = {
    "success": True,
    "data": [
        SAMPLE_HB,
        {**SAMPLE_HB, "name": "HBN-06572", "bill_number": 6572,
         "title": "PUBLIC SCHOOLS OF THE FUTURE ACT", "subjects": ["Education"]},
    ],
    "pagination": {"total": 3, "limit": 2, "offset": 0, "has_more": True, "next_cursor": "2"},
}

BETTERGOV_PAGE_2 = {
    "success": True,
    "data": [{**SAMPLE_HB, "name": "HBN-06573", "bill_number": 6573,
               "title": "FREE LEGAL AID ACT", "subjects": []}],
    "pagination": {"total": 3, "limit": 1, "offset": 2, "has_more": False, "next_cursor": None},
}

# Page where some bills have no title — should be skipped
BETTERGOV_MIXED = {
    "success": True,
    "data": [
        {**SAMPLE_HB, "name": "HBN-00001", "title": None, "long_title": None},  # skip
        SAMPLE_HB,  # keep
    ],
    "pagination": {"total": 2, "limit": 2, "offset": 0, "has_more": False, "next_cursor": None},
}

# Wikipedia-style HTML for House members table
WIKIPEDIA_HTML = """
<html><body>
<table class="wikitable">
  <tr>
    <th>Constituency</th><th>Portrait</th><th>Representative</th><th>Party flag</th>
    <th>Party</th><th>Bloc</th><th>Born</th><th>Prior experience</th><th>Took office</th>
  </tr>
  <tr>
    <td><a href="/wiki/Abra">Abra at-large</a></td>
    <td></td>
    <td><a href="/wiki/Joseph_Bernos">Joseph Bernos</a></td>
    <td></td>
    <td>Lakas</td>
    <td>Majority</td>
    <td>(1978-10-06)October 6, 1978(age 47)</td>
    <td>Mayor of La Paz</td>
    <td>June 30, 2022</td>
  </tr>
  <tr>
    <td><a href="/wiki/Agusan_del_Norte">Agusan del Norte at-large</a></td>
    <td></td>
    <td><a href="/wiki/Dale_Corvera">Dale Corvera</a></td>
    <td></td>
    <td>NUP</td>
    <td>Majority</td>
    <td>(1955-06-27)June 27, 1955(age 70)</td>
    <td>Governor of Agusan del Norte</td>
    <td>June 30, 2019</td>
  </tr>
</table>
</body></html>
"""

# Table without "Representative" header — should be skipped
IRRELEVANT_TABLE_HTML = """
<html><body>
<table class="wikitable">
  <tr><th>Party</th><th>Seats</th></tr>
  <tr><td>Lakas</td><td>120</td></tr>
</table>
</body></html>
"""


def _mock_response(data, status=200):
    resp = MagicMock()
    resp.status_code = status
    if isinstance(data, dict):
        resp.json.return_value = data
    else:
        resp.content = data.encode() if isinstance(data, str) else data
        resp.text = data if isinstance(data, str) else data.decode()
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _hb_to_doc
# ---------------------------------------------------------------------------

class TestHbToDoc:
    def test_sets_source_to_bettergov(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert doc["source"] == "bettergov.ph"

    def test_sets_source_type_to_bill(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert doc["source_type"] == "bill"

    def test_title_includes_bill_name_and_title(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert "HBN-06571" in doc["title"]
        assert "RIGHT-OF-WAY" in doc["title"]

    def test_date_comes_from_date_filed(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert doc["date"] == "2022-12-07"

    def test_missing_date_falls_back_to_today(self):
        from datetime import datetime
        bill = {**SAMPLE_HB, "date_filed": None}
        doc = _hb_to_doc(bill, congress=19)
        datetime.strptime(doc["date"], "%Y-%m-%d")  # must be a valid date

    def test_politician_is_authors_raw_when_present(self):
        bill = {**SAMPLE_HB, "authors_raw": "Cayetano, Alan Peter"}
        doc = _hb_to_doc(bill, congress=19)
        assert doc["politician"] == "Cayetano, Alan Peter"

    def test_politician_is_empty_string_when_authors_raw_is_none(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)  # authors_raw is None
        assert doc["politician"] == ""

    def test_text_includes_title_and_long_title(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert "RIGHT-OF-WAY ACT" in doc["text"]
        assert "EXPEDITIOUS ACQUISITION" in doc["text"]

    def test_text_includes_subjects(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert "Infrastructure" in doc["text"]
        assert "Land Acquisition" in doc["text"]

    def test_text_without_subjects_still_works(self):
        bill = {**SAMPLE_HB, "subjects": []}
        doc = _hb_to_doc(bill, congress=19)
        assert "Subjects:" not in doc["text"]

    def test_congress_stored_correctly(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert doc["congress"] == 19

    def test_bill_number_stored(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)
        assert doc["bill_number"] == 6571


# ---------------------------------------------------------------------------
# scrape_house_bills
# ---------------------------------------------------------------------------

class TestScrapeHouseBills:
    def test_returns_empty_list_when_api_fails(self):
        with patch("data_ingestion.scrapers.congress._get", return_value=None):
            docs = scrape_house_bills(congress=19, max_items=10)
        assert docs == []

    def test_returns_documents_from_api(self):
        resp1 = _mock_response(BETTERGOV_PAGE_1)
        resp2 = _mock_response(BETTERGOV_PAGE_2)
        with patch("data_ingestion.scrapers.congress._get", side_effect=[resp1, resp2]):
            docs = scrape_house_bills(congress=19, max_items=3)
        assert len(docs) == 3

    def test_respects_max_items(self):
        resp = _mock_response(BETTERGOV_PAGE_1)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_house_bills(congress=19, max_items=1)
        assert len(docs) == 1

    def test_stops_when_has_more_is_false(self):
        single_page = {**BETTERGOV_PAGE_1, "pagination": {
            "total": 2, "limit": 2, "offset": 0, "has_more": False, "next_cursor": None
        }}
        resp = _mock_response(single_page)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp) as mock_get:
            scrape_house_bills(congress=19, max_items=100)
        assert mock_get.call_count == 1

    def test_skips_bills_without_title(self):
        resp = _mock_response(BETTERGOV_MIXED)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_house_bills(congress=19, max_items=10)
        assert len(docs) == 1
        assert "RIGHT-OF-WAY" in docs[0]["title"]

    def test_returns_empty_when_api_returns_failure(self):
        resp = _mock_response({"success": False, "error": "not found"})
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_house_bills(congress=19)
        assert docs == []

    def test_documents_have_required_fields(self):
        resp = _mock_response({**BETTERGOV_PAGE_1,
                                "pagination": {**BETTERGOV_PAGE_1["pagination"], "has_more": False}})
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_house_bills(congress=19, max_items=5)
        doc = docs[0]
        assert doc["source"] == "bettergov.ph"
        assert doc["source_type"] == "bill"
        assert "date" in doc
        assert "politician" in doc
        assert "title" in doc
        assert "url" in doc
        assert "text" in doc


# ---------------------------------------------------------------------------
# _parse_members_table
# ---------------------------------------------------------------------------

class TestParseMembersTable:
    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_returns_correct_number_of_members(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert len(rows) == 2

    def test_extracts_member_name(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert rows[0]["name"] == "Joseph Bernos"

    def test_extracts_constituency(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert rows[0]["constituency"] == "Abra at-large"

    def test_extracts_wiki_path(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert rows[0]["wiki_path"] == "/wiki/Joseph_Bernos"

    def test_bio_text_contains_party(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert "Lakas" in rows[0]["bio_text"]

    def test_bio_text_contains_bloc(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert "Majority" in rows[0]["bio_text"]

    def test_bio_text_contains_constituency(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert "Abra at-large" in rows[0]["bio_text"]

    def test_bio_text_strips_sortkey_from_born(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert "(1978-10-06)" not in rows[0]["bio_text"]
        assert "1978" in rows[0]["bio_text"]

    def test_skips_table_without_representative_header(self):
        rows = _parse_members_table(self._soup(IRRELEVANT_TABLE_HTML))
        assert rows == []

    def test_second_member_parsed(self):
        rows = _parse_members_table(self._soup(WIKIPEDIA_HTML))
        assert rows[1]["name"] == "Dale Corvera"
        assert "NUP" in rows[1]["bio_text"]


# ---------------------------------------------------------------------------
# scrape_members
# ---------------------------------------------------------------------------

class TestScrapeMembers:
    def test_returns_empty_when_wikipedia_fails(self):
        with patch("data_ingestion.scrapers.congress._get", return_value=None):
            docs = scrape_members()
        assert docs == []

    def test_returns_one_doc_per_member(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_resp = _mock_response("<html><body><div class='mw-parser-output'><p>Bio text.</p></div></body></html>")
        bio_resp.content = b"<html><body><div class='mw-parser-output'><p>Bio text.</p></div></body></html>"
        with patch("data_ingestion.scrapers.congress._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_members()
        assert len(docs) == 2

    def test_document_has_required_metadata_fields(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_html = b"<html><body><div class='mw-parser-output'><p>Rep bio.</p></div></body></html>"
        bio_resp = _mock_response("")
        bio_resp.content = bio_html
        with patch("data_ingestion.scrapers.congress._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_members()
        doc = docs[0]
        assert doc["source"] == "wikipedia.org"
        assert doc["source_type"] == "profile"
        assert doc["politician"] == "Joseph Bernos"
        assert "Representative Profile" in doc["title"]
        assert "wikipedia.org/wiki" in doc["url"]

    def test_uses_wiki_bio_text_when_available(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_html = b"<html><body><div class='mw-parser-output'><p>Joseph Bernos is a Filipino politician.</p></div></body></html>"
        bio_resp = _mock_response("")
        bio_resp.content = bio_html
        with patch("data_ingestion.scrapers.congress._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_members()
        assert "Filipino politician" in docs[0]["text"]

    def test_falls_back_to_table_bio_when_wiki_page_fails(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        with patch("data_ingestion.scrapers.congress._get", side_effect=[list_resp, None, None]):
            docs = scrape_members()
        assert len(docs) == 2
        assert "Lakas" in docs[0]["text"]

    def test_strips_footnote_markers_from_bio_text(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_html = b"<html><body><div class='mw-parser-output'><p>He served[1] as rep[2].</p></div></body></html>"
        bio_resp = _mock_response("")
        bio_resp.content = bio_html
        with patch("data_ingestion.scrapers.congress._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_members()
        assert "[1]" not in docs[0]["text"]
        assert "[2]" not in docs[0]["text"]

    def test_returns_empty_when_no_member_table_found(self):
        list_resp = _mock_response(IRRELEVANT_TABLE_HTML)
        list_resp.content = IRRELEVANT_TABLE_HTML.encode()
        with patch("data_ingestion.scrapers.congress._get", return_value=list_resp):
            docs = scrape_members()
        assert docs == []
