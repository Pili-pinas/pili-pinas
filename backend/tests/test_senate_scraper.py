"""
Tests for data_ingestion/scrapers/senate.py

HTTP calls are mocked — no real network requests made.
"""

import pytest
from unittest.mock import MagicMock, patch
from data_ingestion.scrapers.senate import (
    scrape_bills,
    scrape_senators,
    _bill_to_doc,
    _parse_senators_table,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_BILL = {
    "id": "abc123",
    "type": "bill",
    "subtype": "SB",
    "name": "SBN-01321",
    "bill_number": 1321,
    "congress": 20,
    "title": "NATIONAL PREVENTIVE MECHANISM ACT",
    "long_title": "AN ACT ESTABLISHING NATIONAL PREVENTIVE MECHANISM",
    "date_filed": "2025-09-02",
    "scope": "NATIONAL",
    "subjects": ["Torture Prevention", "Human Rights"],
    "authors_raw": "Ejercito Estrada, Jinggoy",
    "senate_website_permalink": "https://web.senate.gov.ph/lis/bill_res.aspx?congress=20&q=SBN-1321",
    "download_url_sources": ["https://web.senate.gov.ph/lisdata/4796443986!.pdf"],
    "authors": [{"first_name": "Jose", "last_name": "Ejercito-Estrada"}],
}

BETTERGOV_PAGE_1 = {
    "success": True,
    "data": [SAMPLE_BILL, {**SAMPLE_BILL, "name": "SBN-01322", "bill_number": 1322,
                            "title": "DIGITAL ID ACT", "long_title": "AN ACT ON DIGITAL IDS",
                            "subjects": ["Digital ID"], "date_filed": "2025-09-03"}],
    "pagination": {"total": 3, "limit": 2, "offset": 0, "has_more": True, "next_cursor": "2"},
}

BETTERGOV_PAGE_2 = {
    "success": True,
    "data": [{**SAMPLE_BILL, "name": "SBN-01323", "bill_number": 1323,
               "title": "CLIMATE ACT", "long_title": "AN ACT ON CLIMATE", "subjects": []}],
    "pagination": {"total": 3, "limit": 1, "offset": 2, "has_more": False, "next_cursor": None},
}

WIKIPEDIA_HTML = """
<html><body>
<table class="wikitable">
  <tr>
    <th>Portrait</th><th>Senator</th><th>Party</th><th>Bloc</th>
    <th>Born</th><th>Occupation(s)</th><th>Previous elective office(s)</th>
    <th>Education</th><th>Took office</th><th>Term ending</th><th>Term</th>
  </tr>
  <tr>
    <td></td>
    <td><a href="/wiki/Bam_Aquino">Bam Aquino</a></td>
    <td></td>
    <td>KANP</td>
    <td>Majority</td>
    <td>(1977-05-07)May 7, 1977(age 48)</td>
    <td>Social entrepreneurTelevision host</td>
    <td>None</td>
    <td>Ateneo</td>
    <td>June 30, 2025</td>
    <td>June 30, 2031</td>
    <td>1</td>
  </tr>
  <tr>
    <td></td>
    <td><a href="/wiki/Pia_Cayetano">Pia Cayetano</a></td>
    <td></td>
    <td>Nacionalista</td>
    <td>Majority</td>
    <td>(1966-03-22)March 22, 1966(age 59)</td>
    <td>Lawyer</td>
    <td>None</td>
    <td>UP</td>
    <td>June 30, 2025</td>
    <td>June 30, 2031</td>
    <td>2</td>
  </tr>
</table>
</body></html>
"""

# Table without "Senator" header — should be skipped
IRRELEVANT_TABLE_HTML = """
<html><body>
<table class="wikitable">
  <tr><th>Party</th><th>Seats</th></tr>
  <tr><td>NPC</td><td>6</td></tr>
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
# _bill_to_doc
# ---------------------------------------------------------------------------

class TestBillToDoc:
    def test_sets_source_to_bettergov(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert doc["source"] == "bettergov.ph"

    def test_sets_source_type_to_bill(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert doc["source_type"] == "bill"

    def test_title_includes_bill_name_and_title(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert "SBN-01321" in doc["title"]
        assert "NATIONAL PREVENTIVE MECHANISM ACT" in doc["title"]

    def test_date_comes_from_date_filed(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert doc["date"] == "2025-09-02"

    def test_politician_is_authors_raw(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert doc["politician"] == "Ejercito Estrada, Jinggoy"

    def test_url_is_senate_permalink(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert "SBN-1321" in doc["url"]

    def test_text_includes_title_and_long_title(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert "NATIONAL PREVENTIVE MECHANISM ACT" in doc["text"]
        assert "AN ACT ESTABLISHING" in doc["text"]

    def test_text_includes_authors(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert "Ejercito Estrada, Jinggoy" in doc["text"]
        assert "Authors:" in doc["text"]

    def test_text_includes_subjects(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert "Torture Prevention" in doc["text"]
        assert "Human Rights" in doc["text"]

    def test_text_without_subjects_still_works(self):
        bill = {**SAMPLE_BILL, "subjects": []}
        doc = _bill_to_doc(bill, congress=20)
        assert "Subjects:" not in doc["text"]

    def test_text_without_authors_omits_authors_line(self):
        bill = {**SAMPLE_BILL, "authors_raw": None}
        doc = _bill_to_doc(bill, congress=20)
        assert "Authors:" not in doc["text"]

    def test_missing_date_falls_back_to_today(self):
        from datetime import datetime
        bill = {**SAMPLE_BILL, "date_filed": None}
        doc = _bill_to_doc(bill, congress=20)
        datetime.strptime(doc["date"], "%Y-%m-%d")  # must be a valid date

    def test_congress_stored_correctly(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert doc["congress"] == 20

    def test_bill_number_stored(self):
        doc = _bill_to_doc(SAMPLE_BILL, congress=20)
        assert doc["bill_number"] == 1321


# ---------------------------------------------------------------------------
# scrape_bills
# ---------------------------------------------------------------------------

class TestScrapeBills:
    def test_returns_empty_list_when_api_fails(self):
        with patch("data_ingestion.scrapers.senate._get", return_value=None):
            docs = scrape_bills(congress=20, max_items=10)
        assert docs == []

    def test_returns_documents_from_api(self):
        resp = _mock_response(BETTERGOV_PAGE_1)
        resp2 = _mock_response(BETTERGOV_PAGE_2)
        with patch("data_ingestion.scrapers.senate._get", side_effect=[resp, resp2]):
            docs = scrape_bills(congress=20, max_items=3)
        assert len(docs) == 3

    def test_respects_max_items(self):
        resp = _mock_response(BETTERGOV_PAGE_1)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_bills(congress=20, max_items=1)
        assert len(docs) == 1

    def test_stops_when_has_more_is_false(self):
        single_page = {**BETTERGOV_PAGE_1, "pagination": {
            "total": 2, "limit": 2, "offset": 0, "has_more": False, "next_cursor": None
        }}
        resp = _mock_response(single_page)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp) as mock_get:
            docs = scrape_bills(congress=20, max_items=100)
        assert mock_get.call_count == 1  # only one page fetched

    def test_returns_empty_when_api_returns_failure(self):
        resp = _mock_response({"success": False, "error": "not found"})
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_bills(congress=20)
        assert docs == []

    def test_documents_have_required_fields(self):
        resp = _mock_response({**BETTERGOV_PAGE_1,
                                "pagination": {**BETTERGOV_PAGE_1["pagination"], "has_more": False}})
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_bills(congress=20, max_items=5)
        doc = docs[0]
        assert doc["source"] == "bettergov.ph"
        assert doc["source_type"] == "bill"
        assert "date" in doc
        assert "politician" in doc
        assert "title" in doc
        assert "url" in doc
        assert "text" in doc


# ---------------------------------------------------------------------------
# _parse_senators_table
# ---------------------------------------------------------------------------

class TestParseSenators:
    def _soup(self, html):
        return BeautifulSoup(html, "lxml")

    def test_returns_correct_number_of_senators(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        assert len(rows) == 2

    def test_extracts_senator_name(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        assert rows[0]["name"] == "Bam Aquino"

    def test_extracts_wiki_path(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        assert rows[0]["wiki_path"] == "/wiki/Bam_Aquino"

    def test_bio_text_contains_party(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        assert "KANP" in rows[0]["bio_text"]

    def test_bio_text_contains_bloc(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        assert "Majority" in rows[0]["bio_text"]

    def test_bio_text_strips_sortkey_from_born(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        # Sortkey like "(1977-05-07)" should be stripped
        assert "(1977-05-07)" not in rows[0]["bio_text"]
        assert "1977" in rows[0]["bio_text"]

    def test_skips_table_without_senator_header(self):
        rows = _parse_senators_table(self._soup(IRRELEVANT_TABLE_HTML))
        assert rows == []

    def test_second_senator_parsed(self):
        rows = _parse_senators_table(self._soup(WIKIPEDIA_HTML))
        assert rows[1]["name"] == "Pia Cayetano"
        assert "Nacionalista" in rows[1]["bio_text"]


# ---------------------------------------------------------------------------
# scrape_senators
# ---------------------------------------------------------------------------

class TestScrapeSenators:
    def test_returns_empty_when_wikipedia_fails(self):
        with patch("data_ingestion.scrapers.senate._get", return_value=None):
            docs = scrape_senators()
        assert docs == []

    def test_returns_one_doc_per_senator(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_resp = _mock_response("<html><body><div class='mw-parser-output'><p>Senator bio.</p></div></body></html>")
        bio_resp.content = b"<html><body><div class='mw-parser-output'><p>Senator bio.</p></div></body></html>"
        with patch("data_ingestion.scrapers.senate._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_senators()
        assert len(docs) == 2

    def test_document_has_required_metadata_fields(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_html = b"<html><body><div class='mw-parser-output'><p>Senator bio.</p></div></body></html>"
        bio_resp = _mock_response("")
        bio_resp.content = bio_html
        with patch("data_ingestion.scrapers.senate._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_senators()
        doc = docs[0]
        assert doc["source"] == "wikipedia.org"
        assert doc["source_type"] == "profile"
        assert doc["politician"] == "Bam Aquino"
        assert "Senator Profile" in doc["title"]
        assert "wikipedia.org/wiki" in doc["url"]

    def test_uses_wiki_bio_text_when_available(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_html = b"<html><body><div class='mw-parser-output'><p>Bam Aquino is a Filipino senator.</p></div></body></html>"
        bio_resp = _mock_response("")
        bio_resp.content = bio_html
        with patch("data_ingestion.scrapers.senate._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_senators()
        assert "Filipino senator" in docs[0]["text"]

    def test_falls_back_to_table_bio_when_wiki_page_fails(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        # Only the list page succeeds; individual pages fail
        with patch("data_ingestion.scrapers.senate._get", side_effect=[list_resp, None, None]):
            docs = scrape_senators()
        assert len(docs) == 2
        # Should still have bio_text from table
        assert "KANP" in docs[0]["text"]

    def test_strips_footnote_markers_from_bio_text(self):
        list_resp = _mock_response(WIKIPEDIA_HTML)
        list_resp.content = WIKIPEDIA_HTML.encode()
        bio_html = b"<html><body><div class='mw-parser-output'><p>He served[1] as senator[2].</p></div></body></html>"
        bio_resp = _mock_response("")
        bio_resp.content = bio_html
        with patch("data_ingestion.scrapers.senate._get", side_effect=[list_resp, bio_resp, bio_resp]):
            docs = scrape_senators()
        assert "[1]" not in docs[0]["text"]
        assert "[2]" not in docs[0]["text"]

    def test_returns_empty_when_no_senator_table_found(self):
        list_resp = _mock_response(IRRELEVANT_TABLE_HTML)
        list_resp.content = IRRELEVANT_TABLE_HTML.encode()
        with patch("data_ingestion.scrapers.senate._get", return_value=list_resp):
            docs = scrape_senators()
        assert docs == []
