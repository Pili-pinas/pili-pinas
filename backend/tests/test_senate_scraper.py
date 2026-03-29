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
)


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

BETTERGOV_SENATOR = {
    "id": "person-s1",
    "first_name": "Risa",
    "middle_name": "Calderon",
    "last_name": "Hontiveros",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": ["Risa"],
    "senate_website_keys": ["hontiveros"],
    "congress_website_primary_keys": None,
    "congresses_served": [
        {"congress_number": 18, "congress_ordinal": "18th", "position": "Senator"},
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Senator"},
    ],
}

BETTERGOV_SENATOR_2 = {
    "id": "person-s2",
    "first_name": "Nancy",
    "middle_name": "",
    "last_name": "Binay",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": [],
    "senate_website_keys": ["binay"],
    "congress_website_primary_keys": None,
    "congresses_served": [
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Senator"},
    ],
}

BETTERGOV_REP_PERSON = {
    "id": "person-r1",
    "first_name": "Joseph",
    "middle_name": "",
    "last_name": "Bernos",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": [],
    "senate_website_keys": None,
    "congress_website_primary_keys": [123],
    "congresses_served": [
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Representative"},
    ],
}

PEOPLE_PAGE_1 = {
    "success": True,
    "data": [BETTERGOV_SENATOR, BETTERGOV_REP_PERSON],
    "pagination": {"has_more": False, "next_cursor": None},
}


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
# scrape_senators
# ---------------------------------------------------------------------------

class TestScrapeSenators:
    def test_returns_empty_when_api_fails(self):
        with patch("data_ingestion.scrapers.senate._get", return_value=None):
            docs = scrape_senators()
        assert docs == []

    def test_returns_only_senators_not_representatives(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_senators()
        assert len(docs) == 1
        assert "Hontiveros" in docs[0]["politician"]

    def test_document_has_required_metadata_fields(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_senators()
        doc = docs[0]
        assert doc["source"] == "open-congress-api.bettergov.ph"
        assert doc["source_type"] == "profile"
        assert "Hontiveros" in doc["politician"]
        assert "Senator Profile" in doc["title"]

    def test_profile_text_includes_congresses_served(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_senators()
        assert "18th" in docs[0]["text"]
        assert "19th" in docs[0]["text"]

    def test_profile_text_includes_aliases(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_senators()
        assert "Risa" in docs[0]["text"]

    def test_excludes_person_with_no_senator_role(self):
        rep_only = {**PEOPLE_PAGE_1, "data": [BETTERGOV_REP_PERSON]}
        resp = _mock_response(rep_only)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            docs = scrape_senators()
        assert docs == []

    def test_handles_pagination(self):
        page1 = {
            "success": True,
            "data": [BETTERGOV_SENATOR],
            "pagination": {"has_more": True, "next_cursor": "cursor-2"},
        }
        page2 = {
            "success": True,
            "data": [BETTERGOV_SENATOR_2],
            "pagination": {"has_more": False, "next_cursor": None},
        }
        resp1 = _mock_response(page1)
        resp2 = _mock_response(page2)
        with patch("data_ingestion.scrapers.senate._get", side_effect=[resp1, resp2]):
            docs = scrape_senators()
        assert len(docs) == 2

    def test_filters_by_congress(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp):
            # BETTERGOV_SENATOR served in 18th and 19th; filter to only 20th → excluded
            docs = scrape_senators(congresses=[20])
        assert docs == []

    def test_stops_pagination_when_has_more_true_but_no_cursor(self):
        """has_more=True with next_cursor=None must not cause an infinite loop."""
        page = {
            "success": True,
            "data": [BETTERGOV_SENATOR],
            "pagination": {"has_more": True, "next_cursor": None},
        }
        resp = _mock_response(page)
        with patch("data_ingestion.scrapers.senate._get", return_value=resp) as mock_get:
            docs = scrape_senators()
        # Should fetch exactly once and stop — not loop forever
        assert mock_get.call_count == 1
        assert len(docs) >= 0  # whatever was in that page
