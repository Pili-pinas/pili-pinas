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
)


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

BETTERGOV_REP = {
    "id": "person-r1",
    "first_name": "Joseph",
    "middle_name": "",
    "last_name": "Bernos",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": ["Joe"],
    "senate_website_keys": None,
    "congress_website_primary_keys": [123],
    "congresses_served": [
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Representative"},
    ],
}

BETTERGOV_REP_2 = {
    "id": "person-r2",
    "first_name": "Dale",
    "middle_name": "",
    "last_name": "Corvera",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": [],
    "senate_website_keys": None,
    "congress_website_primary_keys": [456],
    "congresses_served": [
        {"congress_number": 18, "congress_ordinal": "18th", "position": "Representative"},
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Representative"},
    ],
}

BETTERGOV_SENATOR_PERSON = {
    "id": "person-s1",
    "first_name": "Risa",
    "middle_name": "",
    "last_name": "Hontiveros",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": [],
    "senate_website_keys": ["hontiveros"],
    "congress_website_primary_keys": None,
    "congresses_served": [
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Senator"},
    ],
}

PEOPLE_PAGE_1 = {
    "success": True,
    "data": [BETTERGOV_REP, BETTERGOV_SENATOR_PERSON],
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

    def test_text_includes_authors_when_present(self):
        bill = {**SAMPLE_HB, "authors_raw": "Cayetano, Alan Peter"}
        doc = _hb_to_doc(bill, congress=19)
        assert "Cayetano, Alan Peter" in doc["text"]
        assert "Authors:" in doc["text"]

    def test_text_without_authors_omits_authors_line(self):
        doc = _hb_to_doc(SAMPLE_HB, congress=19)  # authors_raw is None
        assert "Authors:" not in doc["text"]

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

    def test_stops_after_scanning_too_many_null_title_records(self):
        """When all pages have null titles and has_more=True, should not loop forever."""
        null_page = {
            "success": True,
            "data": [
                {**SAMPLE_HB, "title": None, "long_title": None},
                {**SAMPLE_HB, "title": None, "long_title": None},
            ],
            "pagination": {"total": 9999, "limit": 2, "has_more": True, "next_cursor": "next"},
        }
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(null_page)

        with patch("data_ingestion.scrapers.congress._get", side_effect=fake_get):
            docs = scrape_house_bills(congress=20, max_items=5)

        assert docs == []
        assert call_count < 20  # must stop well before exhausting the full dataset

    def test_page_cap_does_not_scale_linearly_with_max_items(self):
        """max_items=2000 must not cause 2000 page fetches through null-title records."""
        null_page = {
            "success": True,
            "data": [{**SAMPLE_HB, "title": None}] * 50,
            "pagination": {"total": 99999, "limit": 50, "has_more": True, "next_cursor": "next"},
        }
        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(null_page)

        with patch("data_ingestion.scrapers.congress._get", side_effect=fake_get):
            scrape_house_bills(congress=20, max_items=2000)

        assert call_count < 300  # must be far less than 2000

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
# scrape_members
# ---------------------------------------------------------------------------

class TestScrapeMembers:
    def test_returns_empty_when_api_fails(self):
        with patch("data_ingestion.scrapers.congress._get", return_value=None):
            docs = scrape_members()
        assert docs == []

    def test_returns_only_representatives_not_senators(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_members()
        assert len(docs) == 1
        assert "Bernos" in docs[0]["politician"]

    def test_document_has_required_metadata_fields(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_members()
        doc = docs[0]
        assert doc["source"] == "open-congress-api.bettergov.ph"
        assert doc["source_type"] == "profile"
        assert "Bernos" in doc["politician"]
        assert "Representative Profile" in doc["title"]

    def test_profile_text_includes_congresses_served(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_members()
        assert "19th" in docs[0]["text"]

    def test_profile_text_includes_aliases(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_members()
        assert "Joe" in docs[0]["text"]

    def test_excludes_person_with_no_representative_role(self):
        senator_only = {**PEOPLE_PAGE_1, "data": [BETTERGOV_SENATOR_PERSON]}
        resp = _mock_response(senator_only)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            docs = scrape_members()
        assert docs == []

    def test_handles_pagination(self):
        page1 = {
            "success": True,
            "data": [BETTERGOV_REP],
            "pagination": {"has_more": True, "next_cursor": "cursor-2"},
        }
        page2 = {
            "success": True,
            "data": [BETTERGOV_REP_2],
            "pagination": {"has_more": False, "next_cursor": None},
        }
        resp1 = _mock_response(page1)
        resp2 = _mock_response(page2)
        with patch("data_ingestion.scrapers.congress._get", side_effect=[resp1, resp2]):
            docs = scrape_members()
        assert len(docs) == 2

    def test_filters_by_congress(self):
        resp = _mock_response(PEOPLE_PAGE_1)
        with patch("data_ingestion.scrapers.congress._get", return_value=resp):
            # BETTERGOV_REP served in 19th; filter to only 20th → excluded
            docs = scrape_members(congresses=[20])
        assert docs == []
