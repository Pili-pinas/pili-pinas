"""
Tests for data_ingestion/scrapers/politicians.py

HTTP calls are mocked — no real network requests made.
"""

import pytest
from unittest.mock import patch
from data_ingestion.scrapers.politicians import (
    _build_full_name,
    _bills_for_person,
    _build_enriched_profile,
    scrape_all_politicians,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LENI = {
    "id": "person-leni",
    "first_name": "Maria Leonor",
    "middle_name": "",
    "last_name": "Robredo",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": ["Leni"],
    "congresses_served": [
        {"congress_number": 15, "congress_ordinal": "15th", "position": "Representative"},
        {"congress_number": 16, "congress_ordinal": "16th", "position": "Representative"},
    ],
}

RISA = {
    "id": "person-risa",
    "first_name": "Risa",
    "middle_name": "Calderon",
    "last_name": "Hontiveros",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": ["Risa"],
    "congresses_served": [
        {"congress_number": 18, "congress_ordinal": "18th", "position": "Senator"},
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Senator"},
    ],
}

PERSON_WITH_PREFIX = {
    "id": "person-prefix",
    "first_name": "Juan",
    "middle_name": "dela",
    "last_name": "Cruz",
    "name_prefix": "Hon.",
    "name_suffix": "Jr.",
    "aliases": [],
    "congresses_served": [
        {"congress_number": 19, "congress_ordinal": "19th", "position": "Representative"},
    ],
}

PERSON_NO_CONGRESSES = {
    "id": "person-nocong",
    "first_name": "Unknown",
    "middle_name": "",
    "last_name": "Person",
    "name_prefix": None,
    "name_suffix": None,
    "aliases": [],
    "congresses_served": [],
}


def _bill(title, authors_raw, congress=16):
    return {
        "source": "bettergov.ph",
        "source_type": "bill",
        "title": title,
        "politician": authors_raw,
        "date": "2014-01-01",
        "text": f"{title}\nAuthors: {authors_raw}",
        "congress": congress,
    }


LENI_BILL_1 = _bill("HBN-001: Housing for All Act", "Robredo, Maria Leonor")
LENI_BILL_2 = _bill("HBN-002: Education Funding Act", "Rep. Robredo, Lagman")
OTHER_BILL   = _bill("SBN-999: Unrelated Act", "Hontiveros, Risa", congress=18)


PEOPLE_PAGE = {
    "success": True,
    "data": [LENI, RISA],
    "pagination": {"has_more": False, "next_cursor": None},
}


def _mock_response(data):
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _build_full_name
# ---------------------------------------------------------------------------

class TestBuildFullName:
    def test_basic_name(self):
        assert _build_full_name(LENI) == "Maria Leonor Robredo"

    def test_includes_prefix_and_suffix(self):
        name = _build_full_name(PERSON_WITH_PREFIX)
        assert "Hon." in name
        assert "Jr." in name
        assert "Juan" in name

    def test_skips_empty_parts(self):
        person = {**LENI, "middle_name": ""}
        name = _build_full_name(person)
        assert "  " not in name  # no double spaces

    def test_handles_all_none_parts(self):
        person = {"first_name": None, "middle_name": None, "last_name": "Cruz",
                  "name_prefix": None, "name_suffix": None}
        assert _build_full_name(person) == "Cruz"


# ---------------------------------------------------------------------------
# _bills_for_person
# ---------------------------------------------------------------------------

class TestBillsForPerson:
    def test_matches_by_last_name(self):
        result = _bills_for_person(LENI, [LENI_BILL_1, LENI_BILL_2, OTHER_BILL])
        assert len(result) == 2

    def test_no_match_returns_empty(self):
        result = _bills_for_person(LENI, [OTHER_BILL])
        assert result == []

    def test_empty_bills_returns_empty(self):
        assert _bills_for_person(LENI, []) == []

    def test_case_insensitive_match(self):
        bill = _bill("HBN-003: Some Act", "ROBREDO, LENI")
        result = _bills_for_person(LENI, [bill])
        assert len(result) == 1

    def test_person_with_no_last_name_returns_empty(self):
        person = {**LENI, "last_name": None}
        result = _bills_for_person(person, [LENI_BILL_1])
        assert result == []

    def test_does_not_match_unrelated_bills(self):
        result = _bills_for_person(RISA, [LENI_BILL_1, LENI_BILL_2])
        assert result == []

    def test_matches_senator_bills(self):
        result = _bills_for_person(RISA, [OTHER_BILL, LENI_BILL_1])
        assert len(result) == 1
        assert result[0]["title"] == OTHER_BILL["title"]


# ---------------------------------------------------------------------------
# _build_enriched_profile
# ---------------------------------------------------------------------------

class TestBuildEnrichedProfile:
    def test_politician_field_is_full_name(self):
        doc = _build_enriched_profile(LENI, [])
        assert doc["politician"] == "Maria Leonor Robredo"

    def test_source_type_is_profile(self):
        doc = _build_enriched_profile(LENI, [])
        assert doc["source_type"] == "profile"

    def test_title_contains_name(self):
        doc = _build_enriched_profile(LENI, [])
        assert "Robredo" in doc["title"]

    def test_text_includes_congresses_served(self):
        doc = _build_enriched_profile(LENI, [])
        assert "15th" in doc["text"]
        assert "16th" in doc["text"]

    def test_text_includes_position(self):
        doc = _build_enriched_profile(LENI, [])
        assert "Representative" in doc["text"]

    def test_text_includes_alias(self):
        doc = _build_enriched_profile(LENI, [])
        assert "Leni" in doc["text"]

    def test_text_includes_authored_bills(self):
        doc = _build_enriched_profile(LENI, [LENI_BILL_1, LENI_BILL_2])
        assert "Housing for All Act" in doc["text"]
        assert "Education Funding Act" in doc["text"]

    def test_text_excludes_unrelated_bills(self):
        doc = _build_enriched_profile(LENI, [LENI_BILL_1, OTHER_BILL])
        assert "Unrelated Act" not in doc["text"]

    def test_no_bills_omits_bills_section(self):
        doc = _build_enriched_profile(LENI, [])
        assert "Bills authored" not in doc["text"]

    def test_no_aliases_omits_aliases_line(self):
        person = {**LENI, "aliases": []}
        doc = _build_enriched_profile(person, [])
        assert "Also known as" not in doc["text"]

    def test_url_contains_person_id(self):
        doc = _build_enriched_profile(LENI, [])
        assert "person-leni" in doc["url"]

    def test_caps_bills_at_twenty(self):
        many_bills = [_bill(f"HBN-{i}: Act {i}", "Robredo") for i in range(30)]
        doc = _build_enriched_profile(LENI, many_bills)
        # should not include all 30 bill titles — count semicolons in bills section
        bills_section = doc["text"].split("Bills authored:")[-1] if "Bills authored:" in doc["text"] else ""
        assert bills_section.count(";") <= 19  # max 20 bills = 19 separators


# ---------------------------------------------------------------------------
# scrape_all_politicians
# ---------------------------------------------------------------------------

class TestScrapeAllPoliticians:
    def test_returns_profile_for_each_person_with_congresses(self):
        with patch("data_ingestion.scrapers.politicians._fetch_all_people",
                   return_value=[LENI, RISA]):
            docs = scrape_all_politicians(bills=[])
        assert len(docs) == 2

    def test_skips_person_with_no_congresses_served(self):
        with patch("data_ingestion.scrapers.politicians._fetch_all_people",
                   return_value=[LENI, PERSON_NO_CONGRESSES]):
            docs = scrape_all_politicians(bills=[])
        assert len(docs) == 1
        assert "Robredo" in docs[0]["politician"]

    def test_enriches_profiles_with_bills(self):
        with patch("data_ingestion.scrapers.politicians._fetch_all_people",
                   return_value=[LENI]):
            docs = scrape_all_politicians(bills=[LENI_BILL_1, LENI_BILL_2, OTHER_BILL])
        assert "Housing for All Act" in docs[0]["text"]
        assert "Unrelated Act" not in docs[0]["text"]

    def test_defaults_to_empty_bills_when_none_passed(self):
        with patch("data_ingestion.scrapers.politicians._fetch_all_people",
                   return_value=[LENI]):
            docs = scrape_all_politicians()
        assert len(docs) == 1
        assert "Bills authored" not in docs[0]["text"]

    def test_returns_empty_when_api_fails(self):
        with patch("data_ingestion.scrapers.politicians._fetch_all_people",
                   return_value=[]):
            docs = scrape_all_politicians(bills=[])
        assert docs == []

    def test_all_docs_have_required_fields(self):
        with patch("data_ingestion.scrapers.politicians._fetch_all_people",
                   return_value=[LENI, RISA]):
            docs = scrape_all_politicians(bills=[])
        for doc in docs:
            assert doc["source_type"] == "profile"
            assert doc["politician"]
            assert doc["title"]
            assert doc["text"]
            assert doc["url"]
