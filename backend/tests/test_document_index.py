"""
Tests for document_index.py — SQLite document store + topic tagging.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from data_ingestion.document_index import init_db, upsert_documents, tag_topics, TOPIC_KEYWORDS


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test_index.db"
    init_db(p)
    return p


@pytest.fixture
def sample_bill():
    return {
        "url": "https://senate.gov.ph/bill/SBN-1234",
        "title": "SBN-1234: Magna Carta of Women",
        "source": "bettergov.ph",
        "source_type": "bill",
        "date": "2025-01-15",
        "politician": "Risa Hontiveros",
        "congress": 20,
        "text": "AN ACT STRENGTHENING MECHANISMS FOR THE PREVENTION AND ELIMINATION OF VIOLENCE AGAINST WOMEN",
    }


@pytest.fixture
def sample_news():
    return {
        "url": "https://rappler.com/article/123",
        "title": "Senate passes education bill",
        "source": "rappler.com",
        "source_type": "news",
        "date": "2025-03-01",
        "politician": "",
        "text": "The Philippine Senate passed a bill on scholarship and free tuition.",
    }


class TestInitDb:
    def test_creates_documents_table(self, tmp_path):
        path = tmp_path / "idx.db"
        init_db(path)
        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "documents" in tables

    def test_creates_document_topics_table(self, tmp_path):
        path = tmp_path / "idx.db"
        init_db(path)
        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "document_topics" in tables

    def test_is_idempotent(self, tmp_path):
        path = tmp_path / "idx.db"
        init_db(path)
        init_db(path)  # second call must not raise


class TestUpsertDocuments:
    def test_creates_one_row_per_unique_url(self, db_path, sample_bill, sample_news):
        upsert_documents([sample_bill, sample_news], db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 2

    def test_stores_correct_fields(self, db_path, sample_bill):
        upsert_documents([sample_bill], db_path=db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT url, title, source, source_type, date, politician, congress FROM documents WHERE url=?",
            (sample_bill["url"],),
        ).fetchone()
        conn.close()
        assert row[0] == sample_bill["url"]
        assert row[1] == sample_bill["title"]
        assert row[3] == "bill"
        assert row[5] == "Risa Hontiveros"
        assert row[6] == 20

    def test_is_idempotent(self, db_path, sample_bill):
        upsert_documents([sample_bill], db_path=db_path)
        upsert_documents([sample_bill], db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 1

    def test_skips_docs_without_url(self, db_path):
        upsert_documents([{"title": "no url", "source_type": "bill", "text": "something"}], db_path=db_path)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 0

    def test_writes_topics_to_db(self, db_path, sample_bill):
        upsert_documents([sample_bill], db_path=db_path)
        conn = sqlite3.connect(db_path)
        topics = {r[0] for r in conn.execute(
            "SELECT topic FROM document_topics WHERE url=?", (sample_bill["url"],)
        ).fetchall()}
        conn.close()
        assert "women's rights" in topics

    def test_handles_empty_list(self, db_path):
        upsert_documents([], db_path=db_path)  # must not raise


class TestTagTopics:
    def test_matches_womens_rights_keywords(self):
        doc = {"title": "Magna Carta of Women", "text": "gender equality and anti-violence against women"}
        topics = tag_topics(doc)
        assert "women's rights" in topics

    def test_matches_education_keywords(self):
        doc = {"title": "Free Tuition Act", "text": "scholarship and CHED programs"}
        topics = tag_topics(doc)
        assert "education" in topics

    def test_matches_health_keywords(self):
        doc = {"title": "Universal Health Care Act", "text": "PhilHealth and hospital coverage"}
        topics = tag_topics(doc)
        assert "health" in topics

    def test_returns_empty_for_no_match(self):
        doc = {"title": "SBN-9999", "text": "miscellaneous provisions"}
        topics = tag_topics(doc)
        assert topics == []

    def test_can_match_multiple_topics(self):
        doc = {"title": "Women in Agriculture Act", "text": "women farmers and agrarian reform"}
        topics = tag_topics(doc)
        assert "women's rights" in topics
        assert "agriculture" in topics

    def test_case_insensitive(self):
        doc = {"title": "EDUCATION REFORM ACT", "text": "SCHOLARSHIP FOR STUDENTS"}
        topics = tag_topics(doc)
        assert "education" in topics

    def test_topic_keywords_dict_has_expected_topics(self):
        expected = {"women's rights", "education", "health", "agriculture", "infrastructure",
                    "anti-corruption", "environment", "labor", "social welfare",
                    "peace and order", "economy", "housing", "youth"}
        assert expected.issubset(set(TOPIC_KEYWORDS.keys()))
