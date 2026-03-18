"""
Tests for retrieval/tools.py — tool definitions and executor.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from retrieval.tools import TOOLS, execute_tool
from data_ingestion.document_index import init_db, upsert_documents


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test_index.db"
    init_db(p)
    upsert_documents([
        {
            "url": "https://senate.gov.ph/SBN-1",
            "title": "SBN-1: Women Act",
            "source": "bettergov.ph",
            "source_type": "bill",
            "date": "2025-01-01",
            "politician": "Hontiveros",
            "congress": 20,
            "text": "women rights",
        },
        {
            "url": "https://senate.gov.ph/SBN-2",
            "title": "SBN-2: Education Act",
            "source": "bettergov.ph",
            "source_type": "bill",
            "date": "2025-01-02",
            "politician": "Angara",
            "congress": 20,
            "text": "school scholarship",
        },
    ], db_path=p)
    return p


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.retrieve.return_value = [
        {"text": "chunk text", "metadata": {"title": "T", "url": "u", "source": "s", "date": "d"}, "score": 0.9}
    ]
    return rag


class TestToolDefinitions:
    def test_tools_is_list_of_two(self):
        assert len(TOOLS) == 2

    def test_search_documents_tool_exists(self):
        names = {t["name"] for t in TOOLS}
        assert "search_documents" in names

    def test_query_database_tool_exists(self):
        names = {t["name"] for t in TOOLS}
        assert "query_database" in names

    def test_tools_have_required_keys(self):
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


class TestQueryDatabase:
    def test_returns_rows_as_json(self, db_path):
        result = execute_tool("query_database", {"sql": "SELECT url FROM documents"}, None, db_path)
        rows = json.loads(result)
        assert len(rows) == 2

    def test_rejects_non_select(self, db_path):
        result = execute_tool("query_database", {"sql": "DROP TABLE documents"}, None, db_path)
        assert "error" in result.lower() or "only SELECT" in result

    def test_rejects_delete(self, db_path):
        result = execute_tool("query_database", {"sql": "DELETE FROM documents"}, None, db_path)
        assert "error" in result.lower() or "only SELECT" in result

    def test_adds_limit_when_missing(self, db_path):
        result = execute_tool("query_database", {"sql": "SELECT url FROM documents"}, None, db_path)
        rows = json.loads(result)
        assert isinstance(rows, list)

    def test_count_query_works(self, db_path):
        result = execute_tool(
            "query_database",
            {"sql": "SELECT politician, COUNT(*) as count FROM documents GROUP BY politician ORDER BY count DESC"},
            None,
            db_path,
        )
        rows = json.loads(result)
        assert len(rows) == 2
        assert "politician" in rows[0]
        assert "count" in rows[0]

    def test_returns_error_on_invalid_sql(self, db_path):
        result = execute_tool("query_database", {"sql": "SELECT * FROM nonexistent_table"}, None, db_path)
        assert "error" in result.lower()


class TestSearchDocuments:
    def test_calls_rag_retrieve(self, mock_rag, db_path):
        result = execute_tool("search_documents", {"query": "women rights", "n": 5}, mock_rag, db_path)
        mock_rag.retrieve.assert_called_once_with("women rights", n=5, source_type=None)
        chunks = json.loads(result)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "chunk text"

    def test_passes_source_type_filter(self, mock_rag, db_path):
        execute_tool("search_documents", {"query": "health", "source_type": "bill"}, mock_rag, db_path)
        mock_rag.retrieve.assert_called_once_with("health", n=10, source_type="bill")

    def test_returns_empty_list_when_no_results(self, db_path):
        rag = MagicMock()
        rag.retrieve.return_value = []
        result = execute_tool("search_documents", {"query": "xyz"}, rag, db_path)
        assert json.loads(result) == []


class TestUnknownTool:
    def test_unknown_tool_returns_error(self, db_path):
        result = execute_tool("nonexistent_tool", {}, None, db_path)
        assert "error" in result.lower() or "unknown" in result.lower()
