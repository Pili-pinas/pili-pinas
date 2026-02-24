"""
Tests for embeddings/vector_store.py

_sanitize_metadata is a pure function; the rest mock ChromaDB.
"""

import pytest
from unittest.mock import MagicMock, patch
from embeddings.vector_store import (
    _sanitize_metadata,
    get_or_create_collection,
    upsert_documents,
    query_collection,
)


class TestSanitizeMetadata:
    def test_passes_through_string_values(self):
        meta = {"source": "senate.gov.ph", "title": "SB 1234"}
        result = _sanitize_metadata(meta)
        assert result["source"] == "senate.gov.ph"
        assert result["title"] == "SB 1234"

    def test_passes_through_int_values(self):
        result = _sanitize_metadata({"year": 2025})
        assert result["year"] == 2025

    def test_passes_through_float_values(self):
        result = _sanitize_metadata({"score": 0.95})
        assert result["score"] == 0.95

    def test_passes_through_bool_values(self):
        result = _sanitize_metadata({"active": True})
        assert result["active"] is True

    def test_none_becomes_empty_string(self):
        result = _sanitize_metadata({"politician": None})
        assert result["politician"] == ""

    def test_list_converted_to_string(self):
        result = _sanitize_metadata({"tags": ["education", "law"]})
        assert isinstance(result["tags"], str)

    def test_dict_value_converted_to_string(self):
        result = _sanitize_metadata({"nested": {"key": "val"}})
        assert isinstance(result["nested"], str)

    def test_excludes_text_key(self):
        result = _sanitize_metadata({"text": "document body", "title": "kept"})
        assert "text" not in result
        assert result["title"] == "kept"

    def test_excludes_chunk_index(self):
        result = _sanitize_metadata({"chunk_index": 0, "title": "kept"})
        assert "chunk_index" not in result
        assert result["title"] == "kept"

    def test_excludes_chunk_total(self):
        result = _sanitize_metadata({"chunk_total": 5, "title": "kept"})
        assert "chunk_total" not in result
        assert result["title"] == "kept"

    def test_empty_metadata_returns_empty_dict(self):
        assert _sanitize_metadata({}) == {}

    def test_mixed_types_all_handled(self):
        meta = {
            "source": "rappler.com",
            "date": "2025-01-01",
            "rank": 1,
            "score": 0.8,
            "text": "excluded",
            "tags": ["politics"],
            "extra": None,
        }
        result = _sanitize_metadata(meta)
        assert "text" not in result
        assert result["source"] == "rappler.com"
        assert result["extra"] == ""
        assert isinstance(result["tags"], str)


class TestGetOrCreateCollection:
    def test_calls_get_or_create_with_cosine_space(self):
        mock_client = MagicMock()
        get_or_create_collection(mock_client)
        mock_client.get_or_create_collection.assert_called_once()
        call_kwargs = mock_client.get_or_create_collection.call_args[1]
        assert call_kwargs["metadata"]["hnsw:space"] == "cosine"

    def test_uses_default_collection_name(self):
        mock_client = MagicMock()
        get_or_create_collection(mock_client)
        call_kwargs = mock_client.get_or_create_collection.call_args[1]
        assert call_kwargs["name"] == "pili_pinas"

    def test_custom_collection_name(self):
        mock_client = MagicMock()
        get_or_create_collection(mock_client, name="custom_collection")
        call_kwargs = mock_client.get_or_create_collection.call_args[1]
        assert call_kwargs["name"] == "custom_collection"


class TestUpsertDocuments:
    def test_sanitizes_metadata_before_upsert(self):
        mock_collection = MagicMock()
        metadatas = [{"text": "excluded", "source": "senate.gov.ph", "extra": None}]

        upsert_documents(
            collection=mock_collection,
            ids=["id1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Document text."],
            metadatas=metadatas,
        )

        mock_collection.upsert.assert_called_once()
        passed_metadatas = mock_collection.upsert.call_args[1]["metadatas"]
        assert "text" not in passed_metadatas[0]
        assert passed_metadatas[0]["source"] == "senate.gov.ph"
        assert passed_metadatas[0]["extra"] == ""

    def test_upsert_called_with_correct_args(self):
        mock_collection = MagicMock()
        upsert_documents(
            collection=mock_collection,
            ids=["id1", "id2"],
            embeddings=[[0.1], [0.2]],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )

        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["id1", "id2"]
        assert call_kwargs["documents"] == ["Doc 1", "Doc 2"]


class TestQueryCollection:
    def test_passes_query_embeddings_and_n_results(self):
        mock_collection = MagicMock()
        query_collection(
            collection=mock_collection,
            query_embeddings=[[0.1, 0.2]],
            n_results=5,
        )
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["query_embeddings"] == [[0.1, 0.2]]
        assert call_kwargs["n_results"] == 5

    def test_includes_documents_metadatas_distances(self):
        mock_collection = MagicMock()
        query_collection(mock_collection, query_embeddings=[[0.1]], n_results=3)
        call_kwargs = mock_collection.query.call_args[1]
        assert "documents" in call_kwargs["include"]
        assert "metadatas" in call_kwargs["include"]
        assert "distances" in call_kwargs["include"]

    def test_where_filter_included_when_provided(self):
        mock_collection = MagicMock()
        query_collection(
            mock_collection,
            query_embeddings=[[0.1]],
            n_results=3,
            where={"source_type": "news"},
        )
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"source_type": "news"}

    def test_no_where_filter_when_none(self):
        mock_collection = MagicMock()
        query_collection(mock_collection, query_embeddings=[[0.1]], n_results=3, where=None)
        call_kwargs = mock_collection.query.call_args[1]
        assert "where" not in call_kwargs
