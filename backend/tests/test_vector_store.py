"""
Tests for embeddings/vector_store.py and embeddings/base.py
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from embeddings.base import VectorStore
from embeddings.vector_store import ChromaVectorStore, get_vector_store, COLLECTION_NAME


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_store(collection_name: str = COLLECTION_NAME) -> ChromaVectorStore:
    """Return a ChromaVectorStore with a fully mocked ChromaDB client."""
    mock_collection = MagicMock()
    mock_collection.name = collection_name
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch("embeddings.vector_store.chromadb.PersistentClient", return_value=mock_client), \
         patch("embeddings.vector_store.Path.mkdir"):
        store = ChromaVectorStore(path="/fake/path", collection_name=collection_name)

    return store


# ── VectorStore ABC ───────────────────────────────────────────────────────────

class TestVectorStoreABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_concrete_subclass_must_implement_all_methods(self):
        class Incomplete(VectorStore):
            pass  # missing all abstract methods

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass_can_be_instantiated(self):
        class Minimal(VectorStore):
            @property
            def name(self): return "test"
            def upsert(self, ids, embeddings, documents, metadatas): pass
            def query(self, query_embeddings, n_results=5, where=None): return {}
            def count(self): return 0

        store = Minimal()
        assert store.name == "test"
        assert store.count() == 0


# ── ChromaVectorStore.name ────────────────────────────────────────────────────

class TestChromaVectorStoreName:
    def test_returns_collection_name(self):
        store = _make_store("pili_pinas")
        assert store.name == "pili_pinas"

    def test_uses_cosine_space(self):
        mock_collection = MagicMock()
        mock_collection.name = COLLECTION_NAME
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("embeddings.vector_store.chromadb.PersistentClient", return_value=mock_client), \
             patch("embeddings.vector_store.Path.mkdir"):
            ChromaVectorStore(path="/fake")

        call_kwargs = mock_client.get_or_create_collection.call_args[1]
        assert call_kwargs["metadata"]["hnsw:space"] == "cosine"

    def test_uses_default_collection_name(self):
        mock_collection = MagicMock()
        mock_collection.name = COLLECTION_NAME
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("embeddings.vector_store.chromadb.PersistentClient", return_value=mock_client), \
             patch("embeddings.vector_store.Path.mkdir"):
            ChromaVectorStore(path="/fake")

        call_kwargs = mock_client.get_or_create_collection.call_args[1]
        assert call_kwargs["name"] == COLLECTION_NAME


# ── ChromaVectorStore.upsert ──────────────────────────────────────────────────

class TestChromaVectorStoreUpsert:
    def test_sanitizes_metadata_before_upsert(self):
        store = _make_store()
        metadatas = [{"text": "excluded", "source": "senate.gov.ph", "extra": None}]

        store.upsert(
            ids=["id1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Document text."],
            metadatas=metadatas,
        )

        store._collection.upsert.assert_called_once()
        passed = store._collection.upsert.call_args[1]["metadatas"]
        assert "text" not in passed[0]
        assert passed[0]["source"] == "senate.gov.ph"
        assert passed[0]["extra"] == ""

    def test_passes_ids_and_documents_unchanged(self):
        store = _make_store()
        store.upsert(
            ids=["id1", "id2"],
            embeddings=[[0.1], [0.2]],
            documents=["Doc 1", "Doc 2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )
        kwargs = store._collection.upsert.call_args[1]
        assert kwargs["ids"] == ["id1", "id2"]
        assert kwargs["documents"] == ["Doc 1", "Doc 2"]


# ── ChromaVectorStore.query ───────────────────────────────────────────────────

class TestChromaVectorStoreQuery:
    def test_passes_embeddings_and_n_results(self):
        store = _make_store()
        store.query(query_embeddings=[[0.1, 0.2]], n_results=5)
        kwargs = store._collection.query.call_args[1]
        assert kwargs["query_embeddings"] == [[0.1, 0.2]]
        assert kwargs["n_results"] == 5

    def test_includes_documents_metadatas_distances(self):
        store = _make_store()
        store.query(query_embeddings=[[0.1]], n_results=3)
        kwargs = store._collection.query.call_args[1]
        assert "documents" in kwargs["include"]
        assert "metadatas" in kwargs["include"]
        assert "distances" in kwargs["include"]

    def test_where_filter_passed_when_provided(self):
        store = _make_store()
        store.query(query_embeddings=[[0.1]], n_results=3, where={"source_type": "news"})
        kwargs = store._collection.query.call_args[1]
        assert kwargs["where"] == {"source_type": "news"}

    def test_no_where_key_when_filter_is_none(self):
        store = _make_store()
        store.query(query_embeddings=[[0.1]], n_results=3, where=None)
        kwargs = store._collection.query.call_args[1]
        assert "where" not in kwargs


# ── ChromaVectorStore.count ───────────────────────────────────────────────────

class TestChromaVectorStoreCount:
    def test_delegates_to_collection_count(self):
        store = _make_store()
        store._collection.count.return_value = 42
        assert store.count() == 42


# ── _sanitize_metadata ────────────────────────────────────────────────────────

class TestSanitizeMetadata:
    def test_passes_through_string_values(self):
        result = ChromaVectorStore._sanitize_metadata({"source": "senate.gov.ph"})
        assert result["source"] == "senate.gov.ph"

    def test_passes_through_int_float_bool(self):
        result = ChromaVectorStore._sanitize_metadata({"year": 2025, "score": 0.9, "active": True})
        assert result == {"year": 2025, "score": 0.9, "active": True}

    def test_none_becomes_empty_string(self):
        assert ChromaVectorStore._sanitize_metadata({"x": None})["x"] == ""

    def test_list_converted_to_string(self):
        result = ChromaVectorStore._sanitize_metadata({"tags": ["a", "b"]})
        assert isinstance(result["tags"], str)

    def test_excludes_text_key(self):
        result = ChromaVectorStore._sanitize_metadata({"text": "body", "title": "kept"})
        assert "text" not in result

    def test_excludes_chunk_index_and_chunk_total(self):
        result = ChromaVectorStore._sanitize_metadata({"chunk_index": 0, "chunk_total": 5})
        assert "chunk_index" not in result
        assert "chunk_total" not in result

    def test_empty_dict_returns_empty_dict(self):
        assert ChromaVectorStore._sanitize_metadata({}) == {}


# ── get_vector_store factory ──────────────────────────────────────────────────

class TestGetVectorStore:
    def test_returns_chroma_store_by_default(self, monkeypatch):
        monkeypatch.delenv("VECTOR_STORE_BACKEND", raising=False)
        with patch("embeddings.vector_store.chromadb.PersistentClient"), \
             patch("embeddings.vector_store.Path.mkdir"):
            store = get_vector_store()
        assert isinstance(store, ChromaVectorStore)

    def test_returns_chroma_store_when_explicitly_set(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma")
        with patch("embeddings.vector_store.chromadb.PersistentClient"), \
             patch("embeddings.vector_store.Path.mkdir"):
            store = get_vector_store()
        assert isinstance(store, ChromaVectorStore)

    def test_raises_for_unknown_backend(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_BACKEND", "turso")
        with pytest.raises(ValueError, match="Unknown VECTOR_STORE_BACKEND"):
            get_vector_store()
