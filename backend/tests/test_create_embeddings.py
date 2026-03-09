"""
Tests for embeddings/create_embeddings.py

The SentenceTransformer model and vector store are mocked to avoid loading
the actual ML model or touching ChromaDB.
"""

import json
import hashlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import embeddings.create_embeddings as ce
from embeddings.create_embeddings import (
    doc_id,
    load_jsonl,
    embed_collection,
    run_embedding_pipeline,
    BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# doc_id
# ---------------------------------------------------------------------------

class TestDocId:
    def test_stable_for_same_url_and_index(self):
        doc = {"url": "https://senate.gov.ph/bill/1", "chunk_index": 0}
        assert doc_id(doc, 0) == doc_id(doc, 0)

    def test_different_for_different_chunk_index(self):
        doc = {"url": "https://senate.gov.ph/bill/1", "chunk_index": 0}
        doc2 = {"url": "https://senate.gov.ph/bill/1", "chunk_index": 1}
        assert doc_id(doc, 0) != doc_id(doc2, 1)

    def test_different_for_different_url(self):
        doc_a = {"url": "https://senate.gov.ph/bill/1", "chunk_index": 0}
        doc_b = {"url": "https://senate.gov.ph/bill/2", "chunk_index": 0}
        assert doc_id(doc_a, 0) != doc_id(doc_b, 0)

    def test_returns_md5_hex_string(self):
        doc = {"url": "https://example.com", "chunk_index": 2}
        result = doc_id(doc, 2)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_uses_fallback_idx_when_chunk_index_missing(self):
        doc_a = {"url": "https://example.com"}
        doc_b = {"url": "https://example.com"}
        # idx differs → ids differ
        assert doc_id(doc_a, 0) != doc_id(doc_b, 1)

    def test_empty_doc_does_not_raise(self):
        result = doc_id({}, 0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "Hello"}\n{"text": "World"}\n')
        docs = load_jsonl(f)
        assert len(docs) == 2
        assert docs[0]["text"] == "Hello"
        assert docs[1]["text"] == "World"

    def test_skips_empty_lines(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "A"}\n\n{"text": "B"}\n')
        docs = load_jsonl(f)
        assert len(docs) == 2

    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "valid"}\nnot valid json\n{"text": "also valid"}\n')
        docs = load_jsonl(f)
        assert len(docs) == 2

    def test_returns_empty_list_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_jsonl(f) == []

    def test_preserves_unicode(self, tmp_path):
        f = tmp_path / "ph.jsonl"
        f.write_text('{"text": "Ang batas ay para sa lahat."}\n')
        docs = load_jsonl(f)
        assert docs[0]["text"] == "Ang batas ay para sa lahat."


# ---------------------------------------------------------------------------
# embed_collection
# ---------------------------------------------------------------------------

def _make_model(embedding_dim=3):
    """Fake SentenceTransformer that returns deterministic embeddings."""
    import numpy as np
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), embedding_dim))
    return model


class TestEmbedCollection:
    def test_returns_zero_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        model = _make_model()
        store = MagicMock()
        assert embed_collection(f, model, store) == 0

    def test_returns_chunk_count(self, tmp_path):
        f = tmp_path / "docs.jsonl"
        docs = [{"text": f"Chunk {i}", "url": f"https://example.com/{i}", "chunk_index": i}
                for i in range(5)]
        f.write_text("\n".join(json.dumps(d) for d in docs))
        model = _make_model()
        store = MagicMock()
        assert embed_collection(f, model, store) == 5

    def test_calls_upsert_with_correct_count(self, tmp_path):
        f = tmp_path / "docs.jsonl"
        docs = [{"text": f"Chunk {i}", "url": f"https://x.com", "chunk_index": i}
                for i in range(3)]
        f.write_text("\n".join(json.dumps(d) for d in docs))
        model = _make_model()
        store = MagicMock()
        embed_collection(f, model, store)
        store.upsert.assert_called_once()
        kwargs = store.upsert.call_args[1]
        assert len(kwargs["ids"]) == 3
        assert len(kwargs["documents"]) == 3

    def test_batches_encoding_by_batch_size(self, tmp_path):
        n = BATCH_SIZE + 5
        f = tmp_path / "big.jsonl"
        docs = [{"text": f"T{i}", "url": "https://x.com", "chunk_index": i}
                for i in range(n)]
        f.write_text("\n".join(json.dumps(d) for d in docs))
        model = _make_model()
        store = MagicMock()
        embed_collection(f, model, store)
        # encode should be called at least twice (one full batch + one partial)
        assert model.encode.call_count == 2

    def test_upserts_per_batch_not_all_at_once(self, tmp_path):
        """Each batch should be upserted immediately to avoid holding all embeddings in RAM."""
        n = BATCH_SIZE + 5  # two batches
        f = tmp_path / "big.jsonl"
        docs = [{"text": f"T{i}", "url": f"https://x.com/p{i}", "chunk_index": i}
                for i in range(n)]
        f.write_text("\n".join(json.dumps(d) for d in docs))
        model = _make_model()
        store = MagicMock()
        embed_collection(f, model, store)
        # upsert should be called once per batch, not once for all data
        assert store.upsert.call_count == 2

    def test_ids_are_unique_per_chunk(self, tmp_path):
        f = tmp_path / "docs.jsonl"
        docs = [{"text": f"Chunk {i}", "url": f"https://x.com/p{i}", "chunk_index": i}
                for i in range(4)]
        f.write_text("\n".join(json.dumps(d) for d in docs))
        model = _make_model()
        store = MagicMock()
        embed_collection(f, model, store)
        ids = store.upsert.call_args[1]["ids"]
        assert len(set(ids)) == 4  # all unique


# ---------------------------------------------------------------------------
# run_embedding_pipeline
# ---------------------------------------------------------------------------

class TestRunEmbeddingPipeline:
    @pytest.fixture(autouse=True)
    def patch_deps(self, tmp_path, monkeypatch):
        """Redirect PROCESSED_DIR and mock model + store."""
        monkeypatch.setattr(ce, "PROCESSED_DIR", tmp_path)
        self.tmp_path = tmp_path

        self.mock_model = _make_model()
        self.mock_store = MagicMock()
        self.mock_store.name = "pili_pinas"

        monkeypatch.setattr(ce, "load_model", lambda: self.mock_model)
        monkeypatch.setattr(ce, "get_vector_store", lambda: self.mock_store)

    def _write_jsonl(self, name: str, n: int = 2):
        f = self.tmp_path / f"{name}.jsonl"
        docs = [{"text": f"Chunk {i}", "url": f"https://x.com/{i}", "chunk_index": i}
                for i in range(n)]
        f.write_text("\n".join(json.dumps(d) for d in docs))
        return f

    def test_returns_empty_dict_when_no_jsonl_files(self):
        result = run_embedding_pipeline()
        assert result == {}

    def test_returns_stats_per_collection(self):
        self._write_jsonl("news_articles", n=3)
        result = run_embedding_pipeline()
        assert "news_articles" in result
        assert result["news_articles"] == 3

    def test_processes_multiple_collections(self):
        self._write_jsonl("news_articles", n=2)
        self._write_jsonl("senate_bills", n=4)
        result = run_embedding_pipeline()
        assert result["news_articles"] == 2
        assert result["senate_bills"] == 4

    def test_filters_by_collection_name(self):
        self._write_jsonl("news_articles", n=2)
        self._write_jsonl("senate_bills", n=4)
        result = run_embedding_pipeline(collections=["news_articles"])
        assert "news_articles" in result
        assert "senate_bills" not in result

    def test_upsert_called_for_each_collection(self):
        self._write_jsonl("news_articles", n=2)
        self._write_jsonl("senate_bills", n=3)
        run_embedding_pipeline()
        assert self.mock_store.upsert.call_count == 2
