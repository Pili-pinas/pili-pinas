"""
Tests for retrieval/rag_chain.py

External dependencies (embedding model, ChromaDB, Anthropic API) are all mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.rag_chain import PiliPinasRAG, RAGResult, get_rag
from retrieval.prompts import NO_CONTEXT_RESPONSE


def _make_chunk(text="Content.", score=0.8, title="Doc", url="https://example.com"):
    return {
        "text": text,
        "metadata": {
            "title": title,
            "source": "senate.gov.ph",
            "date": "2025-01-01",
            "url": url,
        },
        "score": score,
    }


class TestBuildContext:
    def setup_method(self):
        # Bypass __init__ — we only need _build_context
        self.rag = PiliPinasRAG.__new__(PiliPinasRAG)

    def test_single_chunk_includes_title_source_date(self):
        chunks = [_make_chunk(title="RA 12345", url="https://senate.gov.ph/ra/12345")]
        result = self.rag._build_context(chunks)
        assert "[1] RA 12345 | senate.gov.ph | 2025-01-01" in result

    def test_single_chunk_includes_url(self):
        chunks = [_make_chunk(url="https://senate.gov.ph/bill/123")]
        result = self.rag._build_context(chunks)
        assert "URL: https://senate.gov.ph/bill/123" in result

    def test_single_chunk_includes_text(self):
        chunks = [_make_chunk(text="Education funding was increased.")]
        result = self.rag._build_context(chunks)
        assert "Education funding was increased." in result

    def test_multiple_chunks_are_numbered(self):
        chunks = [_make_chunk(title="Doc A"), _make_chunk(title="Doc B")]
        result = self.rag._build_context(chunks)
        assert "[1]" in result
        assert "[2]" in result

    def test_multiple_chunks_separated_by_divider(self):
        chunks = [_make_chunk(), _make_chunk()]
        result = self.rag._build_context(chunks)
        assert "---" in result

    def test_missing_metadata_uses_defaults(self):
        chunks = [{"text": "Content.", "metadata": {}, "score": 0.5}]
        result = self.rag._build_context(chunks)
        assert "Untitled" in result
        assert "URL: N/A" in result


class TestQuery:
    def setup_method(self):
        self.rag = PiliPinasRAG.__new__(PiliPinasRAG)
        self.rag.top_k = 5
        self.rag._embedding_model = None
        self.rag._collection = None

    def test_returns_no_context_when_no_chunks_retrieved(self):
        self.rag.retrieve = MagicMock(return_value=[])
        result = self.rag.query("Sino si Marcos?")
        assert result.answer == NO_CONTEXT_RESPONSE
        assert result.sources == []
        assert result.chunks_used == 0

    def test_returns_no_context_when_all_chunks_below_min_score(self):
        low_score_chunks = [_make_chunk(score=0.1), _make_chunk(score=0.2)]
        self.rag.retrieve = MagicMock(return_value=low_score_chunks)
        result = self.rag.query("question", min_score=0.3)
        assert result.answer == NO_CONTEXT_RESPONSE
        assert result.chunks_used == 0

    def test_filters_out_low_score_chunks(self):
        chunks = [_make_chunk(score=0.8), _make_chunk(score=0.1)]
        self.rag.retrieve = MagicMock(return_value=chunks)
        self.rag._call_llm = MagicMock(return_value="Answer.")
        result = self.rag.query("question", min_score=0.3)
        assert result.chunks_used == 1  # only the 0.8 chunk

    def test_calls_llm_when_relevant_chunks_found(self):
        self.rag.retrieve = MagicMock(return_value=[_make_chunk(score=0.8)])
        self.rag._call_llm = MagicMock(return_value="Senate approved the bill.")
        result = self.rag.query("What bills passed?")
        assert result.answer == "Senate approved the bill."
        self.rag._call_llm.assert_called_once()

    def test_returns_correct_chunks_used_count(self):
        chunks = [_make_chunk(score=0.9), _make_chunk(score=0.7), _make_chunk(score=0.1)]
        self.rag.retrieve = MagicMock(return_value=chunks)
        self.rag._call_llm = MagicMock(return_value="Answer.")
        result = self.rag.query("question", min_score=0.3)
        assert result.chunks_used == 2

    def test_sources_contain_expected_fields(self):
        chunk = _make_chunk(score=0.75, title="SB 1234", url="https://senate.gov.ph/1")
        self.rag.retrieve = MagicMock(return_value=[chunk])
        self.rag._call_llm = MagicMock(return_value="Answer.")
        result = self.rag.query("question")
        source = result.sources[0]
        assert source["title"] == "SB 1234"
        assert source["url"] == "https://senate.gov.ph/1"
        assert source["score"] == 0.75

    def test_score_is_rounded_to_3_decimal_places(self):
        chunk = _make_chunk(score=0.123456789)
        self.rag.retrieve = MagicMock(return_value=[chunk])
        self.rag._call_llm = MagicMock(return_value="Answer.")
        result = self.rag.query("question")
        assert result.sources[0]["score"] == 0.123

    def test_query_is_stored_in_result(self):
        self.rag.retrieve = MagicMock(return_value=[])
        result = self.rag.query("Ano ang trabaho ni Leni?")
        assert result.query == "Ano ang trabaho ni Leni?"

    def test_passes_source_type_to_retrieve(self):
        self.rag.retrieve = MagicMock(return_value=[])
        self.rag.query("question", source_type="news")
        self.rag.retrieve.assert_called_once_with("question", source_type="news")


class TestRetrieve:
    def test_converts_distance_to_similarity_score(self, mock_chroma_results):
        rag = PiliPinasRAG.__new__(PiliPinasRAG)
        rag.top_k = 5
        rag._embedding_model = None
        rag._collection = None

        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2]])

        with patch.object(rag, "_get_embedding_model", return_value=mock_model), \
             patch.object(rag, "_get_collection") as mock_coll_fn:
            mock_collection = MagicMock()
            mock_collection.query.return_value = mock_chroma_results
            mock_coll_fn.return_value = mock_collection

            # Import and patch query_collection
            with patch("retrieval.rag_chain.query_collection", return_value=mock_chroma_results):
                chunks = rag.retrieve("test question")

        # distance 0.1 → score 0.9, distance 0.25 → score 0.75
        assert len(chunks) == 2
        assert abs(chunks[0]["score"] - 0.9) < 0.01
        assert abs(chunks[1]["score"] - 0.75) < 0.01

    def test_returns_empty_list_when_no_results(self):
        rag = PiliPinasRAG.__new__(PiliPinasRAG)
        rag.top_k = 5
        rag._embedding_model = None
        rag._collection = None

        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1]])

        empty_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        with patch.object(rag, "_get_embedding_model", return_value=mock_model), \
             patch.object(rag, "_get_collection"):
            with patch("retrieval.rag_chain.query_collection", return_value=empty_results):
                chunks = rag.retrieve("question")

        assert chunks == []


class TestGetRagSingleton:
    def test_returns_same_instance_on_repeated_calls(self):
        # Reset singleton first
        import retrieval.rag_chain as rc
        rc._rag_instance = None

        rag1 = get_rag()
        rag2 = get_rag()
        assert rag1 is rag2

        rc._rag_instance = None  # cleanup
