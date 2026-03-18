"""
Tests for the agentic RAG path in rag_chain.py.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from retrieval.rag_chain import PiliPinasRAG, _is_aggregation_query


class TestAggregationDetection:
    def test_detects_most(self):
        assert _is_aggregation_query("Who has the most bills?") is True

    def test_detects_how_many(self):
        assert _is_aggregation_query("How many bills did Hontiveros file?") is True

    def test_detects_rank(self):
        assert _is_aggregation_query("Rank the senators by bills filed") is True

    def test_detects_top(self):
        assert _is_aggregation_query("Top 5 senators who advocated for health") is True

    def test_detects_biggest_advocate(self):
        assert _is_aggregation_query("Who is the biggest advocate for women's rights?") is True

    def test_detects_compare(self):
        assert _is_aggregation_query("Compare Marcos and Robredo on education") is True

    def test_detects_least(self):
        assert _is_aggregation_query("Who filed the least bills?") is True

    def test_simple_lookup_not_aggregation(self):
        assert _is_aggregation_query("What did Senator Padilla file?") is False

    def test_news_query_not_aggregation(self):
        assert _is_aggregation_query("What is the latest news about Sara Duterte?") is False

    def test_profile_query_not_aggregation(self):
        assert _is_aggregation_query("Tell me about Leila de Lima") is False


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, input_dict: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def _make_response(stop_reason: str, blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = blocks
    return resp


class TestAgenticQuery:
    @pytest.fixture
    def rag(self, tmp_path):
        db = tmp_path / "idx.db"
        r = PiliPinasRAG()
        r._agentic_db_path = db
        return r

    @pytest.fixture
    def mock_anthropic(self):
        """Stub the anthropic module so tests work without it installed."""
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            yield mock_module

    def test_routes_aggregation_to_agentic(self, rag):
        with patch.object(rag, "query_agentic") as mock_agentic:
            mock_agentic.return_value = MagicMock(answer="answer", sources=[], chunks_used=1, query="q")
            with patch.object(rag, "retrieve", return_value=[]):
                rag.query("Who filed the most senate bills?")
            mock_agentic.assert_called_once()

    def test_simple_query_does_not_route_to_agentic(self, rag):
        with patch.object(rag, "query_agentic") as mock_agentic:
            with patch.object(rag, "retrieve", return_value=[]):
                rag.query("What did Senator Padilla file?")
            mock_agentic.assert_not_called()

    def test_agentic_loop_executes_tool_and_continues(self, rag, mock_anthropic):
        """Claude calls a tool, gets result, then returns a final answer."""
        from data_ingestion.document_index import init_db
        init_db(rag._agentic_db_path)

        tool_response = _make_response("tool_use", [
            _make_tool_use_block("tu_1", "query_database", {"sql": "SELECT COUNT(*) as n FROM documents"})
        ])
        final_response = _make_response("end_turn", [
            _make_text_block("Hontiveros filed the most bills.")
        ])

        mock_client = mock_anthropic.Anthropic.return_value
        mock_client.messages.create.side_effect = [tool_response, final_response]
        result = rag.query_agentic("Who filed the most senate bills?")

        assert "Hontiveros" in result.answer
        assert mock_client.messages.create.call_count == 2

    def test_agentic_loop_stops_at_max_turns(self, rag, mock_anthropic):
        """If Claude keeps calling tools, stop after MAX_AGENTIC_TURNS."""
        from data_ingestion.document_index import init_db
        init_db(rag._agentic_db_path)

        tool_response = _make_response("tool_use", [
            _make_tool_use_block("tu_1", "query_database", {"sql": "SELECT 1"})
        ])

        mock_client = mock_anthropic.Anthropic.return_value
        mock_client.messages.create.return_value = tool_response
        result = rag.query_agentic("Who filed the most bills?")

        assert result is not None
        assert isinstance(result.answer, str)

    def test_agentic_returns_rag_result(self, rag, mock_anthropic):
        from data_ingestion.document_index import init_db
        init_db(rag._agentic_db_path)

        final_response = _make_response("end_turn", [
            _make_text_block("Senator X filed the most bills about education.")
        ])

        mock_client = mock_anthropic.Anthropic.return_value
        mock_client.messages.create.return_value = final_response
        result = rag.query_agentic("Who is the biggest advocate for education?")

        assert result.answer == "Senator X filed the most bills about education."
        assert isinstance(result.sources, list)
        assert result.chunks_used >= 0
