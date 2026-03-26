"""
Tests for data_ingestion/ingestion.py

Scrapers and processors are mocked so no real HTTP calls are made.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import data_ingestion.ingestion as ingestion


@pytest.fixture
def patched_dirs(tmp_path, monkeypatch):
    """Redirect data paths to a temp directory for isolation."""
    monkeypatch.setattr(ingestion, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(ingestion, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(ingestion, "FAILED_LOG", tmp_path / "failed_urls.log")
    return tmp_path


class TestSaveDocuments:
    def test_creates_jsonl_file(self, patched_dirs):
        docs = [
            {"text": "Chunk one.", "url": "https://a.com"},
            {"text": "Chunk two.", "url": "https://b.com"},
        ]
        ingestion.save_documents(docs, "test_collection")

        out = patched_dirs / "processed" / "test_collection.jsonl"
        assert out.exists()

    def test_writes_one_line_per_document(self, patched_dirs):
        docs = [{"text": f"Chunk {i}."} for i in range(5)]
        ingestion.save_documents(docs, "col")
        out = patched_dirs / "processed" / "col.jsonl"
        lines = [l for l in out.read_text().strip().split("\n") if l]
        assert len(lines) == 5

    def test_each_line_is_valid_json(self, patched_dirs):
        docs = [{"text": "Content.", "source": "senate.gov.ph"}]
        ingestion.save_documents(docs, "col")
        out = patched_dirs / "processed" / "col.jsonl"
        for line in out.read_text().strip().split("\n"):
            parsed = json.loads(line)
            assert parsed["text"] == "Content."

    def test_preserves_unicode(self, patched_dirs):
        docs = [{"text": "Ang batas ay para sa lahat."}]
        ingestion.save_documents(docs, "unicode_col")
        out = patched_dirs / "processed" / "unicode_col.jsonl"
        parsed = json.loads(out.read_text().strip())
        assert parsed["text"] == "Ang batas ay para sa lahat."

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        deep_dir = tmp_path / "a" / "b" / "c"
        monkeypatch.setattr(ingestion, "PROCESSED_DIR", deep_dir)
        ingestion.save_documents([{"text": "doc"}], "col")
        assert (deep_dir / "col.jsonl").exists()


class TestUpdateMetadata:
    def test_creates_metadata_file_if_missing(self, patched_dirs):
        ingestion.update_metadata({"sources": ["news"], "total_chunks": 10})
        assert (patched_dirs / "metadata.json").exists()

    def test_written_entry_contains_timestamp(self, patched_dirs):
        ingestion.update_metadata({"total_chunks": 5})
        data = json.loads((patched_dirs / "metadata.json").read_text())
        assert "timestamp" in data[0]

    def test_written_entry_contains_provided_stats(self, patched_dirs):
        ingestion.update_metadata({"total_chunks": 42, "sources": ["news", "gazette"]})
        data = json.loads((patched_dirs / "metadata.json").read_text())
        assert data[0]["total_chunks"] == 42
        assert data[0]["sources"] == ["news", "gazette"]

    def test_appends_to_existing_metadata(self, patched_dirs):
        meta_file = patched_dirs / "metadata.json"
        meta_file.write_text(json.dumps([{"timestamp": "2025-01-01", "total_chunks": 5}]))

        ingestion.update_metadata({"total_chunks": 20})

        data = json.loads(meta_file.read_text())
        assert len(data) == 2
        assert data[0]["total_chunks"] == 5
        assert data[1]["total_chunks"] == 20

    def test_handles_corrupted_existing_metadata(self, patched_dirs):
        meta_file = patched_dirs / "metadata.json"
        meta_file.write_text("not valid json {{{{")

        # Should not raise — starts fresh
        ingestion.update_metadata({"total_chunks": 1})
        data = json.loads(meta_file.read_text())
        assert len(data) == 1


class TestRunIngestion:
    @pytest.fixture
    def mock_scrapers(self, monkeypatch):
        """Patch all scrapers and process_html_document with no-ops."""
        no_docs = MagicMock(return_value=[])
        one_doc = MagicMock(return_value=[
            {"text": "Article text.", "title": "Article", "url": "https://example.com"}
        ])
        one_chunk = MagicMock(return_value=[
            {"text": "Chunk.", "title": "Article", "url": "https://example.com",
             "chunk_index": 0, "chunk_total": 1}
        ])
        monkeypatch.setattr(ingestion, "scrape_senate_bills", no_docs)
        monkeypatch.setattr(ingestion, "scrape_senators", no_docs)
        monkeypatch.setattr(ingestion, "scrape_laws", no_docs)
        monkeypatch.setattr(ingestion, "scrape_house_bills", no_docs)
        monkeypatch.setattr(ingestion, "scrape_members", no_docs)
        monkeypatch.setattr(ingestion, "scrape_all_comelec", no_docs)
        monkeypatch.setattr(ingestion, "scrape_all_news", one_doc)
        monkeypatch.setattr(ingestion, "process_html_document", one_chunk)
        return {
            "scrape_all_news": one_doc,
            "scrape_senate_bills": no_docs,
        }

    def test_news_only_runs_only_news_scraper(self, patched_dirs, mock_scrapers):
        ingestion.run_ingestion(sources=["news"], max_news=5)
        mock_scrapers["scrape_all_news"].assert_called_once_with(max_items_per_source=5)
        mock_scrapers["scrape_senate_bills"].assert_not_called()

    def test_returns_stats_with_counts(self, patched_dirs, mock_scrapers):
        stats = ingestion.run_ingestion(sources=["news"])
        assert "counts" in stats
        assert "news_articles" in stats["counts"]

    def test_returns_total_chunks(self, patched_dirs, mock_scrapers):
        stats = ingestion.run_ingestion(sources=["news"])
        assert stats["total_chunks"] == stats["counts"]["news_articles"]

    def test_saves_jsonl_file_for_each_source(self, patched_dirs, mock_scrapers):
        ingestion.run_ingestion(sources=["news"])
        assert (patched_dirs / "processed" / "news_articles.jsonl").exists()

    def test_updates_metadata_after_ingestion(self, patched_dirs, mock_scrapers):
        ingestion.run_ingestion(sources=["news"])
        assert (patched_dirs / "metadata.json").exists()

    def test_politicians_source_skips_senators_scraper(self, patched_dirs, monkeypatch):
        senators_mock = MagicMock(return_value=[])
        politicians_mock = MagicMock(return_value=[])
        monkeypatch.setattr(ingestion, "scrape_senators", senators_mock)
        monkeypatch.setattr(ingestion, "scrape_all_politicians", politicians_mock)
        monkeypatch.setattr(ingestion, "scrape_senate_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_laws", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_house_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_members", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_all_comelec", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_all_news", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "process_html_document", MagicMock(return_value=[]))

        ingestion.run_ingestion(sources=["senators", "politicians"])
        senators_mock.assert_not_called()
        politicians_mock.assert_called_once()

    def test_politicians_source_skips_house_members_scraper(self, patched_dirs, monkeypatch):
        members_mock = MagicMock(return_value=[])
        politicians_mock = MagicMock(return_value=[])
        monkeypatch.setattr(ingestion, "scrape_members", members_mock)
        monkeypatch.setattr(ingestion, "scrape_all_politicians", politicians_mock)
        monkeypatch.setattr(ingestion, "scrape_senate_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_senators", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_laws", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_house_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_all_comelec", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_all_news", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "process_html_document", MagicMock(return_value=[]))

        ingestion.run_ingestion(sources=["house_members", "politicians"])
        members_mock.assert_not_called()
        politicians_mock.assert_called_once()

    def test_senators_runs_normally_without_politicians(self, patched_dirs, monkeypatch):
        senators_mock = MagicMock(return_value=[])
        monkeypatch.setattr(ingestion, "scrape_senators", senators_mock)
        monkeypatch.setattr(ingestion, "scrape_senate_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "process_html_document", MagicMock(return_value=[]))

        ingestion.run_ingestion(sources=["senators"])
        senators_mock.assert_called_once()

    def test_docs_with_no_text_are_skipped(self, patched_dirs, monkeypatch):
        monkeypatch.setattr(ingestion, "scrape_all_news", MagicMock(return_value=[
            {"title": "No text doc", "url": "https://example.com"}  # no "text" key
        ]))
        monkeypatch.setattr(ingestion, "process_html_document", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_senate_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_senators", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_laws", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_house_bills", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_members", MagicMock(return_value=[]))
        monkeypatch.setattr(ingestion, "scrape_all_comelec", MagicMock(return_value=[]))

        stats = ingestion.run_ingestion(sources=["news"])
        assert stats["counts"].get("news_articles", 0) == 0
