"""Tests for the keyword-targeted scrape script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from scrape_keyword import filter_by_keyword, scrape_and_filter, SCRAPER_MAP


class TestFilterByKeyword:
    def _doc(self, title="", text=""):
        return {"title": title, "text": text, "source": "test", "source_type": "news"}

    def test_matches_keyword_in_title(self):
        docs = [self._doc(title="Leni Robredo wins case"), self._doc(title="Marcos signs bill")]
        result = filter_by_keyword(docs, "Leni Robredo")
        assert len(result) == 1
        assert result[0]["title"] == "Leni Robredo wins case"

    def test_matches_keyword_in_text(self):
        docs = [self._doc(text="Former VP Leni Robredo said..."), self._doc(text="No mention")]
        result = filter_by_keyword(docs, "Leni Robredo")
        assert len(result) == 1

    def test_case_insensitive(self):
        docs = [self._doc(title="LENI ROBREDO Statement")]
        result = filter_by_keyword(docs, "leni robredo")
        assert len(result) == 1

    def test_no_matches_returns_empty(self):
        docs = [self._doc(title="Marcos signs new law"), self._doc(title="Sara speech")]
        result = filter_by_keyword(docs, "Leni Robredo")
        assert result == []

    def test_empty_docs_returns_empty(self):
        assert filter_by_keyword([], "Leni Robredo") == []

    def test_matches_partial_keyword(self):
        docs = [self._doc(title="Robredo on housing bill")]
        result = filter_by_keyword(docs, "Robredo")
        assert len(result) == 1

    def test_preserves_doc_structure(self):
        doc = self._doc(title="Leni Robredo profile")
        doc["url"] = "https://example.com"
        result = filter_by_keyword([doc], "Leni Robredo")
        assert result[0]["url"] == "https://example.com"

    def test_doc_without_text_field_still_matches_title(self):
        doc = {"title": "Leni Robredo speech", "source": "test", "source_type": "news"}
        result = filter_by_keyword([doc], "Leni Robredo")
        assert len(result) == 1

    def test_doc_without_title_still_matches_text(self):
        doc = {"text": "Leni Robredo spoke today", "source": "test", "source_type": "news"}
        result = filter_by_keyword([doc], "Leni Robredo")
        assert len(result) == 1


class TestScraperMap:
    def test_scraper_map_has_expected_sources(self):
        expected = {"news", "senate_bills", "senators", "gazette",
                    "house_bills", "house_members", "fact_check",
                    "oversight", "statistics", "research", "financial"}
        assert expected.issubset(set(SCRAPER_MAP.keys()))

    def test_all_values_are_callable(self):
        for name, fn in SCRAPER_MAP.items():
            assert callable(fn), f"{name} scraper is not callable"


class TestScrapeAndFilter:
    def _make_docs(self, titles):
        return [{"title": t, "text": "", "source": "test", "source_type": "news"} for t in titles]

    def test_filters_results_from_scraper(self):
        mock_scraper = MagicMock(return_value=self._make_docs([
            "Leni Robredo on housing",
            "Marcos signs bill",
            "Leni Robredo visits Bicol",
        ]))
        scraper_map = {"news": mock_scraper}
        result = scrape_and_filter("Leni Robredo", sources=["news"], scraper_map=scraper_map)
        assert len(result) == 2

    def test_combines_results_from_multiple_scrapers(self):
        scraper_map = {
            "news": MagicMock(return_value=self._make_docs(["Leni Robredo news"])),
            "senate_bills": MagicMock(return_value=self._make_docs(["Leni Robredo bill filed", "Other bill"])),
        }
        result = scrape_and_filter("Leni Robredo", sources=["news", "senate_bills"], scraper_map=scraper_map)
        assert len(result) == 2

    def test_only_runs_requested_sources(self):
        news_mock = MagicMock(return_value=[])
        senate_mock = MagicMock(return_value=[])
        scraper_map = {"news": news_mock, "senate_bills": senate_mock}
        scrape_and_filter("Leni", sources=["news"], scraper_map=scraper_map)
        news_mock.assert_called_once()
        senate_mock.assert_not_called()

    def test_uses_all_sources_when_none_specified(self):
        mock_fn = MagicMock(return_value=[])
        scraper_map = {"news": mock_fn, "senate_bills": mock_fn, "senators": mock_fn}
        scrape_and_filter("Leni", sources=None, scraper_map=scraper_map)
        assert mock_fn.call_count == len(scraper_map)

    def test_skips_unknown_source_gracefully(self):
        scraper_map = {"news": MagicMock(return_value=[])}
        # should not raise
        result = scrape_and_filter("Leni", sources=["news", "nonexistent"], scraper_map=scraper_map)
        assert result == []

    def test_returns_empty_when_no_matches(self):
        scraper_map = {"news": MagicMock(return_value=self._make_docs(["Marcos signs bill"]))}
        result = scrape_and_filter("Leni Robredo", sources=["news"], scraper_map=scraper_map)
        assert result == []
