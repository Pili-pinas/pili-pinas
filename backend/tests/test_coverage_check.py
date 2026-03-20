"""Tests for the coverage check script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from check_coverage import check_keyword, audit_keywords, CoverageResult


class TestCheckKeyword:
    def _make_store(self, docs=None, metas=None):
        store = MagicMock()
        store._collection.get.return_value = {
            "documents": docs or [],
            "metadatas": metas or [],
        }
        return store

    def test_returns_coverage_result(self):
        store = self._make_store(
            docs=["Leni Robredo is a former VP..."],
            metas=[{"source": "rappler.com", "title": "Leni profile", "source_type": "news"}],
        )
        result = check_keyword("Leni Robredo", store)
        assert isinstance(result, CoverageResult)

    def test_found_keyword(self):
        store = self._make_store(
            docs=["Leni Robredo is a former VP..."],
            metas=[{"source": "rappler.com", "title": "Leni profile", "source_type": "news"}],
        )
        result = check_keyword("Leni Robredo", store)
        assert result.keyword == "Leni Robredo"
        assert result.count == 1
        assert result.found is True

    def test_missing_keyword(self):
        store = self._make_store(docs=[], metas=[])
        result = check_keyword("Leni Robredo", store)
        assert result.found is False
        assert result.count == 0
        assert result.samples == []

    def test_samples_capped_at_three(self):
        docs = [f"doc {i}" for i in range(10)]
        metas = [{"source": f"src{i}", "title": f"title{i}", "source_type": "news"} for i in range(10)]
        store = self._make_store(docs=docs, metas=metas)
        result = check_keyword("keyword", store)
        assert len(result.samples) <= 3

    def test_sample_contains_title_and_source(self):
        store = self._make_store(
            docs=["some text"],
            metas=[{"source": "rappler.com", "title": "Leni wins", "source_type": "news"}],
        )
        result = check_keyword("Leni", store)
        assert result.samples[0]["title"] == "Leni wins"
        assert result.samples[0]["source"] == "rappler.com"
        assert result.samples[0]["source_type"] == "news"

    def test_case_insensitive(self):
        """where_document $contains is case-sensitive in Chroma — check_keyword lowercases query."""
        store = self._make_store(docs=[], metas=[])
        check_keyword("LENI ROBREDO", store)
        call_kwargs = store._collection.get.call_args[1]
        assert "leni robredo" in str(call_kwargs).lower()


class TestAuditKeywords:
    def _make_store(self, keyword_doc_map):
        """Return a mock store where .get() returns docs based on the $contains value."""
        store = MagicMock()

        def fake_get(where_document=None, limit=None, include=None):
            term = where_document.get("$contains", "").lower()
            docs = keyword_doc_map.get(term, [])
            metas = [{"source": "test", "title": f"doc about {term}", "source_type": "news"}
                     for _ in docs]
            return {"documents": docs, "metadatas": metas}

        store._collection.get.side_effect = fake_get
        return store

    def test_returns_result_per_keyword(self):
        store = self._make_store({"leni robredo": ["some doc"], "bongbong marcos": []})
        results = audit_keywords(["Leni Robredo", "Bongbong Marcos"], store)
        assert len(results) == 2

    def test_found_and_missing_counts(self):
        store = self._make_store({"leni robredo": ["doc1", "doc2"], "sara duterte": []})
        results = audit_keywords(["Leni Robredo", "Sara Duterte"], store)
        found = [r for r in results if r.found]
        missing = [r for r in results if not r.found]
        assert len(found) == 1
        assert len(missing) == 1

    def test_empty_keyword_list(self):
        store = self._make_store({})
        results = audit_keywords([], store)
        assert results == []
