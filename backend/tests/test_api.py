"""
Tests for api/main.py

Uses FastAPI's TestClient. Heavy dependencies (RAG, vector store, scrapers)
are mocked. The auth dependency is bypassed via app.dependency_overrides.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api.main as main_module
from api.main import app, _jobs, _run_scrape_job, ScrapeRequest
from api.auth import verify_api_key
from retrieval.rag_chain import RAGResult


@pytest.fixture(autouse=True)
def clear_jobs():
    """Reset job store and bypass auth before each test."""
    _jobs.clear()
    app.dependency_overrides[verify_api_key] = lambda: "test-api-key"
    yield
    _jobs.clear()
    app.dependency_overrides.clear()


client = TestClient(app)


def _mock_rag(answer="Test answer.", sources=None):
    if sources is None:
        sources = [{
            "title": "SB 1234",
            "url": "https://senate.gov.ph/1",
            "source": "senate.gov.ph",
            "date": "2025-01-01",
            "score": 0.85,
        }]
    mock = MagicMock()
    mock.query.return_value = RAGResult(
        answer=answer,
        sources=sources,
        query="test question",
        chunks_used=len(sources),
    )
    return mock


def _mock_rag_no_answer():
    """RAG mock that returns no chunks (question cannot be answered)."""
    mock = MagicMock()
    mock.query.return_value = RAGResult(
        answer="Hindi ako makahanap...",
        sources=[],
        query="test question",
        chunks_used=0,
    )
    return mock


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_returns_ok_status(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_returns_service_name(self):
        assert client.get("/health").json()["service"] == "pili-pinas-api"


class TestStats:
    def test_returns_200(self):
        mock_store = MagicMock()
        mock_store.name = "pili_pinas"
        mock_store.count.return_value = 0
        with patch("api.main.get_vector_store", return_value=mock_store):
            assert client.get("/stats").status_code == 200

    def test_returns_collection_name_and_chunk_count(self):
        mock_store = MagicMock()
        mock_store.name = "pili_pinas"
        mock_store.count.return_value = 42
        with patch("api.main.get_vector_store", return_value=mock_store):
            data = client.get("/stats").json()
        assert data["collection"] == "pili_pinas"
        assert data["total_chunks"] == 42

    def test_returns_500_when_store_fails(self):
        with patch("api.main.get_vector_store", side_effect=Exception("DB error")):
            assert client.get("/stats").status_code == 500


class TestQuery:
    def test_valid_question_returns_200(self):
        with patch("api.main.get_rag", return_value=_mock_rag()):
            resp = client.post("/query", json={"question": "What education bills passed?"})
        assert resp.status_code == 200

    def test_response_contains_answer_and_sources(self):
        with patch("api.main.get_rag", return_value=_mock_rag("Bills were passed.")):
            data = client.post("/query", json={"question": "What education bills passed?"}).json()
        assert data["answer"] == "Bills were passed."
        assert len(data["sources"]) == 1
        assert data["chunks_used"] == 1

    def test_question_too_short_returns_422(self):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_top_k_is_applied_to_rag_instance(self):
        mock_rag = _mock_rag()
        with patch("api.main.get_rag", return_value=mock_rag):
            client.post("/query", json={"question": "Test question here", "top_k": 10})
        assert mock_rag.top_k == 10

    def test_source_type_filter_passed_to_rag(self):
        mock_rag = _mock_rag()
        with patch("api.main.get_rag", return_value=mock_rag):
            client.post("/query", json={"question": "Test question here", "source_type": "news"})
        mock_rag.query.assert_called_once_with(question="Test question here", source_type="news")

    def test_top_k_defaults_to_5(self):
        mock_rag = _mock_rag()
        with patch("api.main.get_rag", return_value=mock_rag):
            client.post("/query", json={"question": "Test question here"})
        assert mock_rag.top_k == 5

    def test_rag_exception_returns_500(self):
        mock_rag = MagicMock()
        mock_rag.query.side_effect = RuntimeError("RAG failure")
        with patch("api.main.get_rag", return_value=mock_rag):
            resp = client.post("/query", json={"question": "Test question here"})
        assert resp.status_code == 500

    def test_unanswered_question_is_logged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_module, "UNANSWERED_DB", tmp_path / "unanswered.db")
        with patch("api.main.get_rag", return_value=_mock_rag_no_answer()):
            client.post("/query", json={"question": "Sino si Leni Robredo?"})
        resp = client.get("/unanswered")
        assert resp.status_code == 200
        rows = resp.json()["questions"]
        assert len(rows) == 1
        assert rows[0]["question"] == "Sino si Leni Robredo?"

    def test_answered_question_is_not_logged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_module, "UNANSWERED_DB", tmp_path / "unanswered.db")
        with patch("api.main.get_rag", return_value=_mock_rag()):
            client.post("/query", json={"question": "What bills were passed?"})
        resp = client.get("/unanswered")
        assert resp.json()["questions"] == []

    def test_unanswered_logs_source_type(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main_module, "UNANSWERED_DB", tmp_path / "unanswered.db")
        with patch("api.main.get_rag", return_value=_mock_rag_no_answer()):
            client.post("/query", json={"question": "Ano ang SALN ni Marcos?", "source_type": "saln"})
        rows = client.get("/unanswered").json()["questions"]
        assert rows[0]["source_type"] == "saln"


class TestScrape:
    def test_trigger_returns_202(self):
        with patch("api.main._run_scrape_job"):
            resp = client.post("/scrape", json={})
        assert resp.status_code == 202

    def test_trigger_returns_job_id_and_pending_status(self):
        with patch("api.main._run_scrape_job"):
            data = client.post("/scrape", json={}).json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_trigger_with_valid_sources(self):
        with patch("api.main._run_scrape_job"):
            resp = client.post("/scrape", json={"sources": ["news", "senators"]})
        assert resp.status_code == 202

    def test_invalid_source_returns_422(self):
        resp = client.post("/scrape", json={"sources": ["not_a_real_source"]})
        assert resp.status_code == 422
        assert "Invalid sources" in resp.json()["detail"]

    def test_mix_of_valid_and_invalid_sources_returns_422(self):
        resp = client.post("/scrape", json={"sources": ["news", "fake_source"]})
        assert resp.status_code == 422

    def test_all_valid_sources_accepted(self):
        all_sources = ["senate_bills", "senators", "gazette", "house_bills",
                       "house_members", "comelec", "news"]
        with patch("api.main._run_scrape_job"):
            resp = client.post("/scrape", json={"sources": all_sources})
        assert resp.status_code == 202

    def test_null_sources_triggers_full_scrape(self):
        with patch("api.main._run_scrape_job"):
            resp = client.post("/scrape", json={"sources": None})
        assert resp.status_code == 202

    def test_job_added_to_store(self):
        with patch("api.main._run_scrape_job"):
            job_id = client.post("/scrape", json={}).json()["job_id"]
        assert job_id in _jobs

    def test_background_task_updates_job_status(self):
        """When background task runs, job status transitions from pending."""
        def fake_job(job_id, req):
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["started_at"] = "2025-01-01T00:00:00"
            _jobs[job_id]["finished_at"] = "2025-01-01T00:01:00"
            _jobs[job_id]["stats"] = {"ingestion": {"total_chunks": 5}}

        with patch("api.main._run_scrape_job", side_effect=fake_job):
            job_id = client.post("/scrape", json={"sources": ["news"]}).json()["job_id"]

        # TestClient runs background tasks synchronously
        assert _jobs[job_id]["status"] == "done"


class TestScrapeStatus:
    def test_unknown_job_id_returns_404(self):
        resp = client.get("/scrape/nonexistent-job-id")
        assert resp.status_code == 404

    def test_known_job_returns_200(self):
        _jobs["test-job"] = {
            "status": "running",
            "started_at": "2025-01-01T00:00:00",
            "finished_at": None,
            "stats": None,
            "error": None,
        }
        assert client.get("/scrape/test-job").status_code == 200

    def test_returns_correct_job_id(self):
        _jobs["abc-123"] = {
            "status": "done",
            "started_at": "2025-01-01T00:00:00",
            "finished_at": "2025-01-01T00:05:00",
            "stats": {"ingestion": {"total_chunks": 80}},
            "error": None,
        }
        data = client.get("/scrape/abc-123").json()
        assert data["job_id"] == "abc-123"

    def test_returns_full_stats_when_done(self):
        _jobs["done-job"] = {
            "status": "done",
            "started_at": "2025-01-01T00:00:00",
            "finished_at": "2025-01-01T00:05:00",
            "stats": {"ingestion": {"total_chunks": 100}, "embedding": {"news_articles": 100}},
            "error": None,
        }
        data = client.get("/scrape/done-job").json()
        assert data["status"] == "done"
        assert data["stats"]["ingestion"]["total_chunks"] == 100

    def test_returns_error_message_when_failed(self):
        _jobs["failed-job"] = {
            "status": "failed",
            "started_at": "2025-01-01T00:00:00",
            "finished_at": "2025-01-01T00:01:00",
            "stats": None,
            "error": "Connection refused",
        }
        data = client.get("/scrape/failed-job").json()
        assert data["status"] == "failed"
        assert data["error"] == "Connection refused"


# ---------------------------------------------------------------------------
# _run_scrape_job (background task — called directly, not via HTTP)
# ---------------------------------------------------------------------------

class TestRunScrapeJob:
    def _make_job(self):
        job_id = "test-job-123"
        _jobs[job_id] = {
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "stats": None,
            "error": None,
        }
        return job_id

    def _default_req(self, **kwargs):
        return ScrapeRequest(**kwargs)

    def test_sets_status_to_running_then_done(self):
        job_id = self._make_job()
        # run_ingestion/run_embedding_pipeline are lazy-imported inside _run_scrape_job
        with patch("data_ingestion.ingestion.run_ingestion", return_value={"total_chunks": 5, "counts": {}}), \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value={}):
            _run_scrape_job(job_id, self._default_req(embed=True))
        assert _jobs[job_id]["status"] == "done"

    def test_sets_started_at_and_finished_at(self):
        job_id = self._make_job()
        with patch("data_ingestion.ingestion.run_ingestion", return_value={"total_chunks": 0, "counts": {}}), \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value={}):
            _run_scrape_job(job_id, self._default_req())
        assert _jobs[job_id]["started_at"] is not None
        assert _jobs[job_id]["finished_at"] is not None

    def test_stores_ingestion_stats(self):
        job_id = self._make_job()
        ingest_stats = {"total_chunks": 42, "counts": {"news_articles": 42}}
        with patch("data_ingestion.ingestion.run_ingestion", return_value=ingest_stats), \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value={}):
            _run_scrape_job(job_id, self._default_req(embed=True))
        assert _jobs[job_id]["stats"]["ingestion"]["total_chunks"] == 42

    def test_stores_embedding_stats_when_embed_true(self):
        job_id = self._make_job()
        embed_stats = {"news_articles": 42}
        with patch("data_ingestion.ingestion.run_ingestion", return_value={"total_chunks": 42, "counts": {}}), \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value=embed_stats):
            _run_scrape_job(job_id, self._default_req(embed=True))
        assert _jobs[job_id]["stats"]["embedding"] == embed_stats

    def test_skips_embedding_when_embed_false(self):
        job_id = self._make_job()
        with patch("data_ingestion.ingestion.run_ingestion", return_value={"total_chunks": 0, "counts": {}}), \
             patch("embeddings.create_embeddings.run_embedding_pipeline") as mock_embed:
            _run_scrape_job(job_id, self._default_req(embed=False))
        mock_embed.assert_not_called()
        assert "embedding" not in (_jobs[job_id]["stats"] or {})

    def test_sets_status_failed_on_ingestion_error(self):
        job_id = self._make_job()
        with patch("data_ingestion.ingestion.run_ingestion", side_effect=RuntimeError("DB error")):
            _run_scrape_job(job_id, self._default_req())
        assert _jobs[job_id]["status"] == "failed"
        assert "DB error" in _jobs[job_id]["error"]

    def test_still_sets_finished_at_on_failure(self):
        job_id = self._make_job()
        with patch("data_ingestion.ingestion.run_ingestion", side_effect=RuntimeError("crash")):
            _run_scrape_job(job_id, self._default_req())
        assert _jobs[job_id]["finished_at"] is not None

    def test_embed_only_runs_on_scraped_collections(self):
        """Embedding pipeline receives only the collections just scraped, not all historical data."""
        job_id = self._make_job()
        ingest_stats = {"total_chunks": 30, "counts": {"news_articles": 25, "senate_bills": 5}}
        with patch("data_ingestion.ingestion.run_ingestion", return_value=ingest_stats), \
             patch("embeddings.create_embeddings.run_embedding_pipeline") as mock_embed:
            mock_embed.return_value = {}
            _run_scrape_job(job_id, self._default_req(embed=True))
        called_collections = set(mock_embed.call_args[1]["collections"])
        assert called_collections == {"news_articles", "senate_bills"}

    # ── Checkpoint / resume ──────────────────────────────────────────────────

    def test_checkpoints_sources_done_after_each_source(self, tmp_path, monkeypatch):
        """Progress file is updated after each source is scraped."""
        monkeypatch.setattr(main_module, "PROGRESS_FILE", tmp_path / "progress.json")
        job_id = self._make_job()

        def per_source_ingest(sources, **kwargs):
            return {"counts": {f"{sources[0]}_collection": 5}, "total_chunks": 5}

        with patch("data_ingestion.ingestion.run_ingestion", side_effect=per_source_ingest), \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value={}):
            _run_scrape_job(job_id, ScrapeRequest(sources=["news", "senate_bills"], embed=False))

        progress = json.loads((tmp_path / "progress.json").read_text())
        assert set(progress["sources_done"]) == {"news", "senate_bills"}
        assert progress["status"] == "done"

    def test_resume_skips_completed_sources(self, tmp_path, monkeypatch):
        """With resume=True, sources already in progress file are not re-scraped."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(main_module, "PROGRESS_FILE", progress_file)
        progress_file.write_text(json.dumps({
            "job_id": "old-job",
            "sources_requested": ["news", "senate_bills"],
            "sources_done": ["news"],
            "status": "running",
        }))

        job_id = self._make_job()
        with patch("data_ingestion.ingestion.run_ingestion",
                   return_value={"counts": {"senate_bills": 3}, "total_chunks": 3}) as mock_ingest, \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value={}):
            _run_scrape_job(job_id, ScrapeRequest(sources=["news", "senate_bills"], resume=True, embed=False))

        # run_ingestion called only once — for senate_bills, not news
        assert mock_ingest.call_count == 1
        assert mock_ingest.call_args[1]["sources"] == ["senate_bills"]

    def test_resume_false_ignores_existing_progress(self, tmp_path, monkeypatch):
        """Without resume=True, a fresh run always scrapes all requested sources."""
        progress_file = tmp_path / "progress.json"
        monkeypatch.setattr(main_module, "PROGRESS_FILE", progress_file)
        progress_file.write_text(json.dumps({
            "sources_done": ["news", "senate_bills"],
            "status": "running",
        }))

        job_id = self._make_job()
        with patch("data_ingestion.ingestion.run_ingestion",
                   return_value={"counts": {"news_articles": 5}, "total_chunks": 5}) as mock_ingest, \
             patch("embeddings.create_embeddings.run_embedding_pipeline", return_value={}):
            _run_scrape_job(job_id, ScrapeRequest(sources=["news", "senate_bills"], resume=False, embed=False))

        assert mock_ingest.call_count == 2  # both sources scraped from scratch
