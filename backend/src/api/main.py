"""
Pili-Pinas FastAPI backend.

Endpoints:
    POST /query           — RAG query (main endpoint)
    GET  /health          — health check
    GET  /stats           — ChromaDB collection stats
    POST /scrape          — trigger on-demand scrape job (async, returns job_id)
    GET  /scrape/{job_id} — poll scrape job status

Run:
    uvicorn backend.src.api.main:app --reload
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Allow imports from backend/src
sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retrieval.rag_chain import get_rag
from embeddings.vector_store import get_chroma_client, get_or_create_collection

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pili-Pinas API",
    description="AI-powered Philippine voter information tool",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory job store ───────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}

VALID_SOURCES = [
    "senate_bills", "senators", "gazette",
    "house_bills", "house_members", "comelec", "news",
]


# ── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="User's question")
    source_type: str | None = Field(
        None,
        description="Optional filter: bill | law | news | profile | saln | election"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceDoc(BaseModel):
    title: str
    url: str
    source: str
    date: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    query: str
    chunks_used: int


class ScrapeRequest(BaseModel):
    sources: list[str] | None = Field(
        None,
        description=(
            "Sources to scrape. None = all. "
            "Options: senate_bills, senators, gazette, "
            "house_bills, house_members, comelec, news"
        ),
    )
    congress: int = Field(19, ge=17, le=25, description="Congress number for bill scrapers")
    max_pages: int = Field(3, ge=1, le=20, description="Max pages to scrape per source")
    max_news: int = Field(20, ge=1, le=200, description="Max articles per news source")
    embed: bool = Field(True, description="Auto-run embedding pipeline after ingestion")


class ScrapeJobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | failed
    started_at: str | None = None
    finished_at: str | None = None
    stats: dict | None = None
    error: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "pili-pinas-api"}


@app.get("/stats")
def stats():
    """Return ChromaDB collection statistics."""
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        count = collection.count()
        return {
            "collection": collection.name,
            "total_chunks": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch stats: {e}")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Main RAG endpoint.

    Takes a question and returns an AI-generated answer with source citations
    drawn from Philippine government documents and news articles.
    """
    try:
        rag = get_rag()
        # Override top_k if specified
        rag.top_k = req.top_k

        result = rag.query(
            question=req.question,
            source_type=req.source_type,
        )

        return QueryResponse(
            answer=result.answer,
            sources=[SourceDoc(**s) for s in result.sources],
            query=result.query,
            chunks_used=result.chunks_used,
        )
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Scrape background task ────────────────────────────────────────────────────

def _run_scrape_job(job_id: str, req: ScrapeRequest) -> None:
    """Background task: run ingestion then optionally build embeddings."""
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now().isoformat()
    try:
        from data_ingestion.ingestion import run_ingestion
        ingest_stats = run_ingestion(
            sources=req.sources,
            congress=req.congress,
            max_pages=req.max_pages,
            max_news=req.max_news,
        )
        _jobs[job_id]["stats"] = {"ingestion": ingest_stats}

        if req.embed:
            from embeddings.create_embeddings import run_embedding_pipeline
            embed_stats = run_embedding_pipeline()
            _jobs[job_id]["stats"]["embedding"] = embed_stats

        _jobs[job_id]["status"] = "done"
    except Exception as e:
        logger.exception(f"Scrape job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
    finally:
        _jobs[job_id]["finished_at"] = datetime.now().isoformat()


@app.post("/scrape", status_code=202)
def trigger_scrape(req: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start an on-demand scrape (and optionally embed) job.

    Returns immediately with a job_id. Poll GET /scrape/{job_id} for status.
    Sources: senate_bills | senators | gazette | house_bills | house_members | comelec | news
    """
    if req.sources:
        invalid = [s for s in req.sources if s not in VALID_SOURCES]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid sources: {invalid}. Valid options: {VALID_SOURCES}",
            )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "stats": None,
        "error": None,
    }
    background_tasks.add_task(_run_scrape_job, job_id, req)
    logger.info(f"Scrape job {job_id} queued (sources={req.sources})")
    return {"job_id": job_id, "status": "pending"}


@app.get("/scrape/{job_id}", response_model=ScrapeJobStatus)
def scrape_status(job_id: str):
    """Check the status of a scrape job by its job_id."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return ScrapeJobStatus(job_id=job_id, **job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
