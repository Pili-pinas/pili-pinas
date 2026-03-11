"""
Pili-Pinas FastAPI backend.

Endpoints:
    POST /query                — RAG query (main endpoint)
    GET  /health               — health check
    GET  /stats                — ChromaDB collection stats
    POST /scrape               — trigger on-demand scrape job (async, returns job_id)
    GET  /scrape/{job_id}      — poll scrape job status
    GET  /messenger/webhook    — Meta webhook verification handshake
    POST /messenger/webhook    — receive Messenger events

Run:
    uvicorn backend.src.api.main:app --reload
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Allow imports from backend/src
sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from retrieval.rag_chain import get_rag
from embeddings.vector_store import get_vector_store, VECTOR_DB_DIR
from api.auth import verify_api_key
from api.messenger import messenger_router

logger = logging.getLogger(__name__)


# ── OpenAPI metadata ──────────────────────────────────────────────────────────

_DESCRIPTION = """
**Pili-Pinas** helps Filipino voters make informed decisions using a
RAG (Retrieval-Augmented Generation) pipeline over Philippine government
documents, laws, and news articles.

## How it works

1. Ask a question in **English, Filipino, or Taglish**
2. The API retrieves the most relevant document chunks from its vector database
3. Claude Haiku generates a cited answer grounded in those documents

## Data sources

| Source | Type |
|--------|------|
| senate.gov.ph | Bills, resolutions, senator profiles |
| congress.gov.ph | House bills, member profiles |
| officialgazette.gov.ph | Laws, executive orders |
| comelec.gov.ph | Candidates, election results |
| Rappler, Inquirer, PhilStar, GMA | News articles |

## Scraping

Use the `/scrape` endpoint to refresh the document database on demand.
Jobs run in the background — poll `/scrape/{job_id}` for status.
"""

_TAGS_METADATA = [
    {
        "name": "query",
        "description": "Ask questions about Philippine politicians, laws, and government records.",
    },
    {
        "name": "scraping",
        "description": (
            "Trigger and monitor on-demand data ingestion jobs. "
            "Jobs run asynchronously — start with **POST /scrape**, "
            "then poll **GET /scrape/{job_id}**."
        ),
    },
    {
        "name": "system",
        "description": "Health check and database statistics.",
    },
    {
        "name": "messenger",
        "description": "Facebook Messenger bot webhook — receives and replies to user messages via the RAG pipeline.",
    },
]

app = FastAPI(
    title="Pili-Pinas API",
    description=_DESCRIPTION,
    version="0.1.0",
    contact={
        "name": "Pili-Pinas",
        "url": "https://github.com/your-repo/pili-pinas",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=_TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(messenger_router, prefix="/messenger", tags=["messenger"])


# ── Scrape progress (persisted to the mounted volume) ────────────────────────

PROGRESS_FILE = VECTOR_DB_DIR / "scrape_progress.json"


def _load_progress() -> dict | None:
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        return None


def _save_progress(data: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))


# ── In-memory job store ───────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}

VALID_SOURCES = [
    "senate_bills", "senators", "gazette",
    "house_bills", "house_members", "comelec", "news",
]


# ── Request / Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question": "What education bills has the Senate passed this year?",
            "source_type": "bill",
            "top_k": 5,
        }
    })

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question in English, Filipino, or Taglish.",
    )
    source_type: str | None = Field(
        None,
        description=(
            "Filter results to a specific document type. "
            "Options: `bill`, `law`, `news`, `profile`, `saln`, `election`. "
            "Omit to search across all types."
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve before generating the answer.",
    )


class SourceDoc(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Senate Bill No. 2765 — Basic Education Funding Act",
            "url": "https://senate.gov.ph/lis/bill_res.aspx?congress=19&q=SB02765",
            "source": "senate.gov.ph",
            "date": "2025-01-10",
            "score": 0.923,
        }
    })

    title: str = Field(description="Document title.")
    url: str = Field(description="Source URL — voters can verify the claim here.")
    source: str = Field(description="Domain of the source (e.g. senate.gov.ph).")
    date: str = Field(description="Publication date (YYYY-MM-DD).")
    score: float = Field(description="Similarity score (0–1). Higher = more relevant.")


class QueryResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "answer": (
                "The Senate passed the Basic Education Funding Act (SB 2765) on January 10, 2025, "
                "increasing per-pupil spending by 30% [senate.gov.ph]."
            ),
            "sources": [{
                "title": "Senate Bill No. 2765 — Basic Education Funding Act",
                "url": "https://senate.gov.ph/lis/bill_res.aspx?congress=19&q=SB02765",
                "source": "senate.gov.ph",
                "date": "2025-01-10",
                "score": 0.923,
            }],
            "query": "What education bills has the Senate passed this year?",
            "chunks_used": 1,
        }
    })

    answer: str = Field(description="AI-generated answer with inline source citations.")
    sources: list[SourceDoc] = Field(description="Documents used to generate the answer.")
    query: str = Field(description="The original question.")
    chunks_used: int = Field(description="Number of document chunks passed to the LLM.")


class ScrapeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "sources": ["news"],
            "max_news": 20,
            "embed": True,
        }
    })

    sources: list[str] | None = Field(
        None,
        description=(
            "Data sources to scrape. `null` scrapes all sources. "
            "Valid values: `senate_bills`, `senators`, `gazette`, "
            "`house_bills`, `house_members`, `comelec`, `news`."
        ),
    )
    congress: int = Field(
        19,
        ge=17,
        le=25,
        description="Philippine Congress number to use for bill scrapers (e.g. 19 = 19th Congress).",
    )
    max_pages: int = Field(
        3,
        ge=1,
        le=20,
        description="Maximum pages to scrape per government source.",
    )
    max_news: int = Field(
        20,
        ge=1,
        le=200,
        description="Maximum articles to fetch per news source.",
    )
    embed: bool = Field(
        True,
        description=(
            "If `true`, automatically runs the embedding pipeline after ingestion "
            "so new documents are immediately searchable."
        ),
    )
    resume: bool = Field(
        False,
        description=(
            "If `true`, resumes a previously interrupted scrape job by skipping "
            "sources that were already completed in the last run."
        ),
    )


class ScrapeJobStatus(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "job_id": "a3f1c2d4-1234-5678-abcd-ef0123456789",
            "status": "done",
            "started_at": "2025-01-15T10:00:00.000000",
            "finished_at": "2025-01-15T10:04:32.123456",
            "stats": {
                "ingestion": {
                    "sources": ["news"],
                    "counts": {"news_articles": 87},
                    "total_chunks": 87,
                },
                "embedding": {"news_articles": 87},
            },
            "error": None,
        }
    })

    job_id: str = Field(description="Unique job identifier.")
    status: str = Field(description="Job state: `pending` → `running` → `done` | `failed`.")
    started_at: str | None = Field(None, description="ISO timestamp when the job started.")
    finished_at: str | None = Field(None, description="ISO timestamp when the job finished.")
    stats: dict | None = Field(None, description="Ingestion and embedding counts, populated on completion.")
    error: str | None = Field(None, description="Error message if status is `failed`.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    response_description="Service is up and running.",
)
def health():
    """Returns `200 OK` when the API is reachable. Use this for liveness probes."""
    return {"status": "ok", "service": "pili-pinas-api"}


@app.get(
    "/stats",
    tags=["system"],
    summary="Vector database statistics",
    response_description="ChromaDB collection name and total chunk count.",
    responses={
        500: {"description": "Could not connect to ChromaDB."},
    },
)
def stats():
    """
    Returns the number of document chunks currently stored in ChromaDB.

    Run **POST /scrape** to ingest more documents and increase this count.
    """
    try:
        store = get_vector_store()
        return {
            "collection": store.name,
            "total_chunks": store.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch stats: {e}")


@app.post(
    "/query",
    tags=["query"],
    summary="Ask a question about Philippine politics",
    response_model=QueryResponse,
    response_description="AI-generated answer with cited sources.",
    dependencies=[Depends(verify_api_key)],
    responses={
        401: {"description": "Missing or invalid API key."},
        422: {"description": "Validation error — question too short or invalid field."},
        500: {"description": "RAG pipeline error."},
    },
)
def query(req: QueryRequest):
    """
    **Main endpoint.** Ask anything about Philippine politicians, laws, or government records.

    ### How it works
    1. Your question is embedded and compared against the vector database
    2. The top-`top_k` most relevant document chunks are retrieved
    3. Claude Haiku generates a factual answer citing those documents
    4. If no relevant documents are found, the API says so — it does **not** hallucinate

    ### Language support
    Questions can be in **English**, **Filipino**, or **Taglish**. The multilingual
    embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) handles all three.

    ### Tips
    - Use `source_type` to narrow results (e.g. `"news"` for recent coverage,
      `"bill"` for legislation)
    - Increase `top_k` for complex questions that span multiple documents
    """
    logger.info(f"Query received: question={req.question!r} source_type={req.source_type!r} top_k={req.top_k}")
    try:
        rag = get_rag()
        rag.top_k = req.top_k

        result = rag.query(
            question=req.question,
            source_type=req.source_type,
        )

        logger.info(f"Query answered: chunks_used={result.chunks_used} sources={[s['source'] for s in result.sources]}")
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
    """Background task: run ingestion source-by-source (checkpointed) then embed."""
    from data_ingestion.ingestion import run_ingestion
    from embeddings.create_embeddings import run_embedding_pipeline

    sources = req.sources or list(VALID_SOURCES)

    # Load prior progress when resuming
    sources_done: set[str] = set()
    if req.resume:
        prev = _load_progress()
        if prev and prev.get("status") == "running":
            sources_done = set(prev.get("sources_done", []))
            logger.info(f"Resuming scrape, skipping already-done sources: {sources_done}")

    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now().isoformat()

    progress = {
        "job_id": job_id,
        "started_at": _jobs[job_id]["started_at"],
        "sources_requested": sources,
        "sources_done": list(sources_done),
        "status": "running",
    }
    _save_progress(progress)

    try:
        all_counts: dict[str, int] = {}

        for source in sources:
            if source in sources_done:
                logger.info(f"Skipping already-completed source: {source}")
                continue

            stats = run_ingestion(
                sources=[source],
                congress=req.congress,
                max_pages=req.max_pages,
                max_news=req.max_news,
            )
            all_counts.update(stats.get("counts", {}))

            sources_done.add(source)
            progress["sources_done"] = list(sources_done)
            _save_progress(progress)

        ingest_stats = {
            "sources": sources,
            "counts": all_counts,
            "total_chunks": sum(all_counts.values()),
        }
        _jobs[job_id]["stats"] = {"ingestion": ingest_stats}

        if req.embed:
            scraped_collections = list(all_counts.keys())
            embed_stats = run_embedding_pipeline(collections=scraped_collections)
            _jobs[job_id]["stats"]["embedding"] = embed_stats

        _jobs[job_id]["status"] = "done"
        progress["status"] = "done"
        _save_progress(progress)
    except Exception as e:
        logger.exception(f"Scrape job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        progress["status"] = "failed"
        _save_progress(progress)
    finally:
        _jobs[job_id]["finished_at"] = datetime.now().isoformat()


@app.post(
    "/scrape",
    tags=["scraping"],
    summary="Start a data ingestion job",
    status_code=202,
    response_description="Job accepted. Use the returned `job_id` to poll for status.",
    responses={
        202: {
            "description": "Job queued successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "a3f1c2d4-1234-5678-abcd-ef0123456789",
                        "status": "pending",
                    }
                }
            },
        },
        401: {"description": "Missing or invalid API key."},
        422: {"description": "Invalid source name supplied."},
    },
    dependencies=[Depends(verify_api_key)],
)
def trigger_scrape(req: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    **Starts a background scraping + embedding job** and returns immediately with a `job_id`.

    ### Workflow
    ```
    POST /scrape  →  { job_id: "abc..." }
         ↓  (poll every few seconds)
    GET  /scrape/abc...  →  { status: "running" }
         ↓
    GET  /scrape/abc...  →  { status: "done", stats: { ... } }
    ```

    ### Source options
    | Value | What is scraped |
    |-------|----------------|
    | `senate_bills` | Bills and resolutions from senate.gov.ph |
    | `senators` | Senator profile pages |
    | `gazette` | Laws and EOs from officialgazette.gov.ph |
    | `house_bills` | House bills from congress.gov.ph |
    | `house_members` | House representative profiles |
    | `comelec` | 2025 candidate list from comelec.gov.ph |
    | `news` | RSS feeds from Rappler, Inquirer, PhilStar, MB, GMA |

    Omit `sources` (or set to `null`) to scrape **all** sources.

    ### Notes
    - Scraping respects a 1.5 s delay between requests (robots.txt etiquette)
    - Set `embed: false` to ingest documents without rebuilding embeddings
    - Jobs are stored in memory — they reset on server restart
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


@app.get(
    "/scrape/{job_id}",
    tags=["scraping"],
    summary="Poll a scrape job's status",
    response_model=ScrapeJobStatus,
    response_description="Current state of the scrape job.",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {"description": "Job found — check `status` field for current state."},
        401: {"description": "Missing or invalid API key."},
        404: {"description": "No job found with this ID."},
    },
)
def scrape_status(job_id: str):
    """
    Returns the current state of a scrape job started by **POST /scrape**.

    ### Status values
    | Status | Meaning |
    |--------|---------|
    | `pending` | Job is queued, not yet started |
    | `running` | Scraping / embedding in progress |
    | `done` | Completed successfully — `stats` is populated |
    | `failed` | An error occurred — `error` contains the message |
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return ScrapeJobStatus(job_id=job_id, **job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
