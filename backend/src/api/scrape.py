"""Scrape routes: trigger and poll background ingestion jobs."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from embeddings.vector_store import VECTOR_DB_DIR
from api.auth import verify_api_key
from api.cache import _cache_clear
from api.models import ScrapeRequest, ScrapeJobStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scraping"])

VALID_SOURCES = [
    "senate_bills", "senators", "gazette",
    "house_bills", "house_members", "comelec", "news",
    "fact_check", "oversight", "statistics", "research", "financial",
]

PROGRESS_FILE = VECTOR_DB_DIR / "scrape_progress.json"

# In-memory job store — resets on server restart.
_jobs: dict[str, dict] = {}


def _load_progress() -> dict | None:
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        return None


def _save_progress(data: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))


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
                congresses=req.congresses,
                election_years=req.election_years,
                max_pages=req.max_pages,
                max_news=req.max_news,
                max_laws=req.max_laws,
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
        _cache_clear(scraped_sources=sources)
    except Exception as e:
        logger.exception(f"Scrape job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        progress["status"] = "failed"
        _save_progress(progress)
    finally:
        _jobs[job_id]["finished_at"] = datetime.now().isoformat()


@router.post(
    "/scrape",
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
    | `news` | RSS feeds from Rappler, PhilStar, GMA, Business World, PCIJ |

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


@router.get(
    "/scrape/{job_id}",
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
