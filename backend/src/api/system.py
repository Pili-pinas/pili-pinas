"""System routes: health, stats, unanswered questions, cache management."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from embeddings.vector_store import get_vector_store
from api.auth import verify_api_key
from api.cache import cache_clear_all
from api.unanswered import get_unanswered_questions

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    summary="Health check",
    response_description="Service is up and running.",
)
def health():
    """Returns `200 OK` when the API is reachable. Use this for liveness probes."""
    return {"status": "ok", "service": "pili-pinas-api"}


@router.get(
    "/stats",
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


@router.get(
    "/unanswered",
    summary="Questions that could not be answered",
    response_description="List of questions with no matching documents in the database.",
    dependencies=[Depends(verify_api_key)],
)
def unanswered():
    """Returns questions that returned zero chunks — useful for identifying gaps in the dataset."""
    try:
        return {"questions": get_unanswered_questions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch unanswered questions: {e}")


@router.delete(
    "/cache",
    summary="Clear the query cache",
    response_description="Number of cache entries deleted.",
    dependencies=[Depends(verify_api_key)],
)
def clear_cache():
    """Deletes all cached query responses. Useful after fixing bad cached answers."""
    try:
        deleted = cache_clear_all()
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear cache: {e}")
