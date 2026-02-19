"""
Pili-Pinas FastAPI backend.

Endpoints:
    POST /query        — RAG query (main endpoint)
    GET  /health       — health check
    GET  /stats        — ChromaDB collection stats

Run:
    uvicorn backend.src.api.main:app --reload
"""

import logging
import sys
from pathlib import Path

# Allow imports from backend/src
sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi import FastAPI, HTTPException
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
