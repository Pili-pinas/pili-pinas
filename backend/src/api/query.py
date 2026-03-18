"""Query routes: /query (RAG) and /popular (most asked questions)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from retrieval.rag_chain import get_rag
from api.auth import verify_api_key
from api.cache import _cache_get, _cache_set, get_popular_questions
from api.unanswered import _log_unanswered
from api.models import QueryRequest, QueryResponse, SourceDoc

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.get(
    "/popular",
    summary="Most frequently asked questions",
    response_description="Questions sorted by number of times asked.",
)
def popular_questions(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of questions to return."),
):
    """Returns the most frequently asked questions, sorted by ask count descending."""
    try:
        return {"questions": get_popular_questions(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch popular questions: {e}")


@router.post(
    "/query",
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

    cached = _cache_get(req.question, req.source_type)
    if cached:
        logger.info(f"Cache hit for question={req.question!r}")
        return QueryResponse(
            answer=cached["answer"],
            sources=[SourceDoc(**s) for s in cached["sources"]],
            query=req.question,
            chunks_used=cached["chunks_used"],
        )
    logger.info("Cache miss — running RAG pipeline")

    try:
        rag = get_rag()
        rag.top_k = req.top_k

        result = rag.query(
            question=req.question,
            source_type=req.source_type,
        )

        if result.chunks_used == 0:
            logger.info(f"No relevant chunks found — logging unanswered question: {req.question!r}")
            _log_unanswered(req.question, req.source_type)

        response = QueryResponse(
            answer=result.answer,
            sources=[SourceDoc(**s) for s in result.sources],
            query=result.query,
            chunks_used=result.chunks_used,
        )
        _cache_set(req.question, req.source_type, response)
        logger.info(f"Query answered: chunks_used={result.chunks_used} sources={[s['source'] for s in result.sources]}")
        return response
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
