"""
Pili-Pinas FastAPI backend.

Endpoints:
    POST /query                — RAG query (main endpoint)
    GET  /popular              — most frequently asked questions
    GET  /health               — health check
    GET  /stats                — ChromaDB collection stats
    GET  /unanswered           — questions with no matching documents
    DELETE /cache              — clear query cache
    POST /scrape               — trigger on-demand scrape job (async, returns job_id)
    GET  /scrape/{job_id}      — poll scrape job status
    GET  /messenger/webhook    — Meta webhook verification handshake
    POST /messenger/webhook    — receive Messenger events

Run:
    uvicorn backend.src.api.main:app --reload
"""

import sys
from pathlib import Path

# Allow imports from backend/src
sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.messenger import messenger_router
from api.system import router as system_router
from api.query import router as query_router
from api.scrape import router as scrape_router


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
| Rappler, PhilStar, GMA, PCIJ | News articles |

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

app.include_router(system_router)
app.include_router(query_router)
app.include_router(scrape_router)
app.include_router(messenger_router, prefix="/messenger", tags=["messenger"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
