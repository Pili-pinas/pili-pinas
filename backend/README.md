# Pili-Pinas Backend

FastAPI + LangChain + ChromaDB RAG pipeline.

## Setup

```bash
uv pip install -r requirements.txt
cp ../.env.example ../.env
# Set ANTHROPIC_API_KEY in .env
```

## Running locally

**Required environment variables** — set in `.env` before running:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for API/queries) | Needed to run `uvicorn`. Not required for ingestion alone. |
| `PROCESSED_DIR` | No | Where to save JSONL chunks. Defaults to `backend/data/processed`. |

### Daily scrape (news + current bills + senator profiles)

```bash
# From repo root
./scripts/scrape.sh           # default: 30 news articles
./scripts/scrape.sh 50        # custom max news
```

### Historical backfill (all sources, Congress 17–20, elections 2016–2025)

Run once to populate the full dataset. Skips news since RSS feeds have no historical archive.

```bash
./scripts/backfill.sh         # default: 1000 laws
./scripts/backfill.sh 12500   # all 12,500+ Republic Acts
```

Both scripts run **ingestion → embeddings** in one shot using `uv`.

You can also trigger a backfill remotely via GitHub Actions:
**Actions → Historical backfill → Run workflow**.

### Start the API

```bash
uvicorn src.api.main:app --reload
```

## Endpoints

| Method | Path      | Description               |
|--------|-----------|---------------------------|
| GET    | /health   | Health check              |
| GET    | /stats    | ChromaDB stats            |
| POST   | /query    | RAG query with citations  |

### Query example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What bills has Senator Padilla filed?", "top_k": 5}'
```

## Directory Structure

```
backend/
├── src/
│   ├── data_ingestion/
│   │   ├── scrapers/       # One scraper per source
│   │   ├── processors/     # HTML + PDF text extraction
│   │   └── ingestion.py    # Main pipeline orchestrator
│   ├── embeddings/
│   │   ├── create_embeddings.py   # Embed chunks → ChromaDB
│   │   └── vector_store.py        # ChromaDB wrapper
│   ├── retrieval/
│   │   ├── rag_chain.py    # RAG query logic
│   │   └── prompts.py      # System + user prompts
│   └── api/
│       └── main.py         # FastAPI app
├── data/
│   ├── raw/                # Original HTML/PDFs (gitignored)
│   └── processed/          # JSONL chunks (gitignored)
└── vector_db/              # ChromaDB files (gitignored)
```
