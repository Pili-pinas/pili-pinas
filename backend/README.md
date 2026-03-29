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

./scripts/scrape_gpu.sh       # same, but uses MPS/CUDA for the embedding step
```

### Historical backfill (all sources, Congress 13–20, elections 2007–2025)

Run once to populate the full dataset. Covers 20 years of bills, laws, COMELEC, oversight, statistics, research, financial, and enriched politician profiles.

```bash
./scripts/backfill.sh         # default: 1000 laws
./scripts/backfill.sh 12500   # all 12,500+ Republic Acts

./scripts/backfill_gpu.sh     # same, with GPU-accelerated embeddings (MPS/CUDA)
```

Both scripts run **ingestion → embeddings** in one shot and write new data to a staging directory, swapping it into `data/processed/` only on success — so a failed run never corrupts the existing dataset.

**Resume an interrupted backfill** (skips already-completed congresses/sources):

```bash
RESUME=1 ./scripts/backfill.sh
RESUME=1 ./scripts/backfill_gpu.sh
```

You can also trigger a backfill remotely via GitHub Actions:
**Actions → Historical backfill → Run workflow**.

### Start the API

```bash
uvicorn src.api.main:app --reload
```

## Utility scripts

```bash
# How many vectors are in ChromaDB?
python backend/scripts/vector_count.py

# Which politicians have a profile?
python backend/scripts/list_politicians.py
python backend/scripts/list_politicians.py --search robredo
python backend/scripts/list_politicians.py --source chromadb

# Scrape and filter by keyword
python backend/scripts/scrape_keyword.py "Leni Robredo"
python backend/scripts/scrape_keyword.py "Leni Robredo" --dry-run
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
