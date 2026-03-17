# Pili-Pinas Backend

FastAPI + LangChain + ChromaDB RAG pipeline.

## Setup

```bash
uv pip install -r requirements.txt
cp ../.env.example ../.env
# Edit .env if needed (default: Ollama + llama3.2)
```

## Pipeline

```bash
# 1. Ingest documents from all sources (--sources flag for subset)
python src/data_ingestion/ingestion.py --max-pages 3 --max-news 20

# 2. Build vector embeddings
python src/embeddings/create_embeddings.py

# 3. Start API
uvicorn src.api.main:app --reload
```

## Historical Backfill

Run once to populate 10 years of data (Congress 17–20, elections 2016–2025, 1000 laws).
The script skips news since RSS feeds have no historical archive.

```bash
# From repo root — full backfill
./scripts/backfill.sh

# Specific sources only
./scripts/backfill.sh senate_bills gazette
./scripts/backfill.sh comelec
```

After the script finishes, rebuild the embeddings:

```bash
python backend/src/embeddings/create_embeddings.py
```

**Parameters** (edit `scripts/backfill.sh` to adjust):

| Parameter | Default | Description |
|---|---|---|
| `--congresses` | `17 18 19 20` | Congress sessions to scrape for bills |
| `--election-years` | `2016 2019 2022 2025` | COMELEC election years |
| `--max-pages` | `500` | Max bills per congress session |
| `--max-laws` | `1000` | Max laws from Official Gazette |

You can also trigger a backfill remotely via GitHub Actions:
**Actions → Historical backfill → Run workflow** (scales to 4GB automatically).

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
