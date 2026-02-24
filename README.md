# Pili-Pinas 🗳️

> AI-powered tool helping Filipino voters make informed decisions.

Pili-Pinas uses a RAG (Retrieval-Augmented Generation) pipeline to summarize politician records, voting history, Philippine laws, SALN disclosures, and news coverage — with citations so voters can verify every claim.

---

## Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`pip install uv` or `brew install uv`)
- Anthropic API key (set `ANTHROPIC_API_KEY` in `.env`)

### Setup

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

uv venv .venv --python 3.11
source .venv/bin/activate

# Install backend deps
uv pip install -r backend/requirements.txt

# Scrape and ingest documents
python backend/src/data_ingestion/ingestion.py

# Build vector embeddings
python backend/src/embeddings/create_embeddings.py

# Start API
uvicorn backend.src.api.main:app --reload

# Start UI (separate terminal)
streamlit run frontend/app.py
```

---

## Next Steps

### 1. Environment setup
```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

uv venv .venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

uv pip install -r backend/requirements.txt
uv pip install -r frontend/requirements.txt
```

### 2. Fix scraper selectors
The scrapers have assumed CSS selectors — verify them against the live pages.
Start with the most reliable source first:
```bash
python backend/src/data_ingestion/scrapers/news_sites.py  # RSS feeds — works as-is
python backend/src/data_ingestion/scrapers/senate.py      # HTML — likely needs fixes
```

### 3. Ingest documents (start small)
```bash
python backend/src/data_ingestion/ingestion.py --sources news --max-news 10
```

### 4. Build vector embeddings
```bash
python backend/src/embeddings/create_embeddings.py
```

### 5. Test the RAG chain
```bash
python backend/src/retrieval/rag_chain.py
```

### 6. Run the full stack
```bash
uvicorn backend.src.api.main:app --reload
streamlit run frontend/app.py  # separate terminal
```

---

## Architecture

```
User query
  → Streamlit UI
  → FastAPI /query endpoint
  → ChromaDB similarity search (top-k chunks)
  → Claude Haiku (LLM)
  → Answer with source citations
```

---

## Implementation Phases

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Setup + PoC (50–100 docs, basic RAG) | In Progress |
| 2 | Data pipeline + scraper automation | Planned |
| 3 | FastAPI backend + Streamlit UI | In Progress |
| 4 | Multilingual support (Filipino + English) | Planned |
| 5 | Production deployment | Planned |

---

## Data Sources

- Senate of the Philippines — senate.gov.ph
- House of Representatives — congress.gov.ph
- Official Gazette — officialgazette.gov.ph
- COMELEC — comelec.gov.ph
- Rappler, Inquirer, PhilStar, GMA News
- PCIJ, Transparency International PH
