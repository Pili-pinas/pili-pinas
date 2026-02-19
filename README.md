# Pili-Pinas 🗳️

> AI-powered tool helping Filipino voters make informed decisions.

Pili-Pinas uses a RAG (Retrieval-Augmented Generation) pipeline to summarize politician records, voting history, Philippine laws, SALN disclosures, and news coverage — with citations so voters can verify every claim.

---

## Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) with `llama3.2` pulled (`ollama pull llama3.2`)

### Setup

```bash
# Install backend deps
pip install -r backend/requirements.txt

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

## Architecture

```
User query
  → Streamlit UI
  → FastAPI /query endpoint
  → ChromaDB similarity search (top-k chunks)
  → Ollama / Claude Haiku (LLM)
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
