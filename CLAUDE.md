# Pili-Pinas — CLAUDE.md

> AI-powered tool for Filipino informed voters. Summarizes politician records, voting history, and government documents using a RAG (Retrieval-Augmented Generation) pipeline.

-----

## Project Goal

Help Filipino voters make informed decisions by providing AI-generated summaries of:

- Politician profiles, voting records, and achievements
- Philippine laws, bills, and resolutions
- SALN (financial disclosures) and COMELEC data
- News coverage and investigative reports

-----

## Tech Stack

|Layer      |Tool                                                       |
|-----------|-----------------------------------------------------------|
|Framework  |LangChain                                                  |
|Vector DB  |ChromaDB (default, swappable — see Vector Store below)     |
|Embeddings |sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2|
|LLM        |Claude Haiku (`claude-haiku-4-5-20251001`) via Anthropic SDK|
|Backend API|FastAPI                                                    |
|Frontend   |Streamlit                                                  |
|Language   |Python 3.11+                                               |

-----

## Repository Structure

```
pili-pinas/
├── backend/                        # RAG API (FastAPI)
│   ├── src/
│   │   ├── data_ingestion/
│   │   │   ├── scrapers/
│   │   │   │   ├── official_gazette.py
│   │   │   │   ├── senate.py
│   │   │   │   ├── congress.py
│   │   │   │   ├── comelec.py
│   │   │   │   └── news_sites.py
│   │   │   ├── processors/
│   │   │   │   ├── pdf_processor.py
│   │   │   │   └── html_processor.py
│   │   │   └── ingestion.py
│   │   ├── embeddings/
│   │   │   ├── base.py              # VectorStore ABC
│   │   │   ├── create_embeddings.py
│   │   │   └── vector_store.py      # ChromaVectorStore + get_vector_store() factory
│   │   ├── retrieval/
│   │   │   ├── rag_chain.py
│   │   │   └── prompts.py
│   │   └── api/
│   │       └── main.py
│   ├── data/
│   │   ├── raw/
│   │   │   ├── laws/
│   │   │   ├── politician_profiles/
│   │   │   └── news_articles/
│   │   ├── processed/
│   │   └── metadata.json
│   ├── vector_db/
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                       # Streamlit UI
│   ├── app.py
│   ├── components/
│   │   ├── search_interface.py
│   │   └── results_display.py
│   ├── requirements.txt
│   └── README.md
│
├── docker-compose.yml
├── .gitignore
├── CLAUDE.md                       # This file
└── README.md
```

-----

## Data Sources

### Official Government

- **Official Gazette** — officialgazette.gov.ph (laws, executive orders, proclamations)
- **Senate of the Philippines** — senate.gov.ph (bills, resolutions, voting records)
- **House of Representatives** — congress.gov.ph (congressional records, rep profiles)
- **Commission on Elections** — comelec.gov.ph (candidates, election results, parties)
- **Supreme Court E-Library** — elibrary.judiciary.gov.ph (case law, legal precedents)
- **SALN** — public financial disclosures (sourced from news and advocacy sites)

### News Archives

- Rappler — rappler.com
- Philippine Daily Inquirer — inquirer.net
- Philippine Star — philstar.com
- Manila Bulletin — mb.com.ph
- GMA News — gmanetwork.com

### Watchdog / CSO

- Philippine Center for Investigative Journalism — pcij.org
- iSYSTEM Asia
- Transparency International Philippines

-----

## Document Metadata Schema

Every ingested document must include:

```python
{
  "source": "senate.gov.ph",
  "source_type": "bill | law | news | profile | saln | election",
  "date": "YYYY-MM-DD",
  "politician": "Full Name",  # if applicable
  "title": "Document title",
  "url": "https://..."
}
```

-----

## Scraping Rules

- Always check `robots.txt` before scraping any site
- Rate limit: 1–2 seconds between requests
- Store raw documents before processing — never discard originals
- Track failed URLs in a log for retry

-----

## Key Commands

```bash
# Install dependencies
uv pip install -r backend/requirements.txt

# Run backend API
uvicorn src.api.main:app --reload

# Run frontend
streamlit run frontend/app.py

# Ingest documents
python backend/src/data_ingestion/ingestion.py

# Build vector embeddings
python backend/src/embeddings/create_embeddings.py
```

-----

## Vector Store Abstraction

The vector store is swappable via the `VECTOR_STORE_BACKEND` env var (default: `"chroma"`).

```
embeddings/base.py          ← VectorStore ABC (interface)
embeddings/vector_store.py  ← ChromaVectorStore impl + get_vector_store() factory
```

**To add a new backend (e.g. Turso, Pinecone):**
1. Subclass `VectorStore` from `embeddings.base`
2. Implement `name`, `upsert()`, `query()`, `count()`
3. Register it in `get_vector_store()` in `vector_store.py`
4. Set `VECTOR_STORE_BACKEND=<your_backend>` in `.env`

**Query result format** (all backends must return this shape):
```python
{
    "documents": [["chunk text", ...]],
    "metadatas": [[{"source": ..., "title": ...}, ...]],
    "distances": [[0.05, 0.12, ...]],  # cosine distance, lower = more similar
}
```

-----

## RAG Pipeline Overview

1. **Ingest** — Scrape/download documents from sources
1. **Process** — Extract text from HTML/PDF, clean, chunk
1. **Embed** — Convert chunks to vectors using multilingual embeddings
1. **Store** — Save vectors + metadata via `get_vector_store()` (ChromaDB by default)
1. **Query** — User asks question → retrieve relevant chunks → LLM generates answer with citations

-----

## Implementation Phases

|Phase|Goal                                             |Status |
|-----|-------------------------------------------------|-------|
|1    |Setup + Proof of Concept (50–100 docs, basic RAG)|Planned|
|2    |Data pipeline + scraper automation               |Planned|
|3    |FastAPI backend + Streamlit UI                   |Planned|
|4    |Multilingual support (Filipino + English)        |Planned|
|5    |Production deployment                            |Planned|

-----

## Language Considerations

- Use **multilingual embeddings** to handle both Filipino and English text
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Laws are mostly in English; news and social content may be in Filipino or mixed (Taglish)

-----

## Cost Estimates

|Setup                 |Monthly Cost|Notes                              |
|----------------------|------------|-----------------------------------|
|Dev (local)           |$0          |ChromaDB on disk, ~$0.002/query    |
|Minimal production    |$1–5        |Light traffic, Claude Haiku        |
|Recommended production|$20–50      |Moderate traffic + VPS for ChromaDB|
|High-performance      |$150–220    |High traffic, faster Claude models |

-----

## Notes for Claude

- Project name: **Pili-Pinas**
- This is a solo project by Kiko (Senior Software Engineer, Manila)
- **Workflow: TDD** — write tests first, then implementation (pytest)
- Prefer Python, concise code, and well-commented scrapers
- Data freshness matters — politicians’ records change with elections (next PH election: May 2025)
- Prioritize citation of sources in all LLM outputs so voters can verify claims
- Vector store is swappable — always use `get_vector_store()`, never import ChromaDB directly