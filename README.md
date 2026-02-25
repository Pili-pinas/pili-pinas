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

## Deploying to Fly.io

Use `scripts/fly-deploy.sh` — it handles volume cleanup automatically before every deploy, so you never hit the "insufficient resources" zone-capacity error manually.

### First-time setup
```bash
# Install flyctl
brew install flyctl        # macOS
# or: curl -L https://fly.io/install.sh | sh

# Create the app, set your API key, and deploy in one step
./scripts/fly-deploy.sh --setup
```

`--setup` will:
1. Open the Fly.io login page if you're not already authenticated
2. Create the `pili-pinas-api` app (skips if it already exists)
3. Prompt for your `ANTHROPIC_API_KEY` and set it as a Fly secret
4. Run the full deploy

### Subsequent deploys
```bash
./scripts/fly-deploy.sh
```

The script automatically destroys any unattached `vector_db` volumes before deploying,
which prevents the _"insufficient resources to create new machine with existing volume"_
error that occurs when Fly can't place the machine in the same zone as an orphaned volume.

The volume is auto-created on first deploy via `initial_size = '3gb'` in `fly.toml`.

### IPv6 issues
If `fly logs` or the deploy errors with a `dial tcp [...]:443: connect: no route to host` message, your ISP blocks IPv6. Prefix any `fly` command with `FLYCTL_NO_IPV6=1`, or add it to your shell profile:
```bash
echo 'export FLYCTL_NO_IPV6=1' >> ~/.zshrc
```

### Useful commands
```bash
FLYCTL_NO_IPV6=1 fly logs --app pili-pinas-api    # tail live logs
FLYCTL_NO_IPV6=1 fly status --app pili-pinas-api  # machine health
FLYCTL_NO_IPV6=1 fly ssh console --app pili-pinas-api  # SSH into container
FLYCTL_NO_IPV6=1 fly volumes list --app pili-pinas-api # check volume usage
```

> **Region**: `sin` (Singapore) is the closest Fly region to the Philippines.
> Change `primary_region` in `fly.toml` if you prefer a different region.

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
