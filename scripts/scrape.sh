#!/usr/bin/env bash
# Run the daily scrape locally: ingest recent news + bills + senators → embed.
# Mirrors what the GitHub Actions daily scrape does.
#
# Usage:
#   ./scripts/scrape.sh              # default: 30 news articles
#   ./scripts/scrape.sh 50           # custom max news

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND="$REPO_ROOT/backend"

cd "$REPO_ROOT"

MAX_NEWS="${1:-30}"

echo "=== Pili-Pinas Daily Scrape ==="
echo "Sources  : news senate_bills house_bills fact_check oversight statistics financial politicians"
echo "Max news : $MAX_NEWS"
echo ""

echo "--- Step 1/2: Ingestion ---"
uv run --project "$BACKEND" python backend/src/data_ingestion/ingestion.py \
  --sources news senate_bills house_bills fact_check oversight statistics financial politicians \
  --max-news "$MAX_NEWS"

echo ""
echo "--- Step 2/2: Embeddings ---"
uv run --project "$BACKEND" python backend/src/embeddings/create_embeddings.py

echo ""
echo "=== Scrape complete ==="
