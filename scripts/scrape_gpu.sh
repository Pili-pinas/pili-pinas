#!/usr/bin/env bash
# GPU-accelerated daily scrape: ingest recent content → embed on GPU.
# Uses CUDA (NVIDIA) or MPS (Apple Silicon) for the embedding step.
#
# Usage:
#   ./scripts/scrape_gpu.sh              # default: 30 news articles
#   ./scripts/scrape_gpu.sh 50           # custom max news

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND="$REPO_ROOT/backend"

cd "$REPO_ROOT"

MAX_NEWS="${1:-30}"

# Auto-detect GPU: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU fallback.
# Override by setting EMBEDDING_DEVICE=cuda|mps|cpu before running this script.
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"

# Larger batch size saturates GPU throughput.
# MPS (Apple Silicon): 256 is a safe default.
# CUDA (NVIDIA):       512–1024 depending on VRAM — raise if you have headroom.
BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

echo "=== Pili-Pinas Daily Scrape (GPU) ==="
echo "Sources  : news senate_bills house_bills fact_check oversight statistics financial politicians"
echo "Max news : $MAX_NEWS"
echo "GPU device: $EMBEDDING_DEVICE"
echo "Batch size: $BATCH_SIZE"
echo ""

echo "--- Step 1/2: Ingestion ---"
uv run --project "$BACKEND" python backend/src/data_ingestion/ingestion.py \
  --sources news senate_bills house_bills fact_check oversight statistics financial politicians \
  --max-news "$MAX_NEWS"

echo ""
echo "--- Step 2/2: Embeddings (GPU batch_size=$BATCH_SIZE) ---"
uv run --project "$BACKEND" python backend/src/embeddings/create_embeddings.py \
  --batch-size "$BATCH_SIZE"

echo ""
echo "=== Scrape complete (GPU) ==="
