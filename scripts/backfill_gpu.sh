#!/usr/bin/env bash
# GPU-accelerated historical backfill: scrape → process → embed on GPU.
# Uses CUDA (NVIDIA) or MPS (Apple Silicon) for the embedding step.
#
# Usage:
#   ./scripts/backfill_gpu.sh              # default: 1000 laws
#   ./scripts/backfill_gpu.sh 5000         # custom max laws

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND="$REPO_ROOT/backend"

cd "$REPO_ROOT"

MAX_LAWS="${1:-1000}"

# Auto-detect GPU: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU fallback.
# Override by setting EMBEDDING_DEVICE=cuda|mps|cpu before running this script.
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"

# Larger batch size saturates GPU throughput.
# MPS (Apple Silicon): 256 is a safe default.
# CUDA (NVIDIA):       512–1024 depending on VRAM — raise if you have headroom.
BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

echo "=== Pili-Pinas Historical Backfill (GPU) ==="
echo "Sources   : senate_bills senators gazette house_bills house_members comelec fact_check oversight statistics research financial politicians"
echo "Congresses: 13 14 15 16 17 18 19 20"
echo "Elections : 2007 2010 2013 2016 2019 2022 2025"
echo "Max bills : 2000 per congress"
echo "Max laws  : $MAX_LAWS (cutoff: 2006)"
echo "Max items : 200 per supplemental source"
echo "GPU device: $EMBEDDING_DEVICE"
echo "Batch size: $BATCH_SIZE"
echo ""

# Clear the seen-URLs tracker so the next daily news scrape re-fetches
# any articles that previously returned empty content.
SEEN_URLS="$REPO_ROOT/data/seen_urls.json"
if [[ -f "$SEEN_URLS" ]]; then
  rm "$SEEN_URLS"
  echo "Cleared seen_urls.json"
fi

echo ""
RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG="--resume"
  echo "Mode: RESUME (skipping completed steps)"
else
  echo "Mode: FRESH (starting from scratch)"
fi

echo "--- Step 1/2: Ingestion ---"
uv run --project "$BACKEND" python backend/src/data_ingestion/ingestion.py \
  --sources senate_bills senators gazette house_bills house_members comelec \
    fact_check oversight statistics research financial politicians \
  --congresses 13 14 15 16 17 18 19 20 \
  --election-years 2007 2010 2013 2016 2019 2022 2025 \
  --max-pages 2000 \
  --max-news 200 \
  --max-laws "$MAX_LAWS" \
  --gazette-from-year 2006 \
  --staging \
  $RESUME_FLAG

echo ""
echo "--- Step 2/2: Embeddings (GPU batch_size=$BATCH_SIZE) ---"
uv run --project "$BACKEND" python backend/src/embeddings/create_embeddings.py \
  --batch-size "$BATCH_SIZE"

echo ""
echo "=== Backfill complete (GPU) ==="
