#!/usr/bin/env bash
# Sync local vector_db and processed JSONL files to the Fly.io volume.
#
# The Fly volume at /app/vector_db holds both ChromaDB and JSONL files:
#   /app/vector_db/         ← ChromaDB (chroma.sqlite3, uuid dirs)
#   /app/vector_db/processed/ ← JSONL chunks (mirrors PROCESSED_DIR in fly.toml)
#
# Usage:
#   ./scripts/sync_to_fly.sh              # sync both vector_db and processed/
#   ./scripts/sync_to_fly.sh --db-only    # ChromaDB only (skip JSONL)
#   ./scripts/sync_to_fly.sh --jsonl-only # JSONL only (skip ChromaDB, faster)
#
# Requirements:
#   fly CLI installed and authenticated  (brew install flyctl)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

APP="${FLY_APP:-pili-pinas-api}"
LOCAL_VECTOR_DB="$REPO_ROOT/backend/vector_db"
LOCAL_PROCESSED="$REPO_ROOT/backend/data/processed"

SYNC_DB=true
SYNC_JSONL=true

for arg in "$@"; do
  case "$arg" in
    --db-only)    SYNC_JSONL=false ;;
    --jsonl-only) SYNC_DB=false ;;
  esac
done

echo "=== Pili-Pinas → Fly sync (app: $APP) ==="
echo ""

# Make sure the app is awake before transferring
echo "Waking Fly machine..."
flyctl machine start --app "$APP" 2>/dev/null || true
for i in $(seq 1 12); do
  if flyctl status --app "$APP" 2>/dev/null | grep -q "started"; then
    echo "Machine is up."
    break
  fi
  echo "  waiting... ($i/12)"
  sleep 5
done
echo ""

if $SYNC_JSONL; then
  if [ ! -d "$LOCAL_PROCESSED" ] || [ -z "$(ls -A "$LOCAL_PROCESSED")" ]; then
    echo "⚠  No JSONL files found at $LOCAL_PROCESSED — skipping."
  else
    JSONL_COUNT=$(ls "$LOCAL_PROCESSED"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    echo "--- Syncing JSONL files ($JSONL_COUNT files) → /app/vector_db/processed/ ---"
    tar czf - -C "$LOCAL_PROCESSED" . \
      | flyctl ssh console --app "$APP" -C "mkdir -p /app/vector_db/processed && tar xzf - -C /app/vector_db/processed"
    echo "JSONL sync done."
    echo ""
  fi
fi

if $SYNC_DB; then
  if [ ! -f "$LOCAL_VECTOR_DB/chroma.sqlite3" ]; then
    echo "⚠  No ChromaDB found at $LOCAL_VECTOR_DB — skipping."
  else
    DB_SIZE=$(du -sh "$LOCAL_VECTOR_DB" | cut -f1)
    echo "--- Syncing ChromaDB ($DB_SIZE) → /app/vector_db/ ---"
    echo "    (this may take a few minutes for large databases)"
    # Exclude the processed/ subdir — synced separately above
    tar czf - -C "$LOCAL_VECTOR_DB" --exclude="./processed" . \
      | flyctl ssh console --app "$APP" -C "tar xzf - -C /app/vector_db"
    echo "ChromaDB sync done."
    echo ""
  fi
fi

echo "=== Sync complete. Verifying... ==="
flyctl ssh console --app "$APP" -C \
  "python3 -c \"
import sys; sys.path.insert(0, '/app/src')
from embeddings.vector_store import get_vector_store
print('ChromaDB vectors:', get_vector_store().count())
\"" 2>/dev/null || echo "(verification skipped — app may need a moment to restart)"

echo ""
echo "Done. Your Fly deployment now has the latest local data."
