#!/usr/bin/env bash
# Run the historical backfill locally.
# Scrapes Congress 17-20 for bills, COMELEC 2016/2019/2022/2025, and all laws.
#
# Usage:
#   ./scripts/backfill.sh              # default: 1000 laws
#   ./scripts/backfill.sh 5000         # custom max laws

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

MAX_LAWS="${1:-1000}"

echo "=== Pili-Pinas Historical Backfill ==="
echo "Sources  : senate_bills senators gazette house_bills house_members comelec"
echo "Congresses: 17 18 19 20"
echo "Elections : 2016 2019 2022 2025"
echo "Max bills : 500 per congress"
echo "Max laws  : $MAX_LAWS"
echo ""

# Clear the seen-URLs tracker so the next daily news scrape re-fetches
# any articles that previously returned empty content.
SEEN_URLS="$REPO_ROOT/data/seen_urls.json"
if [[ -f "$SEEN_URLS" ]]; then
  rm "$SEEN_URLS"
  echo "Cleared seen_urls.json"
fi

python backend/src/data_ingestion/ingestion.py \
  --sources senate_bills senators gazette house_bills house_members comelec \
  --congresses 17 18 19 20 \
  --election-years 2016 2019 2022 2025 \
  --max-pages 500 \
  --max-laws "$MAX_LAWS" \
  --max-news 20
