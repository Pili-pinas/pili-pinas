#!/usr/bin/env bash
# Run the historical backfill locally.
# Scrapes Congress 17-20 for bills, COMELEC 2016/2019/2022/2025, and 1000 laws.
#
# Usage:
#   ./scripts/backfill.sh                   # full backfill (all sources)
#   ./scripts/backfill.sh senate_bills      # single source
#   ./scripts/backfill.sh senate_bills gazette  # multiple sources

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

SOURCES="${*:-senate_bills senators gazette house_bills house_members comelec}"

echo "=== Pili-Pinas Historical Backfill ==="
echo "Sources  : $SOURCES"
echo "Congresses: 17 18 19 20"
echo "Elections : 2016 2019 2022 2025"
echo "Max bills : 500 per congress"
echo "Max laws  : 1000"
echo ""

# shellcheck disable=SC2086
python backend/src/data_ingestion/ingestion.py \
  --sources $SOURCES \
  --congresses 17 18 19 20 \
  --election-years 2016 2019 2022 2025 \
  --max-pages 500 \
  --max-laws 1000 \
  --max-news 20
