#!/usr/bin/env bash
# scripts/install.sh
# Create virtualenv and install all project dependencies using uv.
#
# Usage:
#   ./scripts/install.sh          # install all deps
#   ./scripts/install.sh --dev    # include dev/test deps

set -euo pipefail

info() { echo "[info]  $*"; }

# ── virtualenv ────────────────────────────────────────────────────────────────

if [[ ! -d ".venv" ]]; then
    info "Creating .venv with Python 3.11..."
    uv venv .venv --python 3.11
else
    info ".venv already exists — skipping creation."
fi

# ── dependencies ──────────────────────────────────────────────────────────────

info "Installing backend dependencies..."
uv pip install -r backend/requirements.txt

info "Installing frontend dependencies..."
uv pip install -r frontend/requirements.txt

if [[ "${1:-}" == "--dev" ]]; then
    info "Installing dev/test dependencies..."
    uv pip install -r backend/requirements-dev.txt
fi

# ── done ──────────────────────────────────────────────────────────────────────

info "=== Done ==="
info "Activate with: source .venv/bin/activate"
