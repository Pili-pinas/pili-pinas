#!/usr/bin/env bash
# scripts/fly-deploy.sh
# Deploy pili-pinas-api to Fly.io.
# Cleans up stuck/unattached volumes before deploying so the machine
# can always be placed in a zone with available capacity.
#
# Usage:
#   ./scripts/fly-deploy.sh              # deploy
#   ./scripts/fly-deploy.sh --setup      # first-time setup (login + secrets) then deploy
#   FLYCTL_NO_IPV6=1 ./scripts/fly-deploy.sh  # force IPv4 if your ISP blocks IPv6

set -euo pipefail

APP="pili-pinas-api"
VOLUME_NAME="vector_db"
REGION="sin"
VOLUME_SIZE_GB=3

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[info]  $*"; }
warn()  { echo "[warn]  $*" >&2; }
error() { echo "[error] $*" >&2; exit 1; }

fly_cmd() { FLYCTL_NO_IPV6=1 fly "$@"; }

# ── prerequisites ─────────────────────────────────────────────────────────────

command -v fly &>/dev/null || error "flyctl not found. Install: brew install flyctl"

if ! fly_cmd auth whoami &>/dev/null; then
    info "Not logged in to Fly.io — opening browser..."
    fly_cmd auth login
fi

# ── first-time setup (--setup flag) ──────────────────────────────────────────

if [[ "${1:-}" == "--setup" ]]; then
    info "=== First-time setup ==="

    if ! fly_cmd apps list 2>/dev/null | grep -q "^$APP"; then
        info "Creating app '$APP'..."
        fly_cmd launch --no-deploy --name "$APP" --region sin
    else
        info "App '$APP' already exists — skipping launch."
    fi

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        read -rsp "Enter your ANTHROPIC_API_KEY: " ANTHROPIC_API_KEY
        echo
    fi
    info "Setting ANTHROPIC_API_KEY secret..."
    fly_cmd secrets set "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" --app "$APP"

    info "Setup complete. Proceeding to deploy..."
fi

# ── clean up stuck / unattached volumes ──────────────────────────────────────

info "Checking for unattached '$VOLUME_NAME' volumes..."

# List volumes; keep only those with no attached VM (ATTACHED VM column is empty)
STUCK_IDS=$(
    fly_cmd volumes list --app "$APP" --json 2>/dev/null \
    | python3 -c "
import json, sys
vols = json.load(sys.stdin)
stuck = [v['id'] for v in vols if v.get('name') == '$VOLUME_NAME' and not v.get('attached_machine_id')]
print('\n'.join(stuck))
" || true
)

if [[ -n "$STUCK_IDS" ]]; then
    while IFS= read -r vol_id; do
        [[ -z "$vol_id" ]] && continue
        warn "Destroying unattached volume $vol_id (no data — safe to remove)..."
        fly_cmd volumes destroy "$vol_id" --app "$APP" --yes
    done <<< "$STUCK_IDS"
else
    info "No stuck volumes found."
fi

# ── ensure at least one volume exists ────────────────────────────────────────

VOLUME_COUNT=$(
    fly_cmd volumes list --app "$APP" --json 2>/dev/null \
    | python3 -c "
import json, sys
vols = json.load(sys.stdin)
print(sum(1 for v in vols if v.get('name') == '$VOLUME_NAME'))
" || echo "0"
)

if [[ "$VOLUME_COUNT" == "0" ]]; then
    info "No '$VOLUME_NAME' volume found — creating one in region $REGION..."
    fly_cmd volumes create "$VOLUME_NAME" --app "$APP" --region "$REGION" --size "$VOLUME_SIZE_GB" --yes
    info "Volume created."
else
    info "Volume exists ($VOLUME_COUNT found)."
fi

# ── deploy ────────────────────────────────────────────────────────────────────

info "Deploying $APP..."
fly_cmd deploy --remote-only

info "=== Deploy complete ==="
info "Tail logs:   FLYCTL_NO_IPV6=1 fly logs --app $APP"
info "Health:      FLYCTL_NO_IPV6=1 fly status --app $APP"
info "SSH:         FLYCTL_NO_IPV6=1 fly ssh console --app $APP"
