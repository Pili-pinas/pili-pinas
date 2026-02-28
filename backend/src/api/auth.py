"""
API key authentication via external auth service.

Validates X-API-Key headers against:
    GET {AUTH_SERVICE_URL}/api/settings/validate-key

Set AUTH_SERVICE_URL in .env (defaults to the Vercel auth service).
"""

import os
import time
import logging

import requests
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

AUTH_SERVICE_URL = os.getenv(
    "AUTH_SERVICE_URL",
    "https://rag-pipeline-91ct.vercel.app",
)

# Cache validated keys for this many seconds before re-checking the auth service.
_CACHE_TTL = int(os.getenv("AUTH_CACHE_TTL", "86400"))  # default: 24 hours

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# key → expiry timestamp (monotonic)
_valid_key_cache: dict[str, float] = {}


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency — validates the X-API-Key header against the auth service.

    Validated keys are cached in memory for AUTH_CACHE_TTL seconds (default 5 min)
    to avoid a round-trip to the auth service on every request.

    Returns the key on success. Raises HTTPException on failure.
    """
    now = time.monotonic()
    if _valid_key_cache.get(api_key, 0) > now:
        return api_key

    try:
        resp = requests.get(
            f"{AUTH_SERVICE_URL}/api/settings/validate-key",
            headers={"X-API-Key": api_key},
            timeout=5,
        )
    except requests.RequestException as exc:
        logger.error(f"Auth service unreachable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable. Try again shortly.",
        )

    if resp.status_code == 200:
        _valid_key_cache[api_key] = now + _CACHE_TTL
        return api_key

    # Remove stale cache entry if key was previously valid but now isn't.
    _valid_key_cache.pop(api_key, None)
    logger.warning(f"Key validation failed: {resp.status_code} {resp.text[:120]}")
    raise HTTPException(status_code=401, detail="Invalid or expired API key.")
