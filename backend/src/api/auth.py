"""
API key authentication via external auth service.

Validates X-API-Key headers against:
    GET {AUTH_SERVICE_URL}/api/tokens/validate/{token}

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

AUTH_MAX_RETRIES = 3
_AUTH_TIMEOUT = 10  # seconds per attempt

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

    last_exc = None
    for attempt in range(1, AUTH_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{AUTH_SERVICE_URL}/api/tokens/validate/{api_key}",
                timeout=_AUTH_TIMEOUT,
            )
            last_exc = None
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(f"Auth service attempt {attempt}/{AUTH_MAX_RETRIES} failed: {exc}")

    if last_exc is not None:
        logger.error(f"Auth service unreachable after {AUTH_MAX_RETRIES} attempts: {last_exc}")
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
