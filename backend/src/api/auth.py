"""
API key authentication via external auth service.

Validates X-API-Key headers against:
    GET {AUTH_SERVICE_URL}/api/settings/validate-key

Set AUTH_SERVICE_URL in .env (defaults to the Vercel auth service).
"""

import os
import logging

import requests
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

AUTH_SERVICE_URL = os.getenv(
    "AUTH_SERVICE_URL",
    "https://rag-pipeline-91ct.vercel.app",
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency — validates the X-API-Key header against the auth service.

    Returns the key on success. Raises HTTPException on failure.
    """
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
        return api_key

    logger.warning(f"Key validation failed: {resp.status_code} {resp.text[:120]}")
    raise HTTPException(status_code=401, detail="Invalid or expired API key.")
