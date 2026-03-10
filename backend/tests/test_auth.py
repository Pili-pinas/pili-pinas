"""
Tests for api/auth.py — API key validation via external auth service.

Mocks the requests.get call to avoid hitting the real auth service.
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

import api.auth as auth_module
from api.auth import verify_api_key


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the in-memory key cache before each test."""
    auth_module._valid_key_cache.clear()
    yield
    auth_module._valid_key_cache.clear()


def make_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_key_returns_key():
    with patch("api.auth.requests.get", return_value=make_response(200)) as mock_get:
        result = verify_api_key("good-token")
    assert result == "good-token"
    mock_get.assert_called_once_with(
        f"{auth_module.AUTH_SERVICE_URL}/api/tokens/validate/good-token",
        timeout=auth_module._AUTH_TIMEOUT,
    )


def test_valid_key_is_cached():
    with patch("api.auth.requests.get", return_value=make_response(200)) as mock_get:
        verify_api_key("cached-token")
        verify_api_key("cached-token")  # second call — should use cache
    assert mock_get.call_count == 1


def test_cache_expires_after_ttl():
    with patch("api.auth.requests.get", return_value=make_response(200)):
        verify_api_key("expiring-token")

    # Manually expire the cache entry
    auth_module._valid_key_cache["expiring-token"] = time.monotonic() - 1

    with patch("api.auth.requests.get", return_value=make_response(200)) as mock_get:
        verify_api_key("expiring-token")
    assert mock_get.call_count == 1  # re-validated after expiry


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

def test_invalid_key_raises_401():
    with patch("api.auth.requests.get", return_value=make_response(401, "Unauthorized")):
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("bad-token")
    assert exc_info.value.status_code == 401


def test_expired_key_raises_401():
    with patch("api.auth.requests.get", return_value=make_response(403, "Forbidden")):
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("expired-token")
    assert exc_info.value.status_code == 401


def test_invalid_key_not_cached():
    with patch("api.auth.requests.get", return_value=make_response(401)):
        with pytest.raises(HTTPException):
            verify_api_key("bad-token")
    assert "bad-token" not in auth_module._valid_key_cache


def test_previously_valid_key_removed_from_cache_on_rejection():
    # Seed cache as if key was valid
    auth_module._valid_key_cache["revoked-token"] = time.monotonic() - 1  # expired

    with patch("api.auth.requests.get", return_value=make_response(401)):
        with pytest.raises(HTTPException):
            verify_api_key("revoked-token")
    assert "revoked-token" not in auth_module._valid_key_cache


# ---------------------------------------------------------------------------
# Auth service unreachable
# ---------------------------------------------------------------------------

def test_auth_service_unreachable_raises_503():
    import requests as req
    with patch("api.auth.requests.get", side_effect=req.RequestException("timeout")):
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("any-token")
    assert exc_info.value.status_code == 503


def test_auth_service_unreachable_does_not_cache():
    import requests as req
    with patch("api.auth.requests.get", side_effect=req.RequestException("timeout")):
        with pytest.raises(HTTPException):
            verify_api_key("any-token")
    assert "any-token" not in auth_module._valid_key_cache


def test_retries_on_timeout_and_succeeds():
    """Succeeds on the third attempt after two timeouts."""
    import requests as req
    responses = [req.Timeout("slow"), req.Timeout("slow"), make_response(200)]
    with patch("api.auth.requests.get", side_effect=responses) as mock_get:
        result = verify_api_key("retry-token")
    assert result == "retry-token"
    assert mock_get.call_count == 3


def test_retries_exhaust_raises_503():
    """Raises 503 after all retry attempts fail."""
    import requests as req
    with patch("api.auth.requests.get", side_effect=req.Timeout("slow")) as mock_get:
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("retry-fail-token")
    assert exc_info.value.status_code == 503
    assert mock_get.call_count == auth_module.AUTH_MAX_RETRIES


def test_no_retry_on_401():
    """Auth rejection (401) should not be retried — it's not a transient error."""
    with patch("api.auth.requests.get", return_value=make_response(401)) as mock_get:
        with pytest.raises(HTTPException):
            verify_api_key("bad-token")
    assert mock_get.call_count == 1
