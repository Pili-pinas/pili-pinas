"""
Tests for api/messenger.py (Facebook Messenger integration)

All Meta Graph API calls and RAG calls are mocked.

Meta Messenger webhook protocol:
  - GET  /messenger/webhook  — Meta hub verification (hub.challenge handshake)
  - POST /messenger/webhook  — Receive message events

Signature verification uses real HMAC-SHA256 so correct/wrong cases are real.
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.auth import verify_api_key
from retrieval.rag_chain import RAGResult


# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

APP_SECRET = "test_app_secret_abc123"
VERIFY_TOKEN = "my_verify_token"
PAGE_ACCESS_TOKEN = "EAAtest123"

# Standard Messenger webhook payload
TEXT_MESSAGE_EVENT = {
    "object": "page",
    "entry": [{
        "id": "page_123",
        "time": 1700000000,
        "messaging": [{
            "sender": {"id": "user_psid_abc"},
            "recipient": {"id": "page_123"},
            "timestamp": 1700000000,
            "message": {"mid": "msg_001", "text": "Sino si Leni Robredo?"},
        }]
    }]
}

# Attachment event (should be ignored)
ATTACHMENT_EVENT = {
    "object": "page",
    "entry": [{
        "id": "page_123",
        "time": 1700000001,
        "messaging": [{
            "sender": {"id": "user_psid_abc"},
            "recipient": {"id": "page_123"},
            "timestamp": 1700000001,
            "message": {
                "mid": "msg_002",
                "attachments": [{"type": "image", "payload": {"url": "https://cdn.fb.com/img.jpg"}}],
            },
        }]
    }]
}

# Postback event (button click — should be ignored)
POSTBACK_EVENT = {
    "object": "page",
    "entry": [{
        "id": "page_123",
        "time": 1700000002,
        "messaging": [{
            "sender": {"id": "user_psid_abc"},
            "recipient": {"id": "page_123"},
            "timestamp": 1700000002,
            "postback": {"title": "Get Started", "payload": "GET_STARTED"},
        }]
    }]
}

SAMPLE_RAG_RESULT = RAGResult(
    answer="Leni Robredo ay isang Pilipinong politiko at dating Pangalawang Pangulo.",
    sources=[
        {"title": "Senator Profile: Leni Robredo", "url": "https://wikipedia.org/wiki/Leni_Robredo",
         "source": "wikipedia.org", "date": "2025-01-01", "score": 0.92},
        {"title": "Robredo runs for president", "url": "https://rappler.com/1",
         "source": "rappler.com", "date": "2025-02-01", "score": 0.85},
    ],
    query="Sino si Leni Robredo?",
    chunks_used=2,
)

NO_SOURCE_RAG_RESULT = RAGResult(
    answer="Walang impormasyon.",
    sources=[],
    query="some question",
    chunks_used=0,
)


def _make_signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture(autouse=True)
def bypass_auth():
    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


# ---------------------------------------------------------------------------
# TestVerifySignature
# ---------------------------------------------------------------------------

class TestVerifySignature:
    def test_correct_signature_returns_true(self):
        from api.messenger import verify_signature
        body = b'{"object":"page"}'
        sig = _make_signature(body, APP_SECRET)
        with patch("api.messenger.META_APP_SECRET", APP_SECRET):
            assert verify_signature(body, sig) is True

    def test_wrong_signature_returns_false(self):
        from api.messenger import verify_signature
        body = b'{"object":"page"}'
        with patch("api.messenger.META_APP_SECRET", APP_SECRET):
            assert verify_signature(body, "sha256=deadbeef") is False

    def test_missing_sha256_prefix_returns_false(self):
        from api.messenger import verify_signature
        body = b'{"object":"page"}'
        raw = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
        with patch("api.messenger.META_APP_SECRET", APP_SECRET):
            assert verify_signature(body, raw) is False  # no "sha256=" prefix

    def test_empty_secret_always_returns_true(self):
        from api.messenger import verify_signature
        with patch("api.messenger.META_APP_SECRET", ""):
            assert verify_signature(b"anything", "sha256=garbage") is True


# ---------------------------------------------------------------------------
# TestFormatAnswer
# ---------------------------------------------------------------------------

class TestFormatAnswer:
    def test_includes_answer_text(self):
        from api.messenger import format_answer
        text = format_answer(SAMPLE_RAG_RESULT)
        assert "Leni Robredo" in text

    def test_includes_source_urls(self):
        from api.messenger import format_answer
        text = format_answer(SAMPLE_RAG_RESULT)
        assert "wikipedia.org/wiki/Leni_Robredo" in text

    def test_includes_source_titles(self):
        from api.messenger import format_answer
        text = format_answer(SAMPLE_RAG_RESULT)
        assert "Senator Profile: Leni Robredo" in text

    def test_shows_at_most_three_sources(self):
        from api.messenger import format_answer
        result = RAGResult(
            answer="Answer.",
            sources=[{"title": f"S{i}", "url": f"https://s{i}.com", "source": "x",
                      "date": "2025-01-01", "score": 0.9} for i in range(5)],
            query="q",
            chunks_used=5,
        )
        text = format_answer(result)
        assert text.count("https://s") <= 3

    def test_handles_no_sources(self):
        from api.messenger import format_answer
        text = format_answer(NO_SOURCE_RAG_RESULT)
        assert "Walang impormasyon" in text

    def test_truncates_to_2000_chars(self):
        from api.messenger import format_answer
        long_result = RAGResult(
            answer="A" * 3000,
            sources=[],
            query="q",
            chunks_used=0,
        )
        text = format_answer(long_result)
        assert len(text) <= 2000


# ---------------------------------------------------------------------------
# TestSendMessage
# ---------------------------------------------------------------------------

class TestSendMessage:
    def test_posts_to_graph_api(self):
        from api.messenger import send_message
        with patch("api.messenger.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            send_message("user_psid", "Hello!")
        url = mock_post.call_args[0][0]
        assert "graph.facebook.com" in url
        assert "messages" in url

    def test_sends_recipient_id_and_text(self):
        from api.messenger import send_message
        with patch("api.messenger.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            send_message("user_psid_abc", "Hello!")
        payload = mock_post.call_args[1]["json"]
        assert payload["recipient"]["id"] == "user_psid_abc"
        assert payload["message"]["text"] == "Hello!"

    def test_includes_page_access_token_in_params(self):
        from api.messenger import send_message
        with patch("api.messenger.requests.post") as mock_post, \
             patch("api.messenger.META_PAGE_ACCESS_TOKEN", PAGE_ACCESS_TOKEN):
            mock_post.return_value = MagicMock(status_code=200)
            send_message("user_psid", "Hello!")
        params = mock_post.call_args[1].get("params") or mock_post.call_args[0][1]
        assert params["access_token"] == PAGE_ACCESS_TOKEN


# ---------------------------------------------------------------------------
# TestMessengerWebhookVerification (GET)
# ---------------------------------------------------------------------------

class TestMessengerWebhookVerification:
    def test_returns_challenge_with_correct_verify_token(self):
        with patch("api.messenger.META_VERIFY_TOKEN", VERIFY_TOKEN):
            resp = client.get(
                "/messenger/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": VERIFY_TOKEN,
                    "hub.challenge": "challenge_abc",
                },
            )
        assert resp.status_code == 200
        assert resp.text == "challenge_abc"

    def test_returns_403_with_wrong_verify_token(self):
        with patch("api.messenger.META_VERIFY_TOKEN", VERIFY_TOKEN):
            resp = client.get(
                "/messenger/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "wrong_token",
                    "hub.challenge": "challenge_abc",
                },
            )
        assert resp.status_code == 403

    def test_returns_403_without_subscribe_mode(self):
        with patch("api.messenger.META_VERIFY_TOKEN", VERIFY_TOKEN):
            resp = client.get(
                "/messenger/webhook",
                params={
                    "hub.mode": "unsubscribe",
                    "hub.verify_token": VERIFY_TOKEN,
                    "hub.challenge": "challenge_abc",
                },
            )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# TestMessengerWebhookEvents (POST)
# ---------------------------------------------------------------------------

class TestMessengerWebhookEvents:
    def _post(self, event: dict, secret: str = ""):
        body = json.dumps(event).encode()
        sig = _make_signature(body, secret) if secret else "sha256=nosig"
        return client.post(
            "/messenger/webhook",
            content=body,
            headers={"Content-Type": "application/json",
                     "X-Hub-Signature-256": sig},
        )

    def test_text_message_returns_200(self):
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing"), \
             patch("api.messenger._answer_and_reply"):
            resp = self._post(TEXT_MESSAGE_EVENT)
        assert resp.status_code == 200

    def test_attachment_event_is_ignored(self):
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing") as mock_typing, \
             patch("api.messenger._answer_and_reply") as mock_reply:
            resp = self._post(ATTACHMENT_EVENT)
        assert resp.status_code == 200
        mock_typing.assert_not_called()
        mock_reply.assert_not_called()

    def test_postback_event_is_ignored(self):
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing") as mock_typing, \
             patch("api.messenger._answer_and_reply") as mock_reply:
            resp = self._post(POSTBACK_EVENT)
        assert resp.status_code == 200
        mock_typing.assert_not_called()
        mock_reply.assert_not_called()

    def test_invalid_signature_returns_403(self):
        with patch("api.messenger.META_APP_SECRET", APP_SECRET):
            body = json.dumps(TEXT_MESSAGE_EVENT).encode()
            resp = client.post(
                "/messenger/webhook",
                content=body,
                headers={"Content-Type": "application/json",
                         "X-Hub-Signature-256": "sha256=wrong"},
            )
        assert resp.status_code == 403

    def test_valid_signature_passes(self):
        with patch("api.messenger.META_APP_SECRET", APP_SECRET), \
             patch("api.messenger._send_typing"), \
             patch("api.messenger._answer_and_reply"):
            resp = self._post(TEXT_MESSAGE_EVENT, secret=APP_SECRET)
        assert resp.status_code == 200

    def test_no_secret_skips_signature_check(self):
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing"), \
             patch("api.messenger._answer_and_reply"):
            body = json.dumps(TEXT_MESSAGE_EVENT).encode()
            resp = client.post(
                "/messenger/webhook",
                content=body,
                headers={"Content-Type": "application/json",
                         "X-Hub-Signature-256": "sha256=garbage"},
            )
        assert resp.status_code == 200

    def test_typing_indicator_sent_for_text_message(self):
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing") as mock_typing, \
             patch("api.messenger._answer_and_reply"):
            self._post(TEXT_MESSAGE_EVENT)
        mock_typing.assert_called_once_with("user_psid_abc")

    def test_background_task_called_with_sender_and_question(self):
        """TestClient runs background tasks synchronously."""
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing"), \
             patch("api.messenger._answer_and_reply") as mock_reply:
            self._post(TEXT_MESSAGE_EVENT)
        mock_reply.assert_called_once_with("user_psid_abc", "Sino si Leni Robredo?")

    def test_non_page_object_is_ignored(self):
        event = {**TEXT_MESSAGE_EVENT, "object": "user"}
        with patch("api.messenger.META_APP_SECRET", ""), \
             patch("api.messenger._send_typing") as mock_typing:
            resp = self._post(event)
        assert resp.status_code == 200
        mock_typing.assert_not_called()


# ---------------------------------------------------------------------------
# TestAnswerAndReply (background task)
# ---------------------------------------------------------------------------

class TestAnswerAndReply:
    def test_sends_formatted_answer_to_user(self):
        from api.messenger import _answer_and_reply
        mock_rag = MagicMock()
        mock_rag.query.return_value = SAMPLE_RAG_RESULT
        with patch("api.messenger.get_rag", return_value=mock_rag), \
             patch("api.messenger.send_message") as mock_send:
            _answer_and_reply("user_psid", "Sino si Leni?")
        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == "user_psid"
        assert "Leni Robredo" in mock_send.call_args[0][1]

    def test_sends_error_message_when_rag_fails(self):
        from api.messenger import _answer_and_reply
        with patch("api.messenger.get_rag", side_effect=Exception("DB down")), \
             patch("api.messenger.send_message") as mock_send:
            _answer_and_reply("user_psid", "question")
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][1].lower()
        assert "sorry" in msg or "error" in msg or "try again" in msg
