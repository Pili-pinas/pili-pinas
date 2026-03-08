"""
Facebook Messenger bot integration for Pili-Pinas.

Protocol:
  - GET  /messenger/webhook  — Meta hub verification handshake
  - POST /messenger/webhook  — Receive page message events

Flow for incoming text messages:
  1. Verify X-Hub-Signature-256 HMAC-SHA256 (skip in dev when secret is empty)
  2. Filter: only handle text messages from "page" object events
  3. Send typing indicator immediately (best-effort)
  4. Return HTTP 200 to Meta within the required response window
  5. BackgroundTask: run RAG query → send formatted answer via Graph API

Environment variables:
  META_APP_SECRET        — App Secret for signature verification
  META_PAGE_ACCESS_TOKEN — Page Access Token for sending messages
  META_VERIFY_TOKEN      — Arbitrary string set when registering the webhook
"""

import hashlib
import hmac
import logging
import os
from typing import Optional

import requests
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from api.auth import verify_api_key
from retrieval.rag_chain import RAGResult, get_rag

logger = logging.getLogger(__name__)

META_APP_SECRET = os.getenv("META_APP_SECRET", "")
META_PAGE_ACCESS_TOKEN = os.getenv("META_PAGE_ACCESS_TOKEN", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "pilipinas_verify")

GRAPH_API_URL = "https://graph.facebook.com/v21.0/me/messages"

# Messenger text message limit
MAX_MESSAGE_LEN = 2000

messenger_router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verify_signature(body: bytes, signature: str) -> bool:
    """
    Verify the X-Hub-Signature-256 header from Meta.

    In dev mode (empty META_APP_SECRET), always returns True.
    The signature format is "sha256=<hex_digest>".
    """
    if not META_APP_SECRET:
        return True
    if not signature.startswith("sha256="):
        logger.warning(f"Signature missing sha256= prefix: {signature!r}")
        return False
    expected = "sha256=" + hmac.new(
        META_APP_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    match = hmac.compare_digest(expected, signature)
    if not match:
        logger.warning(
            f"Signature mismatch — secret_len={len(META_APP_SECRET)} "
            f"body_len={len(body)} "
            f"expected={expected[:20]}... received={signature[:20]}..."
        )
    return match


def send_message(recipient_id: str, text: str) -> None:
    """Send a text message to a Messenger user via the Graph API."""
    try:
        requests.post(
            GRAPH_API_URL,
            params={"access_token": META_PAGE_ACCESS_TOKEN},
            json={
                "recipient": {"id": recipient_id},
                "message": {"text": text},
                "messaging_type": "RESPONSE",
            },
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Failed to send Messenger message to {recipient_id}: {e}")


def _send_typing(recipient_id: str) -> None:
    """Send a typing indicator (best-effort — failure is non-fatal)."""
    try:
        requests.post(
            GRAPH_API_URL,
            params={"access_token": META_PAGE_ACCESS_TOKEN},
            json={
                "recipient": {"id": recipient_id},
                "sender_action": "typing_on",
            },
            timeout=5,
        )
    except Exception:
        pass  # typing indicator failure is not critical


def format_answer(result: RAGResult) -> str:
    """
    Format a RAGResult for Messenger (≤2,000 chars).

    Includes the answer and up to 3 source citations.
    """
    sources = result.sources[:3]
    source_lines = "\n".join(
        f"{i + 1}. {s['title']} — {s['url']}"
        for i, s in enumerate(sources)
    )

    if source_lines:
        full = f"{result.answer}\n\n📚 Sources:\n{source_lines}"
    else:
        full = result.answer

    if len(full) > MAX_MESSAGE_LEN:
        full = full[: MAX_MESSAGE_LEN - 3] + "..."

    return full


def _answer_and_reply(sender_id: str, question: str) -> None:
    """Background task: run RAG query and send formatted answer."""
    try:
        rag = get_rag()
        result = rag.query(question)
        reply = format_answer(result)
    except Exception as e:
        logger.exception(f"Messenger RAG failed for {sender_id}: {e}")
        reply = "Sorry, I couldn't process your question. Please try again later."

    send_message(sender_id, reply)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@messenger_router.get("/webhook")
def verify_webhook(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_verify_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default=""),
):
    """
    Meta webhook verification handshake.

    Meta sends a GET request when you first register the webhook URL.
    Respond with hub.challenge to confirm ownership.
    """
    if hub_mode == "subscribe" and hub_verify_token == META_VERIFY_TOKEN:
        logger.info("Messenger webhook verified.")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed.")


@messenger_router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive Messenger events from Meta.

    Verifies the HMAC signature, extracts text messages, sends a typing
    indicator, queues the RAG reply as a background task, then returns
    HTTP 200 immediately (required within Meta's response window).
    """
    body = await request.body()

    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(body, sig):
        raise HTTPException(status_code=403, detail="Invalid signature.")

    event = await request.json()

    # Only process Page subscription events
    if event.get("object") != "page":
        return {"status": "ok"}

    for entry in event.get("entry", []):
        for messaging in entry.get("messaging", []):
            message = messaging.get("message", {})

            # Only handle plain text messages; ignore attachments, postbacks, etc.
            if "text" not in message:
                continue

            sender_id: str = messaging["sender"]["id"]
            question: str = message["text"].strip()

            _send_typing(sender_id)
            background_tasks.add_task(_answer_and_reply, sender_id, question)

    return {"status": "ok"}
