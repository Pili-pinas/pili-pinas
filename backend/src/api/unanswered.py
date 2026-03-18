"""Unanswered questions log backed by SQLite."""

import logging
import sqlite3
from datetime import datetime

from embeddings.vector_store import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

UNANSWERED_DB = VECTOR_DB_DIR / "unanswered.db"


def _init_unanswered_db() -> None:
    UNANSWERED_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(UNANSWERED_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unanswered_questions (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                asked_at TEXT    NOT NULL,
                question TEXT    NOT NULL,
                source_type TEXT
            )
        """)


def _log_unanswered(question: str, source_type: str | None) -> None:
    try:
        _init_unanswered_db()
        with sqlite3.connect(UNANSWERED_DB) as conn:
            conn.execute(
                "INSERT INTO unanswered_questions (asked_at, question, source_type) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), question, source_type),
            )
    except Exception:
        logger.exception("Failed to log unanswered question")


def get_unanswered_questions() -> list[dict]:
    """Return all unanswered questions ordered by most recent first."""
    _init_unanswered_db()
    with sqlite3.connect(UNANSWERED_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, asked_at, question, source_type FROM unanswered_questions ORDER BY asked_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
