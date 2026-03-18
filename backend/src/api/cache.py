"""Query result cache backed by SQLite."""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime

from embeddings.vector_store import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

QUERY_CACHE_DB = VECTOR_DB_DIR / "query_cache.db"

# Maps scrape source names to the document domains they produce.
_SCRAPE_SOURCE_DOMAINS: dict[str, set[str]] = {
    "news":          {"rappler.com", "philstar.com", "bworldonline.com", "gmanetwork.com", "pcij.org"},
    "senate_bills":  {"senate.gov.ph"},
    "senators":      {"senate.gov.ph", "en.wikipedia.org"},
    "gazette":       {"elibrary.judiciary.gov.ph"},
    "house_bills":   {"open-congress-api.bettergov.ph"},
    "house_members": {"congress.gov.ph", "en.wikipedia.org"},
    "comelec":       {"comelec.gov.ph"},
}


def _cache_key(question: str, source_type: str | None) -> str:
    normalized = question.lower().strip()
    return hashlib.md5(f"{normalized}|{source_type or ''}".encode()).hexdigest()


def _init_cache_db() -> None:
    QUERY_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(QUERY_CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key   TEXT PRIMARY KEY,
                question    TEXT NOT NULL,
                source_type TEXT,
                answer      TEXT NOT NULL,
                sources     TEXT NOT NULL,
                chunks_used INTEGER NOT NULL,
                cached_at   TEXT NOT NULL,
                hit_count   INTEGER DEFAULT 0
            )
        """)


def _cache_get(question: str, source_type: str | None) -> dict | None:
    try:
        _init_cache_db()
        key = _cache_key(question, source_type)
        with sqlite3.connect(QUERY_CACHE_DB) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT answer, sources, chunks_used FROM query_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE query_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (key,),
                )
                return {
                    "answer": row["answer"],
                    "sources": json.loads(row["sources"]),
                    "chunks_used": row["chunks_used"],
                }
    except Exception:
        logger.exception("Cache get failed")
    return None


def _cache_set(question: str, source_type: str | None, result) -> None:
    try:
        _init_cache_db()
        key = _cache_key(question, source_type)
        with sqlite3.connect(QUERY_CACHE_DB) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO query_cache
                   (cache_key, question, source_type, answer, sources, chunks_used, cached_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    key, question, source_type, result.answer,
                    json.dumps([s.model_dump() for s in result.sources]),
                    result.chunks_used, datetime.now().isoformat(),
                ),
            )
    except Exception:
        logger.exception("Cache set failed")


def _cache_clear(scraped_sources: list[str] | None = None) -> None:
    """Clear cache entries whose sources overlap with the scraped sources.

    If scraped_sources is None, clears the entire cache.
    """
    try:
        _init_cache_db()
        with sqlite3.connect(QUERY_CACHE_DB) as conn:
            if scraped_sources is None:
                deleted = conn.execute("DELETE FROM query_cache").rowcount
                logger.info(f"Query cache fully cleared: {deleted} entries removed")
                return

            domains: set[str] = set()
            for src in scraped_sources:
                domains |= _SCRAPE_SOURCE_DOMAINS.get(src, set())

            if not domains:
                return

            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT cache_key, sources FROM query_cache").fetchall()
            keys_to_delete = [
                row["cache_key"]
                for row in rows
                if set(s.get("source", "") for s in json.loads(row["sources"])) & domains
            ]
            if keys_to_delete:
                placeholders = ",".join("?" * len(keys_to_delete))
                conn.execute(f"DELETE FROM query_cache WHERE cache_key IN ({placeholders})", keys_to_delete)
            logger.info(f"Selective cache clear: {len(keys_to_delete)} entries removed for sources={scraped_sources}")
    except Exception:
        logger.exception("Cache clear failed")


def cache_clear_all() -> int:
    """Delete all cached entries. Returns the number of deleted rows."""
    _init_cache_db()
    with sqlite3.connect(QUERY_CACHE_DB) as conn:
        deleted = conn.execute("DELETE FROM query_cache").rowcount
    logger.info(f"Cache manually cleared: {deleted} entries removed")
    return deleted


def get_popular_questions(limit: int) -> list[dict]:
    """Return the most frequently asked questions, sorted by ask count descending."""
    _init_cache_db()
    with sqlite3.connect(QUERY_CACHE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT question, source_type, hit_count + 1 AS total_asks, cached_at
               FROM query_cache
               ORDER BY hit_count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
