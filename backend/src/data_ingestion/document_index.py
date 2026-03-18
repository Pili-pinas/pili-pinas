"""
SQLite document index for structured queries and topic tagging.

Stores one row per unique ingested document (not per chunk) with author and topic metadata.
Used by the agentic RAG chain to answer aggregation questions like:
  - "Who filed the most bills?"
  - "Who is the biggest advocate for women's rights?"

Tables:
    documents       — one row per URL (title, source, source_type, date, politician, congress)
    document_topics — many-to-many: document ↔ topic labels
"""

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parents[2] / "vector_db" / "document_index.db"
DB_PATH = Path(_DEFAULT_DB_PATH)

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "women's rights": [
        "women", "gender equality", "reproductive", "maternal", "vawc",
        "anti-violence against women", "violence against women",
        "magna carta of women", "solo parent", "gender",
    ],
    "education": [
        "education", "school", "scholarship", "k-12", "tesda", "ched",
        "free tuition", "alternative learning", "learner", "student",
    ],
    "health": [
        "health", "healthcare", "philhealth", "universal health",
        "mental health", "medicine", "hospital", "pandemic", "disease",
    ],
    "agriculture": [
        "agriculture", "farmer", "agrarian", "rice", "corn",
        "fishing", "fishermen", "magsasaka", "crop", "livestock",
    ],
    "infrastructure": [
        "infrastructure", "road", "bridge", "right-of-way", "dpwh",
        "public works", "flood control", "expressway", "airport",
    ],
    "anti-corruption": [
        "corruption", "plunder", "graft", "sandiganbayan", "ombudsman",
        "saln", "ill-gotten", "bribery", "kickback",
    ],
    "environment": [
        "environment", "climate", "green", "carbon", "pollution",
        "forest", "biodiversity", "plastic", "waste", "clean air",
    ],
    "labor": [
        "labor", "workers", "ofw", "overseas filipino", "minimum wage",
        "employment", "trabaho", "worker", "manpower", "overseas",
    ],
    "social welfare": [
        "social welfare", "dswd", "indigent", "poverty", "4ps",
        "senior citizen", "person with disability", "pwd", "pantawid",
    ],
    "peace and order": [
        "peace", "security", "police", "military", "terrorism",
        "drug", "criminality", "crime", "anti-drug",
    ],
    "economy": [
        "economy", "tax", "business", "trade", "investment",
        "gdp", "inflation", "tariff", "fiscal", "budget",
    ],
    "housing": [
        "housing", "shelter", "resettlement", "informal settler",
        "socialized housing", "urban poor", "home",
    ],
    "youth": [
        "youth", "kabataan", "student", "sk", "sangguniang kabataan",
        "young", "children", "child",
    ],
}


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                url         TEXT PRIMARY KEY,
                title       TEXT,
                source      TEXT,
                source_type TEXT,
                date        TEXT,
                politician  TEXT,
                congress    INTEGER,
                ingested_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS document_topics (
                url   TEXT NOT NULL,
                topic TEXT NOT NULL,
                PRIMARY KEY (url, topic)
            );
        """)
        conn.commit()
    finally:
        conn.close()


def tag_topics(doc: dict) -> list[str]:
    """Return topic labels for a document based on keyword matching."""
    text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
    matched = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(topic)
    return matched


def upsert_documents(raw_docs: list[dict], db_path: Path = DB_PATH) -> int:
    """
    Write one row per unique URL into the documents table.
    Skips documents without a URL. Idempotent (INSERT OR REPLACE).

    Returns number of rows written.
    """
    if not raw_docs:
        return 0

    db_path = Path(db_path)
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    written = 0
    try:
        for doc in raw_docs:
            url = doc.get("url", "").strip()
            if not url:
                continue

            topics = tag_topics(doc)

            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (url, title, source, source_type, date, politician, congress)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url,
                    doc.get("title", ""),
                    doc.get("source", ""),
                    doc.get("source_type", ""),
                    doc.get("date", ""),
                    doc.get("politician", ""),
                    doc.get("congress"),
                ),
            )

            # Re-insert topics (delete old first to handle updates)
            conn.execute("DELETE FROM document_topics WHERE url=?", (url,))
            for topic in topics:
                conn.execute(
                    "INSERT OR IGNORE INTO document_topics (url, topic) VALUES (?, ?)",
                    (url, topic),
                )

            written += 1

        conn.commit()
    finally:
        conn.close()

    logger.debug(f"document_index: upserted {written} documents")
    return written
