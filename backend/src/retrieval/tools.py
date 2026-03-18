"""
Tool definitions and executor for the agentic RAG chain.

Two tools are exposed to Claude:
  - search_documents: semantic search via ChromaDB
  - query_database:   read-only SQL queries against the SQLite document index
"""

import json
import logging
import re
import sqlite3
from pathlib import Path

from data_ingestion.document_index import DB_PATH

logger = logging.getLogger(__name__)

TOOLS: list[dict] = [
    {
        "name": "search_documents",
        "description": (
            "Semantic search across all ingested Philippine government documents "
            "(bills, laws, news, senator profiles, COMELEC records). "
            "Returns the most relevant text chunks with source metadata. "
            "Use this to find specific content, quotes, or background on a topic or person."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (English, Filipino, or Taglish)",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of results to return (default 10, max 20)",
                    "default": 10,
                },
                "source_type": {
                    "type": "string",
                    "description": "Optional filter: bill, law, news, profile, election",
                    "enum": ["bill", "law", "news", "profile", "election"],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_database",
        "description": (
            "Run a read-only SQL SELECT query against the documents database. "
            "Use this for counting, ranking, and grouping — e.g. 'who filed the most bills'. "
            "Tables available:\n"
            "  documents(url, title, source, source_type, date, politician, congress)\n"
            "  document_topics(url, topic)\n"
            "Topics include: women's rights, education, health, agriculture, infrastructure, "
            "anti-corruption, environment, labor, social welfare, peace and order, economy, housing, youth.\n"
            "Always include a LIMIT clause."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A read-only SELECT query. Must start with SELECT.",
                },
            },
            "required": ["sql"],
        },
    },
]


def execute_tool(
    name: str,
    tool_input: dict,
    rag,
    db_path: Path = DB_PATH,
) -> str:
    """
    Execute a tool call and return the result as a JSON string.

    Args:
        name:       Tool name (search_documents or query_database)
        tool_input: Tool arguments from the Claude API response
        rag:        PiliPinasRAG instance (needed for search_documents)
        db_path:    Path to the SQLite document index

    Returns:
        JSON string with result or {"error": "..."} on failure
    """
    if name == "search_documents":
        return _search_documents(tool_input, rag)
    elif name == "query_database":
        return _query_database(tool_input, db_path)
    else:
        logger.warning(f"Unknown tool called: {name}")
        return json.dumps({"error": f"Unknown tool: {name}"})


def _search_documents(tool_input: dict, rag) -> str:
    query = tool_input.get("query", "")
    n = min(int(tool_input.get("n", 10)), 20)
    source_type = tool_input.get("source_type")

    try:
        chunks = rag.retrieve(query, n=n, source_type=source_type)
        result = [
            {
                "text": c["text"],
                "title": c["metadata"].get("title", ""),
                "url": c["metadata"].get("url", ""),
                "source": c["metadata"].get("source", ""),
                "date": c["metadata"].get("date", ""),
                "score": round(c["score"], 3),
            }
            for c in chunks
        ]
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"search_documents error: {e}")
        return json.dumps({"error": str(e)})


def _query_database(tool_input: dict, db_path: Path) -> str:
    sql = tool_input.get("sql", "").strip()

    # Only allow SELECT statements
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        return json.dumps({"error": "Only SELECT queries are allowed."})

    # Auto-add LIMIT if missing
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        sql = sql.rstrip(";") + " LIMIT 50"

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
        return json.dumps(rows, ensure_ascii=False)
    except sqlite3.Error as e:
        logger.error(f"query_database error: {e} | sql={sql}")
        return json.dumps({"error": str(e)})
