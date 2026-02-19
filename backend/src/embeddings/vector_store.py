"""
ChromaDB vector store wrapper.
Handles collection creation, upserts, and similarity search.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

VECTOR_DB_DIR = Path(__file__).parents[3] / "vector_db"
COLLECTION_NAME = "pili_pinas"


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create the main vector collection."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )


def upsert_documents(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert document chunks into ChromaDB."""
    # ChromaDB metadata values must be str/int/float/bool
    clean_metadatas = [_sanitize_metadata(m) for m in metadatas]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=clean_metadatas,
    )
    logger.info(f"Upserted {len(ids)} chunks into '{collection.name}'")


def query_collection(
    collection: chromadb.Collection,
    query_embeddings: list[list[float]],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Query ChromaDB for similar chunks.

    Returns ChromaDB results dict with documents, metadatas, distances.
    """
    kwargs: dict[str, Any] = {
        "query_embeddings": query_embeddings,
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str/int/float/bool metadata values.
    Convert everything else to string.
    """
    clean = {}
    for k, v in meta.items():
        if k in ("text", "chunk_index", "chunk_total"):
            continue  # text goes in documents; chunk info is fine as int
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean
