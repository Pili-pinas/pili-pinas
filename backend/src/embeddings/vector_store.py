"""
Vector store implementations and factory.

Usage:
    from embeddings.vector_store import get_vector_store

    store = get_vector_store()
    store.upsert(ids, embeddings, documents, metadatas)
    results = store.query(query_embeddings, n_results=5)

To add a new backend, subclass VectorStore from embeddings.base and register
it in get_vector_store() below. Set VECTOR_STORE_BACKEND in .env to switch.
"""

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from embeddings.base import VectorStore

logger = logging.getLogger(__name__)

VECTOR_DB_DIR = Path(__file__).parents[3] / "vector_db"
COLLECTION_NAME = "pili_pinas"


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store (default)."""

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        db_path = path or str(VECTOR_DB_DIR)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def name(self) -> str:
        return self._collection.name

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        clean = [self._sanitize_metadata(m) for m in metadatas]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean,
        )
        logger.info(f"Upserted {len(ids)} chunks into '{self.name}'")

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """ChromaDB only accepts str/int/float/bool metadata values."""
        clean = {}
        for k, v in meta.items():
            if k in ("text", "chunk_index", "chunk_total"):
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


# ── Factory ───────────────────────────────────────────────────────────────────

def get_vector_store() -> VectorStore:
    """
    Return the configured vector store backend.

    Reads VECTOR_STORE_BACKEND from the environment (default: "chroma").
    To add a new backend, implement VectorStore and register it here.
    """
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    if backend == "chroma":
        return ChromaVectorStore()
    raise ValueError(
        f"Unknown VECTOR_STORE_BACKEND: {backend!r}. "
        "Supported: 'chroma'"
    )
