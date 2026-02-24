"""
Abstract base class for vector store backends.

Implement this interface to add a new backend (Turso, Pinecone, etc.).
ChromaDB is the default implementation in vector_store.py.
"""

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """
    Minimal interface every vector store backend must implement.

    Query results must follow this shape (matches ChromaDB's format):
        {
            "documents": [["chunk text", ...]],
            "metadatas": [[{"source": ..., "title": ...}, ...]],
            "distances": [[0.05, 0.12, ...]],   # cosine distance, lower = more similar
        }
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Collection / table name."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Insert or update document chunks with their embeddings."""

    @abstractmethod
    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Return the top-n most similar chunks for the given query embedding."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks stored."""
