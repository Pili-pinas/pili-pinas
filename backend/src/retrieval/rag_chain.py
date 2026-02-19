"""
RAG chain for Pili-Pinas.

Retrieval flow:
1. Embed user query with the multilingual sentence-transformer
2. Retrieve top-k similar chunks from ChromaDB
3. Build prompt with retrieved context
4. Call LLM (Ollama locally, Claude Haiku in production)
5. Return answer with source citations

Usage:
    from retrieval.rag_chain import PiliPinasRAG
    rag = PiliPinasRAG()
    result = rag.query("Sino si Leni Robredo?")
    print(result["answer"])
    print(result["sources"])
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from sentence_transformers import SentenceTransformer

from embeddings.vector_store import get_chroma_client, get_or_create_collection, query_collection
from retrieval.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE, NO_CONTEXT_RESPONSE

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 5

# LLM backend: "ollama" (dev) or "claude" (prod)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")


@dataclass
class RAGResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    chunks_used: int = 0


class PiliPinasRAG:
    """Main RAG class — loads once and handles all queries."""

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self._embedding_model: Optional[SentenceTransformer] = None
        self._collection = None

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _get_collection(self):
        if self._collection is None:
            client = get_chroma_client()
            self._collection = get_or_create_collection(client)
        return self._collection

    def retrieve(self, question: str, source_type: Optional[str] = None) -> list[dict]:
        """
        Retrieve top-k relevant chunks for the question.

        Args:
            question: User's question.
            source_type: Optional filter (e.g. 'bill', 'news', 'profile').

        Returns:
            List of chunk dicts with text + metadata.
        """
        model = self._get_embedding_model()
        query_embedding = model.encode([question], normalize_embeddings=True).tolist()

        where = {"source_type": source_type} if source_type else None
        results = query_collection(
            collection=self._get_collection(),
            query_embeddings=query_embedding,
            n_results=self.top_k,
            where=where,
        )

        chunks = []
        if results and results.get("documents"):
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                chunks.append({
                    "text": doc_text,
                    "metadata": meta,
                    "score": 1 - distance,  # convert cosine distance to similarity
                })

        return chunks

    def _build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a readable context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            source_line = f"[{i}] {meta.get('title', 'Untitled')} | {meta.get('source', '')} | {meta.get('date', '')}"
            url_line = f"URL: {meta.get('url', 'N/A')}"
            parts.append(f"{source_line}\n{url_line}\n\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Route to the configured LLM backend."""
        if LLM_BACKEND == "claude":
            return self._call_claude(system_prompt, user_prompt)
        return self._call_ollama(system_prompt, user_prompt)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call local Ollama instance."""
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=OLLAMA_MODEL)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            return llm.invoke(full_prompt)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"Error: Could not reach Ollama. Make sure it's running with `ollama serve`. ({e})"

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude API (production)."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"Error calling Claude API: {e}"

    def query(
        self,
        question: str,
        source_type: Optional[str] = None,
        min_score: float = 0.3,
    ) -> RAGResult:
        """
        Full RAG query: retrieve + generate.

        Args:
            question: User's question (Filipino, English, or Taglish).
            source_type: Optional filter for source type.
            min_score: Minimum similarity score to include a chunk.

        Returns:
            RAGResult with answer and sources.
        """
        logger.info(f"Query: {question!r}")

        # Retrieve
        chunks = self.retrieve(question, source_type=source_type)
        relevant = [c for c in chunks if c["score"] >= min_score]

        if not relevant:
            logger.info("No relevant chunks found above threshold.")
            return RAGResult(
                answer=NO_CONTEXT_RESPONSE,
                sources=[],
                query=question,
                chunks_used=0,
            )

        # Build prompt
        context = self._build_context(relevant)
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # Generate
        answer = self._call_llm(RAG_SYSTEM_PROMPT, user_prompt)

        # Extract source list for the response
        sources = [
            {
                "title": c["metadata"].get("title", ""),
                "url": c["metadata"].get("url", ""),
                "source": c["metadata"].get("source", ""),
                "date": c["metadata"].get("date", ""),
                "score": round(c["score"], 3),
            }
            for c in relevant
        ]

        return RAGResult(
            answer=answer,
            sources=sources,
            query=question,
            chunks_used=len(relevant),
        )


# Module-level singleton for reuse across API requests
_rag_instance: Optional[PiliPinasRAG] = None


def get_rag() -> PiliPinasRAG:
    """Return the shared RAG instance (lazy init)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PiliPinasRAG()
    return _rag_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = PiliPinasRAG()
    result = rag.query("What bills has the Senate passed about education?")
    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== SOURCES ===")
    for s in result.sources:
        print(f"  - {s['title']} ({s['url']})")
