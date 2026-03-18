"""
RAG chain for Pili-Pinas.

Retrieval flow:
1. Embed user query with the multilingual sentence-transformer
2. Retrieve top-k similar chunks from ChromaDB
3. Build prompt with retrieved context
4. Call Claude Haiku via the Anthropic API
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

from embeddings.base import VectorStore
from embeddings.model import get_embedding_model
from embeddings.vector_store import get_vector_store
from retrieval.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE, NO_CONTEXT_RESPONSE, AGENTIC_SYSTEM_PROMPT
from retrieval.tools import TOOLS, execute_tool
from data_ingestion.document_index import DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_AGENTIC_TURNS = 5

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

_AGGREGATION_SIGNALS = [
    "most", "least", "how many", "count", "rank", "top ",
    "who has the most", "biggest advocate", "compare", "advocate for",
    "champion of", "filed the most", "authored the most",
]


def _is_aggregation_query(question: str) -> bool:
    """Return True if the question requires counting/ranking across all documents."""
    q = question.lower()
    return any(signal in q for signal in _AGGREGATION_SIGNALS)


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
        self._store: Optional[VectorStore] = None
        self._agentic_db_path = DB_PATH

    def _get_embedding_model(self) -> SentenceTransformer:
        return get_embedding_model()

    def _get_store(self) -> VectorStore:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    def retrieve(self, question: str, source_type: Optional[str] = None, n: Optional[int] = None) -> list[dict]:
        """
        Retrieve relevant chunks for the question.

        Args:
            question: User's question.
            source_type: Optional filter (e.g. 'bill', 'news', 'profile').
            n: Number of results (overrides self.top_k when provided).

        Returns:
            List of chunk dicts with text + metadata.
        """
        store = self._get_store()
        doc_count = store.count()
        if doc_count == 0:
            logger.warning("Vector store is empty — no documents to retrieve. Run /scrape to ingest data.")
            return []

        n_results = min(n if n is not None else self.top_k, doc_count)
        model = self._get_embedding_model()
        query_embedding = model.encode([question], normalize_embeddings=True).tolist()

        where = {"source_type": source_type} if source_type else None
        results = store.query(
            query_embeddings=query_embedding,
            n_results=n_results,
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
        """Call Claude Haiku via the Anthropic API."""
        import anthropic
        logger.info(f"Calling LLM: model={CLAUDE_MODEL} prompt_chars={len(user_prompt)}")
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text
        logger.info(f"LLM response received: {len(text)} chars, input_tokens={message.usage.input_tokens} output_tokens={message.usage.output_tokens}")
        return text

    def query_agentic(self, question: str) -> RAGResult:
        """
        Agentic RAG query using Claude tool use.
        Claude decides which tools to call (search_documents / query_database).
        Used for aggregation and ranking questions.

        Returns:
            RAGResult with answer and sources.
        """
        import anthropic
        logger.info(f"Agentic query: {question!r}")
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": question}]
        final_answer = ""

        for turn in range(MAX_AGENTIC_TURNS):
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=AGENTIC_SYSTEM_PROMPT,
                messages=messages,
                tools=TOOLS,
            )
            logger.info(f"Agentic turn {turn + 1}: stop_reason={response.stop_reason}")

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        final_answer = block.text
                        break
                break

            if response.stop_reason == "tool_use":
                # Append assistant message
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool call and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(f"Executing tool: {block.name} input={block.input}")
                        result = execute_tool(block.name, block.input, self, self._agentic_db_path)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason — extract any text and stop
                for block in response.content:
                    if hasattr(block, "text"):
                        final_answer = block.text
                break

        if not final_answer:
            final_answer = NO_CONTEXT_RESPONSE

        return RAGResult(
            answer=final_answer,
            sources=[],
            query=question,
            chunks_used=0,
        )

    def query(
        self,
        question: str,
        source_type: Optional[str] = None,
        min_score: float = 0.3,
    ) -> RAGResult:
        """
        Full RAG query: retrieve + generate.
        Routes aggregation questions to query_agentic().

        Args:
            question: User's question (Filipino, English, or Taglish).
            source_type: Optional filter for source type.
            min_score: Minimum similarity score to include a chunk.

        Returns:
            RAGResult with answer and sources.
        """
        logger.info(f"Query: {question!r}")

        if _is_aggregation_query(question):
            logger.info("Routing to agentic RAG (aggregation query detected)")
            return self.query_agentic(question)

        # Retrieve
        chunks = self.retrieve(question, source_type=source_type)
        logger.info(f"Retrieved {len(chunks)} chunks — scores: {[round(c['score'], 3) for c in chunks]}")
        relevant = [c for c in chunks if c["score"] >= min_score]
        logger.info(f"After min_score={min_score} filter: {len(relevant)} relevant chunks")

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
