"""
Embedding pipeline.
Reads processed JSONL chunks, generates embeddings using the multilingual
sentence-transformers model, and stores them in ChromaDB.

Run:
    python backend/src/embeddings/create_embeddings.py
"""

import json
import logging
import hashlib
import os
import sys
import argparse
from pathlib import Path

# Ensure 'src/' is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parents[1]))

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from embeddings.base import VectorStore
from embeddings.model import get_embedding_model
from embeddings.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_DEFAULT_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(_DEFAULT_PROCESSED_DIR)))

# Batch size for embedding generation.
# CPU default: 64. GPU (MPS/CUDA) users should pass --batch-size 256 or higher.
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))


def load_model() -> SentenceTransformer:
    """Return the shared embedding model singleton."""
    return get_embedding_model()


def doc_id(doc: dict, idx: int) -> str:
    """Generate a stable unique ID for a chunk."""
    key = f"{doc.get('url', '')}__{doc.get('chunk_index', idx)}"
    return hashlib.md5(key.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load documents from a JSONL file."""
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line in {path.name}: {e}")
    return docs


def embed_collection(jsonl_path: Path, model: SentenceTransformer, store: VectorStore,
                     batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """
    Embed all chunks in a JSONL file and upsert into the vector store.
    Returns number of chunks processed.
    """
    docs = load_jsonl(jsonl_path)
    if not docs:
        logger.warning(f"No documents in {jsonl_path.name}")
        return 0

    logger.info(f"Embedding {len(docs)} chunks from {jsonl_path.name}")

    # Embed and upsert in batches to keep peak memory low
    for i in tqdm(range(0, len(docs), batch_size), desc=jsonl_path.stem):
        batch_docs = docs[i : i + batch_size]
        batch_texts = [d["text"] for d in batch_docs]
        batch_ids = [doc_id(d, i + j) for j, d in enumerate(batch_docs)]

        # Deduplicate within the batch — ChromaDB errors on duplicate IDs in one call
        seen: dict[str, int] = {}
        for pos, uid in enumerate(batch_ids):
            seen[uid] = pos  # last writer wins
        unique_positions = sorted(seen.values())
        batch_ids = [batch_ids[p] for p in unique_positions]
        batch_texts = [batch_texts[p] for p in unique_positions]
        batch_docs = [batch_docs[p] for p in unique_positions]

        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        store.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_docs,
        )

    return len(docs)


def run_embedding_pipeline(collections: list[str] | None = None,
                           batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """
    Embed all processed JSONL files (or a specified subset).

    Returns stats dict.
    """
    model = load_model()
    store = get_vector_store()

    jsonl_files = list(PROCESSED_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {PROCESSED_DIR}. Run ingestion first.")
        return {}

    if collections:
        jsonl_files = [f for f in jsonl_files if f.stem in collections]

    stats = {}
    total = 0

    for path in jsonl_files:
        count = embed_collection(path, model, store, batch_size=batch_size)
        stats[path.stem] = count
        total += count

    logger.info(f"\n=== Embedding complete: {total} total chunks in {store.name} ===")
    for name, count in stats.items():
        logger.info(f"  {name}: {count} chunks")

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build ChromaDB vector embeddings")
    parser.add_argument(
        "--collections", nargs="+",
        help="JSONL collection names to embed (default: all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE}). Use 256+ for GPU."
    )
    args = parser.parse_args()

    stats = run_embedding_pipeline(collections=args.collections, batch_size=args.batch_size)
    total = sum(stats.values())
    print(f"\nDone. {total} chunks embedded into ChromaDB.")
