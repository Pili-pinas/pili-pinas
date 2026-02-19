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
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from vector_store import get_chroma_client, get_or_create_collection, upsert_documents

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"

# Multilingual model — handles both Filipino and English
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Batch size for embedding generation (tune based on RAM)
BATCH_SIZE = 64


def load_model() -> SentenceTransformer:
    """Load the multilingual sentence transformer."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded.")
    return model


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


def embed_collection(jsonl_path: Path, model: SentenceTransformer, collection) -> int:
    """
    Embed all chunks in a JSONL file and upsert into ChromaDB.
    Returns number of chunks processed.
    """
    docs = load_jsonl(jsonl_path)
    if not docs:
        logger.warning(f"No documents in {jsonl_path.name}")
        return 0

    logger.info(f"Embedding {len(docs)} chunks from {jsonl_path.name}")

    # Generate embeddings in batches
    texts = [d["text"] for d in docs]
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=jsonl_path.stem):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True).tolist()
        all_embeddings.extend(embeddings)

    # Prepare ChromaDB upsert args
    ids = [doc_id(d, i) for i, d in enumerate(docs)]
    metadatas = [d for d in docs]

    upsert_documents(
        collection=collection,
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return len(docs)


def run_embedding_pipeline(collections: list[str] | None = None) -> dict:
    """
    Embed all processed JSONL files (or a specified subset).

    Returns stats dict.
    """
    model = load_model()
    client = get_chroma_client()
    chroma_collection = get_or_create_collection(client)

    jsonl_files = list(PROCESSED_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {PROCESSED_DIR}. Run ingestion first.")
        return {}

    if collections:
        jsonl_files = [f for f in jsonl_files if f.stem in collections]

    stats = {}
    total = 0

    for path in jsonl_files:
        count = embed_collection(path, model, chroma_collection)
        stats[path.stem] = count
        total += count

    logger.info(f"\n=== Embedding complete: {total} total chunks in ChromaDB ===")
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
    args = parser.parse_args()

    stats = run_embedding_pipeline(collections=args.collections)
    total = sum(stats.values())
    print(f"\nDone. {total} chunks embedded into ChromaDB.")
