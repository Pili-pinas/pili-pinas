"""
Print the current ChromaDB vector count.

Usage:
    python backend/scripts/vector_count.py
    PYTHONPATH=backend/src python backend/scripts/vector_count.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from embeddings.vector_store import get_vector_store

store = get_vector_store()
count = store.count()
print(f"ChromaDB vectors: {count:,} ({store.name})")
