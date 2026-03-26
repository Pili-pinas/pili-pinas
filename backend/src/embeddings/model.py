"""
Shared SentenceTransformer singleton.

Both the RAG chain and the embedding pipeline use this so the model
is loaded once per process, not once per caller.

Device selection order:
  1. EMBEDDING_DEVICE env var (explicit override)
  2. CUDA  — NVIDIA GPU
  3. MPS   — Apple Silicon GPU
  4. CPU   — fallback
"""

import logging
import os
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model_instance: Optional[SentenceTransformer] = None


def _detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    override = os.getenv("EMBEDDING_DEVICE", "").strip()
    if override:
        return override

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model_instance
    if _model_instance is None:
        device = _detect_device()
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (device={device})")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info(f"Embedding model loaded on {device}.")
    return _model_instance
