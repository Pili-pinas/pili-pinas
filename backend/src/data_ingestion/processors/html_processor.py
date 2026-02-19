"""
HTML processor: cleans raw HTML and splits text into chunks for embedding.
"""

import re
import logging
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Chunk size in characters (~300–400 tokens)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def clean_html(html: str) -> str:
    """
    Extract clean text from raw HTML.
    Strips scripts, styles, nav, footer, and excessive whitespace.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to split on paragraph/sentence boundaries where possible.
    """
    if not text:
        return []

    # Split on double newlines (paragraphs) first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        # If a single paragraph exceeds chunk_size, split it by sentences
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_html_document(doc: dict) -> list[dict]:
    """
    Process a raw document dict (with 'text' as raw HTML or plain text).
    Returns a list of chunk dicts, each inheriting the original metadata.
    """
    raw_text = doc.get("text", "")

    # If text looks like HTML, clean it; otherwise use as-is
    if raw_text.lstrip().startswith("<"):
        clean = clean_html(raw_text)
    else:
        clean = raw_text.strip()

    if not clean:
        logger.warning(f"Empty text after processing: {doc.get('title', 'untitled')}")
        return []

    chunks = chunk_text(clean)
    logger.debug(f"'{doc.get('title', '')[:50]}' → {len(chunks)} chunks")

    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_doc = {**doc, "text": chunk, "chunk_index": i, "chunk_total": len(chunks)}
        chunk_docs.append(chunk_doc)

    return chunk_docs
