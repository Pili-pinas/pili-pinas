"""
PDF processor: extracts text from Philippine government PDFs (laws, SALN, COMELEC docs).
Uses pdfplumber for reliable text extraction from scanned + native PDFs.
"""

import logging
from pathlib import Path

import pdfplumber

from .html_processor import chunk_text

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file.
    Returns concatenated text from all pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ""

    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"[Page {i + 1}]\n{text}")
                else:
                    logger.debug(f"No text on page {i + 1} of {pdf_path.name} (possibly scanned)")
    except Exception as e:
        logger.error(f"Failed to extract PDF {pdf_path}: {e}")
        return ""

    full_text = "\n\n".join(pages_text)
    logger.info(f"Extracted {len(pages_text)} pages from {pdf_path.name}")
    return full_text


def process_pdf_document(pdf_path: str | Path, metadata: dict) -> list[dict]:
    """
    Extract text from a PDF and return chunked document dicts.

    Args:
        pdf_path: Path to the PDF file.
        metadata: Document metadata dict (source, title, date, etc.)

    Returns:
        List of chunk dicts ready for embedding.
    """
    text = extract_pdf_text(pdf_path)
    if not text:
        return []

    chunks = chunk_text(text)
    logger.info(f"'{metadata.get('title', '')[:50]}' → {len(chunks)} chunks")

    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_doc = {
            **metadata,
            "text": chunk,
            "chunk_index": i,
            "chunk_total": len(chunks),
        }
        chunk_docs.append(chunk_doc)

    return chunk_docs


def download_and_process_pdf(url: str, dest_dir: Path, metadata: dict) -> list[dict]:
    """
    Download a PDF from a URL, save it, and process into chunks.
    """
    import time
    import requests

    headers = {
        "User-Agent": (
            "PiliPinas/1.0 (informed-voter research tool; "
            "contact: pilipinas-bot@example.com)"
        )
    }

    safe_name = url.split("/")[-1][:80] or "document.pdf"
    if not safe_name.endswith(".pdf"):
        safe_name += ".pdf"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / safe_name

    if dest_path.exists():
        logger.info(f"PDF already downloaded: {dest_path.name}")
    else:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
            logger.info(f"Downloaded PDF: {dest_path.name}")
            time.sleep(1.5)
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF {url}: {e}")
            return []

    return process_pdf_document(dest_path, metadata)
