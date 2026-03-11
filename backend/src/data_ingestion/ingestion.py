"""
Main ingestion pipeline.
Orchestrates all scrapers, processes documents, and saves to data/processed/.

Run:
    python backend/src/data_ingestion/ingestion.py
"""

import json
import logging
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Ensure 'src/' is on the path when run as a script (python ingestion.py)
sys.path.insert(0, str(Path(__file__).parents[2]))

from data_ingestion.scrapers.senate import scrape_bills as scrape_senate_bills, scrape_senators
from data_ingestion.scrapers.official_gazette import scrape_laws
from data_ingestion.scrapers.congress import scrape_house_bills, scrape_members
from data_ingestion.scrapers.comelec import scrape_all_comelec
from data_ingestion.scrapers.news_sites import scrape_all_news
from data_ingestion.processors.html_processor import process_html_document

logger = logging.getLogger(__name__)

_DEFAULT_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(_DEFAULT_PROCESSED_DIR)))
METADATA_FILE = PROCESSED_DIR.parent / "metadata.json"
FAILED_LOG = PROCESSED_DIR.parent / "failed_urls.log"


def save_documents(docs: list[dict], collection: str) -> Path:
    """Save processed document chunks to a JSONL file."""
    out_path = PROCESSED_DIR / f"{collection}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(docs)} chunks → {out_path}")
    return out_path


def update_metadata(stats: dict) -> None:
    """Append ingestion run stats to metadata.json."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if METADATA_FILE.exists():
        try:
            existing = json.loads(METADATA_FILE.read_text())
        except json.JSONDecodeError:
            existing = []

    existing.append({"timestamp": datetime.now().isoformat(), **stats})
    METADATA_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


def run_ingestion(
    sources: list[str] | None = None,
    congress: int = 19,
    max_pages: int = 3,
    max_news: int = 20,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        sources: Which sources to ingest. None = all.
                 Options: senate_bills, senators, gazette, house_bills,
                          house_members, comelec, news
        congress: Congress number for bill scrapers.
        max_pages: Max pages to scrape per source.
        max_news: Max articles per news source.

    Returns:
        Stats dict with counts per collection.
    """
    all_sources = [
        "senate_bills", "senators", "gazette",
        "house_bills", "house_members", "comelec", "news"
    ]
    sources = sources or all_sources
    stats = {"sources": sources, "counts": {}}

    def _process_and_save(raw_docs: list[dict], collection: str) -> int:
        """Process raw docs into chunks and save."""
        chunks = []
        for doc in raw_docs:
            if doc.get("text"):
                chunks.extend(process_html_document(doc))
            else:
                logger.warning(f"Skipping doc with no text: {doc.get('title', '')[:50]}")
        if chunks:
            save_documents(chunks, collection)
        stats["counts"][collection] = len(chunks)
        return len(chunks)

    if "senate_bills" in sources:
        logger.info("=== Senate Bills ===")
        docs = scrape_senate_bills(congress=congress, max_items=max_pages)
        _process_and_save(docs, "senate_bills")

    if "senators" in sources:
        logger.info("=== Senator Profiles ===")
        docs = scrape_senators()
        _process_and_save(docs, "senators")

    if "gazette" in sources:
        logger.info("=== Official Gazette Laws ===")
        docs = scrape_laws(max_items=max_pages)
        _process_and_save(docs, "gazette_laws")

    if "house_bills" in sources:
        logger.info("=== House Bills ===")
        docs = scrape_house_bills(congress=congress, max_items=max_pages)
        _process_and_save(docs, "house_bills")

    if "house_members" in sources:
        logger.info("=== House Members ===")
        docs = scrape_members()
        _process_and_save(docs, "house_members")

    if "comelec" in sources:
        logger.info("=== COMELEC (candidates + resolutions) ===")
        # scrape_all_comelec returns pre-chunked PDF docs — save directly
        docs = scrape_all_comelec(election_year=2025, max_resolutions=30)
        if docs:
            save_documents(docs, "comelec")
        stats["counts"]["comelec"] = len(docs)

    if "news" in sources:
        logger.info("=== News Articles ===")
        docs = scrape_all_news(max_items_per_source=max_news)
        _process_and_save(docs, "news_articles")

    total = sum(stats["counts"].values())
    stats["total_chunks"] = total
    update_metadata(stats)

    logger.info(f"\n=== Ingestion complete: {total} total chunks ===")
    for collection, count in stats["counts"].items():
        logger.info(f"  {collection}: {count} chunks")

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Pili-Pinas data ingestion pipeline")
    parser.add_argument(
        "--sources", nargs="+",
        choices=["senate_bills", "senators", "gazette", "house_bills",
                 "house_members", "comelec", "news"],
        help="Sources to ingest (default: all)"
    )
    parser.add_argument("--congress", type=int, default=19)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-news", type=int, default=20)
    args = parser.parse_args()

    stats = run_ingestion(
        sources=args.sources,
        congress=args.congress,
        max_pages=args.max_pages,
        max_news=args.max_news,
    )
    print(f"\nDone. Total chunks: {stats['total_chunks']}")
