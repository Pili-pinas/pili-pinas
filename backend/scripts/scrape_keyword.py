"""
Keyword-targeted scraper: fetch and embed documents mentioning a specific keyword.

Runs all (or selected) scrapers, post-filters results to only documents where the
keyword appears in the title or text, then processes and embeds the matches.

Use this to fill coverage gaps found by check_coverage.py.

Usage:
    cd backend
    uv run python scripts/scrape_keyword.py "Leni Robredo"
    uv run python scripts/scrape_keyword.py "Leni Robredo" --sources news senate_bills senators
    uv run python scripts/scrape_keyword.py "pork barrel" --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data_ingestion.scrapers.news_sites import scrape_all_news
from data_ingestion.scrapers.senate import scrape_bills as scrape_senate_bills, scrape_senators
from data_ingestion.scrapers.congress import scrape_house_bills, scrape_members
from data_ingestion.scrapers.official_gazette import scrape_laws
from data_ingestion.scrapers.comelec import scrape_all_comelec
from data_ingestion.scrapers.fact_check import scrape_all_fact_checks
from data_ingestion.scrapers.oversight import scrape_all_oversight
from data_ingestion.scrapers.statistics import scrape_all_statistics
from data_ingestion.scrapers.research import scrape_all_research
from data_ingestion.scrapers.financial import scrape_all_financial

logger = logging.getLogger(__name__)

# Map source name → scraper callable (no-arg call returns list[dict])
SCRAPER_MAP: dict[str, callable] = {
    "news":         scrape_all_news,
    "senate_bills": scrape_senate_bills,
    "senators":     scrape_senators,
    "gazette":      scrape_laws,
    "house_bills":  scrape_house_bills,
    "house_members": scrape_members,
    "comelec":      scrape_all_comelec,
    "fact_check":   scrape_all_fact_checks,
    "oversight":    scrape_all_oversight,
    "statistics":   scrape_all_statistics,
    "research":     scrape_all_research,
    "financial":    scrape_all_financial,
}


def filter_by_keyword(docs: list[dict], keyword: str) -> list[dict]:
    """Keep only docs where keyword appears (case-insensitive) in title or text."""
    term = keyword.lower()
    return [
        doc for doc in docs
        if term in doc.get("title", "").lower()
        or term in doc.get("text", "").lower()
    ]


def scrape_and_filter(
    keyword: str,
    sources: list[str] | None = None,
    scraper_map: dict | None = None,
) -> list[dict]:
    """
    Run each scraper and return only documents that mention the keyword.

    Args:
        keyword: Term to search for (case-insensitive).
        sources: Which scrapers to run. None = all in scraper_map.
        scraper_map: Override for testing. Defaults to the module-level SCRAPER_MAP.
    """
    smap = scraper_map if scraper_map is not None else SCRAPER_MAP
    active_sources = sources if sources is not None else list(smap.keys())

    matched = []
    for source in active_sources:
        fn = smap.get(source)
        if fn is None:
            logger.warning(f"Unknown source '{source}' — skipping")
            continue

        logger.info(f"[{source}] Scraping...")
        try:
            raw_docs = fn()
        except Exception as e:
            logger.error(f"[{source}] Scraper failed: {e}")
            continue

        filtered = filter_by_keyword(raw_docs, keyword)
        logger.info(f"[{source}] {len(filtered)}/{len(raw_docs)} docs match '{keyword}'")
        matched.extend(filtered)

    return matched


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Scrape and embed documents matching a keyword."
    )
    parser.add_argument("keyword", help="Keyword to search for (e.g. 'Leni Robredo')")
    parser.add_argument(
        "--sources", nargs="+", choices=list(SCRAPER_MAP.keys()),
        help="Which scrapers to run (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print matched docs without saving or embedding",
    )
    args = parser.parse_args()

    logger.info(f"Keyword: '{args.keyword}'  Sources: {args.sources or 'all'}")
    matched = scrape_and_filter(args.keyword, sources=args.sources)

    if not matched:
        logger.warning(f"No documents found for '{args.keyword}'")
        sys.exit(1)

    logger.info(f"Total matched: {len(matched)} documents")

    if args.dry_run:
        for doc in matched:
            print(f"  [{doc.get('source_type')}] {doc.get('title', '(no title)')} — {doc.get('source')}")
        return

    # Process, save, and embed
    from data_ingestion.ingestion import save_documents
    from data_ingestion.processors.html_processor import process_html_document
    from data_ingestion.document_index import upsert_documents

    chunks = []
    for doc in matched:
        if doc.get("text"):
            chunks.extend(process_html_document(doc))
        else:
            logger.warning(f"Skipping doc with no text: {doc.get('title', '')[:60]}")

    if not chunks:
        logger.warning("No chunks produced after processing.")
        sys.exit(1)

    slug = args.keyword.lower().replace(" ", "_")[:40]
    save_documents(chunks, f"keyword_{slug}")
    upsert_documents(matched)

    logger.info(f"Done — {len(chunks)} chunks saved and embedded for '{args.keyword}'")


if __name__ == "__main__":
    main()
