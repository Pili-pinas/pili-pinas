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
sys.path.insert(0, str(Path(__file__).parents[1]))

from data_ingestion.scrapers.senate import scrape_bills as scrape_senate_bills, scrape_senators
from data_ingestion.scrapers.official_gazette import scrape_laws
from data_ingestion.scrapers.congress import scrape_house_bills, scrape_members
from data_ingestion.scrapers.comelec import scrape_all_comelec
from data_ingestion.scrapers.news_sites import scrape_all_news
from data_ingestion.scrapers.fact_check import scrape_all_fact_checks
from data_ingestion.scrapers.oversight import scrape_all_oversight
from data_ingestion.scrapers.statistics import scrape_all_statistics
from data_ingestion.scrapers.research import scrape_all_research
from data_ingestion.scrapers.financial import scrape_all_financial
from data_ingestion.scrapers.politicians import scrape_all_politicians
from data_ingestion.processors.html_processor import process_html_document
from data_ingestion.document_index import upsert_documents

logger = logging.getLogger(__name__)

_DEFAULT_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(_DEFAULT_PROCESSED_DIR)))
METADATA_FILE = PROCESSED_DIR.parent / "metadata.json"
FAILED_LOG = PROCESSED_DIR.parent / "failed_urls.log"
CHECKPOINT_FILE = Path(os.getenv(
    "BACKFILL_CHECKPOINT",
    str(Path(__file__).parents[3] / "data" / "backfill_checkpoint.json"),
))

# Default congress/year values for daily scrape
_CURRENT_CONGRESS = 20
_CURRENT_ELECTION_YEAR = 2025


class BackfillCheckpoint:
    """
    Tracks completed backfill steps so interrupted runs can be resumed.

    Steps use the format:
      "senate_bills:16"  — per-congress bill sources
      "comelec:2019"     — per-year COMELEC
      "gazette"          — flat sources (single run per backfill)

    The checkpoint file is written after every completed step so progress
    is preserved even if the process is killed mid-run.
    """

    def __init__(self, path: Path, resume: bool = False):
        self.path = path
        self._completed: set[str] = set()
        if resume and path.exists():
            try:
                data = json.loads(path.read_text())
                self._completed = set(data.get("completed", []))
                logger.info(f"Resuming: {len(self._completed)} steps already done")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e} — starting fresh")
        elif not resume and path.exists():
            path.unlink()

    def is_done(self, step: str) -> bool:
        return step in self._completed

    def mark_done(self, step: str) -> None:
        self._completed.add(step)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "completed": sorted(self._completed),
            "last_updated": datetime.now().isoformat(),
        }, indent=2))
        logger.info(f"Checkpoint: {step} done")


def save_documents(docs: list[dict], collection: str, append: bool = False) -> Path:
    """Save processed document chunks to a JSONL file."""
    out_path = PROCESSED_DIR / f"{collection}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(docs)} chunks → {out_path}")
    return out_path


def _load_bills_from_jsonl() -> list[dict]:
    """
    Load bill docs from saved JSONL files for use when resuming past a completed
    bills step. Returns a flat list of processed chunks (politician field preserved).
    """
    bills = []
    for collection in ("senate_bills", "house_bills"):
        path = PROCESSED_DIR / f"{collection}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        bills.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    logger.info(f"Loaded {len(bills)} bill docs from JSONL for politician enrichment")
    return bills


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
    congresses: list[int] | None = None,
    election_years: list[int] | None = None,
    max_pages: int = 3,
    max_news: int = 20,
    max_laws: int = 50,
    gazette_from_year: int | None = None,
    resume: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        sources: Which sources to ingest. None = all.
                 Options: senate_bills, senators, gazette, house_bills,
                          house_members, comelec, news
        congresses: Congress numbers to scrape for bills (e.g. [18, 19, 20]).
                    None defaults to [current congress]. Pass multiple for backfill.
        election_years: Election years to scrape for COMELEC (e.g. [2019, 2022, 2025]).
                        None defaults to [current year].
        max_pages: Max items per congress session for bill scrapers.
        max_news: Max articles per news source.
        max_laws: Max laws to fetch from the Official Gazette.

    Returns:
        Stats dict with counts per collection.
    """
    all_sources = [
        "senate_bills", "senators", "gazette",
        "house_bills", "house_members", "comelec", "news",
        "fact_check", "oversight", "statistics", "research", "financial",
        "politicians",
    ]
    sources = sources or all_sources
    congresses = congresses or [_CURRENT_CONGRESS]
    election_years = election_years or [_CURRENT_ELECTION_YEAR]
    stats = {"sources": sources, "counts": {}}

    checkpoint = BackfillCheckpoint(CHECKPOINT_FILE, resume=resume)

    # Collect raw bill docs across all bill scrapers so politician profiles
    # can be enriched with authored-bill history in a single pass.
    _collected_bills: list[dict] = []
    # Track which JSONL collections have been written this run (for append mode).
    _written: set[str] = set()

    def _process_and_save(raw_docs: list[dict], collection: str) -> int:
        """Process raw docs into chunks and save (append if collection already written)."""
        chunks = []
        for doc in raw_docs:
            if doc.get("text"):
                chunks.extend(process_html_document(doc))
            else:
                logger.warning(f"Skipping doc with no text: {doc.get('title', '')[:50]}")
        if chunks:
            save_documents(chunks, collection, append=collection in _written)
            _written.add(collection)
        if raw_docs:
            upsert_documents(raw_docs)
        stats["counts"][collection] = stats["counts"].get(collection, 0) + len(chunks)
        return len(chunks)

    if "senate_bills" in sources:
        logger.info(f"=== Senate Bills (congresses={congresses}) ===")
        for c in congresses:
            step = f"senate_bills:{c}"
            if checkpoint.is_done(step):
                logger.info(f"Skipping {step} (already completed)")
                continue
            logger.info(f"Scraping Senate bills — Congress {c}")
            docs = scrape_senate_bills(congress=c, max_items=max_pages)
            _process_and_save(docs, "senate_bills")
            _collected_bills.extend(docs)
            checkpoint.mark_done(step)

    if "senators" in sources:
        if "politicians" in sources:
            logger.info("=== Senator Profiles — skipped (covered by politicians) ===")
        elif not checkpoint.is_done("senators"):
            logger.info("=== Senator Profiles ===")
            docs = scrape_senators()
            _process_and_save(docs, "senators")
            checkpoint.mark_done("senators")
        else:
            logger.info("Skipping senators (already completed)")

    if "gazette" in sources:
        if not checkpoint.is_done("gazette"):
            logger.info(f"=== Official Gazette Laws (max_laws={max_laws}, from_year={gazette_from_year}) ===")
            docs = scrape_laws(max_items=max_laws, from_year=gazette_from_year)
            _process_and_save(docs, "gazette_laws")
            checkpoint.mark_done("gazette")
        else:
            logger.info("Skipping gazette (already completed)")

    if "house_bills" in sources:
        logger.info(f"=== House Bills (congresses={congresses}) ===")
        for c in congresses:
            step = f"house_bills:{c}"
            if checkpoint.is_done(step):
                logger.info(f"Skipping {step} (already completed)")
                continue
            logger.info(f"Scraping House bills — Congress {c}")
            docs = scrape_house_bills(congress=c, max_items=max_pages)
            _process_and_save(docs, "house_bills")
            _collected_bills.extend(docs)
            checkpoint.mark_done(step)

    if "house_members" in sources:
        if "politicians" in sources:
            logger.info("=== House Members — skipped (covered by politicians) ===")
        elif not checkpoint.is_done("house_members"):
            logger.info("=== House Members ===")
            docs = scrape_members()
            _process_and_save(docs, "house_members")
            checkpoint.mark_done("house_members")
        else:
            logger.info("Skipping house_members (already completed)")

    if "comelec" in sources:
        logger.info(f"=== COMELEC (election_years={election_years}) ===")
        for year in election_years:
            step = f"comelec:{year}"
            if checkpoint.is_done(step):
                logger.info(f"Skipping {step} (already completed)")
                continue
            logger.info(f"Scraping COMELEC — {year}")
            docs = scrape_all_comelec(election_year=year, max_resolutions=30)
            if docs:
                save_documents(docs, "comelec", append="comelec" in _written)
                _written.add("comelec")
            stats["counts"]["comelec"] = stats["counts"].get("comelec", 0) + len(docs)
            checkpoint.mark_done(step)

    for source, collection, scrape_fn, fn_kwargs in [
        ("news",       "news_articles", lambda: scrape_all_news(max_items_per_source=max_news), {}),
        ("fact_check", "fact_checks",   lambda: scrape_all_fact_checks(max_items=max_news), {}),
        ("oversight",  "oversight",     lambda: scrape_all_oversight(max_items=max_news), {}),
        ("statistics", "statistics",    lambda: scrape_all_statistics(max_items=max_news), {}),
        ("research",   "research",      lambda: scrape_all_research(max_items=max_news), {}),
        ("financial",  "financial",     lambda: scrape_all_financial(max_items=max_news), {}),
    ]:
        if source not in sources:
            continue
        if checkpoint.is_done(source):
            logger.info(f"Skipping {source} (already completed)")
            continue
        logger.info(f"=== {source.replace('_', ' ').title()} ===")
        docs = scrape_fn()
        _process_and_save(docs, collection)
        checkpoint.mark_done(source)

    if "politicians" in sources:
        if not checkpoint.is_done("politicians"):
            # If all bill steps were skipped on resume, load from saved JSONL
            if not _collected_bills:
                _collected_bills = _load_bills_from_jsonl()
            logger.info(f"=== Politician Profiles (bills available: {len(_collected_bills)}) ===")
            docs = scrape_all_politicians(bills=_collected_bills)
            _process_and_save(docs, "politicians")
            checkpoint.mark_done("politicians")
        else:
            logger.info("Skipping politicians (already completed)")

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
                 "house_members", "comelec", "news",
                 "fact_check", "oversight", "statistics", "research", "financial",
                 "politicians"],
        help="Sources to ingest (default: all)"
    )
    parser.add_argument("--congresses", nargs="+", type=int, default=None,
                        help="Congress numbers to scrape (default: current congress)")
    parser.add_argument("--election-years", nargs="+", type=int, default=None,
                        help="Election years for COMELEC (default: current year)")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-news", type=int, default=20)
    parser.add_argument("--max-laws", type=int, default=50)
    parser.add_argument("--gazette-from-year", type=int, default=None,
                        help="Stop gazette scraping at laws older than this year (e.g. 2006)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint instead of starting fresh")
    args = parser.parse_args()

    stats = run_ingestion(
        sources=args.sources,
        congresses=args.congresses,
        election_years=args.election_years,
        max_pages=args.max_pages,
        max_news=args.max_news,
        max_laws=args.max_laws,
        gazette_from_year=args.gazette_from_year,
        resume=args.resume,
    )
    print(f"\nDone. Total chunks: {stats['total_chunks']}")
