"""
COMELEC scraper.

Scrapes candidate lists and election resolutions from:
  - comelec.gov.ph  (official; JS-heavy — only static PDF links are extracted)
  - lawphil.net     (reliable mirror of all COMELEC resolutions as PDF)

Strategy
--------
comelec.gov.ph renders candidate list pages via JavaScript, so a plain
requests fetch only returns the navigation skeleton — no candidate data.
We handle this in two ways:

  1. Candidate lists — fetch COMELEC pages and extract any embedded PDF
     links from the static HTML.  If the page requires JavaScript and
     no links are found, we log a warning.  (Add Playwright if you need
     full JS rendering in the future.)

  2. COMELEC Resolutions — scraped from lawphil.net, which mirrors all
     resolutions as server-side-rendered HTML + linked PDFs, so it works
     reliably with plain requests.

robots.txt: comelec.gov.ph has no restrictive robots.txt for public pages.
Rate limit: 1.5 seconds between requests.
"""

import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from data_ingestion.processors.pdf_processor import download_and_process_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMELEC_BASE = "https://comelec.gov.ph"
LAWPHIL_BASE = "https://lawphil.net/administ/comelec"

RAW_DIR = Path(__file__).parents[4] / "data" / "raw" / "comelec"

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}

# COMELEC pages to scan for candidate-list PDF links.
# These pages partially render server-side; PDF links may or may not appear
# depending on the server response.
CANDIDATE_PAGES = [
    f"{COMELEC_BASE}/?r=2025NLE/2025BallotFace",
    f"{COMELEC_BASE}/?r=2025NLE/CandidateListCLC",
    f"{COMELEC_BASE}/?r=2025NLE",
]

# Lawphil resolution index pages (most recent first).
# Pattern: /administ/comelec/comres{YEAR}/comres{YEAR}.html
RESOLUTION_INDEX_URLS = {
    2025: f"{LAWPHIL_BASE}/comres2025/comres2025.html",
    2024: f"{LAWPHIL_BASE}/comres2024/comres2024.html",
    2023: f"{LAWPHIL_BASE}/comres2023/comres2023.html",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            time.sleep(1.5)
            return resp
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(3)
    logger.error(f"All retries failed for {url}")
    return None


def _find_pdf_links(html: str, base_url: str) -> list[str]:
    """Return deduplicated absolute URLs of all PDF links in html."""
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    urls: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if ".pdf" in href.lower():
            full = urljoin(base_url, href)
            if full not in seen:
                seen.add(full)
                urls.append(full)
    return urls


def _pdf_title_from_url(url: str) -> str:
    """Derive a human-readable title from a PDF filename."""
    filename = urlparse(url).path.split("/")[-1]
    # e.g. comres_11102_2025.pdf → COMELEC Resolution 11102 (2025)
    m = re.match(r"comres_(\d+)_(\d{4})\.pdf", filename, re.IGNORECASE)
    if m:
        return f"COMELEC Resolution No. {m.group(1)} ({m.group(2)})"
    m = re.match(r"minres_(\d+)_(\d{4})\.pdf", filename, re.IGNORECASE)
    if m:
        return f"COMELEC Minute Resolution No. {m.group(1)} ({m.group(2)})"
    # Generic fallback
    return filename.replace("_", " ").replace(".pdf", "").replace(".PDF", "").title()


# ---------------------------------------------------------------------------
# Candidate list
# ---------------------------------------------------------------------------

def scrape_candidate_pdfs(election_year: int = 2025) -> list[dict]:
    """
    Scan COMELEC's ballot-face / candidate-list pages for PDF links, then
    download and parse any PDFs found.

    Returns chunk dicts ready for embedding.  Returns an empty list (with a
    warning) if no PDFs are found — this happens when the page requires
    JavaScript to render its content.
    """
    all_docs: list[dict] = []
    dest_dir = RAW_DIR / "candidates"

    pdf_urls_found: list[str] = []

    for page_url in CANDIDATE_PAGES:
        logger.info(f"Scanning for candidate PDFs: {page_url}")
        resp = _get(page_url)
        if resp is None:
            continue

        found = _find_pdf_links(resp.text, page_url)
        if not found:
            logger.debug(
                f"No PDF links on {page_url} "
                "(page likely requires JavaScript — consider adding Playwright)"
            )
            continue

        logger.info(f"Found {len(found)} PDF link(s) on {page_url}")
        pdf_urls_found.extend(found)

    if not pdf_urls_found:
        logger.warning(
            "No candidate-list PDFs found on any COMELEC page. "
            "The candidate list pages appear to require JavaScript. "
            "To scrape them fully, add Playwright to the project dependencies."
        )
        return []

    for pdf_url in pdf_urls_found:
        filename = urlparse(pdf_url).path.split("/")[-1]
        metadata = {
            "source": "comelec.gov.ph",
            "source_type": "election",
            "date": f"{election_year}-01-01",
            "politician": "",
            "title": f"COMELEC {election_year} NLE Candidate List: {filename}",
            "url": pdf_url,
        }
        docs = download_and_process_pdf(pdf_url, dest_dir, metadata)
        all_docs.extend(docs)
        logger.info(f"Processed {len(docs)} chunks from {filename}")

    logger.info(f"Total candidate chunks: {len(all_docs)}")
    return all_docs


# ---------------------------------------------------------------------------
# COMELEC resolutions (via lawphil.net)
# ---------------------------------------------------------------------------

def _parse_resolution_date(text: str, year: int) -> str:
    """
    Try to parse a date string from resolution metadata.
    Falls back to January 1 of the given year.
    """
    formats = [
        "%B %d, %Y",   # January 20, 2025
        "%d %B %Y",    # 20 January 2025
        "%Y-%m-%d",
    ]
    text = text.strip()
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return f"{year}-01-01"


def scrape_resolutions(year: int = 2025, max_resolutions: int = 50) -> list[dict]:
    """
    Scrape COMELEC resolutions from lawphil.net's resolution index.

    The index page lists each resolution as a table row with:
      - Date (left column)
      - Resolution number + description link (right column, href → PDF)

    PDFs are downloaded and chunked via pdf_processor.
    """
    index_url = RESOLUTION_INDEX_URLS.get(year)
    if not index_url:
        logger.warning(f"No lawphil resolution index URL configured for year {year}")
        return []

    logger.info(f"Fetching COMELEC resolution index for {year}: {index_url}")
    resp = _get(index_url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    dest_dir = RAW_DIR / "resolutions" / str(year)
    all_docs: list[dict] = []

    # Lawphil renders a simple <table> with rows: date | resolution-link
    rows = soup.select("table tr")
    logger.info(f"Found {len(rows)} table rows on resolution index")

    processed = 0
    for row in rows:
        if processed >= max_resolutions:
            break

        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_text = cells[0].get_text(strip=True)
        link_cell = cells[1]
        link_tag = link_cell.find("a", href=True)
        if not link_tag:
            continue

        href = link_tag["href"].strip()
        resolution_text = link_cell.get_text(separator=" ", strip=True)

        # Build absolute URL (lawphil links may be relative)
        full_url = urljoin(index_url, href)

        resolution_date = _parse_resolution_date(date_text, year)
        title = _pdf_title_from_url(full_url) if href.lower().endswith(".pdf") else resolution_text

        if href.lower().endswith(".pdf"):
            metadata = {
                "source": "comelec.gov.ph",
                "source_type": "resolution",
                "date": resolution_date,
                "politician": "",
                "title": title,
                "url": full_url,
            }
            docs = download_and_process_pdf(full_url, dest_dir, metadata)
            if docs:
                all_docs.extend(docs)
                logger.info(f"Processed {len(docs)} chunks: {title}")
                processed += 1
        else:
            # HTML resolution page — scrape its text content
            detail_resp = _get(full_url)
            if not detail_resp:
                continue
            detail_soup = BeautifulSoup(detail_resp.text, "lxml")
            content = (
                detail_soup.find("div", class_="resolution-content")
                or detail_soup.find("div", class_="content")
                or detail_soup.find("main")
                or detail_soup.find("article")
                or detail_soup.find("body")
            )
            text = content.get_text(separator="\n", strip=True) if content else ""
            if text:
                all_docs.append({
                    "source": "comelec.gov.ph",
                    "source_type": "resolution",
                    "date": resolution_date,
                    "politician": "",
                    "title": title,
                    "url": full_url,
                    "text": text,
                })
                processed += 1

            # Also extract any nested PDF links (annexes)
            nested_pdfs = _find_pdf_links(detail_resp.text, full_url)
            for pdf_url in nested_pdfs:
                annex_title = _pdf_title_from_url(pdf_url)
                metadata = {
                    "source": "comelec.gov.ph",
                    "source_type": "resolution",
                    "date": resolution_date,
                    "politician": "",
                    "title": f"{title} — {annex_title}",
                    "url": pdf_url,
                }
                docs = download_and_process_pdf(pdf_url, dest_dir, metadata)
                all_docs.extend(docs)

    logger.info(f"Total resolution chunks for {year}: {len(all_docs)}")
    return all_docs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scrape_all_comelec(election_year: int = 2025, max_resolutions: int = 30) -> list[dict]:
    """
    Scrape all available COMELEC data:
      1. Candidate list PDFs from comelec.gov.ph (if JS-accessible)
      2. COMELEC resolutions from lawphil.net
    """
    all_docs: list[dict] = []

    logger.info("=== COMELEC: candidate lists ===")
    all_docs.extend(scrape_candidate_pdfs(election_year))

    logger.info("=== COMELEC: resolutions ===")
    all_docs.extend(scrape_resolutions(election_year, max_resolutions=max_resolutions))

    logger.info(f"Total COMELEC docs: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_comelec(max_resolutions=5)
    print(f"Total: {len(docs)} chunks")
    for d in docs[:3]:
        print(f"  [{d['source_type']}] {d['title'][:80]}")
