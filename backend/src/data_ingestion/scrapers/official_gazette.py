"""
Philippine Republic Acts scraper.

Data source: Supreme Court E-Library (elibrary.judiciary.gov.ph/republic_acts)
  - officialgazette.gov.ph is Cloudflare-blocked; this is the authoritative
    alternative with 12,000+ Republic Acts, fully accessible.

How it works:
  1. GET the index page to obtain the session cookie and CSRF token.
  2. POST to /republic_acts/fetch_ra (DataTable server-side API) with
     pagination params (start, length).
  3. Each row contains short title, date, and a link with the long title.
  4. Optionally fetch each law's detail page to get the full enacted text
     from div.single_content.

Rate limit: 1–2 seconds between requests.
"""

import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ELIBRARY_BASE = "https://elibrary.judiciary.gov.ph"
ELIBRARY_RA_INDEX = f"{ELIBRARY_BASE}/republic_acts"
ELIBRARY_RA_FETCH = f"{ELIBRARY_BASE}/republic_acts/fetch_ra"

RAW_DIR = Path(__file__).parents[4] / "data" / "raw" / "laws"

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fetch_index(session: requests.Session) -> Optional[str]:
    """
    GET the Republic Acts index page and return the CSRF token.

    Returns the token string, or None if the request fails or no token found.
    """
    try:
        resp = session.get(ELIBRARY_RA_INDEX, timeout=15)
        resp.raise_for_status()
        time.sleep(1.5)
    except Exception as e:
        logger.warning(f"Failed to fetch elibrary index: {e}")
        return None

    match = re.search(r"csrf_test_name.*?'([a-f0-9]+)'", resp.text)
    if not match:
        logger.warning("CSRF token not found in elibrary index page.")
        return None

    return match.group(1)


def _fetch_ra_page(
    session: requests.Session,
    csrf: str,
    start: int,
    length: int,
) -> Optional[dict]:
    """
    POST to the DataTable fetch endpoint and return the JSON payload.

    Returns the parsed JSON dict, or None on failure.
    """
    data = {
        "csrf_test_name": csrf,
        "draw": str(start // length + 1),
        "start": str(start),
        "length": str(length),
        "search[value]": "",
        "search[regex]": "false",
    }
    try:
        resp = session.post(ELIBRARY_RA_FETCH, data=data, timeout=15)
        resp.raise_for_status()
        time.sleep(1.5)
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch RA page (start={start}): {e}")
        return None


def _fetch_law_text(session: requests.Session, url: str) -> str:
    """
    Fetch a law's detail page and return the full text from div.single_content.

    Returns empty string on failure or if the content div is not found.
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        time.sleep(1.5)
    except Exception as e:
        logger.warning(f"Failed to fetch law detail {url}: {e}")
        return ""

    # Save raw HTML
    safe_name = url.rstrip("/").split("/")[-1]
    raw_path = RAW_DIR / f"ra_{safe_name}.html"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        raw_path.write_bytes(resp.content)
    except Exception:
        pass  # non-critical

    soup = BeautifulSoup(resp.content, "lxml")
    content = soup.find("div", class_="single_content")
    if not content:
        return ""

    return content.get_text(separator="\n", strip=True)


def _parse_law_row(row: list) -> dict:
    """
    Convert a DataTable row to a metadata document dict.

    Row columns:
      [0] short title   — e.g. "REPUBLIC ACT NO. 12312"
      [1] date          — "YYYY-MM-DD"
      [2] link HTML     — <a href="...">long title</a>
    """
    short_title = row[0] if row else ""
    date = row[1] if len(row) > 1 else datetime.now().strftime("%Y-%m-%d")
    link_html = row[2] if len(row) > 2 else ""

    # Parse link for URL and long title
    link_soup = BeautifulSoup(link_html, "lxml")
    a_tag = link_soup.find("a")
    url = a_tag["href"] if a_tag else ""
    long_title = a_tag.get_text(strip=True) if a_tag else link_html

    return {
        "source": "elibrary.judiciary.gov.ph",
        "source_type": "law",
        "date": date,
        "politician": "",
        "title": short_title,
        "url": url,
        "text": long_title,  # overwritten with full text if detail page succeeds
    }


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape_laws(max_items: int = 50, from_year: Optional[int] = None) -> list[dict]:
    """
    Scrape Republic Acts from the Supreme Court E-Library.

    The API returns laws newest-first, so from_year stops pagination as soon
    as a law older than the cutoff is encountered.

    Args:
        max_items: Maximum number of laws to return.
        from_year: Stop scraping when a law's year is older than this value.
                   None (default) means no cutoff — scrape up to max_items.

    Returns:
        List of document dicts matching the metadata schema.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    csrf = _fetch_index(session)
    if csrf is None:
        return []

    documents = []
    page_size = min(50, max_items)
    start = 0
    cutoff_reached = False

    while len(documents) < max_items and not cutoff_reached:
        payload = _fetch_ra_page(session, csrf, start=start, length=page_size)
        if payload is None:
            break

        rows = payload.get("data", [])
        total = payload.get("recordsTotal", 0)

        for row in rows:
            doc = _parse_law_row(row)

            if from_year is not None:
                try:
                    law_year = int(doc["date"][:4])
                except (ValueError, TypeError):
                    law_year = None
                if law_year is not None and law_year < from_year:
                    cutoff_reached = True
                    break

            # Fetch full text from detail page if URL is available
            if doc["url"]:
                full_text = _fetch_law_text(session, doc["url"])
                if full_text:
                    doc["text"] = full_text

            documents.append(doc)
            logger.info(f"Scraped law: {doc['title'][:70]}")

            if len(documents) >= max_items:
                break

        start += len(rows)
        if start >= total or len(documents) >= max_items or not rows:
            break

    logger.info(f"Total laws scraped: {len(documents)}")
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    laws = scrape_laws(max_items=5)
    print(f"Scraped {len(laws)} laws")
    if laws:
        print(laws[0]["title"])
        print(laws[0]["text"][:300])
