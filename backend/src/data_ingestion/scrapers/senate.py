"""
Senate of the Philippines scraper.

Data sources (senate.gov.ph is Cloudflare-blocked):
  - Bills  : BetterGov Open Congress API (open-congress-api.bettergov.ph)
  - Senators: Wikipedia current senators list + individual senator pages

Rate limit: 1–2 seconds between requests.
"""

import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BETTERGOV_URL = "https://open-congress-api.bettergov.ph/api/documents"
WIKIPEDIA_SENATORS_URL = (
    "https://en.wikipedia.org/wiki/List_of_current_senators_of_the_Philippines"
)
WIKIPEDIA_BASE = "https://en.wikipedia.org"

# Output directory for raw HTML (saved before processing)
RAW_DIR = Path(__file__).parents[4] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}


def _get(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[requests.Response]:
    """GET with retries and rate limiting."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(1.5)
            return resp
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(3)
    logger.error(f"All retries failed for {url}")
    return None


# ---------------------------------------------------------------------------
# Bills — BetterGov Open Congress API
# ---------------------------------------------------------------------------

def scrape_bills(congress: int = 20, max_items: int = 100) -> list[dict]:
    """
    Scrape Senate Bills for a given Congress via the BetterGov API.

    Args:
        congress: Congress number (default 20 = current as of 2025).
        max_items: Maximum number of bills to return.

    Returns:
        List of document dicts matching the metadata schema.
    """
    documents = []
    cursor: Optional[str] = None
    page_size = min(50, max_items)

    while len(documents) < max_items:
        params: dict = {"type": "SB", "congress": congress, "limit": page_size}
        if cursor:
            params["cursor"] = cursor

        resp = _get(BETTERGOV_URL, params=params)
        if resp is None:
            break

        try:
            payload = resp.json()
        except Exception as e:
            logger.error(f"Failed to parse BetterGov response: {e}")
            break

        if not payload.get("success"):
            logger.error(f"BetterGov API error: {payload}")
            break

        for bill in payload.get("data", []):
            doc = _bill_to_doc(bill, congress)
            documents.append(doc)
            logger.info(f"Scraped bill: {doc['title'][:70]}")

            if len(documents) >= max_items:
                break

        pagination = payload.get("pagination", {})
        if not pagination.get("has_more") or len(documents) >= max_items:
            break
        cursor = pagination.get("next_cursor")

    logger.info(f"Total senate bills scraped: {len(documents)}")
    return documents


def _bill_to_doc(bill: dict, congress: int) -> dict:
    """Convert a BetterGov API bill record to our metadata schema."""
    name = bill.get("name", "")       # e.g. "SBN-01321"
    title = bill.get("title") or ""
    long_title = bill.get("long_title") or ""
    subjects = bill.get("subjects") or []
    authors_raw = bill.get("authors_raw") or ""

    text_parts = [title, long_title]
    if subjects:
        text_parts.append("Subjects: " + ", ".join(subjects))

    date_filed = bill.get("date_filed") or datetime.now().strftime("%Y-%m-%d")

    return {
        "source": "bettergov.ph",
        "source_type": "bill",
        "date": date_filed,
        "politician": authors_raw,
        "title": f"{name}: {title}" if name else title,
        "url": bill.get("senate_website_permalink") or "",
        "text": "\n".join(filter(None, text_parts)),
        "bill_number": bill.get("bill_number", ""),
        "congress": congress,
    }


# ---------------------------------------------------------------------------
# Senators — Wikipedia
# ---------------------------------------------------------------------------

def scrape_senators() -> list[dict]:
    """
    Scrape current senator profiles from Wikipedia.

    Fetches the list page for basic metadata, then each senator's Wikipedia
    article for a biographical summary.

    Returns:
        List of profile document dicts matching the metadata schema.
    """
    resp = _get(WIKIPEDIA_SENATORS_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.content, "lxml")
    senator_rows = _parse_senators_table(soup)

    if not senator_rows:
        logger.warning("No senators found in Wikipedia table.")
        return []

    documents = []
    for row in senator_rows:
        name = row["name"]
        wiki_path = row["wiki_path"]

        bio_text = row["bio_text"]  # basic text from table
        wiki_url = WIKIPEDIA_BASE + wiki_path if wiki_path else ""

        # Fetch individual Wikipedia article for rich bio content
        if wiki_path:
            detail_text = _scrape_senator_wiki_page(name, wiki_path)
            if detail_text:
                bio_text = detail_text

        doc = {
            "source": "wikipedia.org",
            "source_type": "profile",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "politician": name,
            "title": f"Senator Profile: {name}",
            "url": wiki_url,
            "text": bio_text,
        }
        documents.append(doc)
        logger.info(f"Scraped senator: {name}")

    logger.info(f"Total senators scraped: {len(documents)}")
    return documents


def _parse_senators_table(soup: BeautifulSoup) -> list[dict]:
    """
    Extract senator rows from the Wikipedia wikitable.

    Column order (12 cells per row):
      0: Portrait (image)
      1: Senator name (with link)
      2: Party flag (image, text is empty)
      3: Party name
      4: Bloc (Majority/Minority)
      5: Born
      6: Occupation(s)
      7: Previous elective office(s)
      8: Education
      9: Took office
      10: Term ending
      11: Term
    """
    results = []
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Senator" not in headers:
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            name = cells[1].get_text(strip=True)
            if not name:
                continue

            party = cells[3].get_text(strip=True)
            bloc = cells[4].get_text(strip=True)

            # Clean born date — strip sortkey prefix like "(1977-05-07)"
            born_raw = cells[5].get_text(strip=True) if len(cells) > 5 else ""
            born = re.sub(r"^\(\d{4}-\d{2}-\d{2}\)", "", born_raw).strip()

            occupation = cells[6].get_text(separator=", ", strip=True) if len(cells) > 6 else ""
            took_office = cells[9].get_text(strip=True) if len(cells) > 9 else ""
            term_ending = cells[10].get_text(strip=True) if len(cells) > 10 else ""

            link_tag = cells[1].find("a")
            wiki_path = link_tag["href"] if link_tag else ""

            bio_text = (
                f"Senator {name}. Party: {party}. Bloc: {bloc}. "
                f"Occupation: {occupation}. Born: {born}. "
                f"Took office: {took_office}. Term ending: {term_ending}."
            )

            results.append({
                "name": name,
                "wiki_path": wiki_path,
                "bio_text": bio_text,
            })

        break  # only need the first matching table

    return results


def _scrape_senator_wiki_page(name: str, wiki_path: str) -> str:
    """Fetch a senator's Wikipedia article and return the introductory text."""
    url = WIKIPEDIA_BASE + wiki_path
    resp = _get(url)
    if resp is None:
        return ""

    # Save raw HTML
    safe_name = name.replace(" ", "_").replace(".", "")
    raw_path = RAW_DIR / "politician_profiles" / f"senator_{safe_name}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(resp.content)

    soup = BeautifulSoup(resp.content, "lxml")
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return ""

    # Grab all top-level paragraphs up to the first h2 section heading
    parts = []
    for tag in content.children:
        if tag.name == "h2":
            break
        if tag.name == "p":
            text = tag.get_text(strip=True)
            if text:
                # Strip inline footnote markers like [1][2]
                text = re.sub(r"\[\d+\]", "", text)
                parts.append(text)

    return "\n\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bills = scrape_bills(congress=20, max_items=5)
    print(f"\nScraped {len(bills)} bills")
    if bills:
        print(bills[0]["title"])

    senators = scrape_senators()
    print(f"\nScraped {len(senators)} senators")
    if senators:
        print(senators[0]["title"])
        print(senators[0]["text"][:200])
