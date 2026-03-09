"""
House of Representatives of the Philippines scraper.

Data sources (congress.gov.ph is Cloudflare-blocked):
  - Bills   : BetterGov Open Congress API (open-congress-api.bettergov.ph)
              Note: HB title data is sparse; bills without titles are skipped.
  - Members : Wikipedia current members list + individual member pages

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
WIKIPEDIA_MEMBERS_URL = (
    "https://en.wikipedia.org/wiki/List_of_current_members_of_the_House_of_Representatives_of_the_Philippines"
)
WIKIPEDIA_BASE = "https://en.wikipedia.org"

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

def scrape_house_bills(congress: int = 19, max_items: int = 100) -> list[dict]:
    """
    Scrape House Bills for a given Congress via the BetterGov API.

    Note: HB records frequently have null titles. Bills without a title are
    skipped since they carry no useful text for the RAG index.

    Args:
        congress: Congress number (default 19 = most complete HB dataset).
        max_items: Maximum number of bills to return.

    Returns:
        List of document dicts matching the metadata schema.
    """
    documents = []
    cursor: Optional[str] = None
    page_size = min(50, max_items)
    pages_fetched = 0
    max_pages = max(max_items, 10)  # cap pages fetched to avoid infinite scan of null-title records

    while len(documents) < max_items and pages_fetched < max_pages:
        params: dict = {"type": "HB", "congress": congress, "limit": page_size}
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

        pages_fetched += 1

        for bill in payload.get("data", []):
            if not bill.get("title"):
                continue  # skip bills with no title (common for HB data)
            doc = _hb_to_doc(bill, congress)
            documents.append(doc)
            logger.info(f"Scraped HB: {doc['title'][:70]}")

            if len(documents) >= max_items:
                break

        pagination = payload.get("pagination", {})
        if not pagination.get("has_more") or len(documents) >= max_items:
            break
        cursor = pagination.get("next_cursor")

    logger.info(f"Total house bills scraped: {len(documents)}")
    return documents


def _hb_to_doc(bill: dict, congress: int) -> dict:
    """Convert a BetterGov API HB record to our metadata schema."""
    name = bill.get("name", "")       # e.g. "HBN-06571"
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
        "url": bill.get("congress_website_permalink") or "",
        "text": "\n".join(filter(None, text_parts)),
        "bill_number": bill.get("bill_number", ""),
        "congress": congress,
    }


# ---------------------------------------------------------------------------
# Members — Wikipedia
# ---------------------------------------------------------------------------

def scrape_members() -> list[dict]:
    """
    Scrape current House member profiles from Wikipedia.

    Fetches the list page for basic metadata, then each member's Wikipedia
    article for a biographical summary.

    Returns:
        List of profile document dicts matching the metadata schema.
    """
    resp = _get(WIKIPEDIA_MEMBERS_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.content, "lxml")
    member_rows = _parse_members_table(soup)

    if not member_rows:
        logger.warning("No members found in Wikipedia table.")
        return []

    documents = []
    for row in member_rows:
        name = row["name"]
        wiki_path = row["wiki_path"]

        bio_text = row["bio_text"]  # basic text from table
        wiki_url = WIKIPEDIA_BASE + wiki_path if wiki_path else ""

        # Fetch individual Wikipedia article for rich bio content
        if wiki_path:
            detail_text = _scrape_member_wiki_page(name, wiki_path)
            if detail_text:
                bio_text = detail_text

        doc = {
            "source": "wikipedia.org",
            "source_type": "profile",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "politician": name,
            "title": f"Representative Profile: {name}",
            "url": wiki_url,
            "text": bio_text,
        }
        documents.append(doc)
        logger.info(f"Scraped member: {name}")

    logger.info(f"Total house members scraped: {len(documents)}")
    return documents


def _parse_members_table(soup: BeautifulSoup) -> list[dict]:
    """
    Extract member rows from the Wikipedia wikitable.

    Column order (9 cells per row):
      0: Constituency (with link)
      1: Portrait (image)
      2: Representative name (with link)
      3: Party flag (image, text is empty)
      4: Party name
      5: Bloc (Majority/Minority)
      6: Born
      7: Prior experience
      8: Took office
    """
    results = []
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Representative" not in headers:
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            name = cells[2].get_text(strip=True)
            if not name:
                continue

            constituency = cells[0].get_text(strip=True)
            party = cells[4].get_text(strip=True)
            bloc = cells[5].get_text(strip=True) if len(cells) > 5 else ""

            # Clean born date — strip sortkey prefix like "(1978-10-06)"
            born_raw = cells[6].get_text(strip=True) if len(cells) > 6 else ""
            born = re.sub(r"^\(\d{4}-\d{2}-\d{2}\)", "", born_raw).strip()

            prior_exp = cells[7].get_text(separator=", ", strip=True) if len(cells) > 7 else ""
            took_office = cells[8].get_text(strip=True) if len(cells) > 8 else ""

            link_tag = cells[2].find("a")
            wiki_path = link_tag["href"] if link_tag else ""

            bio_text = (
                f"Representative {name}, {constituency}. Party: {party}. Bloc: {bloc}. "
                f"Prior experience: {prior_exp}. Born: {born}. "
                f"Took office: {took_office}."
            )

            results.append({
                "name": name,
                "constituency": constituency,
                "wiki_path": wiki_path,
                "bio_text": bio_text,
            })

        break  # only need the first matching table

    return results


def _scrape_member_wiki_page(name: str, wiki_path: str) -> str:
    """Fetch a member's Wikipedia article and return the introductory text."""
    url = WIKIPEDIA_BASE + wiki_path
    resp = _get(url)
    if resp is None:
        return ""

    # Save raw HTML
    safe_name = name.replace(" ", "_").replace(".", "")
    raw_path = RAW_DIR / "politician_profiles" / f"rep_{safe_name}.html"
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
    bills = scrape_house_bills(congress=19, max_items=5)
    print(f"\nScraped {len(bills)} house bills")
    if bills:
        print(bills[0]["title"])

    members = scrape_members()
    print(f"\nScraped {len(members)} house members")
    if members:
        print(members[0]["title"])
        print(members[0]["text"][:200])
