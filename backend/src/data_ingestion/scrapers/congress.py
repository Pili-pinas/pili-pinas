"""
House of Representatives of the Philippines scraper.

Data sources (congress.gov.ph is Cloudflare-blocked):
  - Bills   : BetterGov Open Congress API (open-congress-api.bettergov.ph)
              Note: HB title data is sparse; bills without titles are skipped.
  - Members : BetterGov Open Congress API /people endpoint

Rate limit: 1–2 seconds between requests.
"""

import time
import logging
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BETTERGOV_URL = "https://open-congress-api.bettergov.ph/api/documents"
BETTERGOV_PEOPLE_URL = "https://open-congress-api.bettergov.ph/api/people"

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
    if authors_raw:
        text_parts.append(f"Authors: {authors_raw}")
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
# Members — BetterGov Open Congress API
# ---------------------------------------------------------------------------

def scrape_members(congresses: Optional[list] = None) -> list[dict]:
    """
    Scrape House member profiles from the BetterGov Open Congress API.

    Fetches all politicians via pagination, filters to those who served
    as representatives in the given congress numbers.

    Args:
        congresses: Congress numbers to include (default: 17-20).

    Returns:
        List of profile document dicts matching the metadata schema.
    """
    target = set(congresses or [17, 18, 19, 20])
    people = _fetch_all_people()
    documents = []

    for person in people:
        served = [
            c for c in person.get("congresses_served", [])
            if c.get("position", "").lower() == "representative"
            and c.get("congress_number") in target
        ]
        if not served:
            continue
        doc = _build_profile_doc(person, served)
        documents.append(doc)
        logger.info(f"Scraped member: {doc['politician']}")

    logger.info(f"Total house members scraped: {len(documents)}")
    return documents


def _fetch_all_people() -> list[dict]:
    """Paginate through all BetterGov people records."""
    people = []
    cursor = None
    page = 0
    while True:
        params: dict = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        resp = _get(BETTERGOV_PEOPLE_URL, params=params)
        if resp is None:
            break
        payload = resp.json()
        if not payload.get("success", True):
            logger.error(f"BetterGov people API error: {payload}")
            break
        batch = payload.get("data", [])
        if not batch:
            break
        people.extend(batch)
        page += 1
        logger.info(f"Fetched people page {page} ({len(people)} total so far)")
        pagination = payload.get("pagination", {})
        if not pagination.get("has_more"):
            break
        cursor = pagination.get("next_cursor")
    logger.info(f"Finished fetching people: {len(people)} records total")
    return people


def _build_profile_doc(person: dict, served: list[dict]) -> dict:
    """Build a representative profile document from a BetterGov person record."""
    name_parts = [
        person.get("name_prefix") or "",
        person.get("first_name") or "",
        person.get("middle_name") or "",
        person.get("last_name") or "",
        person.get("name_suffix") or "",
    ]
    full_name = " ".join(p for p in name_parts if p)

    congresses_str = ", ".join(c["congress_ordinal"] for c in served)
    text_parts = [
        f"Representative: {full_name}.",
        f"Served in the Philippine House of Representatives during the {congresses_str} Congress.",
    ]
    aliases = [a for a in (person.get("aliases") or []) if a]
    if aliases:
        text_parts.append(f"Also known as: {', '.join(aliases)}.")

    return {
        "source": "open-congress-api.bettergov.ph",
        "source_type": "profile",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "politician": full_name,
        "title": f"Representative Profile: {full_name}",
        "url": f"https://open-congress-api.bettergov.ph/api/people/{person['id']}",
        "text": " ".join(text_parts),
    }


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
