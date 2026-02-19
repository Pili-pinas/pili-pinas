"""
Senate of the Philippines scraper.
Scrapes bills, resolutions, and senator profiles from senate.gov.ph.

Respects robots.txt — senate.gov.ph allows crawling of public pages.
Rate limit: 1–2 seconds between requests.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.senate.gov.ph"
BILLS_URL = f"{BASE_URL}/lis/bill_res.aspx"
SENATORS_URL = f"{BASE_URL}/senators"

# Output directory (raw HTML saved before processing)
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
            time.sleep(1.5)  # rate limit
            return resp
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(3)
    logger.error(f"All retries failed for {url}")
    return None


def scrape_bills(congress: int = 19, max_pages: int = 5) -> list[dict]:
    """
    Scrape Senate bills and resolutions for a given Congress.

    Returns a list of document dicts matching the metadata schema.
    """
    documents = []
    page = 1

    while page <= max_pages:
        params = {
            "congress": congress,
            "type": "SB",  # Senate Bill
            "pg": page,
        }
        resp = _get(BILLS_URL, params=params)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table.bills-table tr")[1:]  # skip header

        if not rows:
            logger.info(f"No more rows at page {page}, stopping.")
            break

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            bill_number = cells[0].get_text(strip=True)
            title = cells[1].get_text(strip=True)
            author = cells[2].get_text(strip=True)
            status = cells[3].get_text(strip=True)

            # Try to get detail page link
            link_tag = cells[1].find("a")
            detail_url = BASE_URL + link_tag["href"] if link_tag else ""

            doc = {
                "source": "senate.gov.ph",
                "source_type": "bill",
                "date": datetime.now().strftime("%Y-%m-%d"),  # updated when detail scraped
                "politician": author,
                "title": f"Senate Bill {bill_number}: {title}",
                "url": detail_url,
                "bill_number": bill_number,
                "status": status,
                "congress": congress,
                "text": "",  # filled by _scrape_bill_detail
            }

            if detail_url:
                doc = _scrape_bill_detail(doc)

            documents.append(doc)
            logger.info(f"Scraped: {bill_number} — {title[:60]}")

        page += 1

    logger.info(f"Total bills scraped: {len(documents)}")
    return documents


def _scrape_bill_detail(doc: dict) -> dict:
    """Fetch the bill detail page and extract full text + date."""
    resp = _get(doc["url"])
    if resp is None:
        return doc

    # Save raw HTML
    safe_name = doc["bill_number"].replace(" ", "_").replace("/", "-")
    raw_path = RAW_DIR / "laws" / f"senate_{safe_name}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(resp.text, encoding="utf-8")

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract date filed
    date_tag = soup.find("td", string=lambda t: t and "Date Filed" in t)
    if date_tag and date_tag.find_next_sibling("td"):
        raw_date = date_tag.find_next_sibling("td").get_text(strip=True)
        try:
            doc["date"] = datetime.strptime(raw_date, "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass  # keep today's date as fallback

    # Extract bill text / abstract
    content_div = soup.find("div", class_="bill-content") or soup.find("div", id="billContent")
    if content_div:
        doc["text"] = content_div.get_text(separator="\n", strip=True)
    else:
        # Fallback: grab all paragraph text from main content area
        main = soup.find("div", class_="container") or soup.find("main")
        if main:
            doc["text"] = main.get_text(separator="\n", strip=True)

    return doc


def scrape_senators() -> list[dict]:
    """
    Scrape current senator profiles from senate.gov.ph/senators.

    Returns a list of profile document dicts.
    """
    resp = _get(SENATORS_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    documents = []

    # Senator cards are typically in anchor tags with senator photos + names
    senator_cards = soup.select("div.senator-card, div.senator-item, td.senator-cell")

    for card in senator_cards:
        name_tag = card.find("h3") or card.find("strong") or card.find("a")
        name = name_tag.get_text(strip=True) if name_tag else "Unknown"

        link_tag = card.find("a")
        profile_url = BASE_URL + link_tag["href"] if link_tag and link_tag.get("href") else ""

        doc = {
            "source": "senate.gov.ph",
            "source_type": "profile",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "politician": name,
            "title": f"Senator Profile: {name}",
            "url": profile_url,
            "text": "",
        }

        if profile_url:
            doc = _scrape_senator_detail(doc)

        documents.append(doc)
        logger.info(f"Scraped senator: {name}")

    logger.info(f"Total senators scraped: {len(documents)}")
    return documents


def _scrape_senator_detail(doc: dict) -> dict:
    """Fetch individual senator profile page."""
    resp = _get(doc["url"])
    if resp is None:
        return doc

    # Save raw HTML
    safe_name = doc["politician"].replace(" ", "_").replace(".", "")
    raw_path = RAW_DIR / "politician_profiles" / f"senator_{safe_name}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(resp.text, encoding="utf-8")

    soup = BeautifulSoup(resp.text, "lxml")

    bio_div = (
        soup.find("div", class_="senator-bio")
        or soup.find("div", class_="profile-content")
        or soup.find("div", id="senatorProfile")
    )
    if bio_div:
        doc["text"] = bio_div.get_text(separator="\n", strip=True)

    return doc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke test — scrape first page of bills only
    bills = scrape_bills(congress=19, max_pages=1)
    print(f"Scraped {len(bills)} bills")
    if bills:
        print(bills[0]["title"])
