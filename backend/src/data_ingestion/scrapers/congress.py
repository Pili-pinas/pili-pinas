"""
House of Representatives scraper.
Scrapes House bills and representative profiles from congress.gov.ph.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.congress.gov.ph"
BILLS_URL = f"{BASE_URL}/legisdocs"
MEMBERS_URL = f"{BASE_URL}/members"

RAW_DIR = Path(__file__).parents[4] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}


def _get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(1.5)
            return resp
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(3)
    return None


def scrape_house_bills(congress: int = 19, max_pages: int = 5) -> list[dict]:
    """Scrape House Bills from congress.gov.ph."""
    documents = []

    for page in range(1, max_pages + 1):
        params = {"congress": congress, "type": "HB", "page": page}
        resp = _get(BILLS_URL, params=params)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table tr")[1:]

        if not rows:
            break

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            bill_number = cells[0].get_text(strip=True)
            title = cells[1].get_text(strip=True)
            author = cells[2].get_text(strip=True)

            link_tag = cells[1].find("a")
            detail_url = BASE_URL + link_tag["href"] if link_tag else ""

            doc = {
                "source": "congress.gov.ph",
                "source_type": "bill",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "politician": author,
                "title": f"House Bill {bill_number}: {title}",
                "url": detail_url,
                "bill_number": bill_number,
                "congress": congress,
                "text": "",
            }

            documents.append(doc)
            logger.info(f"Scraped HB: {bill_number}")

    logger.info(f"Total house bills scraped: {len(documents)}")
    return documents


def scrape_members() -> list[dict]:
    """Scrape House member profiles."""
    resp = _get(MEMBERS_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    documents = []

    member_rows = soup.select("table.members-table tr")[1:]

    for row in member_rows:
        cells = row.find_all("td")
        if not cells:
            continue

        name_tag = cells[0].find("a") or cells[0]
        name = name_tag.get_text(strip=True)
        link_tag = cells[0].find("a")
        profile_url = BASE_URL + link_tag["href"] if link_tag else ""

        district = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        party = cells[2].get_text(strip=True) if len(cells) > 2 else ""

        doc = {
            "source": "congress.gov.ph",
            "source_type": "profile",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "politician": name,
            "title": f"Representative Profile: {name}",
            "url": profile_url,
            "district": district,
            "party": party,
            "text": f"{name}, Representative for {district}, Party: {party}",
        }
        documents.append(doc)

    logger.info(f"Total members scraped: {len(documents)}")
    return documents
