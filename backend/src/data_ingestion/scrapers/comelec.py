"""
COMELEC scraper.
Scrapes candidate lists and election results from comelec.gov.ph.
Note: COMELEC publishes PDFs heavily; this scraper handles their HTML pages.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://comelec.gov.ph"
CANDIDATES_URL = f"{BASE_URL}/?r=Statistics/VoterRegistration"

RAW_DIR = Path(__file__).parents[4] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}


def _get(url: str) -> Optional[requests.Response]:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(1.5)
            return resp
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(3)
    return None


def scrape_candidate_list(election_year: int = 2025) -> list[dict]:
    """
    Scrape candidate list for a given election year.
    COMELEC publishes these as PDFs and HTML tables.
    """
    # COMELEC election results page (actual URL structure may vary per election)
    url = f"{BASE_URL}/?r=2025NLE/CandidateList"
    resp = _get(url)
    if resp is None:
        logger.warning("Could not reach COMELEC candidates page.")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    documents = []

    rows = soup.select("table.candidates-table tr, table tr")[1:]
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        name = cells[0].get_text(strip=True)
        position = cells[1].get_text(strip=True)
        party = cells[2].get_text(strip=True)

        doc = {
            "source": "comelec.gov.ph",
            "source_type": "election",
            "date": f"{election_year}-01-01",
            "politician": name,
            "title": f"COMELEC Candidate: {name} ({position})",
            "url": url,
            "position": position,
            "party": party,
            "election_year": election_year,
            "text": (
                f"{name} is a candidate for {position} under {party} "
                f"in the {election_year} Philippine National Elections."
            ),
        }
        documents.append(doc)

    logger.info(f"Total candidates scraped: {len(documents)}")
    return documents
