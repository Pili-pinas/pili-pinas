"""
Philippine financial transparency scrapers.
Covers: PhilGEPS (procurement notices), SOCE/COMELEC (campaign finance).
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}

PHILGEPS_URL = (
    "https://notices.philgeps.gov.ph/GEPSNONPILOT/Tender/SplashOpenOpportunitiesUI.aspx"
)
SOCE_URL = "https://comelec.gov.ph/?r=PoliticalFinance/SOCE"


def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(1.5)
            return resp
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(3)
    return None


def _parse_date(text: str) -> str:
    patterns = [
        ("%B %d, %Y", r"\b\w+ \d{1,2}, \d{4}\b"),
        ("%d %B %Y",  r"\b\d{1,2} \w+ \d{4}\b"),
        ("%Y-%m-%d",  r"\d{4}-\d{2}-\d{2}"),
        ("%B %d,%Y",  r"\b\w+ \d{1,2},\d{4}\b"),
    ]
    for fmt, pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return datetime.strptime(m.group(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return datetime.now().strftime("%Y-%m-%d")


def scrape_philgeps(max_items: int = 20) -> list[dict]:
    """
    Scrape PhilGEPS open procurement opportunities.
    Returns documents with source_type='procurement'.
    """
    resp = _get(PHILGEPS_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    documents = []

    # PhilGEPS uses a table with id="bodyContent" or similar; rows have class odd/even
    rows = soup.find_all("tr", class_=re.compile(r"odd|even", re.I))

    for row in rows[:max_items]:
        cols = row.find_all("td")
        if not cols:
            continue
        link = cols[0].find("a", href=True) if cols else None
        if not link:
            continue

        title = link.get_text(strip=True)
        href = link["href"]
        if not href.startswith("http"):
            parsed = urlparse(PHILGEPS_URL)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"

        entity = cols[1].get_text(strip=True) if len(cols) > 1 else ""
        date_text = cols[2].get_text(strip=True) if len(cols) > 2 else ""
        amount = cols[3].get_text(strip=True) if len(cols) > 3 else ""

        text_parts = [title]
        if entity:
            text_parts.append(f"Procuring Entity: {entity}")
        if amount:
            text_parts.append(f"Amount: {amount}")

        documents.append({
            "source": "philgeps.gov.ph",
            "source_type": "procurement",
            "date": _parse_date(date_text) if date_text else datetime.now().strftime("%Y-%m-%d"),
            "politician": "",
            "title": title,
            "url": href,
            "text": " | ".join(text_parts),
        })

    return documents


def scrape_soce(max_items: int = 20) -> list[dict]:
    """
    Scrape COMELEC SOCE (Statement of Contributions and Expenditures) listings.
    Returns documents with source_type='campaign_finance'.
    """
    resp = _get(SOCE_URL)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    documents = []

    # SOCE page lists links to candidate SOCE PDFs
    candidates: list[tuple[str, str, str]] = []

    # Pattern 1: <ul>/<li> with SOCE links
    for li in soup.find_all("li"):
        link = li.find("a", href=True)
        if not link:
            continue
        title = link.get_text(strip=True)
        href = link["href"]
        date_el = li.find(class_=re.compile(r"date|filed|time", re.I))
        date_text = date_el.get_text(strip=True) if date_el else ""
        if title and len(title) > 5:
            candidates.append((title, href, date_text))

    # Pattern 2: any <a> with "SOCE" in title or href
    if not candidates:
        for link in soup.find_all("a", href=True):
            title = link.get_text(strip=True)
            href = link["href"]
            if "soce" in title.lower() or "soce" in href.lower():
                if title and len(title) > 5:
                    candidates.append((title, href, ""))

    for title, href, date_text in candidates[:max_items]:
        if not href.startswith("http"):
            parsed = urlparse(SOCE_URL)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"

        documents.append({
            "source": "comelec.gov.ph",
            "source_type": "campaign_finance",
            "date": _parse_date(date_text) if date_text else datetime.now().strftime("%Y-%m-%d"),
            "politician": "",
            "title": title,
            "url": href,
            "text": title,
        })

    return documents


def scrape_all_financial(max_items: int = 20) -> list[dict]:
    """Aggregate procurement and campaign finance documents."""
    all_docs = []
    for fn, label in [(scrape_philgeps, "PhilGEPS"), (scrape_soce, "SOCE")]:
        logger.info(f"Scraping financial source: {label}")
        try:
            docs = fn(max_items=max_items)
            all_docs.extend(docs)
            logger.info(f"  → {len(docs)} items from {label}")
        except Exception as e:
            logger.warning(f"Failed to scrape {label}: {e}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_financial(max_items=5)
    print(f"Total: {len(docs)} financial items")
