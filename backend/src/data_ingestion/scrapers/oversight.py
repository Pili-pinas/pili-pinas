"""
Philippine independent oversight body scrapers.
Covers: COA (Commission on Audit), Office of the Ombudsman,
        Sandiganbayan (anti-graft court), Civil Service Commission.

Scrapes press releases and news listings from each body's website.
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}

# Press-release / news listing URLs for each oversight body.
OVERSIGHT_SOURCES = {
    "coa.gov.ph":           "https://www.coa.gov.ph/index.php/press-releases",
    "ombudsman.gov.ph":     "https://www.ombudsman.gov.ph/news-and-announcements",
    "sandiganbayan.gov.ph": "https://sandiganbayan.gov.ph/index.php/press-releases",
    "csc.gov.ph":           "https://www.csc.gov.ph/latest-news",
}


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
    """Try to parse common Philippine date formats into YYYY-MM-DD."""
    patterns = [
        ("%B %d, %Y", r"\b\w+ \d{1,2}, \d{4}\b"),
        ("%d %B %Y", r"\b\d{1,2} \w+ \d{4}\b"),
        ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
    ]
    for fmt, pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return datetime.strptime(m.group(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return datetime.now().strftime("%Y-%m-%d")


def _scrape_press_releases(source: str, url: str, max_items: int = 20) -> list[dict]:
    """
    Scrape a press-release listing page and return document dicts.

    Tries multiple common HTML patterns:
    - <article> tags with <a> inside
    - <li> items in news lists
    - <div> containers with headings and links
    """
    resp = _get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    documents = []

    # Collect (title, href, date_text) tuples from common listing patterns.
    candidates: list[tuple[str, str, str]] = []

    # Pattern 1: <article> with a heading link
    for article in soup.find_all("article"):
        link = article.find("a", href=True)
        if not link:
            continue
        title = link.get_text(strip=True)
        href = link["href"]
        date_el = article.find(class_=re.compile(r"date|time|published", re.I))
        date_text = date_el.get_text(strip=True) if date_el else ""
        if title:
            candidates.append((title, href, date_text))

    # Pattern 2: <li> with <a> links (news lists)
    if not candidates:
        for li in soup.find_all("li"):
            link = li.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            date_el = li.find(class_=re.compile(r"date|time|published", re.I))
            date_text = date_el.get_text(strip=True) if date_el else ""
            # Skip navigation/menu items (too short, generic words)
            if len(title) > 15 and not any(w in title.lower() for w in ("home", "about", "contact", "menu")):
                candidates.append((title, href, date_text))

    # Pattern 3: heading tags (h2/h3) with links
    if not candidates:
        for heading in soup.find_all(["h2", "h3"]):
            link = heading.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            if len(title) > 15:
                candidates.append((title, href, ""))

    for title, href, date_text in candidates[:max_items]:
        # Resolve relative URLs
        if href.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"

        documents.append({
            "source": source,
            "source_type": "oversight",
            "date": _parse_date(date_text) if date_text else datetime.now().strftime("%Y-%m-%d"),
            "politician": "",
            "title": title,
            "url": href,
            "text": title,  # listing page; full text would require a follow-up fetch
        })

    return documents


def scrape_all_oversight(max_items: int = 20) -> list[dict]:
    """Scrape all configured oversight body press-release pages."""
    all_docs = []
    for source, url in OVERSIGHT_SOURCES.items():
        logger.info(f"Scraping oversight body: {source}")
        try:
            docs = _scrape_press_releases(source, url, max_items=max_items)
            all_docs.extend(docs)
            logger.info(f"  → {len(docs)} items from {source}")
        except Exception as e:
            logger.warning(f"Failed to scrape {source}: {e}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_oversight(max_items=5)
    print(f"Total: {len(docs)} oversight items")
