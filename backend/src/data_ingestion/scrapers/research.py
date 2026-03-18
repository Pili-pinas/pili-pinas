"""
Philippine research institution publication scrapers.
Covers: PIDS, ADB Philippines, UNDP Philippines, IMF Philippines,
        Transparency International Philippines, UP CIDS.

Scrapes publication listings from each institution's website.
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

RESEARCH_SOURCES = {
    "pids.gov.ph":          "https://pids.gov.ph/publications",
    "adb.org":              "https://www.adb.org/countries/philippines/publications",
    "undp.org":             "https://www.undp.org/philippines/publications",
    "imf.org":              "https://www.imf.org/en/countries/PHL",
    "transparency.org":     "https://www.transparency.org/en/countries/philippines",
    "cids.up.edu.ph":       "https://cids.up.edu.ph/publications/",
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
    """Try to parse common publication date formats into YYYY-MM-DD."""
    patterns = [
        ("%B %Y",    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b"),
        ("%b %Y",    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b"),
        ("%d %b %Y", r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b"),
        ("%B %d, %Y",r"\b\w+ \d{1,2}, \d{4}\b"),
        ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
    ]
    for fmt, pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            try:
                parsed = datetime.strptime(m.group(), fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return datetime.now().strftime("%Y-%m-%d")


def _scrape_publications(source: str, url: str, max_items: int = 20) -> list[dict]:
    """
    Scrape a research publication listing page and return document dicts.

    Tries multiple common HTML patterns:
    - <div class="publication-*"> containers (PIDS-style)
    - <li class="pub-*"> items in publication lists (ADB-style)
    - <article> tags with links
    - <h2>/<h3> heading links
    """
    resp = _get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    candidates: list[tuple[str, str, str]] = []

    # Pattern 1: publication-item divs (PIDS-style)
    pub_items = soup.find_all("div", class_=re.compile(r"publication-item|pub-item|publication_item", re.I))
    for item in pub_items:
        link = item.find("a", href=True)
        if not link:
            continue
        title = link.get_text(strip=True)
        href = link["href"]
        date_el = item.find(class_=re.compile(r"date|time|pub-date", re.I))
        date_text = date_el.get_text(strip=True) if date_el else ""
        if title:
            candidates.append((title, href, date_text))

    # Pattern 2: <li class="pub-item"> (ADB-style)
    if not candidates:
        for li in soup.find_all("li", class_=re.compile(r"pub-item|publication", re.I)):
            link = li.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            time_el = li.find("time")
            date_text = time_el.get("datetime", time_el.get_text(strip=True)) if time_el else ""
            if title:
                candidates.append((title, href, date_text))

    # Pattern 3: <article> tags
    if not candidates:
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

    # Pattern 4: heading links (h2/h3)
    if not candidates:
        for heading in soup.find_all(["h2", "h3"]):
            link = heading.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            if len(title) > 15:
                candidates.append((title, href, ""))

    if not candidates:
        return []

    documents = []
    for title, href, date_text in candidates[:max_items]:
        if href.startswith("/"):
            href = f"{base_url}{href}"
        elif not href.startswith("http"):
            href = f"{base_url}/{href}"

        documents.append({
            "source": source,
            "source_type": "research",
            "date": _parse_date(date_text) if date_text else datetime.now().strftime("%Y-%m-%d"),
            "politician": "",
            "title": title,
            "url": href,
            "text": title,
        })

    return documents


def scrape_all_research(max_items: int = 20) -> list[dict]:
    """Scrape all configured research institution publication pages."""
    all_docs = []
    for source, url in RESEARCH_SOURCES.items():
        logger.info(f"Scraping research source: {source}")
        try:
            docs = _scrape_publications(source, url, max_items=max_items)
            all_docs.extend(docs)
            logger.info(f"  → {len(docs)} items from {source}")
        except Exception as e:
            logger.warning(f"Failed to scrape {source}: {e}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_research(max_items=5)
    print(f"Total: {len(docs)} research items")
