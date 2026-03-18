"""
Philippine fact-checking site scrapers.
Covers: Vera Files, Tsek.ph

Uses RSS feeds. No politics filter needed — all fact-check content is politically relevant.
"""

import logging
import time
import xml.etree.ElementTree as ET
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

# Fact-checking RSS feeds.
FACT_CHECK_FEEDS = {
    "verafiles.org": "https://verafiles.org/feed",
    "tsek.ph":       "https://tsek.ph/feed/",
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


def _parse_rss_date(date_str: str) -> str:
    """Parse RSS pubDate to YYYY-MM-DD."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.now().strftime("%Y-%m-%d")


def scrape_rss_feed(source: str, feed_url: str, max_items: int = 20) -> list[dict]:
    """Parse an RSS feed and return fact-check document dicts."""
    resp = _get(feed_url)
    if resp is None:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse RSS from {feed_url}: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    documents = []
    for item in items:
        title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
        url = (item.findtext("link") or item.findtext("atom:link", namespaces=ns) or "").strip()
        pub_date = item.findtext("pubDate") or item.findtext("atom:published", namespaces=ns) or ""
        description = item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or ""

        date = _parse_rss_date(pub_date) if pub_date else datetime.now().strftime("%Y-%m-%d")
        text = BeautifulSoup(description, "lxml").get_text(strip=True)

        documents.append({
            "source": source,
            "source_type": "fact_check",
            "date": date,
            "politician": "",
            "title": title,
            "url": url,
            "text": text or title,
        })

        if len(documents) >= max_items:
            break

    return documents


def scrape_all_fact_checks(max_items: int = 20) -> list[dict]:
    """Scrape all configured fact-checking RSS feeds."""
    all_docs = []
    for source, feed_url in FACT_CHECK_FEEDS.items():
        logger.info(f"Scraping fact-check feed: {source}")
        docs = scrape_rss_feed(source, feed_url, max_items=max_items)
        all_docs.extend(docs)
        logger.info(f"  → {len(docs)} items from {source}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_fact_checks(max_items=5)
    print(f"Total: {len(docs)} fact-check items")
