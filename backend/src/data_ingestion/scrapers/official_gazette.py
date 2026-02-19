"""
Official Gazette of the Philippines scraper.
Scrapes laws, executive orders, and proclamations from officialgazette.gov.ph.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://officialgazette.gov.ph"
LAWS_URL = f"{BASE_URL}/section/laws"

RAW_DIR = Path(__file__).parents[4] / "data" / "raw" / "laws"

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
    logger.error(f"Failed to fetch: {url}")
    return None


def scrape_laws(max_pages: int = 5) -> list[dict]:
    """
    Scrape Republic Acts and other laws from the Official Gazette.
    Returns document dicts matching the metadata schema.
    """
    documents = []

    for page in range(1, max_pages + 1):
        params = {"page": page}
        resp = _get(LAWS_URL, params=params)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        articles = soup.select("article.post, div.entry, li.law-entry")

        if not articles:
            logger.info(f"No articles found on page {page}, stopping.")
            break

        for article in articles:
            title_tag = article.find("h2") or article.find("h3") or article.find("a")
            title = title_tag.get_text(strip=True) if title_tag else "Untitled"

            link_tag = article.find("a")
            url = link_tag["href"] if link_tag else ""
            if url and not url.startswith("http"):
                url = BASE_URL + url

            date_tag = article.find("time") or article.find("span", class_="date")
            raw_date = date_tag.get("datetime", "") or (date_tag.get_text(strip=True) if date_tag else "")
            try:
                date = datetime.fromisoformat(raw_date[:10]).strftime("%Y-%m-%d")
            except ValueError:
                date = datetime.now().strftime("%Y-%m-%d")

            doc = {
                "source": "officialgazette.gov.ph",
                "source_type": "law",
                "date": date,
                "politician": "",
                "title": title,
                "url": url,
                "text": "",
            }

            if url:
                doc = _scrape_law_detail(doc)

            documents.append(doc)
            logger.info(f"Scraped law: {title[:70]}")

        logger.info(f"Page {page}: {len(articles)} items")

    logger.info(f"Total laws scraped: {len(documents)}")
    return documents


def _scrape_law_detail(doc: dict) -> dict:
    """Fetch individual law page and extract full text."""
    resp = _get(doc["url"])
    if resp is None:
        return doc

    safe_name = doc["title"][:50].replace(" ", "_").replace("/", "-")
    raw_path = RAW_DIR / f"gazette_{safe_name}.html"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(resp.text, encoding="utf-8")

    soup = BeautifulSoup(resp.text, "lxml")

    content = (
        soup.find("div", class_="entry-content")
        or soup.find("div", class_="post-content")
        or soup.find("article")
    )
    if content:
        doc["text"] = content.get_text(separator="\n", strip=True)

    return doc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    laws = scrape_laws(max_pages=1)
    print(f"Scraped {len(laws)} laws")
