"""
Philippine government statistics scrapers.
Covers: PSA, BSP, DBM, NEDA (press releases) + World Bank Philippines API.

For validating politicians' performance claims against actual data.
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

# Press-release listing URLs for government statistics agencies.
STATISTICS_SOURCES = {
    "psa.gov.ph":  "https://psa.gov.ph/statistics/press-releases",
    "bsp.gov.ph":  "https://www.bsp.gov.ph/MediaPublications/Media/PressReleases",
    "dbm.gov.ph":  "https://www.dbm.gov.ph/index.php/news-and-updates",
    "neda.gov.ph": "https://neda.gov.ph/news/",
    "dilg.gov.ph": "https://dilg.gov.ph/news/",
}

# World Bank indicators for the Philippines (public REST API, no auth required).
WORLD_BANK_API = "https://api.worldbank.org/v2/country/PH/indicator"
WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG":    "Inflation, consumer prices (annual %)",
    "SI.POV.NAHC":       "Poverty headcount ratio at national poverty lines",
    "SL.UEM.TOTL.ZS":    "Unemployment, total (% of total labor force)",
    "SE.ADT.LITR.ZS":    "Literacy rate, adult total (% of people ages 15 and above)",
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
    """Parse common Philippine date formats into YYYY-MM-DD."""
    patterns = [
        ("%B %d, %Y", r"\b\w+ \d{1,2}, \d{4}\b"),
        ("%d %B %Y", r"\b\d{1,2} \w+ \d{4}\b"),
        ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
        ("%d %b %Y", r"\b\d{1,2} \w{3} \d{4}\b"),
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
    Scrape a government statistics agency press-release page.
    Tries multiple common HTML patterns.
    """
    resp = _get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    parsed_base = urlparse(url)
    base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"
    candidates: list[tuple[str, str, str]] = []

    # Pattern 1: Drupal-style views rows
    for row in soup.find_all(class_=re.compile(r"views-row|list-item|news-item|article-item")):
        link = row.find("a", href=True)
        if not link:
            continue
        title = link.get_text(strip=True)
        href = link["href"]
        date_el = row.find(class_=re.compile(r"date|time|published", re.I))
        date_text = date_el.get_text(strip=True) if date_el else ""
        if len(title) > 10:
            candidates.append((title, href, date_text))

    # Pattern 2: <article> tags
    if not candidates:
        for article in soup.find_all("article"):
            link = article.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            date_el = article.find(class_=re.compile(r"date|time", re.I))
            date_text = date_el.get_text(strip=True) if date_el else ""
            if len(title) > 10:
                candidates.append((title, href, date_text))

    # Pattern 3: <table> rows (BSP style)
    if not candidates:
        for row in soup.find_all("tr"):
            link = row.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            cells = row.find_all("td")
            date_text = cells[-1].get_text(strip=True) if len(cells) > 1 else ""
            if len(title) > 10:
                candidates.append((title, href, date_text))

    # Pattern 4: any link with meaningful title
    if not candidates:
        for link in soup.find_all("a", href=True):
            title = link.get_text(strip=True)
            if len(title) > 20 and not any(w in title.lower() for w in ("home", "about", "contact")):
                candidates.append((title, link["href"], ""))

    documents = []
    for title, href, date_text in candidates[:max_items]:
        if href.startswith("/"):
            href = base_url + href
        documents.append({
            "source": source,
            "source_type": "statistics",
            "date": _parse_date(date_text) if date_text else datetime.now().strftime("%Y-%m-%d"),
            "politician": "",
            "title": title,
            "url": href,
            "text": title,
        })

    return documents


def scrape_world_bank(max_items: int = 20) -> list[dict]:
    """
    Fetch key Philippine economic indicators from the World Bank public API.
    Returns one document per indicator data point (latest 5 years each).
    """
    documents = []
    per_page = min(5, max_items)

    for indicator_id, indicator_name in WB_INDICATORS.items():
        url = f"{WORLD_BANK_API}/{indicator_id}?format=json&per_page={per_page}&mrv={per_page}"
        resp = _get(url)
        if resp is None:
            continue

        try:
            payload = resp.json()
            # World Bank API returns [metadata, data_array]
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            data_points = payload[1]
            if not data_points:
                continue

            # Build a single summary doc covering the last N years
            values = [
                f"{p['date']}: {p['value']:.2f}"
                for p in data_points
                if p.get("value") is not None
            ]
            if not values:
                continue

            text = (
                f"Philippines {indicator_name}. "
                f"Recent data (World Bank): {'; '.join(values)}."
            )

            documents.append({
                "source": "data.worldbank.org",
                "source_type": "statistics",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "politician": "",
                "title": f"Philippines {indicator_name} — World Bank Data",
                "url": f"https://data.worldbank.org/indicator/{indicator_id}?locations=PH",
                "text": text,
            })

        except Exception as e:
            logger.warning(f"Failed to parse World Bank indicator {indicator_id}: {e}")

        if len(documents) >= max_items:
            break

    return documents


def scrape_all_statistics(max_items: int = 20) -> list[dict]:
    """Scrape all configured statistics sources and the World Bank API."""
    all_docs = []

    for source, url in STATISTICS_SOURCES.items():
        logger.info(f"Scraping statistics source: {source}")
        try:
            docs = _scrape_press_releases(source, url, max_items=max_items)
            all_docs.extend(docs)
            logger.info(f"  → {len(docs)} items from {source}")
        except Exception as e:
            logger.warning(f"Failed to scrape {source}: {e}")

    logger.info("Scraping World Bank API...")
    try:
        wb_docs = scrape_world_bank(max_items=max_items)
        all_docs.extend(wb_docs)
        logger.info(f"  → {len(wb_docs)} World Bank indicator docs")
    except Exception as e:
        logger.warning(f"World Bank API failed: {e}")

    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_statistics(max_items=5)
    print(f"Total: {len(docs)} statistics items")
