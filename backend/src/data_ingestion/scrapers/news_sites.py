"""
Philippine news site scrapers.
Covers: Rappler, Inquirer, PhilStar, Manila Bulletin, GMA News.

Each scraper fetches recent articles about Philippine politics/government.
We use RSS feeds where available (less brittle than HTML scraping).
"""

import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[4] / "data" / "raw" / "news_articles"

HEADERS = {
    "User-Agent": (
        "PiliPinas/1.0 (informed-voter research tool; "
        "contact: pilipinas-bot@example.com)"
    )
}

# Politics-focused RSS feeds (one per source, most specific available)
RSS_FEEDS = {
    "rappler.com": "https://www.rappler.com/politics-government/feed/",
    "inquirer.net": "https://politics.inquirer.net/feed/",
    "philstar.com": "https://www.philstar.com/rss/politics",
    "mb.com.ph": "https://mb.com.ph/category/news/national/feed/",
    "gmanetwork.com": "https://data.gmanetwork.com/gno/rss/news/government/feed.xml",
}

# Keywords used to filter article titles/descriptions.
# An article must match at least one keyword to be kept.
POLITICS_KEYWORDS = [
    # Institutions
    "senate", "senado", "congress", "house of representatives", "comelec",
    "supreme court", "ombudsman", "malacañang", "palace",
    # Roles
    "senator", "congressman", "representative", "president", "vice president",
    "governor", "mayor", "pangulo", "gobernador", "alkalde", "cabinet",
    # Process
    "election", "halalan", "vote", "boto", "campaign", "kandidato", "candidate",
    "bill", "law", "batas", "republic act", "resolution", "budget", "appropriation",
    "impeach", "resign", "appoint", "saln", "corruption", "plunder",
    # Parties / coalitions (generic)
    "political party", "coalition", "opposition", "administration",
    # High-profile names (add more as relevant)
    "marcos", "duterte", "robredo", "cayetano", "lacson", "escudero",
    "bongbong", "sara", "leni",
]

_KEYWORDS_LOWER = [kw.lower() for kw in POLITICS_KEYWORDS]


def _is_politics_related(title: str, description: str) -> bool:
    """Return True if the article title or description matches any politics keyword."""
    text = f"{title} {description}".lower()
    return any(kw in text for kw in _KEYWORDS_LOWER)


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
    """
    Parse an RSS feed and return document dicts.
    Fetches full article text for each item.
    """
    resp = _get(feed_url)
    if resp is None:
        return []

    documents = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse RSS from {feed_url}: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    for item in items[:max_items * 3]:  # scan more items to hit max_items after filtering
        # Extract fields (handles both RSS 2.0 and Atom)
        title = (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "").strip()
        url = (item.findtext("link") or item.findtext("atom:link", namespaces=ns) or "").strip()
        pub_date = item.findtext("pubDate") or item.findtext("atom:published", namespaces=ns) or ""
        description = item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or ""

        # Skip non-political articles before fetching full text
        if not _is_politics_related(title, description):
            logger.debug(f"[{source}] Skipped (not politics): {title[:70]}")
            continue

        date = _parse_rss_date(pub_date) if pub_date else datetime.now().strftime("%Y-%m-%d")

        doc = {
            "source": source,
            "source_type": "news",
            "date": date,
            "politician": "",  # extracted in post-processing if needed
            "title": title,
            "url": url,
            "text": BeautifulSoup(description, "lxml").get_text(strip=True),
        }

        # Fetch full article text
        if url:
            full_text = _fetch_article_text(url, source)
            if full_text:
                doc["text"] = full_text

        documents.append(doc)
        logger.info(f"[{source}] {title[:70]}")

        if len(documents) >= max_items:
            break

    return documents


def _fetch_article_text(url: str, source: str) -> str:
    """Fetch and extract article body text."""
    resp = _get(url)
    if resp is None:
        return ""

    # Save raw HTML
    safe_name = url.split("/")[-1][:50] or "article"
    raw_path = RAW_DIR / source / f"{safe_name}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(resp.text, encoding="utf-8")

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove nav, ads, scripts
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    # Try common article content selectors
    content = (
        soup.find("div", class_="article-body")
        or soup.find("div", class_="story-body")
        or soup.find("div", class_="entry-content")
        or soup.find("article")
        or soup.find("main")
    )

    if content:
        return content.get_text(separator="\n", strip=True)
    return ""


def scrape_all_news(max_items_per_source: int = 20) -> list[dict]:
    """Scrape political news from all configured RSS feeds."""
    all_docs = []
    for source, feed_url in RSS_FEEDS.items():
        logger.info(f"Scraping {source}...")
        docs = scrape_rss_feed(source, feed_url, max_items=max_items_per_source)
        all_docs.extend(docs)
        logger.info(f"  → {len(docs)} articles from {source}")
    logger.info(f"Total news articles scraped: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_all_news(max_items_per_source=5)
    print(f"Total: {len(docs)} articles")
