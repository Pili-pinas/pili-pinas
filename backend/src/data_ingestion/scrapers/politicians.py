"""
Enriched politician profile scraper.

Builds one profile document per politician by combining:
  - Identity + roles from BetterGov /api/people (all congresses, all positions)
  - Bills they authored, matched from pre-scraped bill documents

Unlike scrape_senators() / scrape_members(), this covers everyone in the API
regardless of position or congress — including former reps, VPs, and candidates.

Usage:
    from data_ingestion.scrapers.politicians import scrape_all_politicians

    bills = scrape_senate_bills(...) + scrape_house_bills(...)
    profiles = scrape_all_politicians(bills=bills)
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Reuse the existing paginator from the senate scraper to avoid duplication.
from data_ingestion.scrapers.senate import _fetch_all_people


def _build_full_name(person: dict) -> str:
    parts = [
        person.get("name_prefix") or "",
        person.get("first_name") or "",
        person.get("middle_name") or "",
        person.get("last_name") or "",
        person.get("name_suffix") or "",
    ]
    return " ".join(p for p in parts if p)


def _bills_for_person(person: dict, bills: list[dict]) -> list[dict]:
    """
    Return bills where the person's last name appears in the politician/authors field.

    Matching by last name covers the common BetterGov formats:
      "Robredo, Maria Leonor"  /  "Rep. Leni Robredo"  /  "ROBREDO"
    """
    last_name = (person.get("last_name") or "").strip().lower()
    if not last_name:
        return []
    return [
        b for b in bills
        if last_name in (b.get("politician") or "").lower()
    ]


def _build_enriched_profile(person: dict, bills: list[dict]) -> dict:
    full_name = _build_full_name(person)

    congresses_served = person.get("congresses_served") or []
    role_parts = []
    for c in congresses_served:
        pos = c.get("position", "")
        ordinal = c.get("congress_ordinal", "")
        if pos and ordinal:
            role_parts.append(f"{pos} ({ordinal} Congress)")

    text_parts = [f"{full_name}."]
    if role_parts:
        text_parts.append("Roles: " + "; ".join(role_parts) + ".")

    aliases = [a for a in (person.get("aliases") or []) if a]
    if aliases:
        text_parts.append(f"Also known as: {', '.join(aliases)}.")

    authored = _bills_for_person(person, bills)
    if authored:
        titles = [b["title"] for b in authored[:20]]
        text_parts.append("Bills authored: " + "; ".join(titles) + ".")

    return {
        "source": "open-congress-api.bettergov.ph",
        "source_type": "profile",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "politician": full_name,
        "title": f"Politician Profile: {full_name}",
        "url": f"https://open-congress-api.bettergov.ph/api/people/{person['id']}",
        "text": " ".join(text_parts),
    }


def scrape_all_politicians(bills: list[dict] | None = None) -> list[dict]:
    """
    Build enriched profile documents for all politicians in the BetterGov API.

    Args:
        bills: Pre-scraped bill documents (output of scrape_bills + scrape_house_bills
               combined). Each profile will list the bills that person authored.
               Pass None or omit to build profiles without bill history.

    Returns:
        List of profile document dicts matching the metadata schema.
    """
    people = _fetch_all_people()
    bills = bills or []
    docs = []

    for person in people:
        if not person.get("congresses_served"):
            continue
        doc = _build_enriched_profile(person, bills)
        docs.append(doc)
        logger.info(f"Built profile: {doc['politician']}")

    logger.info(f"Total politician profiles built: {len(docs)}")
    return docs
