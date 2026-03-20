"""
Coverage audit: check which keywords/politicians have data in the vector DB.

Uses ChromaDB's where_document $contains filter (text match, not vector similarity)
so results reflect actual document content regardless of embedding quality.

Usage:
    cd backend
    python scripts/check_coverage.py
    python scripts/check_coverage.py --keywords "Leni Robredo" "Bongbong Marcos"
    python scripts/check_coverage.py --keywords-file keywords.txt
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

DEFAULT_KEYWORDS = [
    # Prominent politicians
    "Leni Robredo",
    "Bongbong Marcos",
    "Sara Duterte",
    "Rodrigo Duterte",
    "Manny Pacquiao",
    "Kiko Pangilinan",
    "Isko Moreno",
    "Ping Lacson",
    "Sonny Angara",
    "Cynthia Villar",
    "Leila de Lima",
    "Nancy Binay",
    "Imee Marcos",
    "Risa Hontiveros",
    "Grace Poe",
    # Institutions
    "Commission on Audit",
    "Ombudsman",
    "Supreme Court",
    "COMELEC",
    "PhilGEPS",
    # Key topics
    "SALN",
    "pork barrel",
    "anti-dynasty",
    "Maharlika",
    "Mandanas ruling",
]


@dataclass
class CoverageResult:
    keyword: str
    count: int
    found: bool
    samples: list[dict] = field(default_factory=list)


def check_keyword(keyword: str, store) -> CoverageResult:
    """
    Check if `keyword` appears in any document in the vector store.
    Uses case-insensitive text matching via ChromaDB's where_document $contains.
    """
    term = keyword.lower()
    result = store._collection.get(
        where_document={"$contains": term},
        limit=100,
        include=["documents", "metadatas"],
    )

    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    count = len(docs)

    samples = [
        {
            "title": m.get("title", ""),
            "source": m.get("source", ""),
            "source_type": m.get("source_type", ""),
        }
        for m in metas[:3]
    ]

    return CoverageResult(
        keyword=keyword,
        count=count,
        found=count > 0,
        samples=samples,
    )


def audit_keywords(keywords: list[str], store) -> list[CoverageResult]:
    return [check_keyword(kw, store) for kw in keywords]


def print_report(results: list[CoverageResult]) -> None:
    found = [r for r in results if r.found]
    missing = [r for r in results if not r.found]

    print(f"\n{'='*60}")
    print(f"COVERAGE AUDIT  |  {len(found)}/{len(results)} keywords found")
    print(f"{'='*60}")

    if found:
        print(f"\n✓ FOUND ({len(found)})")
        for r in sorted(found, key=lambda x: -x.count):
            print(f"  [{r.count:4d} chunks]  {r.keyword}")
            for s in r.samples:
                label = f"{s['source_type']} | {s['source']}"
                title = s['title'][:60] + "…" if len(s['title']) > 60 else s['title']
                print(f"             → {title}  ({label})")

    if missing:
        print(f"\n✗ MISSING ({len(missing)})")
        for r in missing:
            print(f"  {r.keyword}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Audit vector DB coverage for target keywords.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--keywords", nargs="+", metavar="KW", help="Keywords to check")
    group.add_argument("--keywords-file", metavar="FILE", help="File with one keyword per line")
    args = parser.parse_args()

    if args.keywords:
        keywords = args.keywords
    elif args.keywords_file:
        keywords = [l.strip() for l in Path(args.keywords_file).read_text().splitlines() if l.strip()]
    else:
        keywords = DEFAULT_KEYWORDS

    from embeddings.vector_store import get_vector_store
    store = get_vector_store()
    total = store.count()
    print(f"Vector DB: {total} total chunks")

    results = audit_keywords(keywords, store)
    print_report(results)

    missing = [r for r in results if not r.found]
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
