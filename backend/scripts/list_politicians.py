"""
List all politicians that have a profile document in the processed data.

Usage:
    python backend/scripts/list_politicians.py
    python backend/scripts/list_politicians.py --source chromadb   # query vector store instead
    python backend/scripts/list_politicians.py --search robredo    # filter by name
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

_DEFAULT_PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(_DEFAULT_PROCESSED_DIR)))


_PROFILE_JSONL = ["politicians.jsonl", "senators.jsonl", "house_members.jsonl"]


def from_jsonl(search: str | None = None) -> list[str]:
    names = set()
    found_any = False

    for filename in _PROFILE_JSONL:
        path = PROCESSED_DIR / filename
        if not path.exists():
            continue
        found_any = True
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    name = doc.get("politician", "").strip()
                    if name:
                        names.add(name)
                except json.JSONDecodeError:
                    pass

    if not found_any:
        print(f"No profile JSONL files found in {PROCESSED_DIR}")
        print("Run ingestion with 'senators', 'house_members', or 'politicians' sources first.")

    results = sorted(names)
    if search:
        results = [n for n in results if search.lower() in n.lower()]
    return results


def from_chromadb(search: str | None = None) -> list[str]:
    from embeddings.vector_store import get_vector_store

    store = get_vector_store()
    # ChromaDB .get() with a where filter — bypasses embedding step
    raw = store._collection.get(
        where={"source_type": "profile"},
        include=["metadatas"],
        limit=10_000,
    )

    names = set()
    for meta in raw.get("metadatas", []):
        name = (meta or {}).get("politician", "").strip()
        if name:
            names.add(name)

    results = sorted(names)
    if search:
        results = [n for n in results if search.lower() in n.lower()]
    return results


def main():
    parser = argparse.ArgumentParser(description="List politicians with profiles")
    parser.add_argument("--source", choices=["jsonl", "chromadb"], default="jsonl",
                        help="Data source to query (default: jsonl)")
    parser.add_argument("--search", default=None,
                        help="Filter results by name (case-insensitive)")
    args = parser.parse_args()

    if args.source == "chromadb":
        names = from_chromadb(args.search)
        label = "ChromaDB"
    else:
        names = from_jsonl(args.search)
        label = "politicians.jsonl"

    if not names:
        print("No politicians found.")
        return

    print(f"Politicians with profiles ({label}): {len(names)}\n")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
