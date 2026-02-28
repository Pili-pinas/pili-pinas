"""
Persistent URL tracker to skip already-ingested articles on repeat scrapes.

Stores seen URLs in data/seen_urls.json with timestamps.
URLs older than TTL_DAYS are evicted so the file doesn't grow forever.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_SEEN_FILE = Path(__file__).parents[3] / "data" / "seen_urls.json"
TTL_DAYS = 90  # forget URLs older than this so content can be re-scraped eventually


class SeenURLTracker:
    def __init__(self, path: Path = _SEEN_FILE):
        self._path = path
        self._urls: dict[str, str] = {}  # url → ISO date first seen
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._urls = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load seen_urls: {e} — starting fresh")
                self._urls = {}

    def save(self) -> None:
        """Persist to disk, evicting entries older than TTL_DAYS."""
        cutoff = (datetime.now() - timedelta(days=TTL_DAYS)).strftime("%Y-%m-%d")
        self._urls = {u: d for u, d in self._urls.items() if d >= cutoff}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._urls, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def seen(self, url: str) -> bool:
        return url in self._urls

    def mark(self, url: str) -> None:
        if url not in self._urls:
            self._urls[url] = datetime.now().strftime("%Y-%m-%d")

    def __len__(self) -> int:
        return len(self._urls)
