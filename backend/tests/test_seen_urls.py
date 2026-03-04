"""
Tests for data_ingestion/seen_urls.py (SeenURLTracker)

All disk I/O goes to a temp directory — no side effects on the real data/.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from data_ingestion.seen_urls import SeenURLTracker, TTL_DAYS


@pytest.fixture
def tracker(tmp_path):
    """A fresh SeenURLTracker backed by a temp file."""
    return SeenURLTracker(path=tmp_path / "seen_urls.json")


@pytest.fixture
def populated_tracker(tmp_path):
    """A tracker pre-loaded with one URL."""
    t = SeenURLTracker(path=tmp_path / "seen_urls.json")
    t.mark("https://example.com/article-1")
    return t


class TestSeenURLTrackerInit:
    def test_starts_empty_when_no_file(self, tmp_path):
        t = SeenURLTracker(path=tmp_path / "new.json")
        assert len(t) == 0

    def test_loads_existing_file_on_init(self, tmp_path):
        path = tmp_path / "seen.json"
        path.write_text(json.dumps({"https://example.com/": "2025-01-01"}))
        t = SeenURLTracker(path=path)
        assert t.seen("https://example.com/")

    def test_starts_fresh_on_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{{{{ not json")
        t = SeenURLTracker(path=path)
        assert len(t) == 0


class TestMark:
    def test_mark_adds_url(self, tracker):
        tracker.mark("https://example.com/1")
        assert tracker.seen("https://example.com/1")

    def test_mark_records_todays_date(self, tracker):
        tracker.mark("https://example.com/1")
        today = datetime.now().strftime("%Y-%m-%d")
        assert tracker._urls["https://example.com/1"] == today

    def test_mark_does_not_overwrite_existing_entry(self, tracker):
        tracker._urls["https://example.com/1"] = "2025-01-01"
        tracker.mark("https://example.com/1")
        assert tracker._urls["https://example.com/1"] == "2025-01-01"

    def test_mark_multiple_urls(self, tracker):
        urls = [f"https://example.com/{i}" for i in range(5)]
        for url in urls:
            tracker.mark(url)
        assert len(tracker) == 5


class TestSeen:
    def test_seen_returns_false_for_unknown_url(self, tracker):
        assert tracker.seen("https://example.com/unknown") is False

    def test_seen_returns_true_after_mark(self, tracker):
        tracker.mark("https://example.com/article")
        assert tracker.seen("https://example.com/article") is True

    def test_seen_is_exact_match(self, tracker):
        tracker.mark("https://example.com/article")
        assert tracker.seen("https://example.com/article/") is False


class TestSave:
    def test_save_creates_file(self, tracker, tmp_path):
        tracker.mark("https://example.com/1")
        tracker.save()
        assert (tmp_path / "seen_urls.json").exists()

    def test_save_persists_urls(self, tracker, tmp_path):
        tracker.mark("https://example.com/1")
        tracker.save()
        data = json.loads((tmp_path / "seen_urls.json").read_text())
        assert "https://example.com/1" in data

    def test_save_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "seen.json"
        t = SeenURLTracker(path=deep_path)
        t.mark("https://example.com/1")
        t.save()
        assert deep_path.exists()

    def test_save_evicts_entries_older_than_ttl(self, tracker, tmp_path):
        cutoff = datetime.now() - timedelta(days=TTL_DAYS + 1)
        tracker._urls["https://old.example.com/"] = cutoff.strftime("%Y-%m-%d")
        tracker._urls["https://new.example.com/"] = datetime.now().strftime("%Y-%m-%d")
        tracker.save()

        data = json.loads((tmp_path / "seen_urls.json").read_text())
        assert "https://old.example.com/" not in data
        assert "https://new.example.com/" in data

    def test_save_keeps_entries_within_ttl(self, tracker, tmp_path):
        recent = (datetime.now() - timedelta(days=TTL_DAYS - 1)).strftime("%Y-%m-%d")
        tracker._urls["https://recent.example.com/"] = recent
        tracker.save()

        data = json.loads((tmp_path / "seen_urls.json").read_text())
        assert "https://recent.example.com/" in data

    def test_roundtrip_save_and_reload(self, tmp_path):
        path = tmp_path / "seen.json"
        t1 = SeenURLTracker(path=path)
        t1.mark("https://example.com/a")
        t1.mark("https://example.com/b")
        t1.save()

        t2 = SeenURLTracker(path=path)
        assert t2.seen("https://example.com/a")
        assert t2.seen("https://example.com/b")
        assert len(t2) == 2


class TestLen:
    def test_len_zero_on_empty_tracker(self, tracker):
        assert len(tracker) == 0

    def test_len_increases_with_marks(self, tracker):
        tracker.mark("https://a.com/")
        tracker.mark("https://b.com/")
        assert len(tracker) == 2

    def test_len_does_not_double_count(self, tracker):
        tracker.mark("https://a.com/")
        tracker.mark("https://a.com/")  # duplicate
        assert len(tracker) == 1
