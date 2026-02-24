"""
Shared fixtures for Pili-Pinas backend tests.
"""

import pytest


@pytest.fixture
def sample_doc():
    """A minimal document dict as produced by scrapers."""
    return {
        "source": "senate.gov.ph",
        "source_type": "bill",
        "date": "2025-01-15",
        "politician": "Juan dela Cruz",
        "title": "Senate Bill No. 1234",
        "url": "https://senate.gov.ph/bill/1234",
        "text": "This bill seeks to improve public education funding across all regions.",
    }


@pytest.fixture
def sample_chunks(sample_doc):
    """A list of chunk dicts as produced by process_html_document."""
    return [
        {**sample_doc, "text": f"Chunk {i}.", "chunk_index": i, "chunk_total": 3}
        for i in range(3)
    ]


@pytest.fixture
def mock_chroma_results():
    """Fake ChromaDB query results."""
    return {
        "documents": [["Relevant chunk about education.", "Another relevant chunk."]],
        "metadatas": [[
            {"title": "SB 1234", "source": "senate.gov.ph", "date": "2025-01-01", "url": "https://senate.gov.ph/1"},
            {"title": "News Article", "source": "rappler.com", "date": "2025-01-02", "url": "https://rappler.com/1"},
        ]],
        "distances": [[0.1, 0.25]],  # cosine distances (lower = more similar)
    }
