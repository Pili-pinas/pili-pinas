"""
Tests for data_ingestion/processors/html_processor.py

These are pure functions — no mocking needed.
"""

import pytest
from data_ingestion.processors.html_processor import (
    clean_html,
    chunk_text,
    process_html_document,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class TestCleanHtml:
    def test_extracts_paragraph_text(self):
        html = "<html><body><p>Ang batas ay para sa lahat.</p></body></html>"
        result = clean_html(html)
        assert "Ang batas ay para sa lahat." in result

    def test_strips_script_tags(self):
        html = "<html><script>alert('xss')</script><p>Safe content.</p></html>"
        result = clean_html(html)
        assert "alert" not in result
        assert "Safe content." in result

    def test_strips_style_tags(self):
        html = "<html><style>body { color: red; }</style><p>Content.</p></html>"
        result = clean_html(html)
        assert "color: red" not in result
        assert "Content." in result

    def test_strips_nav_and_footer(self):
        html = (
            "<html><nav>Home | About</nav>"
            "<main><p>Main content.</p></main>"
            "<footer>Copyright 2025</footer></html>"
        )
        result = clean_html(html)
        assert "Home | About" not in result
        assert "Copyright 2025" not in result
        assert "Main content." in result

    def test_collapses_multiple_blank_lines(self):
        html = "<p>Para 1.</p>\n\n\n\n<p>Para 2.</p>"
        result = clean_html(html)
        assert "\n\n\n" not in result

    def test_collapses_multiple_spaces(self):
        html = "<p>Word1    Word2    Word3</p>"
        result = clean_html(html)
        assert "  " not in result

    def test_returns_stripped_string(self):
        html = "   <p>Content</p>   "
        result = clean_html(html)
        assert result == result.strip()


class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short text."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self):
        # ~2000 characters — exceeds default CHUNK_SIZE of 1500
        long_text = ("The Philippine Congress passed a landmark education bill. " * 40).strip()
        chunks = chunk_text(long_text)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_double_chunk_size(self):
        # Generous bound: sentence-boundary logic may produce slightly larger chunks
        long_text = ("Education funding bill. " * 80).strip()
        chunks = chunk_text(long_text, chunk_size=CHUNK_SIZE)
        for chunk in chunks:
            assert len(chunk) < CHUNK_SIZE * 2

    def test_no_empty_chunks_produced(self):
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = chunk_text(text)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_paragraph_boundaries_are_respected(self):
        # Two short paragraphs should stay together in one chunk
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert "First paragraph." in chunks[0]
        assert "Second paragraph." in chunks[0]

    def test_custom_chunk_size(self):
        text = "Word " * 100  # 500 chars
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # All chunks should be produced
        assert len(chunks) > 1


class TestProcessHtmlDocument:
    def test_html_document_is_cleaned(self):
        doc = {
            "text": "<html><script>badCode()</script><p>Good content here.</p></html>",
            "title": "Test Bill",
            "url": "https://senate.gov.ph/bill/1",
        }
        chunks = process_html_document(doc)
        assert len(chunks) >= 1
        assert all("badCode()" not in c["text"] for c in chunks)
        assert any("Good content here." in c["text"] for c in chunks)

    def test_plain_text_document_not_html_parsed(self):
        doc = {
            "text": "Plain text without any HTML tags.",
            "title": "Plain Doc",
            "url": "https://example.com",
        }
        chunks = process_html_document(doc)
        assert len(chunks) == 1
        assert "Plain text without any HTML tags." in chunks[0]["text"]

    def test_chunks_inherit_all_metadata(self, sample_doc):
        chunks = process_html_document(sample_doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["source"] == sample_doc["source"]
            assert chunk["source_type"] == sample_doc["source_type"]
            assert chunk["title"] == sample_doc["title"]
            assert chunk["url"] == sample_doc["url"]
            assert chunk["date"] == sample_doc["date"]

    def test_chunks_have_correct_index_fields(self):
        doc = {"text": "Short content.", "title": "T", "url": "https://example.com"}
        chunks = process_html_document(doc)
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["chunk_total"] == len(chunks)

    def test_multiple_chunks_have_sequential_indices(self):
        long_text = "The Senate convened today. " * 80
        doc = {"text": long_text, "title": "Long Doc", "url": "https://senate.gov.ph"}
        chunks = process_html_document(doc)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert chunk["chunk_total"] == len(chunks)

    def test_empty_text_returns_empty_list(self):
        doc = {"text": "", "title": "Empty", "url": "https://example.com"}
        assert process_html_document(doc) == []

    def test_html_with_only_nav_and_scripts_returns_empty(self):
        doc = {
            "text": "<html><nav>Menu</nav><script>code()</script></html>",
            "title": "Empty Page",
            "url": "https://example.com",
        }
        result = process_html_document(doc)
        assert result == []

    def test_bilingual_text_preserved(self):
        doc = {
            "text": "Ang batas ay inaprubahan ng Senado. The law was approved by the Senate.",
            "title": "Batas",
            "url": "https://senate.gov.ph",
        }
        chunks = process_html_document(doc)
        assert any("Ang batas" in c["text"] for c in chunks)
        assert any("The law" in c["text"] for c in chunks)
