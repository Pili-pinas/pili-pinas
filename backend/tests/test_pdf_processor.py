"""
Tests for data_ingestion/processors/pdf_processor.py

pdfplumber and requests are mocked — no real files or HTTP calls needed.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from data_ingestion.processors.pdf_processor import (
    extract_pdf_text,
    process_pdf_document,
    download_and_process_pdf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_mock(*page_texts):
    """Build a mock pdfplumber PDF object with given per-page texts."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    pdf_obj = MagicMock()
    pdf_obj.pages = pages
    pdf_obj.__enter__ = lambda s: s
    pdf_obj.__exit__ = MagicMock(return_value=False)
    return pdf_obj


# ---------------------------------------------------------------------------
# extract_pdf_text
# ---------------------------------------------------------------------------

class TestExtractPdfText:
    def test_returns_empty_string_for_missing_file(self, tmp_path):
        result = extract_pdf_text(tmp_path / "nonexistent.pdf")
        assert result == ""

    def test_extracts_text_from_single_page(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")

        mock_pdf = _make_pdf_mock("Republic Act content here.")
        with patch("data_ingestion.processors.pdf_processor.pdfplumber.open", return_value=mock_pdf):
            result = extract_pdf_text(pdf_path)

        assert "Republic Act content here." in result

    def test_includes_page_numbers_in_output(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")

        mock_pdf = _make_pdf_mock("Page one text.", "Page two text.")
        with patch("data_ingestion.processors.pdf_processor.pdfplumber.open", return_value=mock_pdf):
            result = extract_pdf_text(pdf_path)

        assert "[Page 1]" in result
        assert "[Page 2]" in result

    def test_skips_pages_with_no_text(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")

        mock_pdf = _make_pdf_mock("Page one.", None, "Page three.")
        with patch("data_ingestion.processors.pdf_processor.pdfplumber.open", return_value=mock_pdf):
            result = extract_pdf_text(pdf_path)

        assert "[Page 1]" in result
        assert "[Page 2]" not in result
        assert "[Page 3]" in result

    def test_returns_empty_string_on_pdfplumber_exception(self, tmp_path):
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_bytes(b"not a pdf")

        with patch("data_ingestion.processors.pdf_processor.pdfplumber.open",
                   side_effect=Exception("Invalid PDF")):
            result = extract_pdf_text(pdf_path)

        assert result == ""

    def test_concatenates_multiple_pages(self, tmp_path):
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF fake")

        mock_pdf = _make_pdf_mock("First.", "Second.", "Third.")
        with patch("data_ingestion.processors.pdf_processor.pdfplumber.open", return_value=mock_pdf):
            result = extract_pdf_text(pdf_path)

        assert "First." in result
        assert "Second." in result
        assert "Third." in result


# ---------------------------------------------------------------------------
# process_pdf_document
# ---------------------------------------------------------------------------

class TestProcessPdfDocument:
    METADATA = {
        "source": "comelec.gov.ph",
        "source_type": "election",
        "date": "2025-01-01",
        "title": "2025 Candidate List",
        "url": "https://comelec.gov.ph/candidates.pdf",
    }

    def test_returns_empty_list_when_pdf_has_no_text(self, tmp_path):
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF fake")

        with patch("data_ingestion.processors.pdf_processor.extract_pdf_text", return_value=""):
            result = process_pdf_document(pdf_path, self.METADATA)

        assert result == []

    def test_returns_chunk_dicts(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")
        text = "Sample content. " * 50  # enough to produce chunks

        with patch("data_ingestion.processors.pdf_processor.extract_pdf_text", return_value=text):
            result = process_pdf_document(pdf_path, self.METADATA)

        assert len(result) > 0

    def test_chunks_include_metadata_fields(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")
        text = "Content. " * 50

        with patch("data_ingestion.processors.pdf_processor.extract_pdf_text", return_value=text):
            chunks = process_pdf_document(pdf_path, self.METADATA)

        for chunk in chunks:
            assert chunk["source"] == "comelec.gov.ph"
            assert chunk["source_type"] == "election"
            assert chunk["url"] == self.METADATA["url"]

    def test_chunks_have_chunk_index_and_total(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")
        text = "Word. " * 200

        with patch("data_ingestion.processors.pdf_processor.extract_pdf_text", return_value=text):
            chunks = process_pdf_document(pdf_path, self.METADATA)

        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert chunk["chunk_total"] == len(chunks)

    def test_each_chunk_has_text_field(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF fake")
        text = "Legislation text. " * 50

        with patch("data_ingestion.processors.pdf_processor.extract_pdf_text", return_value=text):
            chunks = process_pdf_document(pdf_path, self.METADATA)

        for chunk in chunks:
            assert "text" in chunk
            assert len(chunk["text"]) > 0


# ---------------------------------------------------------------------------
# download_and_process_pdf
# ---------------------------------------------------------------------------

class TestDownloadAndProcessPdf:
    METADATA = {
        "source": "comelec.gov.ph",
        "title": "Candidate List",
        "url": "https://comelec.gov.ph/list.pdf",
        "date": "2025-01-01",
        "source_type": "election",
    }

    def test_downloads_pdf_and_returns_chunks(self, tmp_path):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF fake content"
        mock_response.raise_for_status = MagicMock()

        fake_chunks = [{"text": "chunk", "chunk_index": 0, "chunk_total": 1}]

        # requests/time are lazy-imported inside the function — patch at source
        with patch("requests.get", return_value=mock_response), \
             patch("data_ingestion.processors.pdf_processor.process_pdf_document", return_value=fake_chunks), \
             patch("time.sleep"):
            result = download_and_process_pdf(
                "https://comelec.gov.ph/list.pdf", tmp_path, self.METADATA
            )

        assert result == fake_chunks

    def test_skips_download_if_file_already_exists(self, tmp_path):
        existing = tmp_path / "list.pdf"
        existing.write_bytes(b"%PDF existing")

        fake_chunks = [{"text": "chunk", "chunk_index": 0, "chunk_total": 1}]

        with patch("requests.get") as mock_get, \
             patch("data_ingestion.processors.pdf_processor.process_pdf_document", return_value=fake_chunks):
            download_and_process_pdf(
                "https://comelec.gov.ph/list.pdf", tmp_path, self.METADATA
            )

        mock_get.assert_not_called()

    def test_returns_empty_list_on_download_failure(self, tmp_path):
        import requests
        with patch("requests.get", side_effect=requests.RequestException("timeout")):
            result = download_and_process_pdf(
                "https://comelec.gov.ph/list.pdf", tmp_path, self.METADATA
            )

        assert result == []

    def test_creates_dest_dir_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b"
        mock_response = MagicMock()
        mock_response.content = b"%PDF"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response), \
             patch("data_ingestion.processors.pdf_processor.process_pdf_document", return_value=[]), \
             patch("time.sleep"):
            download_and_process_pdf(
                "https://comelec.gov.ph/list.pdf", nested, self.METADATA
            )

        assert nested.exists()

    def test_pdf_filename_derived_from_url(self, tmp_path):
        mock_response = MagicMock()
        mock_response.content = b"%PDF"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response), \
             patch("data_ingestion.processors.pdf_processor.process_pdf_document", return_value=[]), \
             patch("time.sleep"):
            download_and_process_pdf(
                "https://comelec.gov.ph/candidates_2025.pdf", tmp_path, self.METADATA
            )

        assert (tmp_path / "candidates_2025.pdf").exists()
