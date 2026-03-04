"""
Tests for data_ingestion/scrapers/comelec.py

HTTP calls and PDF processing are mocked — no real network requests made.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from data_ingestion.scrapers.comelec import (
    _find_pdf_links,
    _pdf_title_from_url,
    _parse_resolution_date,
    scrape_candidate_pdfs,
    scrape_resolutions,
    scrape_all_comelec,
)


# ---------------------------------------------------------------------------
# Sample HTML / RSS fixtures
# ---------------------------------------------------------------------------

HTML_WITH_PDFS = """
<html><body>
  <a href="/files/CandidateList_National_2025.pdf">National Candidates</a>
  <a href="https://comelec.gov.ph/files/BallotFace_2025.pdf">Ballot Face</a>
  <a href="/about">About COMELEC</a>
</body></html>
"""

HTML_NO_PDFS = """
<html><body>
  <p>This page requires JavaScript to view candidates.</p>
  <a href="/about">About</a>
</body></html>
"""

RESOLUTION_INDEX_HTML = """
<html><body>
<table>
  <tr>
    <td>January 20, 2025</td>
    <td><a href="comres_11100_2025.pdf">Resolution No. 11100</a></td>
  </tr>
  <tr>
    <td>February 5, 2025</td>
    <td><a href="comres_11109_2025.pdf">Resolution No. 11109</a></td>
  </tr>
  <tr>
    <td></td>
    <td><span>No link here</span></td>
  </tr>
</table>
</body></html>
"""


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _find_pdf_links
# ---------------------------------------------------------------------------

class TestFindPdfLinks:
    def test_finds_absolute_pdf_link(self):
        html = '<a href="https://comelec.gov.ph/file.pdf">Download</a>'
        links = _find_pdf_links(html, "https://comelec.gov.ph/page")
        assert "https://comelec.gov.ph/file.pdf" in links

    def test_resolves_relative_pdf_link(self):
        html = '<a href="/files/candidates.pdf">Candidates</a>'
        links = _find_pdf_links(html, "https://comelec.gov.ph/page")
        assert "https://comelec.gov.ph/files/candidates.pdf" in links

    def test_ignores_non_pdf_links(self):
        html = '<a href="/about">About</a><a href="page.html">Page</a>'
        links = _find_pdf_links(html, "https://comelec.gov.ph/")
        assert links == []

    def test_deduplicates_repeated_links(self):
        html = '<a href="/f.pdf">1</a><a href="/f.pdf">2</a>'
        links = _find_pdf_links(html, "https://comelec.gov.ph/")
        assert len(links) == 1

    def test_returns_multiple_pdfs(self):
        links = _find_pdf_links(HTML_WITH_PDFS, "https://comelec.gov.ph/page")
        assert len(links) == 2

    def test_empty_html_returns_empty_list(self):
        assert _find_pdf_links("", "https://comelec.gov.ph/") == []


# ---------------------------------------------------------------------------
# _pdf_title_from_url
# ---------------------------------------------------------------------------

class TestPdfTitleFromUrl:
    def test_formats_resolution_url(self):
        title = _pdf_title_from_url("https://lawphil.net/comres_11102_2025.pdf")
        assert "11102" in title
        assert "2025" in title

    def test_formats_minute_resolution_url(self):
        title = _pdf_title_from_url("https://lawphil.net/minres_0544_2025.pdf")
        assert "Minute Resolution" in title
        assert "0544" in title

    def test_generic_fallback_for_unknown_pattern(self):
        title = _pdf_title_from_url("https://example.com/candidate_list_final.pdf")
        assert title  # non-empty
        assert "pdf" not in title.lower()  # extension stripped


# ---------------------------------------------------------------------------
# _parse_resolution_date
# ---------------------------------------------------------------------------

class TestParseResolutionDate:
    def test_parses_long_month_format(self):
        assert _parse_resolution_date("January 20, 2025", 2025) == "2025-01-20"

    def test_parses_day_month_year(self):
        assert _parse_resolution_date("20 January 2025", 2025) == "2025-01-20"

    def test_parses_iso_format(self):
        assert _parse_resolution_date("2025-01-20", 2025) == "2025-01-20"

    def test_returns_jan_1_fallback_for_unrecognized(self):
        result = _parse_resolution_date("not a date", 2025)
        assert result == "2025-01-01"


# ---------------------------------------------------------------------------
# scrape_candidate_pdfs
# ---------------------------------------------------------------------------

class TestScrapeCandidatePdfs:
    def test_returns_empty_when_all_pages_fail(self):
        with patch("data_ingestion.scrapers.comelec._get", return_value=None):
            docs = scrape_candidate_pdfs()
        assert docs == []

    def test_returns_empty_when_no_pdf_links_found(self):
        resp = _mock_response(HTML_NO_PDFS)
        with patch("data_ingestion.scrapers.comelec._get", return_value=resp):
            docs = scrape_candidate_pdfs()
        assert docs == []

    def test_downloads_and_processes_found_pdfs(self):
        resp = _mock_response(HTML_WITH_PDFS)
        mock_chunks = [{"text": "Candidate data.", "source": "comelec.gov.ph",
                        "source_type": "election", "date": "2025-01-01",
                        "politician": "", "title": "Candidate List", "url": "https://x.com/f.pdf",
                        "chunk_index": 0, "chunk_total": 1}]
        with patch("data_ingestion.scrapers.comelec._get", return_value=resp), \
             patch("data_ingestion.scrapers.comelec.download_and_process_pdf",
                   return_value=mock_chunks) as mock_dl:
            docs = scrape_candidate_pdfs(election_year=2025)

        assert mock_dl.called
        assert len(docs) == mock_dl.call_count  # one chunk per PDF call

    def test_metadata_has_correct_source_type(self):
        resp = _mock_response(HTML_WITH_PDFS)
        mock_chunk = {"text": "chunk", "source": "comelec.gov.ph",
                      "source_type": "election", "date": "2025-01-01",
                      "politician": "", "title": "t", "url": "u",
                      "chunk_index": 0, "chunk_total": 1}
        with patch("data_ingestion.scrapers.comelec._get", return_value=resp), \
             patch("data_ingestion.scrapers.comelec.download_and_process_pdf",
                   return_value=[mock_chunk]) as mock_dl:
            scrape_candidate_pdfs(election_year=2025)

        metadata_arg = mock_dl.call_args_list[0][0][2]  # third positional arg
        assert metadata_arg["source"] == "comelec.gov.ph"
        assert metadata_arg["source_type"] == "election"
        assert metadata_arg["date"] == "2025-01-01"


# ---------------------------------------------------------------------------
# scrape_resolutions
# ---------------------------------------------------------------------------

class TestScrapeResolutions:
    def test_returns_empty_for_unconfigured_year(self):
        docs = scrape_resolutions(year=1999)
        assert docs == []

    def test_returns_empty_when_index_fetch_fails(self):
        with patch("data_ingestion.scrapers.comelec._get", return_value=None):
            docs = scrape_resolutions(year=2025)
        assert docs == []

    def test_processes_pdf_links_from_index(self):
        resp = _mock_response(RESOLUTION_INDEX_HTML)
        mock_chunk = {"text": "resolution text", "source": "comelec.gov.ph",
                      "source_type": "resolution", "date": "2025-01-20",
                      "politician": "", "title": "Resolution No. 11100",
                      "url": "https://lawphil.net/comres_11100_2025.pdf",
                      "chunk_index": 0, "chunk_total": 1}
        with patch("data_ingestion.scrapers.comelec._get", return_value=resp), \
             patch("data_ingestion.scrapers.comelec.download_and_process_pdf",
                   return_value=[mock_chunk]) as mock_dl:
            docs = scrape_resolutions(year=2025, max_resolutions=10)

        assert mock_dl.called
        assert len(docs) > 0

    def test_respects_max_resolutions_limit(self):
        resp = _mock_response(RESOLUTION_INDEX_HTML)
        mock_chunk = {"text": "t", "source": "comelec.gov.ph", "source_type": "resolution",
                      "date": "2025-01-01", "politician": "", "title": "t",
                      "url": "u", "chunk_index": 0, "chunk_total": 1}
        with patch("data_ingestion.scrapers.comelec._get", return_value=resp), \
             patch("data_ingestion.scrapers.comelec.download_and_process_pdf",
                   return_value=[mock_chunk]):
            docs = scrape_resolutions(year=2025, max_resolutions=1)

        # Only 1 resolution should have been processed
        assert len(docs) == 1

    def test_resolution_metadata_has_correct_source_type(self):
        resp = _mock_response(RESOLUTION_INDEX_HTML)
        captured_metadata = {}

        def capture(url, dest_dir, metadata):
            captured_metadata.update(metadata)
            return [{"text": "t", **metadata, "chunk_index": 0, "chunk_total": 1}]

        with patch("data_ingestion.scrapers.comelec._get", return_value=resp), \
             patch("data_ingestion.scrapers.comelec.download_and_process_pdf",
                   side_effect=capture):
            scrape_resolutions(year=2025, max_resolutions=1)

        assert captured_metadata.get("source") == "comelec.gov.ph"
        assert captured_metadata.get("source_type") == "resolution"

    def test_skips_rows_without_links(self):
        html = """<html><body><table>
          <tr><td>Jan 1, 2025</td><td><span>no link here</span></td></tr>
        </table></body></html>"""
        resp = _mock_response(html)
        with patch("data_ingestion.scrapers.comelec._get", return_value=resp), \
             patch("data_ingestion.scrapers.comelec.download_and_process_pdf",
                   return_value=[]) as mock_dl:
            docs = scrape_resolutions(year=2025, max_resolutions=5)
        mock_dl.assert_not_called()
        assert docs == []


# ---------------------------------------------------------------------------
# scrape_all_comelec
# ---------------------------------------------------------------------------

class TestScrapeAllComelec:
    def test_combines_candidates_and_resolutions(self):
        candidate_chunk = {"text": "candidate", "source": "comelec.gov.ph",
                           "source_type": "election"}
        resolution_chunk = {"text": "resolution", "source": "comelec.gov.ph",
                            "source_type": "resolution"}
        with patch("data_ingestion.scrapers.comelec.scrape_candidate_pdfs",
                   return_value=[candidate_chunk]) as mock_cand, \
             patch("data_ingestion.scrapers.comelec.scrape_resolutions",
                   return_value=[resolution_chunk]) as mock_res:
            docs = scrape_all_comelec(election_year=2025, max_resolutions=10)

        mock_cand.assert_called_once_with(2025)
        mock_res.assert_called_once_with(2025, max_resolutions=10)
        assert len(docs) == 2

    def test_returns_empty_when_both_sources_empty(self):
        with patch("data_ingestion.scrapers.comelec.scrape_candidate_pdfs", return_value=[]), \
             patch("data_ingestion.scrapers.comelec.scrape_resolutions", return_value=[]):
            docs = scrape_all_comelec()
        assert docs == []
