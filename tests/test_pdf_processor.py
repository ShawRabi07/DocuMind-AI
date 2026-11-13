"""
Unit Tests for PDF Processor Module
====================================
Tests PDF extraction, metadata preservation, empty PDF handling, and corrupted file handling.
"""

import tempfile
from pathlib import Path
import pytest
import pymupdf

from core.pdf_processor import PDFProcessor, ProcessedDocument


@pytest.fixture
def pdf_processor():
    return PDFProcessor(max_file_size_mb=10)


@pytest.fixture
def sample_pdf(tmp_path):
    """Generates a valid 2-page test PDF file."""
    pdf_path = tmp_path / "valid_document.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), "This is page 1 of DocuMind AI. It introduces RAG concepts.")

    p2 = doc.new_page()
    p2.insert_text((72, 72), "This is page 2 with detailed architectural explanations.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_valid_pdf_extraction(pdf_processor, sample_pdf):
    """Test standard extraction from a multi-page PDF."""
    result: ProcessedDocument = pdf_processor.extract_document(sample_pdf)

    assert result.has_text is True
    assert result.total_pages == 2
    assert result.file_name == "valid_document.pdf"
    assert result.document_id.startswith("doc_")
    assert len(result.pages) == 2

    # Check page 1
    assert result.pages[0].page_number == 1
    assert "DocuMind AI" in result.pages[0].text
    assert result.pages[0].source == "valid_document.pdf"

    # Check page 2
    assert result.pages[1].page_number == 2
    assert "architectural" in result.pages[1].text


def test_empty_page_pdf(pdf_processor, tmp_path):
    """Test PDF containing blank pages without text."""
    blank_pdf = tmp_path / "blank_document.pdf"
    doc = pymupdf.open()
    doc.new_page()  # Blank page
    doc.save(str(blank_pdf))
    doc.close()

    result = pdf_processor.extract_document(blank_pdf)
    assert result.total_pages == 1
    assert result.has_text is False
    assert result.pages[0].is_empty is True
    assert "blank" in result.pages[0].warning.lower()


def test_corrupted_pdf_handling(pdf_processor, tmp_path):
    """Test handling of invalid or corrupted PDF bytes."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4 Not a real PDF header or content at all")

    with pytest.raises(ValueError, match="Corrupted or invalid PDF"):
        pdf_processor.extract_document(corrupt_file)


def test_zero_byte_file(pdf_processor, tmp_path):
    """Test rejecting 0-byte file."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="File is empty"):
        pdf_processor.extract_document(empty_file)


def test_invalid_extension(pdf_processor, tmp_path):
    """Test rejecting non-PDF extensions."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("Hello")

    with pytest.raises(ValueError, match="Invalid file extension"):
        pdf_processor.extract_document(txt_file)


def test_sha256_reproducibility(pdf_processor, sample_pdf):
    """Test that document_id is deterministic and based on SHA-256."""
    res1 = pdf_processor.extract_document(sample_pdf)
    res2 = pdf_processor.extract_document(sample_pdf)
    assert res1.document_id == res2.document_id
    assert res1.file_hash == res2.file_hash

