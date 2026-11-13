"""
DocuMind AI – PDF Processing Module
====================================
Extracts text and page-level metadata from PDF files using PyMuPDF (pymupdf).
Handles corrupt PDFs, blank pages, scanned documents (OCR requirement flags),
and computes SHA-256 hashes to prevent duplicate ingestion.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents text and metadata extracted from an individual PDF page."""

    document_id: str
    source: str
    page_number: int  # 1-based index
    total_pages: int
    text: str
    char_count: int
    is_empty: bool = False
    is_scanned_likely: bool = False
    warning: Optional[str] = None


@dataclass
class ProcessedDocument:
    """Represents a fully processed PDF document with page data and file metadata."""

    document_id: str
    file_name: str
    file_path: Path
    file_hash: str
    file_size_bytes: int
    total_pages: int
    pages: List[ExtractedPage] = field(default_factory=list)
    has_text: bool = True
    is_scanned_only: bool = False
    error_message: Optional[str] = None

    @property
    def total_character_count(self) -> int:
        return sum(page.char_count for page in self.pages)


class PDFProcessor:
    """High-performance, robust PDF extraction engine powered by PyMuPDF."""

    def __init__(self, max_file_size_mb: int = 25) -> None:
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Calculates the SHA-256 hash of a file to reliably identify duplicate uploads."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_file(self, file_path: Path) -> None:
        """Validates that a file exists, has a .pdf extension, and is within size limits."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid file extension '{file_path.suffix}'. Only .pdf is supported.")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty (0 bytes): {file_path.name}")

        if file_size > self.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.0f} MB."
            )

    def extract_document(self, file_path: Path) -> ProcessedDocument:
        """
        Extracts all pages, text, and metadata from a PDF file.

        Args:
            file_path: Absolute or relative Path to the PDF file.

        Returns:
            ProcessedDocument containing per-page ExtractedPage instances.
        """
        self.validate_file(file_path)
        file_hash = self.compute_sha256(file_path)
        # Use first 16 chars of hash as clean, reproducible document ID
        document_id = f"doc_{file_hash[:16]}"
        file_size = file_path.stat().st_size

        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            logger.error("Failed to open PDF %s: %s", file_path.name, str(e))
            raise ValueError(f"Corrupted or invalid PDF file '{file_path.name}': {e}") from e

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return ProcessedDocument(
                document_id=document_id,
                file_name=file_path.name,
                file_path=file_path,
                file_hash=file_hash,
                file_size_bytes=file_size,
                total_pages=0,
                has_text=False,
                error_message="PDF contains 0 pages.",
            )

        extracted_pages: List[ExtractedPage] = []
        pages_with_text = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-based indexing for citations
            try:
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text") or ""
                # Clean basic whitespace
                cleaned_text = raw_text.strip()
                char_count = len(cleaned_text)

                # Check if page has images but no text (indicative of scanned pages)
                images = page.get_images()
                is_scanned_likely = bool(images and char_count == 0)

                warning = None
                if char_count == 0:
                    if is_scanned_likely:
                        warning = (
                            f"Page {page_num} contains images but no extractable text. "
                            f"OCR is required to index scanned content."
                        )
                    else:
                        warning = f"Page {page_num} is blank."
                else:
                    pages_with_text += 1

                extracted_pages.append(
                    ExtractedPage(
                        document_id=document_id,
                        source=file_path.name,
                        page_number=page_num,
                        total_pages=total_pages,
                        text=cleaned_text,
                        char_count=char_count,
                        is_empty=(char_count == 0),
                        is_scanned_likely=is_scanned_likely,
                        warning=warning,
                    )
                )
            except Exception as e:
                logger.warning("Error reading page %d in %s: %s", page_num, file_path.name, str(e))
                extracted_pages.append(
                    ExtractedPage(
                        document_id=document_id,
                        source=file_path.name,
                        page_number=page_num,
                        total_pages=total_pages,
                        text="",
                        char_count=0,
                        is_empty=True,
                        warning=f"Page read error: {e}",
                    )
                )

        doc.close()

        has_text = pages_with_text > 0
        is_scanned_only = (not has_text) and any(p.is_scanned_likely for p in extracted_pages)

        error_msg = None
        if not has_text:
            if is_scanned_only:
                error_msg = (
                    "Document appears to be scanned or image-based with no extractable text. "
                    "OCR preprocessing is required."
                )
            else:
                error_msg = "Document contains no readable text across any of its pages."

        return ProcessedDocument(
            document_id=document_id,
            file_name=file_path.name,
            file_path=file_path,
            file_hash=file_hash,
            file_size_bytes=file_size,
            total_pages=total_pages,
            pages=extracted_pages,
            has_text=has_text,
            is_scanned_only=is_scanned_only,
            error_message=error_msg,
        )

