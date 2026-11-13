"""
DocuMind AI – Text Chunking Module
===================================
Splits document pages into semantically cohesive chunks using LangChain's
RecursiveCharacterTextSplitter while maintaining rigorous page-level source traceability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.pdf_processor import ExtractedPage, ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents an atomic chunk of text ready for vector embedding and retrieval."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_langchain_document(self) -> Document:
        """Converts into a standard LangChain Document object."""
        meta = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            **self.metadata,
        }
        return Document(page_content=self.text, metadata=meta)


class DocumentChunker:
    """
    Configurable text splitting engine preserving granular page citations.

    Why chunk page-by-page?
    Splitting text on a page-by-page boundary guarantees that every chunk has
    an exact, unambiguous page citation. Crossing page boundaries inside a chunk
    leads to hallucinated or ambiguous page numbers during citation display.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default multi-level hierarchical separators
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_page(self, page: ExtractedPage, global_chunk_start_idx: int = 0) -> List[DocumentChunk]:
        """
        Splits a single ExtractedPage into chunks.

        Args:
            page: The extracted page object.
            global_chunk_start_idx: Starting counter for document-level chunk ordering.

        Returns:
            List of DocumentChunk instances for this page.
        """
        text = page.text.strip()
        if not text:
            return []

        raw_chunks = self._splitter.split_text(text)
        chunks: List[DocumentChunk] = []

        for local_idx, chunk_text in enumerate(raw_chunks):
            cleaned = chunk_text.strip()
            if not cleaned:
                continue

            chunk_idx = global_chunk_start_idx + local_idx
            # Deterministic, unique ID: doc_<hash>_p<page>_c<idx>
            chunk_id = f"{page.document_id}_p{page.page_number}_c{chunk_idx:04d}"

            # Fast estimation: roughly 4 characters per token in English
            token_estimate = max(1, len(cleaned) // 4)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    text=cleaned,
                    char_count=len(cleaned),
                    token_estimate=token_estimate,
                    metadata={
                        "total_pages": page.total_pages,
                        "chunk_size_setting": self.chunk_size,
                        "chunk_overlap_setting": self.chunk_overlap,
                    },
                )
            )

        return chunks

    def split_document(self, processed_doc: ProcessedDocument) -> List[DocumentChunk]:
        """
        Splits an entire ProcessedDocument across all its pages.

        Args:
            processed_doc: The processed document object.

        Returns:
            List of DocumentChunk instances across all pages.
        """
        all_chunks: List[DocumentChunk] = []
        global_counter = 0

        for page in processed_doc.pages:
            page_chunks = self.split_page(page, global_chunk_start_idx=global_counter)
            all_chunks.extend(page_chunks)
            global_counter += len(page_chunks)

        logger.info(
            "Document '%s' (%s) split into %d chunks from %d pages.",
            processed_doc.file_name,
            processed_doc.document_id,
            len(all_chunks),
            processed_doc.total_pages,
        )
        return all_chunks

