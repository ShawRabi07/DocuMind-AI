"""
DocuMind AI – Semantic Retriever Module
========================================
Coordinates query embedding generation, similarity search against vector store,
distance score normalization, and source attribution extraction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.embeddings import EmbeddingManager
from core.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a single retrieved context chunk with full source traceability."""

    chunk_id: str
    text: str
    document_id: str
    source: str
    page_number: int
    chunk_index: int
    char_count: int
    distance: float
    similarity_score: float

    @property
    def citation_str(self) -> str:
        """Formatted human-readable citation string."""
        return f"[{self.source} - Page {self.page_number} (Chunk #{self.chunk_index})]"


@dataclass
class RetrievalResult:
    """Encapsulates the complete result of a semantic search operation."""

    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    total_retrieved: int = 0
    retrieval_latency_ms: float = 0.0
    sources_cited: List[str] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return len(self.chunks) > 0

    def formatted_context(self) -> str:
        """
        Formats retrieved chunks into an injection-safe context block for the LLM prompt.
        Each chunk is clearly tagged with its source and page number.
        """
        if not self.chunks:
            return "No relevant document context found."

        context_blocks = []
        for i, chunk in enumerate(self.chunks, start=1):
            block = (
                f"--- DOCUMENT EXCERPT {i} ---\n"
                f"Source: {chunk.source}\n"
                f"Page: {chunk.page_number}\n"
                f"Content:\n{chunk.text}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)


class SemanticRetriever:
    """
    Retrieves the most semantically relevant document excerpts for a given query.

    Understanding Distance vs. Similarity:
    - ChromaDB with HNSW cosine space returns Cosine Distance: d = 1 - cos(theta).
    - Range: 0 (identical orientation) to 2 (oppositely oriented).
    - A smaller distance value indicates higher semantic closeness.
    - We normalize this to an intuitive similarity score: sim = 1.0 - (d / 2.0).
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_manager: Optional[EmbeddingManager] = None,
        default_top_k: int = 5,
        max_distance_threshold: float = 1.6,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager or EmbeddingManager.get_instance()
        self.default_top_k = default_top_k
        self.max_distance_threshold = max_distance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Executes semantic retrieval for a user question.

        Args:
            query: The user's question or search query.
            top_k: Optional override for the number of chunks to return.
            document_id: Optional filter to restrict retrieval to a single document.

        Returns:
            RetrievalResult containing ranked RetrievedChunk instances and latency metrics.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return RetrievalResult(query=query)

        k = top_k if (top_k is not None and top_k > 0) else self.default_top_k

        start_time = time.perf_counter()

        # Step 1: Generate dense vector embedding for query
        query_embedding = self.embedding_manager.embed_query(cleaned_query)

        # Step 2: Query vector store
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            document_id=document_id,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 3: Parse and filter results by distance threshold
        chunks: List[RetrievedChunk] = []
        sources_set: Set[str] = set()

        for item in raw_results:
            dist = item.get("distance", 1.0)
            # Filter out chunks that are practically orthogonal / irrelevant
            if dist > self.max_distance_threshold:
                continue

            chunk = RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                document_id=item["document_id"],
                source=item["source"],
                page_number=int(item["page_number"]),
                chunk_index=int(item["chunk_index"]),
                char_count=int(item["char_count"]),
                distance=dist,
                similarity_score=float(item.get("similarity_score", 0.0)),
            )
            chunks.append(chunk)
            sources_set.add(f"{chunk.source} (p. {chunk.page_number})")

        return RetrievalResult(
            query=cleaned_query,
            chunks=chunks,
            total_retrieved=len(chunks),
            retrieval_latency_ms=round(latency_ms, 2),
            sources_cited=sorted(list(sources_set)),
        )

