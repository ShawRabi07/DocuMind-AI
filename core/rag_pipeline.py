"""
DocuMind AI – Unified RAG Pipeline Orchestrator
================================================
Central architectural spine connecting:
PDF Extraction -> Text Splitting -> Vector Embeddings -> ChromaDB ->
Semantic Retrieval -> Prompt Grounding -> Ollama Generation -> Structured Output.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.embeddings import EmbeddingManager
from core.llm import LLMResponse, MockOllamaClient, OllamaClient, OllamaHealthStatus
from core.pdf_processor import PDFProcessor, ProcessedDocument
from core.retriever import RetrievalResult, RetrievedChunk, SemanticRetriever
from core.text_splitter import DocumentChunk, DocumentChunker
from core.vector_store import BaseVectorStore, get_vector_store
from utils.metrics import metrics_tracker

logger = logging.getLogger(__name__)


@dataclass
class CitationInfo:
    """Explicit, verifiable source citation."""

    source: str
    page_number: int
    chunk_index: int
    chunk_id: str
    similarity_score: float
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "similarity_score": round(self.similarity_score, 4),
            "excerpt": self.excerpt,
        }


@dataclass
class RAGResponse:
    """Standardized response object returned for every user query."""

    query: str
    answer: str
    citations: List[CitationInfo] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    model_used: str = ""
    has_sufficient_context: bool = True
    is_mock: bool = False
    tokens_used: int = 0
    error: Optional[str] = None


class RAGPipeline:
    """
    End-to-End Orchestrator for Multi-Document Question-Answering.

    Ensures that every answer is strictly grounded in retrieved evidence,
    unambiguously attributed with citations, and timed with real latency tracking.
    """

    def __init__(
        self,
        pdf_processor: Optional[PDFProcessor] = None,
        chunker: Optional[DocumentChunker] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[BaseVectorStore] = None,
        retriever: Optional[SemanticRetriever] = None,
        llm_client: Optional[OllamaClient] = None,
        use_mock_llm: bool = False,
    ) -> None:
        self.pdf_processor = pdf_processor or PDFProcessor(
            max_file_size_mb=settings.max_upload_size_mb
        )
        self.chunker = chunker or DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding_manager = embedding_manager or EmbeddingManager.get_instance(
            model_name=settings.embedding_model_name,
            device=settings.embedding_device,
        )
        self.vector_store = vector_store or get_vector_store(
            backend=settings.vector_store_backend,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self.retriever = retriever or SemanticRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            default_top_k=settings.top_k,
        )

        if use_mock_llm:
            self.llm_client = MockOllamaClient()
        else:
            self.llm_client = llm_client or OllamaClient()

        logger.info(
            "RAGPipeline initialized with backend='%s', collection='%s', embedding='%s'.",
            settings.vector_store_backend,
            settings.chroma_collection_name,
            settings.embedding_model_name,
        )

    # -------------------------------------------------------------------------
    # Ingestion Workflow
    # -------------------------------------------------------------------------
    def ingest_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingests a single PDF file through the entire pipeline:
        Validate -> Extract -> Chunk -> Embed -> Index in ChromaDB.

        Returns:
            Dictionary summary containing status, document_id, chunks_added, and warnings.
        """
        logger.info("Starting ingestion for: %s", file_path.name)

        # Step 1: Extract document and metadata
        processed_doc: ProcessedDocument = self.pdf_processor.extract_document(file_path)

        if not processed_doc.has_text:
            return {
                "file_name": file_path.name,
                "document_id": processed_doc.document_id,
                "status": "failed",
                "chunks_added": 0,
                "error": processed_doc.error_message or "No readable text found in PDF.",
            }

        # Step 2: Check for duplicate document
        if self.vector_store.document_exists(processed_doc.document_id):
            return {
                "file_name": file_path.name,
                "document_id": processed_doc.document_id,
                "status": "skipped",
                "chunks_added": 0,
                "message": "Document is already indexed. Duplicate addition skipped.",
            }

        # Step 3: Split document into chunks
        chunks: List[DocumentChunk] = self.chunker.split_document(processed_doc)
        if not chunks:
            return {
                "file_name": file_path.name,
                "document_id": processed_doc.document_id,
                "status": "failed",
                "chunks_added": 0,
                "error": "Document splitting produced zero chunks.",
            }

        # Step 4: Batch embed chunk texts
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.embed_texts(chunk_texts)

        # Step 5: Persist vectors, text, and metadata in ChromaDB
        added_count = self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        return {
            "file_name": file_path.name,
            "document_id": processed_doc.document_id,
            "status": "success",
            "chunks_added": added_count,
            "total_pages": processed_doc.total_pages,
            "file_size_bytes": processed_doc.file_size_bytes,
        }

    def ingest_documents(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Ingests multiple PDF files sequentially with progress reporting."""
        results = []
        for p in file_paths:
            res = self.ingest_document(p)
            results.append(res)
        return results

    # -------------------------------------------------------------------------
    # Retrieval and Question Answering
    # -------------------------------------------------------------------------
    def ask_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """
        Executes an end-to-end question answering turn:
        Question -> Semantic Search -> Grounding Prompt -> LLM -> Citations.
        """
        total_start = time.perf_counter()
        cleaned_query = query.strip()

        if not cleaned_query:
            return RAGResponse(
                query=query,
                answer="Please enter a non-empty question.",
                has_sufficient_context=False,
            )

        # Step 1: Semantic Retrieval
        retrieval_res: RetrievalResult = self.retriever.retrieve(
            query=cleaned_query,
            top_k=top_k,
            document_id=document_id,
        )

        retrieval_ms = retrieval_res.retrieval_latency_ms

        # Step 2: Handle Empty Context (Early Return to avoid LLM hallucination)
        if not retrieval_res.has_context:
            total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            ans = "The uploaded documents do not contain sufficient information to answer this question."

            # Record real metrics
            metrics_tracker.record_query(
                query=cleaned_query,
                retrieval_ms=retrieval_ms,
                llm_ms=0.0,
                chunks_count=0,
                sources_count=0,
                is_mock=isinstance(self.llm_client, MockOllamaClient),
            )

            return RAGResponse(
                query=cleaned_query,
                answer=ans,
                retrieved_chunks=[],
                citations=[],
                retrieval_latency_ms=retrieval_ms,
                llm_latency_ms=0.0,
                total_latency_ms=round(total_elapsed_ms, 2),
                model_used=self.llm_client.model,
                has_sufficient_context=False,
            )

        # Step 3: Construct Grounding Prompt
        context_str = retrieval_res.formatted_context()
        prompt = self.llm_client.build_prompt(
            query=cleaned_query,
            context=context_str,
            conversation_history=conversation_history,
        )

        # Step 4: Generate Grounded Answer via LLM
        llm_resp: LLMResponse = self.llm_client.generate(prompt)

        llm_ms = llm_resp.generation_time_ms
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        # Step 5: Format Citations
        citations: List[CitationInfo] = []
        for chunk in retrieval_res.chunks:
            # Create a short 100-character excerpt
            snippet = chunk.text[:120].replace("\n", " ").strip() + "..."
            citations.append(
                CitationInfo(
                    source=chunk.source,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    chunk_id=chunk.chunk_id,
                    similarity_score=chunk.similarity_score,
                    excerpt=snippet,
                )
            )

        # Step 6: Record Real System Metrics
        metrics_tracker.record_query(
            query=cleaned_query,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            chunks_count=len(retrieval_res.chunks),
            sources_count=len(retrieval_res.sources_cited),
            is_mock=llm_resp.is_mock,
            tokens_used=llm_resp.total_tokens,
        )

        # Check if answer encountered an Ollama connection error
        final_answer = llm_resp.text
        if llm_resp.error:
            final_answer = (
                f"**Ollama Communication Notice:**\n{llm_resp.error}\n\n"
                f"*Retrieved {len(retrieval_res.chunks)} relevant document excerpt(s) successfully.*"
            )

        return RAGResponse(
            query=cleaned_query,
            answer=final_answer,
            citations=citations,
            retrieved_chunks=retrieval_res.chunks,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=round(total_elapsed_ms, 2),
            model_used=llm_resp.model,
            has_sufficient_context=True,
            is_mock=llm_resp.is_mock,
            tokens_used=llm_resp.total_tokens,
            error=llm_resp.error,
        )

    # -------------------------------------------------------------------------
    # Management and Diagnostics
    # -------------------------------------------------------------------------
    def delete_document(self, document_id: str) -> int:
        """Deletes all indexed vectors and metadata for a document."""
        return self.vector_store.delete_document(document_id)

    def get_document_statistics(self) -> Dict[str, Any]:
        """Returns overview statistics of indexed documents and chunks."""
        docs = self.vector_store.list_indexed_documents()
        total_chunks = self.vector_store.count_chunks()
        total_docs = len(docs)
        avg_chunks = round(total_chunks / total_docs, 1) if total_docs > 0 else 0.0

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "average_chunks_per_doc": avg_chunks,
            "documents": docs,
            "backend": settings.vector_store_backend,
            "embedding_model": settings.embedding_model_name,
            "llm_model": settings.ollama_model,
        }

    def check_system_health(self) -> Dict[str, Any]:
        """Probes status of vector store, embedding engine, and Ollama service."""
        ollama_status: OllamaHealthStatus = self.llm_client.check_health()
        chroma_count = self.vector_store.count_chunks()

        return {
            "ollama_running": ollama_status.is_running,
            "ollama_version": ollama_status.version,
            "ollama_models": ollama_status.available_models,
            "target_model_available": ollama_status.has_target_model,
            "target_model_name": ollama_status.target_model_name,
            "ollama_error": ollama_status.error_message,
            "vector_store_healthy": True,
            "indexed_chunks_count": chroma_count,
            "embedding_dimension": self.embedding_manager.dimension,
        }

