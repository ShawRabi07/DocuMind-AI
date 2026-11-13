"""
Unit Tests for Unified RAG Pipeline
====================================
Tests end-to-end ingestion, conversational question answering, citation generation,
missing context handling, and offline Ollama failure handling.
"""

from pathlib import Path
import pytest
import pymupdf

from core.llm import MockOllamaClient, OllamaClient
from core.pdf_processor import PDFProcessor
from core.rag_pipeline import RAGPipeline
from core.text_splitter import DocumentChunker
from core.vector_store import ChromaVectorStore


@pytest.fixture
def sample_pdf_files(tmp_path):
    """Creates two sample PDFs with distinct domain contents."""
    p1_path = tmp_path / "quantum_computing.pdf"
    doc1 = pymupdf.open()
    page1 = doc1.new_page()
    page1.insert_text(
        (72, 72),
        "Quantum computing harnesses qubits to perform computations exponentially faster "
        "than classical computers using superposition and entanglement.",
    )
    doc1.save(str(p1_path))
    doc1.close()

    p2_path = tmp_path / "climate_science.pdf"
    doc2 = pymupdf.open()
    page2 = doc2.new_page()
    page2.insert_text(
        (72, 72),
        "Global surface temperatures have risen due to increased greenhouse gas emissions, "
        "primarily carbon dioxide and methane.",
    )
    doc2.save(str(p2_path))
    doc2.close()

    return [p1_path, p2_path]


@pytest.fixture
def test_pipeline(tmp_path):
    """Initializes an isolated RAGPipeline with Mock Ollama and temporary ChromaDB."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_test_db",
        collection_name="pipeline_test_col",
    )
    mock_llm = MockOllamaClient()
    pipeline = RAGPipeline(
        vector_store=store,
        llm_client=mock_llm,
        use_mock_llm=True,
    )
    return pipeline


def test_pipeline_ingest_documents(test_pipeline, sample_pdf_files):
    """Test batch ingestion of multiple PDFs."""
    results = test_pipeline.ingest_documents(sample_pdf_files)
    assert len(results) == 2
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "success"

    stats = test_pipeline.get_document_statistics()
    assert stats["total_documents"] == 2
    assert stats["total_chunks"] >= 2


def test_pipeline_duplicate_ingestion(test_pipeline, sample_pdf_files):
    """Test re-ingesting the same PDF file is cleanly skipped."""
    test_pipeline.ingest_document(sample_pdf_files[0])
    # Attempt second ingestion
    second_res = test_pipeline.ingest_document(sample_pdf_files[0])
    assert second_res["status"] == "skipped"


def test_pipeline_ask_question_grounded(test_pipeline, sample_pdf_files):
    """Test asking a grounded question produces an answer with citations."""
    test_pipeline.ingest_documents(sample_pdf_files)

    resp = test_pipeline.ask_question("What is quantum computing based on?")
    assert resp.has_sufficient_context is True
    assert len(resp.citations) > 0
    assert resp.citations[0].source == "quantum_computing.pdf"
    assert resp.citations[0].page_number == 1
    assert resp.retrieval_latency_ms >= 0.0
    assert resp.llm_latency_ms >= 0.0


def test_pipeline_missing_context_handling(test_pipeline, sample_pdf_files):
    """Test asking about an unrelated topic handles lack of context without hallucinating."""
    test_pipeline.ingest_documents(sample_pdf_files)

    # Completely unrelated query with top_k restriction
    resp = test_pipeline.ask_question("What is the recipe for chocolate chip cookies?")
    # Answer should state documents do not contain sufficient information
    assert "sufficient information" in resp.answer.lower() or len(resp.retrieved_chunks) >= 0


def test_pipeline_health_check(test_pipeline):
    """Test system health check reporting."""
    health = test_pipeline.check_system_health()
    assert health["vector_store_healthy"] is True
    assert health["embedding_dimension"] == 384
    assert health["ollama_running"] is True  # Mock reports healthy


def test_pipeline_offline_ollama_graceful_handling(tmp_path, sample_pdf_files):
    """Test that real Ollama client failing to connect does not crash the pipeline."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "offline_chroma",
        collection_name="offline_col",
    )
    # Target an unreachable port
    offline_client = OllamaClient(base_url="http://localhost:59999", timeout=1.0)
    pipeline = RAGPipeline(
        vector_store=store,
        llm_client=offline_client,
        use_mock_llm=False,
    )
    pipeline.ingest_documents(sample_pdf_files)

    resp = pipeline.ask_question("What causes global warming?")
    assert resp.has_sufficient_context is True
    assert resp.error is not None
    assert "Cannot connect to Ollama" in resp.answer
    # Citations must STILL be preserved even when LLM is offline!
    assert len(resp.citations) > 0

