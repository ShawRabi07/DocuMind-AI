"""
Unit Tests for Vector Store and Semantic Retriever
===================================================
Tests ChromaDB persistence, vector indexing, top-K retrieval, distance scores,
duplicate rejection, and document deletion.
"""

from pathlib import Path
import pytest

from core.embeddings import EmbeddingManager
from core.retriever import SemanticRetriever
from core.text_splitter import DocumentChunk
from core.vector_store import ChromaVectorStore


@pytest.fixture
def temp_chroma_store(tmp_path):
    """Provides an isolated, temporary ChromaDB instance."""
    persist_dir = tmp_path / "test_chroma_db"
    store = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name="test_collection",
    )
    return store


@pytest.fixture(scope="module")
def embedding_mgr():
    return EmbeddingManager.get_instance()


@pytest.fixture
def populated_store(temp_chroma_store, embedding_mgr):
    """Populates ChromaDB with 3 distinct test chunks across 2 documents."""
    chunks = [
        DocumentChunk(
            chunk_id="doc_A_p1_c0001",
            document_id="doc_A",
            source="python_guide.pdf",
            page_number=1,
            chunk_index=1,
            text="Python is an interpreted, high-level, general-purpose programming language.",
            char_count=73,
            token_estimate=18,
        ),
        DocumentChunk(
            chunk_id="doc_A_p2_c0002",
            document_id="doc_A",
            source="python_guide.pdf",
            page_number=2,
            chunk_index=2,
            text="List comprehensions provide a concise way to create lists in Python.",
            char_count=68,
            token_estimate=17,
        ),
        DocumentChunk(
            chunk_id="doc_B_p1_c0001",
            document_id="doc_B",
            source="database_notes.pdf",
            page_number=1,
            chunk_index=1,
            text="Relational databases use SQL to query structured data organized in tables.",
            char_count=73,
            token_estimate=18,
        ),
    ]

    embeddings = embedding_mgr.embed_texts([c.text for c in chunks])
    temp_chroma_store.add_chunks(chunks, embeddings)
    return temp_chroma_store


def test_add_chunks_and_count(populated_store):
    """Test chunks count after insertion."""
    assert populated_store.count_chunks() == 3


def test_duplicate_document_prevention(populated_store, embedding_mgr):
    """Test that attempting to re-index the same document_id is rejected."""
    duplicate_chunks = [
        DocumentChunk(
            chunk_id="doc_A_p1_c0001",
            document_id="doc_A",
            source="python_guide.pdf",
            page_number=1,
            chunk_index=1,
            text="Python is an interpreted, high-level, general-purpose programming language.",
            char_count=73,
            token_estimate=18,
        )
    ]
    embs = embedding_mgr.embed_texts([duplicate_chunks[0].text])
    added = populated_store.add_chunks(duplicate_chunks, embs)
    assert added == 0
    assert populated_store.count_chunks() == 3


def test_semantic_retrieval_ranking(populated_store, embedding_mgr):
    """Test that retriever ranks python chunks highest for python query."""
    retriever = SemanticRetriever(
        vector_store=populated_store,
        embedding_manager=embedding_mgr,
        default_top_k=2,
    )

    result = retriever.retrieve("How do list comprehensions work in Python?")
    assert result.has_context is True
    assert len(result.chunks) <= 2

    # Top chunk should be the list comprehension chunk
    top_chunk = result.chunks[0]
    assert "comprehension" in top_chunk.text.lower()
    assert top_chunk.source == "python_guide.pdf"
    assert top_chunk.page_number == 2
    assert 0.0 <= top_chunk.similarity_score <= 1.0


def test_retrieval_document_filtering(populated_store, embedding_mgr):
    """Test restricting retrieval to a specific document_id."""
    retriever = SemanticRetriever(
        vector_store=populated_store,
        embedding_manager=embedding_mgr,
    )

    # Search query about SQL, but filtered to doc_A (Python guide)
    result = retriever.retrieve("SQL relational queries", document_id="doc_A")
    for chunk in result.chunks:
        assert chunk.document_id == "doc_A"


def test_document_deletion(populated_store):
    """Test purging a document by document_id."""
    assert populated_store.document_exists("doc_A") is True
    deleted_count = populated_store.delete_document("doc_A")
    assert deleted_count == 2
    assert populated_store.document_exists("doc_A") is False
    assert populated_store.count_chunks() == 1


def test_empty_collection_retrieval(temp_chroma_store, embedding_mgr):
    """Test querying an empty collection does not crash."""
    retriever = SemanticRetriever(
        vector_store=temp_chroma_store,
        embedding_manager=embedding_mgr,
    )
    result = retriever.retrieve("Any random query")
    assert result.has_context is False
    assert len(result.chunks) == 0

