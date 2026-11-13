"""
Unit Tests for Embeddings Module
=================================
Tests vector dimension validation, batch processing, single query encoding, and caching.
"""

import pytest
import numpy as np
from core.embeddings import EmbeddingManager


@pytest.fixture(scope="module")
def embedding_mgr():
    """Module-scoped embedding manager (cached for test speed)."""
    return EmbeddingManager.get_instance(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )


def test_embedding_dimensions(embedding_mgr):
    """Test that all-MiniLM-L6-v2 outputs exactly 384 dimensions."""
    assert embedding_mgr.dimension == 384


def test_embed_query_shape(embedding_mgr):
    """Test single query embedding length and type."""
    vec = embedding_mgr.embed_query("What is deep learning?")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_embed_empty_query(embedding_mgr):
    """Test error on empty query."""
    with pytest.raises(ValueError, match="cannot be empty"):
        embedding_mgr.embed_query("   ")


def test_embed_texts_batch(embedding_mgr):
    """Test batch embedding generation."""
    texts = [
        "First document excerpt regarding artificial intelligence.",
        "Second document excerpt discussing neural networks.",
        "Third document excerpt covering vector databases.",
    ]
    embeddings = embedding_mgr.embed_texts(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 384


def test_embed_texts_empty(embedding_mgr):
    """Test empty batch returns empty list."""
    assert embedding_mgr.embed_texts([]) == []


def test_singleton_caching():
    """Test that get_instance returns the same singleton object."""
    inst1 = EmbeddingManager.get_instance()
    inst2 = EmbeddingManager.get_instance()
    assert inst1 is inst2

