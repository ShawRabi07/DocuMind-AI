"""
DocuMind AI – Embedding Generation Module
==========================================
Encodes textual chunks and user questions into dense vector embeddings using
Sentence Transformers (sentence-transformers/all-MiniLM-L6-v2).
Implements model caching to prevent expensive re-initialization on every query.
"""

from __future__ import annotations

import logging
from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Centralized embedding engine with singleton caching and batch processing.

    Why sentence-transformers/all-MiniLM-L6-v2?
    - 384-dimensional dense vector space.
    - Highly optimized balance between inference speed, small memory footprint (~80 MB),
      and semantic retrieval accuracy across general domain English documents.
    - Normalized embeddings allow Inner Product / Cosine Similarity to be computed
      via fast Euclidean distance or dot-product.
    """

    _cached_instance: Optional["EmbeddingManager"] = None
    _model_name: Optional[str] = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading embedding model '%s' on %s...", model_name, device)
        try:
            # First attempt loading from local cache for instant offline startup
            try:
                self._model = SentenceTransformer(
                    model_name,
                    device=device,
                    local_files_only=True,
                )
            except Exception:
                self._model = SentenceTransformer(model_name, device=device)

            # Retrieve embedding dimension using forward-compatible method
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimension: int = self._model.get_embedding_dimension()
            else:
                self.dimension = self._model.get_sentence_embedding_dimension()

            logger.info(
                "Embedding model '%s' loaded successfully. Vector dimension: %d.",
                model_name,
                self.dimension,
            )
        except Exception as e:
            logger.error("Failed to load SentenceTransformer '%s': %s", model_name, str(e))
            raise RuntimeError(f"Could not load embedding model '{model_name}': {e}") from e

    @classmethod
    def get_instance(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "EmbeddingManager":
        """Singleton accessor to prevent reloading weights into memory repeatedly."""
        if cls._cached_instance is None or cls._model_name != model_name:
            cls._cached_instance = cls(model_name=model_name, device=device)
            cls._model_name = model_name
        return cls._cached_instance

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of document chunks in batches.

        Args:
            texts: List of chunk text strings.

        Returns:
            List of float vector lists, each of length `self.dimension`.
        """
        if not texts:
            return []

        try:
            # Compatible with both modern and legacy sentence_transformers
            kwargs = {
                "batch_size": self.batch_size,
                "show_progress_bar": False,
                "normalize_embeddings": self.normalize_embeddings,
                "convert_to_numpy": True,
            }
            try:
                embeddings: np.ndarray = self._model.encode(texts, **kwargs)
            except TypeError:
                embeddings = self._model.encode(sentences=texts, **kwargs)

            # Validate output dimensions
            if embeddings.shape[-1] != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[-1]}"
                )

            return embeddings.tolist()
        except Exception as e:
            logger.error("Batch embedding generation failed: %s", str(e))
            raise RuntimeError(f"Embedding generation error: {e}") from e

    def embed_query(self, query: str) -> List[float]:
        """
        Encodes a single user question into the same vector space as the documents.

        Args:
            query: The user question string.

        Returns:
            1D float vector of length `self.dimension`.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        try:
            kwargs = {
                "show_progress_bar": False,
                "normalize_embeddings": self.normalize_embeddings,
                "convert_to_numpy": True,
            }
            try:
                vec: np.ndarray = self._model.encode(cleaned_query, **kwargs)
            except TypeError:
                vec = self._model.encode(sentences=cleaned_query, **kwargs)

            if len(vec) != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, got {len(vec)}"
                )
            return vec.tolist()
        except Exception as e:
            logger.error("Query embedding failed for '%s': %s", query[:50], str(e))
            raise RuntimeError(f"Query embedding generation error: {e}") from e

    # LangChain Embeddings Protocol Compatibility
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain compatible interface."""
        return self.embed_texts(texts)
