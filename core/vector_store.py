"""
DocuMind AI – Vector Store Module
==================================
Manages persistent vector indexing and similarity search with ChromaDB as the primary
backend and an optional FAISS backend, adhering to a unified BaseVectorStore abstraction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.text_splitter import DocumentChunk

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract interface defining the contract for all vector store implementations."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Adds document chunks and their corresponding embeddings to the index."""
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries the vector store for the top_k most similar chunks."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Deletes all chunks associated with a specific document_id."""
        pass

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """Checks if a document_id is already indexed."""
        pass

    @abstractmethod
    def count_chunks(self) -> int:
        """Returns total number of chunks indexed in the store."""
        pass

    @abstractmethod
    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """Returns a list of unique indexed documents with chunk and page stats."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    Persistent ChromaDB vector store implementation.

    Features:
    - Native PersistentClient for robust on-disk SQLite/DuckDB persistence.
    - HNSW space: Cosine distance ("hnsw:space": "cosine").
    - Atomic batch insertion.
    - Metadata-filtered queries.
    - Cascading document deletions by document_id.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "documind_collection",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(
            "Initializing ChromaDB PersistentClient at '%s'...",
            self.persist_directory,
        )
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection with cosine similarity distance metric
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready. Current indexed chunks: %d.",
            self.collection_name,
            self._collection.count(),
        )

    def document_exists(self, document_id: str) -> bool:
        """Checks if any chunk with this document_id is already indexed."""
        results = self._collection.get(
            where={"document_id": document_id},
            limit=1,
        )
        return bool(results and results.get("ids"))

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Stores document chunks, embeddings, and metadata in ChromaDB.

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: Corresponding list of vector embeddings.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        # Prevent duplicate ingestion of already-indexed document
        doc_id = chunks[0].document_id
        if self.document_exists(doc_id):
            logger.warning(
                "Document '%s' is already indexed in ChromaDB. Skipping duplicate addition.",
                doc_id,
            )
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            # ChromaDB requires flat primitive metadata (str, int, float, bool)
            meta: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page_number": int(chunk.page_number),
                "chunk_index": int(chunk.chunk_index),
                "char_count": int(chunk.char_count),
                "token_estimate": int(chunk.token_estimate),
            }
            metadatas.append(meta)

        # Batch insert into ChromaDB (ChromaDB handles internal batching up to max batch size)
        batch_size = 500
        total_added = 0
        for i in range(0, len(ids), batch_size):
            end_idx = i + batch_size
            self._collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )
            total_added += len(ids[i:end_idx])

        logger.info("Successfully added %d chunks for document '%s'.", total_added, doc_id)
        return total_added

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic similarity search against indexed chunks.

        Args:
            query_embedding: Dense vector for user question.
            top_k: Maximum number of relevant chunks to retrieve.
            document_id: Optional filter to restrict retrieval to a single document.

        Returns:
            List of structured dictionary results containing chunk metadata and distance.
        """
        total_chunks = self._collection.count()
        if total_chunks == 0:
            return []

        # Bound top_k to available chunks
        k = min(top_k, total_chunks)

        where_filter: Optional[Dict[str, Any]] = None
        if document_id:
            where_filter = {"document_id": document_id}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            dist = float(distances[i]) if i < len(distances) else 1.0

            # For cosine distance: distance = 1 - cosine_similarity.
            # Range is [0, 2], where 0 is identical and 1 is orthogonal.
            # Similarity score mapped to [0, 1] for intuitive display:
            sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "document_id": meta.get("document_id", ""),
                    "source": meta.get("source", "Unknown"),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "char_count": int(meta.get("char_count", 0)),
                    "distance": dist,
                    "similarity_score": sim_score,
                }
            )

        return retrieved

    def delete_document(self, document_id: str) -> int:
        """
        Deletes all chunks belonging to a document from ChromaDB.

        Args:
            document_id: The ID of the document to purge.

        Returns:
            Number of chunks removed.
        """
        existing = self._collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        if not existing or not existing.get("ids"):
            logger.info("Document '%s' not found in ChromaDB for deletion.", document_id)
            return 0

        count_to_remove = len(existing["ids"])
        self._collection.delete(where={"document_id": document_id})
        logger.info(
            "Purged document '%s' and all %d associated chunks from ChromaDB.",
            document_id,
            count_to_remove,
        )
        return count_to_remove

    def count_chunks(self) -> int:
        """Returns total indexed chunks in collection."""
        return self._collection.count()

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Inspects stored metadata to construct a summary of all unique indexed documents.

        Returns:
            List of dicts with document_id, source, total_chunks, and max_page.
        """
        total = self._collection.count()
        if total == 0:
            return []

        all_items = self._collection.get(include=["metadatas"])
        if not all_items or not all_items.get("metadatas"):
            return []

        docs_map: Dict[str, Dict[str, Any]] = {}
        for meta in all_items["metadatas"]:
            doc_id = meta.get("document_id")
            if not doc_id:
                continue

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "source": meta.get("source", "Unknown"),
                    "chunk_count": 0,
                    "pages": set(),
                }
            docs_map[doc_id]["chunk_count"] += 1
            docs_map[doc_id]["pages"].add(int(meta.get("page_number", 1)))

        summaries: List[Dict[str, Any]] = []
        for doc_id, info in docs_map.items():
            summaries.append(
                {
                    "document_id": doc_id,
                    "source": info["source"],
                    "chunk_count": info["chunk_count"],
                    "total_pages": len(info["pages"]),
                }
            )

        return sorted(summaries, key=lambda x: x["source"])


def get_vector_store(
    backend: str = "chroma",
    persist_dir: Optional[Path] = None,
    collection_name: str = "documind_collection",
) -> BaseVectorStore:
    """Factory function providing the chosen vector database implementation."""
    from config.settings import settings

    target_dir = persist_dir or settings.chroma_persist_dir

    if backend.lower() == "chroma":
        return ChromaVectorStore(
            persist_directory=target_dir,
            collection_name=collection_name,
        )
    else:
        raise ValueError(f"Vector store backend '{backend}' is not supported yet.")

