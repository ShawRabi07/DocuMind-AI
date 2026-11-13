"""
DocuMind AI – Configuration Module
====================================
Centralizes all application configurations, directory paths, model names,
and hyperparameter settings using python-dotenv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass
class AppConfig:
    """Central configuration class with validation and environment variable overrides."""

    # Project directories
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    upload_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
        )
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
        )
    )

    # Embedding settings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # Vector store settings
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION", "documind_collection")

    # Ollama LLM settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "1024"))

    # Text chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval settings
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Document upload limits
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    allowed_extensions: tuple[str, ...] = (".pdf",)

    def __post_init__(self) -> None:
        """Validate configuration values and ensure required directories exist."""
        self.validate()
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Validates configuration sanity, raising ValueError for invalid settings."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

        if self.top_k <= 0:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")

        if self.max_upload_size_mb <= 0:
            raise ValueError(f"max_upload_size_mb must be positive, got {self.max_upload_size_mb}")

        if self.vector_store_backend not in ("chroma", "faiss"):
            raise ValueError(
                f"vector_store_backend must be 'chroma' or 'faiss', got '{self.vector_store_backend}'"
            )

        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"ollama_base_url must begin with http:// or https://, got '{self.ollama_base_url}'"
            )


# Default global settings instance
settings = AppConfig()

