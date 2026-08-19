"""
Central configuration for the Voice-Enabled RAG system.

Uses pydantic-settings to load from .env file with sensible defaults.
All tunable parameters are centralized here — no magic numbers scattered in code.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """All configuration for the RAG system, loaded from .env + defaults."""

    # ── API Keys ──────────────────────────────────────────────────────
    sarvam_api_key: str = Field(default="", description="Sarvam STT API key")
    groq_api_key: str = Field(default="", description="Groq LLM API key")

    # ── Dataset ───────────────────────────────────────────────────────
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_language: str = "default"  # Use default config (all languages)
    dataset_split: str = "validation"
    dataset_subset_size: int = 5000  # Number of records to download
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # ── Embedding ─────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ── Chunking ──────────────────────────────────────────────────────
    default_chunk_strategy: str = "sentence"  # fixed|sentence|paragraph|semantic|parent_child|metadata
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 50
    semantic_similarity_threshold: float = 0.75
    parent_chunk_size_tokens: int = 512

    # ── Retrieval ─────────────────────────────────────────────────────
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fusion_top_k: int = 30  # Candidates sent to reranker
    final_top_k: int = 5  # Final results after reranking
    dense_weight: float = 0.6
    bm25_weight: float = 0.4

    # ── Reranking ─────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    # ── LLM Generation ────────────────────────────────────────────────
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 300
    llm_timeout: float = 10.0  # seconds

    # ── Guardrails ────────────────────────────────────────────────────
    safety_enabled: bool = True
    retrieval_confidence_threshold: float = 0.3  # Min score to proceed
    domain_relevance_threshold: float = 0.2  # Min cosine sim to corpus centroid
    grounding_overlap_threshold: float = 0.3  # Min token overlap ratio
    max_retries: int = 2
    retry_backoff_base: float = 0.5  # seconds

    # ── Index Storage ─────────────────────────────────────────────────
    index_dir: Path = PROJECT_ROOT / "indexes"

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ── Sarvam STT ────────────────────────────────────────────────────
    sarvam_language: str = "en-IN"  # Default STT language
    sarvam_model: str = "saaras:v3"

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance — import this everywhere
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
