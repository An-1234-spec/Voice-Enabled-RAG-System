"""
Base Chunker Interface — Phase 2

PURPOSE: Define a consistent abstract interface for all chunking strategies.
         Every chunker must implement `chunk()` and return a list of `Chunk` objects.

WHY: We're implementing 6 different chunking strategies. A consistent interface lets us:
  1. Swap strategies easily via config
  2. Benchmark them fairly with identical inputs
  3. Compose them (e.g., parent-child uses fixed chunker internally)

DESIGN DECISIONS:
  - Chunks carry metadata (source passage, strategy name, position)
  - Each chunk has a unique chunk_id derived from passage_id + strategy + index
  - Parent-child chunker stores parent_id for context expansion
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    # Identity
    chunk_id: str  # Unique ID: f"{passage_id}_{strategy}_{idx}"
    passage_id: str  # Source passage ID
    document_id: str  # Source document ID

    # Content
    text: str  # The actual chunk text
    token_count: int  # Approximate token count

    # Metadata
    strategy: str  # Which chunking strategy produced this
    chunk_index: int  # Position within the source passage's chunks
    total_chunks: int  # Total chunks from this passage

    # Optional metadata
    query_id: Optional[int] = None
    query: Optional[str] = None
    query_type: Optional[str] = None
    answer: Optional[str] = None
    is_selected: Optional[int] = None
    language: Optional[str] = None
    parent_id: Optional[str] = None  # For parent-child chunking

    # Extra metadata dict for extensibility
    extra: dict = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def chunk(self, text: str, passage_id: str, document_id: str, **metadata) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The passage text to chunk.
            passage_id: Unique ID of the source passage.
            document_id: Unique ID of the source document.
            **metadata: Additional metadata to attach to each chunk
                        (query_id, query, query_type, answer, is_selected, language).

        Returns:
            List of Chunk objects.
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimation.
        Uses the ~4 chars per token heuristic (accurate within ~10% for English).
        This avoids importing a tokenizer for a simple estimate.
        """
        return max(1, len(text) // 4)

    def _make_chunk_id(self, passage_id: str, index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{passage_id}_{self.strategy_name}_{index}"

    def _build_chunk(
        self,
        text: str,
        passage_id: str,
        document_id: str,
        chunk_index: int,
        total_chunks: int,
        **metadata,
    ) -> Chunk:
        """Helper to build a Chunk with consistent metadata."""
        return Chunk(
            chunk_id=self._make_chunk_id(passage_id, chunk_index),
            passage_id=passage_id,
            document_id=document_id,
            text=text,
            token_count=self._estimate_tokens(text),
            strategy=self.strategy_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            query_id=metadata.get("query_id"),
            query=metadata.get("query"),
            query_type=metadata.get("query_type"),
            answer=metadata.get("answer"),
            is_selected=metadata.get("is_selected"),
            language=metadata.get("language"),
            parent_id=metadata.get("parent_id"),
        )
