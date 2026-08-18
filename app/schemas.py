"""
app/schemas.py

Pydantic models for API request/response types. Maps directly onto the
structured JSON output from pipeline/orchestrator.py's RAGResponse and
generation/llm.py's LLMResult.

Used by app/api.py for automatic request validation, response serialization,
and OpenAPI schema generation (/docs).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Text query input for POST /query."""
    query: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    strategy: str = Field(default="fixed_token", description="Chunking strategy to use for retrieval")


class SourceInfo(BaseModel):
    """A single retrieved source passage with its relevance score."""
    chunk_id: str
    text: str
    score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Per-stage timing in milliseconds."""
    safety_ms: Optional[float] = None
    relevance_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    grounding_ms: Optional[float] = None
    total_ms: Optional[float] = None
    stt_ms: Optional[float] = None  # Only present for voice queries


class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""
    request_id: str
    query: str
    answer: str
    grounded: bool
    confidence: float = Field(default=0.0, description="Confidence score 0-1")
    sources: list[SourceInfo] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    stage_reached: str = "structured_response"
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)


class VoiceResponse(RAGResponse):
    """Extended response for voice queries — includes STT transcript info."""
    transcript: str = ""
    detected_language: Optional[str] = None
    stt_latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "ok"
    strategy: str = ""
    index_loaded: bool = False
    model_loaded: bool = False
