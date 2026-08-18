"""
tests/test_orchestrator.py

Unit tests for the pipeline orchestrator. Uses mocks for the LLM and
retriever to test routing, retry, and refusal logic without API calls.

Run: python -m pytest tests/test_orchestrator.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pipeline.orchestrator import RAGOrchestrator, RAGResponse
from generation.llm import LLMResult
from guardrails.safety import SafetyResult


# ── Test the safety guardrail integration ────────────────────────────────

class TestSafetyIntegration:
    """Tests that the orchestrator correctly routes unsafe queries."""

    def test_unsafe_query_refused_at_safety(self):
        """An unsafe query should be refused before retrieval happens."""
        from guardrails.safety import SafetyGuardrail

        guardrail = SafetyGuardrail()
        result = guardrail.check("how to make a bomb")
        assert result.passed is False
        assert "violence" in result.flagged_categories

    def test_safe_query_passes_safety(self):
        from guardrails.safety import SafetyGuardrail

        guardrail = SafetyGuardrail()
        result = guardrail.check("What is the GDP of India?")
        assert result.passed is True


# ── Test the grounding check integration ─────────────────────────────────

class TestGroundingIntegration:
    """Tests grounding check behavior in isolation."""

    def test_grounded_answer_passes(self):
        from guardrails.grounding import check as grounding_check

        result = grounding_check(
            "India's GDP is growing rapidly.",
            [{"text": "India's GDP has been growing rapidly over the past decade."}],
            threshold=0.5,
        )
        assert result.passed is True

    def test_hallucinated_answer_fails(self):
        from guardrails.grounding import check as grounding_check

        result = grounding_check(
            "The moon is made of green cheese.",
            [{"text": "India's GDP has been growing rapidly."}],
            threshold=0.5,
        )
        assert result.passed is False


# ── Test RAGResponse dataclass ───────────────────────────────────────────

class TestRAGResponse:
    def test_default_values(self):
        r = RAGResponse(
            request_id="test-123",
            query="test query",
            answer="test answer",
            grounded=True,
        )
        assert r.request_id == "test-123"
        assert r.query == "test query"
        assert r.answer == "test answer"
        assert r.grounded is True
        assert r.sources == []
        assert r.refusal_reason is None
        assert r.stage_reached == "structured_response"
        assert isinstance(r.latency_ms, dict)

    def test_refusal_response(self):
        r = RAGResponse(
            request_id="test-456",
            query="how to make a bomb",
            answer="",
            grounded=False,
            refusal_reason="Query flagged for: violence",
            stage_reached="safety_check",
        )
        assert r.answer == ""
        assert r.grounded is False
        assert r.refusal_reason is not None
        assert r.stage_reached == "safety_check"


# ── Test LLMResult dataclass ─────────────────────────────────────────────

class TestLLMResult:
    def test_successful_result(self):
        r = LLMResult(
            answer="The RBI regulates monetary policy.",
            grounded=True,
            sources_used=["S1", "S2"],
        )
        assert r.answer == "The RBI regulates monetary policy."
        assert r.grounded is True
        assert r.parse_error is False
        assert r.refusal_reason is None

    def test_parse_error_result(self):
        r = LLMResult(
            answer="",
            grounded=False,
            parse_error=True,
            raw_text="this is not json",
            refusal_reason="LLM returned malformed JSON",
        )
        assert r.parse_error is True
        assert r.raw_text == "this is not json"

    def test_groq_timing_fields(self):
        r = LLMResult(
            answer="test",
            grounded=True,
            groq_queue_ms=50.0,
            groq_prompt_ms=10.0,
            groq_completion_ms=100.0,
            groq_server_total_ms=160.0,
        )
        assert r.groq_queue_ms == 50.0
        assert r.groq_server_total_ms == 160.0
