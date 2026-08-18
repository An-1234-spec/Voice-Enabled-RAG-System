"""
generation/llm.py

Unified LLM client supporting multiple low-latency providers (Groq, Gemini, OpenAI).
Swappable via environment variables (LLM_PROVIDER, LLM_MODEL) or config settings.
Integrates persistent HTTP connection pooling.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from groq import Groq
from openai import OpenAI
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config.settings import settings
from generation.prompts import build_messages


@dataclass
class LLMResult:
    answer: str
    grounded: bool
    sources_used: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    parse_error: bool = False
    raw_text: str | None = None  # populated only on parse_error, for debugging

    # timing breakdown in ms (namespaced for backward compatibility)
    groq_queue_ms: float = 0.0
    groq_prompt_ms: float = 0.0
    groq_completion_ms: float = 0.0
    groq_server_total_ms: float = 0.0


class GroqLLMClient:
    """Unified LLM Client supporting Groq, Gemini, and OpenAI with connection pooling."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = provider or os.environ.get("LLM_PROVIDER") or settings.llm_provider
        self.provider = self.provider.lower()

        # Resolve model name based on provider
        if model:
            self.model = model
        else:
            self.model = os.environ.get("LLM_MODEL") or settings.llm_model
            # Only apply default fallback if self.model is empty or matches the slow/mock default
            if not self.model or self.model == "openai/gpt-oss-20b":
                if self.provider == "gemini":
                    self.model = "gemini-2.5-flash"
                elif self.provider == "openai":
                    self.model = "gpt-4o-mini"
                elif self.provider == "groq":
                    self.model = "llama-3.1-8b-instant"  # highly optimized on Groq

        # Initialize the appropriate client
        self.groq_client = None
        self.openai_client = None
        self.gemini_client = None

        if self.provider == "groq":
            key = api_key or os.environ.get("GROQ_API_KEY") or settings.groq_api_key
            if not key:
                raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env or environment.")
            self.groq_client = Groq(api_key=key)
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai SDK is not installed. Run `pip install google-genai`.")
            key = api_key or os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key
            if not key:
                raise EnvironmentError("GEMINI_API_KEY not set. Add it to your .env or environment.")
            self.gemini_client = genai.Client(api_key=key)
        elif self.provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY") or settings.openai_api_key
            if not key:
                raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env or environment.")
            self.openai_client = OpenAI(api_key=key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'groq', 'gemini', or 'openai'.")

    def generate(
        self,
        query: str,
        chunks: list[dict],
        temperature: float = 0.0,
        max_completion_tokens: int = 100,  # optimized for short 1-2 sentence answers
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> LLMResult:
        messages = build_messages(query, chunks)
        max_tokens = kwargs.get("max_tokens") or max_completion_tokens

        t_start = time.perf_counter()

        if self.provider == "groq":
            extra_body = {}
            # Only add reasoning effort if model is a reasoning model or explicitly requested
            if reasoning_effort and "gpt-oss" in self.model:
                extra_body["reasoning_effort"] = reasoning_effort

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body if extra_body else None,
            )
            raw_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Extract server usage metrics
            u = response.usage
            timing = dict(
                groq_queue_ms=getattr(u, "queue_time", 0.0) * 1000,
                groq_prompt_ms=getattr(u, "prompt_time", 0.0) * 1000,
                groq_completion_ms=getattr(u, "completion_time", 0.0) * 1000,
                groq_server_total_ms=getattr(u, "total_time", 0.0) * 1000,
            )

        elif self.provider == "gemini":
            # For Gemini SDK, we convert chat completion messages structure
            gemini_contents = []
            system_instruction = None
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                else:
                    # Map role to user/model
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )

            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text
            finish_reason = None # Available in candidates if needed

            # Measure wall-clock API duration as server total (no server timing breakdown available)
            wall_ms = (time.perf_counter() - t_start) * 1000
            timing = dict(
                groq_queue_ms=0.0,
                groq_prompt_ms=0.0,
                groq_completion_ms=wall_ms,
                groq_server_total_ms=wall_ms,
            )

        elif self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Measure wall-clock API duration as server total
            wall_ms = (time.perf_counter() - t_start) * 1000
            timing = dict(
                groq_queue_ms=0.0,
                groq_prompt_ms=0.0,
                groq_completion_ms=wall_ms,
                groq_server_total_ms=wall_ms,
            )

        if finish_reason == "length":
            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason="LLM output truncated (finish_reason=length)",
                parse_error=True,
                raw_text=raw_text,
                **timing,
            )

        try:
            parsed = json.loads(raw_text)
            return LLMResult(
                answer=parsed.get("answer", ""),
                grounded=bool(parsed.get("grounded", False)),
                sources_used=parsed.get("sources_used", []),
                refusal_reason=parsed.get("refusal_reason"),
                **timing,
            )
        except (json.JSONDecodeError, TypeError):
            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason="LLM returned malformed JSON",
                parse_error=True,
                raw_text=raw_text,
                **timing,
            )