"""
generation/llm.py

Unified LLM client supporting:
- Gemini
- Groq
- OpenAI

Provider/model are selected through:
    LLM_PROVIDER
    LLM_MODEL

Example:
    $env:LLM_PROVIDER="gemini"
    $env:LLM_MODEL="gemini-2.5-flash-lite"

LATENCY FIX (2026-08-21): the orchestrator calls generate(query, chunks)
with no keyword overrides, so this method's own defaults are what every
Groq request actually used. Two of those defaults were silently costing
hundreds of ms on openai/gpt-oss-20b:

  1. reasoning_effort defaulted to None, so the `if reasoning_effort and
     "gpt-oss" in self.model` guard below never fired, and Groq applied
     its OWN default for gpt-oss models — which is 'medium' (confirmed
     via Groq's API reference: "'medium' is the default value" for
     openai/gpt-oss-20b and openai/gpt-oss-120b). Medium-effort reasoning
     generates hidden reasoning tokens before the visible answer, and
     those tokens are billed against the same completion budget/wall
     clock — for a one-sentence RAG answer this is pure overhead.
     Fixed by defaulting reasoning_effort="low" (a valid value for this
     model family; 'none' is NOT accepted by gpt-oss, only qwen3 accepts
     'none'/'default' — verified against Groq's docs, not assumed).

  2. max_completion_tokens defaulted to 512 — far more than a ~30-word
     grounded answer + citation tag needs. Cut to 120 as a starting
     point (more conservative than Ollama's 48, since gpt-oss reasoning
     tokens may still consume some of this budget even at low effort —
     needs experimental validation via evaluation/latency_eval.py before
     tightening further; going straight to a very low value risks
     reproducing the earlier "truncated answer, citation cut off"
     failure mode already seen once in generation/ollama_llm.py).

Re-run a single-query smoke test (python -m generation.llm --query ...)
to confirm Grounded=True and a visibly lower groq_completion_ms before
trusting a full 200-query eval on this change.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from config.settings import settings
from generation.prompts import build_messages, parse_cited_answer
from retrieval.reranker import RerankedRetriever
from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI


DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash-lite"


@dataclass
class LLMResult:
    answer: str
    grounded: bool
    sources_used: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    parse_error: bool = False
    raw_text: str | None = None

    # Backward-compatible timing names
    groq_queue_ms: float = 0.0
    groq_prompt_ms: float = 0.0
    groq_completion_ms: float = 0.0
    groq_server_total_ms: float = 0.0


class GroqLLMClient:
    """
    Backward-compatible class name.

    Despite the name, this client can use Gemini, Groq, or OpenAI.
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
    ):

        # ---------------------------------------------------------
        # PROVIDER
        # ---------------------------------------------------------
        self.provider = (
            provider
            or os.environ.get("LLM_PROVIDER")
            or getattr(settings, "llm_provider", None)
            or DEFAULT_PROVIDER
        ).lower()

        # ---------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------
        self.model = (
            model
            or os.environ.get("LLM_MODEL")
            or getattr(settings, "llm_model", None)
            or DEFAULT_MODEL
        )

        # ---------------------------------------------------------
        # CLIENTS
        # ---------------------------------------------------------
        self.groq_client = None
        self.gemini_client = None
        self.openai_client = None

        # ---------------------------------------------------------
        # GEMINI
        # ---------------------------------------------------------
        if self.provider == "gemini":

            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-genai is not installed.\n"
                    "Run:\n"
                    "pip install google-genai"
                )

            key = (
                api_key
                or os.environ.get("GEMINI_API_KEY")
                or getattr(settings, "gemini_api_key", None)
            )

            if not key:
                raise EnvironmentError(
                    "GEMINI_API_KEY not set."
                )

            self.gemini_client = genai.Client(api_key=key)

        # ---------------------------------------------------------
        # GROQ
        # ---------------------------------------------------------
        elif self.provider == "groq":

            key = (
                api_key
                or os.environ.get("GROQ_API_KEY")
                or getattr(settings, "groq_api_key", None)
            )

            if not key:
                raise EnvironmentError(
                    "GROQ_API_KEY not set."
                )

            # max_retries=0: the SDK's default retry behavior sleeps and
            # retries on 429 (rate limit) before returning, which silently
            # inflates measured wall-clock latency by seconds per throttled
            # call — this showed up directly in a 200-query eval run as
            # P100=48,394ms even though the underlying Groq server_total
            # for successful calls stayed under 150ms. Fail fast instead:
            # a throttled request should surface immediately as an error
            # the orchestrator can log/refuse on, not a multi-second sleep
            # baked into the latency number.
            self.groq_client = Groq(api_key=key, max_retries=0, timeout=10.0)

        # ---------------------------------------------------------
        # OPENAI
        # ---------------------------------------------------------
        elif self.provider == "openai":

            key = (
                api_key
                or os.environ.get("OPENAI_API_KEY")
                or getattr(settings, "openai_api_key", None)
            )

            if not key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not set."
                )

            self.openai_client = OpenAI(api_key=key)

        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                "Use: gemini, groq, or openai."
            )

    # =============================================================
    # GENERATE
    # =============================================================

    def generate(
        self,
        query: str,
        chunks: list[dict],
        temperature: float = 0.0,
        max_completion_tokens: int = 120,  # was 512 — cut for a one-sentence RAG answer; sweep lower once reasoning_effort="low" is confirmed stable
        reasoning_effort: str | None = "low",  # was None — gpt-oss silently defaulted to Groq's own 'medium', burning hidden reasoning tokens on every call
    ) -> LLMResult:

        messages = build_messages(query, chunks)

        t_start = time.perf_counter()

        timing = {
            "groq_queue_ms": 0.0,
            "groq_prompt_ms": 0.0,
            "groq_completion_ms": 0.0,
            "groq_server_total_ms": 0.0,
        }

        raw_text = ""
        finish_reason = None

        # =========================================================
        # GEMINI
        # =========================================================

        if self.provider == "gemini":

            gemini_contents = []
            system_instruction = None

            for msg in messages:

                if msg["role"] == "system":
                    system_instruction = msg["content"]

                elif msg["role"] == "user":

                    gemini_contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text=msg["content"]
                                )
                            ],
                        )
                    )

            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,

                    max_output_tokens=max_completion_tokens,
                ),
            )

            raw_text = response.text or ""

            wall_ms = (
                time.perf_counter() - t_start
            ) * 1000

            timing["groq_completion_ms"] = wall_ms
            timing["groq_server_total_ms"] = wall_ms

        # =========================================================
        # GROQ
        # =========================================================

        elif self.provider == "groq":

            extra_body = {}

            # NOTE: gpt-oss accepts 'low'/'medium'/'high' only — it does
            # NOT accept 'none' (that's qwen3-only, per Groq's API docs).
            # If you ever swap to a qwen3 model on Groq and want reasoning
            # off entirely, 'none' is valid THERE but not here — don't
            # copy this value across model families without checking.
            if (
                reasoning_effort
                and "gpt-oss" in self.model
            ):
                extra_body["reasoning_effort"] = reasoning_effort

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                extra_body=(
                    extra_body
                    if extra_body
                    else None
                ),
            )

            raw_text = (
                response.choices[0].message.content
                or ""
            )

            finish_reason = (
                response.choices[0].finish_reason
            )

            usage = response.usage

            timing["groq_queue_ms"] = (
                getattr(
                    usage,
                    "queue_time",
                    0.0,
                )
                * 1000
            )

            timing["groq_prompt_ms"] = (
                getattr(
                    usage,
                    "prompt_time",
                    0.0,
                )
                * 1000
            )

            timing["groq_completion_ms"] = (
                getattr(
                    usage,
                    "completion_time",
                    0.0,
                )
                * 1000
            )

            timing["groq_server_total_ms"] = (
                getattr(
                    usage,
                    "total_time",
                    0.0,
                )
                * 1000
            )

        # =========================================================
        # OPENAI
        # =========================================================

        elif self.provider == "openai":

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens,
            )

            raw_text = (
                response.choices[0].message.content
                or ""
            )

            finish_reason = (
                response.choices[0].finish_reason
            )

            wall_ms = (
                time.perf_counter() - t_start
            ) * 1000

            timing["groq_completion_ms"] = wall_ms
            timing["groq_server_total_ms"] = wall_ms

        # =========================================================
        # TRUNCATION
        # =========================================================

        if finish_reason == "length":

            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason=(
                    "LLM output truncated"
                ),
                parse_error=True,
                raw_text=raw_text,
                **timing,
            )

        # =========================================================
        # PARSING
        # =========================================================

        answer, tags, _cited_chunks = parse_cited_answer(raw_text, chunks)

        if not answer:
            return LLMResult(
                answer="",
                grounded=False,
                sources_used=[],
                refusal_reason="Model indicated insufficient information to answer",
                parse_error=False,
                raw_text=raw_text,
                **timing,
            )

        return LLMResult(
            answer=answer,
            grounded=False,
            sources_used=tags,
            refusal_reason="pending_grounding_check",
            parse_error=False,
            raw_text=raw_text,
            **timing,
        )


# =============================================================
# CLI
# =============================================================

def main():

    parser = argparse.ArgumentParser(
        description="RAG generation smoke test."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed_token",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--retrieve-n",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        help="For gpt-oss models on Groq: low, medium, or high. Default 'low' for latency.",
    )

    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=120,
    )

    args = parser.parse_args()

    print(
        "Loading embedding model + retriever..."
    )

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    retriever = RerankedRetriever(
        strategy=args.strategy,
        base_mode="hybrid",
        faiss_dir=Path(
            "data/processed/faiss"
        ),
        chunks_dir=Path(
            "data/processed/chunks"
        ),
        retrieve_n=args.retrieve_n,
        model=embed_model,
    )

    print(
        f"Retrieving top-{args.top_k} chunks "
        f"for: {args.query}"
    )

    chunks = retriever.search(
        args.query,
        top_k=args.top_k,
    )

    for i, chunk in enumerate(
        chunks,
        1,
    ):
        print(
            f"  [S{i}] "
            f"{chunk['text'][:80]}..."
        )

    llm = GroqLLMClient(
        model=args.model
    )

    print(
        f"\nCalling "
        f"{llm.provider.upper()} "
        f"({llm.model}) "
        f"reasoning_effort={args.reasoning_effort} "
        f"max_completion_tokens={args.max_completion_tokens}..."
    )

    t0 = time.perf_counter()
    result = llm.generate(
        args.query,
        chunks,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(
        f"\nAnswer: {result.answer}"
    )

    print(
        f"Grounded: {result.grounded}"
    )

    print(
        f"Sources used: "
        f"{result.sources_used}"
    )

    if result.refusal_reason:
        print(
            f"Refusal reason: "
            f"{result.refusal_reason}"
        )

    if result.parse_error:
        print(
            "[WARNING] JSON parse failed."
        )
        print(
            f"Raw output:\n"
            f"{result.raw_text}"
        )

    if llm.provider == "groq":

        print(
            "\n[Groq server timing] "
            f"queue={result.groq_queue_ms:.0f}ms "
            f"prompt={result.groq_prompt_ms:.0f}ms "
            f"completion={result.groq_completion_ms:.0f}ms "
            f"total={result.groq_server_total_ms:.0f}ms"
        )

    else:

        print(
            f"\n[{llm.provider} API timing] "
            f"total={result.groq_server_total_ms:.0f}ms"
        )

    print(f"Wall-clock (this call): {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()