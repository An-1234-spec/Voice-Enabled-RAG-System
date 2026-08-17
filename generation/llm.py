"""
generation/llm.py

LLM client using the Groq API (OpenAI-compatible) for RAG answer
generation. Structured JSON output, parsed and validated defensively.

MODEL: default is openai/gpt-oss-20b, NOT llama-3.1-8b-instant as
originally specified in the project plan - Groq has deprecated
llama-3.1-8b-instant and llama-3.3-70b-versatile. gpt-oss-20b is their
current recommended small/fast general-purpose replacement (closest fit
to the original plan's speed rationale). Verify current model
availability/pricing at console.groq.com before final submission.

REQUIRES: GROQ_API_KEY set in the environment before running.
    PowerShell:  $env:GROQ_API_KEY="your-key-here"

Usage (CLI smoke test - requires a real API key and network access):
    python -m generation.llm --query "what is a corporation?" --strategy fixed_token
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from groq import Groq
from sentence_transformers import SentenceTransformer

from generation.prompts import build_messages
from retrieval.reranker import RerankedRetriever

DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass
class LLMResult:
    answer: str
    grounded: bool
    sources_used: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    parse_error: bool = False
    raw_text: str | None = None  # populated only on parse_error, for debugging

    # Groq server-side timing breakdown (from response.usage), in ms.
    # Lets the orchestrator/eval scripts separate "our pipeline" latency
    # from "Groq's queue + compute" latency instead of one blended number.
    groq_queue_ms: float = 0.0
    groq_prompt_ms: float = 0.0
    groq_completion_ms: float = 0.0
    groq_server_total_ms: float = 0.0


class GroqLLMClient:
    """Wraps Groq's chat completion endpoint for RAG answer generation."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Set it in your environment before running "
                '(PowerShell: $env:GROQ_API_KEY="your-key-here"), or pass api_key= explicitly.'
            )
        self.model = model
        self.client = Groq(api_key=key)

    def generate(
        self,
        query: str,
        chunks: list[dict],
        temperature: float = 0.0,
        max_completion_tokens: int = 300,
        reasoning_effort: str = "low",
    ) -> LLMResult:
        messages = build_messages(query, chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"},
            extra_body={"reasoning_effort": reasoning_effort},
        )
        choice = response.choices[0]
        raw_text = choice.message.content

        u = response.usage
        groq_timing = dict(
            groq_queue_ms=u.queue_time * 1000,
            groq_prompt_ms=u.prompt_time * 1000,
            groq_completion_ms=u.completion_time * 1000,
            groq_server_total_ms=u.total_time * 1000,
        )

        if choice.finish_reason == "length":
            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason="LLM output truncated (finish_reason=length)",
                parse_error=True,
                raw_text=raw_text,
                **groq_timing,
            )

        try:
            parsed = json.loads(raw_text)
            return LLMResult(
                answer=parsed.get("answer", ""),
                grounded=bool(parsed.get("grounded", False)),
                sources_used=parsed.get("sources_used", []),
                refusal_reason=parsed.get("refusal_reason"),
                **groq_timing,
            )
        except json.JSONDecodeError:
            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason="LLM returned malformed JSON",
                parse_error=True,
                raw_text=raw_text,
                **groq_timing,
            )


def main():
    parser = argparse.ArgumentParser(description="RAG generation smoke test (retrieval + LLM answer).")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-n", type=int, default=10)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    print("Loading embedding model + retriever...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    retriever = RerankedRetriever(
        strategy=args.strategy,
        base_mode="hybrid",
        faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir,
        retrieve_n=args.retrieve_n,
        model=embed_model,
    )

    print(f"Retrieving top-{args.top_k} chunks for: {args.query}")
    chunks = retriever.search(args.query, top_k=args.top_k)
    for i, c in enumerate(chunks, 1):
        print(f"  [S{i}] {c['text'][:80]}...")

    print(f"\nCalling Groq ({args.model})...")
    llm = GroqLLMClient(model=args.model)
    result = llm.generate(args.query, chunks)

    print(f"\nAnswer: {result.answer}")
    print(f"Grounded: {result.grounded}")
    print(f"Sources used: {result.sources_used}")
    if result.refusal_reason:
        print(f"Refusal reason: {result.refusal_reason}")
    if result.parse_error:
        print(f"[WARNING] JSON parse failed. Raw output:\n{result.raw_text}")
    print(
        f"\n[groq server timing] queue={result.groq_queue_ms:.0f}ms "
        f"prompt={result.groq_prompt_ms:.0f}ms completion={result.groq_completion_ms:.0f}ms "
        f"total={result.groq_server_total_ms:.0f}ms"
    )


if __name__ == "__main__":
    main()