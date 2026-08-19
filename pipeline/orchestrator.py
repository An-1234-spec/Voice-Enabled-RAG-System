"""
pipeline/orchestrator.py

Full RAG pipeline, tying together everything built so far into one
callable entry point:

  validate_input -> safety_check -> relevance_check -> retrieve (hybrid+rerank)
  -> generate -> validate_output -> grounding_check -> retry (once) or
  fallback -> structured_response

Every stage timed; each call gets a request_id (per plan spec).

LLM PROVIDER SELECTION:
  Set LLM_PROVIDER env var to choose the generation backend:
    LLM_PROVIDER=gemini   (default) — Gemini via google-genai
    LLM_PROVIDER=groq     — Groq cloud API (requires GROQ_API_KEY)
    LLM_PROVIDER=openai   — OpenAI API (requires OPENAI_API_KEY)
    LLM_PROVIDER=ollama   — local Ollama (requires ollama serve + model pulled)

  Groq server-side timing fields (groq_queue_ms / groq_prompt_ms /
  groq_completion_ms / groq_server_total_ms) are only populated when the
  provider is "groq"; for other providers they remain 0.0 in the latency
  dict and are omitted from the printed breakdown.

SCOPE NOTE: pipeline/router.py and pipeline/retry.py are NOT separate
files here - routing is effectively done by the guardrail chain itself
(safety -> "unsafe", relevance -> "out_of_domain", grounding -> low
confidence), and retry is a short inline loop, not enough logic to
justify a standalone module given the timeline. Split out later if useful.

KNOWN GUARDRAIL LIMITATIONS (documented earlier in the project, carried
into this pipeline as-is):
  - relevance.py has weak discrimination on this broad, multi-domain
    corpus (see guardrails/relevance.py --calibrate findings).
  - grounding.py uses lexical + entity + sentence-level overlap, not real NLI.
Both are real but imperfect signals, not guarantees.

Usage (CLI smoke test):
    python -m pipeline.orchestrator --query "what is a corporation?" --strategy fixed_token
    python -m pipeline.orchestrator --query "what is a corporation?" --strategy fixed_token --llm-provider ollama
"""

import argparse
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sentence_transformers import SentenceTransformer

from guardrails.safety import SafetyGuardrail
from guardrails.relevance import RelevanceGuardrail
from guardrails.output_validator import validate as validate_output
from guardrails import grounding
from generation.llm import GroqLLMClient
from generation.prompts import tag_to_chunk_id
from retrieval.hybrid_retriever import HybridRetriever

DEFAULT_RELEVANCE_THRESHOLD = -0.0613  # from earlier calibration - see guardrails/relevance.py caveats
DEFAULT_GROUNDING_THRESHOLD = 0.5


@dataclass
class RAGResponse:
    request_id: str
    query: str
    answer: str
    grounded: bool
    sources: list[dict] = field(default_factory=list)  # [{chunk_id, text, score}]
    refusal_reason: str | None = None
    stage_reached: str = "structured_response"  # where the pipeline terminated
    latency_ms: dict = field(default_factory=dict)


def _build_llm(provider: str | None, model: str | None):
    """
    Factory: returns the appropriate LLM client for the requested provider.

    Provider resolution order:
      1. explicit `provider` argument
      2. LLM_PROVIDER environment variable
      3. default: "gemini"

    Supported providers: gemini, groq, openai, ollama
    """
    resolved_provider = (
        provider
        or os.environ.get("LLM_PROVIDER")
        or "gemini"
    ).lower()

    if resolved_provider == "ollama":
        from generation.ollama_llm import OllamaLLMClient
        return OllamaLLMClient(model=model or "llama3.2:1b")
    else:
        # GroqLLMClient handles gemini / groq / openai — see generation/llm.py
        return GroqLLMClient(provider=resolved_provider, model=model)


class RAGOrchestrator:
    def __init__(
        self,
        strategy: str,
        faiss_dir: Path = Path("data/processed/faiss"),
        chunks_dir: Path = Path("data/processed/chunks"),
        embeddings_dir: Path = Path("data/processed/embeddings"),
        retrieve_n: int = 10,
        top_k: int = 3,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        grounding_threshold: float = DEFAULT_GROUNDING_THRESHOLD,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ):
        self.strategy = strategy
        self.top_k = top_k

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.safety = SafetyGuardrail()
        self.relevance = RelevanceGuardrail(
            strategy=strategy, embeddings_dir=embeddings_dir, threshold=relevance_threshold, model=self.embed_model
        )
        self.retriever = HybridRetriever(
            strategy=strategy, faiss_dir=faiss_dir, chunks_dir=chunks_dir,
            dense_weight=0.9, bm25_weight=0.1, model=self.embed_model,
        )
        self.llm = _build_llm(llm_provider, llm_model)
        self.grounding_threshold = grounding_threshold

        # Track whether we're on a provider that exposes Groq-style server timing
        resolved_provider = (
            llm_provider
            or os.environ.get("LLM_PROVIDER")
            or "gemini"
        ).lower()
        self._report_groq_timing = (resolved_provider == "groq")

    def _refuse(self, request_id: str, query: str, reason: str, stage: str, latency_ms: dict) -> RAGResponse:
        return RAGResponse(
            request_id=request_id, query=query, answer="", grounded=False,
            refusal_reason=reason, stage_reached=stage, latency_ms=latency_ms,
        )

    def answer(self, query: str) -> RAGResponse:
        request_id = str(uuid.uuid4())
        latency_ms: dict[str, float] = {}
        t_start = time.perf_counter()

        def mark(stage: str, t0: float):
            latency_ms[stage] = (time.perf_counter() - t0) * 1000

        # 1. Safety
        t0 = time.perf_counter()
        safety_result = self.safety.check(query)
        mark("safety_ms", t0)
        if not safety_result.passed:
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, safety_result.reason, "safety_check", latency_ms)

        # 2. Relevance (known weak discrimination on this corpus - see grounding module docstring)
        t0 = time.perf_counter()
        relevance_result = self.relevance.check(query)
        mark("relevance_ms", t0)
        if not relevance_result.passed:
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, relevance_result.reason, "relevance_check", latency_ms)

        # 3. Retrieval (hybrid + rerank)
        t0 = time.perf_counter()
        chunks = self.retriever.search(query, top_k=self.top_k)
        mark("retrieval_ms", t0)
        if not chunks:
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, "No relevant passages retrieved", "retrieval", latency_ms)

        # 4. Generation, with one retry on invalid output or failed grounding
        result, cited_chunks = None, []
        for attempt in range(2):
            t0 = time.perf_counter()
            result = self.llm.generate(query, chunks)
            mark(f"generation_ms_attempt{attempt+1}", t0)

            # Pull Groq's own server-side timing breakdown when using Groq.
            # Other providers (Gemini, Ollama, OpenAI) don't expose this;
            # their fields stay at 0.0 and are omitted from the latency report.
            if self._report_groq_timing:
                latency_ms[f"groq_queue_ms_attempt{attempt+1}"] = result.groq_queue_ms
                latency_ms[f"groq_prompt_ms_attempt{attempt+1}"] = result.groq_prompt_ms
                latency_ms[f"groq_completion_ms_attempt{attempt+1}"] = result.groq_completion_ms
                latency_ms[f"groq_server_total_ms_attempt{attempt+1}"] = result.groq_server_total_ms

            # ── Output validation ──────────────────────────────────────────
            # The output validator checks structural consistency of the LLMResult:
            # sources_used tags resolve to real chunks, grounded/refusal fields
            # are internally consistent, no parse errors.
            # Special case: OllamaLLMClient sets refusal_reason="pending_grounding_check"
            # as a sentinel when it has an answer but grounding hasn't run yet.
            # We temporarily clear it so the validator sees a clean answer+sources
            # pair rather than an apparent refusal without a legitimate reason.
            pending_grounding = (
                result.refusal_reason == "pending_grounding_check"
                and bool(result.answer)
                and bool(result.sources_used)
            )
            if pending_grounding:
                result.refusal_reason = None
                result.grounded = True  # tentatively — overwritten by grounding check below

            t0 = time.perf_counter()
            validation = validate_output(result, chunks)
            mark(f"output_validation_ms_attempt{attempt+1}", t0)

            if not validation.valid:
                if attempt == 0:
                    continue  # retry once
                latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
                return self._refuse(
                    request_id, query, f"Output validation failed: {'; '.join(validation.errors)}",
                    "output_validation", latency_ms,
                )

            # ── Grounding check ────────────────────────────────────────────
            # Resolve the source tags cited by the model to their actual chunk
            # dicts, then verify the answer is supported by those specific chunks.
            # This is done against cited_chunks (what the model cited), NOT all
            # retrieved candidates — grounding is checked against what was cited.
            cited_chunks = [
                c for c in chunks
                if c["chunk_id"] in {
                    tag_to_chunk_id(chunks, tag)
                    for tag in (result.sources_used or [])
                }
            ]

            t0 = time.perf_counter()
            ground_result = grounding.check(result.answer, cited_chunks, threshold=self.grounding_threshold)
            mark(f"grounding_ms_attempt{attempt+1}", t0)

            if ground_result.passed:
                result.grounded = True
                result.refusal_reason = None
                break
            if attempt == 0:
                continue  # retry once
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, f"Grounding check failed: {ground_result.reason}", "grounding_check", latency_ms)

        latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000

        return RAGResponse(
            request_id=request_id,
            query=query,
            answer=result.answer,
            grounded=result.grounded,
            sources=[
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "score": c.get("score"),
                }
                for c in cited_chunks
            ],
            refusal_reason=result.refusal_reason,
            stage_reached="structured_response",
            latency_ms=latency_ms,
        )


def _resolve(chunks: list[dict], tag: str) -> dict | None:
    chunk_id = tag_to_chunk_id(chunks, tag)
    return next((c for c in chunks if c["chunk_id"] == chunk_id), None)


def main():
    parser = argparse.ArgumentParser(description="Full RAG pipeline smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider: gemini (default), groq, openai, ollama. Overrides LLM_PROVIDER env var.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model name override (provider-specific). Overrides LLM_MODEL env var.",
    )
    args = parser.parse_args()

    pipeline = RAGOrchestrator(
        strategy=args.strategy,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )
    response = pipeline.answer(args.query)

    print(f"\nRequest ID: {response.request_id}")
    print(f"Query: {response.query}")
    print(f"Stage reached: {response.stage_reached}")
    print(f"Answer: {response.answer}")
    print(f"Grounded: {response.grounded}")
    if response.refusal_reason:
        print(f"Refusal reason: {response.refusal_reason}")
    print(f"Sources: {[s['chunk_id'] for s in response.sources]}")
    print(f"\nLatency breakdown (ms):")
    for stage, ms in response.latency_ms.items():
        # Skip Groq-timing fields that are 0 (non-Groq providers)
        if isinstance(ms, float) and ms == 0.0 and "groq_" in stage:
            continue
        print(f"  {stage}: {ms:.2f}")


if __name__ == "__main__":
    main()