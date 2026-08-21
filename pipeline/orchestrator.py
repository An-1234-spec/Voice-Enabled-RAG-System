"""
pipeline/orchestrator.py

Full RAG pipeline entry point:
  safety_check -> relevance_check -> retrieve (hybrid+rerank) -> generate
  -> resolve citations/grounding -> validate_output -> retry (once) or
  fallback -> structured_response

LLM_PROVIDER env var selects backend (default "gemini"); pass
llm_provider="ollama" explicitly to force local generation.

RETRY POLICY: retry is reserved for malformed output, empty response, or
an answer that fails grounding against the full retrieved set - never
triggered by missing citation tags alone when the answer is otherwise
grounded (see 2026-08-20 rewrite). Retry temperature is raised
(0.1 -> 0.6) on the second attempt so it's a genuine second sample, not a
near-deterministic repeat of the first.

INSTRUMENTATION: RAGResponse now carries grounding_overlap_ratio /
grounding_entity_ratio from whichever grounding.check() call last ran
(cited-chunks path or citation-fallback path), on BOTH success and
grounding-failure responses - lets a benchmark build the real
pass/fail score distribution instead of only seeing failure examples.
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

DEFAULT_RELEVANCE_THRESHOLD = -0.0613
DEFAULT_GROUNDING_THRESHOLD = 0.30  # matches guardrails/grounding.py's recalibrated default


@dataclass
class RAGResponse:
    request_id: str
    query: str
    answer: str
    grounded: bool
    sources: list[dict] = field(default_factory=list)
    refusal_reason: str | None = None
    stage_reached: str = "structured_response"
    latency_ms: dict = field(default_factory=dict)
    generation_attempts: int = 1
    retry_reason: str | None = None
    used_citation_fallback: bool = False
    grounding_overlap_ratio: float | None = None
    grounding_entity_ratio: float | None = None


def _build_llm(provider: str | None, model: str | None):
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER") or "gemini").lower()

    if resolved_provider == "ollama":
        from generation.ollama_llm import OllamaLLMClient
        resolved_model = model or os.environ.get("LLM_MODEL") or "llama3.2:1b"
        print(f"[LLM] Provider: ollama")
        print(f"[LLM] Model: {resolved_model}")
        return OllamaLLMClient(model=resolved_model)
    else:
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

        resolved_provider = (llm_provider or os.environ.get("LLM_PROVIDER") or "gemini").lower()
        self._report_groq_timing = (resolved_provider == "groq")

    def _refuse(self, request_id, query, reason, stage, latency_ms,
                attempts=1, retry_reason=None, overlap_ratio=None, entity_ratio=None) -> RAGResponse:
        return RAGResponse(
            request_id=request_id, query=query, answer="", grounded=False,
            refusal_reason=reason, stage_reached=stage, latency_ms=latency_ms,
            generation_attempts=attempts, retry_reason=retry_reason,
            grounding_overlap_ratio=overlap_ratio, grounding_entity_ratio=entity_ratio,
        )

    def answer(self, query: str) -> RAGResponse:
        request_id = str(uuid.uuid4())
        latency_ms: dict[str, float] = {}
        t_start = time.perf_counter()

        def mark(stage: str, t0: float):
            latency_ms[stage] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        safety_result = self.safety.check(query)
        mark("safety_ms", t0)
        if not safety_result.passed:
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, safety_result.reason, "safety_check", latency_ms)

        t0 = time.perf_counter()
        relevance_result = self.relevance.check(query)
        mark("relevance_ms", t0)
        if not relevance_result.passed:
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, relevance_result.reason, "relevance_check", latency_ms)

        t0 = time.perf_counter()
        chunks = self.retriever.search(query, top_k=self.top_k)
        mark("retrieval_ms", t0)
        if not chunks:
            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, "No relevant passages retrieved", "retrieval", latency_ms)

        result, cited_chunks = None, []
        retry_reason = "initial"
        used_citation_fallback = False
        attempts_made = 0
        grounding_overlap_last: float | None = None
        grounding_entity_last: float | None = None

        for attempt in range(2):
            attempts_made = attempt + 1
            print(f"GENERATION ATTEMPT {attempts_made}\n  reason: {retry_reason}")

            # Retry gets a real second sample, not a near-repeat of attempt 1.
            gen_temperature = 0.1 if attempt == 0 else 0.6

            t0 = time.perf_counter()
            result = self.llm.generate(query, chunks, temperature=gen_temperature)
            mark(f"generation_ms_attempt{attempts_made}", t0)

            if self._report_groq_timing:
                latency_ms[f"groq_queue_ms_attempt{attempts_made}"] = result.groq_queue_ms
                latency_ms[f"groq_prompt_ms_attempt{attempts_made}"] = result.groq_prompt_ms
                latency_ms[f"groq_completion_ms_attempt{attempts_made}"] = result.groq_completion_ms
                latency_ms[f"groq_server_total_ms_attempt{attempts_made}"] = result.groq_server_total_ms

            if result.refusal_reason == "pending_grounding_check" and result.answer:
                result.refusal_reason = None

            if not result.answer:
                if attempt == 0:
                    retry_reason = "empty_response"
                    continue
                latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
                return self._refuse(request_id, query, "Model returned an empty response",
                                     "generation", latency_ms, attempts_made, retry_reason)

            cited_chunks = [
                c for c in chunks
                if c["chunk_id"] in {tag_to_chunk_id(chunks, tag) for tag in (result.sources_used or [])}
            ]

            fallback_used_this_attempt = False
            if not cited_chunks:
                t0 = time.perf_counter()
                fallback_ground = grounding.check(result.answer, chunks, threshold=self.grounding_threshold)
                mark(f"grounding_fallback_ms_attempt{attempts_made}", t0)
                grounding_overlap_last = fallback_ground.overlap_ratio
                grounding_entity_last = fallback_ground.entity_ratio
                if fallback_ground.passed:
                    cited_chunks = chunks
                    result.sources_used = [c["chunk_id"] for c in cited_chunks]
                    result.grounded = True
                    result.refusal_reason = None
                    fallback_used_this_attempt = True
                    used_citation_fallback = True
                else:
                    result.grounded = False
                    result.refusal_reason = f"Grounding check failed: {fallback_ground.reason}"
            else:
                t0 = time.perf_counter()
                ground_result = grounding.check(result.answer, cited_chunks, threshold=self.grounding_threshold)
                mark(f"grounding_ms_attempt{attempts_made}", t0)
                grounding_overlap_last = ground_result.overlap_ratio
                grounding_entity_last = ground_result.entity_ratio
                if ground_result.passed:
                    result.grounded = True
                    result.refusal_reason = None
                else:
                    result.grounded = False
                    result.refusal_reason = f"Grounding check failed: {ground_result.reason}"

            t0 = time.perf_counter()
            validation = validate_output(result, chunks)
            mark(f"output_validation_ms_attempt{attempts_made}", t0)

            if not validation.valid:
                if attempt == 0:
                    retry_reason = f"malformed_output: {'; '.join(validation.errors)}"
                    continue
                latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
                return self._refuse(
                    request_id, query, f"Output validation failed: {'; '.join(validation.errors)}",
                    "output_validation", latency_ms, attempts_made, retry_reason,
                    grounding_overlap_last, grounding_entity_last,
                )

            if result.grounded:
                break

            if attempt == 0:
                retry_reason = f"ungrounded: {result.refusal_reason}"
                continue

            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, result.refusal_reason or "Answer failed grounding",
                                 "grounding_check", latency_ms, attempts_made, retry_reason,
                                 grounding_overlap_last, grounding_entity_last)

        latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000

        return RAGResponse(
            request_id=request_id,
            query=query,
            answer=result.answer,
            grounded=result.grounded,
            sources=[{"chunk_id": c["chunk_id"], "text": c["text"], "score": c.get("score")} for c in cited_chunks],
            refusal_reason=result.refusal_reason,
            stage_reached="structured_response",
            latency_ms=latency_ms,
            generation_attempts=attempts_made,
            retry_reason=retry_reason if attempts_made > 1 else None,
            used_citation_fallback=used_citation_fallback,
            grounding_overlap_ratio=grounding_overlap_last,
            grounding_entity_ratio=grounding_entity_last,
        )


def main():
    parser = argparse.ArgumentParser(description="Full RAG pipeline smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    args = parser.parse_args()

    pipeline = RAGOrchestrator(strategy=args.strategy, llm_provider=args.llm_provider, llm_model=args.llm_model)
    response = pipeline.answer(args.query)

    print(f"\nAnswer: {response.answer}")
    print(f"Grounded: {response.grounded}")
    print(f"Overlap ratio: {response.grounding_overlap_ratio}  Entity ratio: {response.grounding_entity_ratio}")
    if response.refusal_reason:
        print(f"Refusal reason: {response.refusal_reason}")


if __name__ == "__main__":
    main()