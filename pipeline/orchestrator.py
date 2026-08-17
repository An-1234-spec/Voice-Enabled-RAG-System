"""
pipeline/orchestrator.py

Full RAG pipeline, tying together everything built so far into one
callable entry point:

  validate_input -> safety_check -> relevance_check -> retrieve (hybrid+rerank)
  -> generate -> validate_output -> grounding_check -> retry (once) or
  fallback -> structured_response

Every stage timed; each call gets a request_id (per plan spec).

SCOPE NOTE: pipeline/router.py and pipeline/retry.py are NOT separate
files here - routing is effectively done by the guardrail chain itself
(safety -> "unsafe", relevance -> "out_of_domain", grounding -> low
confidence), and retry is a short inline loop, not enough logic to
justify a standalone module given the timeline. Split out later if useful.

KNOWN GUARDRAIL LIMITATIONS (documented earlier in the project, carried
into this pipeline as-is):
  - relevance.py has weak discrimination on this broad, multi-domain
    corpus (see guardrails/relevance.py --calibrate findings).
  - grounding.py is lexical overlap, not real NLI.
Both are real but imperfect signals, not guarantees.

LATENCY NOTE (added during optimization pass): LLM generation latency
is split into "our pipeline" time (generation_ms_attemptN, wall clock
around the API call) and Groq's own server-side breakdown
(groq_queue_ms / groq_prompt_ms / groq_completion_ms / groq_server_total_ms,
pulled from response.usage). This split exists because Groq deprecated
the fast non-reasoning model (llama-3.1-8b-instant) this project's
original <200ms target was designed around; the replacement
(openai/gpt-oss-20b) is a reasoning model with real per-request queue
time on top of compute. Reporting both numbers is more honest than
collapsing them into one figure - see README for the full breakdown.

Usage (CLI smoke test):
    python -m pipeline.orchestrator --query "what is a corporation?" --strategy fixed_token
"""

import argparse
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
        groq_model: str | None = None,
    ):
        self.strategy = strategy
        self.top_k = top_k

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.safety = SafetyGuardrail()
        self.relevance = RelevanceGuardrail(
            strategy=strategy, embeddings_dir=embeddings_dir, threshold=relevance_threshold, model=self.embed_model
        )
        self.retriever = HybridRetriever(
            strategy=strategy,faiss_dir=faiss_dir, chunks_dir=chunks_dir,
            dense_weight=0.9,bm25_weight=0.1,model=self.embed_model,
        )
        self.llm = GroqLLMClient(model=groq_model) if groq_model else GroqLLMClient()
        self.grounding_threshold = grounding_threshold

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

            # Pull Groq's own server-side timing breakdown into the same
            # latency dict, namespaced per attempt, so eval scripts can
            # separate "our wall-clock around the call" from "Groq queue +
            # compute" instead of only ever seeing one blended number.
            latency_ms[f"groq_queue_ms_attempt{attempt+1}"] = result.groq_queue_ms
            latency_ms[f"groq_prompt_ms_attempt{attempt+1}"] = result.groq_prompt_ms
            latency_ms[f"groq_completion_ms_attempt{attempt+1}"] = result.groq_completion_ms
            latency_ms[f"groq_server_total_ms_attempt{attempt+1}"] = result.groq_server_total_ms

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

            cited_chunks = chunks if result.grounded else []

            t0 = time.perf_counter()
            ground_result = grounding.check(result.answer, cited_chunks, threshold=self.grounding_threshold)
            mark(f"grounding_ms_attempt{attempt+1}", t0)

            if ground_result.passed:
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
    args = parser.parse_args()

    pipeline = RAGOrchestrator(strategy=args.strategy)
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
        print(f"  {stage}: {ms:.2f}")


if __name__ == "__main__":
    main()