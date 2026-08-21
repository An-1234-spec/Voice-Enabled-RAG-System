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
confidence), and retry is a short inline loop.

KNOWN GUARDRAIL LIMITATIONS (documented earlier in the project, carried
into this pipeline as-is):
  - relevance.py has weak discrimination on this broad, multi-domain
    corpus (see guardrails/relevance.py --calibrate findings).
  - grounding.py uses lexical + entity + sentence-level overlap, not real NLI.
Both are real but imperfect signals, not guarantees.

RETRY POLICY (2026-08-20 rewrite):
  Previously, ANY missing/mismatched citation tag caused a grounding
  failure and a full second LLM generation call, because grounding was
  checked against `cited_chunks` resolved strictly from the model's own
  [S1]/[S2] tags. A model that wrote a correct, fully-grounded answer but
  forgot the tag got treated identically to a hallucination — this was
  the dominant cause of the previously-measured 52% retry rate.

  Fix: if the model gives no usable citation tags, Python checks the
  answer against ALL retrieved chunks (not zero chunks) before deciding
  to retry. If it grounds against the retrieved set, Python assigns the
  supporting chunk_ids as sources itself — no second LLM call. Grounding
  is NOT weakened: the answer still has to actually pass grounding.check()
  against real chunk text, it just isn't required to have gone through
  perfect tag formatting first.

  A retry is now reserved for:
    - malformed/invalid output structure (validate_output fails)
    - an answer that fails grounding even against the full retrieved set
    - an empty response from the LLM
  A retry is NEVER triggered by missing/malformed citation tags alone
  when the underlying answer is actually grounded.

  Every attempt is logged with an explicit reason (see `retry_reason`).

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

# FIX (2026-08-21): this used to hardcode 0.5 here, completely independent
# of guardrails/grounding.py's own DEFAULT_OVERLAP_THRESHOLD. When that
# module was evidence-recalibrated 0.50 -> 0.30 (see its docstring), this
# orchestrator kept passing the stale 0.5 into every grounding.check()
# call, silently overriding the recalibration. A live eval run showed the
# exact fingerprint of this bug: every failure message printed "threshold
# 50%" even though grounding.py's own default had already moved to 30%.
# Importing the module's constant directly means the two files can never
# drift apart again — there is now exactly one place this number lives.
DEFAULT_GROUNDING_THRESHOLD = grounding.DEFAULT_OVERLAP_THRESHOLD


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
    # Diagnostic fields — not required by the API contract, but requested
    # by the latency-refactor spec so retries/attempts are auditable.
    generation_attempts: int = 1
    retry_reason: str | None = None
    used_citation_fallback: bool = False


def _build_llm(provider: str | None, model: str | None):
    """
    Factory for selecting the LLM backend.
    """

    resolved_provider = (
        provider
        or os.environ.get("LLM_PROVIDER")
        or "gemini"
    ).lower()

    if resolved_provider == "ollama":
        from generation.ollama_llm import OllamaLLMClient

        resolved_model = (
            model
            or os.environ.get("LLM_MODEL")
            or "llama3.2:1b"
        )

        print(f"[LLM] Provider: ollama")
        print(f"[LLM] Model: {resolved_model}")

        return OllamaLLMClient(model=resolved_model)

    else:
        return GroqLLMClient(
            provider=resolved_provider,
            model=model,
        )


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

        # Loaded ONCE per orchestrator instance. Caller (app/api.py) MUST
        # construct exactly one RAGOrchestrator per server process and
        # reuse it across requests — constructing one per request would
        # reload SentenceTransformer, FAISS, and the LLM client every
        # single call. This class does not defend against that itself;
        # it's a startup-wiring responsibility, not the orchestrator's.
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

    def _refuse(self, request_id: str, query: str, reason: str, stage: str, latency_ms: dict,
                attempts: int = 1, retry_reason: str | None = None) -> RAGResponse:
        return RAGResponse(
            request_id=request_id, query=query, answer="", grounded=False,
            refusal_reason=reason, stage_reached=stage, latency_ms=latency_ms,
            generation_attempts=attempts, retry_reason=retry_reason,
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

        # 4. Generation, with at most ONE retry — and only for genuine
        #    failures (malformed output, empty response, or an answer that
        #    fails grounding against the FULL retrieved set). Missing
        #    citation tags alone are resolved in Python below and never
        #    cause a retry on their own.
        result, cited_chunks = None, []
        retry_reason = "initial"
        used_citation_fallback = False
        attempts_made = 0

        for attempt in range(2):
            attempts_made = attempt + 1
            print(f"GENERATION ATTEMPT {attempts_made}\n  reason: {retry_reason}")

            t0 = time.perf_counter()
            result = self.llm.generate(query, chunks)
            mark(f"generation_ms_attempt{attempts_made}", t0)

            # Pull Groq's own server-side timing breakdown when using Groq.
            if self._report_groq_timing:
                latency_ms[f"groq_queue_ms_attempt{attempts_made}"] = result.groq_queue_ms
                latency_ms[f"groq_prompt_ms_attempt{attempts_made}"] = result.groq_prompt_ms
                latency_ms[f"groq_completion_ms_attempt{attempts_made}"] = result.groq_completion_ms
                latency_ms[f"groq_server_total_ms_attempt{attempts_made}"] = result.groq_server_total_ms

            # ── Sentinel handling ──────────────────────────────────────────
            # Providers that haven't run their own grounding set
            # refusal_reason="pending_grounding_check" when they have a
            # usable answer. We clear that here so output_validator sees a
            # clean answer, and grounding is decided exactly once, below,
            # by this orchestrator — never inside the LLM client itself.
            pending_grounding = (
                result.refusal_reason == "pending_grounding_check"
                and bool(result.answer)
            )
            if pending_grounding:
                # Just clear the sentinel here. Do NOT set grounded yet —
                # setting a tentative True before we've resolved citations
                # is exactly what caused the "grounded=true but sources_used
                # is empty" validation failures: validate_output ran BEFORE
                # the fallback below got a chance to populate sources_used.
                # grounded/refusal_reason are decided once, definitively,
                # below — by the time validate_output runs they will
                # already be in a mutually consistent final state.
                result.refusal_reason = None

            if not result.answer:
                if attempt == 0:
                    retry_reason = "empty_response"
                    continue
                latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
                return self._refuse(request_id, query, "Model returned an empty response",
                                     "generation", latency_ms, attempts_made, retry_reason)

            # ── Resolve cited chunks from the model's own tags (may be empty) ──
            cited_chunks = [
                c for c in chunks
                if c["chunk_id"] in {
                    tag_to_chunk_id(chunks, tag)
                    for tag in (result.sources_used or [])
                }
            ]

            # ── Grounding, resolved fully before validation ──────────────────
            # Two paths, but exactly one grounding.check() call either way:
            #   - model gave usable tags -> check against those cited chunks
            #   - model gave no tags     -> Python-side fallback: check
            #     against the FULL retrieved set instead of failing outright
            #     (grounding.check() hard-fails on an empty chunk list by
            #     design — see guardrails/grounding.py — so this fallback
            #     deliberately supplies `chunks`, not [], to give a citation-
            #     less-but-correct answer a fair chance without a retry).
            # Either way, grounded/sources_used/refusal_reason are ALL set
            # together, consistently, before output_validation ever runs.
            fallback_used_this_attempt = False
            if cited_chunks:
                t0 = time.perf_counter()
                ground_result = grounding.check(result.answer, cited_chunks, threshold=self.grounding_threshold)
                mark(f"grounding_ms_attempt{attempts_made}", t0)
            else:
                t0 = time.perf_counter()
                ground_result = grounding.check(result.answer, chunks, threshold=self.grounding_threshold)
                mark(f"grounding_fallback_ms_attempt{attempts_made}", t0)
                if ground_result.passed:
                    cited_chunks = chunks
                    fallback_used_this_attempt = True
                    used_citation_fallback = True

            if ground_result.passed:
                result.grounded = True
                result.refusal_reason = None
                if fallback_used_this_attempt:
                    # CONFIRMED via debug output (2026-08-21): tag_to_chunk_id
                    # only resolves tag-format strings ("S1", "S2", ...) --
                    # even a raw string that exactly equals a real chunk_id
                    # failed to resolve. Synthesize tags in that format,
                    # 1-indexed by position in `chunks`, matching the
                    # convention already used by the model's own citations
                    # and displayed in the frontend (S1/S2/S3 labels).
                    result.sources_used = [f"S{i+1}" for i in range(len(chunks))]
                # else: tags path — result.sources_used already holds the
                # model's own tags (e.g. ["S1","S3"]), which already resolve
                # correctly. Overwriting these with chunk_ids was the actual
                # bug: it fed tag_to_chunk_id a format it can't resolve,
                # so output_validator's own "drop unresolvable tags" step
                # silently wiped sources_used back to empty every time.
            else:
                result.grounded = False
                result.refusal_reason = ground_result.reason or "grounding check failed"

            # ── Output validation — runs LAST, on the final resolved state ──
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
                )

            if ground_result.passed:
                break

            if attempt == 0:
                retry_reason = f"ungrounded: {ground_result.reason}"
                continue

            latency_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            return self._refuse(request_id, query, f"Grounding check failed: {ground_result.reason}",
                                 "grounding_check", latency_ms, attempts_made, retry_reason)

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
            generation_attempts=attempts_made,
            retry_reason=retry_reason if attempts_made > 1 else None,
            used_citation_fallback=used_citation_fallback,
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
    print(f"Generation attempts: {response.generation_attempts}"
          + (f" (retry reason: {response.retry_reason})" if response.retry_reason else ""))
    print(f"Used citation fallback (Python-assigned sources): {response.used_citation_fallback}")
    if response.refusal_reason:
        print(f"Refusal reason: {response.refusal_reason}")
    print(f"Sources: {[s['chunk_id'] for s in response.sources]}")
    print(f"\nLatency breakdown (ms):")
    for stage, ms in response.latency_ms.items():
        if isinstance(ms, float) and ms == 0.0 and "groq_" in stage:
            continue
        print(f"  {stage}: {ms:.2f}")


if __name__ == "__main__":
    main()