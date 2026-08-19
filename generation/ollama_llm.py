"""
generation/ollama_llm.py

Local LLM client using Ollama's REST API.  Uses plain-text + citation-tag
output (generation/prompts.py), NOT JSON — see that module's docstring
for why.  Reports Ollama's internal load/prompt_eval/eval breakdown via
self.last_timing.

GROUNDING IS THE ORCHESTRATOR'S JOB:
This client no longer calls guardrails.grounding.check() internally.
Grounding is a pipeline-stage concern owned by pipeline/orchestrator.py,
which runs it after generation and retries on failure.  The client's
responsibility is limited to:
  1. Calling Ollama
  2. Parsing the plain-text answer + citation tags
  3. Returning a well-formed LLMResult

refusal_reason is set to a non-None string whenever grounded=False, so
guardrails/output_validator.py's consistency check always passes.

REQUIRES: Ollama running locally, model already pulled.

Usage:
    python -m generation.ollama_llm --query "what is a corporation?" --strategy fixed_token
"""

import argparse
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

from generation.llm import LLMResult
from generation.prompts import build_messages, parse_cited_answer
from retrieval.reranker import RerankedRetriever

DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MAX_TOKENS = 60  # short answer + a few citation tags, per Phase C target


class OllamaLLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.last_timing: dict = {}

    def generate(
        self,
        query: str,
        chunks: list[dict],
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResult:
        """
        Call Ollama and parse the answer.

        Returns LLMResult with:
          - answer: the clean answer string (empty string on refusal)
          - grounded: False — the orchestrator computes the real grounding
            value against cited chunks; this client has no way to know which
            chunks will survive the orchestrator's resolution step
          - sources_used: list of tag strings like ["S1", "S3"]
          - refusal_reason: set whenever the model indicates it cannot answer
            (e.g. [NONE] signal) or when the HTTP call fails
        """
        messages = build_messages(query, chunks)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # NOTE: no "format": "json" - plain text output, see prompts.py docstring
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason=f"Could not reach Ollama at {self.host}. Is `ollama serve` running?",
                parse_error=False,
            )
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                return LLMResult(
                    answer="",
                    grounded=False,
                    refusal_reason=f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}",
                    parse_error=False,
                )
            return LLMResult(
                answer="",
                grounded=False,
                refusal_reason=f"Ollama HTTP error: {e}",
                parse_error=False,
            )

        resp_json = response.json()
        raw_text = resp_json.get("message", {}).get("content", "")

        self.last_timing = {
            "total_ms": resp_json.get("total_duration", 0) / 1e6,
            "load_ms": resp_json.get("load_duration", 0) / 1e6,
            "prompt_eval_ms": resp_json.get("prompt_eval_duration", 0) / 1e6,
            "prompt_tokens": resp_json.get("prompt_eval_count", 0),
            "eval_ms": resp_json.get("eval_duration", 0) / 1e6,
            "eval_tokens": resp_json.get("eval_count", 0),
        }

        answer, tags, _cited_chunks = parse_cited_answer(raw_text, chunks)

        if not answer:
            # Model indicated [NONE] or returned nothing usable
            return LLMResult(
                answer="",
                grounded=False,
                sources_used=[],
                refusal_reason="Model indicated insufficient information to answer",
                parse_error=False,
            )

        # grounded=False here is intentional — the orchestrator will run its own
        # grounding check against the cited chunks and set the real value.
        # refusal_reason must be non-None when grounded=False (output_validator
        # consistency requirement), so we use a sentinel that the orchestrator
        # will overwrite once grounding passes.
        return LLMResult(
            answer=answer,
            grounded=False,
            sources_used=tags,
            refusal_reason="pending_grounding_check",
            parse_error=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Ollama RAG generation smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=3)  # Phase C: test top-k 2-3
    parser.add_argument("--retrieve-n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    retriever = RerankedRetriever(
        strategy=args.strategy, base_mode="hybrid", faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir, retrieve_n=args.retrieve_n, model=embed_model,
    )
    chunks = retriever.search(args.query, top_k=args.top_k)

    llm = OllamaLLMClient(model=args.model)
    t0 = time.perf_counter()
    result = llm.generate(args.query, chunks, max_tokens=args.max_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    t = llm.last_timing
    print(
        f"Ollama breakdown: load={t.get('load_ms',0):.0f}ms  "
        f"prompt_eval={t.get('prompt_eval_ms',0):.0f}ms ({t.get('prompt_tokens',0)} tok)  "
        f"generate={t.get('eval_ms',0):.0f}ms ({t.get('eval_tokens',0)} tok)"
    )
    print(f"Answer: {result.answer}")
    print(f"Sources used: {result.sources_used}")
    print(f"Refusal reason: {result.refusal_reason}")
    print(f"Wall-clock: {elapsed_ms:.2f} ms")
    print("(grounded value is computed by the orchestrator's grounding stage, not here)")


if __name__ == "__main__":
    main()