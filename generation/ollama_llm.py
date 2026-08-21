"""
generation/ollama_llm.py

Local LLM client using Ollama's REST API, optimized for latency:
  - persistent requests.Session() (avoids per-call TCP handshake)
  - keep_alive set so the model stays resident in VRAM between calls
  - 127.0.0.1 not "localhost" (avoids Windows IPv6-then-fallback DNS delay)
  - trimmed num_ctx, sized to actually fit prompt + answer

LATENCY NOTE (post-debugging): `ollama ps` confirms the model stays
resident (keep_alive is working correctly), but each call still shows
~290-320ms of "load" time in the timing breakdown even when warm. That
appears to be fixed per-request runner/context overhead on this machine,
not a reload. This is a measured floor, not something further keep_alive
tuning fixes.

MAX_TOKENS NOTE (revised): previously 180, which was itself a fix for an
earlier bug (24 was too small and caused 100% refusal). 180 is now known
to be far more than needed — the task requires ~1 short sentence, and a
48-token budget was chosen as a starting point pending experimental
validation via evaluation/latency_eval.py (run with --max-tokens sweep
before trusting this value in a submission).

GROUNDING NOTE (2026-08-20 fix): this client used to run its own
grounding.check() internally AND the orchestrator ran a second,
independently-resolved grounding.check() on top of it. The first result
was silently discarded, and the two resolution paths (parse_cited_answer
here vs tag_to_chunk_id in the orchestrator) could disagree, causing
retries on answers that were actually fine. This client now does NOT run
grounding at all — it returns refusal_reason="pending_grounding_check"
as a sentinel, matching the convention already used by the orchestrator's
non-Ollama providers (see orchestrator.py `pending_grounding` handling).
Grounding is the orchestrator's job, exactly once, per attempt.

Construct ONE OllamaLLMClient per process/server and reuse it for every
request - constructing a new one per call defeats both the session reuse
and keep_alive benefits.
"""

import argparse
import time
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

from generation.llm import LLMResult
from generation.prompts import build_messages, parse_cited_answer
from retrieval.reranker import RerankedRetriever

DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_HOST = "http://127.0.0.1:11434"  # NOT "localhost" - avoids Windows DNS delay

# Was 180 (fix for an earlier 24-token truncation bug). Cut to 48 for a
# latency-critical 1-sentence-answer target. VALIDATE EXPERIMENTALLY before
# trusting this in a submission — sweep 32/48/64 against grounding pass rate.
DEFAULT_MAX_TOKENS = 32

# Was 768 (headroom for a 180-token output). Prompt runs ~228 tok per the
# original measurement; 384 leaves ~150 tok of headroom for a 48-tok answer.
# Re-verify against actual prompt token count from generation/prompts.py
# once that file's exact system prompt is known.
DEFAULT_NUM_CTX = 384


class OllamaLLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: int = 60,
        keep_alive: str = "30m",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.session = requests.Session()  # reused across all calls - real connection reuse
        self.last_timing: dict = {}
        self.last_raw_text: str = ""

    def generate(
        self,
        query: str,
        chunks: list[dict],
        temperature: float = 0.05,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> LLMResult:
        """
        Calls Ollama and parses the response. Does NOT run grounding —
        that is the orchestrator's responsibility (see module docstring).
        Returns refusal_reason="pending_grounding_check" as a sentinel
        when there's a usable answer, so the orchestrator knows to run
        grounding exactly once.
        """
        messages = build_messages(query, chunks)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
            },
        }

        try:
            response = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Could not reach Ollama at {self.host}. Is `ollama serve` running? {e}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}") from e
            raise

        resp_json = response.json()
        raw_text = resp_json["message"]["content"]
        self.last_raw_text = raw_text

        self.last_timing = {
            "total_ms": resp_json.get("total_duration", 0) / 1e6,
            "load_ms": resp_json.get("load_duration", 0) / 1e6,
            "prompt_eval_ms": resp_json.get("prompt_eval_duration", 0) / 1e6,
            "prompt_tokens": resp_json.get("prompt_eval_count", 0),
            "eval_ms": resp_json.get("eval_duration", 0) / 1e6,
            "eval_tokens": resp_json.get("eval_count", 0),
        }

        answer, tags, _cited_chunks_unused = parse_cited_answer(raw_text, chunks)

        if not answer:
            return LLMResult(answer="", grounded=False, refusal_reason="empty_response")

        # Sentinel: answer exists, grounding not yet checked. The orchestrator
        # treats this as "tentatively grounded, verify before trusting."
        # sources_used may be empty if the model omitted citation tags — the
        # orchestrator's Python-side fallback handles that case without a retry.
        return LLMResult(
            answer=answer,
            grounded=False,
            sources_used=tags,
            refusal_reason="pending_grounding_check",
        )


def main():
    parser = argparse.ArgumentParser(description="Ollama RAG generation smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=2)
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
    llm.generate(args.query, chunks, max_tokens=args.max_tokens)  # warmup, not timed

    t0 = time.perf_counter()
    result = llm.generate(args.query, chunks, max_tokens=args.max_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    t = llm.last_timing
    print(
        f"Ollama breakdown: load={t['load_ms']:.0f}ms  "
        f"prompt_eval={t['prompt_eval_ms']:.0f}ms ({t['prompt_tokens']} tok)  "
        f"generate={t['eval_ms']:.0f}ms ({t['eval_tokens']} tok)"
    )
    print(f"Answer: {result.answer}")
    print(f"Sources used (model-cited, may be empty): {result.sources_used}")
    print(f"Refusal reason (sentinel expected pre-grounding): {result.refusal_reason}")
    print(f"[debug] raw output: {llm.last_raw_text}")
    print(f"Wall-clock (warm): {elapsed_ms:.2f} ms")
    print("\nNOTE: this smoke test does NOT run grounding — that now happens")
    print("only in pipeline/orchestrator.py. Run the full pipeline to see")
    print("Grounded=True/False.")


if __name__ == "__main__":
    main()