"""
scripts/diagnose_latency.py

Diagnoses WHERE the 754ms in your gpt-oss-20b generation call is actually
going, using your real generation.llm.GroqLLMClient (not a reimplementation)
so the numbers are trustworthy. Two things this checks:

1. Server-side breakdown (queue_time vs prompt_time vs completion_time) for
   your current model — tells us if it's Groq queueing you, processing a
   long prompt, or actually generating (likely: reasoning tokens).
2. Same query against allam-2-7b, a non-reasoning model already on your
   Groq account — if reasoning is the real bottleneck, this should look
   dramatically different from gpt-oss-20b even though it's the same
   provider/infra, which tells us whether to fix the model choice
   (cheap, in-place) vs. migrate providers (bigger, riskier change).

Run from your project root:
    python -m scripts.diagnose_latency --query "what is a corporation?" --strategy fixed_token
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generation.llm import GroqLLMClient
from retrieval.reranker import RerankedRetriever
from sentence_transformers import SentenceTransformer

CANDIDATES = [
    ("openai/gpt-oss-20b", "low"),   # your current setup, for baseline
    ("openai/gpt-oss-20b", "none"),  # try disabling reasoning entirely, if Groq supports it
    ("allam-2-7b", None),            # non-reasoning model — reasoning_effort likely unsupported/ignored
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-n", type=int, default=10)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    print("Loading embedding model + retriever (once, reused for all candidates)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    retriever = RerankedRetriever(
        strategy=args.strategy,
        base_mode="hybrid",
        faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir,
        retrieve_n=args.retrieve_n,
        model=embed_model,
    )
    chunks = retriever.search(args.query, top_k=args.top_k)
    print(f"Retrieved {len(chunks)} chunks for: {args.query}\n")

    results = []
    for model, reasoning_effort in CANDIDATES:
        label = f"{model}" + (f" (reasoning_effort={reasoning_effort})" if reasoning_effort else "")
        print(f"--- {label} ---")
        try:
            client = GroqLLMClient(model=model)
            kwargs = {}
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            result = client.generate(args.query, chunks, **kwargs)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n")
            continue

        print(f"  queue_ms      : {result.groq_queue_ms:.1f}")
        print(f"  prompt_ms     : {result.groq_prompt_ms:.1f}")
        print(f"  completion_ms : {result.groq_completion_ms:.1f}  <- this is the reasoning+answer cost")
        print(f"  server_total  : {result.groq_server_total_ms:.1f}")
        print(f"  parse_error   : {result.parse_error}")
        print(f"  answer        : {result.answer[:120]}")
        print()
        results.append((label, result))

    if len(results) > 1:
        print("=== Summary (sorted by completion_ms — the lever we actually control) ===")
        for label, r in sorted(results, key=lambda x: x[1].groq_completion_ms):
            flag = "  ⚠ parse_error" if r.parse_error else ""
            print(f"  completion={r.groq_completion_ms:>7.1f}ms  total={r.groq_server_total_ms:>7.1f}ms  {label}{flag}")


if __name__ == "__main__":
    main()