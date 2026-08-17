"""
Compare Groq models using the actual GroqLLMClient.

Run:
    python -m scripts.compare_models
"""

from __future__ import annotations

import time

from generation.llm import GroqLLMClient


CANDIDATE_MODELS = [
    "openai/gpt-oss-20b",
    "allam-2-7b",
]

QUERY = "what is a corporation?"

CONTEXTS = [
    {
        "index": 1,
        "text": (
            "Corporation definition, an association of individuals, "
            "created by law or under authority of law, having a continuous "
            "existence independent of the existences of its members."
        ),
        "passage_id": "S1",
        "score": 0.91,
    },
    {
        "index": 2,
        "text": (
            "McDonald's Corporation is one of the most recognizable "
            "corporations in the world."
        ),
        "passage_id": "S2",
        "score": 0.85,
    },
    {
        "index": 3,
        "text": (
            "A company is incorporated in a specific nation, often within "
            "the bounds of a smaller subdivision."
        ),
        "passage_id": "S3",
        "score": 0.80,
    },
    {
        "index": 4,
        "text": (
            "Corporations are owned by their stockholders (shareholders) "
            "who share in profits and losses."
        ),
        "passage_id": "S4",
        "score": 0.77,
    },
]


def main():
    print("=" * 70)
    print("GROQ MODEL LATENCY COMPARISON")
    print("=" * 70)
    print(f"Query: {QUERY}\n")

    results = []

    for model in CANDIDATE_MODELS:
        print(f"--- Testing {model} ---")

        try:
            client = GroqLLMClient(model=model)

            start = time.perf_counter()

            result = client.generate(
                QUERY,
                CONTEXTS,
                temperature=0.0,
                max_tokens=80,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"  latency_ms  : {elapsed_ms:.2f}")
            print(f"  parse_error : {result.parse_error}")
            print(f"  grounded    : {result.grounded}")
            print(f"  answer      : {result.answer[:200]}")
            print()

            results.append(
                {
                    "model": model,
                    "latency": elapsed_ms,
                    "parse_error": result.parse_error,
                }
            )

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print()

    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        results.sort(key=lambda x: x["latency"])

        for r in results:
            status = "OK" if not r["parse_error"] else "PARSE ERROR"
            print(
                f"{r['latency']:>8.2f} ms  "
                f"{r['model']:<30} {status}"
            )


if __name__ == "__main__":
    main()