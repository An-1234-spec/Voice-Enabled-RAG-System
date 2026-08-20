"""
evaluation/full_pipeline_latency_eval.py

Full end-to-end pipeline latency benchmark: safety -> relevance ->
retrieval (hybrid+rerank) -> generation (Ollama) -> output validation ->
grounding, timed as one real user-facing request. One retry max, only on
genuine output-validation or grounding failure (per spec: no unnecessary
retries). Warmup call excluded from stats.

ASSUMPTION: uses guardrails.safety/relevance/output_validator/grounding
as originally built. If your project's versions of these have diverged,
timings for those stages may not reflect your actual current code -
paste them if so and this script gets adjusted to match exactly.

Usage:
    python -m evaluation.full_pipeline_latency_eval --strategy fixed_token \
        --num-queries 200 --top-k 2 --max-tokens 24
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from guardrails.safety import SafetyGuardrail
from guardrails.relevance import RelevanceGuardrail
from guardrails.output_validator import validate as validate_output
from guardrails import grounding
from generation.ollama_llm import OllamaLLMClient
from retrieval.reranker import RerankedRetriever

RELEVANCE_THRESHOLD = -0.0613  # from earlier calibration


def collect_queries(chunks_path: Path) -> list[str]:
    seen, queries = set(), []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("query")
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline warm latency benchmark.")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--retrieve-n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--model", type=str, default="llama3.2:1b")
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/processed/embeddings"))
    parser.add_argument("--baseline-p50", type=float, default=None, help="Prior P50 in ms, for the improvement report")
    args = parser.parse_args()

    chunks_path = args.chunks_dir / f"{args.strategy}.jsonl"

    print("Loading model + building pipeline (once, reused for all queries)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    safety = SafetyGuardrail()
    relevance = RelevanceGuardrail(strategy=args.strategy, embeddings_dir=args.embeddings_dir, threshold=RELEVANCE_THRESHOLD, model=embed_model)
    retriever = RerankedRetriever(strategy=args.strategy, base_mode="hybrid", faiss_dir=args.faiss_dir, chunks_dir=args.chunks_dir, retrieve_n=args.retrieve_n, model=embed_model)
    llm = OllamaLLMClient(model=args.model)

    distinct_queries = collect_queries(chunks_path)
    queries = list(itertools.islice(itertools.cycle(distinct_queries), args.num_queries))

    print("Warmup call (not counted)...")
    warmup_chunks = retriever.search(queries[0], top_k=args.top_k)
    llm.generate(queries[0], warmup_chunks, max_tokens=args.max_tokens)

    stage_times = {"safety_ms": [], "relevance_ms": [], "retrieval_ms": [], "generation_ms": [], "validation_ms": [], "total_ms": []}
    success_count, refusal_count, retry_count = 0, 0, 0

    print(f"Running {len(queries)} timed queries...")
    for i, query in enumerate(queries, 1):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        safety_result = safety.check(query)
        stage_times["safety_ms"].append((time.perf_counter() - t0) * 1000)
        if not safety_result.passed:
            refusal_count += 1
            stage_times["total_ms"].append((time.perf_counter() - t_start) * 1000)
            continue

        t0 = time.perf_counter()
        relevance_result = relevance.check(query)
        stage_times["relevance_ms"].append((time.perf_counter() - t0) * 1000)
        if not relevance_result.passed:
            refusal_count += 1
            stage_times["total_ms"].append((time.perf_counter() - t_start) * 1000)
            continue

        t0 = time.perf_counter()
        chunks = retriever.search(query, top_k=args.top_k)
        stage_times["retrieval_ms"].append((time.perf_counter() - t0) * 1000)

        result, attempt = None, 0
        for attempt in range(2):
            t0 = time.perf_counter()
            result = llm.generate(query, chunks, max_tokens=args.max_tokens)
            gen_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            validation = validate_output(result, chunks)
            val_ms = (time.perf_counter() - t0) * 1000

            if validation.valid and result.grounded:
                break
            if attempt == 0:
                retry_count += 1
                continue
            break  # second attempt also failed - accept as refusal below

        stage_times["generation_ms"].append(gen_ms)
        stage_times["validation_ms"].append(val_ms)

        if result and result.grounded and validation.valid:
            success_count += 1
        else:
            refusal_count += 1

        stage_times["total_ms"].append((time.perf_counter() - t_start) * 1000)

        if i % 25 == 0:
            print(f"  ...{i}/{len(queries)}")

    print("\n" + "=" * 70)
    print(f"FULL PIPELINE LATENCY - N={len(queries)}, model={args.model}, top_k={args.top_k}, max_tokens={args.max_tokens}")
    print("=" * 70)
    print(f"{'Stage':<20}{'P50':<10}{'P70':<10}{'P100':<10}{'Mean':<10}")
    for stage, vals in stage_times.items():
        if vals:
            print(f"{stage:<20}{percentile(vals,50):<10.2f}{percentile(vals,70):<10.2f}{percentile(vals,100):<10.2f}{np.mean(vals):<10.2f}")

    print(f"\nSuccess: {success_count}  Refusals: {refusal_count}  Retries: {retry_count}")

    total = stage_times["total_ms"]
    p50, p70, p100 = percentile(total, 50), percentile(total, 70), percentile(total, 100)
    print(f"\nP50={p50:.2f}ms  P70={p70:.2f}ms  P100={p100:.2f}ms")
    print(f"Target: P50<=150ms P70<=175ms P100<200ms -> "
          f"{'MET' if p50<=150 and p70<=175 and p100<200 else 'NOT MET'}")

    if args.baseline_p50:
        print(f"\nBASELINE P50: {args.baseline_p50:.2f}ms  ->  AFTER: {p50:.2f}ms  "
              f"(reduction: {args.baseline_p50 - p50:.2f}ms, {100*(1-p50/args.baseline_p50):.1f}%)")


if __name__ == "__main__":
    main()