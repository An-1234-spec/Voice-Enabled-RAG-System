"""
evaluation/rerank_tradeoff.py

Sweeps retrieve_n (candidates pulled by hybrid before reranking) across a
list of values, for one strategy, reporting BOTH retrieval quality
(recall@1/5/10, MRR) and reranking latency (P50/P70/P100) side by side -
so the tradeoff of shrinking the rerank candidate pool is visible in one
table, not two separate runs you have to cross-reference by hand.

Reuses:
  - evaluation.retrieval_eval.evaluate_strategy(mode="hybrid_rerank", top_n_raw=N)
    for quality - the exact same code path retrieval_eval.py/chunking_eval.py use.
  - retrieval.reranker.RerankedRetriever directly for latency timing - the
    same object evaluate_strategy builds internally, just also stopwatched.

Usage:
    python -m evaluation.rerank_tradeoff --strategy fixed_token \
        --candidates 10,15,20,30 --num-latency-queries 100 --max-queries 200
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from evaluation.retrieval_eval import evaluate_strategy
from retrieval.reranker import RerankedRetriever


def collect_queries(chunks_path: Path) -> list[str]:
    seen = set()
    queries = []
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


def percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def time_retriever(retriever, queries: list[str], top_k: int) -> list[float]:
    times = []
    for q in queries:
        t0 = time.perf_counter()
        retriever.search(q, top_k=top_k)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description="Sweep rerank candidate-pool size: quality vs latency tradeoff.")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--candidates", type=str, default="10,15,20,30", help="Comma-separated retrieve_n values to test")
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--num-latency-queries", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=None, help="Cap on quality-eval queries")
    parser.add_argument("--k-values", type=str, default="1,5,10")
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    candidate_values = [int(c) for c in args.candidates.split(",")]
    k_values = [int(k) for k in args.k_values.split(",")]
    chunks_path = args.chunks_dir / f"{args.strategy}.jsonl"

    print("Loading shared embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    distinct_queries = collect_queries(chunks_path)
    latency_queries = list(itertools.islice(itertools.cycle(distinct_queries), args.num_latency_queries))

    results = []
    for n in candidate_values:
        print(f"\nTesting retrieve_n={n}...")

        quality = evaluate_strategy(
            mode="hybrid_rerank",
            strategy=args.strategy,
            model=model,
            faiss_dir=args.faiss_dir,
            chunks_dir=args.chunks_dir,
            k_values=k_values,
            top_n_raw=n,
            max_queries=args.max_queries,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )

        retriever = RerankedRetriever(
            strategy=args.strategy,
            base_mode="hybrid",
            faiss_dir=args.faiss_dir,
            chunks_dir=args.chunks_dir,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            retrieve_n=n,
            model=model,
        )
        times_ms = time_retriever(retriever, latency_queries, top_k=args.final_k)

        results.append({
            "n": n,
            "recall@1": quality.get("recall@1", 0.0),
            "recall@5": quality.get("recall@5", 0.0),
            "recall@10": quality.get("recall@10", 0.0),
            "mrr": quality.get("mrr", 0.0),
            "p50_ms": percentile(times_ms, 50),
            "p70_ms": percentile(times_ms, 70),
            "p100_ms": percentile(times_ms, 100),
        })

    print("\n" + "=" * 90)
    print(f"RERANK CANDIDATE-POOL TRADEOFF - strategy='{args.strategy}'")
    print("=" * 90)
    header = f"{'retrieve_n':<12}{'R@1':<10}{'R@5':<10}{'R@10':<10}{'MRR':<10}{'P50ms':<10}{'P70ms':<10}{'P100ms':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['n']:<12}{r['recall@1']:<10.4f}{r['recall@5']:<10.4f}{r['recall@10']:<10.4f}"
            f"{r['mrr']:<10.4f}{r['p50_ms']:<10.2f}{r['p70_ms']:<10.2f}{r['p100_ms']:<10.2f}"
        )


if __name__ == "__main__":
    main()