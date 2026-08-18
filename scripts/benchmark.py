"""
scripts/benchmark.py

Combined benchmark runner — invokes chunking, retrieval, and latency
evaluations in sequence and produces a formatted report.

Orchestrates the existing evaluation scripts rather than reimplementing:
  - chunking/benchmark.py       → chunk count, token distribution
  - evaluation/retrieval_eval.py → Recall@k, Precision@k, MRR
  - evaluation/latency_eval.py  → per-stage P50/P70/P100 timing

Usage:
    python scripts/benchmark.py --strategy fixed_token --max-queries 50
    python scripts/benchmark.py --all-strategies --max-queries 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_chunking_benchmark(strategies, data_path, chunk_size):
    """Run chunking benchmark and print results."""
    print("\n" + "=" * 70)
    print("  CHUNKING BENCHMARK")
    print("=" * 70)

    from chunking.benchmark import load_records, run_benchmark, _print_table

    records = load_records(
        str(data_path) if data_path.exists() else None,
        limit=None,
        text_field="text",
        passage_id_field="passage_id",
        document_id_field="document_id",
    )
    if not records:
        print("  No records to benchmark.")
        return

    results = run_benchmark(records, chunk_size=chunk_size, include_semantic=True)
    print()
    _print_table(results)


def run_retrieval_eval(strategies, max_queries, faiss_dir, chunks_dir):
    """Run retrieval evaluation and print results."""
    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    from evaluation.retrieval_eval import evaluate_strategy, print_mode_summary

    model = SentenceTransformer("all-MiniLM-L6-v2")
    k_values = [1, 5, 10]
    modes = ["dense", "bm25", "hybrid"]

    for mode in modes:
        mode_results = {}
        for strategy in strategies:
            print(f"  Evaluating '{strategy}' (mode={mode})...")
            try:
                scores = evaluate_strategy(
                    mode=mode, strategy=strategy, model=model,
                    faiss_dir=faiss_dir, chunks_dir=chunks_dir,
                    k_values=k_values, top_n_raw=30,
                    max_queries=max_queries, dense_weight=0.6, bm25_weight=0.4,
                )
                mode_results[strategy] = scores
            except Exception as e:
                print(f"    [error] {e}")
                mode_results[strategy] = {}
        print_mode_summary(mode, mode_results, k_values)


def main():
    parser = argparse.ArgumentParser(description="Combined benchmark runner.")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--all-strategies", action="store_true", help="Run across all 5 strategies")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--passages-path", type=Path, default=Path("data/processed/passages.jsonl"))
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    all_strategies = ["fixed_token", "passage", "sentence", "parent_child", "semantic"]
    strategies = all_strategies if args.all_strategies else [args.strategy]

    print("=" * 70)
    print("  VOICE-ENABLED RAG — COMBINED BENCHMARK")
    print("=" * 70)
    print(f"  Strategies: {strategies}")
    print(f"  Max queries: {args.max_queries}")

    t_start = time.perf_counter()

    # 1. Chunking benchmark
    run_chunking_benchmark(strategies, args.passages_path, args.chunk_size)

    # 2. Retrieval evaluation
    run_retrieval_eval(strategies, args.max_queries, args.faiss_dir, args.chunks_dir)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Total benchmark time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
