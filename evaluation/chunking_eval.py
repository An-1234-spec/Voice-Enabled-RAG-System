"""
evaluation/chunking_eval.py

Combines chunk-level statistics (count, token distribution) with retrieval
quality metrics (Recall@k, Precision@k, MRR) per chunking strategy, into
one side-by-side comparison table - per plan spec: "Compare all chunking
strategies: chunk count, avg size, token distribution, retrieval recall,
precision@k, MRR."

Chunk stats are recomputed directly from data/processed/chunks/<strategy>.jsonl
using each Chunk's token_count field (see chunking/base.py), so this
reflects exactly what's actually saved/indexed right now - not a separate
in-memory benchmark run.

Retrieval metrics reuse evaluation.retrieval_eval.evaluate_strategy()
directly (no logic duplicated), at a single retrieval mode - default
hybrid_rerank, since that's the production-target mode and (per prior eval
runs) is also where chunking-strategy differences are smallest, making it
the fairest lens for comparing strategies on equal footing. Pass --mode to
compare strategies at an earlier pipeline stage instead.

Usage:
    python -m evaluation.chunking_eval --strategies all --mode hybrid_rerank \
        --k-values 1,5,10 --max-queries 200
"""

import argparse
import json
import statistics
from pathlib import Path

from sentence_transformers import SentenceTransformer

from evaluation.retrieval_eval import ALL_STRATEGIES, ALL_MODES, evaluate_strategy


def compute_chunk_stats(chunks_path: Path) -> dict:
    token_counts = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tc = rec.get("token_count")
            if tc is not None:
                token_counts.append(tc)

    if not token_counts:
        return {}

    return {
        "chunk_count": len(token_counts),
        "avg_tok": statistics.mean(token_counts),
        "med_tok": statistics.median(token_counts),
        "min_tok": min(token_counts),
        "max_tok": max(token_counts),
        "stdev_tok": statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0,
    }


def print_combined_table(results: dict[str, dict], k_values: list[int], mode: str) -> None:
    print(f"\nChunking strategy comparison (retrieval mode: {mode})")
    print("=" * 140)

    chunk_cols = ["Chunks", "AvgTok", "MedTok", "Min", "Max", "StDev"]
    retrieval_cols = [f"R@{k}" for k in k_values] + ["MRR"]
    header = f"{'Strategy':<14}" + "".join(f"{c:<10}" for c in chunk_cols) + "  " + "".join(f"{c:<10}" for c in retrieval_cols)
    print(header)
    print("-" * len(header))

    for strategy, data in results.items():
        stats = data.get("chunk_stats", {})
        scores = data.get("retrieval_scores", {})
        if not stats or not scores:
            print(f"{strategy:<14}[incomplete - skipped]")
            continue

        row = f"{strategy:<14}"
        row += f"{stats['chunk_count']:<10}"
        row += f"{stats['avg_tok']:<10.1f}"
        row += f"{stats['med_tok']:<10.1f}"
        row += f"{stats['min_tok']:<10}"
        row += f"{stats['max_tok']:<10}"
        row += f"{stats['stdev_tok']:<10.1f}"
        row += "  "
        for k in k_values:
            row += f"{scores.get(f'recall@{k}', 0.0):<10.4f}"
        row += f"{scores.get('mrr', 0.0):<10.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Combined chunk-stats + retrieval-quality comparison per strategy.")
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--mode", type=str, default="hybrid_rerank", choices=ALL_MODES)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--k-values", type=str, default="1,5,10")
    parser.add_argument("--top-n-raw", type=int, default=30, help="Candidate pool size before dedup/rerank")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    args = parser.parse_args()

    strategies = ALL_STRATEGIES if args.strategies == "all" else args.strategies.split(",")
    k_values = [int(k) for k in args.k_values.split(",")]

    print("Loading shared embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {}
    for strategy in strategies:
        print(f"\nProcessing '{strategy}'...")

        chunks_path = args.chunks_dir / f"{strategy}.jsonl"
        chunk_stats = compute_chunk_stats(chunks_path)

        retrieval_scores = evaluate_strategy(
            mode=args.mode,
            strategy=strategy,
            model=model,
            faiss_dir=args.faiss_dir,
            chunks_dir=args.chunks_dir,
            k_values=k_values,
            top_n_raw=args.top_n_raw,
            max_queries=args.max_queries,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )

        results[strategy] = {"chunk_stats": chunk_stats, "retrieval_scores": retrieval_scores}

    print_combined_table(results, k_values, args.mode)


if __name__ == "__main__":
    main()