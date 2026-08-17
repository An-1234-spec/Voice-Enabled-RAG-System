"""
evaluation/retrieval_eval.py

Evaluates retrieval quality per chunking strategy across multiple retrieval
modes (dense / bm25 / hybrid / hybrid_rerank), using the is_selected
relevance labels carried on each Chunk as ground truth: Recall@k,
Precision@k, MRR.

GROUND TRUTH: built directly from data/processed/chunks/<strategy>.jsonl
  - group chunks by query_id
  - a chunk's source passage is "relevant" to that query_id if is_selected == 1
  - query text taken from the first non-null `query` field seen for that query_id

WHY DEDUPE TO PASSAGE-LEVEL: relevance labels are per-passage, but strategies
like sentence/parent_child emit multiple chunks per passage. Scoring raw
chunk hits would let one relevant passage "count" multiple times and let
strategies with more chunks/passage crowd out other passages at low k.
So we retrieve top_n_raw chunks, collapse to a rank-ordered list of unique
passage_ids (first occurrence wins), then score Recall/Precision/MRR at k
against that deduped list. This applies identically regardless of mode,
since dense/bm25/hybrid/hybrid_rerank all return the same result shape
(chunk_id, document_id, passage_id, score, text, strategy).

NOTE ON hybrid_rerank: retrieve_n is set to top_n_raw (not the plan's
production default of 30), so this mode searches the same size candidate
pool as dense/bm25/hybrid - a fair eval comparison. Production config
(top-30 -> rerank -> top-5) is narrower; pass --top-n-raw 30 if you want
eval to match production exactly.

Usage:
    python -m evaluation.retrieval_eval --strategies all --modes dense,bm25,hybrid,hybrid_rerank \
        --k-values 1,5,10 --top-n-raw 50 --max-queries 200
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import RerankedRetriever

ALL_STRATEGIES = ["fixed_token", "passage", "sentence", "parent_child", "semantic"]
ALL_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def build_ground_truth(chunks_path: Path) -> tuple[dict[int, str], dict[int, set[str]]]:
    """
    Returns:
        queries: query_id -> query text
        relevant: query_id -> set of relevant passage_ids (is_selected == 1)
    """
    queries: dict[int, str] = {}
    relevant: dict[int, set[str]] = defaultdict(set)

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id")
            if qid is None:
                continue
            if qid not in queries and rec.get("query"):
                queries[qid] = rec["query"]
            if rec.get("is_selected") == 1:
                relevant[qid].add(rec["passage_id"])

    return queries, relevant


def dedupe_to_passages(chunk_results: list[dict]) -> list[str]:
    """Collapse ranked chunk hits to a rank-ordered list of unique passage_ids."""
    seen = set()
    passage_order = []
    for r in chunk_results:
        pid = r["passage_id"]
        if pid not in seen:
            seen.add(pid)
            passage_order.append(pid)
    return passage_order


def score_query(
    ranked_passage_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> dict:
    """Compute Recall@k, Precision@k for each k, plus MRR, for one query."""
    result = {}
    for k in k_values:
        top_k = ranked_passage_ids[:k]
        hits = sum(1 for pid in top_k if pid in relevant_ids)
        result[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
        result[f"precision@{k}"] = hits / k if k > 0 else 0.0

    mrr = 0.0
    for rank, pid in enumerate(ranked_passage_ids, start=1):
        if pid in relevant_ids:
            mrr = 1.0 / rank
            break
    result["mrr"] = mrr
    return result


def get_retriever(
    mode: str,
    strategy: str,
    model: SentenceTransformer,
    faiss_dir: Path,
    chunks_dir: Path,
    dense_weight: float,
    bm25_weight: float,
    top_n_raw: int,
):
    """Factory: returns any retriever exposing .search(query, top_k) -> list[dict]."""
    if mode == "dense":
        return VectorRetriever(strategy=strategy, faiss_dir=faiss_dir, chunks_dir=chunks_dir, model=model)
    elif mode == "bm25":
        return BM25Retriever(strategy=strategy, chunks_dir=chunks_dir)
    elif mode == "hybrid":
        return HybridRetriever(
            strategy=strategy,
            faiss_dir=faiss_dir,
            chunks_dir=chunks_dir,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            model=model,
        )
    elif mode == "hybrid_rerank":
        return RerankedRetriever(
            strategy=strategy,
            base_mode="hybrid",
            faiss_dir=faiss_dir,
            chunks_dir=chunks_dir,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            retrieve_n=top_n_raw,
            model=model,
        )
    else:
        raise NotImplementedError(f"mode '{mode}' is not supported.")


def evaluate_strategy(
    mode: str,
    strategy: str,
    model: SentenceTransformer,
    faiss_dir: Path,
    chunks_dir: Path,
    k_values: list[int],
    top_n_raw: int,
    max_queries: int | None,
    dense_weight: float,
    bm25_weight: float,
) -> dict:
    chunks_path = chunks_dir / f"{strategy}.jsonl"
    queries, relevant = build_ground_truth(chunks_path)

    # only evaluate queries that actually have at least one relevant passage
    query_ids = [qid for qid in queries if relevant.get(qid)]
    if max_queries:
        query_ids = query_ids[:max_queries]

    if not query_ids:
        print(f"  [skip] {strategy}/{mode}: no queries with relevant passages found")
        return {}

    retriever = get_retriever(mode, strategy, model, faiss_dir, chunks_dir, dense_weight, bm25_weight, top_n_raw)

    totals = defaultdict(list)
    for qid in query_ids:
        query_text = queries[qid]
        relevant_ids = relevant[qid]

        # HybridRetriever additionally takes top_n_raw (candidates pulled from
        # each of dense/bm25 before fusion) - must match top_k here or hybrid
        # silently fuses over its own default (30) instead of the eval's pool size.
        # RerankedRetriever's pool size is fixed at construction (retrieve_n
        # above), so it only needs top_k here, same as dense/bm25.
        if mode == "hybrid":
            raw_results = retriever.search(query_text, top_k=top_n_raw, top_n_raw=top_n_raw)
        else:
            raw_results = retriever.search(query_text, top_k=top_n_raw)

        ranked_passages = dedupe_to_passages(raw_results)

        scores = score_query(ranked_passages, relevant_ids, k_values)
        for metric, value in scores.items():
            totals[metric].append(value)

    averaged = {metric: sum(vals) / len(vals) for metric, vals in totals.items()}
    averaged["num_queries"] = len(query_ids)
    return averaged


def print_mode_summary(mode: str, results: dict[str, dict], k_values: list[int]) -> None:
    metrics = [f"recall@{k}" for k in k_values] + [f"precision@{k}" for k in k_values] + ["mrr"]
    print(f"\n--- mode: {mode} ---")
    header = f"{'Strategy':<14}{'N':<6}" + "".join(f"{m:<14}" for m in metrics)
    print(header)
    print("-" * len(header))
    for strategy, scores in results.items():
        if not scores:
            continue
        row = f"{strategy:<14}{scores['num_queries']:<6}"
        row += "".join(f"{scores.get(m, 0.0):<14.4f}" for m in metrics)
        print(row)


def print_cross_mode_comparison(all_results: dict[str, dict[str, dict]], modes: list[str]) -> None:
    """Compact strategy x mode table for recall@5 and mrr, the two headline metrics."""
    print("\n" + "=" * 60)
    print("CROSS-MODE COMPARISON (recall@5 / mrr)")
    print("=" * 60)
    header = f"{'Strategy':<14}" + "".join(f"{m:<24}" for m in modes)
    print(header)
    print("-" * len(header))
    for strategy in ALL_STRATEGIES:
        row = f"{strategy:<14}"
        for mode in modes:
            scores = all_results.get(mode, {}).get(strategy, {})
            if scores:
                cell = f"r@5={scores.get('recall@5', 0):.4f} mrr={scores.get('mrr', 0):.4f}"
            else:
                cell = "n/a"
            row += f"{cell:<24}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality per chunking strategy and mode.")
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument(
        "--modes",
        type=str,
        default="dense,bm25,hybrid,hybrid_rerank",
        help="Comma-separated: dense,bm25,hybrid,hybrid_rerank",
    )
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--k-values", type=str, default="1,5,10")
    parser.add_argument("--top-n-raw", type=int, default=50, help="Raw chunks pulled before passage-level dedup")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    args = parser.parse_args()

    strategies = ALL_STRATEGIES if args.strategies == "all" else args.strategies.split(",")
    modes = args.modes.split(",")
    k_values = [int(k) for k in args.k_values.split(",")]

    for mode in modes:
        if mode not in ALL_MODES:
            raise NotImplementedError(f"mode '{mode}' is not available. Supported: {ALL_MODES}")

    print("Loading shared embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_results: dict[str, dict[str, dict]] = {}
    for mode in modes:
        mode_results = {}
        for strategy in strategies:
            print(f"\nEvaluating '{strategy}' (mode={mode})...")
            scores = evaluate_strategy(
                mode=mode,
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
            mode_results[strategy] = scores
        all_results[mode] = mode_results
        print_mode_summary(mode, mode_results, k_values)

    if len(modes) > 1:
        print_cross_mode_comparison(all_results, modes)


if __name__ == "__main__":
    main()