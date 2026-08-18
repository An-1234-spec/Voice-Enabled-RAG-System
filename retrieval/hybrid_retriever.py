"""
retrieval/hybrid.py

Hybrid retrieval: fuses dense (FAISS) and BM25 (lexical) results via
min-max score normalization + weighted combination, per plan spec
(dense_weight=0.6, bm25_weight=0.4 default).

Both retrievers are queried independently for top_n_raw candidates each
(wider than the final top_k, so fusion has real candidates to rank rather
than just re-ordering each retriever's own top-5). Results are merged by
chunk_id: a chunk found by only one retriever gets 0 contribution from the
other side, not a penalty - it's just not boosted by agreement between
both signals.

Usage (CLI smoke test):
    python -m retrieval.hybrid --query "what does RBI do" \
        --strategy fixed_token --top-k 5 --dense-weight 0.6 --bm25-weight 0.4
"""

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever


def min_max_normalize(results: list[dict]) -> dict[str, float]:
    """
    Returns chunk_id -> normalized score in [0, 1].
    If all scores are equal (min == max), returns 0.5 for every chunk_id
    rather than dividing by zero.
    """
    if not results:
        return {}

    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)

    if hi == lo:
        return {r["chunk_id"]: 0.5 for r in results}

    return {r["chunk_id"]: (r["score"] - lo) / (hi - lo) for r in results}


class HybridRetriever:
    """Combines VectorRetriever + BM25Retriever via weighted min-max fusion."""

    def __init__(
        self,
        strategy: str,
        faiss_dir: Path = Path("data/processed/faiss"),
        chunks_dir: Path = Path("data/processed/chunks"),
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        model: SentenceTransformer | None = None,
    ):
        self.strategy = strategy
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.dense = VectorRetriever(
            strategy=strategy, faiss_dir=faiss_dir, chunks_dir=chunks_dir, model=model
        )
        self.bm25 = BM25Retriever(strategy=strategy, chunks_dir=chunks_dir)

    def search(self, query: str, top_k: int = 5, top_n_raw: int = 30, query_vec: np.ndarray | None = None) -> list[dict]:
        dense_results = self.dense.search(query, top_k=top_n_raw, query_vec=query_vec)
        bm25_results = self.bm25.search(query, top_k=top_n_raw)

        dense_norm = min_max_normalize(dense_results)
        bm25_norm = min_max_normalize(bm25_results)

        # chunk_id -> full record, so we can recover text/passage_id/document_id
        # after fusion regardless of which retriever(s) found it.
        chunk_lookup: dict[str, dict] = {}
        for r in dense_results + bm25_results:
            chunk_lookup.setdefault(r["chunk_id"], r)

        all_chunk_ids = set(dense_norm) | set(bm25_norm)

        fused = []
        for chunk_id in all_chunk_ids:
            d_score = dense_norm.get(chunk_id, 0.0)
            b_score = bm25_norm.get(chunk_id, 0.0)
            fused_score = self.dense_weight * d_score + self.bm25_weight * b_score

            base = chunk_lookup[chunk_id]
            fused.append({
                "chunk_id": chunk_id,
                "document_id": base.get("document_id"),
                "passage_id": base.get("passage_id"),
                "score": fused_score,
                "dense_score": d_score,
                "bm25_score": b_score,
                "text": base.get("text", ""),
                "strategy": self.strategy,
            })

        fused.sort(key=lambda r: r["score"], reverse=True)
        return fused[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n-raw", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    retriever = HybridRetriever(
        strategy=args.strategy,
        faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )

    results = retriever.search(args.query, top_k=args.top_k, top_n_raw=args.top_n_raw)

    print(f"\nQuery: {args.query}")
    print(f"Strategy: {args.strategy} (dense={args.dense_weight}, bm25={args.bm25_weight})\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] fused={r['score']:.4f} (dense={r['dense_score']:.4f}, bm25={r['bm25_score']:.4f}) chunk_id={r['chunk_id']}")
        print(f"    {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}\n")


if __name__ == "__main__":
    main()