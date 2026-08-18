"""
retrieval/reranker.py

Cross-encoder reranking: takes a shortlist of candidates from a first-stage
retriever (dense/bm25/hybrid) and re-scores each (query, chunk_text) pair
directly with cross-encoder/ms-marco-MiniLM-L-6-v2, per plan spec:
"Retrieve top-30 -> rerank -> return top-5."

WHY THIS EXISTS: dense/bm25/hybrid all score query-vs-chunk independently
(bi-encoder style) which is fast but less accurate. A cross-encoder scores
the (query, chunk) PAIR jointly, which is much more accurate but too slow
to run over the whole corpus (~10ms/pair) - hence: cheap retriever narrows
~1300 chunks to 30, then the expensive-but-accurate reranker picks the
real top 5 from those 30.

Default base retriever is 'hybrid', matching the plan's architecture
diagram (Hybrid Retrieval -> Cross-Encoder Reranker -> Context Selection).
Pass --base-mode dense or --base-mode bm25 to rerank a different retriever's
output instead.

Usage (CLI smoke test):
    python -m retrieval.reranker --query "what is a corporation?" \
        --strategy fixed_token --base-mode hybrid --retrieve-n 30 --top-k 5
"""

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Wraps a cross-encoder to re-score and re-sort a retriever's candidates."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER, cross_encoder: CrossEncoder | None = None):
        self.model = cross_encoder or CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)  # raw relevance scores, higher = more relevant

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)  # don't mutate the retriever's original dicts
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return reranked[:top_k]


class RerankedRetriever:
    """
    Composes a first-stage retriever + Reranker into one .search() call,
    so evaluation/retrieval_eval.py can treat 'hybrid_rerank' like any
    other mode (same input/output shape as dense/bm25/hybrid).
    """

    def __init__(
        self,
        strategy: str,
        base_mode: str = "hybrid",
        faiss_dir: Path = Path("data/processed/faiss"),
        chunks_dir: Path = Path("data/processed/chunks"),
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        retrieve_n: int = 10,
        model: SentenceTransformer | None = None,
        cross_encoder: CrossEncoder | None = None,
    ):
        self.retrieve_n = retrieve_n
        self.reranker = Reranker(cross_encoder=cross_encoder)

        if base_mode == "dense":
            self.base = VectorRetriever(strategy=strategy, faiss_dir=faiss_dir, chunks_dir=chunks_dir, model=model)
        elif base_mode == "bm25":
            self.base = BM25Retriever(strategy=strategy, chunks_dir=chunks_dir)
        elif base_mode == "hybrid":
            self.base = HybridRetriever(
                strategy=strategy, faiss_dir=faiss_dir, chunks_dir=chunks_dir,
                dense_weight=dense_weight, bm25_weight=bm25_weight, model=model,
            )
        else:
            raise ValueError(f"Unknown base_mode '{base_mode}'")

    def search(self, query: str, top_k: int = 5, query_vec: np.ndarray | None = None) -> list[dict]:
        if isinstance(self.base, (VectorRetriever, HybridRetriever)):
            candidates = self.base.search(query, top_k=self.retrieve_n, query_vec=query_vec)
        else:
            candidates = self.base.search(query, top_k=self.retrieve_n)
        return self.reranker.rerank(query, candidates, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Cross-encoder reranking smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--base-mode", type=str, default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--retrieve-n", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    retriever = RerankedRetriever(
        strategy=args.strategy,
        base_mode=args.base_mode,
        faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir,
        retrieve_n=args.retrieve_n,
    )

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Strategy: {args.strategy} (base={args.base_mode}, retrieve_n={args.retrieve_n})\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] rerank_score={r['rerank_score']:.4f} chunk_id={r['chunk_id']}")
        print(f"    {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}\n")


if __name__ == "__main__":
    main()