"""
retrieval/bm25_retriever.py

BM25 (lexical) retrieval using rank_bm25. Builds an in-memory BM25 index
per chunking strategy from data/processed/chunks/<strategy>.jsonl — no
separate build step, since the corpus is small enough (~1000-1500 chunks)
that indexing is sub-second (per plan: "BM25 - Simple, effective, in-memory").

Returns the same result shape as retrieval/vector_retriever.py's
VectorRetriever.search(), so retrieval/hybrid.py can fuse both without
special-casing either retriever.

NOTE ON SCORES: BM25 scores are unbounded (can be 0 to 20+ depending on
corpus/query), unlike the FAISS cosine scores which are ~0-1. Do NOT
compare raw scores between this and VectorRetriever directly - hybrid
fusion must min-max normalize both score sets first.

Usage (CLI smoke test):
    python -m retrieval.bm25_retriever --query "what does RBI do" \
        --strategy fixed_token --top-k 5
"""

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-word tokenization. No stemming/stopword removal."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Loads one strategy's chunks and builds an in-memory BM25 index over them."""

    def __init__(
        self,
        strategy: str,
        chunks_dir: Path = Path("data/processed/chunks"),
    ):
        self.strategy = strategy

        chunks_path = chunks_dir / f"{strategy}.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"{chunks_path} not found. Did you run chunking/generate.py "
                f"for strategy '{strategy}'?"
            )

        self.chunks: list[dict] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        if not self.chunks:
            raise ValueError(f"{chunks_path} is empty.")

        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # top_k indices by score, descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk.get("document_id"),
                "passage_id": chunk.get("passage_id"),
                "score": float(scores[idx]),
                "text": chunk.get("text", ""),
                "strategy": self.strategy,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    retriever = BM25Retriever(strategy=args.strategy, chunks_dir=args.chunks_dir)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Strategy: {args.strategy}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f} chunk_id={r['chunk_id']}")
        print(f"    {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}\n")


if __name__ == "__main__":
    main()