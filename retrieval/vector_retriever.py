"""
retrieval/vector_retriever.py

Dense (vector) retrieval: embed a query with the same model used at index
time (all-MiniLM-L6-v2), search a strategy's FAISS index, and resolve hits
back to real chunk text + metadata.

Depends on artifacts already built:
  - data/processed/faiss/<strategy>.faiss        (from indexing/build_faiss.py)
  - data/processed/faiss/<strategy>_lookup.jsonl  (row -> chunk_id/passage_id/document_id)
  - data/processed/chunks/<strategy>.jsonl        (chunk_id -> full chunk text)

Usage (CLI smoke test):
    python -m retrieval.vector_retriever --query "what does RBI do" \
        --strategy fixed_token --top-k 5
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class VectorRetriever:
    """Loads one strategy's FAISS index + chunk text and serves top-K search."""

    def __init__(
        self,
        strategy: str,
        faiss_dir: Path = Path("data/processed/faiss"),
        chunks_dir: Path = Path("data/processed/chunks"),
        model_name: str = DEFAULT_MODEL,
        model: SentenceTransformer | None = None,
    ):
        self.strategy = strategy

        index_path = faiss_dir / f"{strategy}.faiss"
        lookup_path = faiss_dir / f"{strategy}_lookup.jsonl"
        chunks_path = chunks_dir / f"{strategy}.jsonl"

        for p in (index_path, lookup_path, chunks_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Did you run indexing/build_faiss.py "
                    f"and chunking/generate.py for strategy '{strategy}'?"
                )

        self.index = faiss.read_index(str(index_path))

        # row -> {chunk_id, document_id, passage_id, num_tokens}
        self.row_lookup: list[dict] = []
        with open(lookup_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.row_lookup.append(json.loads(line))

        # chunk_id -> full chunk text (+ its other Chunk fields, e.g. query/answer)
        self.chunk_text: dict[str, dict] = {}
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.chunk_text[rec["chunk_id"]] = rec

        # Reuse a shared model instance if the caller passes one in (avoids
        # reloading MiniLM once per strategy when querying multiple indexes).
        self.model = model or SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5, query_vec: np.ndarray | None = None) -> list[dict]:
        if query_vec is None:
            query_vec = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,  # must match embedder.py, or IP scores are meaningless
            ).astype(np.float32)
        else:
            query_vec = np.atleast_2d(query_vec).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, row in zip(scores[0], indices[0]):
            if row == -1:  # FAISS pads with -1 if top_k > number of vectors
                continue
            meta = self.row_lookup[row]
            chunk = self.chunk_text.get(meta["chunk_id"], {})
            results.append({
                "chunk_id": meta["chunk_id"],
                "document_id": meta.get("document_id"),
                "passage_id": meta.get("passage_id"),
                "score": float(score),
                "text": chunk.get("text", ""),
                "strategy": self.strategy,
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval smoke test.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    retriever = VectorRetriever(
        strategy=args.strategy,
        faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir,
    )

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Strategy: {args.strategy}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f} chunk_id={r['chunk_id']}")
        print(f"    {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}\n")


if __name__ == "__main__":
    main()