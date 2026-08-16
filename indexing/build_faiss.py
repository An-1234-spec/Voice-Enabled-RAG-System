"""
indexing/faiss_index.py

Builds one FAISS index per chunking strategy from the .npy embeddings
produced by embeddings/embedder.py, and saves each index to disk alongside
a row->chunk_id lookup (reusing the *_meta.jsonl embedder already wrote).

Since embedder.py L2-normalizes embeddings (normalize_embeddings=True),
we use IndexFlatIP (inner product) so similarity search == cosine similarity.
This matches your plan's tech stack choice: "FAISS IndexFlatIP - exact
search, best recall, fast enough for <50K chunks."

Usage:
    python -m indexing.faiss_index --embeddings-dir data\processed\embeddings \
        --output-dir data\processed\faiss --strategies all
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

ALL_STRATEGIES = ["fixed_token", "passage", "sentence", "parent_child", "semantic"]


def build_index(strategy: str, embeddings_dir: Path, output_dir: Path) -> None:
    vec_path = embeddings_dir / f"{strategy}.npy"
    meta_path = embeddings_dir / f"{strategy}_meta.jsonl"

    if not vec_path.exists():
        print(f"  [skip] {vec_path} not found")
        return
    if not meta_path.exists():
        print(f"  [skip] {meta_path} not found")
        return

    embeddings = np.load(vec_path).astype(np.float32)
    n, dim = embeddings.shape

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{strategy}.faiss"
    faiss.write_index(index, str(index_path))

    # Row -> chunk_id lookup, so a FAISS result index can be mapped back to
    # a real chunk. We just copy the embedder's meta file alongside the
    # index under a stable name, rather than re-deriving it.
    lookup_path = output_dir / f"{strategy}_lookup.jsonl"
    with open(meta_path, "r", encoding="utf-8") as src, open(lookup_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())

    print(f"  [done] {strategy}: {n} vectors ({dim}-D) -> {index_path.name}, {lookup_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indexes per chunking strategy.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/processed/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated list, or 'all'",
    )
    args = parser.parse_args()

    strategies = ALL_STRATEGIES if args.strategies == "all" else args.strategies.split(",")

    print(f"Building FAISS indexes for: {strategies}\n")
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        build_index(strategy, args.embeddings_dir, args.output_dir)
        print()

    print("All indexes built.")


if __name__ == "__main__":
    main()