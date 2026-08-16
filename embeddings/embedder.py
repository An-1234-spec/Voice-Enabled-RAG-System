"""
embeddings/embedder.py

Encodes chunked passages into dense vectors using Sentence-Transformers
(all-MiniLM-L6-v2, 384-D) and saves one .npy + metadata.jsonl pair per
chunking strategy, ready for FAISS indexing.

Assumes each chunking strategy's output lives at:
    data/processed/chunks/<strategy>.jsonl
with each line containing at least: chunk_id, text (adjust CHUNK_TEXT_KEY /
CHUNK_ID_KEY below if your chunker uses different field names).

Usage:
    python -m embeddings.embedder --chunks-dir data/processed/chunks \
        --strategies fixed_token,passage,sentence,parent_child,semantic \
        --output-dir data/processed/embeddings \
        --model all-MiniLM-L6-v2 --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_TEXT_KEY = "text"
CHUNK_ID_KEY = "chunk_id"

ALL_STRATEGIES = ["fixed_token", "passage", "sentence", "parent_child", "semantic"]


def load_chunks(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}") from e
            if CHUNK_TEXT_KEY not in rec:
                raise KeyError(
                    f"Missing '{CHUNK_TEXT_KEY}' key in {path}:{line_no}. "
                    f"Available keys: {list(rec.keys())}"
                )
            records.append(rec)
    return records


def encode_strategy(
    model: SentenceTransformer,
    strategy: str,
    chunks_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    in_path = chunks_dir / f"{strategy}.jsonl"
    if not in_path.exists():
        print(f"  [skip] {in_path} not found")
        return

    records = load_chunks(in_path)
    if not records:
        print(f"  [skip] {in_path} is empty")
        return

    texts = [r[CHUNK_TEXT_KEY] for r in records]

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine sim = dot product, plays nice with FAISS IndexFlatIP
    )
    elapsed = time.time() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    vec_path = output_dir / f"{strategy}.npy"
    meta_path = output_dir / f"{strategy}_meta.jsonl"

    np.save(vec_path, embeddings.astype(np.float32))
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            meta = {
                "row": i,
                CHUNK_ID_KEY: rec.get(CHUNK_ID_KEY, f"{strategy}_{i}"),
                "num_tokens": rec.get("num_tokens"),
                "document_id": rec.get("document_id"),
                "passage_id": rec.get("passage_id"),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(
        f"  [done] {strategy}: {len(records)} chunks -> {embeddings.shape} "
        f"in {elapsed:.1f}s ({len(records)/elapsed:.1f} chunks/sec)"
    )
    print(f"         saved {vec_path.name}, {meta_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Encode chunks into dense vectors.")
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/embeddings"))
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(ALL_STRATEGIES),
        help="Comma-separated list, or 'all'",
    )
    args = parser.parse_args()

    strategies = ALL_STRATEGIES if args.strategies == "all" else args.strategies.split(",")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model '{args.model}'...")
    model = SentenceTransformer(args.model, device=device)

    print(f"Encoding strategies: {strategies}\n")
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        encode_strategy(model, strategy, args.chunks_dir, args.output_dir, args.batch_size)
        print()

    print("All done. Vectors are L2-normalized -> use IndexFlatIP in FAISS for cosine similarity.")


if __name__ == "__main__":
    main()