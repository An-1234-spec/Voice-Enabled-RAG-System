"""
chunking/generate.py

Runs all chunking strategies over data/processed/passages.jsonl and saves
each strategy's output to data/processed/chunks/<strategy>.jsonl.

Each chunker's chunk() method (per chunking/base.py's BaseChunker interface)
operates on ONE passage's text at a time and returns List[Chunk]. This
script loops over all passages, calls chunk() per passage per strategy,
and flattens+writes the results.

ASSUMPTIONS FLAGGED BELOW (adjust if wrong):
  - passages.jsonl records have keys: passage_id, document_id, text,
    and optionally query_id, query, query_type, answer, is_selected, language.
  - Chunker constructors for fixed/sentence/passage/semantic take a single
    `chunk_size` kwarg (mirroring parent_chunk_size/child_chunk_size in
    ParentChildChunker). If any constructor differs, you'll get a TypeError
    at that specific chunker — fix just that one line in STRATEGY_BUILDERS.

Usage:
    python -m chunking.generate --data data\processed\passages.jsonl \
        --output-dir data\processed\chunks --chunk-size 80
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from chunking.base import BaseChunker, Chunk

# ── ADJUST THESE IMPORTS/CLASS NAMES if they don't match your actual files ──
from chunking.fixed_token import FixedTokenChunker
from chunking.sentence import SentenceChunker
from chunking.passage import PassageChunker
from chunking.semantic import SemanticChunker
from chunking.parent_child import ParentChildChunker

# metadata.py (metadata-aware strategy) intentionally excluded — plan lists 6
# strategies but your benchmark only ran these 5. Add it here once it exists.

# Maps strategy name -> factory(chunk_size) -> chunker instance.
# ParentChildChunker doesn't take chunk_size directly, so it gets a small
# lambda that derives parent/child sizes from it (4x split — ADJUST if you
# want a different parent:child ratio).
STRATEGY_BUILDERS = {
    "fixed_token": lambda chunk_size: FixedTokenChunker(chunk_size=chunk_size),
    "passage": lambda chunk_size: PassageChunker(chunk_size=chunk_size),
    "sentence": lambda chunk_size: SentenceChunker(chunk_size=chunk_size),
    "semantic": lambda chunk_size: SemanticChunker(chunk_size=chunk_size),
    "parent_child": lambda chunk_size: ParentChildChunker(
        parent_chunk_size=chunk_size * 4, child_chunk_size=chunk_size
    ),
}

# ADJUST: keys expected on each passage record in passages.jsonl.
PASSAGE_TEXT_KEY = "text"
METADATA_KEYS = ["query_id", "query", "query_type", "answer", "is_selected", "language"]


def load_passages(path: Path) -> list[dict]:
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
            if PASSAGE_TEXT_KEY not in rec:
                raise KeyError(
                    f"Missing '{PASSAGE_TEXT_KEY}' key in {path}:{line_no}. "
                    f"Available keys: {list(rec.keys())}"
                )
            if "passage_id" not in rec or "document_id" not in rec:
                raise KeyError(
                    f"Missing 'passage_id'/'document_id' in {path}:{line_no}. "
                    f"Available keys: {list(rec.keys())}"
                )
            records.append(rec)
    return records


def run_strategy(
    strategy: str,
    chunker: BaseChunker,
    passages: list[dict],
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for passage in passages:
        metadata = {k: passage.get(k) for k in METADATA_KEYS}
        chunks = chunker.chunk(
            text=passage[PASSAGE_TEXT_KEY],
            passage_id=passage["passage_id"],
            document_id=passage["document_id"],
            **metadata,
        )
        all_chunks.extend(chunks)
    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate and save chunks for all strategies.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/passages.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated subset, e.g. 'fixed_token,semantic', or 'all'",
    )
    args = parser.parse_args()

    strategies = (
        list(STRATEGY_BUILDERS.keys())
        if args.strategies == "all"
        else args.strategies.split(",")
    )

    print(f"Loading passages from {args.data}...")
    passages = load_passages(args.data)
    print(f"Loaded {len(passages)} passages.\n")

    for strategy in strategies:
        if strategy not in STRATEGY_BUILDERS:
            print(f"  [skip] Unknown strategy '{strategy}'")
            continue

        print(f"Running '{strategy}'...")
        chunker = STRATEGY_BUILDERS[strategy](args.chunk_size)
        chunks = run_strategy(strategy, chunker, passages)

        out_path = args.output_dir / f"{strategy}.jsonl"
        save_chunks(chunks, out_path)
        print(f"  [done] {len(chunks)} chunks -> {out_path}\n")

    print("All chunk files generated.")


if __name__ == "__main__":
    main()