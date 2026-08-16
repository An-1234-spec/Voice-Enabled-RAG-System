"""
chunking/benchmark.py

Compares all chunking strategies (fixed_token, passage, sentence, semantic,
parent_child) on the same input data: chunk counts, token-size distribution,
and throughput. This is the "chunk count, avg size, token distribution"
half of evaluation/chunking_eval.py from the plan — retrieval-quality
metrics (recall@k, precision@k, MRR) belong in evaluation/retrieval_eval.py
once FAISS + embeddings exist, since those require an actual index to
query against.

Usage:
    python -m chunking.benchmark --data path/to/preprocessed.jsonl
    python -m chunking.benchmark --data path/to/preprocessed.jsonl --limit 200
    python -m chunking.benchmark                      # uses built-in synthetic sample

Expected JSONL record shape (field names are configurable via CLI flags —
these are just the defaults):
    {
      "passage_id": "...",       (--passage-id-field, default "passage_id")
      "document_id": "...",      (--document-id-field, default "document_id")
      "text": "...",             (--text-field, default "text")
      "query_id": 123,           (optional, passed through as chunk metadata)
      "query": "...",            (optional)
      "query_type": "...",       (optional)
      "answer": "...",           (optional)
      "is_selected": 1,          (optional)
      "language": "en"           (optional)
    }
If passage_id/document_id are missing, the script generates stable
"row_{i}" IDs so it still runs — but you'll want the real IDs for the
metadata store once indexing.py exists.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from chunking.base import BaseChunker, Chunk
from chunking.fixed_token import FixedTokenChunker
from chunking.passage import PassageChunker
from chunking.sentence import SentenceChunker
from chunking.parent_child import ParentChildChunker

try:
    from chunking.semantic import SemanticChunker
    _SEMANTIC_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover
    SemanticChunker = None
    _SEMANTIC_IMPORT_ERROR = e

_PASSTHROUGH_METADATA_FIELDS = (
    "query_id",
    "query",
    "query_type",
    "answer",
    "is_selected",
    "language",
)

_SYNTHETIC_SAMPLE = [
    {
        "passage_id": "p0",
        "document_id": "d0",
        "text": (
            "The Reserve Bank of India regulates monetary policy in the country. "
            "It was established in 1935 under the Reserve Bank of India Act. "
            "RBI's key functions include issuing currency, managing foreign exchange "
            "reserves, and acting as a banker to the government and other banks."
        ),
        "query_id": 1,
        "query": "what does the RBI do",
        "language": "en",
    },
    {
        "passage_id": "p1",
        "document_id": "d1",
        "text": (
            "Cricket is one of the most popular sports in India. The Indian national "
            "cricket team has won two Cricket World Cups, in 1983 and 2011. Virat Kohli "
            "is widely regarded as one of the greatest batsmen of the modern era."
        ),
        "query_id": 2,
        "query": "has india won the cricket world cup",
        "language": "en",
    },
    {
        "passage_id": "p2",
        "document_id": "d2",
        "text": (
            "Goa is a state on India's western coast known for its beaches, nightlife, "
            "and Portuguese colonial architecture. It was a Portuguese colony until 1961. "
            "Tourism is a major part of Goa's economy."
        ),
        "query_id": 3,
        "query": "what is goa known for",
        "language": "en",
    },
]


def load_records(
    path: Optional[str],
    limit: Optional[int],
    text_field: str,
    passage_id_field: str,
    document_id_field: str,
) -> List[Dict[str, Any]]:
    if not path:
        print("No --data path given — using built-in synthetic sample (3 passages).")
        return _SYNTHETIC_SAMPLE[:limit] if limit else _SYNTHETIC_SAMPLE

    p = Path(path)
    if not p.exists():
        print(f"'{path}' not found — falling back to built-in synthetic sample (3 passages).")
        return _SYNTHETIC_SAMPLE[:limit] if limit else _SYNTHETIC_SAMPLE

    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            text = raw.get(text_field)
            if not text or not str(text).strip():
                continue

            record = {
                "passage_id": raw.get(passage_id_field, f"row_{i}"),
                "document_id": raw.get(document_id_field, raw.get(passage_id_field, f"row_{i}")),
                "text": str(text),
            }
            for field in _PASSTHROUGH_METADATA_FIELDS:
                if field in raw and raw[field] is not None:
                    record[field] = raw[field]
            records.append(record)

    print(f"Loaded {len(records)} records from {path}.")
    return records


def _run_strategy(
    name: str,
    chunker: BaseChunker,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    all_chunks: List[Chunk] = []
    start = time.perf_counter()

    for rec in records:
        metadata = {k: rec[k] for k in _PASSTHROUGH_METADATA_FIELDS if k in rec}
        chunks = chunker.chunk(
            text=rec["text"],
            passage_id=rec["passage_id"],
            document_id=rec["document_id"],
            **metadata,
        )
        all_chunks.extend(chunks)

    elapsed = time.perf_counter() - start
    token_counts = [c.token_count for c in all_chunks]

    stats: Dict[str, Any] = {
        "strategy": name,
        "num_passages": len(records),
        "num_chunks": len(all_chunks),
        "chunks_per_passage": round(len(all_chunks) / len(records), 2) if records else 0,
        "elapsed_sec": round(elapsed, 4),
        "passages_per_sec": round(len(records) / elapsed, 1) if elapsed > 0 else float("inf"),
    }

    if token_counts:
        stats.update(
            {
                "avg_tokens": round(statistics.mean(token_counts), 1),
                "median_tokens": statistics.median(token_counts),
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
                "stdev_tokens": round(statistics.pstdev(token_counts), 1)
                if len(token_counts) > 1
                else 0.0,
            }
        )
    else:
        stats.update(
            {"avg_tokens": 0, "median_tokens": 0, "min_tokens": 0, "max_tokens": 0, "stdev_tokens": 0}
        )

    return stats


def _print_table(rows: List[Dict[str, Any]]) -> None:
    columns = [
        ("strategy", "Strategy", 13),
        ("num_chunks", "Chunks", 8),
        ("chunks_per_passage", "Chunks/Pass", 12),
        ("avg_tokens", "Avg Tok", 8),
        ("median_tokens", "Med Tok", 8),
        ("min_tokens", "Min", 6),
        ("max_tokens", "Max", 6),
        ("stdev_tokens", "StDev", 7),
        ("passages_per_sec", "Pass/sec", 9),
    ]

    header = "  ".join(title.ljust(w) for _, title, w in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(str(row.get(key, "")).ljust(w) for key, _, w in columns)
        print(line)


def run_benchmark(
    records: List[Dict[str, Any]],
    chunk_size: int = 384,
    chunk_overlap: int = 50,
    parent_chunk_size: int = 512,
    child_chunk_size: int = 128,
    include_semantic: bool = True,
) -> List[Dict[str, Any]]:
    strategies: List[tuple] = [
        ("fixed_token", FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)),
        ("passage", PassageChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)),
        ("sentence", SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)),
        (
            "parent_child",
            ParentChildChunker(
                parent_chunk_size=parent_chunk_size, child_chunk_size=child_chunk_size
            ),
        ),
    ]

    if include_semantic:
        if SemanticChunker is None:
            print(
                f"Skipping 'semantic' strategy — sentence-transformers not "
                f"installed ({_SEMANTIC_IMPORT_ERROR}). Run "
                f"`pip install sentence-transformers` to include it."
            )
        else:
            try:
                strategies.append(("semantic", SemanticChunker(chunk_size=chunk_size)))
            except Exception as e:  # pragma: no cover
                print(f"Skipping 'semantic' strategy — failed to initialize: {e}")

    results = []
    for name, chunker in strategies:
        print(f"Running '{name}'...")
        try:
            results.append(_run_strategy(name, chunker, records))
        except ImportError as e:
            # SemanticChunker raises this lazily inside .chunk() if no
            # embedder/model is available at call time, not at construction.
            print(f"Skipping '{name}' strategy — {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark chunking strategies.")
    parser.add_argument("--data", type=str, default=None, help="Path to preprocessed JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Max records to load.")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--passage-id-field", type=str, default="passage_id")
    parser.add_argument("--document-id-field", type=str, default="document_id")
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--parent-chunk-size", type=int, default=512)
    parser.add_argument("--child-chunk-size", type=int, default=128)
    parser.add_argument("--no-semantic", action="store_true", help="Skip the semantic strategy.")
    parser.add_argument(
        "--output", type=str, default=None, help="Optional path to write results as JSON."
    )
    args = parser.parse_args()

    records = load_records(
        args.data, args.limit, args.text_field, args.passage_id_field, args.document_id_field
    )
    if not records:
        print("No records to benchmark — exiting.")
        return

    results = run_benchmark(
        records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parent_chunk_size=args.parent_chunk_size,
        child_chunk_size=args.child_chunk_size,
        include_semantic=not args.no_semantic,
    )

    print()
    _print_table(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()