"""
Data Preprocessing — Phase 2

PURPOSE: Take the raw JSONL from download.py, deduplicate passages, clean text,
         and assign unique document_id + passage_id for indexing.

WHY: The raw data has passages nested inside query records, with many duplicates
     across queries. We need a flat, deduplicated passage corpus with stable IDs
     for the vector index and BM25 index.

OUTPUT: A flat JSONL file where each line is a unique passage with:
  - document_id: str (unique per query)
  - passage_id: str (unique per passage)
  - text: str (cleaned passage text)
  - query_id: int (original query ID)
  - query: str (associated query)
  - query_type: str
  - answer: str (ground truth answer)
  - is_selected: int (relevance label)
  - language: str
  - passage_idx: int (position within the query's passage list)

USAGE: python -m data.preprocess
"""

import json
import re
import sys
import hashlib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def clean_text(text: str) -> str:
    """Clean passage text: normalize whitespace, strip, remove control chars."""
    if not text:
        return ""
    # Remove control characters (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_passage_id(query_id: int, passage_idx: int) -> str:
    """Generate a stable, unique passage ID."""
    return f"q{query_id}_p{passage_idx}"


def generate_document_id(query_id: int) -> str:
    """Generate a stable document ID from query ID."""
    return f"doc_{query_id}"


def preprocess(
    input_path: Path = None,
    output_path: Path = None,
):
    """
    Flatten and deduplicate passages from the raw JSONL download.
    """
    input_path = input_path or next(settings.data_dir.glob("msmarco_subset_*.jsonl"), None)
    if input_path is None or not input_path.exists():
        print(f"ERROR: No raw data found in {settings.data_dir}")
        print("Run 'python -m data.download' first.")
        sys.exit(1)

    output_path = output_path or (settings.processed_dir / "passages.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MSMARCO-XI PREPROCESSING")
    print("=" * 70)
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print()

    # Statistics
    stats = {
        "total_records": 0,
        "total_passages": 0,
        "unique_passages": 0,
        "empty_passages_skipped": 0,
        "short_passages_skipped": 0,
        "duplicate_passages_skipped": 0,
        "query_types": Counter(),
        "selected_passages": 0,
        "passage_lengths": [],
    }

    seen_texts = set()  # For deduplication by text hash

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line.strip())
            stats["total_records"] += 1
            stats["query_types"][record.get("query_type", "unknown")] += 1

            query_id = record["query_id"]
            document_id = generate_document_id(query_id)

            for passage in record.get("passages", []):
                stats["total_passages"] += 1

                text = clean_text(passage.get("text", ""))

                # Skip empty passages
                if not text:
                    stats["empty_passages_skipped"] += 1
                    continue

                # Skip very short passages (< 20 chars = likely garbage)
                if len(text) < 20:
                    stats["short_passages_skipped"] += 1
                    continue

                # Deduplicate by text hash
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen_texts:
                    stats["duplicate_passages_skipped"] += 1
                    continue
                seen_texts.add(text_hash)

                passage_id = generate_passage_id(query_id, passage["passage_idx"])
                is_selected = passage.get("is_selected", 0)

                if is_selected == 1:
                    stats["selected_passages"] += 1

                # Build flat passage record
                flat_record = {
                    "document_id": document_id,
                    "passage_id": passage_id,
                    "text": text,
                    "query_id": query_id,
                    "query": clean_text(record.get("query", "")),
                    "query_type": record.get("query_type", ""),
                    "answer": clean_text(record.get("answer", "")),
                    "is_selected": is_selected,
                    "language": record.get("target_lang", "eng_Latn"),
                    "target_language": record.get("target_lang", ""),
                    "passage_idx": passage["passage_idx"],
                }

                fout.write(json.dumps(flat_record, ensure_ascii=False) + "\n")
                stats["unique_passages"] += 1
                stats["passage_lengths"].append(len(text))

    # Print statistics
    print("PREPROCESSING STATISTICS")
    print("-" * 70)
    print(f"  Input records:           {stats['total_records']}")
    print(f"  Total passages:          {stats['total_passages']}")
    print(f"  Empty skipped:           {stats['empty_passages_skipped']}")
    print(f"  Short skipped (<20ch):   {stats['short_passages_skipped']}")
    print(f"  Duplicates skipped:      {stats['duplicate_passages_skipped']}")
    print(f"  Unique passages saved:   {stats['unique_passages']}")
    print(f"  Selected (relevant):     {stats['selected_passages']}")
    print()

    if stats["passage_lengths"]:
        lengths = stats["passage_lengths"]
        print(f"  Passage length (chars):")
        print(f"    Min:    {min(lengths)}")
        print(f"    Max:    {max(lengths)}")
        print(f"    Mean:   {sum(lengths) / len(lengths):.0f}")
        print(f"    Median: {sorted(lengths)[len(lengths) // 2]}")
        print()

    print(f"  Query types: {dict(stats['query_types'])}")
    print()
    print(f"✅ Preprocessed data saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path, stats


if __name__ == "__main__":
    preprocess()
