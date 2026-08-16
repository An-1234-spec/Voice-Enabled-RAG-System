"""
Controlled Dataset Download — Phase 2

PURPOSE: Download a controlled subset of MSMARCO-XI without downloading the entire
         dataset. Uses HuggingFace streaming to iterate and collect only the records
         we need, then saves them as JSONL.

WHY: The full dataset is 10M-100M records across 14 Indic languages. We cannot
     store it locally. Instead we stream and save only what we need.

STRATEGY:
  1. Stream the dataset (zero disk usage until we save)
  2. Collect N records (configurable, default 5000)
  3. Extract English passages + English queries + metadata
  4. Save as a compact JSONL file

USAGE: python -m data.download [--size 5000] [--split validation]
"""

import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import settings


def download_subset(
    subset_size: int = None,
    split: str = None,
    output_path: Path = None,
):
    """
    Stream MSMARCO-XI and save a controlled subset as JSONL.

    Each saved record contains:
      - query_id: int
      - query: str (English query)
      - query_type: str
      - answer: str (English answer)
      - passages: list[dict] with keys {text, is_selected}
      - target_lang: str
      - num_passages: int
      - num_selected: int
    """
    from datasets import load_dataset

    subset_size = subset_size or settings.dataset_subset_size
    split = split or settings.dataset_split
    output_path = output_path or (settings.data_dir / f"msmarco_subset_{subset_size}.jsonl")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MSMARCO-XI CONTROLLED DOWNLOAD")
    print("=" * 70)
    print(f"  Dataset:     {settings.dataset_name}")
    print(f"  Split:       {split}")
    print(f"  Target size: {subset_size} records")
    print(f"  Output:      {output_path}")
    print()

    # Estimate: each record is ~2-5 KB as JSONL
    est_size_mb = subset_size * 3 / 1024  # ~3KB avg per record
    print(f"  Estimated output size: ~{est_size_mb:.1f} KB")
    if est_size_mb > 1000:
        print(f"  ⚠️  WARNING: Estimated size > 1 GB! Consider reducing subset_size.")
        response = input("  Continue? [y/N]: ").strip().lower()
        if response != "y":
            print("  Aborted.")
            return
    print()

    # Stream the dataset
    print(f"[1/2] Streaming dataset (no full download)...")
    ds = load_dataset(
        settings.dataset_name,
        name="default",
        split=split,
        streaming=True,
    )

    # Collect and save records
    print(f"[2/2] Collecting {subset_size} records...")
    saved = 0
    seen_query_ids = set()

    with open(output_path, "w", encoding="utf-8") as f:
        for record in tqdm(ds, total=subset_size, desc="Downloading"):
            if saved >= subset_size:
                break

            # Skip duplicates by query_id
            qid = record.get("query_id")
            if qid in seen_query_ids:
                continue
            seen_query_ids.add(qid)

            # Extract passages into a clean format
            raw_passages = record.get("passages", {})
            eng_passages = raw_passages.get("English_passages", [])
            trans_passages = raw_passages.get("Translated_passages", [])
            is_selected = raw_passages.get("is_selected", [])

            # Build clean passage list
            passages = []
            for i in range(len(eng_passages)):
                passage = {
                    "text": eng_passages[i] if i < len(eng_passages) else "",
                    "translated_text": trans_passages[i] if i < len(trans_passages) else "",
                    "is_selected": is_selected[i] if i < len(is_selected) else 0,
                    "passage_idx": i,
                }
                passages.append(passage)

            # Build clean record
            clean_record = {
                "query_id": qid,
                "query": record.get("Eng_Query", record.get("query", "")),
                "query_translated": record.get("query", ""),
                "query_type": record.get("query_type", ""),
                "answer": record.get("Eng_Answer", record.get("Answer", "")),
                "answer_translated": record.get("Answer", ""),
                "passages": passages,
                "source_lang": record.get("source_lang", ""),
                "target_lang": record.get("target_lang", ""),
                "num_passages": len(passages),
                "num_selected": sum(1 for p in passages if p["is_selected"] == 1),
            }

            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
            saved += 1

    print()
    print(f"✅ Saved {saved} records to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Quick stats
    print()
    print("Quick statistics:")
    print(f"  Unique query IDs: {len(seen_query_ids)}")
    print(f"  Records saved: {saved}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a controlled subset of MSMARCO-XI")
    parser.add_argument("--size", type=int, default=None, help="Number of records to download")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (train/validation)")
    args = parser.parse_args()

    download_subset(subset_size=args.size, split=args.split)
