"""
Dataset Inspection Script — Phase 1

PURPOSE: Safely inspect the ai4bharat/MSMARCO-XI dataset using streaming mode.
         We NEVER download the full dataset. We stream a few records, print their
         structure, field names, types, sizes, and sample values.

WHY: The requirements explicitly say "Do not make assumptions about dataset fields."
     We must see the real data before building any preprocessing pipeline.

USAGE: python -m data.inspect
"""

import sys
import json
from collections import defaultdict


def inspect_dataset():
    """Stream a small sample from MSMARCO-XI and print detailed field analysis."""

    print("=" * 70)
    print("MSMARCO-XI DATASET INSPECTION")
    print("=" * 70)
    print()

    # Import here so we fail fast with a clear message if datasets isn't installed
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    # ── Step 1: Stream a small sample (NO full download) ─────────────
    print("[1/5] Loading dataset in streaming mode (no full download)...")
    print("       Dataset: ai4bharat/MSMARCO-XI")
    print("       Config:  default")
    print("       Split:   validation")
    print()

    try:
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI",
            name="default",
            split="validation",
            streaming=True,
        )
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        print()
        print("Trying with 'train' split instead...")
        try:
            ds = load_dataset(
                "ai4bharat/MSMARCO-XI",
                name="default",
                split="train",
                streaming=True,
            )
        except Exception as e2:
            print(f"ERROR with train split too: {e2}")
            sys.exit(1)

    # ── Step 2: Grab first 5 records ─────────────────────────────────
    print("[2/5] Fetching first 5 records...")
    print()

    samples = []
    for i, record in enumerate(ds):
        if i >= 5:
            break
        samples.append(record)

    if not samples:
        print("ERROR: No records returned from streaming.")
        sys.exit(1)

    print(f"       Successfully fetched {len(samples)} records.")
    print()

    # ── Step 3: Analyze field structure ──────────────────────────────
    print("[3/5] FIELD ANALYSIS")
    print("-" * 70)

    first = samples[0]

    def analyze_value(key, value, indent=0):
        """Recursively analyze a field value."""
        prefix = "  " * indent
        type_name = type(value).__name__

        if isinstance(value, dict):
            print(f"{prefix}  {key}: dict with {len(value)} keys")
            for sub_key, sub_val in value.items():
                analyze_value(sub_key, sub_val, indent + 1)
        elif isinstance(value, list):
            if len(value) > 0:
                elem_type = type(value[0]).__name__
                # Show first element preview
                preview = str(value[0])[:80] + "..." if len(str(value[0])) > 80 else str(value[0])
                print(f"{prefix}  {key}: list[{elem_type}] length={len(value)}")
                print(f"{prefix}    [0] = {preview}")
            else:
                print(f"{prefix}  {key}: list (empty)")
        elif isinstance(value, str):
            preview = value[:100] + "..." if len(value) > 100 else value
            print(f"{prefix}  {key}: str length={len(value)}")
            print(f'{prefix}    = "{preview}"')
        elif isinstance(value, (int, float)):
            print(f"{prefix}  {key}: {type_name} = {value}")
        else:
            print(f"{prefix}  {key}: {type_name} = {str(value)[:100]}")

    for key, value in first.items():
        analyze_value(key, value)
    print()

    # ── Step 4: Print top-level field summary ────────────────────────
    print("[4/5] FIELD SUMMARY TABLE")
    print("-" * 70)
    print(f"{'Field':<25} {'Type':<15} {'Example Size/Value':<30}")
    print("-" * 70)

    for key, value in first.items():
        type_name = type(value).__name__
        if isinstance(value, str):
            size_info = f"len={len(value)}"
        elif isinstance(value, list):
            size_info = f"len={len(value)}"
        elif isinstance(value, dict):
            size_info = f"keys={list(value.keys())}"
        else:
            size_info = str(value)[:30]
        print(f"{key:<25} {type_name:<15} {str(size_info):<30}")
    print()

    # ── Step 5: Print full sample records ────────────────────────────
    print("[5/5] FULL SAMPLE RECORDS")
    print("-" * 70)

    for i, sample in enumerate(samples[:3]):  # Show 3 full records
        print(f"\n--- Record {i} ---")
        for key, value in sample.items():
            if isinstance(value, (dict, list)):
                # Pretty print nested structures, but truncate long text
                if isinstance(value, dict):
                    truncated = {}
                    for k, v in value.items():
                        if isinstance(v, list):
                            truncated[k] = [str(item)[:80] + "..." if len(str(item)) > 80 else item for item in v[:3]]
                            if len(v) > 3:
                                truncated[k].append(f"... ({len(v)} total)")
                        elif isinstance(v, str) and len(v) > 100:
                            truncated[k] = v[:100] + "..."
                        else:
                            truncated[k] = v
                    print(f"  {key}: {json.dumps(truncated, indent=4, ensure_ascii=False)}")
                elif isinstance(value, list):
                    truncated = [str(item)[:100] + "..." if len(str(item)) > 100 else item for item in value[:3]]
                    if len(value) > 3:
                        truncated.append(f"... ({len(value)} total)")
                    print(f"  {key}: {json.dumps(truncated, indent=4, ensure_ascii=False)}")
            elif isinstance(value, str) and len(value) > 150:
                print(f"  {key}: \"{value[:150]}...\"")
            else:
                print(f"  {key}: {json.dumps(value, ensure_ascii=False)}")

    # ── Summary statistics across all samples ────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS (across 5 samples)")
    print("=" * 70)

    # Count passages per record
    passage_counts = []
    selected_counts = []
    languages = set()

    for s in samples:
        if "passages" in s and isinstance(s["passages"], dict):
            eng = s["passages"].get("English_passages", [])
            selected = s["passages"].get("is_selected", [])
            passage_counts.append(len(eng))
            selected_counts.append(sum(1 for x in selected if x == 1))
        if "target_lang" in s:
            languages.add(s["target_lang"])
        if "source_lang" in s:
            languages.add(s["source_lang"])

    if passage_counts:
        print(f"  Passages per record: min={min(passage_counts)}, max={max(passage_counts)}, avg={sum(passage_counts)/len(passage_counts):.1f}")
        print(f"  Selected passages per record: min={min(selected_counts)}, max={max(selected_counts)}, avg={sum(selected_counts)/len(selected_counts):.1f}")

    print(f"  Languages seen: {languages}")
    print(f"  Available fields: {list(first.keys())}")

    # Check query types
    query_types = set(s.get("query_type", "N/A") for s in samples)
    print(f"  Query types seen: {query_types}")

    print()
    print("✅ Dataset inspection complete. No full download was performed.")
    print("   Use these findings to design the preprocessing pipeline.")


if __name__ == "__main__":
    inspect_dataset()
