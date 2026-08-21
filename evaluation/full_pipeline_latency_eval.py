"""
evaluation/full_pipeline_latency_eval.py

Full end-to-end pipeline latency benchmark, wrapping the real
pipeline.orchestrator.RAGOrchestrator directly (no reimplemented guardrail
logic - see prior version's postmortem for why that matters).

Includes a grounding-failure signal breakdown: every GroundingResult
carries a `reason` string naming exactly which signal(s) failed
(lexical/entity/sentence/no-sources) - this script now surfaces that
instead of only reporting the aggregate refusal count, so root-causing a
high grounding_check refusal rate doesn't require guessing.

Usage:
    python -m evaluation.full_pipeline_latency_eval --strategy fixed_token --num-queries 200
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from pipeline.orchestrator import RAGOrchestrator


def collect_queries(chunks_path: Path) -> list[str]:
    seen, queries = set(), []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("query")
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline latency benchmark (via real orchestrator).")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--llm-model", type=str, default="llama3.2:1b")
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    args = parser.parse_args()

    chunks_path = args.chunks_dir / f"{args.strategy}.jsonl"

    print("Building RAGOrchestrator (provider=ollama, explicit)...")
    pipeline = RAGOrchestrator(
        strategy=args.strategy,
        top_k=args.top_k,
        llm_provider="ollama",
        llm_model=args.llm_model,
    )

    distinct_queries = collect_queries(chunks_path)
    queries = list(itertools.islice(itertools.cycle(distinct_queries), args.num_queries))

    print("Warmup call (not counted)...")
    pipeline.answer(queries[0])

    print(f"Running {len(queries)} timed queries through the real orchestrator...")
    all_latency: dict[str, list[float]] = {}
    success_count, refusal_count, retry_count = 0, 0, 0
    refusal_stages: dict[str, int] = {}
    grounding_refusal_reasons: list[str] = []

    for i, query in enumerate(queries, 1):
        response = pipeline.answer(query)

        for stage, ms in response.latency_ms.items():
            all_latency.setdefault(stage, []).append(ms)

        if response.grounded and response.answer:
            success_count += 1
        else:
            refusal_count += 1
            refusal_stages[response.stage_reached] = refusal_stages.get(response.stage_reached, 0) + 1
            if response.stage_reached == "grounding_check":
                grounding_refusal_reasons.append(response.refusal_reason or "<no reason recorded>")

        if response.generation_attempts > 1:
            retry_count += 1

        if i % 25 == 0:
            print(f"  ...{i}/{len(queries)}")

    print("\n" + "=" * 70)
    print(f"FULL PIPELINE LATENCY (via real orchestrator) - N={len(queries)}, model={args.llm_model}, top_k={args.top_k}")
    print("=" * 70)
    for stage in sorted(all_latency.keys()):
        vals = all_latency[stage]
        if vals:
            print(f"{stage:<35}P50={percentile(vals,50):<9.2f}P70={percentile(vals,70):<9.2f}P100={percentile(vals,100):<9.2f}n={len(vals)}")

    print(f"\nSuccess: {success_count}  Refusals: {refusal_count}  Retries (2nd attempt needed): {retry_count}")
    print(f"Refusals by stage: {refusal_stages}")

    if grounding_refusal_reasons:
        lexical_fails = sum("lexical" in r for r in grounding_refusal_reasons)
        entity_fails = sum("entity" in r for r in grounding_refusal_reasons)
        sentence_fails = sum("sentence" in r for r in grounding_refusal_reasons)
        no_sources = sum("no sources cited" in r for r in grounding_refusal_reasons)
        print(f"\nGrounding failure signal breakdown (n={len(grounding_refusal_reasons)}, signals can overlap):")
        print(f"  lexical:  {lexical_fails}")
        print(f"  entity:   {entity_fails}")
        print(f"  sentence: {sentence_fails}")
        print(f"  no cited sources: {no_sources}")
        print(f"\nFirst 8 example reasons:")
        for r in grounding_refusal_reasons[:8]:
            print(f"  - {r}")

    total = all_latency.get("total_ms", [])
    p50, p70, p100 = percentile(total, 50), percentile(total, 70), percentile(total, 100)
    print(f"\nP50={p50:.2f}ms  P70={p70:.2f}ms  P100={p100:.2f}ms")
    print(f"Target: P50<=150ms P70<=175ms P100<200ms -> {'MET' if p50<=150 and p70<=175 and p100<200 else 'NOT MET'}")


if __name__ == "__main__":
    main()