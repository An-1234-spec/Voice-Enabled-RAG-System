"""
evaluation/e2e_latency_eval.py

True end-to-end RAG pipeline latency, measured via RAGOrchestrator.answer()
on N queries - the actual production path (safety -> relevance -> retrieval
[hybrid, no rerank - matches prod, see project status notes] -> generation
[Groq] -> output validation -> grounding -> structured response).

Complements evaluation/latency_eval.py, which measures retrieval sub-stages
(dense/bm25/fusion/rerank) in isolation for the rerank tradeoff study.
This script measures the whole pipeline as a user would actually experience
it, including real Groq network + queue time.

WHY THIS EXISTS: latency_eval.py's docstring flags that pipeline/*,
guardrails/*, and generation/llm.py didn't exist yet when it was written,
so it could only report retrieval-only numbers. Those pieces are all built
now - this script fills that gap with the real end-to-end figure.

WHAT'S REPORTED, per stage, P50/P70/P100:
  - safety_ms, relevance_ms, retrieval_ms      (guardrails + retrieval)
  - generation_ms                              (wall clock around Groq call,
                                                 summed across retry attempts)
  - groq_queue_ms, groq_completion_ms,
    groq_server_total_ms                       (Groq's own server clock,
                                                 summed across retry attempts -
                                                 see generation/llm.py)
  - output_validation_ms, grounding_ms
  - total_ms                                   (full pipeline wall clock)
  - retry_rate                                 (fraction of queries that
                                                 needed a 2nd generation attempt)
  - refusal_rate, refusal_by_stage             (fraction refused, broken
                                                 down by which stage refused)

Percentiles for generation/groq/validation/grounding stages are computed
ONLY over queries that actually reached generation - queries refused at
safety/relevance/retrieval never call Groq, and padding those with zeros
would understate real generation latency.

NOTE ON COST/RATE LIMITS: this makes N real Groq API calls. Default is 50,
not 200, to keep runtime and free-tier rate-limit risk reasonable - bump
with --num-queries once you've confirmed your tier can handle it. Use
--sleep-s to add a delay between calls if you hit 429s.

Usage:
    python -m evaluation.e2e_latency_eval --strategy fixed_token --num-queries 50
"""

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.orchestrator import RAGOrchestrator


def collect_queries(chunks_path: Path) -> list[str]:
    """Distinct query strings from a chunk file."""
    seen = set()
    queries = []
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


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def sum_attempt_keys(latency_ms: dict, prefix: str) -> float:
    """Sums every key like '{prefix}_attempt1', '{prefix}_attempt2', ... .
    Represents total wall-clock/server cost actually paid across retries,
    not just the first attempt."""
    return sum(v for k, v in latency_ms.items() if k.startswith(prefix + "_attempt"))


def main():
    parser = argparse.ArgumentParser(description="Measure true end-to-end RAG pipeline latency (P50/P70/P100).")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Delay between queries, e.g. 0.5 to avoid rate limits")
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/processed/embeddings"))
    args = parser.parse_args()

    chunks_path = args.chunks_dir / f"{args.strategy}.jsonl"

    print(f"Building orchestrator for strategy '{args.strategy}' (loads embed model, guardrails, retriever, Groq client once)...")
    pipeline = RAGOrchestrator(
        strategy=args.strategy,
        faiss_dir=args.faiss_dir,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
    )

    distinct_queries = collect_queries(chunks_path)
    if not distinct_queries:
        raise ValueError(f"No queries found in {chunks_path}")

    if len(distinct_queries) < args.num_queries:
        print(
            f"  [note] only {len(distinct_queries)} distinct queries available; "
            f"cycling to reach {args.num_queries} timed runs."
        )
    queries = list(itertools.islice(itertools.cycle(distinct_queries), args.num_queries))

    # Stages present on every query (even refusals)
    always_present = {"safety_ms": [], "relevance_ms": [], "total_ms": []}
    # Stages only present once generation is reached
    gen_only = {
        "retrieval_ms": [], "generation_ms": [],
        "groq_queue_ms": [], "groq_completion_ms": [], "groq_server_total_ms": [],
        "output_validation_ms": [], "grounding_ms": [],
    }
    refusal_counts = defaultdict(int)
    retry_count = 0
    reached_generation_count = 0
    error_count = 0

    print(f"Running {len(queries)} queries through the full pipeline...")
    for i, query in enumerate(queries, 1):
        try:
            response = pipeline.answer(query)
        except Exception as e:
            error_count += 1
            print(f"  [ERROR] query {i} failed: {e}")
            if args.sleep_s:
                time.sleep(args.sleep_s)
            continue

        lm = response.latency_ms
        always_present["safety_ms"].append(lm.get("safety_ms", 0.0))
        always_present["relevance_ms"].append(lm.get("relevance_ms", 0.0))
        always_present["total_ms"].append(lm.get("total_ms", 0.0))

        reached_gen = "generation_ms_attempt1" in lm
        if reached_gen:
            reached_generation_count += 1
            gen_only["retrieval_ms"].append(lm.get("retrieval_ms", 0.0))
            gen_only["generation_ms"].append(sum_attempt_keys(lm, "generation_ms"))
            gen_only["groq_queue_ms"].append(sum_attempt_keys(lm, "groq_queue_ms"))
            gen_only["groq_completion_ms"].append(sum_attempt_keys(lm, "groq_completion_ms"))
            gen_only["groq_server_total_ms"].append(sum_attempt_keys(lm, "groq_server_total_ms"))
            gen_only["output_validation_ms"].append(sum_attempt_keys(lm, "output_validation_ms"))
            gen_only["grounding_ms"].append(sum_attempt_keys(lm, "grounding_ms"))
            if "generation_ms_attempt2" in lm:
                retry_count += 1

        if response.refusal_reason:
            refusal_counts[response.stage_reached] += 1

        if i % 10 == 0:
            print(f"  ...{i}/{len(queries)}")
        if args.sleep_s:
            time.sleep(args.sleep_s)

    n_ok = len(queries) - error_count
    print("\n" + "=" * 78)
    print(f"END-TO-END PIPELINE LATENCY - strategy='{args.strategy}', N={n_ok} queries ({error_count} errored)")
    print("=" * 78)
    print(f"{'Stage':<32}{'P50 (ms)':<12}{'P70 (ms)':<12}{'P100 (ms)':<12}{'n':<6}")
    print("-" * 74)

    labels_always = {
        "safety_ms": "Safety guardrail",
        "relevance_ms": "Relevance guardrail",
    }
    for key, label in labels_always.items():
        vals = always_present[key]
        print(f"{label:<32}{percentile(vals,50):<12.2f}{percentile(vals,70):<12.2f}{percentile(vals,100):<12.2f}{len(vals):<6}")

    labels_gen = {
        "retrieval_ms": "Retrieval (hybrid, no rerank)",
        "generation_ms": "Generation (wall clock, all attempts)",
        "groq_queue_ms": "  Groq: queue time",
        "groq_completion_ms": "  Groq: completion time",
        "groq_server_total_ms": "  Groq: server total",
        "output_validation_ms": "Output validation",
        "grounding_ms": "Grounding check",
    }
    for key, label in labels_gen.items():
        vals = gen_only[key]
        print(f"{label:<32}{percentile(vals,50):<12.2f}{percentile(vals,70):<12.2f}{percentile(vals,100):<12.2f}{len(vals):<6}")

    print("-" * 74)
    vals = always_present["total_ms"]
    print(f"{'TOTAL (full pipeline)':<32}{percentile(vals,50):<12.2f}{percentile(vals,70):<12.2f}{percentile(vals,100):<12.2f}{len(vals):<6}")

    print("\n" + "-" * 74)
    print(f"Reached generation: {reached_generation_count}/{n_ok} ({100*reached_generation_count/max(n_ok,1):.0f}%)")
    print(f"Retry rate (needed 2nd generation attempt): {retry_count}/{max(reached_generation_count,1)} "
          f"({100*retry_count/max(reached_generation_count,1):.0f}%)")
    if refusal_counts:
        print("Refusals by stage:")
        for stage, count in sorted(refusal_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {stage}: {count}")
    if error_count:
        print(f"[WARNING] {error_count} queries raised exceptions (network/API issues) - excluded from stage stats.")

    print("\n[READ THIS] Pipeline stages (safety+relevance+retrieval+validation+grounding)")
    print("are your own engineering and are fully in your control. Groq queue/completion")
    print("time is external - queue_ms in particular reflects your Groq tier's request")
    print("priority, not your code. See README latency section for the full breakdown")
    print("and why the original <200ms target assumed a model Groq has since deprecated.")


if __name__ == "__main__":
    main()