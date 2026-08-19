"""
evaluation/latency_eval.py  (also the end-to-end pipeline latency eval)

True end-to-end RAG pipeline latency, measured via RAGOrchestrator.answer()
on N queries — the actual production path (safety → relevance → retrieval
[hybrid, no rerank — matches prod] → generation → output validation →
grounding → structured response).

WHAT'S REPORTED per stage, at P50 / P70 / P100:
  - safety_ms, relevance_ms          (guardrails — always run)
  - retrieval_ms                     (hybrid BM25 + dense)
  - generation_ms                    (wall clock around LLM call, all attempts)
  - groq_queue_ms, groq_completion_ms, groq_server_total_ms
                                     (Groq server-side clock — only when
                                      LLM_PROVIDER=groq, else 0)
  - output_validation_ms, grounding_ms
  - total_ms                         (full pipeline wall clock)

Percentiles for generation/groq/validation/grounding are computed ONLY
over queries that reached generation — queries refused earlier would
understate real generation latency if padded with zeros.

NOTE ON COST/RATE LIMITS: this makes N real LLM API calls. Default is 50
to stay within free-tier rate limits. Use --sleep-s to pace calls if
you hit 429s.

Usage:
    python -m evaluation.latency_eval --strategy fixed_token --num-queries 50
    python -m evaluation.latency_eval --strategy fixed_token --num-queries 50 --sleep-s 0.5
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
    """Sums every key like '{prefix}_attempt1', '{prefix}_attempt2', ...
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

    print(f"Building orchestrator for strategy '{args.strategy}' (loads embed model, guardrails, retriever, LLM client once)...")
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

    # ── Detailed per-stage table ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"END-TO-END PIPELINE LATENCY — strategy='{args.strategy}', N={n_ok} queries ({error_count} errored)")
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
        "groq_queue_ms": "  LLM: queue time (Groq only)",
        "groq_completion_ms": "  LLM: completion time (Groq only)",
        "groq_server_total_ms": "  LLM: server total (Groq only)",
        "output_validation_ms": "Output validation",
        "grounding_ms": "Grounding check (multi-signal)",
    }
    for key, label in labels_gen.items():
        vals = gen_only[key]
        # Skip Groq-specific rows when all values are 0 (non-Groq provider)
        if all(v == 0.0 for v in vals) and "groq" in key:
            continue
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
        print(f"[WARNING] {error_count} queries raised exceptions (network/API issues) — excluded from stage stats.")

    # ── SUBMISSION SUMMARY: P50 / P70 / P100 ────────────────────────────────
    total_vals  = always_present["total_ms"]
    ret_vals    = gen_only["retrieval_ms"]
    gen_vals    = gen_only["generation_ms"]
    safety_vals = always_present["safety_ms"]
    rel_vals    = always_present["relevance_ms"]
    ground_vals = gen_only["grounding_ms"]

    print("\n" + "=" * 78)
    print("SUBMISSION LATENCY SUMMARY (P50 / P70 / P100)")
    print("=" * 78)
    print(f"  Full pipeline  (total_ms)  :  "
          f"P50={percentile(total_vals,50):.1f}ms  "
          f"P70={percentile(total_vals,70):.1f}ms  "
          f"P100={percentile(total_vals,100):.1f}ms  (n={len(total_vals)})")
    print(f"  Retrieval      (hybrid)    :  "
          f"P50={percentile(ret_vals,50):.1f}ms  "
          f"P70={percentile(ret_vals,70):.1f}ms  "
          f"P100={percentile(ret_vals,100):.1f}ms  (n={len(ret_vals)})")
    print(f"  Generation     (wall clock):  "
          f"P50={percentile(gen_vals,50):.1f}ms  "
          f"P70={percentile(gen_vals,70):.1f}ms  "
          f"P100={percentile(gen_vals,100):.1f}ms  (n={len(gen_vals)})")
    print(f"  Safety check               :  "
          f"P50={percentile(safety_vals,50):.2f}ms  "
          f"P70={percentile(safety_vals,70):.2f}ms  "
          f"P100={percentile(safety_vals,100):.2f}ms  (n={len(safety_vals)})")
    print(f"  Relevance check            :  "
          f"P50={percentile(rel_vals,50):.2f}ms  "
          f"P70={percentile(rel_vals,70):.2f}ms  "
          f"P100={percentile(rel_vals,100):.2f}ms  (n={len(rel_vals)})")
    print(f"  Grounding check            :  "
          f"P50={percentile(ground_vals,50):.2f}ms  "
          f"P70={percentile(ground_vals,70):.2f}ms  "
          f"P100={percentile(ground_vals,100):.2f}ms  (n={len(ground_vals)})")
    print("=" * 78)
    print()
    print("[NOTE] Pipeline stages (safety+relevance+retrieval+grounding) are fully")
    print("in your control. LLM generation latency includes external API round-trip")
    print("time (Groq/Gemini queue + compute) and will dominate the total_ms figure.")
    print("See README latency section for the full breakdown.")


if __name__ == "__main__":
    main()