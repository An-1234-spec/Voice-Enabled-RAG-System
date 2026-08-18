"""
scripts/profile_system.py

Profiles the Voice-Enabled RAG system and outputs P50, P70, and P100 latency
statistics for each component and pipeline stage.
Runs across N queries (default 100).
"""

import argparse
import time
import json
import statistics
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from pipeline.orchestrator import RAGOrchestrator
from sentence_transformers import SentenceTransformer
from guardrails.safety import SafetyGuardrail
from guardrails.relevance import RelevanceGuardrail
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import RerankedRetriever


def collect_queries(chunks_dir: Path, strategy: str) -> list[str]:
    chunks_path = chunks_dir / f"{strategy}.jsonl"
    seen = set()
    queries = []
    if not chunks_path.exists():
        # Fall back to any jsonl in chunks dir
        paths = list(chunks_dir.glob("*.jsonl"))
        if not paths:
            return ["what is a corporation?"]
        chunks_path = paths[0]
        
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
    return float(np.percentile(values, p)) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Profile each component of the RAG system.")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--faiss-dir", type=Path, default=Path("data/processed/faiss"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/processed/embeddings"))
    args = parser.parse_args()

    queries = collect_queries(args.chunks_dir, args.strategy)
    if not queries:
        queries = ["what is a corporation?"]
        
    # Cycle queries to reach the requested count
    import itertools
    test_queries = list(itertools.islice(itertools.cycle(queries), args.num_queries))

    print(f"Profiling {len(test_queries)} queries using strategy '{args.strategy}'...")
    
    # Initialize components
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    safety = SafetyGuardrail()
    relevance = RelevanceGuardrail(strategy=args.strategy, embeddings_dir=args.embeddings_dir, model=embed_model)
    
    # We will profile all retrieval configs
    dense_retriever = relevance.model # uses SentenceTransformer
    hybrid = HybridRetriever(strategy=args.strategy, faiss_dir=args.faiss_dir, chunks_dir=args.chunks_dir, model=embed_model)
    reranker = RerankedRetriever(strategy=args.strategy, base_mode="hybrid", faiss_dir=args.faiss_dir, chunks_dir=args.chunks_dir, retrieve_n=10, model=embed_model)

    # Initialize full orchestrator
    orchestrator = RAGOrchestrator(strategy=args.strategy, faiss_dir=args.faiss_dir, chunks_dir=args.chunks_dir, embeddings_dir=args.embeddings_dir)

    metrics = {
        "safety": [],
        "relevance": [],
        "embedding": [],
        "faiss": [],
        "bm25": [],
        "fusion": [],
        "reranking": [],
        "llm_wall": [],
        "llm_queue": [],
        "llm_server": [],
        "validation": [],
        "grounding": [],
        "total": [],
    }

    for i, q in enumerate(test_queries, 1):
        # 1. Safety
        t0 = time.perf_counter()
        safety.check(q)
        metrics["safety"].append((time.perf_counter() - t0) * 1000)

        # 2. Embedding
        t0 = time.perf_counter()
        query_vec = embed_model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        metrics["embedding"].append((time.perf_counter() - t0) * 1000)

        # 3. Relevance (re-normalizes and dots)
        t0 = time.perf_counter()
        relevance.check(q)
        metrics["relevance"].append((time.perf_counter() - t0) * 1000)

        # 4. FAISS Search
        t0 = time.perf_counter()
        hybrid.dense.search(q, top_k=10)
        metrics["faiss"].append((time.perf_counter() - t0) * 1000)

        # 5. BM25 Search
        t0 = time.perf_counter()
        hybrid.bm25.search(q, top_k=10)
        metrics["bm25"].append((time.perf_counter() - t0) * 1000)

        # 6. Hybrid Fusion (runs both + min-max normalizes + fuses)
        t0 = time.perf_counter()
        hybrid.search(q, top_k=5, top_n_raw=10)
        metrics["fusion"].append((time.perf_counter() - t0) * 1000)

        # 7. Reranking (rerank top-10 hybrid results to top-5)
        t0 = time.perf_counter()
        reranker.search(q, top_k=5)
        metrics["reranking"].append((time.perf_counter() - t0) * 1000)

        # 8. Full pipeline call to capture LLM and post-gen stages
        try:
            res = orchestrator.answer(q)
            lm = res.latency_ms
            
            # Extract LLM times (attempt 1)
            metrics["llm_wall"].append(lm.get("generation_ms_attempt1", 0.0))
            metrics["llm_queue"].append(lm.get("groq_queue_ms_attempt1", 0.0))
            metrics["llm_server"].append(lm.get("groq_server_total_ms_attempt1", 0.0))
            
            metrics["validation"].append(lm.get("output_validation_ms_attempt1", 0.0))
            metrics["grounding"].append(lm.get("grounding_ms_attempt1", 0.0))
            metrics["total"].append(lm.get("total_ms", 0.0))
        except Exception as e:
            # If API keys are missing or overload, log 0 or fail gracefully
            pass

        if i % 20 == 0:
            print(f"  Processed {i}/{len(test_queries)}...")

    # Print results
    print("\n" + "=" * 80)
    print(f"BASELINE SYSTEM PROFILING REPORT (N={len(test_queries)})")
    print("=" * 80)
    print(f"{'Component/Stage':<30}{'P50 (ms)':<15}{'P70 (ms)':<15}{'P100 (ms)':<15}")
    print("-" * 80)
    
    for comp, vals in metrics.items():
        if not vals:
            print(f"{comp:<30}{'N/A':<15}{'N/A':<15}{'N/A':<15}")
        else:
            print(f"{comp:<30}{percentile(vals, 50):<15.2f}{percentile(vals, 70):<15.2f}{percentile(vals, 100):<15.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
