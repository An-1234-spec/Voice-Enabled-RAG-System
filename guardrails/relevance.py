"""
guardrails/relevance.py

Domain relevance guardrail: cosine similarity of the query embedding
against a corpus centroid (mean of all chunk embeddings for a strategy).
Below threshold -> out-of-domain, per plan spec.

Reuses embeddings/*.npy already built by embeddings/embedder.py - no new
embedding work, just a mean + a query-time cosine check.

THRESHOLD: there's no principled universal default here since it depends
entirely on your corpus and what "out-of-domain" means for your demo. Use
--calibrate to compute the similarity distribution over REAL in-domain
queries (same ones used for ground-truth retrieval eval) and get a
data-driven suggested threshold, rather than trusting an arbitrary number.
The DEFAULT_THRESHOLD constant below is a placeholder until you calibrate.

Usage:
    # See the in-domain similarity distribution and a suggested threshold
    python -m guardrails.relevance --strategy fixed_token --calibrate

    # Check a single query against a threshold
    python -m guardrails.relevance --strategy fixed_token \
        --query "what is a corporation?" --threshold 0.25
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_THRESHOLD = 0.25  # PLACEHOLDER - run --calibrate and set this from real data


@dataclass
class RelevanceResult:
    passed: bool
    similarity: float
    threshold: float
    reason: str | None = None


class RelevanceGuardrail:
    """Checks query relevance via cosine similarity to a precomputed corpus centroid."""

    def __init__(
        self,
        strategy: str,
        embeddings_dir: Path = Path("data/processed/embeddings"),
        threshold: float = DEFAULT_THRESHOLD,
        model: SentenceTransformer | None = None,
    ):
        self.strategy = strategy
        self.threshold = threshold

        vec_path = embeddings_dir / f"{strategy}.npy"
        if not vec_path.exists():
            raise FileNotFoundError(f"{vec_path} not found. Did you run embeddings/embedder.py?")

        embeddings = np.load(vec_path).astype(np.float32)  # already L2-normalized by embedder.py
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        self.centroid = centroid / norm if norm > 0 else centroid  # re-normalize: mean of unit vectors isn't unit length

        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")

    def check(self, query: str, query_vec: np.ndarray | None = None) -> RelevanceResult:
        if query_vec is None:
            query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        similarity = float(np.dot(query_vec, self.centroid))  # both unit-length -> dot product == cosine similarity

        if similarity < self.threshold:
            return RelevanceResult(
                passed=False,
                similarity=similarity,
                threshold=self.threshold,
                reason=f"Query similarity {similarity:.4f} below threshold {self.threshold:.4f} - likely out-of-domain",
            )
        return RelevanceResult(passed=True, similarity=similarity, threshold=self.threshold)


def calibrate(
    strategy: str,
    embeddings_dir: Path,
    chunks_dir: Path,
    model: SentenceTransformer,
    percentile: int = 5,
) -> None:
    """
    Computes centroid similarity for every real (deduplicated) in-domain
    query in this strategy's chunk file, and prints the distribution plus
    a suggested threshold at the given low percentile - i.e. "X% of known
    in-domain queries score at or above this value."
    """
    chunks_path = chunks_dir / f"{strategy}.jsonl"
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

    if not queries:
        print(f"No queries found in {chunks_path}")
        return

    guardrail = RelevanceGuardrail(strategy=strategy, embeddings_dir=embeddings_dir, model=model)
    similarities = [guardrail.check(q).similarity for q in queries]

    suggested = float(np.percentile(similarities, percentile))

    print(f"\nCalibration: {len(queries)} distinct in-domain queries, strategy='{strategy}'")
    print(f"  min:    {min(similarities):.4f}")
    print(f"  p{percentile}:    {suggested:.4f}  <- suggested threshold")
    print(f"  median: {np.median(similarities):.4f}")
    print(f"  max:    {max(similarities):.4f}")
    print(
        f"\nSetting threshold={suggested:.4f} would reject ~{percentile}% of these known "
        f"in-domain queries as false positives - adjust percentile if that's too aggressive."
    )


def main():
    parser = argparse.ArgumentParser(description="Domain relevance guardrail.")
    parser.add_argument("--strategy", type=str, default="fixed_token")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/processed/embeddings"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--calibrate", action="store_true", help="Print similarity distribution over real in-domain queries")
    parser.add_argument("--percentile", type=int, default=5, help="Percentile used for suggested threshold in --calibrate")
    parser.add_argument("--query", type=str, help="Single query to check (requires --threshold)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.calibrate:
        calibrate(args.strategy, args.embeddings_dir, args.chunks_dir, model, args.percentile)
        return

    if not args.query:
        parser.error("Provide --query, or use --calibrate to see the similarity distribution first.")

    guardrail = RelevanceGuardrail(
        strategy=args.strategy, embeddings_dir=args.embeddings_dir, threshold=args.threshold, model=model
    )
    result = guardrail.check(args.query)

    print(f"\nQuery: {args.query}")
    print(f"Similarity to corpus centroid: {result.similarity:.4f} (threshold: {result.threshold:.4f})")
    print(f"Passed: {result.passed}")
    if not result.passed:
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()