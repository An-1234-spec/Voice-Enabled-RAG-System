"""
guardrails/grounding.py

Grounding check: verifies the LLM's generated answer is actually supported
by its cited source text, independent of the model's own self-reported
"grounded" flag (models can claim grounded=true while still hallucinating).

HEURISTIC, NOT NLI: this is lexical overlap - what fraction of the answer's
words appear literally in the cited sources - not real entailment/NLI, per
plan's own "NLI-lite heuristic" framing. It won't catch a fabricated claim
built entirely from words that DO appear in the source (e.g. reversing a
relationship between two facts that are individually present), and - same
as bm25_retriever.py's tokenizer, reused here for consistency - no
stopword removal, so the ratio has some baseline inflation from common
words. Treat this as a real but weak independent signal, not a guarantee.
"""

from dataclasses import dataclass

from retrieval.bm25_retriever import tokenize  # reuse same tokenizer as BM25 for consistency

DEFAULT_OVERLAP_THRESHOLD = 0.5  # placeholder, not calibrated against labeled data


@dataclass
class GroundingResult:
    passed: bool
    overlap_ratio: float
    reason: str | None = None


def check(answer: str, cited_chunks: list[dict], threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> GroundingResult:
    """
    cited_chunks: the chunk dicts actually cited in sources_used (already
    resolved from tags via generation.prompts.tag_to_chunk_id), NOT all
    retrieved candidates - grounding is checked against what was cited.
    """
    answer = (answer or "").strip()
    if not answer:
        # An empty answer (refusal) makes no factual claims to ground.
        return GroundingResult(passed=True, overlap_ratio=1.0, reason="empty answer, nothing to ground")

    if not cited_chunks:
        return GroundingResult(passed=False, overlap_ratio=0.0, reason="answer given but no sources cited")

    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return GroundingResult(passed=True, overlap_ratio=1.0, reason="answer has no scorable tokens")

    source_tokens = set()
    for chunk in cited_chunks:
        source_tokens |= set(tokenize(chunk.get("text", "")))

    overlap = len(answer_tokens & source_tokens) / len(answer_tokens)

    if overlap < threshold:
        return GroundingResult(
            passed=False,
            overlap_ratio=overlap,
            reason=f"only {overlap:.0%} of answer's words found in cited sources (threshold {threshold:.0%})",
        )
    return GroundingResult(passed=True, overlap_ratio=overlap)