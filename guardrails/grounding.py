"""
guardrails/grounding.py

Multi-signal grounding check: verifies the LLM's generated answer is
actually supported by its cited source text, independent of the model's
own self-reported "grounded" flag.

SIGNAL STACK (fastest -> most expensive, applied in order):
  1. Lexical overlap   -- fraction of answer tokens present in cited sources
                         (baseline; fast; no stopword removal, so common words
                         inflate the ratio slightly -- same tokenizer as BM25
                         for consistency)
  2. Entity consistency -- numbers and capitalised tokens in the answer must
                         appear in the cited sources (catches numeric
                         hallucinations like dates, figures, IDs)
  3. Sentence-level coverage -- each answer sentence must share at least a
                         minimum token overlap with at least one source
                         sentence (catches correct-word-wrong-claim
                         constructions that fool pure bag-of-words)

All three signals are computed; a result is "passed" only when ALL signals
that apply pass. The threshold only governs the lexical pass; the entity
and sentence signals use fixed minimum thresholds documented below.

LIMITATIONS:
  - Still NOT real NLI/entailment -- won't catch a fabricated claim built
    entirely from words individually present in the source (e.g. reversing
    a relationship).
  - Entity check is heuristic: "any capitalised word or pure number" -- not
    true NER, so it will miss lowercased entities and flag common words that
    happen to be capitalised mid-sentence in a source.
  - Sentence coverage uses the same simple tokenizer, so sentence boundary
    detection is a naive "split on '. '" which may mis-split abbreviations.
  Treat each signal as a real but weak guard, not a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.bm25_retriever import tokenize  # reuse same tokenizer as BM25 for consistency

DEFAULT_OVERLAP_THRESHOLD = 0.5      # lexical: fraction of answer tokens in sources
ENTITY_PRESENCE_THRESHOLD = 0.80     # entity: fraction of answer entities found in sources
SENTENCE_COVERAGE_MIN_OVERLAP = 0.20 # sentence: min token overlap for a sentence to be "covered"


# -- Entity extraction --------------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d[\d,.\-]*\b")
_CAMEL_RE  = re.compile(r"\b[A-Z][a-z]{1,}\b")  # CapitalisedWords (heuristic proper noun)


def _extract_entities(text: str) -> set[str]:
    """
    Heuristic named-entity extraction: numbers + CapitalisedWords.
    Returns lowercased forms so comparison is case-insensitive.
    """
    entities: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        entities.add(m.group().lower())
    for m in _CAMEL_RE.finditer(text):
        entities.add(m.group().lower())
    return entities


# -- Sentence splitting -------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '. ' and '? ' and '! '."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# -- Result dataclass ---------------------------------------------------------

@dataclass
class GroundingResult:
    passed: bool
    overlap_ratio: float
    reason: str | None = None
    # Detailed signal breakdown -- useful for debugging / latency analytics
    entity_ratio: float = 0.0
    sentence_coverage: float = 0.0
    failed_signals: list[str] = field(default_factory=list)


# -- Main check ---------------------------------------------------------------

def check(
    answer: str,
    cited_chunks: list[dict],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> GroundingResult:
    """
    cited_chunks: the chunk dicts actually cited in sources_used (already
    resolved from tags via generation.prompts.tag_to_chunk_id), NOT all
    retrieved candidates -- grounding is checked against what was cited.
    """
    answer = (answer or "").strip()

    if not answer:
        # An empty answer (refusal) makes no factual claims to ground.
        return GroundingResult(passed=True, overlap_ratio=1.0, reason="empty answer, nothing to ground")

    if not cited_chunks:
        return GroundingResult(
            passed=False,
            overlap_ratio=0.0,
            reason="answer given but no sources cited",
            failed_signals=["lexical", "entity", "sentence"],
        )

    # Collect all source text
    all_source_text = " ".join(chunk.get("text", "") for chunk in cited_chunks)

    # -- Signal 1: Lexical overlap --------------------------------------------
    answer_tokens = set(tokenize(answer))
    source_tokens = set(tokenize(all_source_text))

    if not answer_tokens:
        return GroundingResult(passed=True, overlap_ratio=1.0, reason="answer has no scorable tokens")

    overlap_ratio = len(answer_tokens & source_tokens) / len(answer_tokens)

    failed_signals: list[str] = []
    reasons: list[str] = []

    if overlap_ratio < threshold:
        failed_signals.append("lexical")
        reasons.append(
            f"lexical: only {overlap_ratio:.0%} of answer words in cited sources "
            f"(threshold {threshold:.0%})"
        )

    # -- Signal 2: Entity consistency -----------------------------------------
    answer_entities = _extract_entities(answer)
    entity_ratio = 1.0  # default: no entities = vacuously passes

    if answer_entities:
        source_entities = _extract_entities(all_source_text)
        found = answer_entities & source_entities
        entity_ratio = len(found) / len(answer_entities)

        if entity_ratio < ENTITY_PRESENCE_THRESHOLD:
            failed_signals.append("entity")
            missing = answer_entities - source_entities
            reasons.append(
                f"entity: {entity_ratio:.0%} of answer entities found in sources "
                f"(need {ENTITY_PRESENCE_THRESHOLD:.0%}); "
                f"missing: {', '.join(sorted(missing)[:5])}"
            )

    # -- Signal 3: Sentence-level coverage ------------------------------------
    answer_sentences = _split_sentences(answer)
    source_sentences = _split_sentences(all_source_text)
    source_sent_token_sets = [set(tokenize(s)) for s in source_sentences]

    covered = 0
    for sent in answer_sentences:
        sent_tokens = set(tokenize(sent))
        if not sent_tokens:
            covered += 1  # empty / punctuation-only sentence, skip
            continue
        # Check if this sentence has >= SENTENCE_COVERAGE_MIN_OVERLAP overlap with ANY source sentence
        best_overlap = max(
            (len(sent_tokens & src_set) / len(sent_tokens) for src_set in source_sent_token_sets),
            default=0.0,
        )
        if best_overlap >= SENTENCE_COVERAGE_MIN_OVERLAP:
            covered += 1

    sentence_coverage = covered / len(answer_sentences) if answer_sentences else 1.0

    # Sentence check only triggers when the answer has more than one sentence
    # (single-sentence answers are already covered by lexical + entity signals)
    if len(answer_sentences) > 1 and sentence_coverage < 1.0:
        failed_signals.append("sentence")
        reasons.append(
            f"sentence: {covered}/{len(answer_sentences)} answer sentences "
            f"covered by cited sources (need all)"
        )

    passed = len(failed_signals) == 0
    reason = "; ".join(reasons) if reasons else None

    return GroundingResult(
        passed=passed,
        overlap_ratio=overlap_ratio,
        entity_ratio=entity_ratio,
        sentence_coverage=sentence_coverage,
        failed_signals=failed_signals,
        reason=reason,
    )